"""CleanRL-style PPO+IMPALA-CNN for node-gym envs.

The win over SB3 on Mac/MPS:
  - Rollout buffer is pre-allocated on the device (GPU) once. Each env step
    does one host->device copy of an (n_envs, H, W, C) uint8 obs and writes
    it directly into the buffer slot. SB3 keeps the buffer in numpy and
    copies per-forward, paying the MPS transfer tax twice per step.
  - Single forward pass per rollout step (action sampling). SB3 also does a
    second forward in compute_returns_and_advantage; we do that inline.
  - No SB3 callback / VecMonitor / VecTransposeImage layers between the env
    and the loop. We transpose obs once, on the GPU.

On CUDA boxes the gain shrinks (~1.5x) since transfers are cheap there.
On MPS we observed ~3-4x over SB3 in early runs.

Usage:
    uv run python -m playtrain_trainers.train_ppo_clean --config configs/impala_quickstart.json
"""
from __future__ import annotations

import argparse
import json
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.vec_env import SubprocVecEnv
from torch.utils.tensorboard import SummaryWriter

from playtrain_trainers.wandb_tracking import finish_wandb, init_wandb

from playtrain.runtime import PlayTrainEnv
from playtrain_trainers.policy import ActorCritic
from playtrain_trainers.intrinsic import RND, NovelD, RewardForwardFilter


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
@dataclass
class Config:
    game: str
    total_timesteps: int = 1_000_000
    # Policy encoder: "impala" (default, procgen-paper recommendation) or
    # "nature" (tiny stem; matches the IMPALA throughput-push runs).
    net: str = "impala"
    n_envs: int = 16
    n_steps: int = 128
    n_minibatches: int = 4
    n_epochs: int = 3
    learning_rate: float = 2.5e-4
    gamma: float = 0.999
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    anneal_lr: bool = True
    seed: int = 0
    device: str = "auto"
    log_dir: str = "outputs/ppo_clean"
    save_every_updates: int = 25
    # Reward clipping. The grid env emits rewards on wildly different scales
    # (+500 pickup, ±1000 value/kill, -5000 death, +50000-80000 win) which
    # makes value-function regression hard. "sign" maps reward -> {-1, 0, +1}
    # (Atari-style); "abs_one" clamps to [-1, 1] (IMPALA-parity: sub-unit
    # rewards like the stepcost -0.005 penalty pass through, big rewards
    # saturate); "symlog" sign*log1p(|r|) preserves ordering; "none" leaves
    # rewards untouched. Raw ep_return is still logged so the per-update ret=
    # number stays interpretable.
    reward_clip: str = "none"
    # Reward normalization: divide rewards by running std of discounted returns
    # before GAE. Standard practice in CleanRL/SB3/OpenAI-baselines PPO on
    # sparse-reward envs with wide reward magnitudes. Reuses the existing
    # RewardForwardFilter (Burda RND convention). Compose with reward_clip:
    # clip first (sign/symlog/none), then normalize. Off by default to keep
    # existing configs unchanged.
    reward_norm: bool = False
    # Env backend: "playtrain" (default) or "ale". With "ale", `game` is a
    # gymnasium ALE id like "ALE/Pong-v5" and obs is resized to (64,64,3)
    # so the CNN architecture is byte-identical to node-gym runs. Used for
    # throughput + difficulty calibration vs Atari baselines.
    env_backend: str = "playtrain"  # "playtrain" (JS games) | "minigrid"; "node_gym" = legacy alias
    # If set, every env reset is forced to this seed (via SeedRangeWrapper).
    # Use for sanity checks: memorize one fixed instance instead of training
    # on the full procedural distribution. None = default procedural sampling.
    fixed_env_seed: int | None = None
    # Binding-level generalization sweep (see playtrain_trainers.generalization). When set,
    # training is restricted to a finite pool of distinct-binding seeds (the
    # held-out "sword=key" configs are excluded) instead of the full procedural
    # distribution. Spec dict: {n_train_bindings, split_seed, scan,
    # placements_per_binding}. Mutually exclusive with fixed_env_seed. The Y-axis
    # (held-out win-rate) is computed post-hoc via tools/eval_generalization.py.
    train_pool: dict | None = None
    # Episode truncation horizon (passed to node-gym runtime). Default 2000
    # matches node-gym's default. Shorter horizons can help on sparse-reward
    # navigation by making "harvest-then-idle" less attractive (less time to
    # idle), at the cost of fewer chances per episode to find the terminal.
    max_steps: int = 2000
    # Action-repeat: run this many game frames per agent decision (node-gym
    # GameEnv frame-skip), holding the action across them and returning one
    # transition. Default 1 = unchanged. Set to MOVE_COOLDOWN+1 (=7 for these
    # grid games) to collapse the per-move cooldown so 1 step == 1 move.
    # max_steps still counts FRAMES, so the per-frame living cost stays calibrated.
    frame_skip: int = 1
    # Observation resolution passed to node-gym. Default 64 matches node-gym's
    # default (4:1 downsample of a 256×256 canvas). Higher values (128, 256)
    # reduce downsample throwaway: at 128 the BLUE_KEY tool sprite goes from
    # ~3×3 obs px to ~6×6 obs px, making color/coarse-shape parsing tractable.
    # Throughput cost is ~50-80% slower wall time at 128.
    obs_size: int = 64
    # Warm-start: path to a .pt checkpoint (or a dir containing final.pt)
    # whose model weights to load before training. Optimizer starts fresh.
    # Use to seed a v2 run from a converged v1 checkpoint, etc. Shapes must
    # match (same in_channels / input_hw / n_actions). None = fresh init.
    init_from: str | None = None
    # RND (Random Network Distillation, Burda et al. 2018). Off by default
    # so existing configs keep their behavior. See intrinsic.py.
    use_rnd: bool = False
    rnd_coef: float = 1.0
    rnd_lr: float = 1e-4
    rnd_int_gamma: float = 0.99
    rnd_update_proportion: float = 0.25
    # NovelD (Zhang et al. 2021): RND novelty *difference* gated to the first
    # episodic visit of each state. Reuses the RND net/optimizer/normalizer
    # below — enabling it implies the RND scaffolding. Stronger than flat RND on
    # hard-exploration MiniGrid (KeyCorridor). Mutually exclusive with use_rnd.
    use_noveld: bool = False
    noveld_alpha: float = 0.5
    noveld_coef: float = 1.0
    # GAE bootstrap mask. True (default, Gymnasium-correct): only real
    # terminations zero V_next; truncations bootstrap. False: truncations
    # also zero V_next (pre-dbe9a70 behavior, biases V downward near the
    # truncation horizon). The False mode implicitly changes the objective
    # to finite-horizon-with-terminal-bonus, which empirically helps PPO
    # commit to terminal-reaching policies on sparse-reward navigation
    # tasks (see 14285151 v5_2rooms_door escape vs the post-fix reruns).
    bootstrap_truncation: bool = True
    # torch.compile the policy. "reduce-overhead" uses CUDA graphs and helps
    # most on small models where kernel-launch overhead dominates (IMPALA-CNN
    # on A100). Pays a one-time 30s-2min compile cost on first forward and
    # again the first time a new batch shape is seen (rollout n_envs vs update
    # minibatch differ). None = off. Cuda-only — ignored on cpu/mps.
    # Modes: None | "default" | "reduce-overhead" | "max-autotune".
    compile_mode: str | None = None
    # bf16 mixed precision. Wraps the model forward + loss in
    # torch.autocast(bfloat16) so the (conv-heavy) IMPALA-CNN runs on A100 tensor
    # cores; GAE/advantages stay fp32, optimizer master weights stay fp32, no
    # GradScaler needed (bf16 has fp32 range). Cuda-only — no-op on cpu/mps.
    # Changes numerics, so it's OFF by default: validate learning parity before
    # using it in a matrix run. Composes with compile_mode.
    bf16: bool = False
    # Vectorised env backend.
    #   "subproc" (default) — SB3 SubprocVecEnv around N node-gym child Python
    #                          procs + pickle round-trip. Battle-tested.
    #   "nodevec"           — playtrain.runtime.PlayTrainVecEnv: one Python proc, N Node
    #                          workers via direct pipes + mmap. Same SB3-style
    #                          autoreset semantics. FASRC bench (job 12978195):
    #                          +13.8% trainer sps for grid_v4 N=8 vs subproc.
    #                          Only valid when env_backend="playtrain".
    #   "native"            — playtrain.runtime.native_vec_env.NativeVecEnv:
    #                          in-process C++ threadpool over QuickJS+rasterizer
    #                          (the IMPALA vec-worker backend). SAME_STEP
    #                          autoreset; ~10-100x nodevec env throughput.
    #                          Only valid when env_backend="playtrain".
    vec_backend: str = "subproc"
    # native backend only: threadpool size (0 = auto/cores) and games dir
    # override (None = playtrain's bundled examples/games/js).
    native_env_threads: int = 0
    native_games_dir: str | None = None
    # native backend only: double-buffered ("ping-pong") sampling. Splits the
    # envs into two groups sharing one threadpool, so one group's C++ step runs
    # while the other group's obs go through the policy — the same overlap the
    # IMPALA vec worker uses (impala/vec_actor.py). ON-POLICY IS PRESERVED: both
    # groups act under the same weights within a rollout, and each env's
    # trajectory stays contiguous in time, so GAE is unaffected.
    #
    # The gain is bounded by the ratio of the two phases (overlap can hide only
    # the SMALLER one), so env-heavy games gain most: measured ceilings are
    # 1.20x bigfish, 1.37x breakout, 1.61x miner.
    #
    # Feedforward extrinsic-reward runs only — the recurrent and RND/NovelD
    # paths carry per-step state that the interleaved loop does not thread
    # through, so they are rejected rather than silently mis-trained.
    double_buffer: bool = False
    # Fill a whole node with one PPO run: launch under torchrun and each rank
    # takes n_envs/world_size environments on its own GPU, then gradients are
    # averaged across ranks.
    #
    # THE MATH IS UNCHANGED. n_envs stays the TOTAL, so rank r holds envs
    # [r*local, (r+1)*local) with the same seeds the single-GPU run would have
    # given them, per-rank minibatches are 1/world of the single-GPU minibatch,
    # and averaging the gradient over ranks reproduces the single-GPU gradient
    # on the full minibatch. Advantage normalization is reduced across ranks for
    # the same reason (normalizing per-shard would NOT be equivalent). So this
    # is a pure hardware change: same batch, same GAE horizon, same
    # gradient-steps-per-transition.
    #
    # Not bit-identical, and should not be advertised as such: the k-th global
    # minibatch is the union of each rank's k-th local chunk, so the partition
    # is stratified across ranks rather than a free shuffle of all n_envs. Same
    # distribution, same sizes, different draw.
    #
    # Contrast with raising n_envs, which buys throughput by shrinking
    # optimization per frame. This does not.
    #
    # Requires n_envs divisible by world_size. Feedforward extrinsic path only
    # (RND/NovelD keep unsynced running statistics; reward_norm keeps an
    # unsynced running std) -- those are rejected rather than silently
    # producing rank-dependent training.
    ddp: bool = False
    # Report where wall-clock actually goes (rollout vs update vs everything
    # else) on each log line. Adds a cuda synchronize at each phase boundary,
    # so it costs a little throughput -- diagnostic only, off for real runs.
    profile_phases: bool = False
    # Experiment tracking (Weights & Biases). Off by default -> TensorBoard
    # only, behavior unchanged. When use_wandb=True, wandb.init mirrors the
    # existing TensorBoard scalars (sync_tensorboard) — no extra logging code.
    # Needs `wandb login` once on the machine. See playtrain_trainers.wandb_tracking.
    use_wandb: bool = False
    wandb_project: str = "playtrain"
    wandb_group: str | None = None
    # W&B run title. None -> the log_dir basename (the per-job dir). The SBATCH
    # wrapper sets this to the config filename so the run reads as the config you
    # launched, not the job id. See playtrain_trainers.wandb_tracking.init_wandb.
    wandb_name: str | None = None
    # Recurrent (LSTM) policy. False (default) = feedforward, behavior unchanged.
    # True inserts an nn.LSTM core between the IMPALA-CNN encoder and the heads
    # (playtrain_trainers.policy.ActorCritic use_lstm), with monobeast/CleanRL done-reset
    # semantics. The PPO update then minibatches over whole ENV-COLUMNS (not
    # shuffled timesteps), replaying the LSTM through each env's full n_steps
    # sequence from the stored per-rollout state — so n_envs must be divisible
    # by n_minibatches. ~20-25% slower than feedforward (cf. IMPALA LSTM).
    use_lstm: bool = False
    # Feed a_{t-1} (one-hot) and r_{t-1} (clamped to [-1,1]) into the LSTM input,
    # as IMPALA/monobeast/R2D2 do. Default True but ONLY applies when use_lstm
    # (feeding action history without a recurrent core makes argmax rollouts
    # brittle — see memory/feedback_impala_argmax.md). Set False for a
    # Markov-clean LSTM that matches the validated IMPALA-LSTM exactly.
    feed_prev_action_reward: bool = True
    # How the r_{t-1} fed to the LSTM is bounded (only when feed_prev_action_reward).
    # True (default): feed the RAW reward, squashed by symlog in the net
    # (bounded-growth, magnitude-preserving, independent of reward_clip). False:
    # feed the reward_clip-SHAPED reward verbatim, so bounding follows reward_clip
    # (symlog config -> same as True; sign -> +/-1; none -> raw, your call).
    lstm_symlog_reward: bool = True

    @classmethod
    def from_json(cls, path: Path) -> "Config":
        raw = json.loads(path.read_text())
        # Ignore SB3-only keys that we don't use
        for k in ("batch_size",):
            raw.pop(k, None)
        return cls(**{k: v for k, v in raw.items() if k in cls.__dataclass_fields__})


# ----------------------------------------------------------------------
# Env wiring
# ----------------------------------------------------------------------
def compute_gae(
    rewards: torch.Tensor,         # (n_steps, n_envs)
    values:  torch.Tensor,         # (n_steps, n_envs)
    bootstrap_mask: torch.Tensor,  # (n_steps, n_envs) — 1 where V_next should be zeroed
    next_value: torch.Tensor,      # (n_envs,)
    bootstrap_last: torch.Tensor,  # (n_envs,) — 1 if final-step V_next should be zeroed
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalized Advantage Estimation. Returns (advantages, returns).

    `bootstrap_mask[t]` controls whether V at step t is treated as terminal:
      - With bootstrap_truncation=True the caller passes `terms_buf` here, so
        only true terminations zero V_next (truncations still bootstrap).
      - With bootstrap_truncation=False the caller passes `dones_buf`, so
        truncations also zero V_next (Pardo Case 1-ish partial impl).

    Extracted from the inline GAE loop in PPO training so it can be unit-tested
    in isolation. The trainer calls this and must produce bit-identical output
    to the previous inline implementation.
    """
    n_steps = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros_like(rewards[0])
    for t in reversed(range(n_steps)):
        if t == n_steps - 1:
            next_nonterm = 1.0 - bootstrap_last
            next_v = next_value
        else:
            next_nonterm = 1.0 - bootstrap_mask[t + 1]
            next_v = values[t + 1]
        delta = rewards[t] + gamma * next_v * next_nonterm - values[t]
        last_gae = delta + gamma * gae_lambda * next_nonterm * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


def derive_terminated(info_list: list[dict], done_np: np.ndarray) -> np.ndarray:
    """Per-env REAL-termination flag for GAE bootstrap, backend-agnostic.

    The bootstrap_truncation logic needs to tell a real termination (V_next=0)
    apart from a truncation/timeout (V_next bootstraps). The two vec backends
    surface that distinction differently:

      1. ``NodeVecAdapter`` sets an explicit ``info["terminated"]`` from the
         env's (terminated, truncated) split — used directly when present.
      2. SB3 ``SubprocVecEnv`` sets ``info["TimeLimit.truncated"] = truncated and
         not terminated`` on every step but NO ``"terminated"`` key. So a done
         that is not a TimeLimit-truncation is a real termination:
         ``done and not info["TimeLimit.truncated"]``.

    Returns a bool ndarray (n_envs,).

    NOTE (was a bug): the previous inline code did
    ``info.get("terminated", done_np[i])`` — but SubprocVecEnv never sets
    "terminated", so it always fell back to done_np, treating EVERY timeout as a
    termination. That silently made bootstrap_truncation=True behave like False
    on the default subproc backend (and on every MiniGrid RND/NovelD config).
    """
    out = np.empty(len(done_np), dtype=bool)
    for i in range(len(done_np)):
        info = info_list[i]
        if "terminated" in info:
            out[i] = bool(info["terminated"])
        else:
            out[i] = bool(done_np[i]) and not bool(info.get("TimeLimit.truncated", False))
    return out


def make_env(game: str, seed: int, backend: str = "playtrain", fixed_env_seed: int | None = None,
             frame_skip: int = 1, train_seeds: list[int] | None = None):
    def _thunk():
        # IMPORTANT: do NOT call env.reset() here. SubprocVecEnv calls reset()
        # itself after construction (via the user's first venv.reset()), so an
        # in-thunk reset would cause one extra tick() per worker — putting
        # frameCount out of sync with the PlayTrainVecEnv path by 1 frame. That
        # offset shows up as 30+ cells of pixel divergence on any sprite that
        # uses `frameCount % N` blink animations (see archive/scripts/compare_vec_backends).
        if backend == "ale":
            # ale_env.py archived to archive/src/ — all Atari configs were
            # archived alongside it. Restore both if you want to rerun
            # Atari experiments.
            raise ValueError(
                "env_backend='ale' requires an ale_env module from a consumer repo, "
                "which was retired to archive/src/ale_env.py (no active "
                "configs use the ALE backend). Restore the module to "
                "re-enable."
            )
        if backend in ("playtrain", "node_gym"):  # "node_gym" = legacy alias
            env = PlayTrainEnv(game=game, frame_skip=frame_skip)
            if train_seeds is not None:
                # Generalization sweep: restrict resets to the train pool.
                from playtrain_trainers.plugins import SeedSetWrapper
                env = SeedSetWrapper(env, train_seeds, rng_seed=seed)
            elif fixed_env_seed is not None:
                from playtrain.runtime.env import SeedRangeWrapper
                env = SeedRangeWrapper(env, fixed_env_seed, fixed_env_seed + 1)
            return env
        if backend == "minigrid":
            from playtrain_trainers.plugins import make_minigrid_env
            return make_minigrid_env(game, seed)
        raise ValueError(f"unknown env_backend={backend!r} (expected 'playtrain'|'minigrid')")
    return _thunk


class _NodeVecAdapter:
    """SB3-shaped wrapper around playtrain.runtime.PlayTrainVecEnv.

    The trainer was written against SB3 SubprocVecEnv's API (reset() returns
    just obs; step() returns 4-tuple (obs, rewards, dones, infos)). PlayTrainVecEnv
    is Gymnasium 1.0-shaped (reset returns (obs, info); step returns 5-tuple).
    This thin shim translates so the rest of the loop is unchanged.
    """

    def __init__(self, venv, *, num_envs: int):
        self._venv = venv
        self.num_envs = num_envs
        self.observation_space = venv.single_observation_space  # SB3 reads single
        self.action_space = venv.single_action_space

    def reset(self):
        obs, _info = self._venv.reset()
        return obs

    def step(self, actions):
        obs, rewards, terms, truncs, info = self._venv.step(actions)
        dones = terms | truncs
        # SB3-style infos: list-of-dicts with terminal_observation in done envs.
        # Also expose per-env `terminated` so the trainer can distinguish a real
        # terminal (V_next=0) from a truncation (bootstrap V_next). Without
        # this, GAE treats every timeout as a sink state and the value function
        # systematically underestimates V near the truncation horizon.
        infos: list[dict] = [{"terminated": bool(terms[i])} for i in range(self.num_envs)]
        if "_final_observation" in info:
            mask = info["_final_observation"]
            final = info["final_observation"]
            for i in np.where(mask)[0]:
                infos[i]["terminal_observation"] = final[i]
        return obs, rewards, dones, infos

    def close(self):
        self._venv.close()


class _NativeVecAdapter:
    """SB3-shaped wrapper around playtrain.runtime.native_vec_env.NativeVecEnv.

    NativeVecEnv steps a C++ threadpool of QuickJS envs in-process and (with
    autoreset=True) applies SAME_STEP autoreset: done envs surface the fresh
    post-reset obs in the same step, matching the SB3 semantics this trainer
    was written against. Terminal observations are NOT surfaced (the reset
    frame replaces them), so truncation bootstrap falls back to the trainer's
    terminated-flag path — same trade the IMPALA vec path makes.
    """

    def __init__(self, nv, *, num_envs: int, n_actions: int = 8):
        import gymnasium as gym
        self._nv = nv
        self.num_envs = num_envs
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(nv.obs_size, nv.obs_size, 3), dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(n_actions)

    def reset(self):
        return self._nv.reset(seeds=self._initial_seeds)

    def step(self, actions):
        obs, rewards, terms, truncs, _info = self._nv.step(actions)
        dones = terms | truncs
        infos: list[dict] = [{"terminated": bool(terms[i])} for i in range(self.num_envs)]
        return obs, rewards, dones, infos

    def close(self):
        self._nv.close()


class _NullWriter:
    """Stand-in SummaryWriter for non-zero ranks: swallow every call."""

    def add_scalar(self, *a, **k):
        pass

    def flush(self):
        pass

    def close(self):
        pass


def _ddp_setup(cfg: "Config"):
    """Join the torchrun process group. -> (world_size, rank, local_rank).

    Returns (1, 0, 0) when ddp is off or the job was not launched under
    torchrun, so every downstream code path is identical in the single-GPU case.
    """
    import os
    if not cfg.ddp:
        return 1, 0, 0
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        raise RuntimeError(
            "ddp=True but RANK/WORLD_SIZE are unset -- launch with "
            "`torchrun --standalone --nproc_per_node=N -m "
            "playtrain_trainers.train_ppo_clean --config ...`")
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if world > 1:
        torch.distributed.init_process_group("nccl", rank=rank, world_size=world)
        torch.cuda.set_device(local_rank)
    return world, rank, local_rank


def _ddp_average_grads(model, world: int) -> None:
    """Average gradients across ranks, in place.

    Done explicitly rather than via DistributedDataParallel because the rollout
    calls model.act() and the update calls model.forward()/get_action_and_value()
    -- DDP only syncs on its own forward(), so wrapping would either miss the
    sync or force the rollout through the wrapper. The encoder is small (a
    Nature/IMPALA CNN), so the unbucketed all-reduce is cheap; measure before
    optimizing it.

    Must run BEFORE grad-norm clipping so the clip sees the same gradient the
    single-GPU run would have clipped.
    """
    if world <= 1:
        return
    for p in model.parameters():
        if p.grad is not None:
            torch.distributed.all_reduce(p.grad, op=torch.distributed.ReduceOp.SUM)
            p.grad /= world


def _ddp_normalize_adv(mb_adv, world: int):
    """Standardize advantages over the GLOBAL minibatch (all ranks).

    Per-rank standardization would make each rank's update depend on its own
    shard's mean/std, which is NOT what the single-GPU run computes. Reduces
    sum, sum-of-squares and count so mean/std match the full minibatch.
    """
    if world <= 1:
        return (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
    stats = torch.stack([mb_adv.sum(), (mb_adv * mb_adv).sum(),
                         torch.tensor(float(mb_adv.numel()), device=mb_adv.device)])
    torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)
    total, sq, n = stats[0], stats[1], stats[2]
    mean = total / n
    # population variance, matching torch.std(unbiased=True) closely enough at
    # these minibatch sizes; use the unbiased form to match exactly.
    var = (sq - n * mean * mean) / (n - 1)
    return (mb_adv - mean) / (var.clamp_min(0).sqrt() + 1e-8)


def _shape_reward(reward_np, reward_clip: str):
    """Apply cfg.reward_clip to one step's rewards (shared by both rollouts)."""
    if reward_clip == "sign":
        return np.sign(reward_np).astype(np.float32)
    if reward_clip == "abs_one":
        # Clamp to [-1, 1] (matches IMPALA's "abs_one"). Unlike "sign",
        # sub-unit rewards pass through unchanged, so the v5_stepcost/v7
        # per-frame STEP_PENALTY (~0.005) stays at -0.005 rather than
        # collapsing to -1/step. Big terminal/pickup rewards saturate
        # at +/-1, which is the farming pathology to watch for.
        return np.clip(reward_np, -1.0, 1.0).astype(np.float32)
    if reward_clip == "symlog":
        return (np.sign(reward_np) * np.log1p(np.abs(reward_np))).astype(np.float32)
    if reward_clip == "none":
        return reward_np
    raise ValueError(f"unknown reward_clip={reward_clip!r} (expected 'none'|'sign'|'abs_one'|'symlog')")


class _PingPongVecAdapter:
    """Double-buffered counterpart of _NativeVecAdapter.

    Exposes send/wait per group rather than a blocking step(), because the
    overlap only exists if the caller runs inference BETWEEN the two. Per-env
    semantics are identical to the serial path (same env_step, SAME_STEP
    autoreset); only the dispatch is split.

    wait() returns VIEWS into the shared buffers, so anything retained past the
    next send(group) must be copied. The rollout loop copies obs onto the GPU
    and the flags into fresh arrays, which satisfies this.
    """

    def __init__(self, pp, *, n_actions: int = 8):
        import gymnasium as gym
        self._pp = pp
        self.num_envs = pp.num_envs
        self.group_size = pp.group_size
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(pp.obs_size, pp.obs_size, 3), dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(n_actions)

    def reset(self):
        return self._pp.reset(seeds=self._initial_seeds)

    def send(self, group, actions):
        self._pp.send(group, actions)

    def wait(self, group):
        """-> (obs, rewards, dones, terms) for this group's envs only."""
        obs, rew, term, trunc = self._pp.wait(group)
        # rew/term are views into buffers that this group's NEXT send overwrites,
        # so copy them out. obs stays a view: the caller uploads it to the GPU
        # before sending again, which is the only safe way to consume it.
        return obs, rew.copy(), term | trunc, term.copy()

    def close(self):
        self._pp.close()


def build_venv(cfg: "Config", env_seed_offset: int = 0):
    """Build the vectorised env per cfg.vec_backend. Returns an SB3-shaped venv
    (reset()->obs, step()->4-tuple) so the trainer loop is backend-agnostic."""
    # MiniGrid is a Gymnasium-native env, not a PlayTrainVecEnv backend — force subproc.
    if cfg.env_backend == "minigrid" and cfg.vec_backend != "subproc":
        raise ValueError(
            f"env_backend='minigrid' requires vec_backend='subproc', "
            f"got vec_backend={cfg.vec_backend!r}")

    # Generalization sweep: resolve the finite train-seed pool (distinct
    # bindings, held-out sword=key configs excluded). Mutually exclusive with
    # fixed_env_seed (memorize-one-instance).
    train_seeds = None
    if cfg.train_pool is not None:
        if cfg.fixed_env_seed is not None:
            raise ValueError("train_pool and fixed_env_seed are mutually exclusive")
        from playtrain_trainers.plugins import resolve_pools
        train_seeds, _test_seeds = resolve_pools(cfg.train_pool)
        print(f"[train_ppo_clean] generalization pool: "
              f"{cfg.train_pool.get('n_train_bindings')} train bindings -> "
              f"{len(train_seeds)} seeds (held-out eval handled post-hoc)")
    if cfg.vec_backend == "native":
        if cfg.env_backend not in ("playtrain", "node_gym"):
            raise ValueError(
                f"vec_backend='native' requires env_backend='playtrain', "
                f"got env_backend={cfg.env_backend!r}")
        if cfg.double_buffer:
            from playtrain.runtime.native_vec_env import PingPongVecEnv
            nv = PingPongVecEnv(
                game=cfg.game, group_size=cfg.n_envs // 2, obs_size=cfg.obs_size,
                max_steps=cfg.max_steps, num_threads=cfg.native_env_threads,
                frame_skip=cfg.frame_skip, render_skip=cfg.frame_skip > 1,
                games_dir=cfg.native_games_dir)
        else:
            from playtrain.runtime.native_vec_env import NativeVecEnv
            nv = NativeVecEnv(
                game=cfg.game, num_envs=cfg.n_envs, obs_size=cfg.obs_size,
                max_steps=cfg.max_steps, num_threads=cfg.native_env_threads,
                autoreset=True, frame_skip=cfg.frame_skip,
                render_skip=cfg.frame_skip > 1,
                games_dir=cfg.native_games_dir)
        if cfg.fixed_env_seed is not None:
            nv.set_autoreset_seeds("fixed", fixed_seed=cfg.fixed_env_seed)
            initial = np.full(cfg.n_envs, cfg.fixed_env_seed, dtype=np.int32)
        elif train_seeds is not None:
            # env_seed_offset keeps ranks from drawing the same pool sample.
            nv.set_autoreset_seeds("pool", pool=train_seeds,
                                   rng_seed=cfg.seed + env_seed_offset)
            rng = np.random.default_rng(cfg.seed + env_seed_offset)
            initial = rng.choice(np.asarray(train_seeds, dtype=np.int32), cfg.n_envs)
        else:
            # Under DDP, env_seed_offset = rank * envs-per-rank, so the ranks
            # tile exactly the seed range a single-GPU run of the same total
            # n_envs would have used.
            initial = (np.arange(cfg.n_envs, dtype=np.int32) + env_seed_offset
                       + cfg.seed * 1_000_000)
        adapter = (_PingPongVecAdapter(nv) if cfg.double_buffer
                   else _NativeVecAdapter(nv, num_envs=cfg.n_envs))
        adapter._initial_seeds = initial
        return adapter

    if cfg.vec_backend == "nodevec":
        if cfg.env_backend not in ("playtrain", "node_gym"):
            raise ValueError(
                f"vec_backend='nodevec' requires env_backend='playtrain', "
                f"got env_backend={cfg.env_backend!r}")
        from playtrain.runtime import PlayTrainVecEnv
        nv = PlayTrainVecEnv(
            games=[cfg.game] * cfg.n_envs,
            autoreset_mode="same_step",   # SB3-compatible semantics
            autoreset_seed=cfg.seed,
            # When fixed_env_seed is set, every reset (initial + autoreset)
            # uses that same seed — matches SubprocVecEnv + SeedRangeWrapper
            # behavior on the subproc path. When train_seeds is set, every
            # reset draws from that pool (generalization sweep).
            fixed_env_seed=cfg.fixed_env_seed,
            seed_pool=train_seeds,
            max_steps=cfg.max_steps,
            frame_skip=cfg.frame_skip,
            obs_size=cfg.obs_size,
        )
        # Initial reset. If fixed_env_seed is set, PlayTrainVecEnv handles it
        # internally (overrides any per-env seeds we pass); otherwise use
        # per-env seeds (matches SubprocVecEnv pattern where each child gets
        # seed = cfg.seed + i).
        if cfg.fixed_env_seed is not None or train_seeds is not None:
            nv.reset()  # PlayTrainVecEnv applies fixed_env_seed / seed_pool automatically
        else:
            nv.reset(seed=[cfg.seed + i for i in range(cfg.n_envs)])
        return _NodeVecAdapter(nv, num_envs=cfg.n_envs)

    if cfg.vec_backend == "subproc":
        return SubprocVecEnv([
            make_env(cfg.game, cfg.seed + i, cfg.env_backend, cfg.fixed_env_seed,
                     cfg.frame_skip, train_seeds=train_seeds)
            for i in range(cfg.n_envs)
        ])

    raise ValueError(
        f"unknown vec_backend={cfg.vec_backend!r} (expected 'subproc'|'nodevec')")


def pick_device(spec: str) -> torch.device:
    if spec == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(spec)


# ----------------------------------------------------------------------
# Training loop
# ----------------------------------------------------------------------
def train(cfg: Config) -> None:
    world, rank, local_rank = _ddp_setup(cfg)
    is_main = rank == 0
    # n_envs is the TOTAL across ranks. Record it before sharding so config.json
    # describes the run, then rewrite cfg.n_envs to this rank's share -- every
    # buffer, the venv, and the per-rank batch then size themselves correctly
    # with no further changes.
    total_envs = cfg.n_envs
    if world > 1:
        if total_envs % world != 0:
            raise ValueError(
                f"ddp requires n_envs ({total_envs}) divisible by world_size "
                f"({world}).")
        for flag in ("use_rnd", "use_noveld", "reward_norm"):
            if getattr(cfg, flag):
                raise ValueError(
                    f"ddp does not support {flag}=True: it keeps running "
                    f"statistics that are not synchronized across ranks, so "
                    f"each rank would train on differently-scaled rewards.")

    log_dir = Path(cfg.log_dir)
    if is_main:
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "config.json").write_text(json.dumps(cfg.__dict__, indent=2))

    if world > 1:
        cfg.n_envs = total_envs // world
        cfg.device = f"cuda:{local_rank}"
        if is_main:
            print(f"[train_ppo_clean] DDP: {world} ranks x {cfg.n_envs} envs "
                  f"= {total_envs} total; per-rank minibatch "
                  f"{cfg.n_envs * cfg.n_steps // cfg.n_minibatches}, "
                  f"effective {total_envs * cfg.n_steps // cfg.n_minibatches}")
    # Start W&B before the SummaryWriter so sync_tensorboard mirrors every
    # scalar (no-op unless cfg.use_wandb). TensorBoard stays the source of truth.
    # Only rank 0 owns the run's outputs; the other ranks would otherwise write
    # competing TensorBoard event files into the same directory.
    wandb_run = init_wandb(cfg, log_dir) if is_main else None
    writer = SummaryWriter(str(log_dir / "tb")) if is_main else _NullWriter()

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = pick_device(cfg.device)
    print(f"[train_ppo_clean] device={device}  n_envs={cfg.n_envs}  "
          f"n_steps={cfg.n_steps}  total_timesteps={cfg.total_timesteps}  "
          f"reward_clip={cfg.reward_clip}  backend={cfg.env_backend}  "
          f"vec_backend={cfg.vec_backend}  game={cfg.game}  "
          f"bootstrap_truncation={cfg.bootstrap_truncation}  "
          f"reward_norm={cfg.reward_norm}  max_steps={cfg.max_steps}  obs_size={cfg.obs_size}")

    venv = build_venv(cfg, env_seed_offset=rank * cfg.n_envs)
    obs_space = venv.observation_space   # (H, W, C) uint8
    act_space = venv.action_space        # Discrete(n)
    H, W, C = obs_space.shape
    n_actions = int(act_space.n)

    # LSTM minibatching is over env-columns (each env's full sequence stays
    # intact for the recurrent replay), so n_minibatches must divide n_envs.
    if cfg.use_lstm and cfg.n_envs % cfg.n_minibatches != 0:
        raise ValueError(
            f"use_lstm requires n_envs ({cfg.n_envs}) divisible by n_minibatches "
            f"({cfg.n_minibatches}): the PPO update minibatches over whole "
            f"env-columns to replay the recurrent state.")

    # Double-buffered sampling only covers the feedforward extrinsic path. The
    # recurrent and intrinsic-reward paths carry per-step state that the
    # interleaved loop does not thread through, so reject them here rather than
    # let a run finish with quietly wrong credit assignment.
    if cfg.double_buffer:
        if cfg.vec_backend != "native":
            raise ValueError(
                f"double_buffer requires vec_backend='native', got "
                f"{cfg.vec_backend!r}: the ping-pong dispatch is a native-"
                f"backend construct.")
        if cfg.n_envs % 2 != 0:
            raise ValueError(
                f"double_buffer requires an even n_envs ({cfg.n_envs}): the "
                f"envs split into two equal groups.")
        for flag in ("use_lstm", "use_rnd", "use_noveld"):
            if getattr(cfg, flag):
                raise ValueError(
                    f"double_buffer does not support {flag}=True: that path "
                    f"keeps per-step state the interleaved rollout does not "
                    f"carry across groups.")
        print(f"[train_ppo_clean] double-buffered rollout: 2 groups of "
              f"{cfg.n_envs // 2} envs")
    feed_ar = cfg.use_lstm and cfg.feed_prev_action_reward
    model = ActorCritic(n_actions=n_actions, in_channels=C, input_hw=H, net=cfg.net,
                        use_lstm=cfg.use_lstm,
                        feed_prev_action_reward=feed_ar,
                        symlog_reward=cfg.lstm_symlog_reward).to(device)
    if world > 1:
        # Every rank seeds identically so init already matches; broadcast anyway
        # because a silent divergence here would not crash, it would just train
        # a different model on every rank and average nonsense.
        for p_ in model.parameters():
            torch.distributed.broadcast(p_.data, src=0)
        for b_ in model.buffers():
            torch.distributed.broadcast(b_.data, src=0)
    if cfg.use_lstm:
        print(f"[train_ppo_clean] LSTM core enabled  feed_prev_action_reward={feed_ar}  "
              f"symlog_reward={cfg.lstm_symlog_reward}  "
              f"(env-column minibatching: {cfg.n_envs // cfg.n_minibatches} envs/minibatch)")
    if cfg.init_from is not None:
        ckpt_path = Path(cfg.init_from)
        if ckpt_path.is_dir():
            ckpt_path = ckpt_path / "final.pt"
        payload = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(payload["model"])
        src_game = payload.get("config", {}).get("game", "?")
        print(f"[train_ppo_clean] warm-start: loaded model weights from {ckpt_path} "
              f"(source game={src_game})  optimizer reset")
    # Wrap *after* init_from load so checkpoint keys don't collide with the
    # `_orig_mod.` prefix torch.compile adds. Optimizer is built on the
    # compiled module so its parameter refs match.
    if cfg.compile_mode is not None and device.type == "cuda":
        print(f"[train_ppo_clean] torch.compile(mode={cfg.compile_mode!r})")
        model = torch.compile(model, mode=cfg.compile_mode)
    elif cfg.compile_mode is not None:
        print(f"[train_ppo_clean] compile_mode={cfg.compile_mode!r} ignored on device={device.type}")
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, eps=1e-5)

    # bf16 autocast context for the forward+loss (cuda-only; no-op otherwise).
    # Fresh context per use so it nests cleanly inside torch.no_grad().
    use_amp = cfg.bf16 and device.type == "cuda"
    def amp_ctx():
        return torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else nullcontext()
    if cfg.bf16:
        print(f"[train_ppo_clean] bf16 autocast {'ENABLED' if use_amp else f'ignored on device={device.type}'}")

    if cfg.use_rnd and cfg.use_noveld:
        raise ValueError("use_rnd and use_noveld are mutually exclusive "
                         "(NovelD reuses the RND novelty net internally)")

    # RND scaffolding — also used as NovelD's novelty source (no-op when both off)
    if cfg.use_rnd or cfg.use_noveld:
        rnd = RND(in_channels=C, input_hw=H).to(device)
        rnd.build_obs_rms((C, H, W), device=device)
        rnd_optimizer = torch.optim.Adam(rnd.predictor.parameters(),
                                         lr=cfg.rnd_lr, eps=1e-5)
        reward_filter = RewardForwardFilter(cfg.n_envs, cfg.rnd_int_gamma, device=device)
        intr_buf = torch.zeros((cfg.n_steps, cfg.n_envs), device=device)
    else:
        rnd = None
        rnd_optimizer = None
        reward_filter = None
        intr_buf = None

    # NovelD helper (built on the RND novelty net above). int_coef selects the
    # intrinsic coefficient for whichever bonus is active.
    if cfg.use_noveld:
        noveld = NovelD(cfg.n_envs, device=device, alpha=cfg.noveld_alpha)
        int_coef = cfg.noveld_coef
        print(f"[train_ppo_clean] NovelD enabled  coef={cfg.noveld_coef}  "
              f"alpha={cfg.noveld_alpha}  lr={cfg.rnd_lr}  "
              f"int_gamma={cfg.rnd_int_gamma}  update_prop={cfg.rnd_update_proportion}")
    else:
        noveld = None
        int_coef = cfg.rnd_coef
        if cfg.use_rnd:
            print(f"[train_ppo_clean] RND enabled  coef={cfg.rnd_coef}  "
                  f"lr={cfg.rnd_lr}  int_gamma={cfg.rnd_int_gamma}  "
                  f"update_prop={cfg.rnd_update_proportion}")

    # Extrinsic reward normalizer (separate stream from RND's reward_filter).
    if cfg.reward_norm:
        ext_reward_filter = RewardForwardFilter(cfg.n_envs, cfg.gamma, device=device)
        print(f"[train_ppo_clean] reward_norm enabled  (divide rewards by running std of discounted returns)")
    else:
        ext_reward_filter = None

    # Rollout buffer, allocated once on device
    obs_buf      = torch.zeros((cfg.n_steps, cfg.n_envs, C, H, W), dtype=torch.uint8, device=device)
    actions_buf  = torch.zeros((cfg.n_steps, cfg.n_envs), dtype=torch.long, device=device)
    logprobs_buf = torch.zeros((cfg.n_steps, cfg.n_envs), device=device)
    rewards_buf  = torch.zeros((cfg.n_steps, cfg.n_envs), device=device)
    dones_buf    = torch.zeros((cfg.n_steps, cfg.n_envs), device=device)
    values_buf   = torch.zeros((cfg.n_steps, cfg.n_envs), device=device)
    # LSTM-only: a_{t-1}/r_{t-1} aligned with obs_buf[t], for the recurrent input.
    if cfg.use_lstm:
        prev_actions_buf = torch.zeros((cfg.n_steps, cfg.n_envs), dtype=torch.long, device=device)
        prev_rewards_buf = torch.zeros((cfg.n_steps, cfg.n_envs), device=device)

    # Initial obs (SB3 SubprocVecEnv returns just obs, not (obs, info))
    obs_np = venv.reset()
    next_obs = torch.as_tensor(obs_np, device=device).permute(0, 3, 1, 2).contiguous()
    next_done = torch.zeros(cfg.n_envs, device=device)
    # Recurrent state carried across rollouts (() in feedforward mode). next_prev_*
    # hold the a_{t-1}/r_{t-1} entering the next step (zero at the very start).
    next_lstm_state = model.initial_state(cfg.n_envs)
    next_prev_action = torch.zeros(cfg.n_envs, dtype=torch.long, device=device)
    next_prev_reward = torch.zeros(cfg.n_envs, device=device)
    if noveld is not None:
        noveld.reset(rnd.intrinsic_reward(next_obs), obs_np)

    # Episode-return tracking (per env)
    ep_returns = np.zeros(cfg.n_envs, dtype=np.float64)
    ep_lengths = np.zeros(cfg.n_envs, dtype=np.int64)
    completed_returns: list[float] = []
    completed_lengths: list[int] = []

    n_updates = cfg.total_timesteps // (cfg.n_steps * total_envs)
    batch_size = cfg.n_steps * cfg.n_envs
    minibatch_size = batch_size // cfg.n_minibatches
    global_step = 0
    start_time = time.perf_counter()
    # Cumulative phase totals for profile_phases. Whatever the two do not
    # account for is startup + logging + checkpointing, reported as "other".
    _t_rollout = _t_update = 0.0

    # terms_buf flags REAL terminations only (truncations bootstrap V_next).
    # dones_buf still flags any episode end (term|trunc) for stats/auto-reset.
    terms_buf = torch.zeros((cfg.n_steps, cfg.n_envs), device=device)
    next_term = torch.zeros(cfg.n_envs, device=device)

    for update in range(1, n_updates + 1):
        # LR anneal
        if cfg.anneal_lr:
            frac = 1.0 - (update - 1) / n_updates
            for g in optimizer.param_groups:
                g["lr"] = frac * cfg.learning_rate

        # ---- Rollout ----
        if cfg.profile_phases:
            if device.type == "cuda":
                torch.cuda.synchronize()
            _t_phase = time.perf_counter()
        # Snapshot the recurrent state at the rollout's first step. The update
        # replays each env's sequence forward from exactly this state (CleanRL
        # ppo_atari_lstm), so it must be the state, not a fresh zero.
        if cfg.use_lstm:
            initial_lstm_state = (next_lstm_state[0].clone(), next_lstm_state[1].clone())

        if cfg.double_buffer:
            # Double-buffered rollout. Each group is dispatched (send) and only
            # collected (wait) after the OTHER group has been dispatched, so the
            # C++ threadpool is stepping one group's envs while this group's obs
            # go through the policy. Both groups act under the same weights for
            # the whole rollout, so the data stays on-policy, and each env's
            # trajectory occupies one buffer column in time order, so GAE is
            # identical to the serial path.
            B = cfg.n_envs // 2
            slices = (slice(0, B), slice(B, 2 * B))

            # Prime: act for both groups, then dispatch both. From here on there
            # is always exactly one group in flight while the other is inferring.
            pend = [None, None]
            for g in (0, 1):
                with torch.no_grad(), amp_ctx():
                    pend[g] = model.act(next_obs[slices[g]])
                venv.send(g, pend[g][0].cpu().numpy())

            for step in range(cfg.n_steps):
                global_step += total_envs
                for g in (0, 1):
                    sl = slices[g]
                    action, logprob, value = pend[g]
                    # Pre-step state, recorded against the action taken from it —
                    # same (s_t, a_t) pairing as the serial loop.
                    obs_buf[step, sl] = next_obs[sl]
                    dones_buf[step, sl] = next_done[sl]
                    terms_buf[step, sl] = next_term[sl]
                    actions_buf[step, sl] = action
                    logprobs_buf[step, sl] = logprob
                    values_buf[step, sl] = value

                    obs_np, reward_np, done_np, term_np = venv.wait(g)
                    shaped_reward_np = _shape_reward(reward_np, cfg.reward_clip)
                    rewards_buf[step, sl] = torch.as_tensor(
                        shaped_reward_np, dtype=torch.float32, device=device)
                    # Upload obs BEFORE the next send — wait() handed back a view
                    # into the buffer that send is about to overwrite.
                    next_obs[sl] = torch.as_tensor(
                        obs_np, device=device).permute(0, 3, 1, 2)
                    next_done[sl] = torch.as_tensor(
                        done_np, dtype=torch.float32, device=device)
                    next_term[sl] = torch.as_tensor(
                        term_np, dtype=torch.float32, device=device)

                    # Act and dispatch for the next step. Skipped on the last
                    # step so the rollout ends with no envs in flight; next_obs
                    # is then the bootstrap state the value head reads below.
                    if step + 1 < cfg.n_steps:
                        with torch.no_grad(), amp_ctx():
                            pend[g] = model.act(next_obs[sl])
                        venv.send(g, pend[g][0].cpu().numpy())

                    ep_returns[sl] += reward_np
                    ep_lengths[sl] += 1
                    for i in np.where(done_np)[0]:
                        j = sl.start + i
                        completed_returns.append(float(ep_returns[j]))
                        completed_lengths.append(int(ep_lengths[j]))
                        ep_returns[j] = 0.0
                        ep_lengths[j] = 0
        else:
            for step in range(cfg.n_steps):
                global_step += total_envs
                obs_buf[step] = next_obs
                dones_buf[step] = next_done
                terms_buf[step] = next_term

                with torch.no_grad(), amp_ctx():
                    if cfg.use_lstm:
                        prev_actions_buf[step] = next_prev_action
                        prev_rewards_buf[step] = next_prev_reward
                        action, logprob, value, next_lstm_state = model.act_recurrent(
                            next_obs, next_lstm_state, next_done,
                            next_prev_action, next_prev_reward)
                    else:
                        action, logprob, value = model.act(next_obs)
                actions_buf[step] = action
                logprobs_buf[step] = logprob
                values_buf[step] = value

                if rnd is not None and noveld is None:
                    # plain RND: novelty of the current obs s_t
                    intr_buf[step] = rnd.intrinsic_reward(next_obs)

                actions_np = action.cpu().numpy()
                # SB3 SubprocVecEnv: step -> (obs, rewards, dones, infos). dones already
                # combines terminated|truncated, and SB3 auto-resets on done with the
                # post-reset obs returned in info[i]['terminal_observation'] saved.
                obs_np, reward_np, done_np, info_list = venv.step(actions_np)

                # Extract per-env terminated flag for GAE bootstrap correctness.
                # Backend-agnostic: nodevec exposes info["terminated"] directly;
                # subproc exposes info["TimeLimit.truncated"] (see derive_terminated).
                term_np = derive_terminated(info_list, done_np)

                shaped_reward_np = _shape_reward(reward_np, cfg.reward_clip)

                rewards_buf[step] = torch.as_tensor(shaped_reward_np, dtype=torch.float32, device=device)
                next_obs = torch.as_tensor(obs_np, device=device).permute(0, 3, 1, 2).contiguous()
                next_done = torch.as_tensor(done_np, dtype=torch.float32, device=device)
                next_term = torch.as_tensor(term_np, dtype=torch.float32, device=device)
                if cfg.use_lstm:
                    # a_t / r_t become the a_{t-1} / r_{t-1} entering step t+1.
                    # symlog mode: feed the RAW reward, net symlogs it (cue is
                    # independent of reward_clip). Otherwise: feed the reward_clip-
                    # SHAPED reward verbatim, so the cue's bounding follows reward_clip.
                    next_prev_action = action
                    if cfg.lstm_symlog_reward:
                        next_prev_reward = torch.as_tensor(reward_np, dtype=torch.float32, device=device)
                    else:
                        next_prev_reward = rewards_buf[step]

                if noveld is not None:
                    # NovelD bonus for the transition into s_{t+1} (= next_obs / obs_np).
                    # [N(s_{t+1}) - alpha*N(s_t)]_+ gated to first episodic visit.
                    novelty_next = rnd.intrinsic_reward(next_obs)
                    intr_buf[step] = noveld.bonus(novelty_next, obs_np, done_np)

                ep_returns += reward_np
                ep_lengths += 1
                for i in np.where(done_np)[0]:
                    completed_returns.append(float(ep_returns[i]))
                    completed_lengths.append(int(ep_lengths[i]))
                    ep_returns[i] = 0.0
                    ep_lengths[i] = 0

        if cfg.profile_phases:
            if device.type == "cuda":
                torch.cuda.synchronize()
            _now = time.perf_counter()
            _t_rollout += _now - _t_phase
            _t_phase = _now

        # ---- Normalize extrinsic rewards (post-clip, pre-GAE) ----
        if ext_reward_filter is not None:
            rewards_buf = ext_reward_filter.update(rewards_buf)

        # ---- Combine extrinsic + intrinsic (RND) before GAE ----
        if rnd is not None:
            # obs_rms must see the same scale that _normalize uses internally
            # ([0,1] floats), otherwise mean/std would be in 0–255 space and
            # the standardization in _normalize would cancel incorrectly.
            rnd.obs_rms.update(obs_buf.float() / 255.0)
            intr_normalized = reward_filter.update(intr_buf)
            gae_rewards = rewards_buf + int_coef * intr_normalized
            intr_mean = float(intr_normalized.mean().item())
            intr_std = float(intr_normalized.std().item())
        else:
            gae_rewards = rewards_buf
            intr_mean = intr_std = 0.0

        # ---- GAE ----
        # Bootstrap mask. With bootstrap_truncation=True (default): only real
        # terminations zero V_next, truncations bootstrap V_next. With
        # bootstrap_truncation=False: truncations also zero V_next (pre-dbe9a70
        # behavior, biases V downward near truncation horizon).
        bootstrap_buf = terms_buf if cfg.bootstrap_truncation else dones_buf
        bootstrap_last = next_term if cfg.bootstrap_truncation else next_done
        with torch.no_grad():
            with amp_ctx():
                if cfg.use_lstm:
                    next_value = model.get_value(next_obs, next_lstm_state, next_done,
                                                 next_prev_action, next_prev_reward)
                else:
                    next_value = model.forward(next_obs)[1]
            next_value = next_value.float()  # GAE runs in fp32
            advantages, returns = compute_gae(
                rewards=gae_rewards,
                values=values_buf,
                bootstrap_mask=bootstrap_buf,
                next_value=next_value,
                bootstrap_last=bootstrap_last,
                gamma=cfg.gamma,
                gae_lambda=cfg.gae_lambda,
            )

        # ---- Flatten + PPO update ----
        b_obs        = obs_buf.reshape(batch_size, C, H, W)
        b_actions    = actions_buf.reshape(batch_size)
        b_logprobs   = logprobs_buf.reshape(batch_size)
        b_advantages = advantages.reshape(batch_size)
        b_returns    = returns.reshape(batch_size)
        b_values     = values_buf.reshape(batch_size)
        b_dones      = dones_buf.reshape(batch_size)
        if cfg.use_lstm:
            b_prev_actions = prev_actions_buf.reshape(batch_size)
            b_prev_rewards = prev_rewards_buf.reshape(batch_size)
            # Env-column minibatching: each minibatch is a set of whole envs
            # whose full n_steps sequence is replayed from its stored initial
            # state. flat_inds[t, e] is that (step, env)'s row in the [T*N]
            # flatten; gathering flat_inds[:, mb_envs].ravel() keeps the
            # (timestep-major, env-minor) order get_states' view(-1, B) expects.
            envs_per_batch = cfg.n_envs // cfg.n_minibatches
            flat_inds = np.arange(batch_size).reshape(cfg.n_steps, cfg.n_envs)
            env_inds = np.arange(cfg.n_envs)

        idx = np.arange(batch_size)
        clip_fracs = []
        rnd_losses: list[float] = []
        for _ in range(cfg.n_epochs):
            if cfg.use_lstm:
                np.random.shuffle(env_inds)
                minibatches = [env_inds[s:s + envs_per_batch]
                               for s in range(0, cfg.n_envs, envs_per_batch)]
            else:
                np.random.shuffle(idx)
                minibatches = [idx[s:s + minibatch_size]
                               for s in range(0, batch_size, minibatch_size)]
            for mb_units in minibatches:
                # Index setup stays in fp32/outside autocast (integer gathers).
                if cfg.use_lstm:
                    mb_t = torch.as_tensor(flat_inds[:, mb_units].ravel(),
                                           dtype=torch.long, device=device)
                    mb_envs_t = torch.as_tensor(mb_units, dtype=torch.long, device=device)
                    mb_state = (initial_lstm_state[0][:, mb_envs_t].contiguous(),
                                initial_lstm_state[1][:, mb_envs_t].contiguous())
                else:
                    mb_t = torch.as_tensor(mb_units, dtype=torch.long, device=device)

                with amp_ctx():  # forward + loss in bf16 when enabled
                    if cfg.use_lstm:
                        _, new_logprob, entropy, new_value, _ = model.get_action_and_value(
                            b_obs[mb_t], mb_state, b_dones[mb_t], b_actions[mb_t],
                            b_prev_actions[mb_t], b_prev_rewards[mb_t])
                        entropy = entropy.mean()
                    else:
                        dist, new_value = model.forward(b_obs[mb_t])
                        new_logprob = dist.log_prob(b_actions[mb_t])
                        entropy = dist.entropy().mean()

                    logratio = new_logprob - b_logprobs[mb_t]
                    ratio = logratio.exp()

                    mb_adv = b_advantages[mb_t]
                    mb_adv = _ddp_normalize_adv(mb_adv, world)

                    pg1 = -mb_adv * ratio
                    pg2 = -mb_adv * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
                    pg_loss = torch.max(pg1, pg2).mean()

                    v_loss = 0.5 * (new_value - b_returns[mb_t]).pow(2).mean()
                    loss = pg_loss + cfg.vf_coef * v_loss - cfg.ent_coef * entropy

                optimizer.zero_grad()
                loss.backward()
                _ddp_average_grads(model, world)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()

                with torch.no_grad():
                    clip_fracs.append(((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item())

                if rnd is not None:
                    # Train predictor on a Bernoulli subsample of this minibatch
                    # (Burda "update proportion"). Default 0.25 — full batch
                    # makes the predictor too easy and intrinsic reward decays
                    # before the policy has learned from it.
                    mask = torch.rand(mb_t.shape[0], device=device) < cfg.rnd_update_proportion
                    if mask.any():
                        rnd_loss = rnd.predictor_loss(b_obs[mb_t][mask])
                        rnd_optimizer.zero_grad()
                        rnd_loss.backward()
                        rnd_optimizer.step()
                        rnd_losses.append(rnd_loss.item())

        # ---- Logging ----
        if cfg.profile_phases:
            if device.type == "cuda":
                torch.cuda.synchronize()
            _t_update += time.perf_counter() - _t_phase

        elapsed = time.perf_counter() - start_time
        sps = int(global_step / elapsed)
        writer.add_scalar("charts/sps", sps, global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy.item(), global_step)
        writer.add_scalar("losses/clip_frac", float(np.mean(clip_fracs)), global_step)
        writer.add_scalar("losses/lr", optimizer.param_groups[0]["lr"], global_step)
        if rnd is not None:
            writer.add_scalar("rnd/intr_reward_mean", intr_mean, global_step)
            writer.add_scalar("rnd/intr_reward_std", intr_std, global_step)
            if rnd_losses:
                writer.add_scalar("rnd/predictor_loss", float(np.mean(rnd_losses)), global_step)
        if completed_returns:
            # Window must cover at least one full episode per env or games whose
            # episodes end in sync (no failure state -> all n_envs truncate
            # together) alias the mean: a truncation wave floods a 100-slot
            # window and early points average over wins alone. Same defect and
            # fix as the IMPALA learner's deque (e270ece), scaled to n_envs=192.
            recent = completed_returns[-1024:]
            recent_lens = completed_lengths[-1024:]
            writer.add_scalar("charts/ep_return_mean", float(np.mean(recent)), global_step)
            writer.add_scalar("charts/ep_length_mean", float(np.mean(recent_lens)), global_step)
            ep_str = f"ret={np.mean(recent):7.1f} len={np.mean(recent_lens):6.0f}"
        else:
            ep_str = "ret=    n/a len=   n/a"
        rnd_str = f" intr={intr_mean:+.3f}" if rnd is not None else ""
        if cfg.profile_phases:
            _acc = _t_rollout + _t_update
            rnd_str += (f" | rollout={100*_t_rollout/elapsed:4.1f}%"
                        f" update={100*_t_update/elapsed:4.1f}%"
                        f" other={100*(elapsed-_acc)/elapsed:4.1f}%")
        if is_main:
            print(f"upd {update:4d}/{n_updates}  step {global_step:>9d}  {sps:>5d} sps  "
                  f"{ep_str}  ent={entropy.item():.3f} clip={np.mean(clip_fracs):.3f}{rnd_str}")

        if is_main and (update % cfg.save_every_updates == 0 or update == n_updates):
            ckpt = log_dir / f"ckpt_{update:05d}.pt"
            payload = {"model": model.state_dict(), "config": cfg.__dict__,
                       "global_step": global_step}
            if rnd is not None:
                payload["rnd"] = rnd.state_dict()
            torch.save(payload, ckpt)

    final = log_dir / "final.pt"
    if is_main:
        final_payload = {"model": model.state_dict(), "config": cfg.__dict__,
                         "global_step": global_step}
        if rnd is not None:
            final_payload["rnd"] = rnd.state_dict()
        torch.save(final_payload, final)
        print(f"saved final checkpoint -> {final}")
    venv.close()
    writer.close()
    if world > 1:
        # Ranks must all arrive before rank 0 starts the W&B rollout below,
        # otherwise an early exit tears the group down under them.
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
        if not is_main:
            return

    # Log a final greedy (argmax) rollout to W&B so the agent's *behavior* lives
    # on the run page next to the curves (the curves are already in W&B, so a
    # separate run-card file would be redundant). Best-effort — never fails the
    # run. Rolled out at the training instance (fixed_env_seed) when memorizing.
    if wandb_run is not None:
        try:
            from playtrain_trainers.ppo_eval import greedy_rollout, log_rollout_to_wandb
            eval_seed = cfg.fixed_env_seed if cfg.fixed_env_seed is not None else cfg.seed
            eval_env = PlayTrainEnv(game=cfg.game, max_steps=cfg.max_steps,
                                  frame_skip=cfg.frame_skip, obs_size=cfg.obs_size)
            try:
                roll = greedy_rollout(model, eval_env, seed=eval_seed,
                                      max_steps=cfg.max_steps, device=device,
                                      reward_clip=cfg.reward_clip)
                log_rollout_to_wandb(wandb_run, roll)
                print(f"[train_ppo_clean] logged greedy rollout to W&B: "
                      f"return={roll['total_return']:.0f} won={roll['won']} "
                      f"len={roll['length']}")
            finally:
                eval_env.close()
        except Exception as e:  # noqa: BLE001 - telemetry must never fail the run
            print(f"[train_ppo_clean] WARN: W&B rollout logging failed (non-fatal): {e}")

    finish_wandb(wandb_run)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=Path)
    args = p.parse_args()
    cfg = Config.from_json(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
