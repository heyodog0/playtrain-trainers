"""Async actor-learner training loop (port of monobeast.train).

Architecture (matches monobeast):
  - 1 shared CPU `model` for actors (kept in shared memory)
  - N actor processes, each with its own env, looping rollouts into shared
    buffers via free_queue / full_queue coordination
  - 1+ learner threads pulling batches off full_queue, running V-trace
    learn() on `learner_model` (GPU if available), then syncing weights
    back to `model`

Differences from monobeast:
  - Uses our ImpalaNet (IMPALA-CNN encoder) instead of AtariNet.
  - Uses gymnasium envs via playtrain_trainers.impala.Environment.
  - Caller-supplied env factory (env_fn(seed) -> gym env) decouples this
    module from minigrid / node_gym specifics.
  - TensorBoard SummaryWriter for logging (matches train_ppo_clean).
"""
from __future__ import annotations

import contextlib
import copy
import dataclasses
import json
import logging
import pprint
import threading
import time
import timeit
import traceback
from pathlib import Path
from typing import Callable

import torch
from torch import multiprocessing as mp
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from playtrain_trainers.impala.actor import act
from playtrain_trainers.impala.buffers import (
    create_buffers,
    create_initial_agent_state_buffers,
)
from playtrain_trainers.impala.central import (
    InferenceServer,
    act_central,
    create_channel,
)
from playtrain_trainers.impala.diagnostics import log_rss, rss_mb, rss_note
from playtrain_trainers.impala.eval import eval_seeds, greedy_eval
from playtrain_trainers.impala.learn import learn
from playtrain_trainers.impala.net import ImpalaNet
from playtrain_trainers.wandb_tracking import finish_wandb, init_wandb


@dataclasses.dataclass
class ImpalaConfig:
    # env
    game: str = "MiniGrid-KeyCorridorS3R3-v0"
    env_backend: str = "minigrid"  # "minigrid" | "playtrain" ("node_gym" = legacy alias)
    # Action-repeat for the node_gym backend: run this many game frames per
    # env step, holding the action across them (1 = unchanged). Set to 7 for
    # the grid games to absorb MOVE_COOLDOWN so 1 step == 1 move.
    frame_skip: int = 1
    # Frame-stacking (node_gym only): stack this many consecutive RGB frames on
    # the channel axis so the net can perceive velocity/motion — needed for
    # momentum games where a single frame doesn't reveal (vx,vy). obs_shape is
    # auto-derived to 3*frame_stack channels in train(). 1 = off (single frame).
    frame_stack: int = 1
    # Episode truncation horizon in AGENT DECISIONS (node_gym only). node-gym's
    # max_steps counts FRAMES, so with frame_skip>1 a frame budget of 2000 gives
    # only 2000/frame_skip decisions. Setting this pins the decision horizon
    # independent of frame_skip: the env is truncated at max_decisions*frame_skip
    # frames. None = leave node-gym's default (2000 FRAMES) — preserves the prior
    # behavior of every config that doesn't set it.
    max_decisions: int | None = None
    # training
    total_steps: int = 30_000_000
    num_actors: int = 8
    batch_size: int = 8
    unroll_length: int = 80
    num_buffers: int | None = None  # default = max(2*num_actors, batch_size)
    num_learner_threads: int = 2
    # loss
    discounting: float = 0.99
    baseline_cost: float = 0.5
    entropy_cost: float = 0.0006
    # Linear entropy annealing: entropy_cost -> entropy_cost_final over the first
    # entropy_anneal_frac of training. None = constant entropy_cost (default).
    # Control tasks (e.g. asteroids) need high entropy early to explore but
    # near-zero late to execute a precise trajectory without deviating off it.
    entropy_cost_final: float | None = None
    entropy_anneal_frac: float = 1.0
    reward_clipping: str = "abs_one"  # "abs_one" | "symlog" | "none"
    # Hybrid clip: with reward_clipping="abs_one", override the terminal win
    # (reward >= win_bonus_threshold) to a fixed win_bonus instead of +1, so the
    # win dominates the sub-goal breadcrumbs and the policy gets a gradient to
    # CONSOLIDATE winning (fixes argmax-collapse where only exploration wins).
    # None = plain abs_one. Value scale stays bounded (~win_bonus), so no PopArt.
    win_bonus: float | None = None
    win_bonus_threshold: float = 10_000.0
    # GRADED win (terminal efficiency): if win_bonus_slope is set, the terminal
    # reward maps to win_bonus + slope*(reward - threshold), clamped to
    # [win_bonus, win_bonus_max] — a faster / higher-lives win gives a bigger
    # learner signal (rewards EFFICIENCY) without a per-step exploration tax and
    # without symlog-style scale blow-up. Pair with a game whose terminal reward
    # encodes speed (e.g. a speed-encoding config). None = fixed win_bonus.
    win_bonus_slope: float | None = None
    win_bonus_max: float | None = None
    # PopArt value-target normalization. When True, the learner trains on RAW
    # ordering-preserving rewards (reward_clipping is ignored) with an
    # auto-rescaled value head, so reward scale is decoupled from stability.
    use_popart: bool = False
    popart_beta: float = 3e-4
    grad_norm_clipping: float = 40.0
    # optimizer (RMSProp; matches monobeast default)
    learning_rate: float = 0.00048
    rmsprop_alpha: float = 0.99
    rmsprop_momentum: float = 0.0
    rmsprop_epsilon: float = 0.01
    # Learner numerics: "fp32" (default — bit-faithful to every historical
    # run), "tf32" (tensor-core matmuls/convs, fp32 range), "bf16" (autocast
    # mixed precision: weights, optimizer state, V-trace and loss reductions
    # stay fp32; only conv/matmul inner products run in bf16, no GradScaler
    # needed). tf32/bf16 change numerics — validate with a twin run before
    # trusting curves. Measured H100 learner ceilings (T=80, B=64, h2d):
    # fp32 43k / tf32 53k / bf16+compile 102k frames/s.
    learner_precision: str = "fp32"
    # io
    log_dir: str = "outputs/impala"
    seed: int = 0
    # device
    device: str = "auto"  # "auto" | "cuda" | "cpu"
    # observation
    obs_shape: tuple[int, int, int] = (3, 64, 64)  # CHW
    num_actions: int | None = None  # required; set from env spec when None
    # net
    features_dim: int = 256  # CNN feature width == LSTM hidden size
    # Encoder: "impala" (procgen-paper ImpalaCNN, science default) or
    # "nature" (tiny DQN stem — the throughput-push baseline; ~6-10x
    # cheaper per frame, see policy.NatureCNN). Applies to every model
    # instance in the run (state_dicts differ across nets).
    net: str = "impala"
    # Recurrent core. False (default) = purely-Markov feedforward (see
    # feedback_impala_argmax.md for why Markov matters for argmax rollouts).
    # True = nn.LSTM core between the CNN features and the heads, with the
    # hidden state threaded across unrolls and reset to zero on episode
    # boundaries. State handling matches monobeast: the actor owns its
    # recurrent state, snapshots the per-unroll initial state into
    # initial_agent_state_buffers, and the learner replays from it. In
    # central_gpu mode the state rides through the inference channel (the
    # server stays stateless) — validated against IMPALA/monobeast's
    # state-with-data pattern rather than SEED RL's server-resident state,
    # since our channel is shared memory, not a network hop.
    use_lstm: bool = False
    # Recurrent core between the CNN features and the heads: "ff", "lstm",
    # "deltanet" or "compfwp". Empty falls back to use_lstm, so every existing
    # config keeps its meaning; see impala.net.resolve_core.
    core: str = ""
    # Side of the square fast-weight matrix, for the deltanet/compfwp cores.
    # 128 keeps 64 envs' state under ~4 MB in fp32. Unused by ff/lstm.
    fwp_dim: int = 128
    # Number of read heads the fast-weight cores retrieve per step. A width,
    # not a count of anything in the world. Unused by ff/lstm.
    fwp_heads: int = 8
    # CompFWP ablation table (review §12). joint/joint/delta is the
    # contribution: the delta-rule error is measured after the competitive
    # read. indep/indep/delta reproduces DeltaNet exactly.
    fwp_read: str = "joint"     # joint | indep
    fwp_error: str = "joint"    # joint | indep
    fwp_write: str = "delta"    # delta | additive
    # Per-step multiplicative forgetting on the fast-weight state, applied
    # before each write. 0.0 is off; roughly a 1/fwp_decay step memory horizon.
    fwp_decay: float = 0.0
    # Log pre-clip gradient norm per component (encoder, each core projection,
    # the set block, the heads). Diagnostic: adds a reduction per parameter
    # every step, so it is off unless a run is being measured.
    log_grad_groups: bool = False
    # Force EVERY env reset (initial + auto-reset on done) to this seed.
    # Mirrors PPO's fixed_env_seed: makes the agent memorize one specific
    # instance instead of generalizing across the procedural distribution.
    # If None, each reset gets a fresh seed (random for node_gym, advancing
    # RNG for gymnasium). Set to match the PPO baseline for apples-to-apples
    # comparison on the same task variant.
    fixed_env_seed: int | None = None
    # Binding-level generalization sweep (see playtrain_trainers.generalization). When set,
    # actors are restricted to a finite pool of distinct-binding seeds (held-out
    # "sword=key" configs excluded) instead of the full procedural distribution.
    # Spec dict: {n_train_bindings, split_seed, scan, placements_per_binding}.
    # Mutually exclusive with fixed_env_seed. Held-out win-rate (the Y-axis) is
    # computed post-hoc via tools/eval_generalization.py.
    train_pool: dict | None = None
    # inference architecture:
    #   "shared_cpu" — monobeast-style: actors hold a share_memory_() CPU
    #                  model and run forward locally. Slow on small batches
    #                  (~315 SPS on MiniGrid) but matches torchbeast lineage.
    #   "central_gpu" — SEED-style: actors send obs to a central inference
    #                   thread that batches across actors and runs one GPU
    #                   forward. Targets ~PPO speed (~1500+ SPS).
    #   "vec"         — vectorized rollout workers (node_gym backend only,
    #                   feedforward only): each worker owns a NativeVecEnv of
    #                   batch_size QuickJS envs + its own GPU inference copy,
    #                   one batched forward + one GIL-released vec_step per
    #                   vector step. A rollout fills a whole (T+1, B) slot =
    #                   one learner batch. Replaces the per-actor flag channel
    #                   entirely; targets 100k+ SPS. See impala/vec_actor.py.
    inference_mode: str = "shared_cpu"
    # vec mode: number of rollout worker processes. Envs per worker ==
    # batch_size (one rollout slot = one learner batch). Throughput scales
    # with workers until the GPU (inference+learner) or the CPU pool
    # saturates; 2-4 is the right range on one H100 node.
    vec_workers: int = 2
    # vec mode: C++ env-threadpool size per worker. 0 = the host's default
    # (all perf cores) — right for 1 worker, oversubscribed for several; set
    # ~cores/vec_workers explicitly on shared nodes.
    vec_env_threads: int = 0
    # Env implementation behind vec workers: "native" (playtrain QuickJS,
    # default) or "envpool" (real ProcGen/Atari via EnvPool's C++ batcher —
    # the 1-to-1 layer-2 comparison; requires explicit num_actions, fs=1,
    # single-buffer, seed policies inapplicable). impala/envpool_vec.py.
    vec_backend: str = "native"
    # Extra kwargs forwarded to envpool.make_gymnasium when vec_backend="envpool".
    # Needed to make an ALE row matchable: EnvPool's Atari defaults are 84x84
    # grayscale / stack_num=4 / frame_skip=4, none of which match a PlayTrain
    # replica's 64x64x3 RGB single frame at fs=1. Set
    #   {"img_height":64,"img_width":64,"gray_scale":false,"stack_num":1,"frame_skip":1}
    # so the A/B differs only in the environment, not in what the encoder sees.
    # ProcGen ignores these (natively 64x64x3), so one field serves both.
    envpool_kwargs: dict | None = None
    # vec mode: skip rendering on all but the final tick of each frame_skip
    # window (game logic runs untouched; obs bit-exact — proven invisible by
    # node-gym test_render_skip_is_invisible, valid because vec mode always
    # autoresets). Measured +69% env throughput on cavequest fs=7. Only
    # meaningful when frame_skip > 1.
    vec_render_skip: bool = True
    # vec mode: double-buffered sampling (Sample-Factory style). Each worker
    # runs 2x batch_size envs as two groups on one shared threadpool; group
    # A's env stepping (C++) overlaps group B's batch build + inference
    # (Python/GPU). Slot/alignment semantics identical to the single-buffer
    # path (node-gym test_pingpong_bit_exact_vs_sync + the act_vec_db
    # alignment tests). Doubles per-worker env count — halve vec_workers or
    # env memory accordingly.
    vec_double_buffer: bool = False
    # vec mode: run worker inference under bf16 autocast. Halves the
    # forward's GPU cost — decisive when workers share the GPU with the
    # learner (1-GPU + MPS). Stored logits remain the exact distribution the
    # action was sampled from; behavior-vs-learner numeric drift is precisely
    # what V-trace's off-policy correction handles.
    vec_infer_bf16: bool = False
    # vec mode: capture worker inference as a CUDA graph (one launch per
    # vector step instead of per-op dispatch; ~10-15% less inference GPU
    # time). Weight reloads stay visible (in-place param copies). Falls back
    # to eager if capture fails.
    vec_infer_graphs: bool = False
    # vec mode: games dir for NativeVecEnv (None = node-gym's examples/games/js,
    # the catalog verified bit-exact vs the V8 path by native/gate_qjs.sh).
    vec_games_dir: str | None = None
    # vec mode: device the rollout workers run inference on. None = same as
    # the learner. On multi-GPU nodes set "cuda:1": at 50M-run scale the
    # learner saturates its GPU, and co-located worker inference cost ~2x
    # sustained SPS (job 30293786: 51k co-located vs ~100k component benches).
    vec_worker_device: str | None = None
    # remote_vec mode (SEED-style network-fed rollouts; see impala/remote_vec
    # module docstring). Workers listen on remote_port_base + worker_index;
    # each accepts up to remote_groups_per_worker connections from the
    # external env-actor fleet (tools/remote_env_actor.py on CPU-partition
    # nodes), one connection per group of batch_size envs. Requires explicit
    # num_actions, feedforward net, eval_every_steps=0.
    remote_port_base: int = 23000
    remote_groups_per_worker: int = 4
    # Abort if the fleet never dials in. remote_vec's acceptor otherwise waits
    # forever on a 1s socket timeout, so a remote_vec run launched with no fleet
    # (wrong host/port, hetjob's fleet half never started, actors died on
    # import) holds its GPU allocation idle until the wall clock kills it. The
    # deadline applies only until the FIRST group connects.
    remote_connect_timeout_s: float = 300.0
    # Data-parallel learner ranks (impala/ddp_learner.py): >1 spawns that
    # many learner PROCESSES on cuda:0..N-1 with DDP grad all-reduce; put
    # vec workers on the remaining GPUs via vec_worker_device. vec/
    # remote_vec: eval/save/popart must be off. LSTM is supported — the
    # recurrent state lives on the trainer's inference workers (remote_vec
    # threads it per group), so the env-actor fleet stays torch-free.
    learner_gpus: int = 1
    ddp_rdzv_port: int = 29400
    # torch.compile the learner model (measured +10% on the bf16 learner).
    # First learn steps pay a ~1min compile; steady state is faster.
    compile_learner: bool = False
    # torch.compile mode for the learner: "default", "reduce-overhead"
    # (CUDA-graphs the compiled regions), or "max-autotune" (autotuned
    # kernels + graphs; slowest compile, fastest steady state).
    compile_mode: str = "default"
    # NHWC (channels_last) memory format for the learner's conv stem —
    # numerics-neutral layout change that tensor cores prefer in bf16.
    channels_last: bool = False
    # Central-inference batching window. Server collects requests for up to
    # this many seconds after the first one arrives, then runs the forward.
    # Trade-off: higher = bigger batches (better GPU utilization) but higher
    # actor latency. 5ms is generous for ~4-8 actors.
    inference_batch_timeout_s: float = 0.005
    # Sync the inference model from the learner every N learn steps
    # (central_gpu only). 1 = after every gradient step, the monobeast-
    # equivalent cadence. Each sync is a full-net GPU copy issued from Python
    # under the GIL, so raising this trades a small amount of extra policy lag
    # (already bounded by V-trace's off-policy correction) for learner/
    # inference throughput.
    inference_sync_every: int = 1
    # Sync + log learner stats (losses, episode returns) to TensorBoard every
    # N learn steps. Between syncs the stats stay on the GPU; episode returns
    # are accumulated across the window so none are dropped from
    # charts/ep_return_mean. Every sync stalls the CUDA stream and holds the
    # GIL (starving the inference thread), hence the cadence. 1 reproduces the
    # old every-step logging exactly.
    stats_log_every: int = 10
    # Experiment tracking (Weights & Biases). Off by default -> TensorBoard
    # only, behavior unchanged. When use_wandb=True, wandb.init mirrors the
    # existing TensorBoard scalars (sync_tensorboard) — no extra logging code.
    # Needs `wandb login` once on the machine. See playtrain_trainers.wandb_tracking.
    use_wandb: bool = False
    wandb_project: str = "playtrain"
    wandb_group: str | None = None
    # W&B run title. None -> the log_dir basename (the per-job dir). run_impala.sh
    # sets this to the config filename so the run reads as the config you
    # launched, not the job id. See playtrain_trainers.wandb_tracking.init_wandb.
    wandb_name: str | None = None
    # Greedy (argmax) evaluation logged during training. The behavior policy's
    # training return can saturate while the *deterministic* argmax policy is
    # brittle — logging greedy return/win-rate live surfaces that immediately.
    # Every eval_every_steps env steps, run eval_episodes greedy episodes on a
    # held-out (procedural) or the fixed (memorized) instance and log
    # eval/greedy_return + eval/greedy_win_rate. Set eval_every_steps=0 to
    # disable. Cheap: runs in the monitor thread on a weight snapshot, so it
    # never blocks the learners.
    eval_every_steps: int = 500_000
    eval_episodes: int = 8
    eval_win_threshold: float = 30_000.0
    # Every save_every_steps env steps, snapshot the learner weights to
    # ckpt_<step>.pt (same payload as final.pt) so post-hoc evals can trace a
    # metric vs training steps from a single run. 0 = off (only final.pt, the
    # default). Cheap: one state_dict copy in the monitor thread under the lock.
    save_every_steps: int = 0
    # Resume policy. "auto": load the newest ckpt_*.pt in log_dir (weights +
    # optimizer + scheduler + step) if one exists — this makes preempted
    # kempner_requeue jobs restartable, since SLURM reruns the batch script
    # with the SAME job id and run_impala.sh derives log_dir from it. "off":
    # always start fresh. Any other value: explicit path to a checkpoint.
    # Requires save_every_steps > 0 to have anything to resume from.
    resume: str = "auto"


def _resolve_resume_ckpt(cfg: "ImpalaConfig") -> Path | None:
    if not cfg.resume or cfg.resume == "off":
        return None
    if cfg.resume != "auto":
        p = Path(cfg.resume)
        if not p.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {p}")
        return p
    cands = sorted(Path(cfg.log_dir).glob("ckpt_*.pt"),
                   key=lambda p: int(p.stem.split("_")[1]))
    return cands[-1] if cands else None


def _pick_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def _default_env_fn(cfg: ImpalaConfig) -> Callable[[int], tuple]:
    """Build a callable (actor_index)->(gym_env, initial_seed).

    The initial_seed mixes cfg.seed with actor_index so:
      - Different cfg.seed runs see different env distributions (the bug
        previously masked on node_gym backend where cfg.seed was a no-op).
      - Different actors within a run see different starting episodes
        (encourages behavioral diversity in the rollout pool).

    Stride is 1_000_000 between cfg.seed values — large enough that even
    with 1000s of actors per run they can't collide across runs.
    """
    def _seed_for(actor_index: int) -> int:
        return cfg.seed * 1_000_000 + actor_index

    # Generalization sweep: resolve the finite train-seed pool once. Each actor
    # wraps its env in a SeedSetWrapper, which forces every reset (initial AND
    # auto-reset) to a pool sample, overriding initial_seed / fixed_env_seed.
    train_seeds = None
    if cfg.train_pool is not None:
        if cfg.fixed_env_seed is not None:
            raise ValueError("train_pool and fixed_env_seed are mutually exclusive")
        from playtrain_trainers.plugins import resolve_pools
        train_seeds, _test_seeds = resolve_pools(cfg.train_pool)

    def _maybe_pool(env, actor_index: int):
        if train_seeds is None:
            return env
        from playtrain_trainers.plugins import SeedSetWrapper
        return SeedSetWrapper(env, train_seeds, rng_seed=actor_index)

    if cfg.env_backend == "minigrid":
        from playtrain_trainers.plugins import make_minigrid_env
        def _fn(actor_index: int):
            s = _seed_for(actor_index)
            env = _maybe_pool(make_minigrid_env(cfg.game, s), actor_index)
            return env, s
        return _fn
    if cfg.env_backend in ("playtrain", "node_gym"):  # "node_gym" = legacy alias
        from playtrain.runtime.env import PlayTrainEnv
        env_kwargs = dict(game=cfg.game, frame_skip=cfg.frame_skip,
                          frame_stack=cfg.frame_stack)
        if cfg.max_decisions is not None:
            # node-gym truncates on FRAMES; convert the decision horizon.
            env_kwargs["max_steps"] = cfg.max_decisions * cfg.frame_skip
        def _fn(actor_index: int):
            s = _seed_for(actor_index)
            env = _maybe_pool(PlayTrainEnv(**env_kwargs), actor_index)
            return env, s
        return _fn
    raise ValueError(f"unknown env_backend={cfg.env_backend!r}")


def _infer_action_space(env_fn: Callable[[int], tuple]) -> int:
    probe, _seed = env_fn(0)
    n = int(probe.action_space.n)
    probe.close()
    return n


def _get_batch(
    free_queue: "mp.SimpleQueue",
    full_queue: "mp.SimpleQueue",
    buffers: dict,
    initial_agent_state_buffers: list,
    batch_size: int,
    device: torch.device,
    lock: threading.Lock,
) -> tuple[dict, tuple]:
    with lock:
        indices = [full_queue.get() for _ in range(batch_size)]
    if any(m is None for m in indices):
        return None, None  # shutdown sentinel (see train() finally)
    batch = {
        key: torch.stack([buffers[key][m] for m in indices], dim=1) for key in buffers
    }
    # Stack each slot's per-unroll initial recurrent state along the batch dim
    # (dim=1), matching the (num_layers, B, hidden) the learner forward expects.
    # In feedforward mode every slot is an empty tuple, so zip(*[(), ...]) is
    # empty and initial_agent_state collapses to () — the learner replays from
    # no state, exactly as before. (monobeast get_batch pattern.)
    initial_agent_state = tuple(
        torch.cat(ts, dim=1)
        for ts in zip(*[initial_agent_state_buffers[m] for m in indices])
    )
    for m in indices:
        free_queue.put(m)
    batch = {k: t.to(device=device, non_blocking=True) for k, t in batch.items()}
    initial_agent_state = tuple(
        s.to(device=device, non_blocking=True) for s in initial_agent_state
    )
    return batch, initial_agent_state


def _get_batch_vec(
    free_queue: "mp.SimpleQueue",
    full_queue: "mp.SimpleQueue",
    buffers: dict,
    state_buffers: list,
    device: torch.device,
    lock: threading.Lock,
) -> tuple[dict, tuple]:
    """vec mode: one (T+1, B) slot IS a learner batch — no stacking. The
    slot's recurrent-state snapshot is likewise already (layers, B, hidden);
    () in feedforward mode."""
    with lock:
        index = full_queue.get()
    if index is None:
        return None, None  # shutdown sentinel (see train() finally)
    batch = {k: buffers[k][index].to(device=device, non_blocking=True)
             for k in buffers}
    initial_agent_state = tuple(
        t.to(device=device, non_blocking=True) for t in state_buffers[index]
    )
    # The slots may be cudaHostRegister-pinned (_pin_vec_buffers), making the
    # .to() copies asynchronous DMA — synchronize before recycling the slot so
    # a worker can't overwrite pages mid-transfer. (Unpinned .to() is already
    # synchronous; the extra sync is then a no-op.)
    if device.type == "cuda":
        torch.cuda.current_stream().synchronize()
    free_queue.put(index)
    return batch, initial_agent_state


def check_vec_backend_compatible(vec_backend: str, vec_double_buffer: bool) -> None:
    """Reject vec_double_buffer=True with any non-native vec_backend.

    ``act_vec_db`` (the double-buffered vec worker) constructs ``PingPongVecEnv``
    unconditionally and never reads ``vec_backend`` — only the single-buffered
    ``act_vec`` has the envpool branch. So the combination does not fail; it
    quietly runs PlayTrain's OWN envs while the logs, the TensorBoard run name,
    and the saved ``config.json`` all say ``envpool``.

    That is the one misconfiguration a benchmark cannot self-detect: the run
    completes, learns, and produces a plausible number — with the system under
    test standing in for its own baseline. Hence a hard error rather than a
    warning. Split out of ``train()`` so it is testable without reaching the
    actor spawn below it.
    """
    if vec_double_buffer and vec_backend != "native":
        raise ValueError(
            f"vec_double_buffer=True is only implemented for vec_backend='native' "
            f"(got {vec_backend!r}). act_vec_db ignores vec_backend and would run "
            f"PlayTrain envs, silently mislabelling the run. Either set "
            f"vec_double_buffer=False — which makes the arm single-buffered, and "
            f"must be disclosed in any A/B against a double-buffered arm — or "
            f"teach act_vec_db the {vec_backend!r} branch.")


def _pin_vec_buffers(buffers: dict, state_buffers: list) -> bool:
    """cudaHostRegister every vec slot so learner H2D copies run as direct
    DMA (~5x faster than unpinned paging for the ~157MB batch-128 slots).
    Registers the EXISTING shared-memory pages — no staging copy. Call after
    CUDA init (post-fork), learner process only. Returns False (and leaves
    the unpinned path, which is correct just slower) on any failure."""
    try:
        cudart = torch.cuda.cudart()
        tensors = [t for ts in buffers.values() for t in ts]
        tensors += [t for slot in state_buffers for t in slot]
        for t in tensors:
            r = cudart.cudaHostRegister(
                t.data_ptr(), t.numel() * t.element_size(), 0)
            if int(r) != 0:
                logging.warning("cudaHostRegister failed (rc=%s); vec slots "
                                "stay unpinned.", r)
                return False
        logging.info("Pinned %d vec buffer tensors for DMA H2D.", len(tensors))
        return True
    except Exception as e:  # noqa: BLE001
        logging.warning("Buffer pinning unavailable (%s); unpinned path.", e)
        return False


def train(cfg: ImpalaConfig, env_fn: Callable[[int], "object"] | None = None) -> dict:
    """Run async IMPALA. Returns a summary dict (final step count, stats)."""
    log_dir = Path(cfg.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    # Frame-stacking: node-gym concatenates `frame_stack` RGB frames on the
    # channel axis (env.py _stacked_obs), so the net sees 3*frame_stack channels.
    # Derive obs_shape from frame_stack (idempotent — always 3 per RGB frame) so
    # the conv stem + shared-mem buffers + eval all size to match the env; every
    # downstream consumer reads cfg.obs_shape.
    if cfg.frame_stack > 1:
        cfg.obs_shape = (3 * cfg.frame_stack, cfg.obs_shape[1], cfg.obs_shape[2])
        logging.info("frame_stack=%d -> obs_shape=%s", cfg.frame_stack, cfg.obs_shape)
    (log_dir / "config.json").write_text(
        json.dumps({k: v for k, v in dataclasses.asdict(cfg).items()}, indent=2,
                   default=str)
    )
    # Start W&B before the SummaryWriter so sync_tensorboard mirrors every
    # scalar (no-op unless cfg.use_wandb). TensorBoard stays the source of truth.
    wandb_run = init_wandb(cfg, log_dir)
    writer = SummaryWriter(str(log_dir / "tb"))

    device = _pick_device(cfg.device)
    if device.type == "cuda":
        _tc = cfg.learner_precision in ("tf32", "bf16")
        torch.backends.cuda.matmul.allow_tf32 = _tc
        torch.backends.cudnn.allow_tf32 = _tc
    # Build the probe/eval env factory LAZILY: vec/remote configs with
    # explicit num_actions and eval off never touch it, and constructing it
    # imports the playtrain runtime (gymnasium>=1.0) — which must not be a
    # hard dependency of e.g. the envpool-backend comparison runs.
    if env_fn is None and (cfg.num_actions is None or cfg.eval_every_steps > 0):
        env_fn = _default_env_fn(cfg)
    if cfg.inference_mode not in ("shared_cpu", "central_gpu", "vec",
                                  "remote_vec"):
        raise ValueError(
            f"inference_mode must be 'shared_cpu', 'central_gpu', 'vec' or "
            f"'remote_vec', got {cfg.inference_mode!r}"
        )
    # remote_vec: envs live on OTHER nodes (tools/remote_env_actor.py fleets
    # on cheap CPU partitions); workers are network-fed but fill the same
    # slots, so everything downstream of the queues is shared with vec mode.
    remote_mode = cfg.inference_mode == "remote_vec"
    if remote_mode:
        # num_actions can only be inferred from a local env, and remote mode has
        # none unless eval is on (which builds one above).
        if cfg.num_actions is None and cfg.eval_every_steps <= 0:
            raise ValueError("remote_vec needs explicit num_actions "
                             "(envs are remote; no probe env)")
        if cfg.eval_every_steps > 0:
            # The rollout envs live on the actor fleet, so the greedy eval needs
            # its own LOCAL env. Prove it can be built HERE and now — before any
            # worker process exists — so a missing games dir on the trainer node
            # fails immediately instead of after the fleet has connected.
            try:
                _probe, _ = env_fn(10_000)
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    f"remote_vec: cannot build the local env needed for greedy "
                    f"eval ({type(exc).__name__}: {exc}). The rollout envs are "
                    f"remote, so eval needs a local copy of {cfg.game!r} on the "
                    f"trainer node — check PLAYTRAIN_GAMES_DIR / vec_games_dir, "
                    f"pass env_fn=, or set eval_every_steps=0 and evaluate "
                    f"final.pt in a separate pass.") from exc
            with contextlib.suppress(Exception):
                _probe.close()
    if cfg.num_actions is None:
        cfg.num_actions = _infer_action_space(env_fn)
    vec_mode = cfg.inference_mode in ("vec", "remote_vec")
    ddp_mode = cfg.learner_gpus > 1
    if ddp_mode:
        if not vec_mode or cfg.use_popart:
            raise ValueError("learner_gpus>1 needs vec/remote_vec mode, "
                             "no popart")
        if cfg.eval_every_steps or cfg.save_every_steps:
            raise ValueError("learner_gpus>1: set eval_every_steps=0 and "
                             "save_every_steps=0")
    if vec_mode:
        if not remote_mode and cfg.env_backend not in ("playtrain", "node_gym"):
            raise ValueError("inference_mode='vec' requires env_backend='playtrain'")
        check_vec_backend_compatible(cfg.vec_backend, cfg.vec_double_buffer)
        # One rollout slot = one learner batch; a handful per worker (or per
        # remote group) keeps the learner fed while rollouts are mid-flight.
        if cfg.num_buffers is None:
            cfg.num_buffers = (3 * cfg.vec_workers * cfg.remote_groups_per_worker
                               if remote_mode else 4 * cfg.vec_workers)
    else:
        if cfg.num_buffers is None:
            cfg.num_buffers = max(2 * cfg.num_actors, cfg.batch_size)
        if cfg.num_actors >= cfg.num_buffers:
            raise ValueError("num_buffers should be larger than num_actors")
        if cfg.num_buffers < cfg.batch_size:
            raise ValueError("num_buffers should be larger than batch_size")

    torch.manual_seed(cfg.seed)

    if cfg.learner_precision not in ("fp32", "tf32", "bf16"):
        raise ValueError(f"unknown learner_precision={cfg.learner_precision!r}")

    # Shared CPU model: needed in shared_cpu mode for actor forwards. In
    # central_gpu mode it's still allocated as the "weight seed" so we can
    # initialize the learner_model from it deterministically, but actors
    # never call .forward() on it.
    model = ImpalaNet(cfg.obs_shape, cfg.num_actions,
                      features_dim=cfg.features_dim, use_lstm=cfg.use_lstm,
                      use_popart=cfg.use_popart, net=cfg.net,
                      core=cfg.core, fwp_dim=cfg.fwp_dim, fwp_heads=cfg.fwp_heads,
                      fwp_read=cfg.fwp_read, fwp_error=cfg.fwp_error,
                      fwp_write=cfg.fwp_write, fwp_decay=cfg.fwp_decay)
    # Resume BEFORE workers spawn / weight_state is created, so actors start
    # from the resumed weights; optimizer/scheduler/step restore below, after
    # they exist.
    resume_state = None
    _resume_path = _resolve_resume_ckpt(cfg)
    if _resume_path is not None:
        resume_state = torch.load(_resume_path, map_location="cpu")
        model.load_state_dict(resume_state["model_state_dict"])
        logging.info("RESUME: loaded %s (step=%s)", _resume_path,
                     resume_state.get("step", "?"))
    vec_state_buffers: list = []
    if vec_mode:
        from playtrain_trainers.impala.vec_actor import (
            create_vec_buffers,
            create_vec_state_buffers,
        )
        buffers = create_vec_buffers(
            cfg.obs_shape, cfg.num_actions, cfg.unroll_length,
            cfg.batch_size, cfg.num_buffers,
            frame_hwc=remote_mode,  # zero-copy wire ingest; GPU-side permute
        )
        # Per-slot (h, c) snapshots, (layers, batch_size, hidden) each; ()
        # tuples in feedforward mode. Allocated BEFORE fork (shared memory).
        vec_state_buffers = create_vec_state_buffers(
            model, cfg.batch_size, cfg.num_buffers
        )
    else:
        buffers = create_buffers(
            cfg.obs_shape, cfg.num_actions, cfg.unroll_length, cfg.num_buffers
        )
    # Per-slot recurrent state snapshots (monobeast initial_agent_state_buffers).
    # Empty tuples per slot in feedforward mode. Allocated BEFORE fork so actor
    # children inherit the shared-memory mappings.
    initial_agent_state_buffers = create_initial_agent_state_buffers(
        model, cfg.num_buffers
    )
    if cfg.inference_mode == "shared_cpu":
        model.share_memory()

    T, B = cfg.unroll_length, cfg.batch_size

    # CRITICAL (fork modes): fork actors BEFORE touching CUDA. PyTorch's CUDA
    # context can't cross fork() cleanly — children that inherit an
    # initialized CUDA state hang on their first cuda-adjacent operation
    # (job 16733100).
    #
    # vec mode SPAWNS instead: its workers create their own CUDA contexts,
    # and even torch's parent-side device probe can poison fork for that
    # ("Cannot re-initialize CUDA in forked subprocess", job 30285906).
    # Spawned children re-import cleanly; the shared-memory buffers /
    # weight_state / queues all pass through spawn pickling.
    ctx = mp.get_context("spawn" if vec_mode else "fork")
    free_queue = ctx.SimpleQueue()
    full_queue = ctx.SimpleQueue()

    # vec mode: shared weight-publication state, allocated BEFORE fork so
    # workers inherit the mappings. Holds the initial weights already.
    weight_state = None
    if vec_mode:
        from playtrain_trainers.impala.vec_actor import create_weight_state, publish_weights
        weight_state = create_weight_state(model)

    # Build the inference channel for central_gpu mode BEFORE fork so the
    # shared-memory mappings are inherited by children. Skipped in shared_cpu
    # mode (where actors use the shared CPU model directly).
    channel = None
    if cfg.inference_mode == "central_gpu":
        channel = create_channel(
            cfg.num_actors, cfg.obs_shape, cfg.num_actions, ctx,
            use_lstm=cfg.use_lstm, features_dim=cfg.features_dim,
        )
        logging.info("Central inference channel created (n_actors=%d, A=%d, "
                     "use_lstm=%s).", cfg.num_actors, cfg.num_actions, cfg.use_lstm)

    # Zero recurrent state template the actors clone at startup. () in
    # feedforward mode. CPU tensors, inherited across fork.
    initial_core_state = model.initial_state(batch_size=1)

    actor_processes: list = []
    if remote_mode:
        from playtrain_trainers.impala.remote_vec import act_remote
        model_spec = dict(obs_shape=cfg.obs_shape, num_actions=cfg.num_actions,
                          features_dim=cfg.features_dim,
                          use_lstm=cfg.use_lstm, net=cfg.net,
                          core=cfg.core, fwp_dim=cfg.fwp_dim, fwp_heads=cfg.fwp_heads,
                      fwp_read=cfg.fwp_read, fwp_error=cfg.fwp_error,
                      fwp_write=cfg.fwp_write, fwp_decay=cfg.fwp_decay)
        _wdevs = (cfg.vec_worker_device or str(device)).split(",")
        for i in range(cfg.vec_workers):
            remote_spec = dict(
                port=cfg.remote_port_base + i,
                groups=cfg.remote_groups_per_worker,
                num_envs=cfg.batch_size, obs_size=cfg.obs_shape[1],
                infer_bf16=cfg.vec_infer_bf16,
                sync_every=cfg.inference_sync_every,
                connect_timeout_s=cfg.remote_connect_timeout_s,
            )
            p = ctx.Process(
                target=act_remote,
                args=(i, free_queue, full_queue, buffers, vec_state_buffers,
                      weight_state, remote_spec, model_spec,
                      cfg.unroll_length, _wdevs[i % len(_wdevs)].strip()),
            )
            p.start()
            actor_processes.append(p)
        logging.info("remote_vec: %d workers listening on ports %d-%d "
                     "(%d groups x %d envs each) — start the actor fleet "
                     "(tools/remote_env_actor.py) against this host.",
                     cfg.vec_workers, cfg.remote_port_base,
                     cfg.remote_port_base + cfg.vec_workers - 1,
                     cfg.remote_groups_per_worker, cfg.batch_size)
    elif vec_mode:
        from playtrain_trainers.plugins import resolve_pools
        from playtrain_trainers.impala.vec_actor import act_vec, act_vec_db
        vec_target = act_vec_db if cfg.vec_double_buffer else act_vec
        # Seed policy for every episode (initial resets + host-side autoreset).
        if cfg.train_pool is not None:
            train_seeds, _ = resolve_pools(cfg.train_pool)
            seed_mode, seed_pool, fixed = "pool", list(train_seeds), None
        elif cfg.fixed_env_seed is not None:
            seed_mode, seed_pool, fixed = "fixed", None, cfg.fixed_env_seed
        else:
            seed_mode, seed_pool, fixed = "formula", None, None
        game_path = cfg.game
        if cfg.vec_games_dir is not None and not game_path.endswith(".js"):
            game_path = str(Path(cfg.vec_games_dir) / f"{cfg.game}.js")
        # node-gym max_steps counts FRAMES (same convention as PlayTrainEnv);
        # convert the decision horizon like _default_env_fn does.
        max_steps = (cfg.max_decisions * cfg.frame_skip
                     if cfg.max_decisions is not None else 2000)
        model_spec = dict(obs_shape=cfg.obs_shape, num_actions=cfg.num_actions,
                          features_dim=cfg.features_dim, use_lstm=cfg.use_lstm,
                          use_popart=cfg.use_popart, net=cfg.net,
                          core=cfg.core, fwp_dim=cfg.fwp_dim, fwp_heads=cfg.fwp_heads,
                      fwp_read=cfg.fwp_read, fwp_error=cfg.fwp_error,
                      fwp_write=cfg.fwp_write, fwp_decay=cfg.fwp_decay)
        for i in range(cfg.vec_workers):
            env_spec = dict(
                game_path=game_path, num_envs=cfg.batch_size,
                frame_skip=cfg.frame_skip, frame_stack=cfg.frame_stack,
                max_steps=max_steps, obs_size=cfg.obs_shape[1],
                env_threads=cfg.vec_env_threads,
                vec_backend=cfg.vec_backend,
                envpool_kwargs=cfg.envpool_kwargs,
                render_skip=cfg.vec_render_skip,
                infer_bf16=cfg.vec_infer_bf16,
                infer_graphs=cfg.vec_infer_graphs,
                seed_mode=seed_mode, seed_pool=seed_pool, fixed_seed=fixed,
                base_seed=cfg.seed * 1_000_000 + i * 10_000,
            )
            # vec_worker_device accepts a comma-list ("cuda:1,cuda:2,cuda:3");
            # workers round-robin across it. One inference GPU saturates
            # around ~8 workers at NatureCNN batch-128 — full-node (4-GPU)
            # runs spread 12-16 workers over the 3 non-learner GPUs.
            _wdevs = (cfg.vec_worker_device or str(device)).split(",")
            p = ctx.Process(
                target=vec_target,
                args=(i, free_queue, full_queue, buffers, vec_state_buffers,
                      weight_state, env_spec, model_spec, cfg.unroll_length,
                      _wdevs[i % len(_wdevs)].strip()),
            )
            p.start()
            actor_processes.append(p)
    for i in range(cfg.num_actors if not vec_mode else 0):
        if cfg.inference_mode == "shared_cpu":
            p = ctx.Process(
                target=act,
                args=(i, free_queue, full_queue, model, buffers,
                      initial_agent_state_buffers, env_fn,
                      cfg.unroll_length, True, cfg.fixed_env_seed),
            )
        else:  # central_gpu
            p = ctx.Process(
                target=act_central,
                args=(i, free_queue, full_queue, channel, buffers,
                      initial_agent_state_buffers, initial_core_state, env_fn,
                      cfg.unroll_length, True, cfg.fixed_env_seed),
            )
        p.start()
        actor_processes.append(p)

    # Now safe to initialize CUDA in the main process (actors already forked
    # off a CUDA-clean parent).
    learner_model = ImpalaNet(cfg.obs_shape, cfg.num_actions,
                              features_dim=cfg.features_dim,
                              use_lstm=cfg.use_lstm,
                              channels_last=cfg.channels_last,
                              use_popart=cfg.use_popart,
                              net=cfg.net, core=cfg.core,
                              fwp_dim=cfg.fwp_dim, fwp_heads=cfg.fwp_heads,
                      fwp_read=cfg.fwp_read, fwp_error=cfg.fwp_error,
                      fwp_write=cfg.fwp_write, fwp_decay=cfg.fwp_decay).to(device)
    learner_model.load_state_dict(model.state_dict())
    if cfg.channels_last:
        learner_model = learner_model.to(memory_format=torch.channels_last)
    if cfg.compile_learner:
        # In-place Module.compile() (NOT torch.compile(module), whose wrapper
        # prefixes state_dict keys with _orig_mod. and would break weight
        # publication, eval snapshots, and final.pt loading).
        learner_model.compile(mode=cfg.compile_mode)
    if vec_mode and device.type == "cuda":
        _pin_vec_buffers(buffers, vec_state_buffers)

    # In central_gpu mode, also build an inference-side GPU model. Separate
    # from learner_model to avoid lock contention during learner forward+
    # backward (which would block all actor inference requests).
    inference_server: InferenceServer | None = None
    inference_thread: threading.Thread | None = None
    if cfg.inference_mode == "central_gpu":
        inference_model = ImpalaNet(cfg.obs_shape, cfg.num_actions,
                                    features_dim=cfg.features_dim,
                                    use_lstm=cfg.use_lstm,
                                    use_popart=cfg.use_popart,
                                    net=cfg.net, core=cfg.core,
                                    fwp_dim=cfg.fwp_dim, fwp_heads=cfg.fwp_heads,
                      fwp_read=cfg.fwp_read, fwp_error=cfg.fwp_error,
                      fwp_write=cfg.fwp_write, fwp_decay=cfg.fwp_decay).to(device)
        inference_model.load_state_dict(learner_model.state_dict())
        inference_server = InferenceServer(
            model=inference_model,
            channel=channel,
            device=device,
            obs_shape=cfg.obs_shape,
            num_actions=cfg.num_actions,
            batch_timeout_s=cfg.inference_batch_timeout_s,
            stochastic_actions=True,  # multinomial sampling for training rollouts
        )
        inference_thread = threading.Thread(
            target=inference_server.run, name="inference-server", daemon=False,
        )
        inference_thread.start()

    optimizer = torch.optim.RMSprop(
        learner_model.parameters(),
        lr=cfg.learning_rate,
        momentum=cfg.rmsprop_momentum,
        eps=cfg.rmsprop_epsilon,
        alpha=cfg.rmsprop_alpha,
    )

    def lr_lambda(epoch: int) -> float:
        return 1 - min(epoch * T * B, cfg.total_steps) / cfg.total_steps
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    step = 0
    if resume_state is not None:
        if "optimizer_state_dict" in resume_state:
            optimizer.load_state_dict(resume_state["optimizer_state_dict"])
        if "scheduler_state_dict" in resume_state:
            scheduler.load_state_dict(resume_state["scheduler_state_dict"])
        step = int(resume_state.get("step", 0))
        resume_state = None  # free the CPU copies
    stats: dict = {}
    step_lock = threading.Lock()
    batch_lock = threading.Lock()
    learn_lock = threading.Lock()
    publish_lock = threading.Lock()
    # Staged warmup: each learner thread's FIRST learn() — the one that runs
    # torch.compile and RECORDS its CUDA graph — executes alone, in thread
    # order, and no thread enters steady state until every thread has
    # recorded. cudagraph-trees recording is not safe against concurrent
    # replay from another thread: overlapping them dies with
    # "beginAllocateToPool: already recording to mempool_id" or "cuDNN
    # DEVICE_ALLOCATION_FAILED during capture" (reproduced on cold-cache
    # H100 and H200 nodes; warm caches sometimes just never overlapped).
    first_learn_done = [threading.Event()
                        for _ in range(cfg.num_learner_threads)]

    def batch_and_learn(thread_idx: int) -> None:
        nonlocal step, stats
        if thread_idx != 0:
            first_learn_done[thread_idx - 1].wait()
        learn_steps = 0
        # Episode-return tensors accumulated (on device) between stat syncs so
        # the cadence drops no returns from charts/ep_return_mean.
        pending_returns: list[torch.Tensor] = []
        while step < cfg.total_steps:
            if vec_mode:
                batch, initial_agent_state = _get_batch_vec(
                    free_queue, full_queue, buffers, vec_state_buffers,
                    device, batch_lock
                )
                if remote_mode and batch is not None:
                    # Remote slots store frames HWC (zero-copy wire ingest);
                    # recover the (T+1, B, C, H, W) contract as a free view —
                    # the memory IS channels_last, which the learner prefers.
                    batch["frame"] = batch["frame"].permute(0, 1, 4, 2, 3)
            else:
                batch, initial_agent_state = _get_batch(
                    free_queue, full_queue, buffers, initial_agent_state_buffers,
                    cfg.batch_size, device, batch_lock
                )
            if batch is None:
                break  # shutdown sentinel
            # bf16: autocast the learn step (forward+loss run conv/matmul in
            # bf16; parameters, optimizer math and reductions stay fp32).
            autocast = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if cfg.learner_precision == "bf16" and device.type == "cuda"
                else contextlib.nullcontext()
            )
            # Entropy annealing (linear, based on progress). `step` is read
            # lock-free — annealing is smooth, an off-by-one-batch step is fine.
            if cfg.entropy_cost_final is not None:
                _anneal_span = max(1.0, cfg.total_steps * cfg.entropy_anneal_frac)
                _frac = min(step / _anneal_span, 1.0)
                cur_entropy = (cfg.entropy_cost
                               + _frac * (cfg.entropy_cost_final - cfg.entropy_cost))
            else:
                cur_entropy = cfg.entropy_cost
            with autocast:
                new_stats = learn(
                    # Central mode: actors never forward on the shared CPU
                    # model, so skip the full GPU->CPU state_dict copy per
                    # gradient step.
                    actor_model=model if cfg.inference_mode == "shared_cpu" else None,
                    learner_model=learner_model,
                    batch=batch,
                    initial_agent_state=initial_agent_state,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    discounting=cfg.discounting,
                    baseline_cost=cfg.baseline_cost,
                    entropy_cost=cur_entropy,
                    grad_norm_clipping=cfg.grad_norm_clipping,
                    log_grad_groups=cfg.log_grad_groups,
                    reward_clipping=cfg.reward_clipping,
                    win_bonus=cfg.win_bonus,
                    win_bonus_threshold=cfg.win_bonus_threshold,
                    win_bonus_slope=cfg.win_bonus_slope,
                    win_bonus_max=cfg.win_bonus_max,
                    use_popart=cfg.use_popart,
                    popart_beta=cfg.popart_beta,
                    lock=learn_lock,
                )
            learn_steps += 1
            if learn_steps == 1:
                first_learn_done[thread_idx].set()
                # Hold until ALL threads have recorded their graphs — a
                # steady-state replay here would race the next thread's
                # recording (see warmup note above).
                first_learn_done[-1].wait()
            with step_lock:
                step += T * B
                pending_returns.append(new_stats["episode_returns"])
                cur_step = step
            if learn_steps % cfg.stats_log_every == 0:
                # The one place this thread pays the GPU->CPU sync: cat the
                # windowed returns, .item() the latest losses, log, publish.
                rets = torch.cat(pending_returns)
                pending_returns = []
                # Win-rate: fraction of the windowed TRAINING episodes whose
                # (raw) return cleared the win threshold. This is the honest
                # headline metric for a win-based task — it measures success
                # directly and, unlike raw ep_return_mean, isn't amplified by
                # the win's magnitude, so it reads the true (much steadier)
                # learning signal. From the same episode returns, no extra sync.
                _win_thr = cfg.eval_win_threshold if cfg.eval_win_threshold is not None else 30_000.0
                win_rate = (
                    float((rets >= _win_thr).float().mean().item())
                    if rets.numel() > 0 else float("nan")
                )
                synced = {
                    "episode_returns": tuple(rets.cpu().numpy()),
                    "mean_episode_return": (
                        float(rets.mean().item()) if rets.numel() > 0
                        else float("nan")
                    ),
                    "win_rate": win_rate,
                    "total_loss": float(new_stats["total_loss"].item()),
                    "pg_loss": float(new_stats["pg_loss"].item()),
                    "baseline_loss": float(new_stats["baseline_loss"].item()),
                    "entropy_loss": float(new_stats["entropy_loss"].item()),
                }
                # Fast-weight state size, present only for the matrix cores.
                # This is the trace the collapse probe reads: the state is
                # cleared only on done, so over 5000-step episodes it is the
                # one thing that differs structurally from the LSTM.
                for _k in ("grad_norm", "core_out_scale"):
                    if _k in new_stats:
                        synced[_k] = float(new_stats[_k].item())
                for _k in [k for k in new_stats if k.startswith("gradgrp/")]:
                    synced[_k] = float(new_stats[_k].item())
                for _k in ("fwp_state_norm_mean", "fwp_state_norm_max"):
                    if _k in new_stats:
                        synced[_k] = float(new_stats[_k].item())
                with step_lock:
                    stats = synced
                    writer.add_scalar("losses/total", synced["total_loss"], cur_step)
                    writer.add_scalar("losses/pg", synced["pg_loss"], cur_step)
                    writer.add_scalar(
                        "losses/baseline", synced["baseline_loss"], cur_step)
                    writer.add_scalar(
                        "losses/entropy", synced["entropy_loss"], cur_step)
                    if synced["episode_returns"]:
                        writer.add_scalar(
                            "charts/ep_return_mean",
                            synced["mean_episode_return"], cur_step,
                        )
                        writer.add_scalar(
                            "charts/ep_win_rate", synced["win_rate"], cur_step,
                        )
                    writer.add_scalar("charts/entropy_cost", cur_entropy, cur_step)
                    for _k, _tag in (("grad_norm", "charts/grad_norm"),
                                     ("core_out_scale", "charts/core_out_scale")):
                        if _k in synced:
                            writer.add_scalar(_tag, synced[_k], cur_step)
                    for _k in [k for k in synced if k.startswith("gradgrp/")]:
                        writer.add_scalar(_k, synced[_k], cur_step)
                    if "fwp_state_norm_mean" in synced:
                        writer.add_scalar(
                            "fwp/state_norm_mean", synced["fwp_state_norm_mean"], cur_step)
                        writer.add_scalar(
                            "fwp/state_norm_max", synced["fwp_state_norm_max"], cur_step)
            # In central_gpu mode, push fresh weights to the inference model
            # so subsequent actor requests reflect the just-trained policy.
            # Device-to-device copies; cadence configurable via
            # inference_sync_every (1 = every gradient step, monobeast-style).
            if (inference_server is not None
                    and learn_steps % cfg.inference_sync_every == 0):
                inference_server.sync_weights_from(learner_model)
            # vec mode: publish weights to the shared CPU tensors the rollout
            # workers reload from at rollout boundaries. The seqlock assumes a
            # SINGLE writer, so publishers serialize on publish_lock.
            if (weight_state is not None
                    and learn_steps % cfg.inference_sync_every == 0):
                with publish_lock:
                    publish_weights(weight_state, learner_model)
            if learn_steps <= 5 or learn_steps % 25 == 0:
                log_rss("learner", thread_idx, step=step, learn_steps=learn_steps)
            # Force the TB SummaryWriter to flush its in-memory scalar
            # buffer to disk every 100 learn steps. Without this, the
            # default flush_secs=120 lets the buffer accumulate scalars
            # between writes — observed as ~6 MB / 100K env steps of
            # slow drift in the main process RSS over a 15M run. At 16
            # actors × 30M+ steps that drift would breach an 8G cgroup;
            # explicit flushes keep RSS bounded.
            if learn_steps % 100 == 0:
                writer.flush()

    # Prime free queue with all buffer slots.
    for m in range(cfg.num_buffers):
        free_queue.put(m)

    def _learner_thread_main(thread_idx: int) -> None:
        # A learner thread dying (compile crash, CUDA fault) must END the run,
        # not zombie it: without this, the monitor loop waited at a frozen
        # step until the SLURM wall — 50 min of idle GPU per incident.
        nonlocal step
        try:
            batch_and_learn(thread_idx)
        except Exception:
            logging.exception("learner-%d died; aborting run", thread_idx)
            with step_lock:
                step = cfg.total_steps
            for ev in first_learn_done:  # unblock warmup-gated threads
                ev.set()

    # DDP mode: learner RANKS are processes (one GPU each), not threads —
    # they pull whole slots data-parallel and all-reduce grads in backward
    # (see impala/ddp_learner.py). The monitor mirrors the shared step
    # counter into `step` each tick so all logging/shutdown logic is common.
    threads = []
    rank_processes: list = []
    ddp_step_value = None
    if ddp_mode:
        from playtrain_trainers.impala.ddp_learner import ddp_learner
        ddp_step_value = ctx.Value("l", step)
        cfg_d = dict(
            obs_shape=tuple(cfg.obs_shape), num_actions=cfg.num_actions,
            features_dim=cfg.features_dim, net=cfg.net,
            core=cfg.core, fwp_dim=cfg.fwp_dim, fwp_heads=cfg.fwp_heads,
                      fwp_read=cfg.fwp_read, fwp_error=cfg.fwp_error,
                      fwp_write=cfg.fwp_write, fwp_decay=cfg.fwp_decay,
            unroll_length=cfg.unroll_length, batch_size=cfg.batch_size,
            total_steps=cfg.total_steps, discounting=cfg.discounting,
            baseline_cost=cfg.baseline_cost, entropy_cost=cfg.entropy_cost,
            grad_norm_clipping=cfg.grad_norm_clipping,
            learning_rate=cfg.learning_rate,
            rmsprop_alpha=cfg.rmsprop_alpha,
            rmsprop_epsilon=cfg.rmsprop_epsilon,
            learner_precision=cfg.learner_precision,
            compile_mode=(cfg.compile_mode if cfg.compile_learner else "off"),
            frame_hwc=remote_mode, sync_every=cfg.inference_sync_every,
            stats_log_every=cfg.stats_log_every, log_dir=cfg.log_dir,
        )
        for r in range(cfg.learner_gpus):
            p = ctx.Process(
                target=ddp_learner,
                args=(r, cfg.learner_gpus, cfg.ddp_rdzv_port, cfg_d,
                      free_queue, full_queue, buffers, vec_state_buffers,
                      weight_state, ddp_step_value),
            )
            p.start()
            rank_processes.append(p)
        logging.info("DDP: %d learner ranks on cuda:0-%d", cfg.learner_gpus,
                     cfg.learner_gpus - 1)
    else:
        for i in range(cfg.num_learner_threads):
            t = threading.Thread(
                target=_learner_thread_main, name=f"learner-{i}", args=(i,)
            )
            t.start()
            threads.append(t)

    # Greedy-eval setup. Runs in this monitor thread on a weight snapshot +
    # a dedicated eval env, so it never blocks the learner threads.
    eval_model = None
    eval_gym_env = None
    eval_seed_list: list[int] = []
    last_eval_step = step
    last_save_step = step
    if cfg.eval_every_steps > 0:
        eval_model = ImpalaNet(cfg.obs_shape, cfg.num_actions,
                               features_dim=cfg.features_dim,
                               use_lstm=cfg.use_lstm,
                               use_popart=cfg.use_popart,
                               net=cfg.net, core=cfg.core,
                               fwp_dim=cfg.fwp_dim, fwp_heads=cfg.fwp_heads,
                      fwp_read=cfg.fwp_read, fwp_error=cfg.fwp_error,
                      fwp_write=cfg.fwp_write, fwp_decay=cfg.fwp_decay).to(device)
        eval_gym_env, _ = env_fn(10_000)  # dedicated; seed set per episode
        eval_seed_list = eval_seeds(cfg.fixed_env_seed, cfg.eval_episodes)

    logging.info("Main loop starting%s.", rss_note())

    timer = timeit.default_timer
    aborted_by_dead_actor = False
    try:
        last_log = timer()
        while step < cfg.total_steps:
            if ddp_mode:
                step = ddp_step_value.value
            start_step, start_t = step, timer()
            time.sleep(5)
            if ddp_mode:
                step = ddp_step_value.value
            sps = (step - start_step) / max(timer() - start_t, 1e-9)

            # Bail fast if an actor died (OOM-killed, segfault, etc.). Without
            # this, learner threads block forever on full_queue.get() while
            # the main thread keeps logging the same step until SLURM's
            # wall-time hits. See job 16739061 — frozen 23 min after OOM.
            dead_actors = [
                (i, p.exitcode) for i, p in enumerate(actor_processes)
                if not p.is_alive()
            ] + [
                (f"rank{i}", p.exitcode) for i, p in enumerate(rank_processes)
                if not p.is_alive()
            ]
            if dead_actors:
                logging.error("Actor(s) died: %s — aborting.", dead_actors)
                aborted_by_dead_actor = True
                break

            # Periodic greedy (argmax) eval — snapshot weights under the lock
            # (fast), then eval off to the side without holding it.
            if eval_model is not None and step - last_eval_step >= cfg.eval_every_steps:
                last_eval_step = step
                with learn_lock:
                    eval_model.load_state_dict(learner_model.state_dict())
                win_rate, greedy_ret = greedy_eval(
                    eval_model, eval_gym_env, seeds=eval_seed_list,
                    device=device,
                    max_steps=(cfg.max_decisions if cfg.max_decisions is not None else 2000),
                    win_threshold=cfg.eval_win_threshold,
                )
                writer.add_scalar("eval/greedy_return", greedy_ret, step)
                writer.add_scalar("eval/greedy_win_rate", win_rate, step)
                logging.info("greedy eval @ step=%d: return=%.0f win_rate=%.2f",
                             step, greedy_ret, win_rate)

            # Periodic checkpoint — snapshot weights under the lock (fast), write
            # ckpt_<step>.pt off-lock. Enables a metric-vs-steps trace from one
            # run (e.g. the sprite-swap generalization curve). Same payload as
            # final.pt so downstream evaluation loads it unchanged.
            if cfg.save_every_steps > 0 and step - last_save_step >= cfg.save_every_steps:
                last_save_step = step
                with learn_lock:
                    ckpt_sd = {k: v.detach().cpu().clone()
                               for k, v in learner_model.state_dict().items()}
                    # Optimizer/scheduler snapshots make the ckpt resumable
                    # (RESUME path above). deepcopy under the lock so learner
                    # threads can't mutate RMSprop state mid-serialization.
                    opt_sd = copy.deepcopy(optimizer.state_dict())
                    sched_sd = scheduler.state_dict()
                torch.save({"model_state_dict": ckpt_sd,
                            "optimizer_state_dict": opt_sd,
                            "scheduler_state_dict": sched_sd,
                            "step": step,
                            "config": dataclasses.asdict(cfg)},
                           log_dir / f"ckpt_{step}.pt")
                logging.info("saved ckpt_%d.pt (resumable)", step)

            if timer() - last_log > 60:
                main_rss = rss_mb()
                actor_rss = [rss_mb(p.pid) for p in actor_processes]
                # rss_mb reads /proc and returns NaN off Linux. Logging "nanMB" there
                # reads as a bug in the first line a new user sees, so drop the fields.
                if main_rss == main_rss:
                    logging.info(
                        "step=%d sps=%.1f total_rss=%.0fMB main=%.0f actors=%s",
                        step, sps, main_rss + sum(r for r in actor_rss if r == r), main_rss,
                        [f"{r:.0f}" for r in actor_rss],
                    )
                else:
                    logging.info("step=%d sps=%.1f", step, sps)
                logging.info("stats=%s", pprint.pformat(stats))
                writer.add_scalar("charts/sps", sps, step)
                if main_rss == main_rss:  # RSS is unavailable off Linux
                    writer.add_scalar("mem/total_rss_mb",
                                      main_rss + sum(r for r in actor_rss if r == r), step)
                    writer.add_scalar("mem/main_rss_mb", main_rss, step)
                    for i, r in enumerate(actor_rss):
                        if r == r:
                            writer.add_scalar(f"mem/actor{i}_rss_mb", r, step)
                last_log = timer()
    except KeyboardInterrupt:
        pass
    finally:
        # Ask learner threads to stop. They check `step < total_steps`, so we
        # bump step past the threshold to break their loop in case we exited
        # via the dead-actor abort path.
        if aborted_by_dead_actor:
            with step_lock:
                step = cfg.total_steps
            if ddp_step_value is not None:
                ddp_step_value.value = cfg.total_steps
        # Unblock learner threads/ranks parked inside full_queue.get() —
        # without a sentinel they hang forever and, being non-daemon, keep
        # the process (and its SLURM allocation) alive long after train()
        # logged Done (job 30285906 idled a 2-GPU node for 2h this way).
        # Harmless on the normal path: a consumed None just breaks a loop
        # that was ending.
        for _ in range(cfg.learner_gpus if ddp_mode
                       else cfg.num_learner_threads):
            try:
                full_queue.put(None)
            except Exception:  # noqa: BLE001
                break
        for t in threads:
            t.join(timeout=10)
        for p in rank_processes:
            p.join(timeout=60)
            if p.is_alive():
                p.terminate()
        # CENTRAL MODE: set should_stop FIRST. Actors that are mid-request
        # (spinning on response_flags) will see should_stop and have
        # request_inference return None — they exit act_central cleanly.
        # Actors between rollouts (waiting on free_queue.get) need the None
        # sentinel below to unblock. Both paths covered.
        if channel is not None:
            channel.should_stop.fill_(1)
        for _ in range(len(actor_processes)):
            try:
                free_queue.put(None)
            except Exception:  # noqa: BLE001
                break
        for p in actor_processes:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
        if inference_thread is not None:
            inference_thread.join(timeout=10)
        if eval_gym_env is not None:
            eval_gym_env.close()
        writer.close()

    # DDP: main's learner_model was never trained — the last rank-0-published
    # weights in weight_state are the real final params.
    final_sd = (dict(zip(weight_state["names"], weight_state["tensors"]))
                if ddp_mode else learner_model.state_dict())
    torch.save(
        {"model_state_dict": final_sd,
         "optimizer_state_dict": {} if ddp_mode else optimizer.state_dict(),
         "config": dataclasses.asdict(cfg)},
        log_dir / "final.pt",
    )
    finish_wandb(wandb_run)
    return {"final_step": step, "stats": stats}
