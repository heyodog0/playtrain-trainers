"""BBF configuration — a dataclass mirror of the frozen protocol.

Every default in :class:`BBFConfig` is transcribed from one of two sources and
the source is named in the comment above it:

  gin   -- ``bigger_better_faster/bbf/configs/BBF.gin`` (Schwarzer et al. 2023,
           ICML; google-research/bigger_better_faster). Copy in
           ``playtrain-internal/bbf-loop/reference/BBF.gin``.
  D-0NN -- a PlayTrain mapping choice, recorded in
           ``playtrain-internal/bbf-loop/DEVIATIONS.md``.

The gin-sourced values are frozen: this loop does not tune BBF. A value that
looks wrong for PlayTrain is a finding for the report, not an edit here.

JSON loading follows ``playtrain_trainers.train_impala`` (a flat dict of field
names), with one deliberate difference: unknown keys are a hard error rather
than a warning. A misspelled frozen hyperparameter that silently reverts to its
default would be indistinguishable from a faithful run in the results.
"""
from __future__ import annotations

import dataclasses
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np


@dataclasses.dataclass
class BBFConfig:
    # ------------------------------------------------------------------
    # Identity / bookkeeping
    # ------------------------------------------------------------------
    # Names the run directory under outputs/bbf/ and results/bbf/.
    run_id: str = "bbf_playtrain_frostbite_rr2"
    seed: int = 0

    # ------------------------------------------------------------------
    # Environment (PROTOCOL section 2, mapped by section 4)
    # ------------------------------------------------------------------
    # "playtrain" -> PlayTrainEnv on the JS game (D-013);
    # "ale"       -> gymnasium/ale-py, the section 5A sanity arm.
    env_backend: str = "playtrain"
    # PlayTrain game id, or the ALE env id when env_backend == "ale".
    game: str = "frostbite"
    # gin: AtariPreprocessing default; D-001 keeps 4 game frames per decision.
    frame_skip: int = 4
    # gin: 84x84 grayscale, 4-frame stack. D-002 asks the runtime for 84 px
    # directly rather than resizing its native 64.
    obs_size: int = 84
    obs_mode: str = "grayscale"
    frame_stack: int = 4
    # Max-pool over the last 2 frames exists for ALE sprite flicker; the p5
    # game does not flicker (D-002). True only on the ALE arm.
    max_pool_frames: bool = False
    # gin: atari_lib.create_atari_environment.sticky_actions = False
    sticky_actions: bool = False
    # gin: DataEfficientAtariRunner.max_noops = 30 (D-005)
    max_noops: int = 30
    # gin: AtariPreprocessing.terminal_on_life_loss = True (D-004)
    terminal_on_life_loss: bool = True
    # gin: Runner.max_steps_per_episode = 27000 AGENT steps. The PlayTrain
    # runtime's max_steps counts FRAMES, so envs.py passes
    # max_steps_per_episode * frame_skip (D-012).
    max_steps_per_episode: int = 27_000
    # D-003: PlayTrain's 8-action set, against ALE Frostbite's 18.
    action_space: str = "default8"
    # Rewards clipped to [-clip, clip] for learning; raw score kept in info
    # (D-011).
    reward_clip: float = 1.0
    # gin: DataEfficientAtariRunner.num_train_envs = 1
    num_train_envs: int = 1

    # ------------------------------------------------------------------
    # Budget and evaluation (PROTOCOL section 2)
    # ------------------------------------------------------------------
    # gin: Runner.training_steps = 100000 agent steps.
    training_steps: int = 100_000
    # gin: num_eval_episodes = 100, num_eval_envs = 100, at the END only.
    num_eval_episodes: int = 100
    num_eval_envs: int = 100
    # D-006: a cheap curve on top of the specified final eval. 0 disables.
    eval_every: int = 10_000
    eval_episodes_curve: int = 10
    # D-017: the eval episodes run on a FIXED game-seed pool so that every arm
    # of the comparison (random, PPO, IMPALA, BBF) faces the same floe
    # layouts. Dopamine just runs 100 episodes; pinning the seeds removes
    # layout luck from the comparison instead of averaging it per arm.
    # Chosen clear of the pools already in use in this repo: the human study's
    # 90_000+i and the trainers' 9_000_000+i.
    eval_seed_base: int = 8_000_000

    # ------------------------------------------------------------------
    # Agent (PROTOCOL section 1 -- all gin, all frozen)
    # ------------------------------------------------------------------
    # gin: encoder_type = "impala", ImpalaCNN.num_blocks = 2, width_scale = 4
    encoder_type: str = "impala"
    num_blocks: int = 2
    width_scale: int = 4
    renormalize: bool = True
    hidden_dim: int = 2048
    # gin: dueling True, double_dqn True, distributional True, noisy False
    dueling: bool = True
    double_dqn: bool = True
    distributional: bool = True
    num_atoms: int = 51
    # NOT in the gin: the C51 support range falls back to the Dopamine default
    # vmax = 10, vmin = -vmax. Recorded as D-015 because it is an
    # implementation choice we cannot check against the reference we hold.
    v_max: float = 10.0
    noisy: bool = False

    # gin: spr_weight = 5, jumps = 5
    spr_weight: float = 5.0
    jumps: int = 5
    # gin: data_augmentation = True (DrQ/SPR random shift + intensity)
    data_augmentation: bool = True
    aug_shift_pad: int = 4
    aug_intensity_scale: float = 0.05

    # gin: adam (AdamW with decoupled wd in create_scaling_optimizer),
    # learning_rate = encoder_learning_rate = 1e-4, eps = 1.5e-4, wd = 0.1
    learning_rate: float = 1e-4
    encoder_learning_rate: float = 1e-4
    adam_eps: float = 1.5e-4
    weight_decay: float = 0.1
    # gin: half_precision = False
    half_precision: bool = False

    # gin: target_update_tau = 0.005, target_update_period = 1,
    # target_action_selection = True, update_period = 1
    target_update_tau: float = 0.005
    target_update_period: int = 1
    target_action_selection: bool = True
    update_period: int = 1

    # gin: batch_size = 32, batches_to_group = 2. In the official agent
    # `batches_to_group` only sets how many batches one jitted `train` call
    # scans over; EVERY batch is its own Adam step and its own EMA target
    # update. It has no effect on the training semantics and none here
    # (D-039). Kept so the gin transcribes 1:1.
    batch_size: int = 32
    batches_to_group: int = 2
    # gin: replay_ratio = 64. Gradient steps per env step is
    # replay_ratio * num_train_envs / batch_size = 2 ("RR=2"). The paper's
    # Table A.1 headline is RR=8, i.e. replay_ratio = 256 (flag F-001).
    replay_ratio: int = 64

    # gin: max_update_horizon 10 -> update_horizon 3 and min_gamma 0.97 ->
    # gamma 0.997, both annealed exponentially over cycle_steps = 10000
    # GRADIENT steps following each reset (`cycle_grad_steps`).
    update_horizon: int = 3
    max_update_horizon: int = 10
    gamma: float = 0.997
    min_gamma: float = 0.97
    cycle_steps: int = 10_000

    # gin: reset_every = 20000, no_resets_after = 100000 -- both in ENV
    # (agent) steps: `spr_agent._train_step` compares them against
    # `training_steps`, which ticks once per env step (D-033). At RR=2 the
    # 20k-env-step interval is the paper's "every 40k gradient steps"; the
    # paper's RR=8 value is 5000. shrink_perturb_keys =
    # "encoder,transition_model", shrink 0.5 / perturb 0.5.
    reset_every: int = 20_000
    shrink_perturb_keys: tuple[str, ...] = ("encoder", "transition_model")
    shrink_factor: float = 0.5
    perturb_factor: float = 0.5
    no_resets_after: int = 100_000

    # gin: epsilon_train = 0.0, epsilon_eval = 0.001, and
    # linearly_decaying_epsilon over epsilon_decay_period = 2001 steps after
    # min_replay_history = 2000 env steps of random actions.
    epsilon_train: float = 0.0
    epsilon_eval: float = 0.001
    epsilon_decay_period: int = 2001
    min_replay_history: int = 2_000

    # gin: subsequence replay buffer, prioritized, capacity 200000, n_envs 1
    replay_capacity: int = 200_000
    replay_scheme: str = "prioritized"
    # Dopamine prioritized-replay exponent; not in the gin, part of D-015.
    priority_exponent: float = 0.5
    # D-042: reproduce the official buffer's off-by-one bootstrap.
    # `subsequence_replay_buffer.sample_transition_batch` sums n rewards at
    # offsets 0..n-1 with gamma^0..gamma^(n-1) -- a correct n-step return --
    # but then sets `next_indices = state_indices + (update_horizon - 1)` and
    # `discount = cumulative_discount_vector[update_horizon]`, i.e. it
    # bootstraps from s_{t+n-1} while applying gamma^n. That counts
    # r_{t+n-1} twice: once in the sum and once inside V(s_{t+n-1}).
    # We use the textbook s_{t+n}. Set True only to TEST whether their
    # off-by-one explains the score gap; it is not the protocol.
    official_offby1_bootstrap: bool = False

    # ------------------------------------------------------------------
    # Runtime / io
    # ------------------------------------------------------------------
    # Run trees, relative to the repo root. outputs/ is scratch and is not
    # committed; results/<run_id>/ holds config + metrics.json and is.
    output_root: str = "outputs/bbf"
    results_root: str = "results/bbf"
    device: str = "auto"  # "auto" | "cuda" | "mps" | "cpu"
    log_every: int = 100  # gin: BBFAgent.log_every = 100
    checkpoint_every: int = 10_000  # env steps; 0 disables

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------
    @property
    def gradient_steps_per_env_step(self) -> int:
        """BBF's ``num_updates_per_train_step``: replay_ratio * n_envs // batch.

        At the gin defaults this is 2, which is what the loop calls "RR=2".
        """
        return self.replay_ratio * self.num_train_envs // self.batch_size

    @property
    def reset_env_steps(self) -> list[int]:
        """Env steps at which the official reset rule fires over this run."""
        from playtrain_trainers.bbf.schedules import reset_env_steps

        return reset_env_steps(self)

    @property
    def total_gradient_steps(self) -> int:
        """Gradient steps over the whole run, for the reset schedule."""
        return self.training_steps * self.gradient_steps_per_env_step

    @property
    def max_env_frames(self) -> int:
        """Game frames the budget corresponds to (400k at the defaults)."""
        return self.training_steps * self.frame_skip

    @property
    def obs_channels(self) -> int:
        return self.frame_stack * (1 if self.obs_mode == "grayscale" else 3)

    @property
    def obs_shape(self) -> tuple[int, int, int]:
        """Network input, CHW."""
        return (self.obs_channels, self.obs_size, self.obs_size)

    @property
    def v_min(self) -> float:
        return -self.v_max

    @property
    def max_steps_frames(self) -> int:
        """Episode cap in game FRAMES, which is the unit the runtime uses."""
        return self.max_steps_per_episode * self.frame_skip

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validate(self) -> None:
        """Reject configs the protocol cannot mean. Raises ValueError."""
        if self.env_backend not in ("playtrain", "ale"):
            raise ValueError(f"env_backend must be playtrain|ale, got {self.env_backend!r}")
        if self.obs_mode not in ("grayscale", "rgb"):
            raise ValueError(f"obs_mode must be grayscale|rgb, got {self.obs_mode!r}")
        if self.replay_scheme not in ("prioritized", "uniform"):
            raise ValueError(f"replay_scheme must be prioritized|uniform, got {self.replay_scheme!r}")
        if self.replay_ratio * self.num_train_envs % self.batch_size:
            raise ValueError(
                "replay_ratio * num_train_envs must be a multiple of batch_size; "
                f"got {self.replay_ratio} * {self.num_train_envs} % {self.batch_size}"
            )
        if self.gradient_steps_per_env_step < 1:
            raise ValueError(
                "replay_ratio is too small for batch_size: "
                f"gradient_steps_per_env_step = {self.gradient_steps_per_env_step}"
            )
        if self.batches_to_group < 1:
            raise ValueError("batches_to_group must be >= 1")
        if self.gradient_steps_per_env_step % self.batches_to_group:
            # The official agent asserts this too (set_replay_settings).
            raise ValueError(
                "batches_to_group must divide gradient_steps_per_env_step; got "
                f"{self.batches_to_group} vs {self.gradient_steps_per_env_step}"
            )
        # The n-step / gamma anneal runs on gradient steps; a zero cycle would
        # divide by zero in the schedule.
        if self.cycle_steps < 1:
            raise ValueError("cycle_steps must be >= 1")
        if self.max_update_horizon < self.update_horizon:
            raise ValueError(
                f"max_update_horizon ({self.max_update_horizon}) must be >= "
                f"update_horizon ({self.update_horizon})"
            )
        if not 0.0 < self.min_gamma <= self.gamma < 1.0:
            raise ValueError(
                f"need 0 < min_gamma ({self.min_gamma}) <= gamma ({self.gamma}) < 1"
            )
        if self.jumps < 0:
            raise ValueError("jumps must be >= 0")
        if self.num_atoms < 2:
            raise ValueError("num_atoms must be >= 2")
        if self.v_max <= 0:
            raise ValueError("v_max must be > 0")
        if not 0.0 <= self.shrink_factor <= 1.0:
            raise ValueError("shrink_factor must be in [0, 1]")
        if not 0.0 <= self.perturb_factor <= 1.0:
            raise ValueError("perturb_factor must be in [0, 1]")
        if self.frame_skip < 1 or self.frame_stack < 1:
            raise ValueError("frame_skip and frame_stack must be >= 1")
        if self.max_noops < 0:
            raise ValueError("max_noops must be >= 0")
        if self.num_eval_episodes < 1 or self.eval_episodes_curve < 1:
            raise ValueError("eval episode counts must be >= 1")
        if self.eval_seed_base < 0:
            raise ValueError("eval_seed_base must be >= 0")
        if self.replay_capacity <= self.min_replay_history:
            raise ValueError("replay_capacity must exceed min_replay_history")
        # A subsequence buffer must be able to hold a jumps+1 window.
        if self.replay_capacity < self.jumps + 1:
            raise ValueError("replay_capacity is smaller than one SPR window")

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Field values plus the derived quantities, for config.json."""
        d = dataclasses.asdict(self)
        d["shrink_perturb_keys"] = list(self.shrink_perturb_keys)
        d["_derived"] = {
            "gradient_steps_per_env_step": self.gradient_steps_per_env_step,
            "reset_env_steps": self.reset_env_steps,
            "total_gradient_steps": self.total_gradient_steps,
            "max_env_frames": self.max_env_frames,
            "max_steps_frames": self.max_steps_frames,
            "obs_shape": list(self.obs_shape),
            "v_min": self.v_min,
            "v_max": self.v_max,
        }
        return d


def config_from_dict(raw: dict[str, Any]) -> BBFConfig:
    """Build a validated BBFConfig, rejecting unknown keys.

    Unlike ``train_impala``'s loader, an unknown key raises. A typo in a frozen
    hyperparameter would otherwise leave no trace in the saved run.
    """
    valid = {f.name for f in dataclasses.fields(BBFConfig)}
    # _derived is emitted by to_dict for readers; accept and drop it so a saved
    # config.json round-trips.
    raw = {k: v for k, v in raw.items() if k != "_derived"}
    unknown = sorted(set(raw) - valid)
    if unknown:
        raise ValueError(
            f"unknown BBF config keys: {unknown}. Known keys: {sorted(valid)}"
        )
    kwargs = dict(raw)
    if "shrink_perturb_keys" in kwargs:
        kwargs["shrink_perturb_keys"] = tuple(kwargs["shrink_perturb_keys"])
    cfg = BBFConfig(**kwargs)
    cfg.validate()
    return cfg


def load_config(path: str | os.PathLike[str]) -> BBFConfig:
    """Load a validated BBFConfig from a JSON file."""
    return config_from_dict(json.loads(Path(path).read_text()))


# ----------------------------------------------------------------------
# Run-directory convention
# ----------------------------------------------------------------------
def run_dir(cfg: BBFConfig, seed: int | None = None, root: str | os.PathLike[str] | None = None) -> Path:
    """Scratch directory for a run: ``outputs/bbf/<run_id>/seed<N>/``.

    Not committed: checkpoints, TensorBoard, frames.
    """
    base = Path(root) if root is not None else Path(cfg.output_root)
    s = cfg.seed if seed is None else seed
    return base / cfg.run_id / f"seed{s}"


def results_dir(cfg: BBFConfig, root: str | os.PathLike[str] | None = None) -> Path:
    """Committed directory for a run: ``results/bbf/<run_id>/``.

    Holds config.json, metrics.json, the plot and the provenance stamp. One
    directory per run_id, with per-seed results inside metrics.json, so the
    committed tree has one entry per reported number.
    """
    base = Path(root) if root is not None else Path(cfg.results_root)
    return base / cfg.run_id


# ----------------------------------------------------------------------
# Seeding
# ----------------------------------------------------------------------
def seed_for(base_seed: int, index: int) -> int:
    """Derive the ``index``-th run seed from a base seed.

    Spreading the seeds apart keeps the per-seed RNG streams from sharing a
    prefix, which consecutive small integers can do in some generators.
    """
    if index < 0:
        raise ValueError("index must be >= 0")
    return int((base_seed * 1_000_003 + index * 7_919 + 1) % (2**31 - 1))


def set_global_seeds(seed: int) -> None:
    """Seed python, numpy and torch (including cuda/mps) for one process."""
    import torch

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
