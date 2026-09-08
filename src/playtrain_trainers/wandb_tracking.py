"""Optional Weights & Biases tracking, mirrored from TensorBoard.

Gated on ``cfg.use_wandb`` (default False) so existing runs are byte-for-byte
unchanged — TensorBoard stays the source of truth. When enabled,
``wandb.init(sync_tensorboard=True)`` transparently mirrors every scalar
already written to the run's ``SummaryWriter`` to W&B — no per-metric
``wandb.log`` calls anywhere in the training loop.

Contract:
  - Call ``init_wandb(cfg, log_dir)`` BEFORE constructing the SummaryWriter so
    the TensorBoard patch is in place for every subsequent write.
  - Call ``finish_wandb(run)`` once at the end (no-op when run is None).

Auth/usage: ``wandb login`` once per machine (or set ``WANDB_API_KEY``). Online
mode works on FASRC Cannon compute nodes (outbound egress is open); only set
``WANDB_MODE=offline`` + ``wandb sync`` later if a particular node lacks egress.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path


def init_wandb(cfg, log_dir):
    """Start a W&B run if ``cfg.use_wandb`` is set, else return None.

    The run name is ``cfg.wandb_name`` when set, else the log-dir basename.
    The SBATCH wrappers (run.sh / run_impala.sh) rewrite ``log_dir`` to a
    per-job dir (``outputs/ppo_<jobid>``) so outputs don't collide, then set
    ``wandb_name`` to the config filename — so the W&B run is titled by the
    config you launched, not the opaque job id. ``cfg.wandb_group`` groups
    related runs (e.g. a seed sweep / the LSTM matrix) in the UI.
    """
    if not getattr(cfg, "use_wandb", False):
        return None
    try:
        import wandb
    except ImportError as e:  # pragma: no cover - guarded dependency
        raise ImportError(
            "use_wandb=True but wandb is not importable. Run `uv sync` "
            "(wandb is a base dependency) or set use_wandb=False."
        ) from e
    cfg_dict = (dataclasses.asdict(cfg) if dataclasses.is_dataclass(cfg)
                else dict(vars(cfg)))
    name = getattr(cfg, "wandb_name", None) or Path(log_dir).name
    run = wandb.init(
        project=getattr(cfg, "wandb_project", "playtrain"),
        group=getattr(cfg, "wandb_group", None),
        name=name,
        config=cfg_dict,
        sync_tensorboard=True,
        dir=str(log_dir),
    )
    # Default every curve's x-axis to `global_step` (env steps) instead of
    # wandb's internal `_step` log-counter. Both trainers write their TB scalars
    # with global_step as the step, and the TB sync logs `global_step` as a
    # metric — so this makes charts read 0 -> total_timesteps out of the box,
    # no per-chart axis fiddling. Best-effort: never let telemetry config fail a run.
    try:
        run.define_metric("global_step")
        run.define_metric("*", step_metric="global_step")
    except Exception:  # noqa: BLE001
        pass
    return run


def finish_wandb(run) -> None:
    """Close the W&B run (no-op when tracking was disabled)."""
    if run is not None:
        import wandb
        wandb.finish()
