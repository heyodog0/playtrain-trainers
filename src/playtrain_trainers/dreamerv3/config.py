"""DreamerV3 configuration — a dataclass tree mirroring the pinned ``configs.yaml``.

Source of every default below, named per block:

  yaml:<path>  -- ``dreamerv3/configs.yaml`` at the pinned commit e3f0224, vendored
                  at ``playtrain-internal/dreamerv3-loop/reference/dreamerv3_official/``.
                  The resolved config is ``defaults`` overridden by the ``atari100k``
                  section; ``defaults`` already carries the 200M sizes, so the
                  ``size200m`` preset is a no-op on it (verified in U01).
  atari.py     -- a constructor default of ``embodied/envs/atari.py`` that the yaml
                  does not mention (``length``, ``pooling``, ``aggregate``).
  D-0NN        -- a PlayTrain mapping choice, recorded in
                  ``playtrain-internal/dreamerv3-loop/DEVIATIONS.md``.
  ours         -- a field the official code does not have, because its job is done
                  there by JAX/portal/elements machinery we do not port.

The yaml-sourced values are frozen (MISSION rule 4): this loop does not tune
DreamerV3. A value that looks wrong for PlayTrain is a finding for the report, not
an edit here. ``validate()`` enforces that, including refusing any model size other
than 200M unless ``allow_size_override`` is set.

JSON loading takes the same nested shape as the dataclass tree. Unknown keys are a
hard error at every level: a misspelled frozen hyperparameter that silently reverted
to its default would be indistinguishable from a faithful run in the results.
"""

from __future__ import annotations

import dataclasses
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np

# ----------------------------------------------------------------------
# Size presets (yaml:size1m..size400m). The official file applies these as regex
# overrides over the whole tree; the three numbers each one touches are the rssm
# widths, the conv depth and the MLP units.
# ----------------------------------------------------------------------
SIZE_PRESETS: dict[str, dict[str, int]] = {
    # name:      deter, hidden, classes, depth, units
    "size1m": {"deter": 512, "hidden": 64, "classes": 4, "depth": 4, "units": 64},
    "size12m": {"deter": 2048, "hidden": 256, "classes": 16, "depth": 16, "units": 256},
    "size25m": {"deter": 3072, "hidden": 384, "classes": 24, "depth": 24, "units": 384},
    "size50m": {"deter": 4096, "hidden": 512, "classes": 32, "depth": 32, "units": 512},
    "size100m": {"deter": 6144, "hidden": 768, "classes": 48, "depth": 48, "units": 768},
    "size200m": {"deter": 8192, "hidden": 1024, "classes": 64, "depth": 64, "units": 1024},
    "size400m": {"deter": 12288, "hidden": 1536, "classes": 96, "depth": 96, "units": 1536},
}


# ----------------------------------------------------------------------
# agent.*
# ----------------------------------------------------------------------
@dataclasses.dataclass
class LossScales:
    """yaml:defaults.agent.loss_scales"""

    rec: float = 1.0
    rew: float = 1.0
    con: float = 1.0
    dyn: float = 1.0
    rep: float = 0.1
    policy: float = 1.0
    value: float = 1.0
    repval: float = 0.3


@dataclasses.dataclass
class OptConfig:
    """yaml:defaults.agent.opt — LaProp chain, see PROTOCOL section 2 and U06."""

    lr: float = 4e-5
    agc: float = 0.3
    eps: float = 1e-20
    beta1: float = 0.9
    beta2: float = 0.999  # paper says 0.99; the code wins (PROTOCOL section 3).
    momentum: bool = True
    wd: float = 0.0
    schedule: str = "const"
    warmup: int = 1000
    anneal: int = 0


@dataclasses.dataclass
class RSSMConfig:
    """yaml:defaults.agent.dyn.rssm (already the 200M widths)."""

    deter: int = 8192
    hidden: int = 1024
    stoch: int = 32
    classes: int = 64
    act: str = "silu"
    norm: str = "rms"
    unimix: float = 0.01
    outscale: float = 1.0
    winit: str = "trunc_normal_in"
    imglayers: int = 2
    obslayers: int = 1
    dynlayers: int = 1
    absolute: bool = False
    blocks: int = 8
    free_nats: float = 1.0


@dataclasses.dataclass
class EncoderConfig:
    """yaml:defaults.agent.enc.simple"""

    depth: int = 64
    mults: tuple[int, ...] = (2, 3, 4, 4)
    layers: int = 3
    units: int = 1024
    act: str = "silu"
    norm: str = "rms"
    winit: str = "trunc_normal_in"
    symlog: bool = True
    outer: bool = False
    kernel: int = 5
    strided: bool = False


@dataclasses.dataclass
class DecoderConfig:
    """yaml:defaults.agent.dec.simple"""

    depth: int = 64
    mults: tuple[int, ...] = (2, 3, 4, 4)
    layers: int = 3
    units: int = 1024
    act: str = "silu"
    norm: str = "rms"
    outscale: float = 1.0
    winit: str = "trunc_normal_in"
    outer: bool = False
    kernel: int = 5
    bspace: int = 8
    strided: bool = False


@dataclasses.dataclass
class HeadConfig:
    """yaml:defaults.agent.rewhead / .conhead / .value.

    ``bins`` is unused by the ``binary`` output (the continue head); it is kept on
    the one dataclass so the three heads stay comparable field by field.
    """

    layers: int = 1
    units: int = 1024
    act: str = "silu"
    norm: str = "rms"
    output: str = "symexp_twohot"
    outscale: float = 0.0
    winit: str = "trunc_normal_in"
    bins: int = 255


@dataclasses.dataclass
class PolicyConfig:
    """yaml:defaults.agent.policy"""

    layers: int = 3
    units: int = 1024
    act: str = "silu"
    norm: str = "rms"
    minstd: float = 0.1
    maxstd: float = 1.0
    outscale: float = 0.01
    unimix: float = 0.01
    winit: str = "trunc_normal_in"


@dataclasses.dataclass
class ImagLossConfig:
    """yaml:defaults.agent.imag_loss"""

    slowtar: bool = False
    lam: float = 0.95
    actent: float = 3e-4
    slowreg: float = 1.0


@dataclasses.dataclass
class ReplLossConfig:
    """yaml:defaults.agent.repl_loss"""

    slowtar: bool = False
    lam: float = 0.95
    slowreg: float = 1.0


@dataclasses.dataclass
class SlowValueConfig:
    """yaml:defaults.agent.slowvalue — EMA regularizer, not the bootstrap."""

    rate: float = 0.02
    every: int = 1


@dataclasses.dataclass
class NormalizerConfig:
    """yaml:defaults.agent.retnorm / .valnorm / .advnorm"""

    impl: str = "perc"
    rate: float = 0.01
    limit: float = 1.0
    perclo: float = 5.0
    perchi: float = 95.0
    debias: bool = False


@dataclasses.dataclass
class AgentConfig:
    """yaml:defaults.agent"""

    loss_scales: LossScales = dataclasses.field(default_factory=LossScales)
    opt: OptConfig = dataclasses.field(default_factory=OptConfig)
    ac_grads: bool = False
    dyn_typ: str = "rssm"  # yaml:defaults.agent.dyn.typ
    rssm: RSSMConfig = dataclasses.field(default_factory=RSSMConfig)
    enc_typ: str = "simple"  # yaml:defaults.agent.enc.typ
    enc: EncoderConfig = dataclasses.field(default_factory=EncoderConfig)
    dec_typ: str = "simple"  # yaml:defaults.agent.dec.typ
    dec: DecoderConfig = dataclasses.field(default_factory=DecoderConfig)
    rewhead: HeadConfig = dataclasses.field(default_factory=HeadConfig)
    conhead: HeadConfig = dataclasses.field(
        default_factory=lambda: HeadConfig(output="binary", outscale=1.0)
    )
    policy: PolicyConfig = dataclasses.field(default_factory=PolicyConfig)
    value: HeadConfig = dataclasses.field(
        default_factory=lambda: HeadConfig(layers=3, output="symexp_twohot", outscale=0.0)
    )
    policy_dist_disc: str = "categorical"
    policy_dist_cont: str = "bounded_normal"
    imag_last: int = 0
    imag_length: int = 15
    horizon: int = 333
    contdisc: bool = True
    imag_loss: ImagLossConfig = dataclasses.field(default_factory=ImagLossConfig)
    repl_loss: ReplLossConfig = dataclasses.field(default_factory=ReplLossConfig)
    slowvalue: SlowValueConfig = dataclasses.field(default_factory=SlowValueConfig)
    retnorm: NormalizerConfig = dataclasses.field(default_factory=NormalizerConfig)
    valnorm: NormalizerConfig = dataclasses.field(
        default_factory=lambda: NormalizerConfig(impl="none", limit=1e-8)
    )
    advnorm: NormalizerConfig = dataclasses.field(
        default_factory=lambda: NormalizerConfig(impl="none", limit=1e-8)
    )
    reward_grad: bool = True
    repval_loss: bool = True
    repval_grad: bool = True
    report: bool = True
    report_gradnorms: bool = False


# ----------------------------------------------------------------------
# env.*, replay.*, run.*
# ----------------------------------------------------------------------
@dataclasses.dataclass
class EnvConfig:
    """yaml:defaults.env.atari100k, plus the three ``atari.py`` constructor defaults
    that section does not override (``length``, ``pooling``, ``aggregate``)."""

    size: tuple[int, int] = (64, 64)
    repeat: int = 4
    sticky: bool = False
    gray: bool = False
    actions: str = "needed"
    lives: str = "unused"
    noops: int = 30
    autostart: bool = False
    resize: str = "pillow"
    clip_reward: bool = False
    length: int = 108000  # atari.py: frames, not agent steps.
    pooling: int = 2  # atari.py: max over the last 2 frames of the repeat.
    aggregate: str = "max"  # atari.py


@dataclasses.dataclass
class ReplayConfig:
    """yaml:defaults.replay. The ``prio``/``recency`` machinery is not ported: the
    frozen config is ``fracs.uniform == 1.0``, so only the uniform selector can ever
    run (D-022). ``fracs_*`` are kept so a config that tries to turn prioritization
    on is refused loudly rather than ignored."""

    size: float = 5e6
    online: bool = True
    fracs_uniform: float = 1.0
    fracs_priority: float = 0.0
    fracs_recency: float = 0.0
    chunksize: int = 1024


@dataclasses.dataclass
class RunConfig:
    """yaml:defaults.run overridden by yaml:atari100k.run.

    The parallel-runner fields (``actor_addr``, ``remote_replay``, ``eval_envs``, …)
    are not ported: our script is ``train`` on one process (D-023).
    """

    steps: int = 110_000  # yaml:atari100k.run.steps 1.1e5, in AGENT steps.
    train_ratio: float = 256.0  # yaml:atari100k.run.train_ratio; replayed steps per agent step.
    envs: int = 1  # yaml:atari100k.run.envs
    log_every: float = 120.0  # yaml:defaults.run, seconds
    report_every: float = 300.0  # seconds
    save_every: float = 900.0  # seconds
    report_batches: int = 1
    from_checkpoint: str = ""


# ----------------------------------------------------------------------
# Top level
# ----------------------------------------------------------------------
@dataclasses.dataclass
class DreamerConfig:
    """The resolved ``defaults`` + ``atari100k`` config, plus our run bookkeeping."""

    # ----- ours: identity and placement -----
    run_id: str = "dreamerv3_playtrain_frostbite"
    output_root: str = "outputs/dreamerv3"
    results_root: str = "results/dreamerv3"

    # ----- ours: which backend this config runs on -----
    # "playtrain" -> PlayTrainEnv on the JS game (D-002); "ale" -> ale-py, the
    # PROTOCOL section 5 A gate that must pass before any PlayTrain arm (MISSION rule 2).
    env_backend: str = "playtrain"
    game: str = "frostbite"

    # ----- yaml:defaults (top level) -----
    seed: int = 0  # yaml:defaults.seed
    script: str = "train"  # yaml:defaults.script
    batch_size: int = 16  # yaml:defaults.batch_size
    batch_length: int = 64  # yaml:defaults.batch_length
    report_length: int = 32  # yaml:defaults.report_length
    consec_train: int = 1  # yaml:defaults.consec_train
    consec_report: int = 1  # yaml:defaults.consec_report
    replay_context: int = 1  # yaml:defaults.replay_context

    # ----- ours: what jax.* does in the official code -----
    # yaml:defaults.jax.compute_dtype is bfloat16; on CUDA that is autocast, on
    # MPS/CPU (local smoke only) it is float32 (D-012).
    device: str = "auto"
    compute_dtype: str = "bfloat16"

    # ----- the frozen model size (yaml:defaults == size200m) -----
    size: str = "size200m"
    allow_size_override: bool = False

    env: EnvConfig = dataclasses.field(default_factory=EnvConfig)
    replay: ReplayConfig = dataclasses.field(default_factory=ReplayConfig)
    run: RunConfig = dataclasses.field(default_factory=RunConfig)
    agent: AgentConfig = dataclasses.field(default_factory=AgentConfig)

    # ------------------------------------------------------------------
    # Derived quantities (all three clocks live here; MISSION rule 3)
    # ------------------------------------------------------------------
    @property
    def batch_steps(self) -> int:
        """Replayed steps per gradient step: ``batch_size * batch_length`` = 1024.

        This is the divisor in ``run/train.py``'s
        ``Ratio(train_ratio / batch_steps)``, and it uses ``batch_length``, NOT the
        sampled window length (which is one longer; see :attr:`sequence_length`).
        """
        return self.batch_size * self.batch_length

    @property
    def train_ratio_per_agent_step(self) -> float:
        """Gradient steps per agent step: 256 / 1024 = 0.25 (D-014)."""
        return self.run.train_ratio / self.batch_steps

    @property
    def sequence_length(self) -> int:
        """Replay window length: ``consec_train * batch_length + replay_context`` = 65.

        From ``main.py`` ``make_replay``/``make_stream``: the extra leading step is
        the replay-context step used to initialize the latent, not a trained step.
        """
        return self.consec_train * self.batch_length + self.replay_context

    @property
    def train_warmup_steps(self) -> int:
        """Agent steps collected before the first gradient step.

        ``run/train.py``'s ``trainfn`` returns early while
        ``len(replay) < batch_size * batch_length``, and it does not call the ratio
        accumulator at all during that window, so there is no catch-up burst.

        ``len(replay)`` is ``len(self.items)``, and an item only appears once
        ``sequence_length`` steps exist from its start, so the item count lags the
        number of ADDED steps by ``sequence_length - 1``. The warmup in agent steps is
        therefore ``batch_steps + sequence_length - 1`` = **1088**, not 1024. Found in
        U07 by the exact accumulator check in ``train.py``.
        """
        return self.batch_steps + self.sequence_length - 1

    @property
    def total_gradient_steps(self) -> int:
        """Gradient steps over the whole run, simulating the official accumulator.

        Matches ``elements.when.Ratio``: the first call returns 1, thereafter the
        fractional remainder is carried exactly. 27,245 at the frozen settings.
        """
        ratio = self.train_ratio_per_agent_step
        if ratio <= 0:
            return 0
        prev: float | None = None
        total = 0
        for step in range(self.train_warmup_steps, self.run.steps + 1):
            if prev is None:
                prev = float(step)
                total += 1
                continue
            repeats = int((step - prev) * ratio)
            prev += repeats / ratio
            total += repeats
        return total

    @property
    def total_frames(self) -> int:
        """Environment frames over the run: agent steps times the action repeat."""
        return self.run.steps * self.env.repeat

    @property
    def obs_shape(self) -> tuple[int, int, int]:
        """Channel-first image shape handed to the encoder."""
        channels = 1 if self.env.gray else 3
        return (channels, self.env.size[0], self.env.size[1])

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validate(self) -> None:
        """Refuse anything the frozen protocol does not allow.

        This is the guard behind MISSION rule 4: the only knobs a config file may
        legitimately turn are identity, backend, seed and device.
        """
        if self.size not in SIZE_PRESETS:
            raise ValueError(f"unknown size preset {self.size!r}; one of {sorted(SIZE_PRESETS)}")
        if self.size != "size200m" and not self.allow_size_override:
            raise ValueError(
                f"size={self.size!r} deviates from the frozen 200M model (PROTOCOL section 1, "
                "D-013). Set allow_size_override=true and record the run as a labelled "
                "extra arm, or leave it at size200m."
            )
        preset = SIZE_PRESETS[self.size]
        mismatched = {
            "agent.rssm.deter": (self.agent.rssm.deter, preset["deter"]),
            "agent.rssm.hidden": (self.agent.rssm.hidden, preset["hidden"]),
            "agent.rssm.classes": (self.agent.rssm.classes, preset["classes"]),
            "agent.enc.depth": (self.agent.enc.depth, preset["depth"]),
            "agent.dec.depth": (self.agent.dec.depth, preset["depth"]),
            "agent.enc.units": (self.agent.enc.units, preset["units"]),
            "agent.dec.units": (self.agent.dec.units, preset["units"]),
            "agent.rewhead.units": (self.agent.rewhead.units, preset["units"]),
            "agent.conhead.units": (self.agent.conhead.units, preset["units"]),
            "agent.policy.units": (self.agent.policy.units, preset["units"]),
            "agent.value.units": (self.agent.value.units, preset["units"]),
        }
        bad = {k: v for k, v in mismatched.items() if v[0] != v[1]}
        if bad and not self.allow_size_override:
            detail = ", ".join(f"{k}={got} (preset {want})" for k, (got, want) in bad.items())
            raise ValueError(
                f"widths disagree with the {self.size} preset: {detail}. The size presets are "
                "regex overrides in configs.yaml; set allow_size_override=true to break them apart."
            )

        if self.env_backend not in ("playtrain", "ale"):
            raise ValueError(f"env_backend must be 'playtrain' or 'ale', got {self.env_backend!r}")
        if self.script != "train":
            raise ValueError(f"only the 'train' script is ported (D-023), got {self.script!r}")
        if self.run.envs != 1:
            raise ValueError(
                f"run.envs must be 1 (yaml:atari100k.run.envs; our driver is single-env, D-023), "
                f"got {self.run.envs}"
            )
        if self.replay.fracs_uniform != 1.0 or self.replay.fracs_priority or self.replay.fracs_recency:
            raise ValueError(
                "replay is uniform-only (yaml:defaults.replay.fracs uniform 1.0); prioritized and "
                "recency selectors are not ported (D-022)"
            )
        if self.batch_size * self.sequence_length > self.replay.size:
            raise ValueError("replay capacity cannot hold one batch (main.py make_replay assert)")
        if self.compute_dtype not in ("bfloat16", "float32"):
            raise ValueError(f"compute_dtype must be bfloat16 or float32, got {self.compute_dtype!r}")
        if self.agent.dyn_typ != "rssm" or self.agent.enc_typ != "simple" or self.agent.dec_typ != "simple":
            raise ValueError("only dyn.typ=rssm, enc.typ=simple, dec.typ=simple are ported")
        if self.env.repeat < 1 or self.env.pooling < 1 or self.env.pooling > self.env.repeat:
            raise ValueError("env.pooling must be in [1, env.repeat] (atari.py assert)")
        if self.env.lives != "unused":
            raise ValueError(f"env.lives is frozen at 'unused' (PROTOCOL section 3), got {self.env.lives!r}")
        if self.run.steps <= self.train_warmup_steps:
            raise ValueError("run.steps must exceed the replay warmup, or no gradient step ever runs")

    def to_dict(self) -> dict[str, Any]:
        """Plain nested dict, JSON-round-trippable, for the run's committed config."""
        return _asdict(self)


# ----------------------------------------------------------------------
# JSON / dict conversion
# ----------------------------------------------------------------------
def _asdict(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj):
        return {f.name: _asdict(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, tuple):
        return list(obj)
    return obj


# The nested dataclass members, by field name, so a JSON file can address them.
_NESTED: dict[str, type] = {
    "env": EnvConfig,
    "replay": ReplayConfig,
    "run": RunConfig,
    "agent": AgentConfig,
}
_AGENT_NESTED: dict[str, type] = {
    "loss_scales": LossScales,
    "opt": OptConfig,
    "rssm": RSSMConfig,
    "enc": EncoderConfig,
    "dec": DecoderConfig,
    "rewhead": HeadConfig,
    "conhead": HeadConfig,
    "policy": PolicyConfig,
    "value": HeadConfig,
    "imag_loss": ImagLossConfig,
    "repl_loss": ReplLossConfig,
    "slowvalue": SlowValueConfig,
    "retnorm": NormalizerConfig,
    "valnorm": NormalizerConfig,
    "advnorm": NormalizerConfig,
}


def _merge(base: Any, raw: dict[str, Any], path: str, nested: dict[str, type]) -> None:
    """Overwrite fields of a constructed dataclass in place, rejecting unknown keys."""
    fields = {f.name for f in dataclasses.fields(base)}
    unknown = sorted(set(raw) - fields)
    if unknown:
        where = f"{path}." if path else ""
        raise ValueError(f"unknown config key(s): {', '.join(where + k for k in unknown)}")
    for name, value in raw.items():
        sub = f"{path}.{name}" if path else name
        if name in nested:
            if not isinstance(value, dict):
                raise TypeError(f"{sub}: expected an object, got {type(value).__name__}")
            child = getattr(base, name)
            grandchildren = _AGENT_NESTED if name == "agent" else {}
            _merge(child, value, sub, grandchildren)
        else:
            current = getattr(base, name)
            if isinstance(current, tuple) and isinstance(value, list):
                value = tuple(value)
            setattr(base, name, value)


def config_from_dict(raw: dict[str, Any]) -> DreamerConfig:
    """Build a validated :class:`DreamerConfig` from a nested dict.

    Missing keys keep the frozen default; unknown keys raise at the level they
    appear, naming the full dotted path.
    """
    if not isinstance(raw, dict):
        raise TypeError(f"expected an object at the top level, got {type(raw).__name__}")
    raw = dict(raw)
    preset = raw.get("preset")
    if preset is not None:
        raw.pop("preset")
        cfg = preset_config(preset)
    else:
        cfg = DreamerConfig()
    _merge(cfg, raw, "", _NESTED)
    cfg.validate()
    return cfg


def load_config(path: str | os.PathLike[str]) -> DreamerConfig:
    """Load a validated :class:`DreamerConfig` from a JSON file."""
    return config_from_dict(json.loads(Path(path).read_text()))


# ----------------------------------------------------------------------
# Presets
# ----------------------------------------------------------------------
def atari100k_config() -> DreamerConfig:
    """``defaults`` + ``atari100k``: the frozen 200M spec, PROTOCOL sections 1-3."""
    return DreamerConfig()


def debug_config() -> DreamerConfig:
    """yaml:debug — the official debug section, for the local smoke only.

    Mirrors it field for field with two recorded exceptions (D-023):
    ``run.envs`` stays 1 because our driver is single-env, and ``jax.platform: cpu``
    becomes ``compute_dtype: float32`` plus device autodetect, because we have no JAX.
    Everything else (batch 8x10, report_length 5, train_ratio 8, replay 1e4, and the
    ``agent`` regex overrides) is copied.
    """
    cfg = DreamerConfig(
        run_id="dreamerv3_debug",
        batch_size=8,
        batch_length=10,
        report_length=5,
        compute_dtype="float32",
        size="size1m",
        allow_size_override=True,
    )
    cfg.run.envs = 1  # yaml:debug.run.envs is 4; D-023, our driver is single-env.
    cfg.run.report_every = 10.0
    cfg.run.log_every = 5.0
    cfg.run.save_every = 15.0
    cfg.run.train_ratio = 8.0
    cfg.run.steps = 3_000  # ours: the U07 local smoke budget, not from the yaml.
    cfg.replay.size = 1e4

    # yaml:debug.agent regex overrides, applied to every field the regex would hit.
    for head in (cfg.agent.rewhead, cfg.agent.conhead, cfg.agent.value):
        head.bins = 5
        head.layers = 1
        head.units = 8
    cfg.agent.policy.layers = 1
    cfg.agent.policy.units = 8
    cfg.agent.enc.layers = 1
    cfg.agent.enc.units = 8
    cfg.agent.enc.depth = 2
    cfg.agent.dec.layers = 1
    cfg.agent.dec.units = 8
    cfg.agent.dec.depth = 2
    cfg.agent.rssm.stoch = 2
    cfg.agent.rssm.classes = 4
    cfg.agent.rssm.deter = 8
    cfg.agent.rssm.hidden = 3
    cfg.agent.rssm.blocks = 4
    return cfg


PRESETS = {
    "atari100k": atari100k_config,
    "debug": debug_config,
}


def preset_config(name: str) -> DreamerConfig:
    if name not in PRESETS:
        raise ValueError(f"unknown preset {name!r}; one of {sorted(PRESETS)}")
    return PRESETS[name]()


# ----------------------------------------------------------------------
# Run-directory convention (same shape as bbf/config.py)
# ----------------------------------------------------------------------
def run_dir(cfg: DreamerConfig, seed: int | None = None, root: str | os.PathLike[str] | None = None) -> Path:
    """Scratch directory for a run: ``outputs/dreamerv3/<run_id>/seed<N>/``."""
    base = Path(root) if root is not None else Path(cfg.output_root)
    s = cfg.seed if seed is None else seed
    return base / cfg.run_id / f"seed{s}"


def results_dir(cfg: DreamerConfig, root: str | os.PathLike[str] | None = None) -> Path:
    """Committed directory for a run: ``results/dreamerv3/<run_id>/``."""
    base = Path(root) if root is not None else Path(cfg.results_root)
    return base / cfg.run_id


# ----------------------------------------------------------------------
# Seeding (same derivation as bbf/config.py, so the two ports share seed streams)
# ----------------------------------------------------------------------
def seed_for(base_seed: int, index: int) -> int:
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
