"""U01 — the resolved 200M config, checked field by field against `configs.yaml`.

The expected values in :data:`EXPECTED` were transcribed BY HAND from the pinned
``dreamerv3/configs.yaml`` (``defaults`` overridden by the ``atari100k`` section), not
read out of the dataclass. That is the point of the test: it compares our port against
the official file, not against itself. Line numbers are from the pinned file.

Vendored copy:
``playtrain-internal/dreamerv3-loop/reference/dreamerv3_official/dreamerv3/configs.yaml``
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from playtrain_trainers.dreamerv3 import config as C

# ----------------------------------------------------------------------
# Hand-transcribed from configs.yaml. Keys are dotted paths into our tree.
# ----------------------------------------------------------------------
EXPECTED: dict[str, object] = {
    # defaults, top level (lines 8-15)
    "seed": 0,
    "script": "train",
    "batch_size": 16,
    "batch_length": 64,
    "report_length": 32,
    "consec_train": 1,
    "consec_report": 1,
    "replay_context": 1,
    # atari100k overrides (lines 170-172)
    "run.steps": 110_000,  # 1.1e5
    "run.envs": 1,
    "run.train_ratio": 256.0,
    # defaults.run (lines 52-58)
    "run.log_every": 120.0,
    "run.report_every": 300.0,
    "run.save_every": 900.0,
    "run.report_batches": 1,
    "run.from_checkpoint": "",
    # defaults.env.atari100k (line 33) + atari.py constructor defaults
    "env.size": (64, 64),
    "env.repeat": 4,
    "env.sticky": False,
    "env.gray": False,
    "env.actions": "needed",
    "env.lives": "unused",
    "env.noops": 30,
    "env.autostart": False,
    "env.resize": "pillow",
    "env.clip_reward": False,
    "env.length": 108000,
    "env.pooling": 2,
    "env.aggregate": "max",
    # defaults.replay (lines 40-46)
    "replay.size": 5e6,
    "replay.online": True,
    "replay.fracs_uniform": 1.0,
    "replay.fracs_priority": 0.0,
    "replay.fracs_recency": 0.0,
    "replay.chunksize": 1024,
    # defaults.agent.loss_scales (line 86)
    "agent.loss_scales.rec": 1.0,
    "agent.loss_scales.rew": 1.0,
    "agent.loss_scales.con": 1.0,
    "agent.loss_scales.dyn": 1.0,
    "agent.loss_scales.rep": 0.1,
    "agent.loss_scales.policy": 1.0,
    "agent.loss_scales.value": 1.0,
    "agent.loss_scales.repval": 0.3,
    # defaults.agent.opt (line 87)
    "agent.opt.lr": 4e-5,
    "agent.opt.agc": 0.3,
    "agent.opt.eps": 1e-20,
    "agent.opt.beta1": 0.9,
    "agent.opt.beta2": 0.999,
    "agent.opt.momentum": True,
    "agent.opt.wd": 0.0,
    "agent.opt.schedule": "const",
    "agent.opt.warmup": 1000,
    "agent.opt.anneal": 0,
    # defaults.agent.ac_grads / dyn (lines 88-91)
    "agent.ac_grads": False,
    "agent.dyn_typ": "rssm",
    "agent.rssm.deter": 8192,
    "agent.rssm.hidden": 1024,
    "agent.rssm.stoch": 32,
    "agent.rssm.classes": 64,
    "agent.rssm.act": "silu",
    "agent.rssm.norm": "rms",
    "agent.rssm.unimix": 0.01,
    "agent.rssm.outscale": 1.0,
    "agent.rssm.winit": "trunc_normal_in",
    "agent.rssm.imglayers": 2,
    "agent.rssm.obslayers": 1,
    "agent.rssm.dynlayers": 1,
    "agent.rssm.absolute": False,
    "agent.rssm.blocks": 8,
    "agent.rssm.free_nats": 1.0,
    # defaults.agent.enc.simple (lines 92-94)
    "agent.enc_typ": "simple",
    "agent.enc.depth": 64,
    "agent.enc.mults": (2, 3, 4, 4),
    "agent.enc.layers": 3,
    "agent.enc.units": 1024,
    "agent.enc.act": "silu",
    "agent.enc.norm": "rms",
    "agent.enc.winit": "trunc_normal_in",
    "agent.enc.symlog": True,
    "agent.enc.outer": False,
    "agent.enc.kernel": 5,
    "agent.enc.strided": False,
    # defaults.agent.dec.simple (lines 95-97)
    "agent.dec_typ": "simple",
    "agent.dec.depth": 64,
    "agent.dec.mults": (2, 3, 4, 4),
    "agent.dec.layers": 3,
    "agent.dec.units": 1024,
    "agent.dec.act": "silu",
    "agent.dec.norm": "rms",
    "agent.dec.outscale": 1.0,
    "agent.dec.winit": "trunc_normal_in",
    "agent.dec.outer": False,
    "agent.dec.kernel": 5,
    "agent.dec.bspace": 8,
    "agent.dec.strided": False,
    # defaults.agent.rewhead (line 98)
    "agent.rewhead.layers": 1,
    "agent.rewhead.units": 1024,
    "agent.rewhead.act": "silu",
    "agent.rewhead.norm": "rms",
    "agent.rewhead.output": "symexp_twohot",
    "agent.rewhead.outscale": 0.0,
    "agent.rewhead.winit": "trunc_normal_in",
    "agent.rewhead.bins": 255,
    # defaults.agent.conhead (line 99) — note outscale 1.0, no bins in the yaml
    "agent.conhead.layers": 1,
    "agent.conhead.units": 1024,
    "agent.conhead.act": "silu",
    "agent.conhead.norm": "rms",
    "agent.conhead.output": "binary",
    "agent.conhead.outscale": 1.0,
    "agent.conhead.winit": "trunc_normal_in",
    # defaults.agent.policy (line 100)
    "agent.policy.layers": 3,
    "agent.policy.units": 1024,
    "agent.policy.act": "silu",
    "agent.policy.norm": "rms",
    "agent.policy.minstd": 0.1,
    "agent.policy.maxstd": 1.0,
    "agent.policy.outscale": 0.01,
    # Present in configs.yaml, but DEAD in the pinned code path: `Head.categorical`
    # never reads it (D-027). Asserted here because the config must still mirror the
    # yaml; what the head does with it is asserted in test_dreamerv3_nets.py.
    "agent.policy.unimix": 0.01,
    "agent.policy.winit": "trunc_normal_in",
    # defaults.agent.value (line 101)
    "agent.value.layers": 3,
    "agent.value.units": 1024,
    "agent.value.act": "silu",
    "agent.value.norm": "rms",
    "agent.value.output": "symexp_twohot",
    "agent.value.outscale": 0.0,
    "agent.value.winit": "trunc_normal_in",
    "agent.value.bins": 255,
    # defaults.agent, rest (lines 102-118)
    "agent.policy_dist_disc": "categorical",
    "agent.policy_dist_cont": "bounded_normal",
    "agent.imag_last": 0,
    "agent.imag_length": 15,
    "agent.horizon": 333,
    "agent.contdisc": True,
    "agent.imag_loss.slowtar": False,
    "agent.imag_loss.lam": 0.95,
    "agent.imag_loss.actent": 3e-4,
    "agent.imag_loss.slowreg": 1.0,
    "agent.repl_loss.slowtar": False,
    "agent.repl_loss.lam": 0.95,
    "agent.repl_loss.slowreg": 1.0,
    "agent.slowvalue.rate": 0.02,
    "agent.slowvalue.every": 1,
    "agent.retnorm.impl": "perc",
    "agent.retnorm.rate": 0.01,
    "agent.retnorm.limit": 1.0,
    "agent.retnorm.perclo": 5.0,
    "agent.retnorm.perchi": 95.0,
    "agent.retnorm.debias": False,
    "agent.valnorm.impl": "none",
    "agent.valnorm.rate": 0.01,
    "agent.valnorm.limit": 1e-8,
    "agent.advnorm.impl": "none",
    "agent.advnorm.rate": 0.01,
    "agent.advnorm.limit": 1e-8,
    "agent.reward_grad": True,
    "agent.repval_loss": True,
    "agent.repval_grad": True,
    "agent.report": True,
    "agent.report_gradnorms": False,
    # defaults.jax.compute_dtype (line 74)
    "compute_dtype": "bfloat16",
}


def _get(obj: object, path: str) -> object:
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


@pytest.mark.parametrize("path", sorted(EXPECTED))
def test_resolved_atari100k_matches_configs_yaml(path: str) -> None:
    cfg = C.atari100k_config()
    assert _get(cfg, path) == EXPECTED[path], path


def test_every_expected_path_is_covered() -> None:
    """Guard against a typo silently dropping a check."""
    cfg = C.atari100k_config()
    for path in EXPECTED:
        _get(cfg, path)  # raises AttributeError if the path is wrong
    assert len(EXPECTED) >= 150


# ----------------------------------------------------------------------
# Derived clocks (MISSION rule 3, D-014)
# ----------------------------------------------------------------------
def test_derived_clocks() -> None:
    cfg = C.atari100k_config()
    assert cfg.batch_steps == 1024  # 16 * 64
    assert cfg.train_ratio_per_agent_step == 0.25  # 256 / 1024
    # main.py make_replay/make_stream: consec_train * batch_length + replay_context
    assert cfg.sequence_length == 65
    # len(replay) counts ITEMS, which lag adds by sequence_length - 1 (U07).
    assert cfg.train_warmup_steps == 1024 + 65 - 1 == 1088
    assert cfg.total_frames == 440_000  # 110k agent steps * repeat 4
    assert cfg.obs_shape == (3, 64, 64)


def test_total_gradient_steps_matches_the_elements_ratio_accumulator() -> None:
    """Independent re-implementation of `elements.when.Ratio`, applied the way
    `embodied/run/train.py` applies it, must give the same total as the property.

    Written from the fetched source in
    `playtrain-internal/dreamerv3-loop/reference/elements_when_3.19.1.py`.
    """
    cfg = C.atari100k_config()

    ratio = cfg.run.train_ratio / (cfg.batch_size * cfg.batch_length)
    prev = None
    total = 0
    first = None
    # `trainfn` returns before consulting the accumulator while len(replay) < 1024
    # items, and an item needs `sequence_length` steps from its start, so the first
    # training call lands at 1024 + 65 - 1 agent steps.
    warmup = cfg.batch_size * cfg.batch_length + cfg.sequence_length - 1
    for step in range(1, cfg.run.steps + 1):
        if step < warmup:
            continue
        if prev is None:
            prev = float(step)
            repeats = 1
        else:
            repeats = int((step - prev) * ratio)
            prev += repeats / ratio
        if repeats and first is None:
            first = step
        total += repeats

    assert first == 1088
    assert total == 27_229
    assert cfg.total_gradient_steps == total
    # Rule 3's tolerance has to be read on the post-warmup clock; the full-run
    # figure is legitimately 0.9% low because the warmup is not credited.
    post_warmup = total / (cfg.run.steps - cfg.train_warmup_steps + 1)
    assert abs(post_warmup - 0.25) / 0.25 < 0.05
    full_run = total / cfg.run.steps
    assert 0.2475 < full_run < 0.2476


# ----------------------------------------------------------------------
# Size override guard (MISSION rule 4)
# ----------------------------------------------------------------------
def test_defaults_are_the_size200m_preset() -> None:
    """`defaults` in configs.yaml already carries the 200M widths, so applying the
    `size200m` regex block is a no-op. If that stops being true the port is reading
    the wrong section."""
    preset = C.SIZE_PRESETS["size200m"]
    cfg = C.atari100k_config()
    assert cfg.agent.rssm.deter == preset["deter"] == 8192
    assert cfg.agent.rssm.hidden == preset["hidden"] == 1024
    assert cfg.agent.rssm.classes == preset["classes"] == 64
    assert cfg.agent.enc.depth == cfg.agent.dec.depth == preset["depth"] == 64
    assert cfg.agent.policy.units == cfg.agent.value.units == preset["units"] == 1024


def test_size_override_is_refused_by_default() -> None:
    with pytest.raises(ValueError, match="allow_size_override"):
        C.config_from_dict({"size": "size100m"})


def test_size_override_allowed_when_flagged() -> None:
    cfg = C.config_from_dict(
        {
            "size": "size100m",
            "allow_size_override": True,
            "agent": {
                "rssm": {"deter": 6144, "hidden": 768, "classes": 48},
                "enc": {"depth": 48, "units": 768},
                "dec": {"depth": 48, "units": 768},
                "rewhead": {"units": 768},
                "conhead": {"units": 768},
                "policy": {"units": 768},
                "value": {"units": 768},
            },
        }
    )
    assert cfg.agent.rssm.deter == 6144


def test_widths_out_of_step_with_the_preset_are_refused() -> None:
    with pytest.raises(ValueError, match="widths disagree"):
        C.config_from_dict({"agent": {"rssm": {"deter": 4096}}})


def test_unknown_size_preset() -> None:
    with pytest.raises(ValueError, match="unknown size preset"):
        C.config_from_dict({"size": "size1000m", "allow_size_override": True})


# ----------------------------------------------------------------------
# Other frozen-value guards
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "raw, match",
    [
        ({"env_backend": "gym"}, "env_backend"),
        ({"script": "parallel"}, "only the 'train' script"),
        ({"run": {"envs": 4}}, "run.envs must be 1"),
        ({"replay": {"fracs_priority": 0.5, "fracs_uniform": 0.5}}, "uniform-only"),
        ({"compute_dtype": "float16"}, "compute_dtype"),
        ({"env": {"lives": "reset"}}, "env.lives is frozen"),
        ({"agent": {"dyn_typ": "gru"}}, "are ported"),
        ({"env": {"pooling": 5}}, "env.pooling"),
        ({"run": {"steps": 500}}, "must exceed the replay warmup"),
    ],
)
def test_validate_refuses(raw: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        C.config_from_dict(raw)


# ----------------------------------------------------------------------
# JSON loading
# ----------------------------------------------------------------------
def test_unknown_key_is_an_error_with_a_dotted_path() -> None:
    with pytest.raises(ValueError, match=r"agent\.rssm\.detr"):
        C.config_from_dict({"agent": {"rssm": {"detr": 8192}}})
    with pytest.raises(ValueError, match=r"unknown config key\(s\): batchsize"):
        C.config_from_dict({"batchsize": 16})


def test_json_round_trip() -> None:
    cfg = C.atari100k_config()
    again = C.config_from_dict(json.loads(json.dumps(cfg.to_dict())))
    assert again.to_dict() == cfg.to_dict()


def test_mults_survive_json_as_a_tuple() -> None:
    cfg = C.config_from_dict({"agent": {"enc": {"mults": [2, 3, 4, 4]}}})
    assert cfg.agent.enc.mults == (2, 3, 4, 4)


@pytest.mark.parametrize(
    "name, backend, run_id",
    [
        ("ale_frostbite.json", "ale", "dreamerv3_ale_frostbite"),
        ("frostbite.json", "playtrain", "dreamerv3_playtrain_frostbite"),
        ("debug_ale.json", "ale", "dreamerv3_debug_ale"),
        ("debug_playtrain.json", "playtrain", "dreamerv3_debug_playtrain"),
    ],
)
def test_shipped_configs_load(name: str, backend: str, run_id: str) -> None:
    path = Path(__file__).resolve().parents[1] / "configs" / "dreamerv3" / name
    cfg = C.load_config(path)
    assert cfg.env_backend == backend
    assert cfg.run_id == run_id
    assert cfg.game == "frostbite"


def test_the_two_full_configs_differ_only_in_identity_and_backend() -> None:
    """The ALE gate and the PlayTrain arm must run the SAME frozen agent: the only
    legitimate differences are the run id and the backend (MISSION rule 2)."""
    root = Path(__file__).resolve().parents[1] / "configs" / "dreamerv3"
    ale = C.load_config(root / "ale_frostbite.json").to_dict()
    pt = C.load_config(root / "frostbite.json").to_dict()
    differing = {k for k in ale if ale[k] != pt[k]}
    assert differing == {"run_id", "env_backend"}


# ----------------------------------------------------------------------
# Debug preset (yaml:debug)
# ----------------------------------------------------------------------
def test_debug_preset_mirrors_the_official_debug_section() -> None:
    cfg = C.debug_config()
    # yaml:debug lines 205-210
    assert cfg.batch_size == 8
    assert cfg.batch_length == 10
    assert cfg.report_length == 5
    assert cfg.run.report_every == 10.0
    assert cfg.run.log_every == 5.0
    assert cfg.run.save_every == 15.0
    assert cfg.run.train_ratio == 8.0
    assert cfg.replay.size == 1e4
    # yaml:debug.agent regex overrides, lines 212-220
    assert cfg.agent.rewhead.bins == cfg.agent.value.bins == 5
    assert cfg.agent.policy.layers == cfg.agent.enc.layers == cfg.agent.dec.layers == 1
    assert cfg.agent.policy.units == cfg.agent.value.units == 8
    assert cfg.agent.rssm.stoch == 2
    assert cfg.agent.rssm.classes == 4
    assert cfg.agent.rssm.deter == 8
    assert cfg.agent.rssm.hidden == 3
    assert cfg.agent.rssm.blocks == 4
    assert cfg.agent.enc.depth == cfg.agent.dec.depth == 2
    # D-023: the two deliberate differences from yaml:debug
    assert cfg.run.envs == 1  # yaml says 4; our driver is single-env
    assert cfg.compute_dtype == "float32"  # yaml:debug.jax.platform is cpu
    cfg.validate()


def test_debug_preset_derived_clocks() -> None:
    cfg = C.debug_config()
    assert cfg.batch_steps == 80  # 8 * 10
    assert cfg.train_ratio_per_agent_step == 0.1  # 8 / 80
    assert cfg.sequence_length == 11  # 1 * 10 + 1


def test_debug_preset_via_json_preset_key() -> None:
    cfg = C.config_from_dict({"preset": "debug", "seed": 3})
    assert cfg.batch_size == 8
    assert cfg.seed == 3


def test_unknown_preset() -> None:
    with pytest.raises(ValueError, match="unknown preset"):
        C.config_from_dict({"preset": "atari200k"})


# ----------------------------------------------------------------------
# Bookkeeping helpers
# ----------------------------------------------------------------------
def test_run_and_results_dirs() -> None:
    cfg = C.atari100k_config()
    assert C.run_dir(cfg, seed=2).as_posix().endswith(
        "outputs/dreamerv3/dreamerv3_playtrain_frostbite/seed2"
    )
    assert C.results_dir(cfg).as_posix().endswith("results/dreamerv3/dreamerv3_playtrain_frostbite")


def test_seed_for_is_deterministic_and_spread() -> None:
    seeds = [C.seed_for(0, i) for i in range(5)]
    assert seeds == [C.seed_for(0, i) for i in range(5)]
    assert len(set(seeds)) == 5
