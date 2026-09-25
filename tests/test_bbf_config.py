"""Tests for playtrain_trainers.bbf.config.

These pin the frozen BBF hyperparameters (PROTOCOL sections 1-2, transcribed
from reference/BBF.gin) so a later edit to the dataclass cannot silently
change what "BBF" means in this repo.
"""
from __future__ import annotations

import dataclasses
import json

import pytest

from playtrain_trainers.bbf.config import (
    BBFConfig,
    config_from_dict,
    load_config,
    results_dir,
    run_dir,
    seed_for,
    set_global_seeds,
)


# ----------------------------------------------------------------------
# The frozen values, transcribed independently of config.py from
# playtrain-internal/bbf-loop/reference/BBF.gin.
# ----------------------------------------------------------------------
GIN_AGENT = {
    "gamma": 0.997,
    "min_replay_history": 2000,
    "update_period": 1,
    "target_update_period": 1,
    "epsilon_train": 0.0,
    "epsilon_eval": 0.001,
    "epsilon_decay_period": 2001,
    "noisy": False,
    "dueling": True,
    "double_dqn": True,
    "distributional": True,
    "num_atoms": 51,
    "update_horizon": 3,
    "max_update_horizon": 10,
    "min_gamma": 0.97,
    "cycle_steps": 10_000,
    "reset_every": 20_000,
    "shrink_factor": 0.5,
    "perturb_factor": 0.5,
    "no_resets_after": 100_000,
    "log_every": 100,
    "replay_ratio": 64,
    "batches_to_group": 2,
    "batch_size": 32,
    "spr_weight": 5.0,
    "jumps": 5,
    "data_augmentation": True,
    "replay_scheme": "prioritized",
    "half_precision": False,
    "learning_rate": 1e-4,
    "encoder_learning_rate": 1e-4,
    "target_update_tau": 0.005,
    "target_action_selection": True,
    "renormalize": True,
    "hidden_dim": 2048,
    "encoder_type": "impala",
    "width_scale": 4,
    "num_blocks": 2,
    "adam_eps": 0.00015,
    "weight_decay": 0.1,
    "replay_capacity": 200_000,
}

GIN_ENV = {
    "sticky_actions": False,
    "terminal_on_life_loss": True,
    "training_steps": 100_000,
    "num_eval_episodes": 100,
    "num_eval_envs": 100,
    "num_train_envs": 1,
    "max_noops": 30,
    "max_steps_per_episode": 27_000,
}


@pytest.mark.parametrize("field,value", sorted(GIN_AGENT.items()))
def test_agent_defaults_match_gin(field, value):
    assert getattr(BBFConfig(), field) == value


@pytest.mark.parametrize("field,value", sorted(GIN_ENV.items()))
def test_env_defaults_match_gin(field, value):
    assert getattr(BBFConfig(), field) == value


def test_shrink_perturb_keys_match_gin():
    # gin: BBFAgent.shrink_perturb_keys = "encoder,transition_model"
    assert BBFConfig().shrink_perturb_keys == ("encoder", "transition_model")


# ----------------------------------------------------------------------
# PlayTrain mapping defaults (PROTOCOL section 4)
# ----------------------------------------------------------------------
def test_playtrain_mapping_defaults():
    cfg = BBFConfig()
    assert cfg.env_backend == "playtrain"   # D-013
    assert cfg.game == "frostbite"
    assert cfg.frame_skip == 4              # D-001
    assert cfg.obs_size == 84               # D-002
    assert cfg.obs_mode == "grayscale"      # D-002
    assert cfg.frame_stack == 4             # D-002
    assert cfg.max_pool_frames is False     # D-002: no flicker to pool
    assert cfg.action_space == "default8"   # D-003
    assert cfg.reward_clip == 1.0           # D-011
    assert cfg.eval_every == 10_000         # D-006
    assert cfg.eval_episodes_curve == 10    # D-006


# ----------------------------------------------------------------------
# Derived quantities
# ----------------------------------------------------------------------
def test_replay_ratio_default_is_rr2():
    # gin replay_ratio 64 / batch 32 / 1 env = 2 gradient steps per env step.
    cfg = BBFConfig()
    assert cfg.gradient_steps_per_env_step == 2
    assert cfg.total_gradient_steps == 200_000
    assert len(cfg.reset_env_steps) == 3  # D-033: ~20k, 40k, 60k env steps


def test_replay_ratio_256_is_rr8():
    # The paper's Table A.1 headline setting (flag F-001).
    cfg = BBFConfig(replay_ratio=256)
    cfg.validate()
    assert cfg.gradient_steps_per_env_step == 8
    assert cfg.total_gradient_steps == 800_000


def test_frame_and_episode_units():
    cfg = BBFConfig()
    # 100k agent steps at skip 4 = 400k game frames (PROTOCOL section 2).
    assert cfg.max_env_frames == 400_000
    # D-012: the runtime's max_steps counts FRAMES, not agent steps.
    assert cfg.max_steps_frames == 108_000


def test_obs_shape_grayscale_and_rgb():
    assert BBFConfig().obs_shape == (4, 84, 84)
    assert BBFConfig(obs_mode="rgb").obs_shape == (12, 84, 84)
    # D-002's second arm: native 64 px RGB stacked on the channel axis.
    assert BBFConfig(obs_mode="rgb", obs_size=64).obs_shape == (12, 64, 64)


def test_c51_support_is_symmetric():
    cfg = BBFConfig()
    assert (cfg.v_min, cfg.v_max) == (-10.0, 10.0)  # D-015


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------
def test_default_config_validates():
    BBFConfig().validate()


@pytest.mark.parametrize(
    "overrides",
    [
        {"env_backend": "dopamine"},
        {"obs_mode": "depth"},
        {"replay_scheme": "sampled"},
        # 64 * 1 % 48 != 0: replay ratio not a whole number of batches.
        {"batch_size": 48},
        # 4 gradient steps do not split into 3 groups.
        {"batches_to_group": 3},
        {"cycle_steps": 0},
        # max_update_horizon below update_horizon would anneal upward.
        {"max_update_horizon": 2},
        {"min_gamma": 0.999},
        {"gamma": 1.0},
        {"num_atoms": 1},
        {"v_max": 0.0},
        {"shrink_factor": 1.5},
        {"perturb_factor": -0.1},
        {"frame_skip": 0},
        {"frame_stack": 0},
        {"max_noops": -1},
        {"replay_capacity": 1000},
    ],
)
def test_validate_rejects(overrides):
    with pytest.raises(ValueError):
        BBFConfig(**overrides).validate()


# ----------------------------------------------------------------------
# JSON loading
# ----------------------------------------------------------------------
def test_config_from_dict_rejects_unknown_key():
    # A typo in a frozen hyperparameter must not silently fall back.
    with pytest.raises(ValueError, match="unknown BBF config keys"):
        config_from_dict({"replay_rato": 256})


def test_config_from_dict_applies_overrides():
    cfg = config_from_dict({"replay_ratio": 256, "run_id": "rr8", "seed": 3})
    assert (cfg.replay_ratio, cfg.run_id, cfg.seed) == (256, "rr8", 3)


def test_config_from_dict_validates():
    with pytest.raises(ValueError):
        config_from_dict({"gamma": 1.5})


def test_to_dict_round_trips(tmp_path):
    cfg = BBFConfig(replay_ratio=256, run_id="rr8", seed=7, env_backend="ale")
    path = tmp_path / "c.json"
    path.write_text(json.dumps(cfg.to_dict(), indent=2))
    back = load_config(path)
    assert back == cfg


def test_to_dict_records_derived_units():
    d = BBFConfig().to_dict()
    assert d["_derived"]["gradient_steps_per_env_step"] == 2
    assert d["_derived"]["max_steps_frames"] == 108_000
    assert d["_derived"]["obs_shape"] == [4, 84, 84]


def test_shipped_configs_load():
    # Every configs/bbf/*.json must be loadable and valid.
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "configs" / "bbf"
    shipped = sorted(root.glob("*.json"))
    assert shipped, f"no configs found under {root}"
    for path in shipped:
        cfg = load_config(path)
        assert cfg.run_id, path


# ----------------------------------------------------------------------
# Run dirs and seeding
# ----------------------------------------------------------------------
def test_run_and_results_dirs():
    cfg = BBFConfig(run_id="rr2", seed=4)
    assert run_dir(cfg).as_posix() == "outputs/bbf/rr2/seed4"
    assert run_dir(cfg, seed=9).as_posix() == "outputs/bbf/rr2/seed9"
    assert results_dir(cfg).as_posix() == "results/bbf/rr2"
    assert run_dir(cfg, root="/tmp/x").as_posix() == "/tmp/x/rr2/seed4"


def test_seed_for_is_deterministic_and_distinct():
    seeds = [seed_for(0, i) for i in range(10)]
    assert seeds == [seed_for(0, i) for i in range(10)]
    assert len(set(seeds)) == 10
    assert all(0 <= s < 2**31 - 1 for s in seeds)
    # A different base gives a different family.
    assert set(seeds).isdisjoint(seed_for(1, i) for i in range(10))


def test_seed_for_rejects_negative():
    with pytest.raises(ValueError):
        seed_for(0, -1)


def test_set_global_seeds_reproduces_streams():
    import random

    import numpy as np
    import torch

    def draw():
        return (random.random(), float(np.random.rand()), float(torch.rand(1)))

    set_global_seeds(123)
    a = draw()
    set_global_seeds(123)
    b = draw()
    assert a == b


def test_every_field_is_json_serializable():
    # config.json must be writable without a custom encoder.
    json.dumps(BBFConfig().to_dict())
    assert {f.name for f in dataclasses.fields(BBFConfig)} >= set(GIN_AGENT)
