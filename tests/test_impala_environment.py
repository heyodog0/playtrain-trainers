"""Tests for playtrain_trainers.impala.environment (the single-env adapter)."""
from __future__ import annotations

import numpy as np
import torch

from playtrain_trainers.impala.environment import Environment


class _MockEnv:
    """Minimal gymnasium-shaped env. Step ramps reward 1.0 per call; episode
    ends after `max_steps`. Obs is a fixed HWC uint8 array."""

    def __init__(self, max_steps: int = 3, obs_shape=(4, 4, 3)):
        self.max_steps = max_steps
        self.obs_shape = obs_shape
        self.t = 0
        self.action_space_n = 5

    def reset(self, *, seed=None):
        self.t = 0
        return np.full(self.obs_shape, 7, dtype=np.uint8), {}

    def step(self, action):
        self.t += 1
        obs = np.full(self.obs_shape, self.t, dtype=np.uint8)
        terminated = self.t >= self.max_steps
        return obs, 1.0, terminated, False, {}

    def close(self):
        pass


def test_initial_emits_time_major_dict():
    env = Environment(_MockEnv(), chw_transpose=True)
    out = env.initial()
    assert out["frame"].shape == (1, 1, 3, 4, 4)  # CHW
    assert out["frame"].dtype == torch.uint8
    assert out["reward"].shape == (1, 1) and out["reward"].item() == 0.0
    assert out["done"].shape == (1, 1) and out["done"].item() is True
    assert out["episode_return"].item() == 0.0
    assert out["episode_step"].item() == 0
    assert out["last_action"].item() == 0


def test_step_bookkeeping_running_then_reset_on_done():
    env = Environment(_MockEnv(max_steps=2), chw_transpose=True)
    env.initial()

    out1 = env.step(torch.tensor(3))
    assert out1["reward"].item() == 1.0
    assert out1["done"].item() is False
    assert out1["episode_return"].item() == 1.0
    assert out1["episode_step"].item() == 1
    assert out1["last_action"].item() == 3

    out2 = env.step(torch.tensor(4))
    # episode ends; episode_return reflects FINAL pre-reset value
    assert out2["done"].item() is True
    assert out2["episode_return"].item() == 2.0
    assert out2["episode_step"].item() == 2
    # but internal bookkeeping is reset for the next episode
    assert env.episode_return.item() == 0.0
    assert env.episode_step.item() == 0


def test_chw_transpose_off_keeps_hwc():
    env = Environment(_MockEnv(), chw_transpose=False)
    out = env.initial()
    assert out["frame"].shape == (1, 1, 4, 4, 3)


def test_fixed_seed_forces_same_seed_on_every_reset():
    """When fixed_seed is set, EVERY reset (initial + auto-reset on done)
    must pass that seed to the underlying env. Mirrors PPO's fixed_env_seed:
    memorize one instance instead of training on a procedural distribution."""
    class _SeedRecordingEnv:
        def __init__(self):
            self.seeds_seen: list = []
            self.t = 0
            self.action_space = type("S", (), {"n": 3})()

        def reset(self, *, seed=None):
            self.seeds_seen.append(seed)
            self.t = 0
            return np.zeros((4, 4, 3), dtype=np.uint8), {}

        def step(self, action):
            self.t += 1
            return (np.zeros((4, 4, 3), dtype=np.uint8), 0.0,
                    self.t >= 2, False, {})

        def close(self):
            pass

    gym_env = _SeedRecordingEnv()
    env = Environment(gym_env, chw_transpose=True, fixed_seed=42)
    env.initial()
    # Step twice to trigger one auto-reset.
    env.step(torch.tensor(0))
    env.step(torch.tensor(0))  # done -> auto-reset
    env.step(torch.tensor(0))
    env.step(torch.tensor(0))  # done -> another auto-reset
    # First reset (initial) + 2 auto-resets = 3 reset calls, all with seed=42.
    assert gym_env.seeds_seen == [42, 42, 42], (
        f"fixed_seed=42 must pass through to every reset; got {gym_env.seeds_seen}"
    )


def test_no_fixed_seed_lets_auto_resets_be_random():
    """When fixed_seed is None, initial_seed seeds only the first reset;
    subsequent auto-resets get seed=None (env's own RNG advance / random)."""
    class _SeedRecordingEnv:
        def __init__(self):
            self.seeds_seen: list = []
            self.t = 0
            self.action_space = type("S", (), {"n": 3})()

        def reset(self, *, seed=None):
            self.seeds_seen.append(seed)
            self.t = 0
            return np.zeros((4, 4, 3), dtype=np.uint8), {}

        def step(self, action):
            self.t += 1
            return (np.zeros((4, 4, 3), dtype=np.uint8), 0.0,
                    self.t >= 1, False, {})

        def close(self):
            pass

    gym_env = _SeedRecordingEnv()
    env = Environment(gym_env, chw_transpose=True, initial_seed=7, fixed_seed=None)
    env.initial()
    env.step(torch.tensor(0))  # done -> auto-reset
    env.step(torch.tensor(0))  # done -> auto-reset
    assert gym_env.seeds_seen == [7, None, None]


def test_truncation_also_counts_as_done():
    class _Trunc(_MockEnv):
        def step(self, action):
            self.t += 1
            return (
                np.full(self.obs_shape, self.t, dtype=np.uint8),
                0.5, False, self.t >= self.max_steps, {},
            )
    env = Environment(_Trunc(max_steps=1))
    env.initial()
    out = env.step(torch.tensor(0))
    assert out["done"].item() is True
    assert out["reward"].item() == 0.5
