"""Greedy-eval helper (playtrain_trainers.impala.eval) used for the live eval metric."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from playtrain_trainers.impala.eval import eval_seeds, greedy_eval
from playtrain_trainers.impala.net import ImpalaNet


OBS_HWC = (16, 16, 3)
A = 5


class _StubEnv:
    """Deterministic, action-independent env: distinct frame per step, gives
    `terminal_reward` once it terminates at `term_at` (so episode return is
    known regardless of the policy's actions)."""

    def __init__(self, term_at=3, terminal_reward=10.0):
        self.action_space = SimpleNamespace(n=A)
        self.t = 0
        self.term_at = term_at
        self.terminal_reward = terminal_reward

    def _obs(self):
        g = torch.Generator().manual_seed(1000 + self.t)
        return torch.randint(0, 256, OBS_HWC, dtype=torch.uint8, generator=g).numpy()

    def reset(self, seed=None):
        self.t = 0
        return self._obs(), {}

    def step(self, action):
        self.t += 1
        terminated = self.t >= self.term_at
        return self._obs(), (self.terminal_reward if terminated else 0.0), \
            terminated, False, {}

    def close(self):
        pass


def test_eval_seeds_fixed_vs_procedural():
    # fixed instance -> the memorized seed, one episode
    assert eval_seeds(7, 8) == [7]
    # procedural -> held-out block disjoint from training seeds
    assert eval_seeds(None, 3) == [9_000_000, 9_000_001, 9_000_002]


@pytest.mark.parametrize("use_lstm", [False, True])
def test_greedy_eval_accounts_returns_and_threshold(use_lstm):
    net = ImpalaNet((3, 16, 16), num_actions=A, features_dim=32, use_lstm=use_lstm)
    cpu = torch.device("cpu")

    # return is 10 per episode -> win at threshold 5, not at threshold 50.
    win_rate, mean_ret = greedy_eval(
        net, _StubEnv(term_at=3, terminal_reward=10.0), seeds=[0, 1],
        device=cpu, max_steps=50, win_threshold=5.0,
    )
    assert mean_ret == 10.0
    assert win_rate == 1.0

    win_rate_hi, _ = greedy_eval(
        net, _StubEnv(term_at=3, terminal_reward=10.0), seeds=[0, 1],
        device=cpu, max_steps=50, win_threshold=50.0,
    )
    assert win_rate_hi == 0.0
