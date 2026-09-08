"""End-to-end smoke test for recurrent (LSTM) PPO in train_ppo_clean.

Runs the real train() loop on a fake SB3-shaped vec-env (no Node workers), so it
exercises the actual rollout state-threading and the env-column minibatched
update — the new trainer code — not just the net. Kept tiny (cpu, a few steps)
so it's fast.

Covers:
  - LSTM training runs end-to-end and writes final.pt with lstm.* weights
  - feed_prev_action_reward grows the LSTM input width as expected
  - the n_envs % n_minibatches divisibility guard fires
  - feedforward path still runs (regression)
  - the LSTM core actually moves (params change over training)
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import playtrain_trainers.train_ppo_clean as T
from playtrain_trainers.train_ppo_clean import Config


class _FakeVenv:
    """SB3-shaped: reset()->obs, step()->(obs, rewards, dones, infos). Random
    obs, a synthetic reward, and a staggered done every few steps so episode
    boundaries (and the LSTM reset path) are exercised."""

    def __init__(self, n_envs: int, H: int = 16, W: int = 16, C: int = 3,
                 n_actions: int = 5):
        self.num_envs = n_envs
        self._shape = (H, W, C)
        self._n_actions = n_actions
        self.observation_space = SimpleNamespace(shape=(H, W, C))
        self.action_space = SimpleNamespace(n=n_actions)
        self._rng = np.random.RandomState(0)
        self._t = np.zeros(n_envs, dtype=int)

    def _obs(self):
        return self._rng.randint(0, 256, (self.num_envs, *self._shape), dtype=np.uint8)

    def reset(self):
        self._t[:] = 0
        return self._obs()

    def step(self, actions):
        self._t += 1
        obs = self._obs()
        rewards = self._rng.randn(self.num_envs).astype(np.float32)
        # env i is "done" every (3 + i) steps -> staggered boundaries
        dones = np.array([self._t[i] % (3 + i) == 0 for i in range(self.num_envs)])
        infos = [{"TimeLimit.truncated": bool(self._t[i] % 2 == 0)}
                 for i in range(self.num_envs)]
        for i in np.where(dones)[0]:
            infos[i]["terminal_observation"] = obs[i]
            self._t[i] = 0
        return obs, rewards, dones, infos

    def close(self):
        pass


def _cfg(**over) -> Config:
    base = dict(
        game="fake", total_timesteps=64, n_envs=4, n_steps=4, n_minibatches=2,
        n_epochs=2, device="cpu", anneal_lr=False, seed=0,
    )
    base.update(over)
    return Config(**base)


def _run(tmp_path, monkeypatch, cfg) -> dict:
    monkeypatch.setattr(T, "build_venv", lambda c, **_: _FakeVenv(c.n_envs))
    T.train(cfg)
    final = tmp_path / "final.pt"
    assert final.exists(), "training did not write final.pt"
    return torch.load(final, map_location="cpu", weights_only=False)


def test_lstm_ppo_runs_end_to_end(tmp_path, monkeypatch):
    cfg = _cfg(use_lstm=True, log_dir=str(tmp_path))
    payload = _run(tmp_path, monkeypatch, cfg)
    keys = payload["model"].keys()
    assert any(k.startswith("lstm.") for k in keys), "no LSTM weights in checkpoint"
    assert payload["global_step"] == cfg.total_timesteps
    assert payload["config"]["use_lstm"] is True
    assert payload["config"]["feed_prev_action_reward"] is True


def test_feed_prev_action_reward_grows_input(tmp_path, monkeypatch):
    """With a/r feeding, weight_ih_l0 has input width features+ n_actions+1."""
    fed = _run(tmp_path / "fed", monkeypatch,
               _cfg(use_lstm=True, feed_prev_action_reward=True,
                    log_dir=str(tmp_path / "fed")))
    plain = _run(tmp_path / "plain", monkeypatch,
                 _cfg(use_lstm=True, feed_prev_action_reward=False,
                      log_dir=str(tmp_path / "plain")))
    fed_in = fed["model"]["lstm.weight_ih_l0"].shape[1]
    plain_in = plain["model"]["lstm.weight_ih_l0"].shape[1]
    assert fed_in == plain_in + 5 + 1, (fed_in, plain_in)  # +one_hot(A=5) +reward


def test_divisibility_guard(tmp_path, monkeypatch):
    """use_lstm with n_envs not divisible by n_minibatches must raise."""
    monkeypatch.setattr(T, "build_venv", lambda c, **_: _FakeVenv(c.n_envs))
    cfg = _cfg(use_lstm=True, n_envs=4, n_minibatches=3, log_dir=str(tmp_path))
    with pytest.raises(ValueError, match="divisible by n_minibatches"):
        T.train(cfg)


def test_lstm_symlog_reward_off_runs(tmp_path, monkeypatch):
    """lstm_symlog_reward=False (feed the reward_clip-shaped reward verbatim)
    trains end-to-end — exercises the trainer's non-symlog reward-cue branch."""
    cfg = _cfg(use_lstm=True, lstm_symlog_reward=False, reward_clip="sign",
               log_dir=str(tmp_path))
    payload = _run(tmp_path, monkeypatch, cfg)
    assert payload["config"]["lstm_symlog_reward"] is False
    assert any(k.startswith("lstm.") for k in payload["model"].keys())


def test_feedforward_still_runs(tmp_path, monkeypatch):
    """Regression: the default (feedforward) path trains end-to-end and has no
    LSTM weights."""
    payload = _run(tmp_path, monkeypatch, _cfg(use_lstm=False, log_dir=str(tmp_path)))
    assert not any(k.startswith("lstm.") for k in payload["model"].keys())


def test_bf16_flag_runs(tmp_path, monkeypatch):
    """bf16 autocast is cuda-only (no-op on the cpu test device), but the flag
    must be accepted and the run must complete + record it — guards the plumbing
    (amp_ctx wrapping the rollout/update) against import/indentation breakage."""
    payload = _run(tmp_path, monkeypatch,
                   _cfg(use_lstm=True, bf16=True, log_dir=str(tmp_path)))
    assert payload["config"]["bf16"] is True
    assert payload["global_step"] == 64  # total_timesteps


def test_lstm_core_actually_trains(tmp_path, monkeypatch):
    """The LSTM weights must change over training — proves the recurrent core is
    in the optimized graph through the env-column minibatched update, not bypassed."""
    monkeypatch.setattr(T, "build_venv", lambda c, **_: _FakeVenv(c.n_envs))
    cfg = _cfg(use_lstm=True, total_timesteps=128, log_dir=str(tmp_path))

    # capture initial LSTM weights by seeding identically and building the net
    torch.manual_seed(cfg.seed)
    from playtrain_trainers.policy import ActorCritic
    ref = ActorCritic(n_actions=5, in_channels=3, input_hw=16, use_lstm=True,
                      feed_prev_action_reward=True)
    w0 = ref.lstm.weight_hh_l0.detach().clone()

    T.train(cfg)
    payload = torch.load(tmp_path / "final.pt", map_location="cpu", weights_only=False)
    w1 = payload["model"]["lstm.weight_hh_l0"]
    assert not torch.allclose(w0, w1), "LSTM core weights did not change during training"
