"""Alignment + bookkeeping contract for the vectorized rollout worker.

Mirrors test_impala_central_alignment.py for inference_mode="vec": the slot
convention learn() assumes is that buffers[t+1] holds the env output OF step
t+1 together with the agent output computed FROM frame t (infer-then-step).
A fake NativeVecEnv with fully deterministic dynamics lets us recompute what
the model saw and assert the stored logits/actions line up, and that
episode_return/episode_step follow Environment's semantics (done rows report
the COMPLETED episode; counters restart next step).
"""
from __future__ import annotations

import queue
import sys
import types

import numpy as np
import torch

from playtrain_trainers.impala.net import ImpalaNet
from playtrain_trainers.impala.vec_actor import (
    act_vec,
    create_vec_buffers,
    create_vec_state_buffers,
    create_weight_state,
    publish_weights,
)

M, T, A = 3, 6, 8
OBS = 64
EP_LEN = 4  # fake env terminates every EP_LEN steps (env 0 only)


class FakeVecEnv:
    """Deterministic stand-in for NativeVecEnv: obs encodes (env, step) so
    every frame is unique; env 0 terminates every EP_LEN steps with reward 5,
    other envs never terminate and pay reward 1 per step."""

    def __init__(self, game, num_envs, **kw):
        self.num_envs = num_envs
        self.t = 0
        self.calls = []

    def set_autoreset_seeds(self, *a, **kw):
        pass

    def _obs(self):
        obs = np.zeros((self.num_envs, OBS, OBS, 3), dtype=np.uint8)
        for i in range(self.num_envs):
            obs[i, :, :, 0] = (self.t * 7 + i * 31) % 251
            obs[i, :, :, 1] = i
        return obs

    def reset(self, seeds=None):
        self.t = 0
        return self._obs()

    def step(self, actions):
        self.calls.append(np.array(actions, copy=True))
        self.t += 1
        rew = np.ones(self.num_envs, dtype=np.float32)
        term = np.zeros(self.num_envs, dtype=bool)
        if self.t % EP_LEN == 0:
            rew[0] = 5.0
            term[0] = True  # autoreset: obs below is already the reset frame
        return self._obs(), rew, term, np.zeros(self.num_envs, dtype=bool), {}

    def close(self):
        pass


def _run_one_rollout(monkeypatch, use_lstm=False, rollouts=1):
    fake_mod = types.ModuleType("playtrain.runtime.native_vec_env")
    fake_mod.NativeVecEnv = FakeVecEnv
    monkeypatch.setitem(sys.modules, "playtrain.runtime.native_vec_env", fake_mod)

    torch.manual_seed(0)
    model = ImpalaNet((3, OBS, OBS), A, features_dim=32, use_lstm=use_lstm)
    weight_state = create_weight_state(model)
    buffers = create_vec_buffers((3, OBS, OBS), A, T, M, num_buffers=rollouts)
    state_buffers = create_vec_state_buffers(model, M, num_buffers=rollouts)

    free_q, full_q = queue.Queue(), queue.Queue()
    for r in range(rollouts):
        free_q.put(r)
    free_q.put(None)  # stop after `rollouts` rollouts

    env_spec = dict(game_path="fake", num_envs=M, frame_skip=1, frame_stack=1,
                    max_steps=10_000, obs_size=OBS, env_threads=1,
                    seed_mode="formula", seed_pool=None, fixed_seed=None,
                    base_seed=7)
    model_spec = dict(obs_shape=(3, OBS, OBS), num_actions=A, features_dim=32,
                      use_lstm=use_lstm)
    act_vec(0, free_q, full_q, buffers, state_buffers, weight_state, env_spec,
            model_spec, T, "cpu")
    for r in range(rollouts):
        assert full_q.get_nowait() == r
    return model, buffers, state_buffers


def test_alignment_agent_output_from_pre_step_frame(monkeypatch):
    model, buffers, _ = _run_one_rollout(monkeypatch)
    model.train()
    # Stored policy_logits at t+1 must equal a forward on the frame stored at
    # t (the PRE-step frame) — the infer-then-step contract learn() assumes.
    for t in range(T):
        frame_t = buffers["frame"][0][t].unsqueeze(0)  # (1, M, C, H, W)
        with torch.no_grad():
            out, _ = model({
                "frame": frame_t,
                "reward": buffers["reward"][0][t].unsqueeze(0),
                "done": buffers["done"][0][t].unsqueeze(0),
                "last_action": buffers["last_action"][0][t].unsqueeze(0),
            }, ())
        torch.testing.assert_close(
            buffers["policy_logits"][0][t + 1], out["policy_logits"][0],
            msg=f"logits at t+1 != forward(frame[t]) at t={t}")
        torch.testing.assert_close(
            buffers["baseline"][0][t + 1], out["baseline"][0])


def test_env_output_is_post_step_and_last_action_matches(monkeypatch):
    _, buffers, _ = _run_one_rollout(monkeypatch)
    # frame[t+1] must be the obs AFTER taking action[t+1]; with the fake env
    # obs R-channel = (step*7 + env*31) % 251, so step index is recoverable.
    for t in range(T):
        for i in range(M):
            expect = ((t + 1) * 7 + i * 31) % 251
            assert int(buffers["frame"][0][t + 1, i, 0, 0, 0]) == expect
        # last_action stored with the env output == the action just taken.
        torch.testing.assert_close(buffers["last_action"][0][t + 1],
                                   buffers["action"][0][t + 1])


def test_episode_bookkeeping_matches_environment_semantics(monkeypatch):
    _, buffers, _ = _run_one_rollout(monkeypatch)
    done = buffers["done"][0]           # (T+1, M)
    ep_ret = buffers["episode_return"][0]
    ep_step = buffers["episode_step"][0]
    # Env 0 terminates at step EP_LEN with reward 5 after (EP_LEN-1) 1-rewards:
    # the done row must report the COMPLETED episode's totals.
    t_done = EP_LEN  # buffer row EP_LEN corresponds to env step EP_LEN
    assert bool(done[t_done, 0])
    assert float(ep_ret[t_done, 0]) == (EP_LEN - 1) * 1.0 + 5.0
    assert int(ep_step[t_done, 0]) == EP_LEN
    # The step after a done restarts the counters.
    assert int(ep_step[t_done + 1, 0]) == 1
    assert float(ep_ret[t_done + 1, 0]) == 1.0
    # A never-done env keeps accumulating.
    assert not done[1:, 1].any()
    assert int(ep_step[T, 1]) == T


def test_weight_publication_seqlock_roundtrip():
    torch.manual_seed(1)
    src = ImpalaNet((3, OBS, OBS), A, features_dim=32)
    dst = ImpalaNet((3, OBS, OBS), A, features_dim=32)
    ws = create_weight_state(src)
    with torch.no_grad():
        for p in src.parameters():
            p.add_(1.0)
    publish_weights(ws, src)
    from playtrain_trainers.impala.vec_actor import maybe_reload_weights
    v = maybe_reload_weights(ws, dst, -1)
    assert v == int(ws["version"].item()) and v % 2 == 0
    for (n1, p1), (n2, p2) in zip(src.state_dict().items(),
                                  dst.state_dict().items()):
        assert n1 == n2
        torch.testing.assert_close(p1, p2)


def test_lstm_learner_replay_reproduces_worker_logits(monkeypatch):
    """The LSTM contract end-to-end: a learner-style forward over the whole
    (T+1, M) slot, starting from the slot's snapshotted (h, c), must
    reproduce the worker's stored logits — out[t] == stored[t+1], across the
    fake env's mid-unroll episode boundary (done-mask state reset)."""
    model, buffers, state_buffers = _run_one_rollout(monkeypatch, use_lstm=True,
                                                     rollouts=2)
    model.train()
    for slot in range(2):  # slot 1 starts from a NON-ZERO carried state
        batch = {k: buffers[k][slot]
                 for k in ("frame", "reward", "done", "last_action")}
        init = tuple(t.clone() for t in state_buffers[slot])
        with torch.no_grad():
            out, _ = model(batch, init)
        torch.testing.assert_close(
            out["policy_logits"][:-1], buffers["policy_logits"][slot][1:],
            msg=f"slot {slot}: learner replay diverges from worker logits")
        torch.testing.assert_close(
            out["baseline"][:-1], buffers["baseline"][slot][1:])


def test_lstm_second_slot_snapshot_is_nonzero(monkeypatch):
    """Guards the snapshot timing: after a full unroll the carried state must
    be non-zero and stored for the next slot (a zeroed snapshot would mean we
    snapshot AFTER reset or at the wrong point)."""
    _, _, state_buffers = _run_one_rollout(monkeypatch, use_lstm=True,
                                           rollouts=2)
    h1, c1 = state_buffers[1]
    assert float(h1.abs().sum()) > 0 and float(c1.abs().sum()) > 0


# ---------------------------------------------------------------------------
# Double-buffered worker (act_vec_db): same contracts, ping-pong env.
# ---------------------------------------------------------------------------

class FakePingPongEnv:
    """Deterministic ping-pong stand-in: same dynamics as FakeVecEnv but with
    per-env step counters and group send/wait (env i in group i // B)."""

    def __init__(self, game, group_size, **kw):
        self.B = group_size
        self.N = 2 * group_size
        self.t = np.zeros(self.N, dtype=np.int64)
        self._pending = {}

    def set_autoreset_seeds(self, *a, **kw):
        pass

    def _obs(self, ids):
        obs = np.zeros((len(ids), OBS, OBS, 3), dtype=np.uint8)
        for j, i in enumerate(ids):
            obs[j, :, :, 0] = (self.t[i] * 7 + i * 31) % 251
            obs[j, :, :, 1] = i
        return obs

    def reset(self, seeds=None):
        self.t[:] = 0
        return self._obs(range(self.N))

    def send(self, g, actions):
        self._pending[g] = np.array(actions, copy=True)

    def wait(self, g):
        ids = list(range(g * self.B, (g + 1) * self.B))
        assert g in self._pending, "wait() before send()"
        del self._pending[g]
        self.t[ids] += 1
        rew = np.ones(self.B, dtype=np.float32)
        term = np.zeros(self.B, dtype=bool)
        for j, i in enumerate(ids):
            if i == 0 and self.t[i] % EP_LEN == 0:
                rew[j] = 5.0
                term[j] = True
        return self._obs(ids), rew, term, np.zeros(self.B, dtype=bool)

    def close(self):
        pass


def _run_db_rollouts(monkeypatch, use_lstm, slots_per_half=2):
    from playtrain_trainers.impala.vec_actor import act_vec_db
    fake_mod = types.ModuleType("playtrain.runtime.native_vec_env")
    fake_mod.PingPongVecEnv = FakePingPongEnv
    monkeypatch.setitem(sys.modules, "playtrain.runtime.native_vec_env", fake_mod)

    torch.manual_seed(0)
    n_slots = 2 * slots_per_half
    model = ImpalaNet((3, OBS, OBS), A, features_dim=32, use_lstm=use_lstm)
    weight_state = create_weight_state(model)
    buffers = create_vec_buffers((3, OBS, OBS), A, T, M, num_buffers=n_slots)
    state_buffers = create_vec_state_buffers(model, M, num_buffers=n_slots)

    free_q, full_q = queue.Queue(), queue.Queue()
    for r in range(n_slots):
        free_q.put(r)
    free_q.put(None)
    free_q.put(None)

    env_spec = dict(game_path="fake", num_envs=M, frame_skip=1, frame_stack=1,
                    max_steps=10_000, obs_size=OBS, env_threads=1,
                    seed_mode="formula", seed_pool=None, fixed_seed=None,
                    base_seed=7, render_skip=False)
    model_spec = dict(obs_shape=(3, OBS, OBS), num_actions=A, features_dim=32,
                      use_lstm=use_lstm)
    act_vec_db(0, free_q, full_q, buffers, state_buffers, weight_state,
               env_spec, model_spec, T, "cpu")
    done_slots = []
    while not full_q.empty():
        done_slots.append(full_q.get_nowait())
    assert len(done_slots) >= n_slots - 1, f"only {done_slots} completed"
    return model, buffers, state_buffers, done_slots


def test_db_alignment_and_env_progression(monkeypatch):
    model, buffers, _, slots = _run_db_rollouts(monkeypatch, use_lstm=False)
    model.train()
    for slot in slots:
        # env column identity: channel-1 of each frame encodes the env id;
        # each slot must hold ONE group's envs, constant across rows.
        ids0 = [int(buffers["frame"][slot][0, i, 1, 0, 0]) for i in range(M)]
        for t in range(T + 1):
            ids_t = [int(buffers["frame"][slot][t, i, 1, 0, 0]) for i in range(M)]
            assert ids_t == ids0, f"slot {slot}: env columns shuffled at t={t}"
        # infer-then-step alignment (same contract as act_vec).
        for t in range(T):
            with torch.no_grad():
                out, _ = model({
                    "frame": buffers["frame"][slot][t].unsqueeze(0),
                    "reward": buffers["reward"][slot][t].unsqueeze(0),
                    "done": buffers["done"][slot][t].unsqueeze(0),
                    "last_action": buffers["last_action"][slot][t].unsqueeze(0),
                }, ())
            torch.testing.assert_close(
                buffers["policy_logits"][slot][t + 1], out["policy_logits"][0],
                msg=f"slot {slot}: logits misaligned at t={t}")


def test_db_lstm_learner_replay(monkeypatch):
    model, buffers, state_buffers, slots = _run_db_rollouts(
        monkeypatch, use_lstm=True)
    model.train()
    for slot in slots:
        batch = {k: buffers[k][slot]
                 for k in ("frame", "reward", "done", "last_action")}
        init = tuple(t.clone() for t in state_buffers[slot])
        with torch.no_grad():
            out, _ = model(batch, init)
        torch.testing.assert_close(
            out["policy_logits"][:-1], buffers["policy_logits"][slot][1:],
            msg=f"slot {slot}: LSTM replay diverges")
