"""End-to-end pipeline smoke test (single-process).

Drives one IMPALA actor synchronously into the shared buffers, then runs
a learn step on the assembled batch. Verifies the whole glue stack —
environment adapter, net contract, buffer layout, batch dim ordering,
learn() — fits together without involving torch.multiprocessing.

The async actor-learner path is best validated with a real config run on
FASRC; this test exists to catch shape/dtype/contract mismatches before
shipping a job.
"""
from __future__ import annotations

import threading

import numpy as np
import torch

from playtrain_trainers.impala.actor import act
from playtrain_trainers.impala.buffers import (
    create_buffers,
    create_initial_agent_state_buffers,
)
from playtrain_trainers.impala.learn import learn as impala_learn
from playtrain_trainers.impala.net import ImpalaNet


class _FakeImageEnv:
    """Gymnasium-shaped env emitting 64x64x3 uint8 obs. Episode length = 5."""

    def __init__(self, seed=0, num_actions=4):
        self.seed = seed
        self.num_actions = num_actions
        self.t = 0
        self.action_space = type("S", (), {"n": num_actions})()

    def reset(self, *, seed=None):
        self.t = 0
        obs = np.full((64, 64, 3), self.seed & 0xff, dtype=np.uint8)
        return obs, {}

    def step(self, action):
        self.t += 1
        obs = np.full((64, 64, 3), (self.seed + self.t) & 0xff, dtype=np.uint8)
        reward = 1.0 if action == 0 else 0.0
        terminated = self.t >= 5
        return obs, reward, terminated, False, {}

    def close(self):
        pass


class _SingleShotQueue:
    """Mock mp.SimpleQueue that yields one index then `None` (shutdown)."""

    def __init__(self, items):
        self._items = list(items)

    def get(self):
        return self._items.pop(0) if self._items else None

    def put(self, item):
        self._items.append(item)


def test_actor_writes_one_rollout_to_buffer_slot():
    T, A = 8, 4
    bufs = create_buffers(
        obs_shape=(3, 64, 64), num_actions=A, unroll_length=T, num_buffers=2
    )
    # torch.empty leaves garbage in shared memory; zero so "untouched slot"
    # checks below are meaningful.
    for key in bufs:
        for slot in bufs[key]:
            slot.zero_()

    model = ImpalaNet(observation_shape=(3, 64, 64), num_actions=A)
    model.eval()
    iasb = create_initial_agent_state_buffers(model, num_buffers=2)

    free_q = _SingleShotQueue([0, None])
    full_q = _SingleShotQueue([])

    act(
        actor_index=0,
        free_queue=free_q,
        full_queue=full_q,
        model=model,
        buffers=bufs,
        initial_agent_state_buffers=iasb,
        env_fn=lambda seed: (_FakeImageEnv(seed=seed, num_actions=A), seed),
        unroll_length=T,
        chw_transpose=True,
    )

    # Slot 0 fully populated
    assert bufs["frame"][0].shape == (T + 1, 3, 64, 64)
    assert bufs["frame"][0].abs().sum() > 0
    # Slot 1 untouched
    assert bufs["frame"][1].abs().sum() == 0
    # Actor signaled completion
    assert full_q._items == [0]


def test_full_pipeline_actor_to_learn_step():
    """Drive 2 buffer slots via actor, stack into a batch, run learn()."""
    T, A, B = 8, 4, 2
    bufs = create_buffers(
        obs_shape=(3, 64, 64), num_actions=A, unroll_length=T, num_buffers=2
    )
    actor_model = ImpalaNet(observation_shape=(3, 64, 64), num_actions=A)
    actor_model.eval()
    learner_model = ImpalaNet(observation_shape=(3, 64, 64), num_actions=A)
    learner_model.load_state_dict(actor_model.state_dict())
    learner_model.eval()
    iasb = create_initial_agent_state_buffers(actor_model, num_buffers=2)

    # Run actor twice — fills slots 0 and 1.
    for idx in [0, 1]:
        free_q = _SingleShotQueue([idx, None])
        full_q = _SingleShotQueue([])
        act(
            actor_index=idx, free_queue=free_q, full_queue=full_q,
            model=actor_model, buffers=bufs, initial_agent_state_buffers=iasb,
            env_fn=lambda seed: (_FakeImageEnv(seed=seed, num_actions=A), seed),
            unroll_length=T, chw_transpose=True,
        )

    # Stack slots into a [T+1, B, ...] batch (mimics _get_batch).
    batch = {key: torch.stack(bufs[key], dim=1) for key in bufs}
    assert batch["frame"].shape == (T + 1, B, 3, 64, 64)
    assert batch["action"].shape == (T + 1, B)

    pre_params = [p.detach().clone() for p in learner_model.parameters()]

    opt = torch.optim.RMSprop(learner_model.parameters(), lr=1e-3)
    stats = impala_learn(
        actor_model=actor_model, learner_model=learner_model, batch=batch,
        initial_agent_state=(), optimizer=opt, scheduler=None,
        discounting=0.99, baseline_cost=0.5, entropy_cost=0.01,
        grad_norm_clipping=40.0,
    )

    # Learn produced finite losses
    for k in ("total_loss", "pg_loss", "baseline_loss", "entropy_loss"):
        assert np.isfinite(stats[k])

    # Learner params actually updated (not bit-identical to pre)
    post_params = list(learner_model.parameters())
    moved = any(
        (a - b).abs().sum().item() > 0 for a, b in zip(pre_params, post_params)
    )
    assert moved, "learner_model parameters did not change after learn()"

    # Actor model synced to match learner
    for (_, ap), (_, lp) in zip(
        actor_model.state_dict().items(), learner_model.state_dict().items()
    ):
        torch.testing.assert_close(ap, lp)


class _SeedRecordingFakeEnv:
    """Like _FakeImageEnv but records every seed passed to reset(). Used
    to verify the cfg.fixed_env_seed -> Environment chain end-to-end."""

    def __init__(self, num_actions=4):
        self.num_actions = num_actions
        self.t = 0
        self.action_space = type("S", (), {"n": num_actions})()
        self.seeds_seen: list = []

    def reset(self, *, seed=None):
        self.seeds_seen.append(seed)
        self.t = 0
        return np.zeros((64, 64, 3), dtype=np.uint8), {}

    def step(self, action):
        self.t += 1
        return (np.zeros((64, 64, 3), dtype=np.uint8), 0.0,
                self.t >= 3, False, {})

    def close(self):
        pass


def test_act_threads_fixed_env_seed_through_to_env_resets():
    """End-to-end wiring check: act(..., fixed_env_seed=K) must cause
    every env.reset() call inside the rollout to receive seed=K. Catches
    regressions in act's signature/positional args, Environment ctor
    keywords, and the train.py spawn site that wires them up.

    Distinct from test_fixed_seed_forces_same_seed_on_every_reset (which
    tests the Environment class in isolation) — this exercises the full
    chain that delivers cfg.fixed_env_seed at runtime."""
    T, A = 8, 4
    bufs = create_buffers(
        obs_shape=(3, 64, 64), num_actions=A, unroll_length=T, num_buffers=1
    )
    for key in bufs:
        for slot in bufs[key]:
            slot.zero_()
    model = ImpalaNet(observation_shape=(3, 64, 64), num_actions=A)
    model.eval()
    iasb = create_initial_agent_state_buffers(model, num_buffers=1)

    seed_envs: list = []
    def _env_fn(seed):
        env = _SeedRecordingFakeEnv(num_actions=A)
        seed_envs.append(env)
        return env, seed

    free_q = _SingleShotQueue([0, None])
    full_q = _SingleShotQueue([])

    act(
        actor_index=0,
        free_queue=free_q,
        full_queue=full_q,
        model=model,
        buffers=bufs,
        initial_agent_state_buffers=iasb,
        env_fn=_env_fn,
        unroll_length=T,
        chw_transpose=True,
        fixed_env_seed=42,  # ← the value under test
    )

    # The actor built one env. Episode length is 3, so an 8-step unroll
    # crosses at least 2 auto-resets. Plus the initial reset = 3+ calls,
    # all with seed=42.
    assert len(seed_envs) == 1
    env = seed_envs[0]
    assert len(env.seeds_seen) >= 3, (
        f"expected ≥3 reset calls (initial + auto-resets); "
        f"got {len(env.seeds_seen)}"
    )
    assert all(s == 42 for s in env.seeds_seen), (
        f"fixed_env_seed=42 should pass through to every reset; "
        f"got {env.seeds_seen}"
    )


def test_act_without_fixed_env_seed_uses_initial_seed_only():
    """The complement: if fixed_env_seed is None, the initial reset gets
    the initial_seed from env_fn, and subsequent auto-resets get None
    (the env picks its own RNG advance / random seed)."""
    T, A = 8, 4
    bufs = create_buffers(
        obs_shape=(3, 64, 64), num_actions=A, unroll_length=T, num_buffers=1
    )
    for key in bufs:
        for slot in bufs[key]:
            slot.zero_()
    model = ImpalaNet(observation_shape=(3, 64, 64), num_actions=A)
    model.eval()
    iasb = create_initial_agent_state_buffers(model, num_buffers=1)

    seed_envs: list = []
    def _env_fn(seed):
        env = _SeedRecordingFakeEnv(num_actions=A)
        seed_envs.append(env)
        return env, 999  # initial_seed handed back via env_fn

    free_q = _SingleShotQueue([0, None])
    full_q = _SingleShotQueue([])
    act(
        actor_index=0, free_queue=free_q, full_queue=full_q,
        model=model, buffers=bufs, initial_agent_state_buffers=iasb,
        env_fn=_env_fn,
        unroll_length=T, chw_transpose=True,
        fixed_env_seed=None,  # default
    )

    env = seed_envs[0]
    assert env.seeds_seen[0] == 999, "first reset should use initial_seed"
    assert all(s is None for s in env.seeds_seen[1:]), (
        f"auto-resets should get seed=None when no fixed_env_seed; "
        f"got {env.seeds_seen}"
    )


def test_pipeline_handles_episode_boundaries():
    """Actor should auto-reset on done; episode_return should reflect it."""
    T, A = 12, 4  # long enough to cross at least one episode (env len = 5)
    bufs = create_buffers(
        obs_shape=(3, 64, 64), num_actions=A, unroll_length=T, num_buffers=1
    )
    model = ImpalaNet(observation_shape=(3, 64, 64), num_actions=A)
    model.eval()
    iasb = create_initial_agent_state_buffers(model, num_buffers=1)

    free_q = _SingleShotQueue([0, None])
    full_q = _SingleShotQueue([])
    act(
        actor_index=0, free_queue=free_q, full_queue=full_q,
        model=model, buffers=bufs, initial_agent_state_buffers=iasb,
        env_fn=lambda seed: (_FakeImageEnv(seed=seed, num_actions=A), seed),
        unroll_length=T, chw_transpose=True,
    )
    # At least one done flag should appear in the rollout
    assert bufs["done"][0].any().item(), "actor never hit an episode end"
    # Episode step should reset somewhere mid-rollout (not monotonic)
    steps = bufs["episode_step"][0].tolist()
    assert min(steps) == 0
    assert max(steps) >= 1


def test_full_pipeline_lstm_actor_to_learn_step():
    """LSTM end-to-end (single-process): actor fills buffers + snapshots its
    recurrent state, the states stack into a non-empty initial_agent_state,
    and learn() consumes it — producing finite losses and updating the LSTM
    core. The recurrent analogue of test_full_pipeline_actor_to_learn_step."""
    T, A, B = 8, 4, 2
    bufs = create_buffers(
        obs_shape=(3, 64, 64), num_actions=A, unroll_length=T, num_buffers=2
    )
    actor_model = ImpalaNet((3, 64, 64), num_actions=A, use_lstm=True)
    actor_model.eval()
    learner_model = ImpalaNet((3, 64, 64), num_actions=A, use_lstm=True)
    learner_model.load_state_dict(actor_model.state_dict())
    learner_model.eval()
    iasb = create_initial_agent_state_buffers(actor_model, num_buffers=2)

    for idx in [0, 1]:
        free_q = _SingleShotQueue([idx, None])
        full_q = _SingleShotQueue([])
        act(
            actor_index=idx, free_queue=free_q, full_queue=full_q,
            model=actor_model, buffers=bufs, initial_agent_state_buffers=iasb,
            env_fn=lambda seed: (_FakeImageEnv(seed=seed, num_actions=A), seed),
            unroll_length=T, chw_transpose=True,
        )

    batch = {key: torch.stack(bufs[key], dim=1) for key in bufs}
    # Stack the per-slot snapshots along the batch dim (mimics _get_batch).
    initial_agent_state = tuple(
        torch.cat(ts, dim=1) for ts in zip(*[iasb[m] for m in [0, 1]])
    )
    assert len(initial_agent_state) == 2  # (h, c)
    # (num_layers, B, hidden); hidden == features_dim default 256
    assert initial_agent_state[0].shape == (1, B, 256)
    assert initial_agent_state[1].shape == (1, B, 256)

    pre_params = [p.detach().clone() for p in learner_model.parameters()]
    opt = torch.optim.RMSprop(learner_model.parameters(), lr=1e-3)
    stats = impala_learn(
        actor_model=actor_model, learner_model=learner_model, batch=batch,
        initial_agent_state=initial_agent_state, optimizer=opt, scheduler=None,
        discounting=0.99, baseline_cost=0.5, entropy_cost=0.01,
        grad_norm_clipping=40.0,
    )
    for k in ("total_loss", "pg_loss", "baseline_loss", "entropy_loss"):
        assert np.isfinite(stats[k])

    # The LSTM core specifically must have received a gradient update.
    core_moved = any(
        (a - b).abs().sum().item() > 0
        for (na, a), b in zip(learner_model.named_parameters(), pre_params)
        if na.startswith("core")
    )
    assert core_moved, "LSTM core params did not update after learn()"


def test_get_batch_stacks_recurrent_state_along_batch_dim():
    """train._get_batch must cat each slot's (num_layers, 1, hidden) snapshot
    into a (num_layers, B, hidden) initial_agent_state. Feedforward slots are
    empty tuples, so it must collapse to ()."""
    import queue as _queue

    from playtrain_trainers.impala.train import _get_batch

    T, A, B, L, H = 4, 3, 2, 1, 8

    # LSTM case: distinct per-slot states so we can verify the cat order.
    net = ImpalaNet((3, 16, 16), num_actions=A, features_dim=H, use_lstm=True)
    bufs = create_buffers((3, 16, 16), A, T, num_buffers=B)
    iasb = create_initial_agent_state_buffers(net, num_buffers=B)
    for m in range(B):
        iasb[m][0].fill_(float(m + 1))   # h
        iasb[m][1].fill_(float(-(m + 1)))  # c

    free_q, full_q = _queue.Queue(), _queue.Queue()
    for m in range(B):
        full_q.put(m)
    batch, state = _get_batch(free_q, full_q, bufs, iasb, B,
                              torch.device("cpu"), threading.Lock())
    assert len(state) == 2
    assert state[0].shape == (L, B, H) and state[1].shape == (L, B, H)
    # batch dim 1 holds slot 0 then slot 1, in full_queue order.
    assert torch.equal(state[0][:, 0], torch.full((L, H), 1.0))
    assert torch.equal(state[0][:, 1], torch.full((L, H), 2.0))
    assert torch.equal(state[1][:, 1], torch.full((L, H), -2.0))
    # slots returned to free queue for reuse
    assert sorted([free_q.get() for _ in range(B)]) == list(range(B))

    # Feedforward case: empty per-slot tuples -> initial_agent_state is ().
    ff_net = ImpalaNet((3, 16, 16), num_actions=A, features_dim=H)
    ff_iasb = create_initial_agent_state_buffers(ff_net, num_buffers=B)
    ff_full = _queue.Queue()
    for m in range(B):
        ff_full.put(m)
    _batch, ff_state = _get_batch(_queue.Queue(), ff_full, bufs, ff_iasb, B,
                                  torch.device("cpu"), threading.Lock())
    assert ff_state == ()
