"""Actor-loop ↔ learner alignment + recurrent-state threading regression tests.

`learn()` is a port of monobeast.learn: it shifts `batch[1:]` against
`learner_outputs[:-1]` before V-trace. That shift assumes a specific slot
convention — the action/policy_logits stored at slot k+1 were produced from
the FRAME at slot k (monobeast's "infer-then-step, store pre-step output").

The defining correctness property, for BOTH inference modes and BOTH feedforward
and LSTM nets, is replay-equivalence:

    net(frame[k], state entering frame[k]) == stored_policy_logits[k+1]

i.e. when the learner re-runs the net over the buffered frames starting from the
snapshotted initial recurrent state, it reproduces the behavior logits the actor
stored (after the [1:]/[:-1] shift). For an LSTM this only holds if the recurrent
state was threaded correctly actor→buffer→learner — so this doubles as the
end-to-end validation of the (h,c) round-trip through the (stateless) central
inference server.

History: `act_central` previously did step-then-infer, storing a one-step-
shifted action/policy_logits — this file's central test caught that
misalignment (it failed at max|diff|~0.01) and now guards the fix.
"""
from __future__ import annotations

import queue
import threading
from types import SimpleNamespace

import numpy as np
import torch

from playtrain_trainers.impala.actor import act
from playtrain_trainers.impala.buffers import (
    create_buffers,
    create_initial_agent_state_buffers,
)
from playtrain_trainers.impala.central import (
    InferenceServer,
    act_central,
    create_channel,
)
from playtrain_trainers.impala.net import ImpalaNet


OBS_HWC = (16, 16, 3)
OBS_CHW = (3, 16, 16)
A = 5
F = 32  # features_dim == LSTM hidden size
T = 4


class _StubEnv:
    """Deterministic env: action-independent transitions, a DISTINCT frame per
    step (seeded by an internal counter). Optionally terminates once at
    `terminate_at` to exercise the done-reset path. The distinct-frames
    property is what makes a one-step misalignment observable."""

    def __init__(self, terminate_at: int | None = None):
        self.action_space = SimpleNamespace(n=A)
        self.c = 0
        self._terminate_at = terminate_at
        self._fired = False

    def _obs(self) -> np.ndarray:
        g = torch.Generator().manual_seed(1000 + self.c)
        return torch.randint(0, 256, OBS_HWC, dtype=torch.uint8,
                             generator=g).numpy()

    def reset(self, seed=None):
        self.c = 0
        return self._obs(), {}

    def step(self, action):
        self.c += 1
        terminated = (self._terminate_at is not None
                      and not self._fired
                      and self.c == self._terminate_at)
        if terminated:
            self._fired = True
        return self._obs(), 0.0, terminated, False, {}

    def close(self):
        pass


def _env_fn_factory(terminate_at=None):
    def _env_fn(actor_index: int):
        return _StubEnv(terminate_at=terminate_at), 0
    return _env_fn


def _net(use_lstm: bool) -> ImpalaNet:
    torch.manual_seed(0)
    net = ImpalaNet(OBS_CHW, num_actions=A, features_dim=F, use_lstm=use_lstm)
    # Default policy head uses orthogonal gain 0.01 -> near-zero logits for any
    # input on an untrained net, which would mask a one-step misalignment.
    # Re-init at gain 1.0 so logits genuinely depend on (frame, state).
    torch.nn.init.orthogonal_(net.policy.weight, gain=1.0)
    torch.nn.init.normal_(net.policy.bias, std=0.1)
    net.eval()  # deterministic logits; argmax actions
    return net


def _recompute_logits(net, buffers, index, initial_state) -> torch.Tensor:
    """Re-run the net over the buffered frames from the snapshotted state,
    using the buffered done flags (so the LSTM reset matches the actor)."""
    frames = buffers["frame"][index]              # (T+1, C, H, W) uint8
    done = buffers["done"][index]                 # (T+1,) bool
    inp = dict(
        frame=frames.unsqueeze(1),                # (T+1, 1, C, H, W)
        reward=torch.zeros(T + 1, 1),
        done=done.unsqueeze(1),                   # (T+1, 1)
        last_action=torch.zeros(T + 1, 1, dtype=torch.int64),
    )
    with torch.no_grad():
        out, _ = net(inp, initial_state)
    return out["policy_logits"][:, 0, :]          # (T+1, A)


def _assert_replay_equivalence(net, buffers, index, initial_state, label):
    recomputed = _recompute_logits(net, buffers, index, initial_state)
    stored = buffers["policy_logits"][index]
    # learn()'s shift needs: net(frame[k]) == stored[k+1]  for k=0..T-1
    assert torch.allclose(recomputed[:-1], stored[1:], atol=1e-4), (
        f"[{label}] replay/alignment broken: re-running the net over the "
        f"buffered frames from the snapshotted state does not reproduce the "
        f"stored behavior logits[1:]. learn()'s shift would compare mismatched "
        f"(frame, state).\nmax|diff| = "
        f"{(recomputed[:-1] - stored[1:]).abs().max().item():.4f}"
    )


# ── shared_cpu (act) ─────────────────────────────────────────────────────────
def test_act_feedforward_alignment():
    net = _net(use_lstm=False)
    buffers = create_buffers(OBS_CHW, A, T, num_buffers=1)
    iasb = create_initial_agent_state_buffers(net, 1)
    fq, full = queue.Queue(), queue.Queue()
    fq.put(0); fq.put(None)
    act(0, fq, full, net, buffers, iasb, _env_fn_factory(), T,
        chw_transpose=True, fixed_env_seed=None)
    assert full.get() == 0
    _assert_replay_equivalence(net, buffers, 0, (), "act/ff")


def test_act_lstm_replay_equivalence():
    net = _net(use_lstm=True)
    buffers = create_buffers(OBS_CHW, A, T, num_buffers=1)
    iasb = create_initial_agent_state_buffers(net, 1)
    fq, full = queue.Queue(), queue.Queue()
    fq.put(0); fq.put(None)
    act(0, fq, full, net, buffers, iasb, _env_fn_factory(), T,
        chw_transpose=True, fixed_env_seed=None)
    assert full.get() == 0
    # learner replays from the snapshotted initial state (iasb slot 0)
    _assert_replay_equivalence(net, buffers, 0, tuple(iasb[0]), "act/lstm")


# ── central_gpu (act_central) ────────────────────────────────────────────────
def _run_central(net, buffers, iasb, env_fn, use_lstm):
    channel = create_channel(1, OBS_CHW, A, use_lstm=use_lstm, features_dim=F)
    server = InferenceServer(
        net, channel, device=torch.device("cpu"),
        obs_shape=OBS_CHW, num_actions=A, stochastic_actions=False,
    )
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    try:
        init_state = net.initial_state(batch_size=1)
        fq, full = queue.Queue(), queue.Queue()
        fq.put(0); fq.put(None)
        act_central(0, fq, full, channel, buffers, iasb, init_state, env_fn, T,
                    chw_transpose=True, fixed_env_seed=None)
        assert full.get() == 0
    finally:
        channel.should_stop.fill_(1)
        th.join(timeout=5)


def test_act_central_feedforward_alignment():
    """Guards the realignment fix: act_central must satisfy the SAME shift
    precondition act() does (this failed before the request-then-step fix)."""
    net = _net(use_lstm=False)
    buffers = create_buffers(OBS_CHW, A, T, num_buffers=1)
    iasb = create_initial_agent_state_buffers(net, 1)
    _run_central(net, buffers, iasb, _env_fn_factory(), use_lstm=False)
    _assert_replay_equivalence(net, buffers, 0, (), "act_central/ff")


def test_act_central_lstm_replay_equivalence():
    """The decisive (h,c)-through-the-channel test: the learner can only
    reproduce the stored logits if the recurrent state threaded correctly
    actor → stateless server → buffer → learner."""
    net = _net(use_lstm=True)
    buffers = create_buffers(OBS_CHW, A, T, num_buffers=1)
    iasb = create_initial_agent_state_buffers(net, 1)
    _run_central(net, buffers, iasb, _env_fn_factory(), use_lstm=True)
    _assert_replay_equivalence(net, buffers, 0, tuple(iasb[0]), "act_central/lstm")


def test_act_central_lstm_replay_equivalence_with_done_reset():
    """Same, but the env terminates mid-rollout. The done-reset must thread
    consistently: server zeros the incoming state on done, and the learner
    (reading the buffered done) zeros it at the same slot — so replay still
    reproduces the stored logits."""
    net = _net(use_lstm=True)
    buffers = create_buffers(OBS_CHW, A, T, num_buffers=1)
    iasb = create_initial_agent_state_buffers(net, 1)
    _run_central(net, buffers, iasb, _env_fn_factory(terminate_at=2), use_lstm=True)
    assert buffers["done"][0].any().item(), "env never terminated; test moot"
    _assert_replay_equivalence(net, buffers, 0, tuple(iasb[0]),
                               "act_central/lstm+done")
