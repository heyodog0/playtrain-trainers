"""Tests for playtrain_trainers.impala.net (the ImpalaNet agent wrapper)."""
from __future__ import annotations

import torch

from playtrain_trainers.impala.net import ImpalaNet


def _make_inputs(T=3, B=2, C=3, H=64, W=64, A=7):
    return dict(
        frame=torch.randint(0, 256, (T, B, C, H, W), dtype=torch.uint8),
        reward=torch.randn(T, B),
        done=torch.zeros(T, B, dtype=torch.bool),
        last_action=torch.randint(0, A, (T, B), dtype=torch.int64),
    )


def test_forward_shapes_train_mode():
    net = ImpalaNet(observation_shape=(3, 64, 64), num_actions=7)
    net.train()
    inputs = _make_inputs(T=3, B=2, A=7)
    out, state = net(inputs)
    assert out["policy_logits"].shape == (3, 2, 7)
    assert out["baseline"].shape == (3, 2)
    assert out["action"].shape == (3, 2)
    assert out["action"].dtype == torch.int64
    assert state == ()  # feedforward => no recurrent state


def test_eval_mode_uses_argmax():
    net = ImpalaNet(observation_shape=(3, 64, 64), num_actions=4)
    net.eval()
    inputs = _make_inputs(T=2, B=3, A=4)
    out, _ = net(inputs)
    # argmax over policy_logits should equal action
    flat_logits = out["policy_logits"].view(-1, 4)
    flat_actions = out["action"].view(-1)
    assert torch.equal(flat_actions, flat_logits.argmax(-1))


def test_initial_state_is_empty_tuple():
    net = ImpalaNet(observation_shape=(3, 64, 64), num_actions=5)
    assert net.initial_state(batch_size=1) == ()
    assert net.initial_state(batch_size=16) == ()


def test_lstm_mode_constructs_and_runs():
    # LSTM core is now implemented (see test_impala_net_lstm.py for the
    # recurrence-principle tests). Here we only assert it constructs and the
    # feedforward I/O contract still holds with a non-empty core_state.
    net = ImpalaNet(observation_shape=(3, 64, 64), num_actions=5, use_lstm=True)
    net.train()
    inputs = _make_inputs(T=3, B=2, A=5)
    out, state = net(inputs, net.initial_state(batch_size=2))
    assert out["policy_logits"].shape == (3, 2, 5)
    assert out["baseline"].shape == (3, 2)
    assert out["action"].shape == (3, 2)
    assert len(state) == 2  # (h, c)


def test_grad_flows_through_policy_and_baseline():
    net = ImpalaNet(observation_shape=(3, 32, 32), num_actions=3)
    net.train()
    inputs = _make_inputs(T=2, B=1, C=3, H=32, W=32, A=3)
    out, _ = net(inputs)
    (out["policy_logits"].sum() + out["baseline"].sum()).backward()
    grads = [p.grad for p in net.parameters() if p.requires_grad]
    assert all(g is not None for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)
