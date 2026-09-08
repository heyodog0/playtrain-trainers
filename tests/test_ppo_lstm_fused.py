"""The fused LSTM unroll must be EXACTLY equivalent to the per-step reference.

get_states replaces the monobeast per-step Python loop with a fused unroll (one
cuDNN call over a no-boundary window, else one fused call per reset-free segment)
for speed. Correctness is non-negotiable: these tests pin the fused path against
``_unroll_steps`` (the retained per-step oracle) over random done patterns —
no resets, a reset at t=0, multiple async per-env resets — at the data level, so
a future edit to the fast path can't silently diverge from the recurrence the
recurrence-principle tests (test_ppo_lstm_net.py) assert.
"""
from __future__ import annotations

import torch

from playtrain_trainers.policy import ActorCritic


OBS = (3, 16, 16)
A = 5
F = 32


def _net(seed: int = 0) -> ActorCritic:
    torch.manual_seed(seed)
    net = ActorCritic(n_actions=A, in_channels=OBS[0], features_dim=F,
                      input_hw=OBS[1], use_lstm=True)
    net.eval()
    return net


def _state(net, B):
    s = net.initial_state(B)
    return (s[0].clone(), s[1].clone())


def _rand_frames(T, B, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (T, B, *OBS), dtype=torch.uint8, generator=g)


def test_fast_path_equals_perstep_when_no_resets():
    """No episode boundary -> the single fused nn.LSTM call over the whole
    sequence must equal the per-step loop."""
    net = _net()
    T, B = 8, 2
    z = torch.randn(T, B, net.lstm.input_size)
    done = torch.zeros(T, B)
    ref, ref_s = net._unroll_steps(z, _state(net, B), done)
    fast, fast_s = net.lstm(z, _state(net, B))
    assert torch.allclose(ref, fast, atol=1e-5)
    assert torch.allclose(ref_s[0], fast_s[0], atol=1e-5)
    assert torch.allclose(ref_s[1], fast_s[1], atol=1e-5)


def test_segmented_equals_perstep_with_async_resets():
    """Per-env resets at different timesteps (incl. t=0) — the segmented fused
    unroll must match the per-step oracle on outputs AND final state."""
    net = _net()
    T, B = 12, 3
    torch.manual_seed(1)
    z = torch.randn(T, B, net.lstm.input_size)
    done = torch.zeros(T, B)
    done[0, 1] = 1            # reset at the very first step (env 1)
    done[3, 2] = 1
    done[5, 0] = 1
    done[5, 2] = 1            # two envs reset same step
    done[9, 1] = 1
    done[11, 0] = 1           # reset on the last step

    ref, ref_s = net._unroll_steps(z, _state(net, B), done)
    reset_cpu = (done != 0).numpy()
    fud, fud_s = net._unroll_segmented(z, _state(net, B), reset_cpu)
    assert torch.allclose(ref, fud, atol=1e-5)
    assert torch.allclose(ref_s[0], fud_s[0], atol=1e-5)
    assert torch.allclose(ref_s[1], fud_s[1], atol=1e-5)


def test_segmented_equals_perstep_every_step_resets():
    """Pathological: a reset on every step degrades segments to length 1 — still
    must equal the per-step loop (each segment is a single fused 1-step call)."""
    net = _net()
    T, B = 6, 2
    torch.manual_seed(2)
    z = torch.randn(T, B, net.lstm.input_size)
    done = torch.ones(T, B)
    ref, ref_s = net._unroll_steps(z, _state(net, B), done)
    fud, fud_s = net._unroll_segmented(z, _state(net, B), (done != 0).numpy())
    assert torch.allclose(ref, fud, atol=1e-5)
    assert torch.allclose(ref_s[0], fud_s[0], atol=1e-5)


def test_get_states_dispatch_matches_reference():
    """End-to-end: get_states (which picks fast/segmented internally) must equal
    encoding + the per-step oracle, with boundaries present."""
    net = _net()
    T, B = 10, 3
    frames = _rand_frames(T, B, seed=5)
    done = torch.zeros(T, B)
    done[4, 0] = 1
    done[7, 2] = 1
    obs = frames.reshape(T * B, *OBS)

    z_fused, s_fused = net.get_states(obs, _state(net, B), done.reshape(T * B))

    zc = net.encoder(obs).view(T, B, net.lstm.input_size)
    z_ref, s_ref = net._unroll_steps(zc, _state(net, B), done)
    z_ref = torch.flatten(z_ref, 0, 1)

    assert torch.allclose(z_fused, z_ref, atol=1e-5)
    assert torch.allclose(s_fused[0], s_ref[0], atol=1e-5)
    assert torch.allclose(s_fused[1], s_ref[1], atol=1e-5)


def test_get_states_dispatch_matches_reference_no_resets():
    """Same, but the no-reset fast path branch."""
    net = _net()
    T, B = 9, 2
    frames = _rand_frames(T, B, seed=6)
    done = torch.zeros(T, B)
    obs = frames.reshape(T * B, *OBS)

    z_fused, s_fused = net.get_states(obs, _state(net, B), done.reshape(T * B))
    zc = net.encoder(obs).view(T, B, net.lstm.input_size)
    z_ref, s_ref = net._unroll_steps(zc, _state(net, B), done)
    z_ref = torch.flatten(z_ref, 0, 1)
    assert torch.allclose(z_fused, z_ref, atol=1e-5)
    assert torch.allclose(s_fused[0], s_ref[0], atol=1e-5)
