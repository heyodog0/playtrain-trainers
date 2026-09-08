"""Recurrence-principle tests for the ImpalaNet LSTM core (use_lstm=True).

These assert the *defining properties* of a correct recurrent policy, not
just that an LSTM module is present. Each test pins one principle:

  1. State shape contract                 (initial_state / forward round-trip)
  2. Non-Markov: output depends on history (the whole point of an LSTM)
  3. Chunk-equivalence / state threading   (online-actor == replay-learner)
  4. Carried state is actually consumed    (guards "forward ignores core_state")
  5. Done resets the hidden state          (episode-boundary independence)
  6. Reset == fresh zero-state start       (the reset truly zeroes, not decays)
  7. BPTT connects timesteps; reset cuts it (gradient path across/through time)
  8. The LSTM core receives gradient        (it's in the optimized graph)
  9. Feedforward control: Markov, ignores done/state

Comparisons use policy_logits / baseline, which are deterministic functions of
(inputs, core_state) regardless of train/eval — only action sampling is
stochastic, and we never assert on actions here. Nets are in eval() so runs
are fully reproducible.

Small obs (3,16,16) and features_dim=32 keep the LSTM tiny and the suite fast.
"""
from __future__ import annotations

import torch

from playtrain_trainers.impala.net import ImpalaNet


OBS = (3, 16, 16)
A = 5
F = 32  # features_dim == LSTM hidden size


def _net(use_lstm: bool, seed: int = 0) -> ImpalaNet:
    torch.manual_seed(seed)
    net = ImpalaNet(OBS, num_actions=A, features_dim=F, use_lstm=use_lstm)
    net.eval()  # deterministic: argmax actions, no sampling RNG
    return net


def _inputs(frame: torch.Tensor, done: torch.Tensor) -> dict:
    T, B = frame.shape[:2]
    return dict(
        frame=frame,
        reward=torch.zeros(T, B),
        done=done,
        last_action=torch.zeros(T, B, dtype=torch.int64),
    )


def _rand_frames(T: int, B: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (T, B, *OBS), dtype=torch.uint8, generator=g)


def _no_done(T: int, B: int) -> torch.Tensor:
    return torch.zeros(T, B, dtype=torch.bool)


# 1 ─────────────────────────────────────────────────────────────────────────
def test_initial_state_shape_contract():
    net = _net(use_lstm=True)
    for B in (1, 4, 16):
        h, c = net.initial_state(batch_size=B)
        assert h.shape == (1, B, F)  # [num_layers, B, hidden]
        assert c.shape == (1, B, F)
        assert torch.count_nonzero(h) == 0 and torch.count_nonzero(c) == 0
    # forward returns a same-shaped (h, c) it can be re-fed
    out, state = net(_inputs(_rand_frames(3, 4, 1), _no_done(3, 4)),
                     net.initial_state(batch_size=4))
    assert len(state) == 2
    assert state[0].shape == (1, 4, F) and state[1].shape == (1, 4, F)
    # feedforward mode: empty state
    assert _net(use_lstm=False).initial_state(4) == ()


# 2 ─────────────────────────────────────────────────────────────────────────
def test_lstm_output_depends_on_history():
    """Two sequences sharing only the FINAL frame must give different logits
    at that step — an LSTM conditions on the past. Feedforward must not."""
    T, B = 4, 1
    frames_a = _rand_frames(T, B, seed=10)
    frames_b = _rand_frames(T, B, seed=20)
    frames_b[T - 1] = frames_a[T - 1]  # identical last frame, different history
    done = _no_done(T, B)
    init = lambda net: net.initial_state(batch_size=B)

    lstm = _net(use_lstm=True)
    la, _ = lstm(_inputs(frames_a, done), init(lstm))
    lb, _ = lstm(_inputs(frames_b, done), init(lstm))
    last_diff = (la["policy_logits"][-1] - lb["policy_logits"][-1]).abs().max()
    assert last_diff > 1e-5, "LSTM ignored history (Markov-like output)"

    ff = _net(use_lstm=False)
    fa, _ = ff(_inputs(frames_a, done), ())
    fb, _ = ff(_inputs(frames_b, done), ())
    assert torch.allclose(fa["policy_logits"][-1], fb["policy_logits"][-1]), \
        "feedforward output changed despite identical final frame"


# 3 ─────────────────────────────────────────────────────────────────────────
def test_chunk_equivalence_state_threading():
    """Running T steps in one call == running them in two chunks while
    threading core_state. This is the online-actor (1 step at a time) vs
    learner-replay (full unroll) consistency that V-trace relies on."""
    net = _net(use_lstm=True)
    T, B, k = 6, 2, 3
    frames = _rand_frames(T, B, seed=30)
    done = _no_done(T, B)  # no resets: pure continuity

    out_full, state_full = net(_inputs(frames, done), net.initial_state(B))

    out1, state1 = net(_inputs(frames[:k], done[:k]), net.initial_state(B))
    out2, state2 = net(_inputs(frames[k:], done[k:]), state1)

    for key in ("policy_logits", "baseline"):
        assert torch.allclose(out_full[key][:k], out1[key], atol=1e-5)
        assert torch.allclose(out_full[key][k:], out2[key], atol=1e-5)
    # the recurrent state after the split must match the monolithic run
    for s_full, s_split in zip(state_full, state2):
        assert torch.allclose(s_full, s_split, atol=1e-5)


# 4 ─────────────────────────────────────────────────────────────────────────
def test_carried_state_is_actually_consumed():
    """Feeding the previous segment's state must differ from feeding a zero
    state (no reset between). Guards against forward() ignoring core_state."""
    net = _net(use_lstm=True)
    T, B, k = 6, 1, 3
    frames = _rand_frames(T, B, seed=40)
    done = _no_done(T, B)

    _, state1 = net(_inputs(frames[:k], done[:k]), net.initial_state(B))
    out_carry, _ = net(_inputs(frames[k:], done[k:]), state1)
    out_zero, _ = net(_inputs(frames[k:], done[k:]), net.initial_state(B))

    diff = (out_carry["policy_logits"] - out_zero["policy_logits"]).abs().max()
    assert diff > 1e-5, "carried core_state had no effect — state not consumed"


# 5 ─────────────────────────────────────────────────────────────────────────
def test_done_resets_hidden_state():
    """done[k]=True must make every output at step >= k independent of all
    frames before k. THE recurrence-specific correctness property."""
    net = _net(use_lstm=True)
    T, B, k = 6, 1, 3
    shared_tail = _rand_frames(T, B, seed=50)

    frames_a = _rand_frames(T, B, seed=51)
    frames_b = _rand_frames(T, B, seed=52)
    frames_a[k:] = shared_tail[k:]          # identical from the reset onward
    frames_b[k:] = shared_tail[k:]

    done = _no_done(T, B)
    done[k] = True                          # episode boundary at step k

    oa, _ = net(_inputs(frames_a, done), net.initial_state(B))
    ob, _ = net(_inputs(frames_b, done), net.initial_state(B))

    for key in ("policy_logits", "baseline"):
        assert torch.allclose(oa[key][k:], ob[key][k:], atol=1e-5), \
            f"{key} after reset depended on pre-reset history"
    # sanity: BEFORE the reset the two DO differ (histories differ there)
    assert not torch.allclose(oa["policy_logits"][:k], ob["policy_logits"][:k])


# 6 ─────────────────────────────────────────────────────────────────────────
def test_reset_equals_fresh_zero_state_start():
    """Output at/after a done boundary must equal a fresh run on the post-reset
    subsequence started from zero state — confirms the reset truly zeroes the
    state rather than merely attenuating it."""
    net = _net(use_lstm=True)
    T, B, k = 6, 1, 2
    frames = _rand_frames(T, B, seed=60)
    done = _no_done(T, B)
    done[k] = True

    out_full, _ = net(_inputs(frames, done), net.initial_state(B))

    # fresh run on frames[k:], with its own done[0]=True and zero init state
    fresh_done = _no_done(T - k, B)
    fresh_done[0] = True
    out_fresh, _ = net(_inputs(frames[k:], fresh_done), net.initial_state(B))

    for key in ("policy_logits", "baseline"):
        assert torch.allclose(out_full[key][k:], out_fresh[key], atol=1e-5)


# 7 ─────────────────────────────────────────────────────────────────────────
def test_bptt_connects_timesteps_and_reset_cuts_it():
    """Perturbing the first frame changes a later-step output when no reset
    intervenes (temporal dependence path exists), but NOT when a done reset
    sits between them (the reset severs the path)."""
    net = _net(use_lstm=True)
    T, B = 4, 1
    frames = _rand_frames(T, B, seed=70)
    perturbed = frames.clone()
    perturbed[0] = _rand_frames(T, B, seed=71)[0]  # change only frame 0

    # (a) no reset: last-step output must react to frame[0]
    done = _no_done(T, B)
    base, _ = net(_inputs(frames, done), net.initial_state(B))
    pert, _ = net(_inputs(perturbed, done), net.initial_state(B))
    assert (base["policy_logits"][-1] - pert["policy_logits"][-1]).abs().max() > 1e-5

    # (b) reset at step 1: frame[0] is severed from steps >= 1
    done_reset = _no_done(T, B)
    done_reset[1] = True
    base_r, _ = net(_inputs(frames, done_reset), net.initial_state(B))
    pert_r, _ = net(_inputs(perturbed, done_reset), net.initial_state(B))
    assert torch.allclose(base_r["policy_logits"][1:], pert_r["policy_logits"][1:],
                          atol=1e-6), "reset failed to sever pre-reset influence"


# 8 ─────────────────────────────────────────────────────────────────────────
def test_lstm_core_receives_gradient():
    """The recurrent weights must be in the optimized graph: a loss on the
    outputs must produce nonzero gradients on self.core's parameters."""
    net = _net(use_lstm=True)
    net.train()
    T, B = 5, 2
    out, _ = net(_inputs(_rand_frames(T, B, seed=80), _no_done(T, B)),
                 net.initial_state(B))
    (out["policy_logits"].sum() + out["baseline"].sum()).backward()
    core_grads = [p.grad for n, p in net.named_parameters() if n.startswith("core")]
    assert core_grads, "no LSTM core parameters found"
    assert all(g is not None for g in core_grads)
    assert any(g.abs().sum() > 0 for g in core_grads)


# 9 ─────────────────────────────────────────────────────────────────────────
def test_feedforward_is_markov_ignores_done_and_state():
    """Control: in feedforward mode the done flags and any passed core_state
    have no effect — output is a pure function of the current frame."""
    net = _net(use_lstm=False)
    T, B = 4, 2
    frames = _rand_frames(T, B, seed=90)

    out_a, _ = net(_inputs(frames, _no_done(T, B)), ())
    done = _no_done(T, B)
    done[1] = done[2] = True
    out_b, _ = net(_inputs(frames, done), ())  # different done flags

    for key in ("policy_logits", "baseline"):
        assert torch.allclose(out_a[key], out_b[key]), \
            f"feedforward {key} reacted to done flags (not Markov)"


def test_segmentwise_lstm_matches_stepwise_reference():
    """The segment-wise unroll (split at any-done timesteps, fused LSTM per
    segment) must reproduce the original per-timestep masked loop exactly —
    stressed with DENSE random dones (every segmentation shape: t=0 done,
    consecutive dones, all-done rows, done at T-1)."""
    import torch
    from playtrain_trainers.impala.net import ImpalaNet

    torch.manual_seed(0)
    T, B, A = 24, 5, 4
    net = ImpalaNet((3, 64, 64), A, features_dim=32, use_lstm=True)
    net.eval()

    for trial, p in enumerate((0.0, 0.15, 0.5, 1.0)):
        g = torch.Generator().manual_seed(trial)
        inputs = {
            "frame": torch.randint(0, 256, (T, B, 3, 64, 64),
                                   dtype=torch.uint8, generator=g),
            "reward": torch.randn(T, B, generator=g),
            "done": torch.rand(T, B, generator=g) < p,
            "last_action": torch.randint(0, A, (T, B), dtype=torch.int64,
                                         generator=g),
        }
        state = tuple(torch.randn(1, B, 32, generator=g) for _ in range(2))

        with torch.no_grad():
            out, (h1, c1) = net(inputs, tuple(s.clone() for s in state))

        # Reference: the original per-timestep masked loop.
        with torch.no_grad():
            x = torch.flatten(inputs["frame"], 0, 1)
            feats = net.encoder(x).view(T, B, -1)
            notdone = (~inputs["done"]).float()
            ref_state = tuple(s.clone() for s in state)
            ref_outs = []
            for inp, nd in zip(feats.unbind(), notdone.unbind()):
                nd = nd.view(1, -1, 1)
                ref_state = tuple(nd * s for s in ref_state)
                o, ref_state = net.core(inp.unsqueeze(0), ref_state)
                ref_outs.append(o)
            ref_core = torch.flatten(torch.cat(ref_outs), 0, 1)
            ref_logits = net.policy(ref_core).view(T, B, A)

        torch.testing.assert_close(out["policy_logits"], ref_logits,
                                   msg=f"p={p}: logits diverge")
        torch.testing.assert_close(h1, ref_state[0], msg=f"p={p}: h diverges")
        torch.testing.assert_close(c1, ref_state[1], msg=f"p={p}: c diverges")
