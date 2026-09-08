"""Recurrence-principle tests for the PPO ActorCritic LSTM core (use_lstm=True).

The PPO analogue of test_impala_net_lstm.py. These assert the *defining
properties* of a correct recurrent actor-critic, not merely that an LSTM module
is present. Each test pins one principle:

  1. State shape contract                  (initial_state / get_states round-trip)
  2. Non-Markov: output depends on history  (the whole point of an LSTM)
  3. Chunk-equivalence / state threading    (online-actor == replay-learner)
  4. Carried state is actually consumed     (guards "get_states ignores state")
  5. Done resets the hidden state           (episode-boundary independence)
  6. Reset == fresh zero-state start        (the reset truly zeroes, not decays)
  7. BPTT connects timesteps; reset cuts it (temporal path across/through time)
  8. The LSTM core receives gradient        (it's in the optimized graph)
  9. Feedforward control: Markov, ignores done/state

PPO-specific (the parts the IMPALA suite doesn't cover, because PPO's update
re-scores stored actions and its GAE/critic must thread state too):

 10. Value head — not just policy — depends on history (GAE needs this)
 11. get_action_and_value scores a *given* action (the PPO update path)
 12. get_value == critic(get_states features)
 13. FF new-API (get_action_and_value) is consistent with the old forward()
 14. Rollout (act_recurrent, 1 step at a time) == update replay
     (get_action_and_value over the full sequence) for the SAME actions —
     THE property PPO's importance ratio depends on, across an episode reset
 15. act_recurrent threads state and has rollout-shaped (B,) outputs
 16. forward()/act() refuse to run in LSTM mode (no silent stateless misuse)

Comparisons use get_states features / get_value / log_prob-of-a-fixed-action,
all deterministic functions of (obs, lstm_state, done); we never assert on
freshly-sampled actions. Nets are in eval() so the encoder is deterministic.

Small obs (3,16,16) and features_dim=32 keep the LSTM tiny and the suite fast.
"""
from __future__ import annotations

import pytest
import torch

from playtrain_trainers.policy import ActorCritic


OBS = (3, 16, 16)  # (C, H, W)
A = 5
F = 32  # features_dim == LSTM hidden size


def _net(use_lstm: bool, seed: int = 0) -> ActorCritic:
    torch.manual_seed(seed)
    net = ActorCritic(
        n_actions=A, in_channels=OBS[0], features_dim=F,
        input_hw=OBS[1], use_lstm=use_lstm,
    )
    net.eval()  # deterministic encoder; argmax/sample RNG never asserted on
    return net


def _rand_frames(T: int, B: int, seed: int) -> torch.Tensor:
    """[T, B, C, H, W] uint8."""
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (T, B, *OBS), dtype=torch.uint8, generator=g)


def _no_done(T: int, B: int) -> torch.Tensor:
    return torch.zeros(T, B, dtype=torch.bool)


def _flat(frames: torch.Tensor, done: torch.Tensor):
    """[T,B,...] -> the (obs[T*B,C,H,W], done[T*B]) the trainer feeds get_states.
    reshape is timestep-major / env-minor, matching get_states' internal
    view(-1, B, ...) recovery."""
    T, B = frames.shape[:2]
    return frames.reshape(T * B, *OBS), done.reshape(T * B)


def _feats(net: ActorCritic, frames: torch.Tensor, done: torch.Tensor, state):
    """Run get_states over a [T,B] sequence; return (features[T,B,F], new_state)."""
    T, B = frames.shape[:2]
    obs, d = _flat(frames, done)
    z, new_state = net.get_states(obs, state, d)
    return z.view(T, B, F), new_state


def _values(net: ActorCritic, frames: torch.Tensor, done: torch.Tensor, state):
    """Run get_value over a [T,B] sequence; return values[T,B]."""
    T, B = frames.shape[:2]
    obs, d = _flat(frames, done)
    return net.get_value(obs, state, d).view(T, B)


# 1 ─────────────────────────────────────────────────────────────────────────
def test_initial_state_shape_contract():
    net = _net(use_lstm=True)
    for B in (1, 4, 16):
        h, c = net.initial_state(batch_size=B)
        assert h.shape == (1, B, F)  # [num_layers, B, hidden]
        assert c.shape == (1, B, F)
        assert torch.count_nonzero(h) == 0 and torch.count_nonzero(c) == 0
    # get_states returns a same-shaped (h, c) it can be re-fed
    _, state = _feats(net, _rand_frames(3, 4, 1), _no_done(3, 4),
                      net.initial_state(batch_size=4))
    assert len(state) == 2
    assert state[0].shape == (1, 4, F) and state[1].shape == (1, 4, F)
    # feedforward mode: empty state
    assert _net(use_lstm=False).initial_state(4) == ()


# 2 ─────────────────────────────────────────────────────────────────────────
def test_lstm_output_depends_on_history():
    """Two sequences sharing only the FINAL frame must give different features
    at that step — an LSTM conditions on the past. Feedforward must not."""
    T, B = 4, 1
    frames_a = _rand_frames(T, B, seed=10)
    frames_b = _rand_frames(T, B, seed=20)
    frames_b[T - 1] = frames_a[T - 1]  # identical last frame, different history
    done = _no_done(T, B)

    lstm = _net(use_lstm=True)
    fa, _ = _feats(lstm, frames_a, done, lstm.initial_state(B))
    fb, _ = _feats(lstm, frames_b, done, lstm.initial_state(B))
    assert (fa[-1] - fb[-1]).abs().max() > 1e-5, "LSTM ignored history (Markov-like)"

    ff = _net(use_lstm=False)
    ga, _ = _feats(ff, frames_a, done, ())
    gb, _ = _feats(ff, frames_b, done, ())
    assert torch.allclose(ga[-1], gb[-1]), \
        "feedforward features changed despite identical final frame"


# 3 ─────────────────────────────────────────────────────────────────────────
def test_chunk_equivalence_state_threading():
    """Running T steps in one call == running them in two chunks while threading
    lstm_state. This is the online-actor (1 step) vs learner-replay (full unroll)
    consistency the PPO ratio relies on."""
    net = _net(use_lstm=True)
    T, B, k = 6, 2, 3
    frames = _rand_frames(T, B, seed=30)
    done = _no_done(T, B)  # no resets: pure continuity

    full, state_full = _feats(net, frames, done, net.initial_state(B))

    f1, state1 = _feats(net, frames[:k], done[:k], net.initial_state(B))
    f2, state2 = _feats(net, frames[k:], done[k:], state1)

    assert torch.allclose(full[:k], f1, atol=1e-5)
    assert torch.allclose(full[k:], f2, atol=1e-5)
    # the recurrent state after the split must match the monolithic run
    for s_full, s_split in zip(state_full, state2):
        assert torch.allclose(s_full, s_split, atol=1e-5)


# 4 ─────────────────────────────────────────────────────────────────────────
def test_carried_state_is_actually_consumed():
    """Feeding the previous segment's state must differ from feeding a zero state
    (no reset between). Guards against get_states ignoring lstm_state."""
    net = _net(use_lstm=True)
    T, B, k = 6, 1, 3
    frames = _rand_frames(T, B, seed=40)
    done = _no_done(T, B)

    _, state1 = _feats(net, frames[:k], done[:k], net.initial_state(B))
    carry, _ = _feats(net, frames[k:], done[k:], state1)
    zero, _ = _feats(net, frames[k:], done[k:], net.initial_state(B))

    assert (carry - zero).abs().max() > 1e-5, \
        "carried lstm_state had no effect — state not consumed"


# 5 ─────────────────────────────────────────────────────────────────────────
def test_done_resets_hidden_state():
    """done[k]=True must make every output at step >= k independent of all frames
    before k. THE recurrence-specific correctness property."""
    net = _net(use_lstm=True)
    T, B, k = 6, 1, 3
    shared_tail = _rand_frames(T, B, seed=50)

    frames_a = _rand_frames(T, B, seed=51)
    frames_b = _rand_frames(T, B, seed=52)
    frames_a[k:] = shared_tail[k:]          # identical from the reset onward
    frames_b[k:] = shared_tail[k:]

    done = _no_done(T, B)
    done[k] = True                          # episode boundary at step k

    fa, _ = _feats(net, frames_a, done, net.initial_state(B))
    fb, _ = _feats(net, frames_b, done, net.initial_state(B))

    assert torch.allclose(fa[k:], fb[k:], atol=1e-5), \
        "features after reset depended on pre-reset history"
    # sanity: BEFORE the reset the two DO differ (histories differ there)
    assert not torch.allclose(fa[:k], fb[:k])


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

    full, _ = _feats(net, frames, done, net.initial_state(B))

    # fresh run on frames[k:], with its own done[0]=True and zero init state
    fresh_done = _no_done(T - k, B)
    fresh_done[0] = True
    fresh, _ = _feats(net, frames[k:], fresh_done, net.initial_state(B))

    assert torch.allclose(full[k:], fresh, atol=1e-5)


# 7 ─────────────────────────────────────────────────────────────────────────
def test_bptt_connects_timesteps_and_reset_cuts_it():
    """Perturbing the first frame changes a later-step output when no reset
    intervenes (temporal dependence path exists), but NOT when a done reset sits
    between them (the reset severs the path)."""
    net = _net(use_lstm=True)
    T, B = 4, 1
    frames = _rand_frames(T, B, seed=70)
    perturbed = frames.clone()
    perturbed[0] = _rand_frames(T, B, seed=71)[0]  # change only frame 0

    # (a) no reset: last-step output must react to frame[0]
    done = _no_done(T, B)
    base, _ = _feats(net, frames, done, net.initial_state(B))
    pert, _ = _feats(net, perturbed, done, net.initial_state(B))
    assert (base[-1] - pert[-1]).abs().max() > 1e-5

    # (b) reset at step 1: frame[0] is severed from steps >= 1
    done_reset = _no_done(T, B)
    done_reset[1] = True
    base_r, _ = _feats(net, frames, done_reset, net.initial_state(B))
    pert_r, _ = _feats(net, perturbed, done_reset, net.initial_state(B))
    assert torch.allclose(base_r[1:], pert_r[1:], atol=1e-6), \
        "reset failed to sever pre-reset influence"


# 8 ─────────────────────────────────────────────────────────────────────────
def test_lstm_core_receives_gradient():
    """The recurrent weights must be in the optimized graph: a loss on the
    outputs must produce nonzero gradients on self.lstm's parameters."""
    net = _net(use_lstm=True)
    net.train()
    T, B = 5, 2
    obs, d = _flat(_rand_frames(T, B, seed=80), _no_done(T, B))
    _, logp, ent, val, _ = net.get_action_and_value(obs, net.initial_state(B), d)
    (logp.sum() + ent.sum() + val.sum()).backward()
    lstm_grads = [p.grad for n, p in net.named_parameters() if n.startswith("lstm")]
    assert lstm_grads, "no LSTM parameters found"
    assert all(g is not None for g in lstm_grads)
    assert any(g.abs().sum() > 0 for g in lstm_grads)


# 9 ─────────────────────────────────────────────────────────────────────────
def test_feedforward_is_markov_ignores_done_and_state():
    """Control: in feedforward mode the done flags have no effect — features are
    a pure function of the current frame."""
    net = _net(use_lstm=False)
    T, B = 4, 2
    frames = _rand_frames(T, B, seed=90)

    fa, _ = _feats(net, frames, _no_done(T, B), ())
    done = _no_done(T, B)
    done[1] = done[2] = True
    fb, _ = _feats(net, frames, done, ())  # different done flags

    assert torch.allclose(fa, fb), "feedforward features reacted to done (not Markov)"


# 10 ────────────────────────────────────────────────────────────────────────
def test_value_head_also_depends_on_history():
    """Not just the policy — the *critic* must be recurrent too, else GAE/returns
    are computed against a Markov value and the recurrence buys nothing for the
    advantage estimate."""
    T, B = 4, 1
    frames_a = _rand_frames(T, B, seed=110)
    frames_b = _rand_frames(T, B, seed=120)
    frames_b[T - 1] = frames_a[T - 1]  # identical final frame, different history
    done = _no_done(T, B)

    net = _net(use_lstm=True)
    va = _values(net, frames_a, done, net.initial_state(B))
    vb = _values(net, frames_b, done, net.initial_state(B))
    assert (va[-1] - vb[-1]).abs().max() > 1e-5, "value head ignored history"


# 11 ────────────────────────────────────────────────────────────────────────
def test_get_action_and_value_scores_given_action():
    """The PPO update path: passing a stored action must score *that* action
    (not resample), and entropy/value/log_prob must be self-consistent and
    correctly shaped [T*B]."""
    net = _net(use_lstm=True)
    T, B = 4, 2
    obs, d = _flat(_rand_frames(T, B, seed=130), _no_done(T, B))
    state = net.initial_state(B)

    given = torch.randint(0, A, (T * B,))
    act, logp, ent, val, new_state = net.get_action_and_value(obs, state, d, given)

    assert torch.equal(act, given), "passed action was not the one returned/scored"
    assert logp.shape == (T * B,) and ent.shape == (T * B,) and val.shape == (T * B,)
    # log_prob of the given action matches a hand-built categorical on the logits
    z, _ = net.get_states(obs, state, d)
    dist = torch.distributions.Categorical(logits=net.actor(z))
    assert torch.allclose(logp, dist.log_prob(given), atol=1e-6)
    assert torch.allclose(ent, dist.entropy(), atol=1e-6)
    assert new_state[0].shape == (1, B, F)


# 12 ────────────────────────────────────────────────────────────────────────
def test_get_value_matches_get_states_critic():
    """get_value is exactly the critic head applied to get_states features."""
    net = _net(use_lstm=True)
    T, B = 5, 2
    obs, d = _flat(_rand_frames(T, B, seed=140), _no_done(T, B))
    state = net.initial_state(B)

    z, _ = net.get_states(obs, state, d)
    expected = net.critic(z).squeeze(-1)
    got = net.get_value(obs, state, d)
    assert torch.allclose(got, expected, atol=1e-6)


# 13 ────────────────────────────────────────────────────────────────────────
def test_ff_new_api_matches_old_forward():
    """In feedforward mode the new state-threaded API must agree with the
    untouched forward()/act() path — proves the LSTM refactor introduced no
    regression on the default (use_lstm=False) configs."""
    net = _net(use_lstm=False)
    N = 8
    obs = _rand_frames(N, 1, seed=150).reshape(N, *OBS)
    done = torch.zeros(N)

    dist, value = net.forward(obs)
    given = torch.randint(0, A, (N,))
    act, logp, ent, val, state = net.get_action_and_value(obs, (), done, given)

    assert state == ()
    assert torch.allclose(val, value, atol=1e-6)
    assert torch.allclose(logp, dist.log_prob(given), atol=1e-6)
    assert torch.allclose(ent, dist.entropy(), atol=1e-6)
    assert torch.allclose(net.get_value(obs, (), done), value, atol=1e-6)


# 14 ────────────────────────────────────────────────────────────────────────
def test_rollout_equals_update_replay():
    """THE PPO-critical property. Collect a rollout one step at a time with
    act_recurrent (threading lstm_state and feeding per-step done, with a reset
    mid-episode), recording each step's action / log_prob / value. Then replay
    the whole sequence in a single get_action_and_value call (the update path)
    scoring those SAME actions from the stored initial state. log_prob and value
    must match step-for-step — otherwise the importance ratio new/old is
    computed against a misaligned hidden state and the PG is corrupted (cf. the
    IMPALA V-trace off-by-one in HANDOFF.md)."""
    net = _net(use_lstm=True)
    T, B = 6, 3
    frames = _rand_frames(T, B, seed=160)
    done = _no_done(T, B)
    done[2] = torch.tensor([True, False, False])   # env 0 resets at step 2
    done[4] = torch.tensor([False, True, False])   # env 1 resets at step 4

    # --- rollout: one step at a time, threading state + done ---
    state = net.initial_state(B)
    roll_actions, roll_logp, roll_val = [], [], []
    for t in range(T):
        a, lp, v, state = net.act_recurrent(frames[t], state, done[t])
        roll_actions.append(a)
        roll_logp.append(lp)
        roll_val.append(v)
    roll_actions = torch.stack(roll_actions)  # [T, B]
    roll_logp = torch.stack(roll_logp)        # [T, B]
    roll_val = torch.stack(roll_val)          # [T, B]

    # --- update replay: full sequence in one call, scoring the rollout actions ---
    obs, d = _flat(frames, done)
    _, logp, _ent, val, _ = net.get_action_and_value(
        obs, net.initial_state(B), d, roll_actions.reshape(T * B)
    )
    logp = logp.view(T, B)
    val = val.view(T, B)

    assert torch.allclose(roll_logp, logp, atol=1e-5), "rollout vs replay log_prob mismatch"
    assert torch.allclose(roll_val, val, atol=1e-5), "rollout vs replay value mismatch"


# 15 ────────────────────────────────────────────────────────────────────────
def test_act_recurrent_threads_state_and_shapes():
    """act_recurrent has rollout-shaped (B,) outputs and returns a usable next
    state. Two steps fed through it equal the 2-step replay."""
    net = _net(use_lstm=True)
    B = 4
    f0 = _rand_frames(1, B, seed=170)[0]
    f1 = _rand_frames(1, B, seed=171)[0]
    done0 = torch.zeros(B)
    done1 = torch.zeros(B)

    s0 = net.initial_state(B)
    a0, lp0, v0, s1 = net.act_recurrent(f0, s0, done0)
    a1, lp1, v1, s2 = net.act_recurrent(f1, s1, done1)
    assert a0.shape == (B,) and lp0.shape == (B,) and v0.shape == (B,)
    assert s1[0].shape == (1, B, F) and s2[0].shape == (1, B, F)

    # value at step 1 (threaded) == 2-step replay value at step 1
    frames = torch.stack([f0, f1])           # [2, B, C, H, W]
    done = torch.stack([done0, done1])       # [2, B]
    vals = _values(net, frames, done, net.initial_state(B))
    assert torch.allclose(vals[0], v0, atol=1e-5)
    assert torch.allclose(vals[1], v1, atol=1e-5)


# 16 ────────────────────────────────────────────────────────────────────────
def test_forward_and_act_refuse_lstm_mode():
    """The feedforward forward()/act() carry no state or episode boundaries, so
    they must hard-error in LSTM mode rather than silently run a stateless step."""
    net = _net(use_lstm=True)
    obs = _rand_frames(1, 2, seed=180).reshape(2, *OBS)
    with pytest.raises(RuntimeError, match="feedforward-only"):
        net.forward(obs)
    with pytest.raises(RuntimeError, match="feedforward-only"):
        net.act(obs)
