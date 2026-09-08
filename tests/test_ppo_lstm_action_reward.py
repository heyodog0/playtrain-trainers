"""Tests for feed_prev_action_reward on the ActorCritic LSTM core.

When enabled, the core sees [cnn(frame_t), r_{t-1}, one_hot(a_{t-1})] (IMPALA /
monobeast / R2D2 convention). These tests pin the properties specific to that
extra conditioning, on top of the recurrence properties in
test_ppo_lstm_net.py (which still hold and are not re-tested here):

  1. LSTM input width grows by (n_actions + 1)
  2. feed_prev_action_reward requires use_lstm (guards the argmax footgun)
  3. None args == explicit zeros (safe default for step 0)
  4. The previous action actually changes the output (one-hot is consumed)
  5. The previous reward actually changes the output (reward channel consumed)
  6. The reward channel is symlog-squashed (bounded-growth, magnitude-preserving)
  7. Rollout (act_recurrent, threading a/r) == update replay
     (get_action_and_value over the sequence) for the SAME actions/rewards,
     across an episode reset — the importance-ratio alignment property, now
     with the action/reward channel in the loop.
  8. symlog_reward=False concats the reward verbatim (caller-side bounding)
"""
from __future__ import annotations

import pytest
import torch

from playtrain_trainers.policy import ActorCritic


OBS = (3, 16, 16)  # (C, H, W)
A = 5
F = 32


def _net(seed: int = 0, feed: bool = True) -> ActorCritic:
    torch.manual_seed(seed)
    net = ActorCritic(
        n_actions=A, in_channels=OBS[0], features_dim=F, input_hw=OBS[1],
        use_lstm=True, feed_prev_action_reward=feed,
    )
    net.eval()
    return net


def _rand_frames(T: int, B: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (T, B, *OBS), dtype=torch.uint8, generator=g)


# 1 ──────────────────────────────────────────────────────────────────────────
def test_lstm_input_width_grows_with_action_reward():
    plain = _net(feed=False)
    fed = _net(feed=True)
    assert plain.lstm.input_size == F
    assert fed.lstm.input_size == F + A + 1  # one-hot action + scalar reward


# 2 ──────────────────────────────────────────────────────────────────────────
def test_feed_requires_lstm():
    """Conditioning on action history without a recurrent core is the exact
    argmax-brittleness footgun — the constructor must refuse it."""
    with pytest.raises(ValueError, match="requires use_lstm"):
        ActorCritic(n_actions=A, in_channels=OBS[0], features_dim=F,
                    input_hw=OBS[1], use_lstm=False, feed_prev_action_reward=True)


# 3 ──────────────────────────────────────────────────────────────────────────
def test_none_args_equal_explicit_zeros():
    """Omitting last_action/reward (e.g. rollout step 0) must equal feeding
    zeros — so the very first step has a well-defined, history-free input."""
    net = _net()
    T, B = 3, 2
    obs = _rand_frames(T, B, seed=1).reshape(T * B, *OBS)
    done = torch.zeros(T * B)
    state = net.initial_state(B)

    z_none, _ = net.get_states(obs, state, done)
    z_zero, _ = net.get_states(
        obs, state, done,
        last_action=torch.zeros(T * B, dtype=torch.long),
        reward=torch.zeros(T * B),
    )
    assert torch.allclose(z_none, z_zero, atol=1e-6)


# 4 ──────────────────────────────────────────────────────────────────────────
def test_previous_action_changes_output():
    """Same frame + same state, different a_{t-1} -> different features. Proves
    the one-hot action channel is actually wired into the core."""
    net = _net()
    B = 4
    obs = _rand_frames(1, B, seed=2)[0]
    done = torch.zeros(B)
    state = net.initial_state(B)
    r = torch.zeros(B)

    z0, _ = net.get_states(obs, state, done, torch.zeros(B, dtype=torch.long), r)
    z1, _ = net.get_states(obs, state, done, torch.full((B,), 3, dtype=torch.long), r)
    assert (z0 - z1).abs().max() > 1e-5, "previous action had no effect"


# 5 ──────────────────────────────────────────────────────────────────────────
def test_previous_reward_changes_output():
    """Same frame + same state + same action, different r_{t-1} -> different
    features (within the clamp range). Proves the reward channel is consumed."""
    net = _net()
    B = 4
    obs = _rand_frames(1, B, seed=3)[0]
    done = torch.zeros(B)
    state = net.initial_state(B)
    a = torch.zeros(B, dtype=torch.long)

    z_neg, _ = net.get_states(obs, state, done, a, torch.full((B,), -1.0))
    z_pos, _ = net.get_states(obs, state, done, a, torch.full((B,), 1.0))
    assert (z_neg - z_pos).abs().max() > 1e-5, "previous reward had no effect"


# 6 ──────────────────────────────────────────────────────────────────────────
def test_reward_is_symlog_squashed():
    """The reward channel is symlog-squashed, NOT clamped to [-1,1]. symlog is
    bounded-GROWTH, not bounded-RANGE: it tames the env's ±80000-scale rewards to
    a sane input magnitude while PRESERVING ordering — so two distinct large
    rewards give distinct outputs (a hard clamp would collapse them to a sign
    bit, undoing the symlog reward shaping the door configs use)."""
    from playtrain_trainers.policy import symlog
    # the squash itself: odd, monotonic, compressing, bounded-growth
    assert torch.isclose(symlog(torch.tensor(0.0)), torch.tensor(0.0))
    assert symlog(torch.tensor(1e6)) > symlog(torch.tensor(1.0)) > 0.0
    assert symlog(torch.tensor(1e6)) < 20.0  # ~13.8 — bounded growth, not 1e6
    assert torch.isclose(symlog(torch.tensor(-3.0)), -symlog(torch.tensor(3.0)))

    # and it reaches the net: two distinct large rewards -> distinct features
    net = _net()
    B = 3
    obs = _rand_frames(1, B, seed=4)[0]
    done = torch.zeros(B)
    state = net.initial_state(B)
    a = torch.zeros(B, dtype=torch.long)
    z_big, _ = net.get_states(obs, state, done, a, torch.full((B,), 500.0))
    z_bigger, _ = net.get_states(obs, state, done, a, torch.full((B,), 80000.0))
    assert (z_big - z_bigger).abs().max() > 1e-5, \
        "symlog collapsed distinct large rewards (clamp-like behavior)"


# 8 ──────────────────────────────────────────────────────────────────────────
def test_symlog_reward_off_concats_verbatim():
    """symlog_reward=False skips the squash and concats the reward unchanged, so
    the caller controls bounding. A net with symlog OFF, fed symlog(r), must
    equal an identically-weighted net with symlog ON, fed raw r — proving the
    only difference between the two modes is exactly that symlog. And OFF fed raw
    differs from ON fed raw (no squash applied in OFF mode)."""
    from playtrain_trainers.policy import symlog
    torch.manual_seed(0)
    on = ActorCritic(n_actions=A, in_channels=OBS[0], features_dim=F, input_hw=OBS[1],
                     use_lstm=True, feed_prev_action_reward=True, symlog_reward=True).eval()
    torch.manual_seed(0)  # identical weights (symlog_reward changes no layer)
    off = ActorCritic(n_actions=A, in_channels=OBS[0], features_dim=F, input_hw=OBS[1],
                      use_lstm=True, feed_prev_action_reward=True, symlog_reward=False).eval()

    B = 3
    obs = _rand_frames(1, B, seed=9)[0]
    done = torch.zeros(B)
    a = torch.zeros(B, dtype=torch.long)
    raw = torch.full((B,), 500.0)

    z_on, _ = on.get_states(obs, on.initial_state(B), done, a, raw)
    z_off_presq, _ = off.get_states(obs, off.initial_state(B), done, a, symlog(raw))
    assert torch.allclose(z_on, z_off_presq, atol=1e-5), "OFF did not concat verbatim"

    z_off_raw, _ = off.get_states(obs, off.initial_state(B), done, a, raw)
    assert (z_on - z_off_raw).abs().max() > 1e-5, "OFF applied a squash it shouldn't"


# 7 ──────────────────────────────────────────────────────────────────────────
def test_rollout_equals_replay_with_action_reward():
    """The PPO-critical alignment property, now with a/r in the loop. Roll out
    one step at a time threading lstm_state AND the previous action/reward (with
    a mid-episode reset), then replay the whole sequence in one
    get_action_and_value call scoring the same actions and feeding the same
    a_{t-1}/r_{t-1}. log_prob and value must match step-for-step."""
    net = _net(seed=7)
    T, B = 6, 3
    frames = _rand_frames(T, B, seed=170)
    done = torch.zeros(T, B)
    done[3] = torch.tensor([True, False, False])

    # buffers aligned with frame[t]: prev_a[t] = a_{t-1}, prev_r[t] = r_{t-1}
    prev_a = torch.zeros(T, B, dtype=torch.long)
    prev_r = torch.zeros(T, B)

    state = net.initial_state(B)
    roll_a, roll_lp, roll_v = [], [], []
    last_a = torch.zeros(B, dtype=torch.long)
    last_r = torch.zeros(B)
    for t in range(T):
        prev_a[t] = last_a
        prev_r[t] = last_r
        a, lp, v, state = net.act_recurrent(frames[t], state, done[t], last_a, last_r)
        roll_a.append(a)
        roll_lp.append(lp)
        roll_v.append(v)
        # synthesize a reward for this transition (any deterministic value works)
        last_r = (a.float() - 2.0) * 0.5      # in [-1, 1] after the net's clamp
        last_a = a
    roll_a = torch.stack(roll_a)
    roll_lp = torch.stack(roll_lp)
    roll_v = torch.stack(roll_v)

    obs = frames.reshape(T * B, *OBS)
    _, lp, _e, v, _ = net.get_action_and_value(
        obs, net.initial_state(B), done.reshape(T * B),
        action=roll_a.reshape(T * B),
        last_action=prev_a.reshape(T * B),
        reward=prev_r.reshape(T * B),
    )
    lp = lp.view(T, B)
    v = v.view(T, B)
    assert torch.allclose(roll_lp, lp, atol=1e-5), "rollout vs replay log_prob mismatch"
    assert torch.allclose(roll_v, v, atol=1e-5), "rollout vs replay value mismatch"
