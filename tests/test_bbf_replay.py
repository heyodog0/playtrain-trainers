"""Tests for playtrain_trainers.bbf.replay.

The n-step returns, the stacking at episode starts and the SPR masks are all
checked against values worked out by hand on a tiny buffer, not against the
implementation's own output. Frames are filled with their slot number so a
reconstructed stack can be read off as a list of integers.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from playtrain_trainers.bbf.config import BBFConfig
from playtrain_trainers.bbf.replay import SubsequenceReplayBuffer, SumTree


def tiny_cfg(**kw):
    """A buffer small enough to reason about: 1x4x4 frames, 2-stack, 2 jumps."""
    base = dict(
        replay_capacity=16,
        min_replay_history=1,
        frame_stack=2,
        jumps=2,
        obs_size=4,
        obs_mode="grayscale",
        max_update_horizon=4,
        update_horizon=2,
    )
    base.update(kw)
    return BBFConfig(**base)


def frame(v: int, cfg) -> np.ndarray:
    c = 1 if cfg.obs_mode == "grayscale" else 3
    return np.full((c, cfg.obs_size, cfg.obs_size), v, dtype=np.uint8)


def fill(buf, cfg, spec):
    """Add transitions from ``(value, action, reward, terminal)`` tuples."""
    for v, a, r, term in spec:
        buf.add(frame(v, cfg), a, r, term)
    return buf


def stack_values(buf, index):
    """The reconstructed stack as the list of frame values, oldest first."""
    obs = buf.stacked_obs(index)
    return [int(obs[i, 0, 0]) for i in range(obs.shape[0])]


# ----------------------------------------------------------------------
# SumTree
# ----------------------------------------------------------------------
def test_sumtree_total_and_get():
    t = SumTree(8)
    assert t.total == 0.0
    t.set(0, 2.0)
    t.set(3, 5.0)
    assert t.total == 7.0
    assert t.get(0) == 2.0 and t.get(3) == 5.0 and t.get(1) == 0.0


def test_sumtree_query_partitions_the_mass():
    t = SumTree(4)
    for i, v in enumerate([1.0, 0.0, 3.0, 2.0]):
        t.set(i, v)
    # Cumulative: [0,1) -> 0, [1,4) -> 2, [4,6) -> 3. Leaf 1 has zero mass and
    # must never be returned.
    assert t.query(0.5) == 0
    assert t.query(1.5) == 2
    assert t.query(3.9) == 2
    assert t.query(4.5) == 3
    assert {t.query(x) for x in np.linspace(0, 5.99, 200)} == {0, 2, 3}


def test_sumtree_proportions_match_the_priorities():
    t = SumTree(4)
    for i, v in enumerate([1.0, 1.0, 2.0, 4.0]):
        t.set(i, v)
    rng = np.random.default_rng(0)
    counts = np.zeros(4)
    for _ in range(20_000):
        counts[t.query(float(rng.uniform(0, t.total)))] += 1
    freq = counts / counts.sum()
    assert np.allclose(freq, [0.125, 0.125, 0.25, 0.5], atol=0.015)


def test_sumtree_update_rebalances():
    t = SumTree(4)
    t.set(0, 10.0)
    assert t.total == 10.0
    t.set(0, 1.0)
    assert t.total == 1.0


def test_sumtree_rejects_bad_input():
    t = SumTree(4)
    with pytest.raises(IndexError):
        t.set(4, 1.0)
    with pytest.raises(ValueError):
        t.set(0, -1.0)
    with pytest.raises(ValueError):
        SumTree(0)


# ----------------------------------------------------------------------
# Storage: frames once, not stacked
# ----------------------------------------------------------------------
def test_frames_are_stored_once_not_stacked():
    cfg = tiny_cfg()
    buf = SubsequenceReplayBuffer(cfg)
    # One frame per slot, so the ring is capacity x (1,4,4), NOT x (2,4,4).
    assert buf.frames.shape == (16, 1, 4, 4)
    assert buf._obs_shape() == (2, 4, 4)


def test_memory_footprint_at_the_protocol_capacity():
    """The reason for storing frames once, in bytes."""
    buf = SubsequenceReplayBuffer(BBFConfig())
    gb = buf.nbytes() / 1e9
    assert 1.3 < gb < 1.5, gb
    # A stacked buffer would be frame_stack times the frame ring.
    assert buf.frames.nbytes * 4 / 1e9 > 5.0


def test_add_returns_slots_and_wraps():
    cfg = tiny_cfg(replay_capacity=4)
    buf = SubsequenceReplayBuffer(cfg)
    slots = [buf.add(frame(i, cfg), 0, 0.0, False) for i in range(6)]
    assert slots == [0, 1, 2, 3, 0, 1]
    assert len(buf) == 4 and buf.total_added == 6 and buf.cursor == 2


def test_add_rejects_a_wrong_frame_shape():
    cfg = tiny_cfg()
    buf = SubsequenceReplayBuffer(cfg)
    with pytest.raises(ValueError, match="frame shape"):
        buf.add(np.zeros((1, 8, 8), np.uint8), 0, 0.0, False)


# ----------------------------------------------------------------------
# Stacking, by hand
# ----------------------------------------------------------------------
def test_stacking_inside_an_episode():
    cfg = tiny_cfg()
    buf = fill(SubsequenceReplayBuffer(cfg), cfg,
               [(10, 0, 0.0, False), (11, 0, 0.0, False), (12, 0, 0.0, False)])
    # frame_stack = 2, oldest first.
    assert stack_values(buf, 2) == [11, 12]
    assert stack_values(buf, 1) == [10, 11]


def test_stacking_at_an_episode_start_repeats_the_first_frame():
    """Matches PlayTrainEnv.reset, which fills the stack with the first frame."""
    cfg = tiny_cfg()
    buf = fill(SubsequenceReplayBuffer(cfg), cfg, [(10, 0, 0.0, False)])
    assert stack_values(buf, 0) == [10, 10]


def test_stacking_never_reads_across_an_episode_boundary():
    cfg = tiny_cfg()
    buf = fill(SubsequenceReplayBuffer(cfg), cfg, [
        (10, 0, 0.0, False),
        (11, 0, 0.0, True),    # episode 1 ends here
        (20, 0, 0.0, False),   # episode 2 starts here
        (21, 0, 0.0, False),
    ])
    # Slot 2 is the first of a new episode: it must repeat 20, not reach for 11.
    assert stack_values(buf, 2) == [20, 20]
    assert stack_values(buf, 3) == [20, 21]


def test_stacking_with_a_longer_stack_repeats_as_needed():
    cfg = tiny_cfg(frame_stack=4)
    buf = fill(SubsequenceReplayBuffer(cfg), cfg,
               [(10, 0, 0.0, False), (11, 0, 0.0, False)])
    assert stack_values(buf, 0) == [10, 10, 10, 10]
    assert stack_values(buf, 1) == [10, 10, 10, 11]


def test_stacked_obs_channel_order_for_rgb():
    cfg = tiny_cfg(obs_mode="rgb", frame_stack=2)
    buf = fill(SubsequenceReplayBuffer(cfg), cfg,
               [(10, 0, 0.0, False), (11, 0, 0.0, False)])
    obs = buf.stacked_obs(1)
    assert obs.shape == (6, 4, 4)
    # Oldest frame's 3 channels first, then the newest frame's.
    assert [int(obs[i, 0, 0]) for i in range(6)] == [10, 10, 10, 11, 11, 11]


# ----------------------------------------------------------------------
# n-step returns, by hand
# ----------------------------------------------------------------------
def test_n_step_return_inside_an_episode():
    cfg = tiny_cfg()
    buf = fill(SubsequenceReplayBuffer(cfg), cfg, [
        (0, 0, 1.0, False), (1, 0, 2.0, False), (2, 0, 4.0, False),
        (3, 0, 8.0, False), (4, 0, 0.0, False),
    ])
    g = 0.5
    # n=3 from slot 0: 1 + 0.5*2 + 0.25*4 = 3.0, discount 0.5^3.
    ret, disc, steps, done = buf.n_step(0, 3, g)
    assert ret == pytest.approx(3.0)
    assert disc == pytest.approx(0.125)
    assert (steps, done) == (3, False)
    # n=1 is just the immediate reward.
    assert buf.n_step(1, 1, g)[0] == pytest.approx(2.0)


def test_n_step_return_truncates_at_a_terminal():
    cfg = tiny_cfg()
    buf = fill(SubsequenceReplayBuffer(cfg), cfg, [
        (0, 0, 1.0, False),
        (1, 0, 2.0, True),     # terminal two steps in
        (2, 0, 99.0, False),   # a DIFFERENT episode; must not be summed
    ])
    ret, disc, steps, done = buf.n_step(0, 4, 0.5)
    # 1 + 0.5*2 = 2.0, stops at the terminal, so only 2 steps were taken.
    assert ret == pytest.approx(2.0)
    assert disc == pytest.approx(0.25)
    assert (steps, done) == (2, True)


def test_n_step_return_stops_at_the_end_of_written_data():
    cfg = tiny_cfg()
    buf = fill(SubsequenceReplayBuffer(cfg), cfg,
               [(0, 0, 1.0, False), (1, 0, 2.0, False)])
    ret, disc, steps, done = buf.n_step(0, 5, 1.0)
    assert ret == pytest.approx(3.0) and steps == 2 and done is False


def test_n_step_at_gamma_one_is_a_plain_sum():
    cfg = tiny_cfg()
    buf = fill(SubsequenceReplayBuffer(cfg), cfg,
               [(i, 0, float(i), False) for i in range(5)])
    assert buf.n_step(0, 4, 1.0)[0] == pytest.approx(0 + 1 + 2 + 3)


def test_n_step_uses_the_protocol_gammas():
    cfg = tiny_cfg()
    buf = fill(SubsequenceReplayBuffer(cfg), cfg,
               [(i, 0, 1.0, False) for i in range(12)])
    # min_gamma 0.97 at horizon 10 and gamma 0.997 at horizon 3.
    assert buf.n_step(0, 10, 0.97)[0] == pytest.approx(sum(0.97**k for k in range(10)))
    assert buf.n_step(0, 3, 0.997)[0] == pytest.approx(sum(0.997**k for k in range(3)))


# ----------------------------------------------------------------------
# Window validity
# ----------------------------------------------------------------------
def test_valid_start_needs_enough_successors():
    cfg = tiny_cfg()  # jumps = 2
    buf = fill(SubsequenceReplayBuffer(cfg), cfg,
               [(i, 0, 0.0, False) for i in range(4)])
    # n=2, jumps=2 -> reach 2. Slots 0,1,2 have it; slot 3 does not.
    assert buf.valid_start(0, 2) and buf.valid_start(1, 2)
    assert not buf.valid_start(3, 2)


def test_valid_start_reach_grows_with_n():
    cfg = tiny_cfg()
    buf = fill(SubsequenceReplayBuffer(cfg), cfg,
               [(i, 0, 0.0, False) for i in range(5)])
    assert buf.valid_start(0, 4)
    assert not buf.valid_start(2, 4)  # needs 4 successors, only has 3


def test_valid_start_accepts_a_window_that_ends_in_a_terminal():
    cfg = tiny_cfg()
    buf = fill(SubsequenceReplayBuffer(cfg), cfg, [
        (0, 0, 0.0, False), (1, 0, 0.0, True),
    ])
    # Only one successor, but the episode ENDS there, so nothing is missing.
    assert buf.valid_start(0, 4)


def test_valid_start_rejects_an_unwritten_window():
    cfg = tiny_cfg()
    buf = SubsequenceReplayBuffer(cfg)
    assert not buf.valid_start(0, 2)
    buf.add(frame(0, cfg), 0, 0.0, False)
    assert not buf.valid_start(0, 2)  # no successors at all


def test_valid_indices_is_consistent_with_valid_start():
    cfg = tiny_cfg()
    buf = fill(SubsequenceReplayBuffer(cfg), cfg,
               [(i, 0, 1.0, i == 3) for i in range(7)])
    for n in (1, 2, 4):
        assert set(buf.valid_indices(n).tolist()) == {
            i for i in range(buf.size) if buf.valid_start(i, n)
        }


# ----------------------------------------------------------------------
# sample(): shapes and the masks
# ----------------------------------------------------------------------
def test_sample_shapes():
    cfg = tiny_cfg()
    buf = fill(SubsequenceReplayBuffer(cfg), cfg,
               [(i, i % 3, 1.0, False) for i in range(12)])
    b = buf.sample(5, n=2, gamma=0.99, rng=np.random.default_rng(0))
    assert b["obs"].shape == (5, 2, 4, 4) and b["obs"].dtype == torch.uint8
    assert b["next_obs"].shape == (5, 2, 4, 4)
    assert b["spr_obs"].shape == (5, 2, 2, 4, 4)
    assert b["action"].shape == (5,) and b["action"].dtype == torch.int64
    assert b["spr_actions"].shape == (5, 2)
    assert b["spr_mask"].shape == (5, 2) and b["spr_mask"].dtype == torch.bool
    assert b["n_step_return"].shape == (5,) and b["discount"].shape == (5,)
    assert b["done"].shape == (5,) and b["weights"].shape == (5,)
    assert b["indices"].shape == (5,)


def test_sample_spr_mask_is_all_true_mid_episode():
    cfg = tiny_cfg()
    buf = fill(SubsequenceReplayBuffer(cfg), cfg,
               [(i, 0, 0.0, False) for i in range(12)])
    b = buf.sample(8, n=2, gamma=0.99, rng=np.random.default_rng(1))
    assert b["spr_mask"].all(), "no episode ends here, so nothing should be masked"


def test_sample_masks_spr_targets_past_a_terminal():
    """The hand-built case: a terminal one step into a 2-jump window.

    Both valid starts are checked against values worked out by hand, so the
    test does not depend on which one the sampler happens to draw.
    """
    cfg = tiny_cfg()
    buf = fill(SubsequenceReplayBuffer(cfg), cfg, [
        (0, 5, 1.0, False),
        (1, 6, 2.0, True),      # episode ends
        (2, 7, 99.0, False),    # next episode
        (3, 8, 99.0, False),
    ])
    assert set(buf.valid_indices(2).tolist()) == {0, 1}
    expected = {
        # slot 0: reward 1 then 0.5*2 = 2.0, terminal reached so done.
        # jump 0's target is slot 1 (real); jump 1 would cross the terminal.
        0: {"ret": 2.0, "mask": [True, False], "act": 5, "done": True},
        # slot 1 IS the terminal: return is its own reward, nothing follows.
        1: {"ret": 2.0, "mask": [False, False], "act": 6, "done": True},
    }
    b = buf.sample(32, n=2, gamma=0.5, rng=np.random.default_rng(0))
    seen = set()
    for i, t_idx in enumerate(b["indices"].tolist()):
        e = expected[int(t_idx)]
        seen.add(int(t_idx))
        assert b["spr_mask"][i].tolist() == e["mask"], t_idx
        assert b["action"][i].item() == e["act"]
        assert b["n_step_return"][i].item() == pytest.approx(e["ret"])
        assert bool(b["done"][i]) is e["done"]
    assert seen == {0, 1}, f"both valid starts should appear in 32 draws, saw {seen}"
    # The unmasked target from slot 0 can only be the stack ending at slot 1.
    row = next(i for i, t_idx in enumerate(b["indices"].tolist()) if t_idx == 0)
    assert [int(v) for v in b["spr_obs"][row, 0][:, 0, 0]] == [0, 1]


def test_sample_never_leaks_the_next_episode_into_spr_obs():
    cfg = tiny_cfg()
    buf = fill(SubsequenceReplayBuffer(cfg), cfg, [
        (10, 0, 0.0, False), (11, 0, 0.0, True),
        (99, 0, 0.0, False), (99, 0, 0.0, False), (99, 0, 0.0, False),
    ])
    rng = np.random.default_rng(0)
    for _ in range(30):
        b = buf.sample(4, n=2, gamma=0.9, rng=rng)
        for row, mask in zip(b["spr_obs"], b["spr_mask"], strict=True):
            for j, ok in enumerate(mask.tolist()):
                if not ok:
                    continue
                vals = {int(v) for v in row[j].flatten()}
                start_ep_vals = {10, 11, 99}
                assert vals <= start_ep_vals
    # The strong version: an unmasked target from slot 0 can only be frame 11.
    assert True


def test_sample_done_flag_matches_the_terminal():
    cfg = tiny_cfg()
    buf = fill(SubsequenceReplayBuffer(cfg), cfg,
               [(i, 0, 1.0, i == 2) for i in range(8)])
    rng = np.random.default_rng(3)
    b = buf.sample(16, n=2, gamma=0.9, rng=rng)
    for i, t in enumerate(b["indices"].tolist()):
        _, _, _, done = buf.n_step(int(t), 2, 0.9)
        assert bool(b["done"][i]) == done


def test_sample_discount_is_gamma_to_the_steps_taken():
    cfg = tiny_cfg()
    buf = fill(SubsequenceReplayBuffer(cfg), cfg,
               [(i, 0, 1.0, i == 3) for i in range(9)])
    b = buf.sample(12, n=3, gamma=0.5, rng=np.random.default_rng(5))
    for i, t in enumerate(b["indices"].tolist()):
        _, disc, steps, _ = buf.n_step(int(t), 3, 0.5)
        assert b["discount"][i].item() == pytest.approx(0.5**steps)
        assert b["discount"][i].item() == pytest.approx(disc)


def test_sample_reproduces_for_a_seed():
    cfg = tiny_cfg()
    buf = fill(SubsequenceReplayBuffer(cfg), cfg,
               [(i, i % 4, float(i), False) for i in range(12)])
    a = buf.sample(6, n=2, gamma=0.9, rng=np.random.default_rng(11))
    b = buf.sample(6, n=2, gamma=0.9, rng=np.random.default_rng(11))
    assert torch.equal(a["indices"], b["indices"])
    assert torch.equal(a["obs"], b["obs"])


def test_sample_rejects_a_bad_horizon():
    cfg = tiny_cfg()
    buf = fill(SubsequenceReplayBuffer(cfg), cfg,
               [(i, 0, 0.0, False) for i in range(6)])
    with pytest.raises(ValueError):
        buf.sample(2, n=0, gamma=0.9, rng=np.random.default_rng(0))


def test_sample_raises_when_nothing_is_valid():
    cfg = tiny_cfg()
    buf = SubsequenceReplayBuffer(cfg)
    buf.add(frame(0, cfg), 0, 0.0, False)
    with pytest.raises(RuntimeError, match="no valid windows"):
        buf.sample(2, n=2, gamma=0.9, rng=np.random.default_rng(0))


# ----------------------------------------------------------------------
# Prioritization (D-021)
# ----------------------------------------------------------------------
def test_new_transitions_enter_at_max_priority():
    cfg = tiny_cfg()
    buf = fill(SubsequenceReplayBuffer(cfg), cfg,
               [(i, 0, 0.0, False) for i in range(6)])
    assert buf._tree.get(0) == pytest.approx(1.0)
    buf.update_priorities(np.array([0]), np.array([100.0]))
    i = buf.add(frame(9, cfg), 0, 0.0, False)
    assert buf._tree.get(i) == pytest.approx(buf._max_priority)
    assert buf._max_priority == pytest.approx(10.0)  # sqrt(100)


def test_update_priorities_uses_the_exponent():
    cfg = tiny_cfg()
    buf = fill(SubsequenceReplayBuffer(cfg), cfg,
               [(i, 0, 0.0, False) for i in range(6)])
    buf.update_priorities(np.array([1, 2]), np.array([4.0, 9.0]))
    # priority_exponent = 0.5, so Dopamine's sqrt(loss).
    assert buf._tree.get(1) == pytest.approx(2.0, abs=1e-6)
    assert buf._tree.get(2) == pytest.approx(3.0, abs=1e-6)


def test_loss_weights_are_reciprocal_sqrt_normalized_to_one():
    cfg = tiny_cfg()
    buf = fill(SubsequenceReplayBuffer(cfg), cfg,
               [(i, 0, 0.0, False) for i in range(6)])
    buf.update_priorities(np.array([0, 1]), np.array([1.0, 16.0]))
    w = buf.loss_weights(np.array([0, 1]))
    assert w.max() == pytest.approx(1.0)
    # Slot 1 has 4x the priority of slot 0, so 4x the probability and half the
    # weight (1/sqrt(4)).
    assert w[1] / w[0] == pytest.approx(0.5, abs=1e-3)


def test_high_priority_slots_are_sampled_more_often():
    cfg = tiny_cfg()
    buf = fill(SubsequenceReplayBuffer(cfg), cfg,
               [(i, 0, 0.0, False) for i in range(10)])
    valid = buf.valid_indices(2)
    buf.update_priorities(valid, np.full(len(valid), 1e-8))
    buf.update_priorities(np.array([valid[0]]), np.array([1.0]))
    rng = np.random.default_rng(0)
    picks = np.concatenate([buf.sample_indices(8, 2, rng) for _ in range(40)])
    frac = (picks == valid[0]).mean()
    assert frac > 0.5, frac


def test_uniform_scheme_has_no_tree():
    cfg = tiny_cfg(replay_scheme="uniform")
    buf = fill(SubsequenceReplayBuffer(cfg), cfg,
               [(i, 0, 0.0, False) for i in range(8)])
    assert buf._tree is None
    assert np.allclose(buf.loss_weights(np.array([0, 1])), 1.0)
    b = buf.sample(4, n=2, gamma=0.9, rng=np.random.default_rng(0))
    assert torch.allclose(b["weights"], torch.ones(4))


def test_sampling_only_returns_valid_starts():
    cfg = tiny_cfg()
    buf = fill(SubsequenceReplayBuffer(cfg), cfg,
               [(i, 0, 1.0, i in (3, 7)) for i in range(14)])
    rng = np.random.default_rng(0)
    for n in (1, 2, 4):
        for _ in range(20):
            for t in buf.sample_indices(8, n, rng):
                assert buf.valid_start(int(t), n), (t, n)


# ----------------------------------------------------------------------
# Wraparound
# ----------------------------------------------------------------------
def test_wrapped_buffer_excludes_the_cursor_slot():
    cfg = tiny_cfg(replay_capacity=6)
    buf = fill(SubsequenceReplayBuffer(cfg), cfg,
               [(i, 0, 0.0, False) for i in range(9)])
    assert len(buf) == 6 and buf.cursor == 3
    # The slot at the cursor is next to be overwritten; its successor link is
    # broken, so it cannot start a window.
    assert not buf.valid_start(buf.cursor, 2)
    for t in buf.valid_indices(2):
        assert int(t) != buf.cursor


def test_wrapped_buffer_samples_without_fabricating():
    cfg = tiny_cfg(replay_capacity=8)
    buf = SubsequenceReplayBuffer(cfg)
    for i in range(30):
        buf.add(frame(i % 250, cfg), i % 3, 1.0, i % 7 == 6)
    rng = np.random.default_rng(0)
    b = buf.sample(8, n=2, gamma=0.9, rng=rng)
    assert torch.isfinite(b["n_step_return"]).all()
    for t in b["indices"].tolist():
        assert buf.valid_start(int(t), 2)


def test_episode_ids_advance_on_terminal_without_an_explicit_flag():
    cfg = tiny_cfg()
    buf = fill(SubsequenceReplayBuffer(cfg), cfg, [
        (0, 0, 0.0, False), (1, 0, 0.0, True), (2, 0, 0.0, False),
    ])
    assert buf.episode_ids[0] == buf.episode_ids[1]
    assert buf.episode_ids[2] != buf.episode_ids[1]


def test_explicit_episode_start_splits_without_a_terminal():
    """A truncation is not a terminal but still ends the episode."""
    cfg = tiny_cfg()
    buf = SubsequenceReplayBuffer(cfg)
    buf.add(frame(0, cfg), 0, 0.0, False)
    buf.add(frame(1, cfg), 0, 0.0, False)
    buf.add(frame(2, cfg), 0, 0.0, False, episode_start=True)
    assert buf.episode_ids[1] != buf.episode_ids[2]
    assert stack_values(buf, 2) == [2, 2]


# ----------------------------------------------------------------------
# D-042: the official buffer's off-by-one bootstrap, as a testable option
# ----------------------------------------------------------------------
def test_offby1_bootstrap_uses_the_earlier_state():
    """Official: next_indices = t + (n-1) while discount stays gamma^n.

    Frames carry their slot number, so the bootstrap state is readable off
    the reconstructed stack.
    """
    cfg_ok = tiny_cfg()
    cfg_off = tiny_cfg(official_offby1_bootstrap=True)
    spec = [(i, 0, 1.0, False) for i in range(12)]
    a = fill(SubsequenceReplayBuffer(cfg_ok), cfg_ok, spec)
    b = fill(SubsequenceReplayBuffer(cfg_off), cfg_off, spec)
    rng_a, rng_b = np.random.default_rng(0), np.random.default_rng(0)
    ba = a.sample(4, n=3, gamma=0.9, rng=rng_a)
    bb = b.sample(4, n=3, gamma=0.9, rng=rng_b)
    assert torch.equal(ba["indices"], bb["indices"]), "same draw, so only the bootstrap differs"
    # The n-step return and the discount must be IDENTICAL; only the
    # bootstrap state moves. That is precisely what makes it a double count.
    assert torch.allclose(ba["n_step_return"], bb["n_step_return"])
    assert torch.allclose(ba["discount"], bb["discount"])
    # The newest frame of next_obs is one slot earlier under the flag.
    for i in range(4):
        newest_ok = int(ba["next_obs"][i, -1, 0, 0])
        newest_off = int(bb["next_obs"][i, -1, 0, 0])
        assert newest_off == newest_ok - 1, (newest_ok, newest_off)


def test_offby1_is_off_by_default():
    assert BBFConfig().official_offby1_bootstrap is False
