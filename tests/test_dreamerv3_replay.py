"""U04 — replay, against `core/replay.py`, `chunk.py`, `selectors.py`, `streams.py`
and `agent._apply_replay_context` at the pinned commit.

Every buffer here is hand-built: steps carry a counter so a sampled window can be
checked to be contiguous, and the chunk size is tiny so boundary crossings happen in
a handful of steps rather than after 1024.
"""

from __future__ import annotations

import numpy as np
import pytest

from playtrain_trainers.dreamerv3 import config as C
from playtrain_trainers.dreamerv3 import replay as RP


def make_step(i: int, is_first: bool = False, is_last: bool = False) -> dict:
    return {
        "count": np.int32(i),
        "image": np.full((2, 2, 3), i % 256, np.uint8),
        "action": np.int32(i % 4),
        "reward": np.float32(i),
        "is_first": np.bool_(is_first),
        "is_last": np.bool_(is_last),
        "is_terminal": np.bool_(is_last),
        "deter": np.full(4, float(i), np.float32),
    }


def fill(rp: RP.Replay, n: int, episode_starts: tuple[int, ...] = ()) -> None:
    for i in range(n):
        rp.add(make_step(i, is_first=i in episode_starts, is_last=(i + 1) in episode_starts))


# ----------------------------------------------------------------------
# When an item becomes sampleable
# ----------------------------------------------------------------------
def test_nothing_is_sampleable_until_length_steps_exist() -> None:
    """`add` inserts only once `len(stream) >= length`, which is why training starts
    at exactly `batch_size * batch_length` steps and not before."""
    rp = RP.Replay(length=5, chunksize=64)
    for i in range(4):
        rp.add(make_step(i))
        assert len(rp) == 0
    rp.add(make_step(4))
    assert len(rp) == 1  # the window starting at step 0 is now complete
    rp.add(make_step(5))
    assert len(rp) == 2


def test_sampled_window_is_contiguous_and_the_right_length() -> None:
    rp = RP.Replay(length=5, chunksize=64, seed=0)
    fill(rp, 50)
    batch = rp.sample(8)
    assert batch["count"].shape == (8, 5)
    for row in batch["count"]:
        assert list(row) == list(range(int(row[0]), int(row[0]) + 5))


def test_every_sampled_start_is_a_real_item() -> None:
    rp = RP.Replay(length=4, chunksize=64, seed=1)
    fill(rp, 20)
    starts = {int(rp.sample(1)["count"][0, 0]) for _ in range(200)}
    # 20 steps, windows of 4: starts 0..16 inclusive are complete.
    assert starts <= set(range(0, 17))
    assert len(starts) > 5  # the uniform sampler really does move around


# ----------------------------------------------------------------------
# Chunk boundaries
# ----------------------------------------------------------------------
def test_window_crossing_a_chunk_boundary() -> None:
    """A window is not confined to one chunk; `_getseq` follows `succ`."""
    rp = RP.Replay(length=6, chunksize=4)
    fill(rp, 30)
    assert len(rp.chunks) > 1
    # Find the item that starts at count 2, which spans chunk 0 (steps 2, 3) and
    # chunk 1 (steps 4..7).
    for itemid, (chunkid, index) in rp.items.items():
        seq = rp._getseq(chunkid, index, concat=True)
        if int(seq["count"][0]) == 2:
            assert list(seq["count"]) == [2, 3, 4, 5, 6, 7]
            break
    else:
        pytest.fail("no item starting at count 2")


def test_window_spanning_three_chunks() -> None:
    rp = RP.Replay(length=7, chunksize=2)
    fill(rp, 40)
    for chunkid, index in rp.items.values():
        seq = rp._getseq(chunkid, index, concat=True)
        assert list(seq["count"]) == list(range(int(seq["count"][0]), int(seq["count"][0]) + 7))


def test_chunks_are_linked_by_succ_in_order() -> None:
    rp = RP.Replay(length=3, chunksize=4)
    fill(rp, 12)
    first = rp.items[0][0]
    seen, node = [], first
    while node is not None and node in rp.chunks:
        seen.append(node)
        node = rp.chunks[node].succ
    assert len(seen) >= 3


# ----------------------------------------------------------------------
# is_first / is_last annotation
# ----------------------------------------------------------------------
def test_is_first_is_forced_at_the_window_start() -> None:
    """`_annotate_batch` sets `is_first[:, 0] = True` whatever was stored, so a
    sampled window always looks like the start of an episode to the RSSM."""
    rp = RP.Replay(length=4, chunksize=64, seed=0)
    fill(rp, 30)
    batch = rp.sample(6)
    assert batch["is_first"][:, 0].all()


def test_is_last_is_set_before_a_real_episode_boundary() -> None:
    """A window whose interior contains an `is_first` must mark the step before it
    `is_last`, or the model would train across the seam as if it were one episode."""
    rp = RP.Replay(length=5, chunksize=64, seed=0)
    fill(rp, 40, episode_starts=(10, 25))
    found = False
    for _ in range(400):
        batch = rp.sample(1)
        counts = [int(c) for c in batch["count"][0]]
        if 10 in counts[1:]:
            i = counts.index(10)
            assert batch["is_first"][0, i]
            assert batch["is_last"][0, i - 1]
            found = True
            break
    assert found, "never drew a window containing the episode boundary"


def test_annotation_does_not_write_into_the_stored_chunk() -> None:
    """`_annotate_batch` copies before forcing the flag; if it did not, the store
    would slowly fill with fake episode starts at every sampled offset."""
    rp = RP.Replay(length=4, chunksize=64, seed=0)
    fill(rp, 20)
    for _ in range(50):
        rp.sample(4)
    stored = [rp.chunks[c].data["is_first"][: rp.chunks[c].length] for c in rp.chunks]
    assert not np.concatenate(stored).any()


# ----------------------------------------------------------------------
# online (the freshest steps are included)
# ----------------------------------------------------------------------
def test_online_queue_holds_every_length_th_item() -> None:
    rp = RP.Replay(length=5, chunksize=64, online=True)
    fill(rp, 40)
    starts = [rp._getseq(c, i, concat=True)["count"][0] for c, i in rp.queue]
    assert list(starts) == [1, 6, 11, 16, 21, 26, 31]


def test_online_sample_drains_the_queue_first() -> None:
    """With `online: True` the newest complete window is guaranteed to be trained
    on, rather than waiting for a uniform draw to find it."""
    rp = RP.Replay(length=5, chunksize=64, online=True, seed=0)
    fill(rp, 40)
    queued = [int(rp._getseq(c, i, concat=True)["count"][0]) for c, i in rp.queue]
    got = [int(rp.sample(1)["count"][0, 0]) for _ in range(len(queued))]
    assert got == queued
    assert not rp.queue
    # Once drained, sampling falls back to the uniform selector.
    rp.sample(4)


def test_online_off_means_no_queue() -> None:
    rp = RP.Replay(length=5, chunksize=64, online=False)
    fill(rp, 40)
    assert not rp.queue


# ----------------------------------------------------------------------
# Capacity and eviction
# ----------------------------------------------------------------------
def test_fifo_eviction_at_capacity() -> None:
    rp = RP.Replay(length=3, chunksize=4, capacity=5)
    fill(rp, 40)
    assert len(rp) == 5
    starts = sorted(int(rp._getseq(c, i, concat=True)["count"][0]) for c, i in rp.items.values())
    # The five most recent complete windows, oldest ones dropped.
    assert starts == [33, 34, 35, 36, 37]


def test_evicted_chunks_are_dropped_from_memory() -> None:
    rp = RP.Replay(length=3, chunksize=4, capacity=5)
    fill(rp, 200)
    # Without reference counting the store would grow to 50 chunks.
    assert len(rp.chunks) <= 6
    assert len(rp.refs) == len(rp.chunks)


def test_sampling_still_works_after_heavy_eviction() -> None:
    rp = RP.Replay(length=4, chunksize=8, capacity=10, seed=0)
    fill(rp, 500)
    batch = rp.sample(8)
    for row in batch["count"]:
        assert list(row) == list(range(int(row[0]), int(row[0]) + 4))


# ----------------------------------------------------------------------
# update / write-back
# ----------------------------------------------------------------------
def test_update_writes_values_back_at_the_stepid_positions() -> None:
    rp = RP.Replay(length=4, chunksize=64, seed=0)
    fill(rp, 20)
    batch = rp.sample(3)
    fresh = np.full(batch["deter"].shape, 7.0, np.float32)
    rp.update({"stepid": batch["stepid"], "deter": fresh})
    again = rp._getseq(*rp.items[int(batch["count"][0, 0])], concat=True)
    assert (again["deter"] == 7.0).all()


def test_update_write_back_crosses_chunk_boundaries() -> None:
    rp = RP.Replay(length=6, chunksize=4, seed=0)
    fill(rp, 30)
    batch = rp.sample(2)
    rp.update({"stepid": batch["stepid"], "deter": np.full(batch["deter"].shape, 9.0, np.float32)})
    for n in range(2):
        start = int(batch["count"][n, 0])
        chunkid, index = rp.items[start]
        assert (rp._getseq(chunkid, index, concat=True)["deter"] == 9.0).all()


def test_update_ignores_evicted_steps() -> None:
    rp = RP.Replay(length=3, chunksize=4, capacity=5, seed=0)
    fill(rp, 40)
    batch = rp.sample(2)
    fill(rp, 200)  # evict everything the batch referenced
    rp.update({"stepid": batch["stepid"], "deter": np.zeros(batch["deter"].shape, np.float32)})


def test_update_counts_metrics_and_drops_priority() -> None:
    rp = RP.Replay(length=3, chunksize=64, seed=0)
    fill(rp, 20)
    batch = rp.sample(4)
    rp.update(
        {
            "stepid": batch["stepid"],
            "priority": np.ones((4, 3), np.float32),
            "deter": np.zeros(batch["deter"].shape, np.float32),
        }
    )
    assert rp.stats()["updates"] == 12


# ----------------------------------------------------------------------
# stepid
# ----------------------------------------------------------------------
def test_stepid_encodes_the_chunk_and_index() -> None:
    rp = RP.Replay(length=2, chunksize=4)
    fill(rp, 10)
    chunkid, index = rp.items[0]
    seq = rp._getseq(chunkid, index, concat=True)
    raw = seq["stepid"][0].tobytes()
    assert len(raw) == RP.STEPID_BYTES
    assert raw[:-4] == chunkid
    assert int.from_bytes(raw[-4:], "big") == index


# ----------------------------------------------------------------------
# Uniform selector
# ----------------------------------------------------------------------
def test_uniform_selector_swap_with_last_keeps_the_index_map_consistent() -> None:
    sel = RP.Uniform(seed=0)
    for k in range(6):
        sel[k] = None
    del sel[2]
    del sel[0]
    assert len(sel) == 4
    assert set(sel.keys) == {1, 3, 4, 5}
    for key, idx in sel.indices.items():
        assert sel.keys[idx] == key


def test_uniform_selector_covers_every_key() -> None:
    sel = RP.Uniform(seed=0)
    for k in range(5):
        sel[k] = None
    assert {sel() for _ in range(300)} == set(range(5))


# ----------------------------------------------------------------------
# Consec stream and the replay-context split
# ----------------------------------------------------------------------
def test_consec_chunk_at_the_frozen_config_is_the_whole_window() -> None:
    cfg = C.atari100k_config()
    batch = {
        "is_first": np.zeros((2, cfg.sequence_length), bool),
        "action": np.zeros((2, cfg.sequence_length), np.int32),
    }
    out = RP.consec_chunk(batch, 0, cfg.batch_length, cfg.replay_context)
    assert out["is_first"].shape == (2, 65)  # 1 * 64 + 1
    assert (out["consec"] == 0).all()


def test_apply_replay_context_splits_context_from_trained_steps() -> None:
    """The window is 65 long; step 0 supplies the carry, steps 1..64 are trained."""
    B, T, K = 2, 65, 1
    data = {
        "consec": np.zeros((B, T), np.int32),
        "deter": np.arange(B * T * 4, dtype=np.float32).reshape(B, T, 4),
        "stoch": np.zeros((B, T, 2, 3), np.float32),
        "image": np.zeros((B, T, 2, 2, 3), np.uint8),
        "action": np.arange(B * T, dtype=np.int32).reshape(B, T),
        "stepid": np.zeros((B, T, 20), np.uint8),
        "is_first": np.zeros((B, T), bool),
    }
    carry, obs, stepid = RP.apply_replay_context(data, ("deter", "stoch"), K)
    assert carry["deter"].shape == (B, 4)
    assert (carry["deter"] == data["deter"][:, 0]).all()  # the context step's latent
    assert obs["image"].shape == (B, 64, 2, 2, 3)
    assert stepid.shape == (B, 64, 20)
    assert (obs["image"] == data["image"][:, 1:]).all()


def test_apply_replay_context_action_alignment() -> None:
    """Step t is trained with the action that PRODUCED it, stored one row earlier.
    Off by one here and every transition the model learns is shifted."""
    B, T, K = 1, 5, 1
    data = {
        "consec": np.zeros((B, T), np.int32),
        "deter": np.zeros((B, T, 2), np.float32),
        "stoch": np.zeros((B, T, 1, 2), np.float32),
        "action": np.array([[10, 11, 12, 13, 14]], np.int32),
        "stepid": np.zeros((B, T, 20), np.uint8),
        "is_first": np.zeros((B, T), bool),
        "image": np.zeros((B, T, 1, 1, 3), np.uint8),
    }
    _, obs, _ = RP.apply_replay_context(data, ("deter", "stoch"), K)
    assert list(obs["action"][0]) == [10, 11, 12, 13]  # data['action'][:, K-1:-1]
    assert obs["image"].shape[1] == 4


def test_apply_replay_context_rejects_a_non_first_chunk() -> None:
    data = {
        "consec": np.ones((1, 3), np.int32),
        "deter": np.zeros((1, 3, 2), np.float32),
        "stoch": np.zeros((1, 3, 1, 2), np.float32),
        "action": np.zeros((1, 3), np.int32),
        "stepid": np.zeros((1, 3, 20), np.uint8),
        "is_first": np.zeros((1, 3), bool),
    }
    with pytest.raises(AssertionError, match="consec_train is 1"):
        RP.apply_replay_context(data)


# ----------------------------------------------------------------------
# The frozen config's numbers
# ----------------------------------------------------------------------
def test_frozen_config_shapes_end_to_end() -> None:
    cfg = C.atari100k_config()
    rp = RP.Replay(
        length=cfg.sequence_length,
        capacity=int(cfg.replay.size),
        chunksize=cfg.replay.chunksize,
        online=cfg.replay.online,
        seed=cfg.seed,
    )
    assert rp.length == 65
    fill(rp, 200)
    batch = rp.sample(cfg.batch_size)
    assert batch["image"].shape[:2] == (16, 65)
    batch = RP.consec_chunk(batch, 0, cfg.batch_length, cfg.replay_context)
    _, obs, stepid = RP.apply_replay_context(batch, ("deter",), cfg.replay_context)
    # 64 trained steps per sequence, 16 sequences = the 1024 of `batch_steps`.
    assert obs["image"].shape[:2] == (16, 64)
    assert obs["image"].shape[0] * obs["image"].shape[1] == cfg.batch_steps
    assert stepid.shape[:2] == (16, 64)


def test_replay_ratio_stat_matches_the_official_formula() -> None:
    rp = RP.Replay(length=4, chunksize=64, seed=0)
    fill(rp, 20)
    rp.sample(2)
    stats = rp.stats()
    assert stats["replay_ratio"] == 4 * 2 / 17  # length * samples / inserts
    assert rp.stats()["samples"] == 0  # stats() resets the counters
