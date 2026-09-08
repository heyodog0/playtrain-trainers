"""Tests for playtrain_trainers.impala.buffers."""
from __future__ import annotations

import torch

from playtrain_trainers.impala.buffers import create_buffers


def test_buffer_shapes_and_dtypes():
    bufs = create_buffers(
        obs_shape=(3, 64, 64), num_actions=7, unroll_length=80, num_buffers=4
    )
    expected_keys = {
        "frame", "reward", "done", "episode_return", "episode_step",
        "policy_logits", "baseline", "last_action", "action",
    }
    assert set(bufs.keys()) == expected_keys
    for key in bufs:
        assert len(bufs[key]) == 4  # num_buffers slots

    # T+1 leading dim everywhere
    assert bufs["frame"][0].shape == (81, 3, 64, 64)
    assert bufs["reward"][0].shape == (81,)
    assert bufs["policy_logits"][0].shape == (81, 7)

    assert bufs["frame"][0].dtype == torch.uint8
    assert bufs["done"][0].dtype == torch.bool
    assert bufs["action"][0].dtype == torch.int64
    assert bufs["episode_step"][0].dtype == torch.int32


def test_buffers_are_in_shared_memory():
    bufs = create_buffers(
        obs_shape=(3, 8, 8), num_actions=3, unroll_length=4, num_buffers=2
    )
    for key, slots in bufs.items():
        for t in slots:
            assert t.is_shared(), f"{key} slot not in shared memory"


def test_writes_to_one_slot_do_not_alias_other_slots():
    bufs = create_buffers(
        obs_shape=(3, 8, 8), num_actions=3, unroll_length=4, num_buffers=3
    )
    bufs["reward"][0].fill_(1.0)
    bufs["reward"][1].fill_(2.0)
    assert bufs["reward"][0].sum().item() == 5.0
    assert bufs["reward"][1].sum().item() == 10.0
    assert bufs["reward"][2].sum().item() == 0.0
