"""Shared-memory rollout buffers (port of monobeast.create_buffers).

Each "buffer slot" holds one (T+1)-step rollout for one actor — index 0 is
the carry-over from the previous rollout (so V-trace has a bootstrap value),
indices 1..T are the new steps. Slots live in shared memory so actor
processes write to them and the learner reads from them without IPC.
"""
from __future__ import annotations

from typing import Dict, List

import torch


Buffers = Dict[str, List[torch.Tensor]]


def create_buffers(
    obs_shape: tuple,
    num_actions: int,
    unroll_length: int,
    num_buffers: int,
    obs_dtype: torch.dtype = torch.uint8,
) -> Buffers:
    """Allocates `num_buffers` rollout slots in shared memory.

    obs_shape: shape of a single env observation (e.g. (3, 64, 64) for CHW).
    The T+1 dim covers carry-over (index 0) + T new steps (indices 1..T).
    """
    T = unroll_length
    specs = dict(
        frame=dict(size=(T + 1, *obs_shape), dtype=obs_dtype),
        reward=dict(size=(T + 1,), dtype=torch.float32),
        done=dict(size=(T + 1,), dtype=torch.bool),
        episode_return=dict(size=(T + 1,), dtype=torch.float32),
        episode_step=dict(size=(T + 1,), dtype=torch.int32),
        policy_logits=dict(size=(T + 1, num_actions), dtype=torch.float32),
        baseline=dict(size=(T + 1,), dtype=torch.float32),
        last_action=dict(size=(T + 1,), dtype=torch.int64),
        action=dict(size=(T + 1,), dtype=torch.int64),
    )
    buffers: Buffers = {k: [] for k in specs}
    for _ in range(num_buffers):
        for key, spec in specs.items():
            buffers[key].append(torch.empty(**spec).share_memory_())
    return buffers


def create_initial_agent_state_buffers(
    model, num_buffers: int
) -> list[tuple[torch.Tensor, ...]]:
    """Per-slot recurrent-state snapshots (monobeast `initial_agent_state_buffers`).

    The recurrent state is per-unroll, not per-step, so it can't live in the
    `(T+1, ...)` step buffers above — it gets its own structure: one tuple per
    buffer slot. The actor writes the state entering each unroll here; the
    learner reads it back (via train._get_batch) to replay the unroll from the
    exact state the rollout began at, which V-trace requires.

    Each slot's tuple matches ``model.initial_state(batch_size=1)``:
      - LSTM mode   -> (h, c), each a [num_layers, 1, hidden] shared tensor
      - feedforward -> () (empty tuple); the learner replays from () and these
        slots are never written.
    Allocate from the main process BEFORE forking actors so children inherit
    the shared-memory mappings.
    """
    state_buffers: list[tuple[torch.Tensor, ...]] = []
    for _ in range(num_buffers):
        state = model.initial_state(batch_size=1)
        for t in state:
            t.share_memory_()
        state_buffers.append(state)
    return state_buffers
