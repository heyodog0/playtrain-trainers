"""Actor loop — runs in its own subprocess.

Pulls a free buffer index off `free_queue`, steps the env `unroll_length`
times using the shared CPU `model` for action inference, writes everything
into the buffer at that index, signals on `full_queue`. Repeat until
`free_queue.get()` returns `None` (shutdown signal).

Matches monobeast.act() with two adaptations:
  - Uses our Environment (gymnasium) instead of torchbeast's (legacy gym).
  - Env factory passed in as a callable so the trainer controls env id /
    seeding / wrappers without this module needing to know about minigrid.
"""
from __future__ import annotations

import gc
import logging
import os
import traceback
from typing import Callable

import torch
from torch import multiprocessing as mp
from torch import nn

from playtrain_trainers.impala.buffers import Buffers
from playtrain_trainers.impala.diagnostics import log_rss, rss_mb, rss_note
from playtrain_trainers.impala.environment import Environment


# Memory diagnostic cadences. RSS is read from /proc each tick; cheap but
# not free, so don't log every single rollout once we're past warm-up.
_MEM_LOG_FIRST_N = 5       # log every rollout for the first N (catch warm-up)
_MEM_LOG_EVERY = 25        # then every Nth rollout (catch slow drift)
_GC_EVERY = 10             # force a cycle collection every N rollouts


def act(
    actor_index: int,
    free_queue: "mp.SimpleQueue",
    full_queue: "mp.SimpleQueue",
    model: nn.Module,
    buffers: Buffers,
    initial_agent_state_buffers: list,
    env_fn: Callable[[int], tuple],
    unroll_length: int,
    chw_transpose: bool = True,
    fixed_env_seed: int | None = None,
) -> None:
    try:
        logging.info("Actor %d starting%s.", actor_index, rss_note())
        # env_fn now returns (env, initial_seed) so cfg.seed propagates
        # through to the env-side RNG. Without this the node_gym backend
        # silently used randomSeed() for every reset, making all cfg.seed
        # values train on the same env distribution.
        gym_env, initial_seed = env_fn(actor_index)
        log_rss("actor", actor_index, phase="env_built")
        env = Environment(gym_env, chw_transpose=chw_transpose,
                          initial_seed=initial_seed,
                          fixed_seed=fixed_env_seed)
        # Surface the env seeding so the run log makes it obvious whether
        # the run is on a fixed instance or the procedural distribution.
        # When fixed_env_seed is set, EVERY episode the actor sees has the
        # same layout (memorize-one-instance, mirrors PPO fixed_env_seed).
        if fixed_env_seed is not None:
            logging.info("Actor %d env seeding: fixed_env_seed=%d "
                         "(every reset uses this seed)",
                         actor_index, fixed_env_seed)
        else:
            logging.info("Actor %d env seeding: initial_seed=%d, "
                         "auto-resets use env's own RNG advance",
                         actor_index, initial_seed)
        env_output = env.initial()
        agent_state = model.initial_state(batch_size=1)
        agent_output, _ = model(env_output, agent_state)
        log_rss("actor", actor_index, phase="first_forward")

        rollouts = 0
        while True:
            index = free_queue.get()
            if index is None:
                break

            # Carry-over: write the LAST step of the previous rollout into
            # slot 0. On the very first rollout this is the initial obs.
            for key in env_output:
                buffers[key][index][0, ...] = env_output[key]
            for key in agent_output:
                buffers[key][index][0, ...] = agent_output[key]
            # Snapshot the recurrent state ENTERING this unroll so the learner
            # can replay it from the same state (monobeast pattern). No-op in
            # feedforward mode where agent_state is an empty tuple.
            for i, tensor in enumerate(agent_state):
                initial_agent_state_buffers[index][i][...] = tensor

            for t in range(unroll_length):
                with torch.no_grad():
                    agent_output, agent_state = model(env_output, agent_state)
                env_output = env.step(agent_output["action"])
                for key in env_output:
                    buffers[key][index][t + 1, ...] = env_output[key]
                for key in agent_output:
                    buffers[key][index][t + 1, ...] = agent_output[key]

            full_queue.put(index)
            rollouts += 1
            # Force cyclic GC. Job 16774848 showed ~60MB/rollout net allocation
            # in actor RSS with stable learner RSS — pointing at refcount
            # cycles in env.step()/info dicts that refcount-only collection
            # leaves behind. gc.collect() is ~5ms on a small heap; cheap.
            if rollouts % _GC_EVERY == 0:
                gc.collect()
            if rollouts <= _MEM_LOG_FIRST_N or rollouts % _MEM_LOG_EVERY == 0:
                log_rss("actor", actor_index, rollouts=rollouts)

        env.close()
        logging.info("Actor %d shutting down cleanly%s.",
                     actor_index, rss_note())
    except KeyboardInterrupt:
        pass
    except Exception as e:  # noqa: BLE001
        logging.error("Actor %d crashed: %s", actor_index, e)
        traceback.print_exc()
        raise
