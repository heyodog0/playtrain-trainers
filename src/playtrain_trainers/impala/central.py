"""Centralized GPU inference for IMPALA actors.

Architecture (matches SEED RL / polybeast's C++ batcher idea, in Python):

  - Each actor process holds NO model. It only steps its env and writes
    rollouts into shared-memory buffers (same as before).
  - For each env step, the actor writes its observation into a per-actor
    shared-memory slot and signals the inference server via an atomic
    flag (shared int64 tensor).
  - One inference thread in the main process scans the flags, batches
    pending requests across actors (with a small batching window), runs
    one batched GPU forward, writes per-actor outputs back to shared
    memory, and sets per-actor response flags.
  - Actor spins on its response flag (with periodic yields), reads the
    action/logits/baseline, and proceeds to env.step().

Why atomic flags instead of mp.Queue + mp.Event:
  mp.Queue.put/get serializes a small int through pickle + lock + pipe
  (~500us-1ms per call). mp.Event.set/wait involves futex syscalls
  (~100us-1ms). Per env step we hit 4-6 such ops × 80 steps × N actors
  per rollout — IPC overhead dominates wall time at small batch sizes.

  Shared int64 tensors give us:
    - Aligned int64 read/write is atomic on x86 (TSO memory model) and
      ARM/AArch64 (load-acquire / store-release semantics for naturally
      aligned int64). On these architectures the ordering "write inputs
      then set flag" is observed correctly by the consumer.
    - ~1-2us per .item() / .fill_() call (just memory access through
      PyTorch's tensor API), no syscalls.
    - 100-500x less IPC overhead than mp.Queue + mp.Event.

  Cost: actor spins burn CPU when no response is pending. We mitigate
  with periodic `time.sleep(0)` to yield to other threads/processes.

Memory leak elimination (verified in job 16792241):
  Per-actor RSS stable at 370MB across 250+ rollouts (vs 6GB plateau in
  shared_cpu mode). Centralizing inference moves the model forward off
  the actor processes entirely, eliminating PyTorch's CPU caching
  allocator drift across N actor procs.
"""
from __future__ import annotations

import dataclasses
import gc
import logging
import threading
import time
import traceback
from typing import Callable

import torch
from torch import multiprocessing as mp
from torch import nn

from playtrain_trainers.impala.buffers import Buffers
from playtrain_trainers.impala.diagnostics import log_rss, rss_mb, rss_note
from playtrain_trainers.impala.environment import Environment


_MEM_LOG_FIRST_N = 5
_MEM_LOG_EVERY = 25
_GC_EVERY = 10

# Actor-side spin tuning. After this many tight-poll iterations with no
# response, yield via time.sleep(0). 1000 iterations ≈ ~1-2ms of polling
# (~1-2us per .item() call), which is the right scale for our ~2-5ms
# expected round-trip in central inference.
_ACTOR_SPIN_BEFORE_YIELD = 1000


@dataclasses.dataclass
class InferenceChannel:
    """Per-actor shared-memory slots + atomic flag signaling.

    All tensors are in shared memory so the inference thread and actor
    processes both observe the same bytes through normal load/store. No
    mp.Queue, no mp.Event — just int64 flags polled by both sides.

    Flag protocol (per actor i):
      request_flags[i]:
        actor sets to 1 after writing input slots (publishes a request)
        server clears to 0 when it consumes the request
      response_flags[i]:
        server sets to 1 after writing output slots (publishes a response)
        actor clears to 0 after reading the response
      should_stop (global):
        main sets to 1 during shutdown; both server loop and actor spin
        loops observe it and exit
    """
    # Per-actor input slots — written by actor, read by server.
    input_frame: list[torch.Tensor]        # per-actor (1, 1, C, H, W) uint8
    input_reward: list[torch.Tensor]       # per-actor (1, 1) float32
    input_done: list[torch.Tensor]         # per-actor (1, 1) bool
    input_last_action: list[torch.Tensor]  # per-actor (1, 1) int64
    # Per-actor output slots — written by server, read by actor.
    output_policy_logits: list[torch.Tensor]  # per-actor (1, 1, A) float32
    output_baseline: list[torch.Tensor]       # per-actor (1, 1) float32
    output_action: list[torch.Tensor]         # per-actor (1, 1) int64
    # Per-actor atomic signaling flags (int64 shared tensors).
    request_flags: list[torch.Tensor]      # per-actor (1,) int64
    response_flags: list[torch.Tensor]     # per-actor (1,) int64
    # Global shutdown signal (1,) int64.
    should_stop: torch.Tensor
    # Recurrent state slots (LSTM only; empty lists in feedforward mode). The
    # server stays STATELESS: the actor ships its current (h, c) in with each
    # request and reads the updated (h, c) back out, exactly mirroring the
    # state-with-data pattern shared_cpu uses. This matches IMPALA/monobeast
    # rather than SEED RL's server-resident state — our channel is a
    # shared-memory memcpy, not the gRPC hop that motivated SEED's choice.
    use_lstm: bool = False
    input_core_h: list[torch.Tensor] = dataclasses.field(default_factory=list)
    input_core_c: list[torch.Tensor] = dataclasses.field(default_factory=list)
    output_core_h: list[torch.Tensor] = dataclasses.field(default_factory=list)
    output_core_c: list[torch.Tensor] = dataclasses.field(default_factory=list)


def create_channel(
    num_actors: int,
    obs_shape: tuple[int, int, int],
    num_actions: int,
    ctx=None,  # kept for backward compat with callers; no mp primitives used
    use_lstm: bool = False,
    features_dim: int = 256,
    num_layers: int = 1,
) -> InferenceChannel:
    """Allocate all shared-memory slots + flags. Call from main process
    BEFORE forking actors so the children inherit the shared mappings.

    When use_lstm=True, also allocate per-actor (h, c) input/output slots
    shaped (num_layers, 1, features_dim) so the actor can thread its recurrent
    state through the server without the server having to keep any state."""
    del ctx  # no mp primitives needed in the atomic-flag design
    C, H, W = obs_shape
    if use_lstm:
        def _state_slots():
            return [torch.zeros(num_layers, 1, features_dim,
                                dtype=torch.float32).share_memory_()
                    for _ in range(num_actors)]
        input_core_h, input_core_c = _state_slots(), _state_slots()
        output_core_h, output_core_c = _state_slots(), _state_slots()
    else:
        input_core_h = input_core_c = output_core_h = output_core_c = []
    return InferenceChannel(
        input_frame=[torch.empty(1, 1, C, H, W, dtype=torch.uint8).share_memory_()
                     for _ in range(num_actors)],
        input_reward=[torch.empty(1, 1, dtype=torch.float32).share_memory_()
                      for _ in range(num_actors)],
        input_done=[torch.empty(1, 1, dtype=torch.bool).share_memory_()
                    for _ in range(num_actors)],
        input_last_action=[torch.empty(1, 1, dtype=torch.int64).share_memory_()
                           for _ in range(num_actors)],
        output_policy_logits=[
            torch.empty(1, 1, num_actions, dtype=torch.float32).share_memory_()
            for _ in range(num_actors)
        ],
        output_baseline=[torch.empty(1, 1, dtype=torch.float32).share_memory_()
                         for _ in range(num_actors)],
        output_action=[torch.empty(1, 1, dtype=torch.int64).share_memory_()
                       for _ in range(num_actors)],
        request_flags=[torch.zeros(1, dtype=torch.int64).share_memory_()
                       for _ in range(num_actors)],
        response_flags=[torch.zeros(1, dtype=torch.int64).share_memory_()
                        for _ in range(num_actors)],
        should_stop=torch.zeros(1, dtype=torch.int64).share_memory_(),
        use_lstm=use_lstm,
        input_core_h=input_core_h,
        input_core_c=input_core_c,
        output_core_h=output_core_h,
        output_core_c=output_core_c,
    )


class InferenceServer:
    """Runs as a thread in the main process. Polls request flags, batches
    pending actors, runs one GPU forward per batch, distributes outputs."""

    def __init__(
        self,
        model: nn.Module,
        channel: InferenceChannel,
        device: torch.device,
        obs_shape: tuple[int, int, int],
        num_actions: int,
        batch_timeout_s: float = 0.001,
        stochastic_actions: bool = True,
    ):
        self.model = model
        self.channel = channel
        self.device = device
        self.obs_shape = obs_shape
        self.num_actions = num_actions
        self.num_actors = len(channel.request_flags)
        # Batching window: after seeing ≥1 pending request, allow more to
        # arrive for this long before launching the forward. Smaller window
        # = lower per-actor latency, bigger window = bigger batches.
        # 1ms is a good default at ~4 actors on MiniGrid.
        self.batch_timeout_s = batch_timeout_s
        self.stochastic_actions = stochastic_actions
        # Stats for diagnostics.
        self.total_inferences = 0
        self.total_actors_served = 0  # sum of batch sizes
        self.empty_scans = 0          # how often we found nothing pending

    @property
    def avg_batch_size(self) -> float:
        return self.total_actors_served / max(self.total_inferences, 1)

    def run(self) -> None:
        try:
            logging.info(
                "InferenceServer starting on %s (stochastic=%s, "
                "batch_window=%.1fms, n_actors=%d).",
                self.device, self.stochastic_actions,
                self.batch_timeout_s * 1000, self.num_actors,
            )
            if self.stochastic_actions:
                self.model.train()
            else:
                self.model.eval()

            while self.channel.should_stop.item() == 0:
                requests = self._collect_batch()
                if not requests:
                    continue
                self._infer_and_distribute(requests)

            logging.info(
                "InferenceServer stopped. n_inferences=%d avg_batch=%.2f "
                "empty_scans=%d",
                self.total_inferences, self.avg_batch_size, self.empty_scans,
            )
        except Exception:  # noqa: BLE001
            logging.error("InferenceServer crashed:")
            traceback.print_exc()
            raise

    def _collect_batch(self) -> list[int]:
        """Scan request flags. Returns the actor_ids of pending requests.
        Claims each by clearing its flag so it's processed exactly once."""
        # First pass — anything already pending?
        requests = []
        for i in range(self.num_actors):
            if self.channel.request_flags[i].item() == 1:
                self.channel.request_flags[i].fill_(0)  # claim
                requests.append(i)

        if not requests:
            # Nothing pending. Sleep briefly so we don't 100%-spin one core
            # in the idle case. 50us is short enough that round-trip latency
            # is dominated by inference itself, not by this poll cadence.
            self.empty_scans += 1
            time.sleep(0.00005)
            return requests

        # We have ≥1 request; allow a small window for more to arrive so we
        # can batch them into one forward.
        deadline = time.monotonic() + self.batch_timeout_s
        while len(requests) < self.num_actors and time.monotonic() < deadline:
            for i in range(self.num_actors):
                if i in requests:
                    continue
                if self.channel.request_flags[i].item() == 1:
                    self.channel.request_flags[i].fill_(0)
                    requests.append(i)
        return requests

    def _infer_and_distribute(self, requests: list[int]) -> None:
        # Stack inputs across batch dim. Each input_frame[a] has shape
        # (1, 1, C, H, W); cat along dim 1 gives (1, B, C, H, W).
        frames = torch.cat(
            [self.channel.input_frame[a] for a in requests], dim=1
        ).to(self.device, non_blocking=True)
        rewards = torch.cat(
            [self.channel.input_reward[a] for a in requests], dim=1
        ).to(self.device, non_blocking=True)
        dones = torch.cat(
            [self.channel.input_done[a] for a in requests], dim=1
        ).to(self.device, non_blocking=True)
        last_actions = torch.cat(
            [self.channel.input_last_action[a] for a in requests], dim=1
        ).to(self.device, non_blocking=True)

        inputs = {
            "frame": frames,
            "reward": rewards,
            "done": dones,
            "last_action": last_actions,
        }

        # Gather each requesting actor's recurrent state (shipped in with the
        # request) into a batched (num_layers, B, hidden) core_state. The net's
        # forward applies the done-reset internally, so the server need not
        # touch episode boundaries — it just threads state in and back out.
        core_state = ()
        if self.channel.use_lstm:
            h = torch.cat([self.channel.input_core_h[a] for a in requests], dim=1)
            c = torch.cat([self.channel.input_core_c[a] for a in requests], dim=1)
            core_state = (h.to(self.device, non_blocking=True),
                          c.to(self.device, non_blocking=True))

        with torch.no_grad():
            out, new_core_state = self.model(inputs, core_state)
        policy_logits = out["policy_logits"].cpu()
        baseline = out["baseline"].cpu()
        actions = out["action"].cpu()
        if self.channel.use_lstm:
            new_h = new_core_state[0].cpu()
            new_c = new_core_state[1].cpu()

        # Write per-actor outputs.
        for i, actor_id in enumerate(requests):
            self.channel.output_policy_logits[actor_id].copy_(
                policy_logits[:, i:i + 1]
            )
            self.channel.output_baseline[actor_id].copy_(baseline[:, i:i + 1])
            self.channel.output_action[actor_id].copy_(actions[:, i:i + 1])
            if self.channel.use_lstm:
                self.channel.output_core_h[actor_id].copy_(new_h[:, i:i + 1])
                self.channel.output_core_c[actor_id].copy_(new_c[:, i:i + 1])
        # Then publish response flags. Writing outputs BEFORE flag avoids
        # any visibility race where actor sees flag=1 but stale outputs
        # (the inputs-then-flag pattern in reverse, on the response side).
        for actor_id in requests:
            self.channel.response_flags[actor_id].fill_(1)

        self.total_inferences += 1
        self.total_actors_served += len(requests)

    def sync_weights_from(self, learner_model: nn.Module) -> None:
        """Called by the learner after each gradient step to update the
        inference-side model with the latest weights."""
        self.model.load_state_dict(learner_model.state_dict())


def request_inference(
    actor_id: int,
    channel: InferenceChannel,
    env_output: dict,
    core_state: tuple = (),
) -> dict | None:
    """Actor-side blocking call. Writes input to shared slots, sets request
    flag, spins on response flag, reads output. Returns agent_output dict.

    In LSTM mode the caller passes its current recurrent state via `core_state`;
    the returned dict carries the updated state under key "core_state". The
    server stays stateless — the state round-trips through the channel.

    Returns None if the channel's should_stop is set while waiting — actor
    should treat this as a shutdown signal and exit its rollout loop.
    """
    # Write inputs FIRST, then set the request flag. On x86 the producer's
    # stores are observed in this order by the consumer (TSO), so the server
    # cannot see request_flag=1 with stale input slots.
    channel.input_frame[actor_id].copy_(env_output["frame"])
    channel.input_reward[actor_id].copy_(env_output["reward"])
    channel.input_done[actor_id].copy_(env_output["done"])
    channel.input_last_action[actor_id].copy_(env_output["last_action"])
    if channel.use_lstm:
        channel.input_core_h[actor_id].copy_(core_state[0])
        channel.input_core_c[actor_id].copy_(core_state[1])
    channel.request_flags[actor_id].fill_(1)

    # Spin-wait for response. Tight poll for the common-case <1ms response,
    # then yield to other threads/processes via time.sleep(0).
    spin = 0
    while channel.response_flags[actor_id].item() == 0:
        if channel.should_stop.item() == 1:
            return None
        spin += 1
        if spin >= _ACTOR_SPIN_BEFORE_YIELD:
            time.sleep(0)
            spin = 0

    # Symmetric ordering: clone outputs (taking a private copy) BEFORE
    # clearing the response flag, so we can't race with the server
    # overwriting slots for the next request.
    out = {
        "policy_logits": channel.output_policy_logits[actor_id].clone(),
        "baseline": channel.output_baseline[actor_id].clone(),
        "action": channel.output_action[actor_id].clone(),
    }
    if channel.use_lstm:
        out["core_state"] = (
            channel.output_core_h[actor_id].clone(),
            channel.output_core_c[actor_id].clone(),
        )
    channel.response_flags[actor_id].fill_(0)
    return out


def act_central(
    actor_index: int,
    free_queue: "mp.SimpleQueue",
    full_queue: "mp.SimpleQueue",
    channel: InferenceChannel,
    buffers: Buffers,
    initial_agent_state_buffers: list,
    initial_core_state: tuple,
    env_fn: Callable[[int], tuple],
    unroll_length: int,
    chw_transpose: bool = True,
    fixed_env_seed: int | None = None,
) -> None:
    """Actor loop using centralized inference via shared-memory atomic flags.

    Mirrors actor.act() exactly — infer-then-step, storing the agent_output
    computed from the PRE-step frame — so both inference modes produce the
    identical slot convention that learn()'s batch[1:]/outputs[:-1] shift
    assumes. (Earlier this loop did step-then-infer, which stored a one-step-
    shifted action/policy_logits and silently misaligned V-trace; see
    tests/test_impala_central_alignment.py.)

    LSTM: the actor owns its recurrent state and threads it through the
    inference channel (server stays stateless), snapshotting the per-unroll
    initial state into initial_agent_state_buffers — the same state-with-data
    pattern actor.act() uses."""
    try:
        logging.info("Actor %d starting [central]%s.",
                     actor_index, rss_note())
        # See note in actor.py: env_fn returns (env, initial_seed) so the
        # node_gym backend actually uses cfg.seed.
        gym_env, initial_seed = env_fn(actor_index)
        log_rss("actor", actor_index, phase="env_built")
        env = Environment(gym_env, chw_transpose=chw_transpose,
                          initial_seed=initial_seed,
                          fixed_seed=fixed_env_seed)
        # Surface env seeding so the run log shows whether the run is on a
        # fixed instance (fixed_env_seed) or the procedural distribution.
        if fixed_env_seed is not None:
            logging.info("Actor %d env seeding: fixed_env_seed=%d "
                         "(every reset uses this seed)",
                         actor_index, fixed_env_seed)
        else:
            logging.info("Actor %d env seeding: initial_seed=%d, "
                         "auto-resets use env's own RNG advance",
                         actor_index, initial_seed)
        env_output = env.initial()
        # Recurrent state to feed the NEXT request. () in feedforward mode.
        agent_state = tuple(t.clone() for t in initial_core_state)
        # Pre-loop request supplies the first carry-over (slot 0). Its returned
        # state is discarded — like act(), the loop recomputes the carry frame
        # from agent_state, so the snapshot stays the state ENTERING that frame.
        agent_output = request_inference(actor_index, channel, env_output,
                                         agent_state)
        if agent_output is None:
            logging.info("Actor %d: server stopped before initial forward.",
                         actor_index)
            return
        agent_output.pop("core_state", None)
        log_rss("actor", actor_index, phase="first_forward")

        rollouts = 0
        while True:
            index = free_queue.get()
            if index is None:
                break

            for key in env_output:
                buffers[key][index][0, ...] = env_output[key]
            for key in agent_output:
                buffers[key][index][0, ...] = agent_output[key]
            # Snapshot the recurrent state ENTERING this unroll (monobeast).
            # No-op in feedforward mode (agent_state is an empty tuple).
            for i, tensor in enumerate(agent_state):
                initial_agent_state_buffers[index][i][...] = tensor

            for t in range(unroll_length):
                # Infer on the CURRENT (pre-step) frame, then step — matching
                # actor.act() so the stored action/policy_logits align to the
                # frame learn() expects.
                agent_output = request_inference(actor_index, channel,
                                                 env_output, agent_state)
                if agent_output is None:
                    logging.info(
                        "Actor %d: server stopped mid-rollout (t=%d). Exiting.",
                        actor_index, t,
                    )
                    return
                new_state = agent_output.pop("core_state", None)
                env_output = env.step(agent_output["action"])
                for key in env_output:
                    buffers[key][index][t + 1, ...] = env_output[key]
                for key in agent_output:
                    buffers[key][index][t + 1, ...] = agent_output[key]
                if new_state is not None:
                    agent_state = new_state

            full_queue.put(index)
            rollouts += 1
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
        logging.error("Actor %d [central] crashed: %s", actor_index, e)
        traceback.print_exc()
        raise
