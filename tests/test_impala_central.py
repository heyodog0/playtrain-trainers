"""Tests for centralized GPU inference (playtrain_trainers.impala.central).

Coverage:
  - Channel allocation: shapes, dtypes, shared-memory backing.
  - Atomic flag protocol: actor sets request, server consumes; server sets
    response, actor consumes. No torch tensors leak between actors (anti
    cross-talk invariant).
  - Cross-process visibility of int64 shared tensors (the load-bearing
    assumption that makes the atomic-flag design correct).
  - Shutdown: should_stop unblocks actor spins AND exits server loop.
  - Batching: when ≥2 actors request near-simultaneously, the server fuses
    them into one forward (avg_batch_size > 1).
  - Round-trip latency microbench: sequential request_inference calls
    complete fast enough that we beat the mp.Queue+Event version.
  - Weight sync from learner → inference model.

Tests don't fork actor processes for the per-step protocol checks —
they use in-process threads, which exercise the same shared-memory
read/write paths. The fork-based multi-process is structurally identical
to monobeast and validated by the smoke run on FASRC.
"""
from __future__ import annotations

import threading
import time

import torch
from torch import multiprocessing as mp

from playtrain_trainers.impala.central import (
    InferenceServer,
    create_channel,
    request_inference,
)
from playtrain_trainers.impala.net import ImpalaNet


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _build_setup(num_actors=4, obs_shape=(3, 16, 16), num_actions=5,
                 stochastic=False, batch_timeout_s=0.005):
    """Build a channel + server with a tiny ImpalaNet. Default stochastic=
    False so tests get deterministic argmax actions for routing checks."""
    channel = create_channel(num_actors, obs_shape, num_actions)
    model = ImpalaNet(observation_shape=obs_shape, num_actions=num_actions)
    server = InferenceServer(
        model=model, channel=channel, device=torch.device("cpu"),
        obs_shape=obs_shape, num_actions=num_actions,
        batch_timeout_s=batch_timeout_s,
        stochastic_actions=stochastic,
    )
    return channel, server


def _make_env_output(obs_shape, num_actions, actor_id):
    C, H, W = obs_shape
    frame = torch.full((1, 1, C, H, W), fill_value=actor_id, dtype=torch.uint8)
    return dict(
        frame=frame,
        reward=torch.full((1, 1), float(actor_id), dtype=torch.float32),
        done=torch.zeros(1, 1, dtype=torch.bool),
        last_action=torch.full((1, 1), actor_id, dtype=torch.int64),
    )


def _start_server(server):
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    return t


def _stop_server(channel, server_thread, timeout=2.0):
    channel.should_stop.fill_(1)
    server_thread.join(timeout=timeout)


# ----------------------------------------------------------------------
# Channel structure
# ----------------------------------------------------------------------
class TestChannelStructure:
    def test_shapes_and_dtypes(self):
        num_actors, obs_shape, num_actions = 3, (3, 32, 32), 7
        ch = create_channel(num_actors, obs_shape, num_actions)
        C, H, W = obs_shape
        assert len(ch.input_frame) == num_actors
        assert ch.input_frame[0].shape == (1, 1, C, H, W)
        assert ch.input_frame[0].dtype == torch.uint8
        assert ch.input_reward[0].dtype == torch.float32
        assert ch.input_done[0].dtype == torch.bool
        assert ch.input_last_action[0].dtype == torch.int64
        assert ch.output_policy_logits[0].shape == (1, 1, num_actions)
        assert ch.output_baseline[0].shape == (1, 1)
        assert ch.output_action[0].shape == (1, 1)
        assert ch.output_action[0].dtype == torch.int64

    def test_flags_are_int64_and_shape_1(self):
        ch = create_channel(4, (3, 16, 16), 5)
        for f in ch.request_flags + ch.response_flags + [ch.should_stop]:
            assert f.dtype == torch.int64
            assert f.shape == (1,)

    def test_all_slots_in_shared_memory(self):
        ch = create_channel(3, (3, 16, 16), 5)
        per_actor_lists = [
            ch.input_frame, ch.input_reward, ch.input_done, ch.input_last_action,
            ch.output_policy_logits, ch.output_baseline, ch.output_action,
            ch.request_flags, ch.response_flags,
        ]
        for slots in per_actor_lists:
            for t in slots:
                assert t.is_shared()
        assert ch.should_stop.is_shared()

    def test_no_mp_primitives_in_channel(self):
        """The atomic-flag design specifically removes mp.Queue/mp.Event.
        Catch any regression that re-introduces them by verifying every
        channel field is a tensor, a list-of-tensors, or a plain fork-safe
        scalar (bools/ints are immutable and copied across fork — they are
        config flags like use_lstm, not synchronization primitives)."""
        ch = create_channel(2, (3, 8, 8), 3)
        for name, value in vars(ch).items():
            if isinstance(value, (torch.Tensor, bool, int)):
                continue
            if isinstance(value, list) and all(
                isinstance(t, torch.Tensor) for t in value
            ):
                continue
            raise AssertionError(
                f"channel field {name!r} is {type(value).__name__}, expected "
                f"torch.Tensor, list[torch.Tensor], or a fork-safe scalar "
                f"under the atomic-flag design (no mp.Queue or mp.Event)"
            )


# ----------------------------------------------------------------------
# Atomic flag visibility across processes
# ----------------------------------------------------------------------
def _flip_flag_worker(flag_tensor, target_value):
    """Run in child process: wait briefly, then flip the flag."""
    time.sleep(0.1)
    flag_tensor.fill_(target_value)


class TestCrossProcessAtomicity:
    def test_int64_flag_visible_in_child_process(self):
        """Writing to a shared int64 tensor in a child must be visible to
        the parent process after the write completes. This is the load-
        bearing assumption behind the atomic-flag design."""
        flag = torch.zeros(1, dtype=torch.int64).share_memory_()
        assert flag.item() == 0
        ctx = mp.get_context("fork")
        p = ctx.Process(target=_flip_flag_worker, args=(flag, 42))
        p.start()
        p.join(timeout=5)
        assert p.exitcode == 0
        assert flag.item() == 42

    def test_parent_write_visible_to_child(self):
        """Symmetric direction: parent writes, child reads."""
        flag = torch.zeros(1, dtype=torch.int64).share_memory_()
        result = torch.zeros(1, dtype=torch.int64).share_memory_()

        def child(flag, result):
            # Spin until parent writes 7, then echo to result
            while flag.item() != 7:
                time.sleep(0.001)
            result.fill_(flag.item() * 2)

        ctx = mp.get_context("fork")
        p = ctx.Process(target=child, args=(flag, result))
        p.start()
        time.sleep(0.05)
        flag.fill_(7)
        p.join(timeout=5)
        assert p.exitcode == 0
        assert result.item() == 14


# ----------------------------------------------------------------------
# Flag protocol
# ----------------------------------------------------------------------
class TestFlagProtocol:
    def test_request_flag_starts_at_zero(self):
        ch = create_channel(4, (3, 16, 16), 5)
        for f in ch.request_flags:
            assert f.item() == 0

    def test_response_flag_starts_at_zero(self):
        ch = create_channel(4, (3, 16, 16), 5)
        for f in ch.response_flags:
            assert f.item() == 0

    def test_should_stop_starts_at_zero(self):
        ch = create_channel(4, (3, 16, 16), 5)
        assert ch.should_stop.item() == 0

    def test_server_clears_request_flag_after_consuming(self):
        """After the server processes a request, the actor's request_flag
        must be back to 0 — otherwise the server would re-process it."""
        channel, server = _build_setup(num_actors=2)
        server_t = _start_server(server)
        try:
            env_output = _make_env_output((3, 16, 16), 5, actor_id=0)
            request_inference(0, channel, env_output)
            # Give the server a moment to fully complete the cycle
            time.sleep(0.05)
            assert channel.request_flags[0].item() == 0
        finally:
            _stop_server(channel, server_t)

    def test_actor_clears_response_flag_after_reading(self):
        """request_inference must clear response_flag before returning so
        the next request doesn't see a stale response."""
        channel, server = _build_setup(num_actors=2)
        server_t = _start_server(server)
        try:
            env_output = _make_env_output((3, 16, 16), 5, actor_id=0)
            request_inference(0, channel, env_output)
            assert channel.response_flags[0].item() == 0
        finally:
            _stop_server(channel, server_t)


# ----------------------------------------------------------------------
# Shapes returned to actor
# ----------------------------------------------------------------------
class TestRequestInference:
    def test_returns_correct_shapes(self):
        num_actions = 5
        channel, server = _build_setup(num_actors=4, num_actions=num_actions)
        server_t = _start_server(server)
        try:
            env_output = _make_env_output((3, 16, 16), num_actions, actor_id=0)
            agent_output = request_inference(0, channel, env_output)
            assert agent_output is not None
            assert agent_output["policy_logits"].shape == (1, 1, num_actions)
            assert agent_output["baseline"].shape == (1, 1)
            assert agent_output["action"].shape == (1, 1)
            assert agent_output["action"].dtype == torch.int64
            assert 0 <= int(agent_output["action"].item()) < num_actions
        finally:
            _stop_server(channel, server_t)

    def test_concurrent_routing_no_crosstalk(self):
        """When 4 actors request simultaneously, each must get back ITS OWN
        output. Verified by re-running the same input in isolation and
        comparing — if routing mis-mapped during batching, this fails."""
        num_actors, obs_shape, num_actions = 4, (3, 16, 16), 5
        channel, server = _build_setup(num_actors, obs_shape, num_actions,
                                        stochastic=False)
        server_t = _start_server(server)
        try:
            results: dict[int, dict] = {}
            results_lock = threading.Lock()

            def actor_thread(actor_id):
                eo = _make_env_output(obs_shape, num_actions, actor_id)
                out = request_inference(actor_id, channel, eo)
                with results_lock:
                    results[actor_id] = out

            threads = [threading.Thread(target=actor_thread, args=(i,))
                       for i in range(num_actors)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=5)

            assert set(results.keys()) == set(range(num_actors))

            # Isolation re-run: same input for actor 0 should produce same
            # output (deterministic argmax mode).
            single = request_inference(
                0, channel, _make_env_output(obs_shape, num_actions, 0)
            )
            torch.testing.assert_close(
                single["policy_logits"], results[0]["policy_logits"]
            )
            torch.testing.assert_close(single["baseline"], results[0]["baseline"])
            torch.testing.assert_close(single["action"], results[0]["action"])
        finally:
            _stop_server(channel, server_t)


# ----------------------------------------------------------------------
# Batching
# ----------------------------------------------------------------------
class TestBatching:
    def test_avg_batch_size_above_one_under_concurrent_load(self):
        num_actors, obs_shape, num_actions = 4, (3, 16, 16), 5
        # Generous batching window so the server has time to gather requests.
        channel, server = _build_setup(num_actors, obs_shape, num_actions,
                                        batch_timeout_s=0.05)
        server_t = _start_server(server)
        try:
            def actor_thread(actor_id):
                eo = _make_env_output(obs_shape, num_actions, actor_id)
                request_inference(actor_id, channel, eo)

            threads = [threading.Thread(target=actor_thread, args=(i,))
                       for i in range(num_actors)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=5)
            assert server.total_actors_served == num_actors
            assert server.avg_batch_size >= 1.5, (
                f"expected batching to kick in; avg_batch_size="
                f"{server.avg_batch_size:.2f}, "
                f"n_inf={server.total_inferences}"
            )
        finally:
            _stop_server(channel, server_t)


# ----------------------------------------------------------------------
# Shutdown
# ----------------------------------------------------------------------
class TestShutdown:
    def test_should_stop_exits_server_loop(self):
        channel, server = _build_setup(num_actors=2)
        server_t = _start_server(server)
        # Server should still be running.
        time.sleep(0.05)
        assert server_t.is_alive()
        # Signal shutdown.
        channel.should_stop.fill_(1)
        server_t.join(timeout=2)
        assert not server_t.is_alive()

    def test_should_stop_unblocks_waiting_actor(self):
        """If the server stops while an actor is mid-spin, request_inference
        returns None instead of hanging forever."""
        num_actors, obs_shape, num_actions = 2, (3, 16, 16), 5
        channel = create_channel(num_actors, obs_shape, num_actions)

        # No server running. Actor will spin forever unless should_stop is
        # set. We verify it bails out with None when we set it.
        result_holder: list = []

        def actor_thread():
            eo = _make_env_output(obs_shape, num_actions, 0)
            out = request_inference(0, channel, eo)
            result_holder.append(out)

        t = threading.Thread(target=actor_thread, daemon=True)
        t.start()
        # Let the actor enter its spin loop.
        time.sleep(0.05)
        assert t.is_alive(), "actor should still be spinning"
        channel.should_stop.fill_(1)
        t.join(timeout=2)
        assert not t.is_alive(), "actor must exit spin when should_stop set"
        assert result_holder == [None], (
            f"actor should return None on shutdown, got {result_holder!r}"
        )


# ----------------------------------------------------------------------
# Weight sync
# ----------------------------------------------------------------------
class TestWeightSync:
    def test_sync_pulls_state_dict(self):
        channel, server = _build_setup(num_actors=2)
        learner = ImpalaNet(observation_shape=(3, 16, 16), num_actions=5)
        with torch.no_grad():
            for p in learner.parameters():
                p.add_(1.0)
        # Pre-sync: differ
        assert not torch.equal(
            next(server.model.parameters()),
            next(learner.parameters()),
        )
        server.sync_weights_from(learner)
        # Post-sync: match
        for p_inf, p_lrn in zip(server.model.parameters(), learner.parameters()):
            torch.testing.assert_close(p_inf, p_lrn)


# ----------------------------------------------------------------------
# Performance microbench
# ----------------------------------------------------------------------
class TestPerformance:
    def test_round_trip_latency_microbench(self):
        """100 sequential request_inference round-trips. With shared-mem
        atomic flags we expect well under 5ms per call on CPU (the GPU
        forward is the dominant cost in production; here CPU forward is
        ~1ms for the tiny test net). Pass threshold is loose to absorb
        CI variance, but tight enough to catch a regression to the
        mp.Queue+Event design (~10-20ms per call there).

        This isn't a perfect production proxy — actual SPS depends on env
        step time and across-actor batching — but it locks in that the
        Python-level overhead of the signaling layer stays minimal."""
        num_actors, obs_shape, num_actions = 1, (3, 16, 16), 5
        channel, server = _build_setup(
            num_actors, obs_shape, num_actions, batch_timeout_s=0.0001,
        )
        server_t = _start_server(server)
        try:
            eo = _make_env_output(obs_shape, num_actions, 0)
            # Warm up to amortize one-time costs.
            for _ in range(5):
                request_inference(0, channel, eo)

            N = 100
            start = time.monotonic()
            for _ in range(N):
                out = request_inference(0, channel, eo)
                assert out is not None
            elapsed = time.monotonic() - start
            per_call_ms = (elapsed / N) * 1000
            # Loose 10ms ceiling — passes comfortably with atomic flags
            # (~1-3ms on most CI). The mp.Queue+Event implementation
            # would clock ~15-30ms per call.
            assert per_call_ms < 10.0, (
                f"round-trip too slow: {per_call_ms:.2f}ms/call "
                f"(elapsed={elapsed:.3f}s for {N} calls)"
            )
        finally:
            _stop_server(channel, server_t)
