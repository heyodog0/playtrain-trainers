"""Vectorized rollout worker: NativeVecEnv + in-worker batched GPU inference.

Replaces the per-actor architecture (one env per process + per-step batch-1
inference round-trips through the shared-memory flag channel) for
inference_mode="vec". Each worker process owns:

  - a NativeVecEnv of M envs (QuickJS + native rasterizer, C++ threadpool;
    bit-exact vs the node/V8 production path — see node-gym
    tests/test_native_vec_env.py and native/gate_qjs.sh), and
  - its own GPU copy of ImpalaNet for inference.

Per vector step it runs ONE batched forward for all M envs and ONE
GIL-released vec_step — Python cost is O(1) per M env steps instead of the
old O(10+) per single env step. A rollout fills a whole (T+1, M) buffer slot,
which the learner consumes as one batch (batch_size == M), so no stacking on
the learner side either.

Weight sync: the learner publishes state_dict tensors into shared pinned CPU
memory under a seqlock (odd version = write in progress); workers reload at
rollout boundaries when the version advanced. Staleness is bounded by
(rollout time + publish cadence), which V-trace's off-policy correction is
built for — the same regime as monobeast/SEED.

Alignment contract is identical to actor.act()/act_central(): slot[t] holds
the env output OF step t together with the agent output computed FROM frame
t-1's... no: with the agent output computed from the PRE-step frame; see
tests/test_impala_central_alignment.py. Concretely: infer on the current
frame, step, store (new env output + that agent output) at t+1.

LSTM: the worker owns its recurrent state outright — (h, c) for all M envs
stay resident on the GPU and thread from forward to forward (no channel
round-trip; the state-with-data plumbing of central mode existed only because
its server was stateless). Episode resets happen inside ImpalaNet.forward via
the done mask. Per rollout, the state ENTERING the unroll is snapshotted into
the slot's shared state buffer (the monobeast initial_agent_state contract);
one slot = one batch, so the snapshot IS the learner's initial_agent_state.
"""
from __future__ import annotations

import contextlib
import errno
import fcntl
import logging
import os
import tempfile
import time
import traceback
from typing import Callable

import numpy as np
import torch

from playtrain_trainers.impala.buffers import Buffers
from playtrain_trainers.impala.diagnostics import log_rss, rss_mb, rss_note
from playtrain_trainers.impala.net import ImpalaNet, resolve_core


def create_vec_buffers(
    obs_shape: tuple,
    num_actions: int,
    unroll_length: int,
    num_envs: int,
    num_buffers: int,
    obs_dtype: torch.dtype = torch.uint8,
    frame_hwc: bool = False,
) -> Buffers:
    """(T+1, M, ...) rollout slots in shared memory — the vectorized version
    of buffers.create_buffers, where one slot is a whole learner batch.

    frame_hwc: store frames (T+1, M, H, W, C) instead of CHW — remote_vec
    readers recv wire bytes (HWC) straight into the row, and the learner
    recovers CHW as a free GPU-side permute (train.py batch_and_learn)."""
    T, M = unroll_length, num_envs
    c, h, w = obs_shape
    frame_size = (T + 1, M, h, w, c) if frame_hwc else (T + 1, M, *obs_shape)
    specs = dict(
        frame=dict(size=frame_size, dtype=obs_dtype),
        reward=dict(size=(T + 1, M), dtype=torch.float32),
        done=dict(size=(T + 1, M), dtype=torch.bool),
        episode_return=dict(size=(T + 1, M), dtype=torch.float32),
        episode_step=dict(size=(T + 1, M), dtype=torch.int32),
        policy_logits=dict(size=(T + 1, M, num_actions), dtype=torch.float32),
        baseline=dict(size=(T + 1, M), dtype=torch.float32),
        last_action=dict(size=(T + 1, M), dtype=torch.int64),
        action=dict(size=(T + 1, M), dtype=torch.int64),
    )
    buffers: Buffers = {k: [] for k in specs}
    for _ in range(num_buffers):
        for key, spec in specs.items():
            buffers[key].append(torch.empty(**spec).share_memory_())
    return buffers


def create_vec_state_buffers(model, num_envs: int,
                             num_buffers: int) -> list[tuple[torch.Tensor, ...]]:
    """Per-slot recurrent-state snapshots for vec mode: each slot holds the
    (h, c) ENTERING that unroll, batched over the worker's M envs — shape
    (num_layers, M, hidden) per tensor. () per slot in feedforward mode.
    The vectorized analogue of buffers.create_initial_agent_state_buffers."""
    state_buffers: list[tuple[torch.Tensor, ...]] = []
    for _ in range(num_buffers):
        state = model.initial_state(batch_size=num_envs)
        for t in state:
            t.share_memory_()
        state_buffers.append(state)
    return state_buffers


# ---------------------------------------------------------------------------
# Weight publication (learner -> workers) via shared CPU tensors + seqlock.
# ---------------------------------------------------------------------------

def create_weight_state(model: torch.nn.Module):
    """Shared-memory copies of every state_dict tensor + a seqlock version.

    Allocate in the main process BEFORE forking workers. version semantics:
    even = stable, odd = publish in progress (seqlock). Starts at 0 with the
    initial weights already written, so workers can load immediately.
    """
    version = torch.zeros(1, dtype=torch.int64).share_memory_()
    names, tensors = [], []
    for name, t in model.state_dict().items():
        shared = t.detach().cpu().clone().share_memory_()
        names.append(name)
        tensors.append(shared)
    return {"version": version, "names": names, "tensors": tensors}


def publish_weights(weight_state: dict, model: torch.nn.Module) -> None:
    """Learner-side: copy state_dict into the shared tensors under the seqlock.
    Callers must serialize (train.py holds a lock)."""
    version = weight_state["version"]
    version += 1  # odd: write in progress
    with torch.no_grad():
        for shared, src in zip(weight_state["tensors"],
                               model.state_dict().values()):
            shared.copy_(src)
    version += 1  # even: stable


def maybe_reload_weights(weight_state: dict, model: torch.nn.Module,
                         last_version: int) -> int:
    """Worker-side: if a newer stable version is published, load it.
    Returns the version now loaded (== last_version if unchanged)."""
    version = weight_state["version"]
    v = int(version.item())
    if v == last_version or (v & 1):  # unchanged, or publish in progress
        return last_version
    while True:
        sd = {n: t for n, t in zip(weight_state["names"],
                                   weight_state["tensors"])}
        model.load_state_dict(sd)
        v2 = int(version.item())
        if v2 == v and not (v2 & 1):
            return v2  # stable read
        # Publisher raced us; converge on the newest stable version.
        while v2 & 1:
            v2 = int(version.item())
        v = v2


@contextlib.contextmanager
def _capture_lock(device):
    """Serialise CUDA-graph capture across the worker processes on one GPU.

    Capture needs exclusive stream ownership. Every vec worker builds its
    inference callable at the same moment, so under MPS a dozen clients attempt
    capture on one device concurrently and collide:

        cudaErrorDevicesUnavailable          (loser of the race)
        cudaErrorStreamCaptureInvalidated    (capture torn down mid-flight)

    The second is the dangerous one -- it poisons the process's CUDA context, so
    the eager fallback is dead too and the run emits no stats while exiting
    cleanly with ``stats: {}``. Observed twice on miner at learner_gpus=2.

    A flock keyed on the device serialises capture without threading a lock
    through the spawn path. Held for the capture only (~1s), so workers still
    start in parallel.
    """
    idx = device.index if device.index is not None else 0
    job = os.environ.get("SLURM_JOB_ID", "local")
    path = os.path.join(tempfile.gettempdir(), f"pt_cudagraph_{job}_{idx}.lock")
    fh = None
    try:
        fh = open(path, "w")
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        yield
    except OSError as e:
        if e.errno not in (errno.ENOSYS, errno.EACCES, errno.EPERM):
            raise
        yield          # locking unavailable on this fs: proceed unserialised
    finally:
        if fh is not None:
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            except Exception:  # noqa: BLE001
                pass
            fh.close()


def _cuda_context_alive(device) -> bool:
    """Is this process's CUDA context still usable after a failed capture?"""
    try:
        torch.cuda.synchronize(device)
        torch.zeros(1, device=device).add_(1)
        return True
    except Exception:  # noqa: BLE001
        return False


def _make_infer(model, env_spec: dict, model_spec: dict, M: int, device):
    """Build the worker's inference callable: (env_output, state) ->
    (agent_output_cpu, new_state).

    infer_bf16: autocast the forward (halves co-located GPU cost; stored
    logits stay self-consistent with the sampled actions — behavior/learner
    drift is V-trace's job). infer_graphs: capture the whole forward as one
    CUDA graph — replay is a single launch instead of per-op dispatch
    (~10-15% less inference GPU time). Weight reloads remain visible to the
    graph because load_state_dict copies parameters IN PLACE (same storage
    the capture recorded). Falls back to eager on any capture failure.
    """
    import contextlib

    use_bf16 = (bool(env_spec.get("infer_bf16", False))
                and device.type == "cuda")
    use_graphs = (bool(env_spec.get("infer_graphs", False))
                  and device.type == "cuda")
    # Whether the model carries recurrent state at all — true for the LSTM and
    # for the fast-weight cores, false only for the Markov feedforward core.
    has_core_state = resolve_core(
        model_spec.get("core"), bool(model_spec.get("use_lstm", False))
    ) != "ff"
    obs_shape = tuple(model_spec["obs_shape"])

    def _ac():
        return (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if use_bf16 else contextlib.nullcontext())

    def eager_infer(eo: dict, state: tuple):
        with torch.no_grad(), _ac():
            out, new_state = model(
                {k: eo[k].to(device, non_blocking=True)
                 for k in ("frame", "reward", "done", "last_action")}, state)
        return ({"policy_logits": out["policy_logits"].float().cpu(),
                 "baseline": out["baseline"].float().cpu(),
                 "action": out["action"].cpu()},
                new_state)

    if use_graphs and os.environ.get("CUDA_MPS_PIPE_DIRECTORY"):
        # MPS and CUDA-graph capture are mutually exclusive here. Under MPS the
        # clients share the server's context and stream capture is
        # context-wide, so concurrent workers cannot each capture: the first
        # gets cudaErrorDevicesUnavailable and the rest
        # cudaErrorStreamCaptureInvalidated. Verified by contrast -- runs with
        # graphs active (logs/3503284*.out) have no MPS; every MPS run fails
        # capture on all workers.
        #
        # MPS wins that trade by a mile: 1.9x (586,442 -> 1,084,599 on breakout)
        # against the graph's ~10-15% of inference time, which is ~2% end-to-end
        # when the learner binds. So skip capture rather than retry into a wall
        # -- four attempts with backoff cost ~9s per worker at startup.
        logging.info("CUDA graphs disabled: MPS is active (capture is "
                     "context-wide under MPS). Eager inference.")
        return eager_infer

    if not use_graphs:
        return eager_infer

    def _capture_once():
        static_in = {
            "frame": torch.zeros(1, M, *obs_shape, dtype=torch.uint8,
                                 device=device),
            "reward": torch.zeros(1, M, device=device),
            "done": torch.zeros(1, M, dtype=torch.bool, device=device),
            "last_action": torch.zeros(1, M, dtype=torch.int64,
                                       device=device),
        }
        static_state_in = (tuple(t.to(device).clone()
                                 for t in model.initial_state(M))
                           if has_core_state else ())
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.no_grad(), _ac():
            for _ in range(3):
                model(static_in, static_state_in)
        torch.cuda.current_stream().wait_stream(s)
        graph = torch.cuda.CUDAGraph()
        with _capture_lock(device):
            with torch.no_grad(), _ac(), torch.cuda.graph(graph):
                static_out, static_state_out = model(static_in, static_state_in)

        def graphed_infer(eo: dict, state: tuple):
            static_in["frame"].copy_(eo["frame"], non_blocking=True)
            static_in["reward"].copy_(eo["reward"], non_blocking=True)
            static_in["done"].copy_(eo["done"], non_blocking=True)
            static_in["last_action"].copy_(eo["last_action"],
                                           non_blocking=True)
            for dst, src in zip(static_state_in, state):
                dst.copy_(src, non_blocking=True)
            graph.replay()
            outs = {"policy_logits": static_out["policy_logits"].float().cpu(),
                    "baseline": static_out["baseline"].float().cpu(),
                    "action": static_out["action"].cpu()}
            new_state = tuple(t.clone() for t in static_state_out)
            return outs, new_state

        logging.info("worker inference: CUDA graph active (bf16=%s, M=%d)",
                     use_bf16, M)
        return graphed_infer

    # Retry: the collision is transient and serialised capture almost always
    # wins the retry. Silently dropping to eager would leave SOME workers ~10-15%
    # slower than others, which is harder to notice than an outright failure.
    last_err = None
    for attempt in range(4):
        try:
            fn = _capture_once()
            if attempt:
                logging.info("CUDA-graph capture succeeded on attempt %d",
                             attempt + 1)
            return fn
        except Exception as e:  # noqa: BLE001
            last_err = e
            if not _cuda_context_alive(device):
                raise RuntimeError(
                    f"CUDA-graph capture invalidated this worker's CUDA context "
                    f"({e}); the eager fallback cannot run either. Under MPS "
                    f"this is concurrent capture on cuda:{device.index} -- see "
                    f"_capture_lock.") from e
            logging.warning("CUDA-graph capture attempt %d/4 failed (%s)",
                            attempt + 1, e)
            time.sleep(1.5 * (attempt + 1))
    logging.warning("CUDA-graph capture failed after 4 attempts (%s); "
                    "eager inference.", last_err)
    return eager_infer


# ---------------------------------------------------------------------------
# The worker loop.
# ---------------------------------------------------------------------------

def act_vec(
    worker_index: int,
    free_queue,
    full_queue,
    buffers: Buffers,
    state_buffers: list,
    weight_state: dict,
    env_spec: dict,
    model_spec: dict,
    unroll_length: int,
    device_str: str,
) -> None:
    """Rollout worker. env_spec:
      game_path, num_envs, frame_skip, max_steps (FRAMES), obs_size,
      frame_stack, env_threads, seed_mode ("formula"|"fixed"|"pool"),
      seed_pool (list|None), fixed_seed (int|None), base_seed (int)
    model_spec: obs_shape (CHW, incl. frame_stack channels), num_actions,
      features_dim, use_lstm.
    state_buffers: per-slot (h, c) snapshot tensors from
      create_vec_state_buffers ((,) tuples in feedforward mode).
    """
    try:
        # Spawned children don't inherit the parent's logging config.
        logging.basicConfig(level=logging.INFO,
                            format="[%(levelname)s %(asctime)s] %(message)s")

        T = unroll_length
        M = int(env_spec["num_envs"])
        S = int(env_spec.get("frame_stack", 1))
        obs_shape = tuple(model_spec["obs_shape"])
        num_actions = int(model_spec["num_actions"])
        # One CPU thread for torch: the heavy lifting is GPU inference + the
        # env host's own C++ threadpool; torch intra-op threads would only
        # fight the env threads for cores.
        torch.set_num_threads(1)
        device = torch.device(device_str)

        logging.info("VecWorker %d starting: M=%d T=%d device=%s%s",
                     worker_index, M, T, device_str, rss_note())

        # vec_backend="envpool" swaps ONLY the env implementation (real
        # ProcGen/Atari via EnvPool's C++ batcher) behind the same worker —
        # the 1-to-1 layer-2 comparison row. Seed policies are native-only.
        if env_spec.get("vec_backend", "native") == "envpool":
            from playtrain_trainers.impala.envpool_vec import EnvPoolVec
            env = EnvPoolVec(env_spec["game_path"], num_envs=M,
                             num_threads=int(env_spec.get("env_threads", 0)),
                             env_kwargs=env_spec.get("envpool_kwargs") or {})
            init_seeds = None
        else:
            from playtrain.runtime.native_vec_env import NativeVecEnv
            env = NativeVecEnv(
                env_spec["game_path"], num_envs=M,
                obs_size=int(env_spec.get("obs_size", 64)),
                max_steps=int(env_spec["max_steps"]),
                num_threads=int(env_spec.get("env_threads", 0)),
                autoreset=True,
                frame_skip=int(env_spec.get("frame_skip", 1)),
                render_skip=bool(env_spec.get("render_skip", False)),
            )

            # Seeding. Initial reset seeds + the autoreset policy for every
            # episode after the first (mirrors Environment._reset_seed +
            # SeedSetWrapper semantics).
            base_seed = int(env_spec.get("base_seed", 0))
            rng = np.random.default_rng(base_seed)
            mode = env_spec.get("seed_mode", "formula")
            if mode == "fixed":
                fixed = int(env_spec["fixed_seed"])
                env.set_autoreset_seeds("fixed", fixed_seed=fixed)
                init_seeds = np.full(M, fixed, dtype=np.int32)
            elif mode == "pool":
                pool = np.asarray(env_spec["seed_pool"], dtype=np.int32)
                env.set_autoreset_seeds("pool", pool=pool, rng_seed=base_seed)
                init_seeds = rng.choice(pool, size=M).astype(np.int32)
            else:
                init_seeds = (base_seed + np.arange(M)).astype(np.int32)

        model = ImpalaNet(obs_shape, num_actions,
                          features_dim=int(model_spec.get("features_dim", 256)),
                          use_lstm=bool(model_spec.get("use_lstm", False)),
                          use_popart=bool(model_spec.get("use_popart", False)),
                          net=str(model_spec.get("net", "impala")),
                          core=str(model_spec.get("core", "")),
                          fwp_dim=int(model_spec.get("fwp_dim", 128)),
                          fwp_heads=int(model_spec.get("fwp_heads", 8)),
                          fwp_read=str(model_spec.get("fwp_read", "joint")),
                          fwp_error=str(model_spec.get("fwp_error", "joint")),
                          fwp_write=str(model_spec.get("fwp_write", "delta")),
                          fwp_decay=float(model_spec.get("fwp_decay", 0.0)),
                          fwp_w_o_gain=float(model_spec.get("fwp_w_o_gain", 0.1)),
                          fwp_read_norm=bool(model_spec.get("fwp_read_norm", False)),
                          fwp_w_p_init=float(model_spec.get("fwp_w_p_init", 0.0)))
        model = model.to(device)
        model.train()  # multinomial sampling for training rollouts
        weight_version = maybe_reload_weights(weight_state, model, -1)
        # Recurrent state for all M envs, resident on the device. () in
        # feedforward mode, so every state operation below no-ops.
        agent_state = tuple(t.to(device)
                            for t in model.initial_state(batch_size=M))

        # Rolling frame-stack window, torch CPU (M, 3*S, H, W). Newest frame
        # occupies the LAST 3 channels (matches playtrain.runtime.env deque order); on
        # reset all S slots are filled with the reset frame (env.py reset()).
        H, W = obs_shape[1], obs_shape[2]
        stack = torch.zeros(M, 3 * S, H, W, dtype=torch.uint8)

        def push_frames(obs_hwc: np.ndarray, done_mask: np.ndarray | None):
            """obs (M, H, W, 3) uint8 -> update `stack` in place."""
            chw = torch.from_numpy(obs_hwc).permute(0, 3, 1, 2)  # view+copy below
            if S > 1:
                stack[:, : 3 * (S - 1)] = stack[:, 3:]
                stack[:, 3 * (S - 1):] = chw
                if done_mask is not None and done_mask.any():
                    idx = torch.from_numpy(np.nonzero(done_mask)[0])
                    stack[idx] = chw[idx].repeat(1, S, 1, 1)
            else:
                stack.copy_(chw)

        # Episode bookkeeping (Environment semantics: on the done step,
        # episode_return/step report the COMPLETED episode; counters restart
        # for the next step).
        ep_return = torch.zeros(M)
        ep_step = torch.zeros(M, dtype=torch.int32)

        obs = env.reset(seeds=init_seeds)
        push_frames(obs, None)

        # env_output for the CURRENT state, (1, M, ...) time-major like
        # Environment.initial(): done=True, reward 0, last_action 0.
        # frame is a VIEW of `stack`, not a clone: the slot write below is the
        # one copy per step. Safe because the H2D in infer() stages
        # synchronously (stack is pageable) and every buffer write happens
        # before push_frames next mutates stack.
        env_output = dict(
            frame=stack.unsqueeze(0),
            reward=torch.zeros(1, M),
            done=torch.ones(1, M, dtype=torch.bool),
            episode_return=ep_return.clone().unsqueeze(0),
            episode_step=ep_step.clone().unsqueeze(0),
            last_action=torch.zeros(1, M, dtype=torch.int64),
        )

        infer = _make_infer(model, env_spec, model_spec, M, device)

        # Pre-loop inference supplies slot 0's carry-over agent_output. Its
        # returned state is DISCARDED — the loop re-infers on the carry frame
        # from agent_state, so the per-unroll snapshot below stays the state
        # ENTERING that frame (same contract as act()/act_central()).
        agent_output, _ = infer(env_output, agent_state)

        act_i32 = np.empty(M, dtype=np.int32)
        rollouts = 0
        while True:
            index = free_queue.get()
            if index is None:
                break

            for key in env_output:
                buffers[key][index][0, ...] = env_output[key][0]
            for key in agent_output:
                buffers[key][index][0, ...] = agent_output[key][0]
            # Snapshot the recurrent state ENTERING this unroll into the
            # slot's shared buffer — the learner replays frames 0..T from it.
            for i, t_ in enumerate(agent_state):
                state_buffers[index][i].copy_(t_.detach().cpu())

            for t in range(T):
                # Infer on the CURRENT frame, then step (alignment contract).
                agent_output, new_state = infer(env_output, agent_state)
                actions = agent_output["action"][0]  # (M,) int64
                np.copyto(act_i32, actions.numpy().astype(np.int32))
                obs, rew, term, trunc, _ = env.step(act_i32)
                done_np = (term | trunc)
                rew_t = torch.from_numpy(rew.copy())

                # Bookkeeping BEFORE zeroing: done rows report the completed
                # episode's totals at this timestep.
                ep_step += 1
                ep_return += rew_t
                episode_return = ep_return.clone()
                episode_step = ep_step.clone()
                done_t = torch.from_numpy(done_np.copy())
                if done_np.any():
                    idx = torch.from_numpy(np.nonzero(done_np)[0])
                    ep_return[idx] = 0.0
                    ep_step[idx] = 0

                push_frames(obs, done_np)
                env_output = dict(
                    frame=stack.unsqueeze(0),
                    reward=rew_t.unsqueeze(0),
                    done=done_t.unsqueeze(0),
                    episode_return=episode_return.unsqueeze(0),
                    episode_step=episode_step.unsqueeze(0),
                    last_action=actions.unsqueeze(0),
                )
                for key in env_output:
                    buffers[key][index][t + 1, ...] = env_output[key][0]
                for key in agent_output:
                    buffers[key][index][t + 1, ...] = agent_output[key][0]
                agent_state = new_state

            full_queue.put(index)
            rollouts += 1
            # Refresh weights at rollout boundaries (bounded staleness; the
            # behavior policy stays fixed within an unroll, like SEED's
            # batched inference between syncs).
            weight_version = maybe_reload_weights(weight_state, model,
                                                  weight_version)
            if rollouts <= 5 or rollouts % 50 == 0:
                log_rss("vec_worker", worker_index, rollouts=rollouts)

        env.close()
        logging.info("VecWorker %d shutting down cleanly%s.",
                     worker_index, rss_note())
    except KeyboardInterrupt:
        pass
    except Exception as e:  # noqa: BLE001
        logging.error("VecWorker %d crashed: %s", worker_index, e)
        traceback.print_exc()
        raise


def act_vec_db(
    worker_index: int,
    free_queue,
    full_queue,
    buffers: Buffers,
    state_buffers: list,
    weight_state: dict,
    env_spec: dict,
    model_spec: dict,
    unroll_length: int,
    device_str: str,
) -> None:
    """Double-buffered rollout worker (Sample-Factory-style): the worker runs
    2x batch_size envs as two GROUPS on one shared C++ threadpool
    (PingPongVecEnv). While group A's envs step in C++, Python/GPU builds
    group B's batch and runs its inference — the two costs hide behind each
    other instead of alternating. Each group fills its OWN slot stream, so
    the learner contract (one (T+1, B) slot = one batch, infer-then-step
    alignment, LSTM snapshot semantics) is byte-identical to act_vec's.
    """
    try:
        # Spawned children don't inherit the parent's logging config.
        logging.basicConfig(level=logging.INFO,
                            format="[%(levelname)s %(asctime)s] %(message)s")
        from playtrain.runtime.native_vec_env import PingPongVecEnv

        T = unroll_length
        B = int(env_spec["num_envs"])   # per GROUP == batch_size
        S = int(env_spec.get("frame_stack", 1))
        obs_shape = tuple(model_spec["obs_shape"])
        num_actions = int(model_spec["num_actions"])
        torch.set_num_threads(1)
        device = torch.device(device_str)

        logging.info("VecWorker %d starting [double-buffer]: 2x%d envs T=%d "
                     "device=%s%s", worker_index, B, T, device_str,
                     rss_note())

        env = PingPongVecEnv(
            env_spec["game_path"], group_size=B,
            obs_size=int(env_spec.get("obs_size", 64)),
            max_steps=int(env_spec["max_steps"]),
            num_threads=int(env_spec.get("env_threads", 0)),
            frame_skip=int(env_spec.get("frame_skip", 1)),
            render_skip=bool(env_spec.get("render_skip", False)),
        )

        base_seed = int(env_spec.get("base_seed", 0))
        rng = np.random.default_rng(base_seed)
        mode = env_spec.get("seed_mode", "formula")
        N = 2 * B
        if mode == "fixed":
            fixed = int(env_spec["fixed_seed"])
            env.set_autoreset_seeds("fixed", fixed_seed=fixed)
            init_seeds = np.full(N, fixed, dtype=np.int32)
        elif mode == "pool":
            pool = np.asarray(env_spec["seed_pool"], dtype=np.int32)
            env.set_autoreset_seeds("pool", pool=pool, rng_seed=base_seed)
            init_seeds = rng.choice(pool, size=N).astype(np.int32)
        else:
            init_seeds = (base_seed + np.arange(N)).astype(np.int32)

        model = ImpalaNet(obs_shape, num_actions,
                          features_dim=int(model_spec.get("features_dim", 256)),
                          use_lstm=bool(model_spec.get("use_lstm", False)),
                          use_popart=bool(model_spec.get("use_popart", False)),
                          net=str(model_spec.get("net", "impala")),
                          core=str(model_spec.get("core", "")),
                          fwp_dim=int(model_spec.get("fwp_dim", 128)),
                          fwp_heads=int(model_spec.get("fwp_heads", 8)),
                          fwp_read=str(model_spec.get("fwp_read", "joint")),
                          fwp_error=str(model_spec.get("fwp_error", "joint")),
                          fwp_write=str(model_spec.get("fwp_write", "delta")),
                          fwp_decay=float(model_spec.get("fwp_decay", 0.0)),
                          fwp_w_o_gain=float(model_spec.get("fwp_w_o_gain", 0.1)),
                          fwp_read_norm=bool(model_spec.get("fwp_read_norm", False)),
                          fwp_w_p_init=float(model_spec.get("fwp_w_p_init", 0.0)))
        model = model.to(device)
        model.train()
        weight_version = maybe_reload_weights(weight_state, model, -1)

        H, W = obs_shape[1], obs_shape[2]

        infer = _make_infer(model, env_spec, model_spec, B, device)

        obs_all = env.reset(seeds=init_seeds)  # (2B, H, W, 3) view

        class Half:
            """Per-group rollout cursor + bookkeeping (mirrors act_vec)."""

            def __init__(self, g: int):
                self.g = g
                self.stack = torch.zeros(B, 3 * S, H, W, dtype=torch.uint8)
                self.ep_return = torch.zeros(B)
                self.ep_step = torch.zeros(B, dtype=torch.int32)
                self.agent_state = tuple(t.to(device)
                                         for t in model.initial_state(B))
                self.act_i64 = torch.zeros(B, dtype=torch.int64)
                self.slot = None
                self.t = 0
                self.push(np.ascontiguousarray(obs_all[g * B:(g + 1) * B]),
                          None)
                # frame is a VIEW of self.stack (see act_vec): write_row does
                # the one copy per step, before push() next mutates the stack.
                self.env_output = dict(
                    frame=self.stack.unsqueeze(0),
                    reward=torch.zeros(1, B),
                    done=torch.ones(1, B, dtype=torch.bool),
                    episode_return=self.ep_return.clone().unsqueeze(0),
                    episode_step=self.ep_step.clone().unsqueeze(0),
                    last_action=torch.zeros(1, B, dtype=torch.int64),
                )
                # Pre-loop carry inference; returned state discarded (the
                # first in-loop infer re-runs from the snapshot — same
                # contract as act()/act_central()/act_vec()).
                self.agent_output, _ = infer(self.env_output,
                                             self.agent_state)

            def push(self, obs_hwc: np.ndarray, done_mask):
                chw = torch.from_numpy(obs_hwc).permute(0, 3, 1, 2)
                if S > 1:
                    self.stack[:, : 3 * (S - 1)] = self.stack[:, 3:]
                    self.stack[:, 3 * (S - 1):] = chw
                    if done_mask is not None and done_mask.any():
                        idx = torch.from_numpy(np.nonzero(done_mask)[0])
                        self.stack[idx] = chw[idx].repeat(1, S, 1, 1)
                else:
                    self.stack.copy_(chw)

            def write_row(self, row: int) -> None:
                for key in self.env_output:
                    buffers[key][self.slot][row, ...] = self.env_output[key][0]
                for key in self.agent_output:
                    buffers[key][self.slot][row, ...] = self.agent_output[key][0]

            def open_slot(self) -> bool:
                index = free_queue.get()
                if index is None:
                    return False
                self.slot = index
                self.t = 0
                self.write_row(0)
                for i, t_ in enumerate(self.agent_state):
                    state_buffers[index][i].copy_(t_.detach().cpu())
                return True

            def infer_and_send(self) -> None:
                self.agent_output, new_state = infer(self.env_output,
                                                     self.agent_state)
                self.act_i64 = self.agent_output["action"][0]
                env.send(self.g, self.act_i64.numpy().astype(np.int32))
                self.agent_state = new_state

            def absorb(self) -> None:
                """wait() the group and build the post-step env_output."""
                obs, rew, term, trunc = env.wait(self.g)
                done_np = term | trunc
                rew_t = torch.from_numpy(rew.copy())
                self.ep_step += 1
                self.ep_return += rew_t
                episode_return = self.ep_return.clone()
                episode_step = self.ep_step.clone()
                done_t = torch.from_numpy(done_np.copy())
                if done_np.any():
                    idx = torch.from_numpy(np.nonzero(done_np)[0])
                    self.ep_return[idx] = 0.0
                    self.ep_step[idx] = 0
                self.push(np.ascontiguousarray(obs), done_np)
                self.env_output = dict(
                    frame=self.stack.unsqueeze(0),
                    reward=rew_t.unsqueeze(0),
                    done=done_t.unsqueeze(0),
                    episode_return=episode_return.unsqueeze(0),
                    episode_step=episode_step.unsqueeze(0),
                    last_action=self.act_i64.unsqueeze(0),
                )

        halves = [Half(0), Half(1)]
        for h in halves:
            if not h.open_slot():
                env.close()
                return
            h.infer_and_send()

        rollouts = 0
        running = True
        while running:
            for h in halves:
                h.absorb()
                h.t += 1
                h.write_row(h.t)
                if h.t == T:
                    full_queue.put(h.slot)
                    rollouts += 1
                    if not h.open_slot():
                        running = False
                        break
                    if h.g == 0:
                        weight_version = maybe_reload_weights(
                            weight_state, model, weight_version)
                        if rollouts <= 4 or rollouts % 50 == 0:
                            log_rss("vec_worker_db", worker_index,
                                    rollouts=rollouts)
                h.infer_and_send()

        env.close()
        logging.info("VecWorker %d [double-buffer] shutting down%s.",
                     worker_index, rss_note())
    except KeyboardInterrupt:
        pass
    except Exception as e:  # noqa: BLE001
        logging.error("VecWorker %d [double-buffer] crashed: %s",
                      worker_index, e)
        traceback.print_exc()
        raise
