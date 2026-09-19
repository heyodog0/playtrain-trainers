"""Remote-vec rollout worker: SEED-style network-fed groups, one training run.

The distributed complement to vec_actor: each worker process owns a TCP
LISTENER instead of a NativeVecEnv. Remote env actors (tools/
remote_env_actor.py, running on cheap CPU-partition nodes) connect one
socket per env GROUP of batch_size envs; the worker does batched GPU
inference for its groups and fills the standard (T+1, B) shared-memory
slots — the learner cannot tell remote rollouts from local ones. Weight
sync rides the existing seqlock weight_state; GIL sharding is free
(workers are processes). Measured constants behind the design: 45.5k SPS
per remote ACTOR worker (zero-copy sends), ~14-23k SPS per group, fabric
10.3+ GB/s, 1.09M SPS aggregate across 4 probe shards.

Slot/alignment contract is byte-identical to act_vec (infer-then-step,
row 0 = carry whose agent fields learn() trims): on each incoming message
write row r = (env fields from the wire, agent fields = last inference);
on r == T push the slot, pull a fresh one, duplicate row T into row 0,
continue at r = 1. Derivation in the remote-probe server.

FRAMES ARE STORED HWC (create_vec_buffers(frame_hwc=True)) so the reader
recv_into()'s the wire bytes directly into the slot row — zero CPU-side
copies; the learner does a free GPU-side permute after H2D (train.py).

LSTM: recurrent state lives HERE, not on the env actor — the actor is
torch-free and never sees it. Each group owns an (h, c) pair batched over
its M envs; the inference loop concatenates the pending groups' states
along the batch dim in frame-staging order, runs one forward, and splits
the returned state back per group. Episode resets need no special
handling: the real `done` row goes into the forward and ImpalaNet's T==1
path zeroes the incoming state where done (same as central.py, which is
the design this follows).

Per-slot state snapshot: `state_buffers[slot]` must hold the state
ENTERING that unroll, i.e. the state used for the inference on the slot's
row 0 — so it is written, PRE-forward, whenever a group's inference lands
on row 0. That is the same contract act_vec satisfies by snapshotting
agent_state at slot acquisition, and it is what learn() replays from.
"""
from __future__ import annotations

import logging
import queue
import socket
import threading
import time
import traceback

import numpy as np
import torch

from playtrain_trainers.impala.buffers import Buffers
from playtrain_trainers.impala.diagnostics import log_rss, rss_mb, rss_note
from playtrain_trainers.impala.net import ImpalaNet
from playtrain_trainers.impala.remote_proto import (
    ACK, HELLO, MAGIC, obs_bytes, recv_exact, tail_bytes, tune,
)
from playtrain_trainers.impala.vec_actor import maybe_reload_weights


class _Group:
    """One remote env group: its socket, slot cursor, and last agent output.
    Mutated by its reader thread (env/slot side) and the worker's inference
    loop (last_* side) — never concurrently, because the per-group protocol
    is lockstep (the actor can't send row r+1 until we replied to r)."""

    def __init__(self, gid: int, sock: socket.socket, m: int,
                 num_actions: int, init_state: tuple = ()):
        self.gid = gid
        self.sock = sock
        self.m = m
        self.row = 0
        self.slot: int | None = None
        self.rollouts = 0
        self.last_logits = torch.zeros(m, num_actions)
        self.last_baseline = torch.zeros(m)
        self.last_action = torch.zeros(m, dtype=torch.int64)
        self.act_out = np.empty(m, dtype=np.int32)
        # Recurrent state for this group's M envs, (num_layers, M, hidden)
        # per tensor; () in feedforward mode. Touched only by the inference
        # loop (the reader thread hands the group over via `requests`).
        self.state = tuple(t.clone() for t in init_state)


def act_remote(
    worker_index: int,
    free_queue,
    full_queue,
    buffers: Buffers,
    state_buffers: list,
    weight_state: dict,
    remote_spec: dict,
    model_spec: dict,
    unroll_length: int,
    device_str: str,
) -> None:
    """Network-fed rollout worker. remote_spec:
      port (this worker's listen port), groups (max concurrent groups),
      num_envs (M == batch_size), obs_size, infer_bf16, sync_every
      (weight-reload cadence in group-rollouts).
    model_spec: obs_shape (CHW), num_actions, features_dim, net.
    """
    try:
        logging.basicConfig(level=logging.INFO,
                            format="[%(levelname)s %(asctime)s] %(message)s")
        T = unroll_length
        M = int(remote_spec["num_envs"])
        G = int(remote_spec["groups"])
        obs = int(remote_spec["obs_size"])
        num_actions = int(model_spec["num_actions"])
        use_lstm = bool(model_spec.get("use_lstm", False))
        torch.set_num_threads(1)
        device = torch.device(device_str)

        model = ImpalaNet(tuple(model_spec["obs_shape"]), num_actions,
                          features_dim=int(model_spec.get("features_dim", 256)),
                          use_lstm=use_lstm,
                          net=str(model_spec.get("net", "impala")),
                          core=str(model_spec.get("core", "")),
                          fwp_dim=int(model_spec.get("fwp_dim", 128)),
                          fwp_heads=int(model_spec.get("fwp_heads", 8)),
                          fwp_read=str(model_spec.get("fwp_read", "joint")),
                          fwp_error=str(model_spec.get("fwp_error", "joint")),
                          fwp_write=str(model_spec.get("fwp_write", "delta")),
                          fwp_decay=float(model_spec.get("fwp_decay", 0.0)))
        model = model.to(device)
        model.train()  # multinomial sampling
        weight_version = maybe_reload_weights(weight_state, model, -1)

        autocast = (torch.autocast("cuda", torch.bfloat16)
                    if remote_spec.get("infer_bf16", True)
                    and device.type == "cuda"
                    else torch.autocast("cuda", enabled=False))
        stage = torch.empty(G * M, obs, obs, 3, dtype=torch.uint8,
                            device=device)
        # Template zero state; each group clones it on connect. () when FF.
        base_state = tuple(s.to(device)
                           for s in model.initial_state(batch_size=M))
        requests: queue.SimpleQueue = queue.SimpleQueue()
        stop = threading.Event()
        counters = {"decisions": 0, "rollouts": 0}
        synced_rollouts = 0

        def reader(sock: socket.socket, gid: int) -> None:
            try:
                hello = memoryview(bytearray(HELLO.size))
                recv_exact(sock, hello)
                magic, m, o, _seed = HELLO.unpack(hello)
                assert magic == MAGIC and m == M and o == obs, "hello mismatch"
                sock.sendall(ACK.pack(num_actions))
                g = _Group(gid, sock, M, num_actions, base_state)
                tail = memoryview(bytearray(tail_bytes(M)))
                logging.info("worker %d: group %d connected (M=%d)",
                             worker_index, gid, M)
                while not stop.is_set():
                    if g.slot is None:
                        g.slot = free_queue.get()
                        if g.slot is None:
                            return  # shutdown sentinel
                    r = g.row
                    frame_row = buffers["frame"][g.slot][r]
                    recv_exact(sock,
                               memoryview(frame_row.numpy()).cast("B"))
                    recv_exact(sock, tail)
                    sm = np.frombuffer(tail, dtype=np.uint8)
                    buffers["reward"][g.slot][r].copy_(torch.from_numpy(
                        sm[:4 * M].view(np.float32).copy()))
                    buffers["done"][g.slot][r].copy_(torch.from_numpy(
                        sm[4 * M:5 * M].astype(bool)))
                    buffers["episode_return"][g.slot][r].copy_(
                        torch.from_numpy(
                            sm[5 * M:9 * M].view(np.float32).copy()))
                    buffers["episode_step"][g.slot][r].copy_(torch.from_numpy(
                        sm[9 * M:13 * M].view(np.int32).copy()))
                    buffers["policy_logits"][g.slot][r].copy_(g.last_logits)
                    buffers["baseline"][g.slot][r].copy_(g.last_baseline)
                    buffers["action"][g.slot][r].copy_(g.last_action)
                    buffers["last_action"][g.slot][r].copy_(g.last_action)

                    if r == T:
                        new_slot = free_queue.get()
                        if new_slot is None:
                            return
                        for k in buffers:  # carry row BEFORE handoff
                            buffers[k][new_slot][0].copy_(
                                buffers[k][g.slot][T])
                        full_queue.put(g.slot)
                        g.slot = new_slot
                        g.row = 1
                        infer_row = 0
                        g.rollouts += 1
                        counters["rollouts"] += 1  # reload happens in the
                        # inference loop — readers must not mutate the model
                        # mid-forward.
                    else:
                        g.row += 1
                        infer_row = r
                    counters["decisions"] += M
                    requests.put((g, g.slot, infer_row))
            except (ConnectionError, OSError):
                logging.info("worker %d: group %d disconnected",
                             worker_index, gid)
            except Exception:  # noqa: BLE001
                traceback.print_exc()
                stop.set()

        def acceptor() -> None:
            listener = socket.create_server(
                ("0.0.0.0", int(remote_spec["port"])), backlog=G + 4)
            listener.settimeout(1.0)
            logging.info("worker %d listening on :%d (G=%d, M=%d, dev=%s, "
                         "rss=%.0fMB)", worker_index, remote_spec["port"],
                         G, M, device_str, rss_note())
            gid = 0
            # Fail loudly if the fleet never dials in. Without a deadline this
            # loop waits forever on a 1s socket timeout, so a remote_vec config
            # launched with no fleet (wrong host/port, fleet half of the hetjob
            # never started, actors died on import) silently holds the GPU
            # allocation doing nothing until the wall clock kills it. That is the
            # single reason remote_vec cannot simply be made the default
            # inference_mode. Deadline applies only until the FIRST group
            # connects; after that a long-running rollout is expected.
            connect_deadline = time.monotonic() + float(
                remote_spec.get("connect_timeout_s", 300.0))
            while not stop.is_set():
                try:
                    sock, _ = listener.accept()
                except socket.timeout:
                    if gid == 0 and time.monotonic() > connect_deadline:
                        logging.error(
                            "worker %d: no env actor connected within %.0fs on "
                            ":%d — is the fleet half running, and pointed at "
                            "this host/port? Aborting rather than holding the "
                            "GPU idle.", worker_index,
                            float(remote_spec.get("connect_timeout_s", 300.0)),
                            remote_spec["port"])
                        stop.set()
                    continue
                if gid >= G:
                    sock.close()
                    continue
                tune(sock)
                threading.Thread(target=reader, args=(sock, gid),
                                 daemon=True).start()
                gid += 1
            listener.close()

        threading.Thread(target=acceptor, daemon=True).start()

        # Inference loop (main thread): batch pending groups, one forward,
        # reply. Eager bf16 — proven sufficient at 600k system SPS.
        window_s = float(remote_spec.get("batch_window_ms", 1.0)) / 1000
        last_log, d0 = time.perf_counter(), 0
        while True:
            try:
                first = requests.get(timeout=0.5)
            except queue.Empty:
                continue
            pending = [first]
            deadline = time.monotonic() + window_s
            while len(pending) < G:
                try:
                    pending.append(requests.get(
                        timeout=max(0.0, deadline - time.monotonic())))
                except queue.Empty:
                    break
            b = len(pending) * M
            for i, (g, slot, r) in enumerate(pending):
                stage[i * M:(i + 1) * M].copy_(buffers["frame"][slot][r],
                                               non_blocking=True)
                # The state ENTERING this unroll is the one used for the
                # inference on row 0 — snapshot it PRE-forward, which is the
                # contract learn() replays from (act_vec does the same at slot
                # acquisition). No-ops in feedforward mode.
                if r == 0:
                    for j, s in enumerate(g.state):
                        state_buffers[slot][j].copy_(s.detach().cpu())
            if use_lstm:
                # Batch the pending groups' states in frame-staging order, and
                # feed the REAL done row: ImpalaNet's T==1 path zeroes the
                # incoming state where done, so episode resets are handled by
                # the net rather than duplicated here.
                core_state = tuple(
                    torch.cat([g.state[j] for g, _, _ in pending], dim=1)
                    for j in range(len(base_state))
                )
                done_row = torch.cat(
                    [buffers["done"][slot][r] for _, slot, r in pending]
                ).to(device, non_blocking=True).unsqueeze(0)
            else:
                core_state = ()
                done_row = torch.zeros(1, b, dtype=torch.bool, device=device)
            inp = {
                "frame": stage[:b].permute(0, 3, 1, 2).unsqueeze(0),
                "reward": torch.zeros(1, b, device=device),
                "done": done_row,
                "last_action": torch.zeros(1, b, dtype=torch.int64,
                                           device=device),
            }
            with torch.no_grad(), autocast:
                out, new_core_state = model(inp, core_state)
            if use_lstm:
                # Split the advanced state back to its groups, same order.
                splits = [torch.split(s.float(), M, dim=1)
                          for s in new_core_state]
                for i, (g, _, _) in enumerate(pending):
                    g.state = tuple(part[i].contiguous() for part in splits)
            logits = out["policy_logits"].float().cpu()[0]
            baseline = out["baseline"].float().cpu()[0]
            actions = out["action"].cpu()[0]
            for i, (g, slot, r) in enumerate(pending):
                sl = slice(i * M, (i + 1) * M)
                g.last_logits.copy_(logits[sl])
                g.last_baseline.copy_(baseline[sl])
                g.last_action.copy_(actions[sl])
                np.copyto(g.act_out, g.last_action.numpy().astype(np.int32))
                try:
                    g.sock.sendall(g.act_out.tobytes())
                except OSError:
                    pass
            if counters["rollouts"] - synced_rollouts >= int(
                    remote_spec.get("sync_every", 1)):
                synced_rollouts = counters["rollouts"]
                weight_version = maybe_reload_weights(
                    weight_state, model, weight_version)
            now = time.perf_counter()
            if now - last_log > 30:
                log_rss("remote_worker", worker_index)
                logging.info("worker %d: %.0f SPS", worker_index,
                             (counters["decisions"] - d0) / (now - last_log))
                d0, last_log = counters["decisions"], now
    except KeyboardInterrupt:
        pass
    except Exception as e:  # noqa: BLE001
        logging.error("remote worker %d crashed: %s", worker_index, e)
        traceback.print_exc()
        raise
