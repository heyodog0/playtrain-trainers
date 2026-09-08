"""CPU-node env-actor fleet for remote_vec training — torch-free.

Each worker process owns one PingPongVecEnv (2 groups x M envs on a shared
C++ threadpool) and 2 TCP connections to a remote_vec trainer worker. Per
group step: wait(g) -> zero-copy obs send -> recv actions -> send(g, ...).
The two groups alternate so the network+inference round trip hides behind
the other group's env stepping. Protocol + bookkeeping semantics:
playtrain_trainers.impala.remote_proto / vec_actor's Half.absorb.

Measured: ~45.5k SPS per worker process (bigfish, group 256, zero-copy);
lean workers (2-3 env threads) pack ~24 to a 92-core node for fast games.

    python tools/remote_env_actor.py --server trainerhost --port-base 23000 \
        --trainer-workers 4 --workers 24 --game .../bigfish.js \
        --group-size 256 --env-threads 3

Workers round-robin across trainer ports. Multi-node: pass
--worker-offset SLURM_NODEID*workers so nodes cover distinct ports.
Exits cleanly when the trainer closes connections, or after --seconds.
"""
from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import socket
import time
import traceback

import numpy as np

from playtrain_trainers.impala.remote_proto import (
    ACK, HELLO, MAGIC, recv_exact, tune,
)


def _connect(host: str, port: int, retries: int = 300) -> socket.socket:
    for _ in range(retries):
        try:
            s = socket.create_connection((host, port), timeout=10)
            s.settimeout(None)
            tune(s)
            return s
        except OSError:
            time.sleep(2.0)
    raise ConnectionError(f"could not reach {host}:{port}")


class Group:
    def __init__(self, g: int, m: int, obs: int, host: str, port: int,
                 base_seed: int):
        self.g = g
        self.m = m
        self.sock = _connect(host, port)
        self.sock.sendall(HELLO.pack(MAGIC, m, obs, base_seed))
        ack = memoryview(bytearray(ACK.size))
        recv_exact(self.sock, ack)
        (self.num_actions,) = ACK.unpack(ack)
        self.ep_return = np.zeros(m, dtype=np.float32)
        self.ep_step = np.zeros(m, dtype=np.int32)
        self.tail = bytearray(13 * m)
        self.act_buf = memoryview(bytearray(4 * m))
        self.sent = 0

    def _send(self, obs_hwc: np.ndarray, rew, done, ep_ret, ep_step) -> None:
        m = self.m
        self.sock.sendall(memoryview(obs_hwc).cast("B"))  # zero-copy
        t = self.tail
        t[:4 * m] = rew.tobytes()
        t[4 * m:5 * m] = done.tobytes()
        t[5 * m:9 * m] = ep_ret.tobytes()
        t[9 * m:13 * m] = ep_step.tobytes()
        self.sock.sendall(t)
        self.sent += 1

    def send_initial(self, obs_hwc: np.ndarray) -> None:
        m = self.m
        self._send(obs_hwc, np.zeros(m, np.float32), np.ones(m, np.uint8),
                   np.zeros(m, np.float32), np.zeros(m, np.int32))

    def send_step(self, obs_hwc: np.ndarray, rew: np.ndarray,
                  done: np.ndarray) -> None:
        self.ep_step += 1
        self.ep_return += rew
        ep_ret_out = self.ep_return.copy()
        ep_step_out = self.ep_step.copy()
        if done.any():
            idx = np.nonzero(done)[0]
            self.ep_return[idx] = 0.0
            self.ep_step[idx] = 0
        self._send(obs_hwc, rew.astype(np.float32), done.astype(np.uint8),
                   ep_ret_out, ep_step_out)

    def recv_actions(self) -> np.ndarray:
        recv_exact(self.sock, self.act_buf)
        return np.frombuffer(self.act_buf, dtype=np.int32)


def worker(idx: int, args) -> None:
    logging.basicConfig(level=logging.INFO,
                        format=f"[env-actor-{idx} %(asctime)s] %(message)s")
    try:
        from playtrain.runtime.native_vec_env import PingPongVecEnv
        m = args.group_size
        # worker_offset makes the fleet MULTI-NODE. idx is local to this
        # process's node, so without an offset every node's workers map onto
        # the same first `workers` trainer ports: an 8-node fleet with 3
        # workers each fed trainer workers 0-2 and left 3-11 with no actor,
        # which aborts after 300s (job 37731308).
        #
        # Each worker opens 2 groups (below), and the trainer wants
        # remote_groups_per_worker=4 per port, so exactly 2 workers must land
        # on each port: total workers across the fleet == 2 x trainer_workers.
        port = args.port_base + (args.worker_offset + idx) % args.trainer_workers
        env = PingPongVecEnv(
            args.game, group_size=m, obs_size=args.obs,
            max_steps=args.max_steps, num_threads=args.env_threads,
            frame_skip=args.frame_skip, render_skip=True,
        )
        base = args.base_seed + (args.worker_offset + idx) * 10_000
        obs_all = env.reset(seeds=(base + np.arange(2 * m)).astype(np.int32))
        groups = [Group(g, m, args.obs, args.server, port, base + g)
                  for g in range(2)]
        logging.info("connected to :%d (2 groups x %d envs, %d threads)",
                     port, m, args.env_threads)
        for h in groups:
            h.send_initial(np.ascontiguousarray(obs_all[h.g * m:(h.g + 1) * m]))
            env.send(h.g, h.recv_actions())

        t0 = time.perf_counter()
        last_log = t0
        while time.perf_counter() - t0 < args.seconds:
            for h in groups:
                obs, rew, term, trunc = env.wait(h.g)
                h.send_step(np.ascontiguousarray(obs), rew, term | trunc)
                env.send(h.g, h.recv_actions())
            now = time.perf_counter()
            if now - last_log > 30:
                rate = sum(h.sent for h in groups) * m / (now - t0)
                logging.info("%.0f env-steps/s", rate)
                last_log = now
        env.close()
    except ConnectionError as e:
        logging.info("trainer closed (%s) — exiting cleanly", e)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        raise


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--server", required=True, help="trainer hostname")
    p.add_argument("--port-base", type=int, default=23000)
    p.add_argument("--trainer-workers", type=int, required=True,
                   help="trainer's vec_workers (ports port_base..+N-1)")
    p.add_argument("--workers", type=int, default=24)
    p.add_argument("--game", required=True)
    p.add_argument("--group-size", type=int, default=256)
    p.add_argument("--obs", type=int, default=64)
    p.add_argument("--env-threads", type=int, default=3)
    p.add_argument("--frame-skip", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--base-seed", type=int, default=0)
    p.add_argument("--worker-offset", type=int, default=0,
                   help="global index of this node's first worker "
                        "(SLURM_NODEID * --workers). 0 for a single-node fleet.")
    p.add_argument("--seconds", type=float, default=100000.0)
    args = p.parse_args()

    procs = [mp.Process(target=worker, args=(i, args)) for i in
             range(args.workers)]
    for pr in procs:
        pr.start()
    for pr in procs:
        pr.join()


if __name__ == "__main__":
    main()
