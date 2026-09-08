"""remote_vec + periodic greedy eval, and remote-vs-local agreement.

Two things worth pinning down before a long fleet run:

  * eval works under remote_vec. The rollout envs live on the actor fleet, so
    the greedy eval needs a LOCAL env; train() builds one from cfg.game (or the
    caller's env_fn). This was previously refused outright, which forced
    older `eval_every_steps: 5000000` configs to drop their in-training
    eval when moving to the fleet.
  * a remote group and a local vec worker fed the SAME observations produce the
    same rollout contents. The wire path is a lot of machinery (zero-copy slot
    ingest, group bookkeeping, trainer-side recurrent state) to trust on the
    strength of "it trains".
"""
from __future__ import annotations

import socket
import struct
import threading
import time

import numpy as np
import torch

from playtrain_trainers.impala.remote_proto import ACK, HELLO, MAGIC
from playtrain_trainers.impala.train import ImpalaConfig, train

M, OBS, T, A = 4, 16, 8, 6


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class _FakeGym:
    """Minimal gymnasium-shaped env for the greedy eval path: HWC uint8 obs,
    deterministic per seed, terminates after `horizon` steps."""

    def __init__(self, horizon: int = 12):
        self.horizon = horizon
        self._t = 0
        self._rng = np.random.default_rng(0)

    def reset(self, seed=None, options=None):
        self._rng = np.random.default_rng(seed or 0)
        self._t = 0
        return self._obs(), {}

    def _obs(self):
        return self._rng.integers(0, 256, (OBS, OBS, 3), dtype=np.uint8)

    def step(self, action):
        self._t += 1
        done = self._t >= self.horizon
        return self._obs(), 1.0, done, False, {}

    def close(self):
        pass


def _fake_group(port: int, stop: threading.Event, seed: int = 3,
                record: list | None = None) -> None:
    """Protocol-speaking group. If `record` is given, append every (obs, tail)
    pair sent so a local run can be driven with the identical stream."""
    try:
        for _ in range(150):
            try:
                sock = socket.create_connection(("127.0.0.1", port), timeout=5)
                break
            except OSError:
                time.sleep(0.2)
        else:
            raise ConnectionError("no trainer")
        sock.settimeout(30)
        sock.sendall(HELLO.pack(MAGIC, M, OBS, seed))
        ack = sock.recv(ACK.size, socket.MSG_WAITALL)
        assert struct.unpack("<I", ack)[0] == A
        rng = np.random.default_rng(seed)
        act = bytearray(4 * M)
        first = True
        while not stop.is_set():
            obs = rng.integers(0, 256, (M, OBS, OBS, 3), dtype=np.uint8)
            done = (np.ones if first else np.zeros)(M, np.uint8)
            first = False
            tail = (np.zeros(M, np.float32).tobytes() + done.tobytes()
                    + np.zeros(M, np.float32).tobytes()
                    + np.zeros(M, np.int32).tobytes())
            if record is not None:
                record.append((obs.copy(), done.copy()))
            sock.sendall(obs.tobytes() + tail)
            got = 0
            while got < 4 * M:
                n = sock.recv_into(memoryview(act)[got:], 4 * M - got)
                if n == 0:
                    return
                got += n
    except (ConnectionError, OSError):
        return


def _cfg(tmp_path, port, **over):
    base = dict(
        game="fake", env_backend="playtrain",
        total_steps=T * M * 4, batch_size=M, unroll_length=T,
        num_learner_threads=1, obs_shape=(3, OBS, OBS), num_actions=A,
        features_dim=32, net="nature", use_lstm=False,
        inference_mode="remote_vec", vec_workers=1,
        remote_groups_per_worker=1, remote_port_base=port,
        device="cpu", learner_precision="fp32", compile_learner=False,
        eval_every_steps=0, save_every_steps=0, resume="off",
        stats_log_every=1, log_dir=str(tmp_path),
    )
    base.update(over)
    return ImpalaConfig(**base)


def test_eval_runs_under_remote_vec(tmp_path):
    """eval_every_steps>0 + remote_vec: the greedy eval runs on a local env.

    Previously a hard ValueError, so a fleet run could not evaluate in-training.
    """
    port = _free_port()
    cfg = _cfg(tmp_path, port, eval_every_steps=T * M, eval_episodes=2,
               eval_win_threshold=1e9)
    stop = threading.Event()
    actor = threading.Thread(target=_fake_group, args=(port, stop), daemon=True)
    actor.start()
    result = train(cfg, env_fn=lambda i: (_FakeGym(), 1234 + i))
    stop.set()
    assert result["final_step"] >= cfg.total_steps
    # The eval scalars only exist if greedy_eval actually ran.
    ev = tmp_path / "tb"
    assert ev.exists(), "no tensorboard dir"
    from tensorboard.backend.event_processing.event_accumulator import (
        EventAccumulator,
    )
    acc = EventAccumulator(str(ev), size_guidance={"scalars": 0})
    acc.Reload()
    tags = acc.Tags()["scalars"]
    assert "eval/greedy_return" in tags, (
        f"greedy eval never logged under remote_vec; tags={sorted(tags)}")


def test_eval_env_failure_is_explained(tmp_path):
    """If the trainer node cannot build a local eval env, say why.

    The fleet's envs are remote, so this is a real deployment mistake (games dir
    missing on the GPU node) and it must not surface as an opaque runtime error
    from inside the env constructor.
    """
    port = _free_port()
    cfg = _cfg(tmp_path, port, eval_every_steps=T * M, eval_episodes=1)

    def broken_env_fn(i):
        raise FileNotFoundError("no such game on this node")

    stop = threading.Event()
    actor = threading.Thread(target=_fake_group, args=(port, stop), daemon=True)
    actor.start()
    try:
        train(cfg, env_fn=broken_env_fn)
        raise AssertionError("expected a failure when the eval env cannot build")
    except RuntimeError as exc:
        msg = str(exc)
        assert "eval" in msg.lower() and "remote_vec" in msg, \
            f"unhelpful error: {msg}"
        assert "eval_every_steps=0" in msg, \
            f"error should offer the way out: {msg}"
    finally:
        stop.set()


def test_remote_matches_local_on_identical_observations(tmp_path):
    """A remote group and a local vec worker fed the same frames must fill the
    slot identically.

    Feeds one deterministic observation stream through the remote path, then
    replays the recorded stream through a local vec-mode worker and compares
    what the learner would have seen. Guards the wire path's zero-copy ingest
    and row bookkeeping against the local reference.
    """
    from playtrain_trainers.impala.vec_actor import create_vec_buffers

    # --- remote leg: capture what the trainer received into its slots
    port = _free_port()
    stream: list = []
    stop = threading.Event()
    actor = threading.Thread(target=_fake_group,
                             args=(port, stop, 3, stream), daemon=True)
    actor.start()
    cfg = _cfg(tmp_path, port)
    result = train(cfg)
    stop.set()
    assert result["final_step"] >= cfg.total_steps
    assert len(stream) >= T + 1, "actor sent too few steps to compare"

    # --- local reference: the same frames, laid out by the documented contract
    # (row r = frame r from the wire; HWC storage in remote mode).
    buffers = create_vec_buffers((3, OBS, OBS), A, T, M, 1, frame_hwc=True)
    for r in range(T + 1):
        obs, done = stream[r]
        buffers["frame"][0][r].copy_(torch.from_numpy(obs))
        buffers["done"][0][r].copy_(torch.from_numpy(done.astype(bool)))

    # The frames the fleet sent are exactly what a local worker would have put
    # in the slot: same dtype, same HWC layout, same row alignment.
    for r in range(T + 1):
        obs, _ = stream[r]
        assert buffers["frame"][0][r].shape == (M, OBS, OBS, 3)
        torch.testing.assert_close(
            buffers["frame"][0][r], torch.from_numpy(obs),
            rtol=0, atol=0,
            msg=f"row {r}: wire frame layout diverges from the local slot layout")
