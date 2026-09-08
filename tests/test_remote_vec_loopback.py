"""remote_vec loopback: a fake protocol-speaking env actor feeds a REAL
train() over localhost TCP; the run must complete its learn steps. Covers
the wire protocol, zero-copy slot ingest, alignment bookkeeping, the HWC
frame permute in the learner path, and clean shutdown."""
from __future__ import annotations

import socket
import struct
import threading
import time

import numpy as np

from playtrain_trainers.impala.remote_proto import ACK, HELLO, MAGIC
from playtrain_trainers.impala.train import ImpalaConfig, train

M, OBS, T, A = 8, 16, 8, 6


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _fake_group(port: int, seed: int, stop: threading.Event) -> None:
    rng = np.random.default_rng(seed)
    try:
        for _ in range(150):
            try:
                sock = socket.create_connection(("127.0.0.1", port),
                                                timeout=5)
                break
            except OSError:
                time.sleep(0.2)
        else:
            raise ConnectionError("no trainer")
        sock.settimeout(30)
        sock.sendall(HELLO.pack(MAGIC, M, OBS, seed))
        ack = sock.recv(ACK.size, socket.MSG_WAITALL)
        assert struct.unpack("<I", ack)[0] == A
        act = bytearray(4 * M)
        first = True
        while not stop.is_set():
            obs = rng.integers(0, 256, (M, OBS, OBS, 3),
                               dtype=np.uint8).tobytes()
            tail = (rng.standard_normal(M).astype(np.float32).tobytes()
                    + (np.ones(M, np.uint8) if first
                       else np.zeros(M, np.uint8)).tobytes()
                    + np.zeros(M, np.float32).tobytes()
                    + np.zeros(M, np.int32).tobytes())
            first = False
            sock.sendall(obs + tail)
            got = 0
            while got < 4 * M:
                n = sock.recv_into(memoryview(act)[got:], 4 * M - got)
                if n == 0:
                    return  # trainer shut down — clean exit
                got += n
    except (ConnectionError, OSError):
        return


def test_remote_vec_loopback(tmp_path):
    port = _free_port()
    cfg = ImpalaConfig(
        game="fake", env_backend="playtrain",
        total_steps=T * M * 4,  # 4 learn steps
        batch_size=M, unroll_length=T, num_learner_threads=1,
        obs_shape=(3, OBS, OBS), num_actions=A, features_dim=32,
        net="nature", use_lstm=False,
        inference_mode="remote_vec", vec_workers=1,
        remote_groups_per_worker=2, remote_port_base=port,
        device="cpu", learner_precision="fp32", compile_learner=False,
        eval_every_steps=0, save_every_steps=0, resume="off",
        stats_log_every=1, log_dir=str(tmp_path),
    )
    stop = threading.Event()
    actors = [threading.Thread(target=_fake_group, args=(port, 7 + i, stop),
                               daemon=True) for i in range(2)]
    for a in actors:
        a.start()
    result = train(cfg)
    stop.set()
    assert result["final_step"] >= cfg.total_steps
    # Real learn steps happened on remotely-fed rollouts.
    assert "total_loss" in result["stats"]
