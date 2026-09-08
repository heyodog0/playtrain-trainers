"""remote_vec + LSTM: the recurrent state lives on the trainer's inference
worker, so these tests pin down the two things that can silently go wrong and
still train (badly):

  1. the per-slot state snapshot is never written, so learn() replays every
     unroll from zeros;
  2. the real `done` row never reaches the forward, so the state is not reset
     at episode boundaries.

Both are invisible end-to-end — the run completes and the loss decreases — so
they are tested by construction instead. With weight sync disabled and a
CONSTANT observation, the state entering each unroll is an exact function of the
done pattern:

  done=True  every step -> the state is zeroed before every cell step, so the
                           state entering each unroll is exactly f(const, 0):
                           identical across unrolls, and SMALL
  done=False every step -> the state advances freely and, under constant input,
                           converges to the cell's fixed point: also identical
                           across unrolls, but a DIFFERENT (larger) value

So "identical within a run" is not the discriminator — both patterns give that.
The discriminator is that the two runs must land on materially different states
(measured ~0.43 vs ~0.90 in norm). If the done row never reached the forward,
the done=True run would converge to the same fixed point as done=False.

The fake env actor is torch-free and byte-identical to the real fleet's
protocol, which is the point: nothing about LSTM support reaches the actor.
"""
from __future__ import annotations

import socket
import struct
import threading
import time

import numpy as np
import torch

from playtrain_trainers.impala import vec_actor
from playtrain_trainers.impala.remote_proto import ACK, HELLO, MAGIC
from playtrain_trainers.impala.train import ImpalaConfig, train

M, OBS, T, A = 4, 16, 8, 6
FRAME_FILL = 7


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _fake_group(port: int, always_done: bool, stop: threading.Event) -> None:
    """One protocol-speaking group sending a CONSTANT frame every step."""
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
        sock.sendall(HELLO.pack(MAGIC, M, OBS, 0))
        ack = sock.recv(ACK.size, socket.MSG_WAITALL)
        assert struct.unpack("<I", ack)[0] == A
        obs = np.full((M, OBS, OBS, 3), FRAME_FILL, dtype=np.uint8).tobytes()
        done_flag = (np.ones if always_done else np.zeros)(M, np.uint8).tobytes()
        tail = (np.zeros(M, np.float32).tobytes()      # reward
                + done_flag                            # done
                + np.zeros(M, np.float32).tobytes()    # episode_return
                + np.zeros(M, np.int32).tobytes())     # episode_step
        act = bytearray(4 * M)
        while not stop.is_set():
            sock.sendall(obs + tail)
            got = 0
            while got < 4 * M:
                n = sock.recv_into(memoryview(act)[got:], 4 * M - got)
                if n == 0:
                    return  # trainer shut down
                got += n
    except (ConnectionError, OSError):
        return


def _run(tmp_path, always_done: bool, monkeypatch):
    """Train briefly over one remote LSTM group; return the state snapshots."""
    captured: list = []
    real = vec_actor.create_vec_state_buffers

    def spy(model, num_envs, num_buffers):
        sb = real(model, num_envs, num_buffers)
        captured.append(sb)
        return sb

    # train() imports this by name at call time, so patching the module
    # attribute is enough to observe the buffers it hands the workers.
    monkeypatch.setattr(vec_actor, "create_vec_state_buffers", spy)

    port = _free_port()
    cfg = ImpalaConfig(
        game="fake", env_backend="playtrain",
        total_steps=T * M * 4,          # 4 learn steps
        batch_size=M, unroll_length=T, num_learner_threads=1,
        obs_shape=(3, OBS, OBS), num_actions=A, features_dim=32,
        net="nature", use_lstm=True,
        inference_mode="remote_vec", vec_workers=1,
        remote_groups_per_worker=1, remote_port_base=port,
        # Never reload weights in the worker: the state map must stay fixed for
        # the invariants above to be exact rather than approximate.
        inference_sync_every=10 ** 9,
        device="cpu", learner_precision="fp32", compile_learner=False,
        eval_every_steps=0, save_every_steps=0, resume="off",
        stats_log_every=1, log_dir=str(tmp_path),
    )
    stop = threading.Event()
    actor = threading.Thread(target=_fake_group,
                             args=(port, always_done, stop), daemon=True)
    actor.start()
    result = train(cfg)
    stop.set()
    assert result["final_step"] >= cfg.total_steps
    assert "total_loss" in result["stats"], "no learn step ran"
    assert captured, "state buffers were never allocated"
    return captured[0]


def _hidden(snapshots):
    """(h tensors) per slot, split into zero and non-zero groups."""
    hs = [s[0] for s in snapshots]
    assert all(h.dim() == 3 and h.shape[1] == M for h in hs), \
        f"expected (layers, M, hidden) snapshots, got {[tuple(h.shape) for h in hs]}"
    zero = [h for h in hs if not h.any()]
    nonzero = [h for h in hs if h.any()]
    return zero, nonzero


def test_lstm_state_is_threaded_and_done_resets_it(tmp_path, monkeypatch):
    """Covers both failure modes in one pair of runs.

    1. non-zero snapshots  -> the recurrent state reaches state_buffers at all
    2. identical within a run -> the state entering each unroll is well-defined
       (not drifting with slot recycling or group bookkeeping)
    3. the two done patterns disagree -> the real done row reaches the forward
    """
    free = _run(tmp_path / "free", always_done=False, monkeypatch=monkeypatch)
    reset = _run(tmp_path / "reset", always_done=True, monkeypatch=monkeypatch)

    _, free_nz = _hidden(free)
    _, reset_nz = _hidden(reset)
    assert free_nz, ("every per-slot state snapshot is zero: the recurrent "
                     "state is not reaching state_buffers")
    assert reset_nz, "no non-zero snapshot in the done=True run"

    for label, nz in (("done=False", free_nz), ("done=True", reset_nz)):
        for other in nz[1:]:
            torch.testing.assert_close(
                nz[0], other, rtol=1e-5, atol=1e-6,
                msg=f"{label}: the state entering successive unrolls differs "
                    "under a constant observation — slot/group state bookkeeping "
                    "is wrong")

    # The load-bearing assertion: with done never set the state runs to the
    # cell's fixed point; with done always set it is pinned one step from zero.
    free_norm = float(free_nz[0].norm())
    reset_norm = float(reset_nz[0].norm())
    assert abs(free_norm - reset_norm) > 0.1 * max(free_norm, reset_norm), (
        f"done=True and done=False produce the same recurrent state "
        f"(norms {reset_norm:.4f} vs {free_norm:.4f}): the done row is not "
        "reaching the forward, so episode boundaries never reset the state")
