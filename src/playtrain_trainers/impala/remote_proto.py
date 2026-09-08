"""Wire protocol for remote env actors (SEED-style split).

One persistent TCP connection per env GROUP. After a fixed-size hello,
every message has a FIXED size derived from (M, obs) — no framing, no
serialization library:

  hello  (actor->server):  <IIII  magic, M, obs_px, base_seed
  ack    (server->actor):  <I     num_actions
  step   (actor->server):  obs  M*obs*obs*3 u8 (HWC, zero-copy sendall)
                           tail M*(4+1+4+4)  reward f32|done u8|ep_ret f32|
                                             ep_step i32
  act    (server->actor):  M i32

The first step message after hello carries the RESET frame with reward=0,
done=1 (Environment.initial() convention). last_action never crosses the
wire — the server chose it. Obs-first ordering lets the server recv_into
the rollout slot row directly (zero-copy ingest).

Validated lineage: remote-probe v1->v3 (32.3k -> 162k SPS/shard;
zero-copy actor sends -> 45.5k SPS per actor worker; 4-shard aggregate
1.09M SPS, jobs 34462770/34574273/34575776).
"""
from __future__ import annotations

import socket
import struct

MAGIC = 0x504C5452  # "PLTR"
HELLO = struct.Struct("<IIII")
ACK = struct.Struct("<I")


def obs_bytes(m: int, obs: int) -> int:
    return m * obs * obs * 3


def tail_bytes(m: int) -> int:
    return m * (4 + 1 + 4 + 4)


def recv_exact(sock: socket.socket, view: memoryview) -> None:
    """Fill `view` completely (recv_into: no Python-level byte copies).
    Raises ConnectionError on EOF — the clean-shutdown signal."""
    got, n = 0, len(view)
    while got < n:
        r = sock.recv_into(view[got:], n - got)
        if r == 0:
            raise ConnectionError("peer closed")
        got += r


def tune(sock: socket.socket) -> None:
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    for opt in (socket.SO_RCVBUF, socket.SO_SNDBUF):
        try:
            sock.setsockopt(socket.SOL_SOCKET, opt, 8 * 1024 * 1024)
        except OSError:
            pass
