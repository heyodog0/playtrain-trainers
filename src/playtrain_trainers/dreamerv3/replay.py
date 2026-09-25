"""Uniform sequence replay — a port of ``embodied/core/replay.py`` and ``chunk.py``.

What the frozen config asks for: windows of ``consec_train * batch_length +
replay_context`` = **65** steps, sampled uniformly from a chunked store of capacity
5e6, with ``online: True`` so the freshest windows are guaranteed to be trained on,
and with the per-step replay-context entries written back after each gradient step.

The observable semantics, all of which the tests pin down:

1. A window becomes sampleable only once ``length`` steps exist from its start, so
   every sampled window is complete. That is why training starts at 1024 steps.
2. Windows cross chunk boundaries through the ``succ`` links; a window is not
   confined to one chunk.
3. ``online`` pushes every ``length``-th inserted item onto a queue that ``sample``
   drains first, so recent experience cannot be missed by an unlucky uniform draw.
4. Eviction is FIFO on items, not on chunks; a chunk is dropped when its last
   referencing item goes.
5. Every sampled window is annotated ``is_first[:, 0] = True`` whatever the stored
   value was, and ``is_last`` is OR-ed with the next step's ``is_first`` so an
   abandoned episode still ends.
6. ``update`` writes values back into the chunks at their ``stepid`` positions.

Not ported (D-017): the npz save/load path, the thread pool and the RW/refs locks
(our driver is single-threaded, MISSION rule 7), the ``Prioritized``/``Recency``/
``Mixture`` selectors and the ``priority`` half of ``update`` (unreachable at
``fracs.uniform: 1.0``, D-022), and the ``limiters`` wait (the caller checks the
buffer size itself, exactly as ``run/train.py``'s ``trainfn`` does).
"""

from __future__ import annotations

import uuid as uuidlib
from collections import defaultdict, deque
from typing import Any

import numpy as np

STEPID_BYTES = 20  # 16-byte chunk uuid + 4-byte big-endian index, as upstream


class Chunk:
    """``chunk.Chunk``: a fixed-size column store with a link to its successor."""

    def __init__(self, size: int = 1024):
        self.uuid = uuidlib.uuid4().bytes
        self.succ: bytes | None = None
        self.length = 0
        self.size = size
        self.data: dict[str, np.ndarray] | None = None

    def append(self, step: dict[str, np.ndarray]) -> None:
        assert self.length < self.size
        if not self.data:
            self.data = {
                k: np.empty((self.size, *v.shape), v.dtype) for k, v in step.items()
            }
        for key, value in step.items():
            self.data[key][self.length] = value
        self.length += 1

    def update(self, index: int, length: int, mapping: dict[str, np.ndarray]) -> None:
        assert 0 <= index <= self.length, (index, self.length)
        assert 0 <= index + length <= self.length, (index, length, self.length)
        for key, value in mapping.items():
            self.data[key][index : index + length] = value

    def slice(self, index: int, length: int) -> dict[str, np.ndarray]:
        assert 0 <= index and index + length <= self.length, (index, length, self.length)
        return {k: v[index : index + length] for k, v in self.data.items()}

    @property
    def nbytes(self) -> int:
        return sum(x.nbytes for x in self.data.values()) if self.data else 0


class Uniform:
    """``selectors.Uniform``: an O(1) swap-with-last index set over item ids."""

    def __init__(self, seed: int = 0):
        self.indices: dict[int, int] = {}
        self.keys: list[int] = []
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.keys)

    def __call__(self) -> int:
        return self.keys[int(self.rng.integers(0, len(self.keys)))]

    def __setitem__(self, key: int, stepids: Any) -> None:
        self.indices[key] = len(self.keys)
        self.keys.append(key)

    def __delitem__(self, key: int) -> None:
        assert 2 <= len(self), len(self)
        index = self.indices.pop(key)
        last = self.keys.pop()
        if index != len(self.keys):
            self.keys[index] = last
            self.indices[last] = index


class Replay:
    """``core.Replay`` for the single-worker, uniform, in-memory case."""

    def __init__(
        self,
        length: int,
        capacity: int | None = None,
        chunksize: int = 1024,
        online: bool = False,
        seed: int = 0,
    ):
        self.length = length
        self.capacity = int(capacity) if capacity else None
        self.chunksize = chunksize
        self.sampler = Uniform(seed)

        self.chunks: dict[bytes, Chunk] = {}
        self.refs: dict[bytes, int] = {}
        self.items: dict[int, tuple[bytes, int]] = {}
        self.fifo: deque[int] = deque()
        self.itemid = 0

        self.current: dict[int, tuple[bytes, int]] = {}
        self.streams: dict[int, deque] = defaultdict(deque)

        self.online = online
        self.lengths: dict[int, int] = defaultdict(int)
        self.queue: deque[tuple[bytes, int]] = deque()

        self.metrics = {"samples": 0, "inserts": 0, "updates": 0}

    def __len__(self) -> int:
        return len(self.items)

    # ------------------------------------------------------------------
    def add(self, step: dict[str, Any], worker: int = 0) -> None:
        step = {k: np.asarray(v) for k, v in step.items() if not k.startswith("log/")}

        if worker not in self.current:
            chunk = Chunk(self.chunksize)
            self.refs[chunk.uuid] = 1
            self.chunks[chunk.uuid] = chunk
            self.current[worker] = (chunk.uuid, 0)

        chunkid, index = self.current[worker]
        step["stepid"] = np.frombuffer(
            chunkid + index.to_bytes(4, "big"), np.uint8
        ).copy()
        stream = self.streams[worker]
        chunk = self.chunks[chunkid]
        assert chunk.length == index, (chunk.length, index)
        chunk.append(step)
        stream.append((chunkid, index))
        self.refs[chunkid] += 1

        index += 1
        if index < chunk.size:
            self.current[worker] = (chunkid, index)
        else:
            self._complete(chunk, worker)

        if len(stream) >= self.length:
            self.metrics["inserts"] += 1
            chunkid, index = stream.popleft()
            self._insert(chunkid, index)
            # Every `length`-th item also goes on the online queue, so the freshest
            # experience is trained on at least once instead of waiting for a uniform
            # draw to find it. The counter is read BEFORE this add increments it.
            if self.online and self.lengths[worker] % self.length == 0:
                self.queue.append((chunkid, index))

        if self.online:
            self.lengths[worker] += 1

    def sample(self, batch: int, mode: str = "train") -> dict[str, np.ndarray]:
        assert len(self.sampler) or self.queue, "replay buffer is empty"
        seqs = [self._sample(mode) for _ in range(batch)]
        data = self._assemble_batch(seqs)
        return self._annotate_batch(data)

    def update(self, data: dict[str, np.ndarray]) -> None:
        """Write per-step values back into the store, addressed by ``stepid``.

        This is how the replay-context latents are refreshed: after a gradient step
        the agent hands back the entries it computed for those exact steps, so the
        next time the window is drawn it starts from a latent the CURRENT world model
        produced rather than the one from whenever the step was collected.
        """
        data = dict(data)
        stepid = data.pop("stepid")
        data.pop("priority", None)  # D-022: no prioritized replay at the frozen config
        assert stepid.ndim == 3, stepid.shape
        self.metrics["updates"] += int(np.prod(stepid.shape[:-1]))
        if not data:
            return
        for i, row in enumerate(stepid):
            raw = np.asarray(row[0], np.uint8).tobytes()
            chunkid = raw[:-4]
            index = int.from_bytes(raw[-4:], "big")
            values = {k: v[i] for k, v in data.items()}
            try:
                self._setseq(chunkid, index, values)
            except KeyError:
                pass  # the chunk was evicted while the batch was in flight

    # ------------------------------------------------------------------
    def _sample(self, mode: str) -> dict[str, list[np.ndarray]]:
        assert mode in ("train", "report", "eval"), mode
        if mode == "train":
            self.metrics["samples"] += 1
        while True:
            try:
                if self.online and self.queue and mode == "train":
                    chunkid, index = self.queue.popleft()
                else:
                    chunkid, index = self.items[self.sampler()]
                return self._getseq(chunkid, index, concat=False)
            except KeyError:
                continue

    def _insert(self, chunkid: bytes, index: int) -> None:
        while self.capacity and len(self.items) >= self.capacity:
            self._remove()
        itemid = self.itemid
        self.itemid += 1
        self.items[itemid] = (chunkid, index)
        self.sampler[itemid] = None
        self.fifo.append(itemid)

    def _remove(self) -> None:
        itemid = self.fifo.popleft()
        del self.sampler[itemid]
        chunkid, index = self.items.pop(itemid)
        self.refs[chunkid] -= 1
        if self.refs[chunkid] < 1:
            del self.refs[chunkid]
            chunk = self.chunks.pop(chunkid)
            if chunk.succ in self.refs:
                self.refs[chunk.succ] -= 1

    def _complete(self, chunk: Chunk, worker: int) -> Chunk:
        succ = Chunk(self.chunksize)
        self.refs[chunk.uuid] -= 1
        self.refs[succ.uuid] = 2
        self.chunks[succ.uuid] = succ
        self.current[worker] = (succ.uuid, 0)
        chunk.succ = succ.uuid
        return succ

    def _getseq(
        self, chunkid: bytes, index: int, concat: bool = True
    ) -> dict[str, Any]:
        """``length`` steps from ``index``, following ``succ`` across chunks."""
        chunk = self.chunks[chunkid]
        available = chunk.length - index
        if available >= self.length:
            seq = chunk.slice(index, self.length)
            return seq if concat else {k: [v] for k, v in seq.items()}
        parts = [chunk.slice(index, available)]
        remaining = self.length - available
        while remaining > 0:
            chunk = self.chunks[chunk.succ]
            used = min(remaining, chunk.length)
            # An item is only inserted once `length` steps exist after it, so the
            # successor always has some. An AssertionError, not a KeyError: the
            # caller retries on KeyError and this condition would spin forever.
            assert used > 0, "successor chunk is empty"
            parts.append(chunk.slice(0, used))
            remaining -= used
        seq = {k: [p[k] for p in parts] for k in parts[0].keys()}
        return {k: np.concatenate(v, 0) for k, v in seq.items()} if concat else seq

    def _setseq(self, chunkid: bytes, index: int, values: dict[str, np.ndarray]) -> None:
        length = len(next(iter(values.values())))
        chunk = self.chunks[chunkid]
        available = chunk.length - index
        if available >= length:
            chunk.update(index, length, values)
            return
        part = {k: v[:available] for k, v in values.items()}
        values = {k: v[available:] for k, v in values.items()}
        chunk.update(index, available, part)
        remaining = length - available
        while remaining > 0:
            chunk = self.chunks[chunk.succ]
            used = min(remaining, chunk.length)
            chunk.update(0, used, {k: v[:used] for k, v in values.items()})
            values = {k: v[used:] for k, v in values.items()}
            remaining -= used

    def _assemble_batch(self, seqs: list[dict[str, list[np.ndarray]]]) -> dict[str, np.ndarray]:
        shape = (len(seqs), self.length)
        data = {
            key: np.empty((*shape, *parts[0].shape[1:]), parts[0].dtype)
            for key, parts in seqs[0].items()
        }
        for n, seq in enumerate(seqs):
            dt = 0
            for p in range(len(seq["stepid"])):
                partlen = len(seq["stepid"][p])
                for k in data:
                    data[k][n, dt : dt + partlen] = seq[k][p]
                dt += partlen
        return data

    def _annotate_batch(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """``_annotate_batch``: force ``is_first`` at the window start, repair
        ``is_last``.

        The forced ``is_first`` means a sampled window ALWAYS looks like the start of
        an episode to the sequence model, whatever was actually stored there. With
        ``replay_context`` that flag is overridden again by the stored latent
        (``_apply_replay_context``); without it, it is what resets the carry.
        """
        data = dict(data)
        if "is_first" in data:
            data["is_first"] = data["is_first"].copy()
            data["is_first"][:, 0] = True
            if "is_last" in data:
                next_is_first = np.roll(data["is_first"], shift=-1, axis=1)
                next_is_first[:, -1] = False
                data["is_last"] = data["is_last"] | next_is_first
        return data

    def stats(self) -> dict[str, float]:
        m = self.metrics
        stats = {
            "items": len(self.items),
            "chunks": len(self.chunks),
            "inserts": m["inserts"],
            "samples": m["samples"],
            "updates": m["updates"],
            "ram_gb": sum(c.nbytes for c in self.chunks.values()) / (1024**3),
            "replay_ratio": (self.length * m["samples"] / m["inserts"]) if m["inserts"] else float("nan"),
        }
        for key in self.metrics:
            self.metrics[key] = 0
        return stats


# ----------------------------------------------------------------------
# The Consec stream and the replay-context split
# ----------------------------------------------------------------------
def consec_chunk(batch: dict[str, np.ndarray], index: int, length: int, prefix: int) -> dict:
    """``streams.Consec.__next__`` for one step of the consec loop.

    At ``consec_train: 1`` there is exactly one chunk per sampled window and it is
    the whole 65 steps, tagged ``consec = 0``.
    """
    start = index * length
    stop = start + length + prefix
    available = batch["is_first"].shape[-1]
    assert length * 1 + prefix <= available, (length, prefix, available)
    out = {k: v[:, start:stop] for k, v in batch.items()}
    out["consec"] = np.full(out["is_first"].shape, index, np.int32)
    return out


def apply_replay_context(
    data: dict[str, np.ndarray],
    entry_keys: tuple[str, ...] = ("deter", "stoch"),
    context: int = 1,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    """``agent._apply_replay_context`` for ``consec_train == 1``.

    Splits a 65-step window into the leading context step and the 64 trained steps::

        carry    = the stored latent entries at step K - 1      (the context step)
        obs      = data[:, K:]                                  (64 steps)
        prevact  = data['action'][:, K - 1 : -1]                (the action INTO each)
        stepid   = data['stepid'][:, K:]

    The action alignment is the part that is easy to get wrong: step ``t`` is trained
    with the action that PRODUCED it, which is stored one row earlier.

    ``first_chunk = (consec[:, 0] == 0)`` selects between this and the carried-over
    path; with ``consec_train: 1`` it is always true, so the replay-context branch
    always wins. Returned here unconditionally, with that fact recorded rather than
    a dead branch ported.
    """
    K = context
    assert (data["consec"][:, 0] == 0).all(), "consec_train is 1; every chunk is the first"
    carry = {k: data[k][:, K - 1] for k in entry_keys}
    obs = {
        k: v[:, K:]
        for k, v in data.items()
        if k not in entry_keys and k not in ("consec", "action", "stepid")
    }
    prevact = data["action"][:, K - 1 : -1]
    stepid = data["stepid"][:, K:]
    return carry, {**obs, "action": prevact}, stepid
