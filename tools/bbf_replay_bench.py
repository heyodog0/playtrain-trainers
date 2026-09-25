"""U05 record: the replay buffer's footprint and sampling throughput.

Sampling sits on the critical path of every gradient step, so its cost has to
be known before U08 is committed to a wall-time estimate. It also records the
memory the "store frames once" decision saves.

Run: uv run --no-sync python tools/bbf_replay_bench.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from playtrain_trainers.bbf.config import BBFConfig
from playtrain_trainers.bbf.replay import SubsequenceReplayBuffer

OUT = Path("results/bbf/replay")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = BBFConfig()
    buf = SubsequenceReplayBuffer(cfg)
    frame = np.zeros(buf.frame_shape, np.uint8)
    rng = np.random.default_rng(0)

    # Fill well past one window so validity is not the limiting factor, but
    # short of capacity -- a 100k-step run never fills a 200k buffer anyway.
    n_add = 20_000
    t0 = time.time()
    for i in range(n_add):
        buf.add(frame, i % 8, 1.0 if i % 50 == 0 else 0.0, i % 400 == 399)
    add_s = time.time() - t0

    results = {
        "capacity": cfg.replay_capacity,
        "frame_shape": list(buf.frame_shape),
        "obs_shape": list(buf._obs_shape()),
        "ring_bytes": buf.nbytes(),
        "ring_gb": round(buf.nbytes() / 1e9, 3),
        "stacked_equivalent_gb": round(buf.frames.nbytes * cfg.frame_stack / 1e9, 3),
        "transitions_added": n_add,
        "add_per_s": round(n_add / add_s),
        "sample": {},
    }

    for n in (cfg.max_update_horizon, cfg.update_horizon):
        k = 200
        t0 = time.time()
        for _ in range(k):
            buf.sample(cfg.batch_size, n=n, gamma=cfg.gamma, rng=rng)
        dt = time.time() - t0
        results["sample"][f"n={n}"] = {
            "batches_per_s": round(k / dt, 1),
            "ms_per_batch": round(1000 * dt / k, 3),
        }

    # What the protocol's budget costs in replay time alone.
    ms = results["sample"][f"n={cfg.update_horizon}"]["ms_per_batch"]
    for label, ratio in (("RR=2", 2), ("RR=8", 8)):
        steps = cfg.training_steps * ratio
        results.setdefault("run_cost", {})[label] = {
            "gradient_steps": steps,
            "replay_seconds": round(steps * ms / 1000, 1),
        }

    (OUT / "bench.json").write_text(json.dumps(results, indent=2) + "\n")

    print(f"capacity           {results['capacity']:,} transitions")
    print(f"frame ring         {results['ring_gb']} GB  (frames stored once)")
    print(f"if stacked instead {results['stacked_equivalent_gb']} GB")
    print(f"add                {results['add_per_s']:,}/s")
    for k, v in results["sample"].items():
        print(f"sample {k:<6}      {v['batches_per_s']:>7.1f} batches/s "
              f"({v['ms_per_batch']} ms at batch {cfg.batch_size})")
    for k, v in results["run_cost"].items():
        print(f"{k:<6} {v['gradient_steps']:>8,} gradient steps "
              f"-> {v['replay_seconds']:>6.1f} s of replay sampling")
    print(f"\nwrote {OUT / 'bench.json'}")


if __name__ == "__main__":
    main()
