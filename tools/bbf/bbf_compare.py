"""Compare two arms the way this project's own audit says they must be compared.

Three rules, each learned the hard way and recorded in bbf-loop/STATE.md:

1. **Unpaired.** Pairing by seed buys nothing here — a code change alone
   reshuffled the seed ordering completely (31677 went worst-to-best), so a
   seed does not pin an outcome and a paired interval is too narrow.
2. **EXACT permutation, not Monte Carlo.** With 5 seeds per arm there are only
   C(10,5) = 252 distinct splits, so the smallest attainable two-sided p is
   2/252 = 0.0079. An MC permutation over that space reported p = 0.007, which
   is not a possible value. `itertools.combinations` enumerates it outright.
3. **Report the floor.** A p at the floor means "as extreme as this design can
   show", not "p = 0.008".

Run: uv run --no-sync python tools/bbf/bbf_compare.py ARM_A ARM_B
"""
from __future__ import annotations

import argparse
import glob
import json
from itertools import combinations

import numpy as np


def scores(run_id: str) -> np.ndarray:
    fs = sorted(glob.glob(f"results/bbf/{run_id}/metrics_seed*.json"))
    if not fs:
        one = f"results/bbf/{run_id}/metrics.json"
        return np.array([json.load(open(one))["summary"]["score_mean"]])
    return np.array([json.load(open(f))["summary"]["score_mean"] for f in fs])


def exact_permutation_p(a: np.ndarray, b: np.ndarray) -> tuple[float, float, int]:
    """Two-sided exact permutation p, plus the design's floor and split count."""
    pool = np.concatenate([a, b])
    n, obs = len(a), b.mean() - a.mean()
    idx = range(len(pool))
    diffs = []
    for left in combinations(idx, n):
        mask = np.zeros(len(pool), bool)
        mask[list(left)] = True
        diffs.append(pool[~mask].mean() - pool[mask].mean())
    diffs = np.asarray(diffs)
    p = float((np.abs(diffs) >= abs(obs) - 1e-12).mean())
    return p, 2.0 / len(diffs), len(diffs)


def bootstrap_ci(a: np.ndarray, b: np.ndarray, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    d = (rng.choice(b, (20000, len(b))).mean(1)
         - rng.choice(a, (20000, len(a))).mean(1))
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("arm_a")
    ap.add_argument("arm_b")
    args = ap.parse_args()
    a, b = scores(args.arm_a), scores(args.arm_b)
    print(f"{args.arm_a:<45}{a.mean():9.2f}  n={len(a)}  {sorted(round(x) for x in a)}")
    print(f"{args.arm_b:<45}{b.mean():9.2f}  n={len(b)}  {sorted(round(x) for x in b)}")
    lo, hi = bootstrap_ci(a, b)
    print(f"\ndifference (B - A): {b.mean() - a.mean():+.2f}")
    print(f"bootstrap CI95 (unpaired): [{lo:+.2f}, {hi:+.2f}]   excludes 0: {lo > 0 or hi < 0}")
    if len(a) > 1 and len(b) > 1:
        p, floor, n = exact_permutation_p(a, b)
        at_floor = " (AT THE FLOOR: as extreme as this design can show)" if p <= floor + 1e-12 else ""
        print(f"exact permutation p = {p:.4f} over {n} splits; floor = {floor:.4f}{at_floor}")
    else:
        print("exact permutation: not applicable (an arm has one sample)")


if __name__ == "__main__":
    main()
