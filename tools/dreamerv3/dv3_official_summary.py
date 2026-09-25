"""Aggregate an official-JAX DreamerV3 logdir tree the way the gate is aggregated.

The official logger writes `episode/score` to `<logdir>/metrics.jsonl` with `step` in
FRAMES (logger multiplier = env repeat 4). An episode counts towards last-50k if its
logged step is in [350k, 400k), matching `dv3_gate_summary.window_mean`.

Writes the same JSON shape as `results/dreamerv3/official_jax_frostbite.json` so the
two official runs can be compared line for line.

Usage::

    python tools/dreamerv3/dv3_official_summary.py results/dreamerv3/official_2411f7d1
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import dv3_gate_summary as G

PUBLISHED = G.OFFICIAL_LAST50K
PINNED_HERE = G.OFFICIAL_HERE_LAST50K


def read_scores(metrics: Path) -> list[list[float]]:
    out = []
    for line in metrics.read_text().splitlines():
        rec = json.loads(line)
        if "episode/score" in rec:
            out.append([int(rec["step"]), float(rec["episode/score"])])
    return out


def last50k(scores: list[list[float]], read_at: int = 400_000) -> float:
    xs = [s for f, s in scores if read_at - 50_000 <= f < read_at]
    return G.mean(xs)


def welch_t(a: list[float], b: list[float]) -> float:
    va, vb = G.stdev(a) ** 2 / len(a), G.stdev(b) ** 2 / len(b)
    return (G.mean(a) - G.mean(b)) / math.sqrt(va + vb)


def fisher_one_sided(k1: int, n1: int, k2: int, n2: int) -> float:
    """P(X >= k1) under the hypergeometric: arm 1 escapes at least this often."""
    K, N = k1 + k2, n1 + n2
    total = math.comb(N, n1)
    return sum(math.comb(K, k) * math.comb(N - K, n1 - k) for k in range(k1, min(K, n1) + 1)) / total


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir")
    args = ap.parse_args()
    root = Path(args.run_dir)
    arms: dict[str, dict[int, dict]] = {}
    for m in sorted((root / "raw" / "logdir").glob("*/metrics.jsonl")):
        task, seed = re.match(r"(.+)_seed(\d+)$", m.parent.name).groups()
        sc = read_scores(m)
        arms.setdefault(task, {})[int(seed)] = {
            "episodes": len(sc),
            "last_frame": max(f for f, _ in sc),
            "last50k": last50k(sc),
            "scores": sc,
        }

    summary = {}
    for task, seeds in arms.items():
        vals = [seeds[s]["last50k"] for s in sorted(seeds)]
        esc = sum(v >= 1000 for v in vals)
        summary[task] = {
            "n": len(vals),
            "last50k_per_seed": [round(v, 1) for v in vals],
            "mean_last50k": round(G.mean(vals), 1),
            "sd_last50k": round(G.stdev(vals), 1),
            "escaped": esc,
            "welch_t_vs_pinned_here": round(welch_t(vals, PINNED_HERE), 2),
            "welch_t_vs_published": round(welch_t(vals, PUBLISHED), 2),
            "fisher_p_published_escapes_more": round(
                fisher_one_sided(sum(v >= 1000 for v in PUBLISHED), len(PUBLISHED), esc, len(vals)), 4
            ),
        }
        (root / f"{task}.json").write_text(
            json.dumps(
                {
                    "source": "official danijar/dreamerv3 @ 2411f7d1, unmodified, run on kempner_h100",
                    "config": f"--configs atari100k --task {task}",
                    "note": "logged step is FRAMES (logger multiplier = env repeat 4)",
                    "seeds": {str(s): seeds[s] for s in sorted(seeds)},
                },
                indent=1,
            )
        )

    print(f"{'arm':<22} {'n':>2} {'mean':>7} {'sd':>7} {'esc':>4}  per seed")
    rows = [
        ("published (score file)", PUBLISHED),
        ("e3f0224 on our cluster", PINNED_HERE),
    ] + [(f"2411f7d1 {t}", [arms[t][s]["last50k"] for s in sorted(arms[t])]) for t in arms]
    for name, vals in rows:
        print(
            f"{name:<22} {len(vals):>2} {G.mean(vals):>7.1f} {G.stdev(vals):>7.1f} "
            f"{sum(v >= 1000 for v in vals):>4}  {[round(v) for v in sorted(vals)]}"
        )
    print()
    print(json.dumps(summary, indent=1))
    (root / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
