"""Aggregate DreamerV3 seeds the way the official score file is aggregated.

The official protocol (PROTOCOL section 5 A, and `reference/official_frostbite_scores.md`)
reads the training-episode return of a SAMPLED policy, binned along the run, and
reports the mean over the last 50k environment frames before 400k. There is no
separate eval; `run/train.py` logs `episode/score` straight from the training driver.

So an episode counts towards the last-50k figure if it ENDED in [350k, 400k) frames.
Both that and the last-100k figure are printed, along with the per-seed spread, so the
report can say which aggregation a number came from rather than leaving it implicit.

Usage::

    python tools/dreamerv3/dv3_gate_summary.py outputs/dreamerv3/ale_frostbite_gate
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

# The official 5 Frostbite seeds, last-50k aggregation, from
# reference/official_frostbite_scores.md. Used only for the "where do we fall"
# column; the gate threshold itself is PROTOCOL section 5 A.
OFFICIAL_LAST50K = [350.5, 2846.9, 2944.2, 4359.1, 4190.0]
OFFICIAL_FINAL_BIN = [370.0, 2560.0, 3620.0, 4800.0, 8220.0]

# The official JAX code RUN ON OUR CLUSTER, first 5 seeds (job 47627202). U14 showed these
# were the unlucky tail: the same setup's next 15 seeds broke out 10/15, and all 45 official
# seeds here average 1449.1 with 20/45 broken out (results/dreamerv3/official_u14/). Kept for
# the U13 comparisons; do NOT use it as the reference. Raw: results/dreamerv3/official_jax_frostbite.json.
OFFICIAL_HERE_LAST50K = [430.0, 281.8, 250.4, 316.5, 252.4]
OFFICIAL_HERE_ALL_MEAN, OFFICIAL_HERE_ALL_BROKE, OFFICIAL_HERE_ALL_N = 1449.1, 20, 45


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def stdev(xs: list[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def window_mean(episodes: list[dict[str, Any]], lo: int, hi: int) -> tuple[float, int]:
    """Mean training-episode score over episodes ending in [lo, hi) frames."""
    scores = [e["score"] for e in episodes if lo <= e["frames"] < hi]
    return mean(scores), len(scores)


def summarize(run_dir: Path, read_at: int = 400_000) -> dict[str, Any]:
    files = sorted(run_dir.glob("metrics_seed*.json"))
    if not files:
        raise SystemExit(f"no metrics_seed*.json in {run_dir}")
    seeds = []
    for path in files:
        d = json.loads(path.read_text())
        eps = d["episodes"]
        last50k, n50 = window_mean(eps, read_at - 50_000, read_at)
        last100k, n100 = window_mean(eps, read_at - 100_000, read_at)
        final_ep = [e["score"] for e in eps if e["frames"] < read_at]
        seeds.append(
            {
                "seed": d["seed"],
                "partial": d.get("partial", False),
                "frames": d["frames"],
                "agent_steps": d["agent_steps"],
                "grad_steps": d["grad_steps"],
                "grad_steps_expected": d.get("grad_steps_expected"),
                "ratio_post_warmup": d["achieved_ratio_post_warmup"],
                "episodes": len(eps),
                "last50k": last50k,
                "last50k_n": n50,
                "last100k": last100k,
                "last100k_n": n100,
                "final_episode": final_ep[-1] if final_ep else float("nan"),
                "max_episode": max(final_ep) if final_ep else float("nan"),
                "wall_hours": round(d["wall_seconds"] / 3600, 2),
                "peak_gpu_gb": d.get("peak_gpu_gb"),
            }
        )
    last50 = [s["last50k"] for s in seeds if not math.isnan(s["last50k"])]
    # The escape rate is the quantity the 15-seed run exists to estimate: how often a
    # seed leaves the ~300 local optimum at all. Frostbite is bimodal, so the MEAN is
    # dominated by how many seeds escaped, not by how well they did.
    escaped = [x for x in last50 if x >= 1000]
    return {
        "run_dir": str(run_dir),
        "read_at_frames": read_at,
        "seeds": seeds,
        "mean_last50k": mean(last50),
        "sd_last50k": stdev(last50),
        "mean_last100k": mean([s["last100k"] for s in seeds if not math.isnan(s["last100k"])]),
        "se_last50k": stdev(last50) / math.sqrt(len(last50)) if len(last50) > 1 else float("nan"),
        "n_scored": len(last50),
        "n_escaped": len(escaped),
        "escape_rate": len(escaped) / len(last50) if last50 else float("nan"),
        "escape_ci95": wilson(len(escaped), len(last50)),
        "official_mean_last50k": mean(OFFICIAL_LAST50K),
        "official_sd_last50k": stdev(OFFICIAL_LAST50K),
        "official_escape_rate": sum(1 for x in OFFICIAL_LAST50K if x >= 1000) / len(OFFICIAL_LAST50K),
    }


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval — the right one for a proportion at small n, where the
    normal approximation would give a nonsense interval (and can include 0 or 1)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def ours_scores(out: dict) -> list[float]:
    return [s["last50k"] for s in out["seeds"] if not math.isnan(s["last50k"])]


def verdict(mean_last50k: float) -> str:
    """PROTOCOL section 5 A: PASS at >= 1500, "plausible" in [1000, 1500), FAIL below."""
    if math.isnan(mean_last50k):
        return "NO DATA"
    if mean_last50k >= 1500:
        return "PASS"
    if mean_last50k >= 1000:
        return "PLAUSIBLE (not confirmed)"
    return "FAIL"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir")
    ap.add_argument("--read-at", type=int, default=400_000)
    ap.add_argument("--json", action="store_true", help="dump the summary as JSON too")
    args = ap.parse_args()

    out = summarize(Path(args.run_dir), args.read_at)
    print(f"{out['run_dir']}  (read at {out['read_at_frames']} frames)")
    print()
    hdr = (
        f"{'seed':>4} {'frames':>8} {'grad':>6} {'ratio':>7} {'eps':>5} "
        f"{'last50k':>9} {'n':>4} {'last100k':>9} {'max':>7} {'hours':>6} {'part':>5}"
    )
    print(hdr)
    print("-" * len(hdr))
    for s in out["seeds"]:
        print(
            f"{s['seed']:>4} {s['frames']:>8} {s['grad_steps']:>6} "
            f"{s['ratio_post_warmup']:>7.4f} {s['episodes']:>5} "
            f"{s['last50k']:>9.1f} {s['last50k_n']:>4} {s['last100k']:>9.1f} "
            f"{s['max_episode']:>7.0f} {s['wall_hours']:>6.2f} "
            f"{'YES' if s['partial'] else '':>5}"
        )
    print("-" * len(hdr))
    n = out["n_scored"]
    print(
        f"  {n}-seed mean last-50k = {out['mean_last50k']:.1f}  "
        f"(sd {out['sd_last50k']:.1f}, se {out['se_last50k']:.1f});  "
        f"last-100k = {out['mean_last100k']:.1f}"
    )
    print(
        f"  official, PUBLISHED  = {out['official_mean_last50k']:.1f}  "
        f"(sd {out['official_sd_last50k']:.1f}), per seed "
        f"{[round(x) for x in sorted(OFFICIAL_LAST50K)]}"
    )
    here = OFFICIAL_HERE_LAST50K
    hm = mean(here)
    print(
        f"  official, OUR CLUSTER= {hm:.1f}  (sd {stdev(here):.1f}), per seed "
        f"{[round(x) for x in sorted(here)]}   <- first 5 only, an unlucky draw (U14)"
    )
    print(
        f"  official, OUR CLUSTER, all {OFFICIAL_HERE_ALL_N} seeds = {OFFICIAL_HERE_ALL_MEAN:.1f}, "
        f"{OFFICIAL_HERE_ALL_BROKE}/{OFFICIAL_HERE_ALL_N} >= 1000   <- the reference"
    )
    if not math.isnan(out["mean_last50k"]):
        print(f"  ours / official-here (all) = {out['mean_last50k'] / OFFICIAL_HERE_ALL_MEAN:.2f}x")
    lo, hi = out["escape_ci95"]
    print(
        f"  escape rate (last-50k >= 1000): {out['n_escaped']}/{out['n_scored']} "
        f"= {out['escape_rate']:.2f}  95% CI [{lo:.2f}, {hi:.2f}];  "
        f"official {out['official_escape_rate']:.2f} (4/5)"
    )
    print(f"  PROTOCOL section 5 A verdict: {verdict(out['mean_last50k'])}")
    print("    PASS >= 1500 | PLAUSIBLE 1000-1499 | FAIL < 1000   (flag F-001)")
    ours = sorted(s["last50k"] for s in out["seeds"] if not math.isnan(s["last50k"]))
    if ours:
        inside = [x for x in ours if min(OFFICIAL_LAST50K) <= x <= max(OFFICIAL_LAST50K)]
        print(
            f"  {len(inside)}/{len(ours)} of our seeds fall inside the official spread "
            f"[{min(OFFICIAL_LAST50K):.0f}, {max(OFFICIAL_LAST50K):.0f}]"
        )
    bad = [s for s in out["seeds"] if s["grad_steps"] != s["grad_steps_expected"]]
    if bad:
        print(f"  !! train-ratio mismatch on seeds {[s['seed'] for s in bad]}")
    if args.json:
        print()
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
