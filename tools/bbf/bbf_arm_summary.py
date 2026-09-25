"""Aggregate one arm's per-seed metrics into the PROTOCOL section 5 B numbers.

Reads only `metrics_seed*.json` and writes `summary.json` beside them, so the
arm-level numbers the report quotes have a committed file behind them rather
than being recomputed by hand each time (MISSION hard rule 3).

Run: uv run --no-sync python tools/bbf/bbf_arm_summary.py results/bbf/<run_id>
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

CEILING = 260.0  # PlayTrain frostbite: 16 floes x 10 + 100 igloo (D-014)
PAPER_FROSTBITE_HNS = 0.543  # BBF's Frostbite HNS, PROTOCOL section 3


def bootstrap(vals, seed=0, resamples=10_000):
    a = np.asarray(vals, dtype=float)
    if a.size < 2:
        return float(a[0]), float(a[0])
    rng = np.random.default_rng(seed)
    means = rng.choice(a, size=(resamples, a.size), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--random-baseline", type=Path,
                    default=Path("results/bbf/random_frostbite/metrics.json"))
    args = ap.parse_args()

    runs = [json.load(open(f))
            for f in sorted(glob.glob(str(args.run_dir / "metrics_seed*.json")))]
    if not runs:
        raise SystemExit(f"no metrics_seed*.json under {args.run_dir}")
    # frostbite's fields (win, floes, the 260 ceiling) exist only where its
    # profile applies (D-044); key off the fields, not the backend.
    playtrain = "floes_visited_mean" in runs[0]["summary"]

    seeds = [d["config"]["seed"] for d in runs]
    scores = [d["summary"]["score_mean"] for d in runs]
    lo, hi = bootstrap(scores)
    out = {
        "run_id": runs[0]["run_id"],
        "unit": runs[0]["unit"],
        "env_backend": runs[0]["config"]["env_backend"],
        "game": runs[0]["config"]["game"],
        "replay_ratio": runs[0]["config"]["replay_ratio"],
        "gradient_steps_per_env_step": runs[0]["config"]["_derived"]["gradient_steps_per_env_step"],
        "commits": sorted({d["commit"] for d in runs}),
        "n_seeds": len(runs),
        "seeds": seeds,
        "per_seed_score_mean": dict(zip(map(str, seeds), scores, strict=True)),
        "score_mean": float(np.mean(scores)),
        "score_sd_over_seeds": float(np.std(scores, ddof=1)),
        "score_ci95_over_seeds": [lo, hi],
        "score_min": min(scores),
        "score_max": max(scores),
        "wall_hours_per_seed": [round(d["wall_seconds"] / 3600, 3) for d in runs],
        # BBF runs record gradient_steps; the PPO/IMPALA reference arms have
        # no such notion and record agent_steps instead. Report whichever the
        # arm actually has rather than assuming the BBF shape.
        "gradient_steps": sorted({d["gradient_steps"] for d in runs})
        if "gradient_steps" in runs[0] else None,
        "agent_steps": sorted({d["agent_steps"] for d in runs})
        if "agent_steps" in runs[0] else None,
        "trainer": runs[0].get("trainer", "bbf"),
    }

    if playtrain:
        rnd = float(json.loads(args.random_baseline.read_text())["summary"]["score_mean"])
        wins = [d["summary"]["win_rate"] for d in runs]
        floes = [d["summary"]["floes_visited_mean"] for d in runs]
        npg = [(s - rnd) / (CEILING - rnd) for s in scores]
        out |= {
            "random_baseline": rnd,
            "score_ceiling": CEILING,
            "win_rate_mean": float(np.mean(wins)),
            "win_rate_per_seed": dict(zip(map(str, seeds), wins, strict=True)),
            "floes_visited_mean": float(np.mean(floes)),
            "floes_visited_per_seed": dict(zip(map(str, seeds), floes, strict=True)),
            "normalized_progress_mean": float(np.mean(npg)),
            "normalized_progress_per_seed": dict(zip(map(str, seeds), npg, strict=True)),
            "normalized_progress_ci95_over_seeds": list(bootstrap(npg)),
            "paper_frostbite_hns_for_reference": PAPER_FROSTBITE_HNS,
            "caveat": (
                "normalized progress is (score - random) / (260 - random) on "
                "PlayTrain frostbite. The paper's Frostbite HNS of 0.543 is "
                "(score - 65.2) / (4334.7 - 65.2) on ALE, where the score is "
                "unbounded across levels. The two are NOT the same quantity "
                "and must not be subtracted or ranked against each other; "
                "they are printed together only to give the reader a scale."
            ),
        }

    # Late-run trend. Deliberately NOT "final minus the best curve point":
    # the curve is a 10-episode eval sampled ten times, so its maximum is an
    # upward-biased estimate of the true level (one seed's best was exactly
    # the 260 ceiling on 10 episodes) and every seed would look like it
    # regressed. The slope of the last three points separates a sustained
    # decline from a lucky peak.
    reg = {}
    for d in runs:
        curve = sorted(d.get("curve") or [], key=lambda c: c["env_step"])
        if len(curve) < 3:
            continue
        tail = curve[-3:]
        xs = np.array([c["env_step"] for c in tail], dtype=float)
        ys = np.array([c["score_mean"] for c in tail], dtype=float)
        slope_per_10k = float(np.polyfit(xs, ys, 1)[0] * 10_000)
        reg[str(d["config"]["seed"])] = {
            "tail_env_steps": [int(x) for x in xs],
            "tail_curve_means": [float(y) for y in ys],
            "slope_per_10k_env_steps": round(slope_per_10k, 2),
            "final_100ep_mean": d["summary"]["score_mean"],
            "best_curve_mean": float(max(c["score_mean"] for c in curve)),
            # A sustained decline means the tail falls at EVERY step and the
            # 100-episode eval confirms the lower level. A slope threshold
            # cannot do this job across arms: the 10-episode curve on ALE
            # Frostbite swings by thousands, so a tail of [2066, 276, 1505]
            # scored a steep negative slope while actually ending flat.
            # Requiring monotonicity is scale-free.
            "sustained_decline": bool(
                all(ys[i] > ys[i + 1] for i in range(len(ys) - 1))
                and d["summary"]["score_mean"] < ys[0]
            ),
        }
    out["late_run_trend"] = reg

    path = args.run_dir / "summary.json"
    path.write_text(json.dumps(out, indent=2) + "\n")

    budget = (f"RR={out['gradient_steps_per_env_step']}"
              if out["trainer"] == "bbf" else f"trainer={out['trainer']}")
    print(f"{out['run_id']}  ({out['unit']}, {out['env_backend']}, "
          f"{budget}, {out['n_seeds']} seeds)")
    print(f"  score      {out['score_mean']:.2f}  sd {out['score_sd_over_seeds']:.2f}  "
          f"CI95 over seeds [{lo:.2f}, {hi:.2f}]  range [{out['score_min']:.1f}, {out['score_max']:.1f}]")
    if playtrain:
        print(f"  win rate   {out['win_rate_mean']:.3f}")
        print(f"  floes      {out['floes_visited_mean']:.2f} / 16")
        print(f"  norm prog  {out['normalized_progress_mean']:.3f}  "
              f"CI95 {[round(x, 3) for x in out['normalized_progress_ci95_over_seeds']]}"
              f"   (random {out['random_baseline']}, ceiling {CEILING:.0f})")
    print("\n  late-run trend (slope of the last three 10-ep curve points):")
    for s, r in reg.items():
        flag = "  <-- SUSTAINED DECLINE" if r["sustained_decline"] else ""
        print(f"    seed {s:>6}: {[round(y) for y in r['tail_curve_means']]} "
              f"slope {r['slope_per_10k_env_steps']:+7.1f}/10k  "
              f"final {r['final_100ep_mean']:6.2f}{flag}")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
