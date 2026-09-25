"""Regenerate the U11 reference numbers from the committed BBF metrics files.

Every number in `results/dreamerv3/REFERENCE_TABLE.md` comes from here, so the table
can be re-derived rather than trusted (MISSION rule 6).
"""

from __future__ import annotations

import glob
import json
import math

ARMS = [
    ("random_frostbite", "PlayTrain random"),
    ("ref_ppo_frostbite_100k", "PPO @100k greedy"),
    ("ref_ppo_frostbite_100k_sampled", "PPO @100k sampled"),
    ("ref_impala_frostbite_100k", "IMPALA @100k greedy"),
    ("ref_impala_frostbite_100k_sampled", "IMPALA @100k sampled"),
    ("bbf_playtrain_frostbite_rr2_v4", "BBF RR=2 v4 (PlayTrain)"),
    ("bbf_playtrain_frostbite_rr8_v4", "BBF RR=8 v4 (PlayTrain)"),
    ("bbf_ale_frostbite_rr2_v4", "BBF RR=2 v4 (ALE)"),
]


def arm(run_id: str, root: str = "results/bbf") -> dict | None:
    files = sorted(glob.glob(f"{root}/{run_id}/metrics_seed*.json"))
    if not files:
        return None
    summaries = [json.load(open(f))["summary"] for f in files]
    scores = [s["score_mean"] for s in summaries]
    mean = sum(scores) / len(scores)
    sd = (
        math.sqrt(sum((x - mean) ** 2 for x in scores) / (len(scores) - 1))
        if len(scores) > 1
        else None
    )

    def avg(key: str):
        vals = [s[key] for s in summaries if s.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    return {
        "n": len(scores),
        "mean": mean,
        "sd": sd,
        "per_seed": sorted(round(x, 1) for x in scores),
        "win_rate": avg("win_rate"),
        "floes": avg("floes_visited_mean"),
        "normalized": avg("normalized_progress"),
    }


def main() -> None:
    for run_id, label in ARMS:
        a = arm(run_id)
        if not a:
            print(f"{label:26} MISSING ({run_id}) -- flag, do not re-run (U11)")
            continue
        sd = f"{a['sd']:7.2f}" if a["sd"] is not None else "      -"
        extra = ""
        if a["win_rate"] is not None:
            extra += f"  win {a['win_rate']:.2f}"
        if a["floes"] is not None:
            extra += f"  floes {a['floes']:.1f}"
        if a["normalized"] is not None:
            extra += f"  norm {a['normalized']:.3f}"
        print(f"{label:26} n={a['n']} mean={a['mean']:8.2f} sd={sd}  {a['per_seed']}{extra}")


if __name__ == "__main__":
    main()
