"""U06 record: the annealed horizon, discount and epsilon, as a table.

BBF's receding update horizon is the component most likely to be misread, so
the realized schedule is written down rather than left implicit in the code.

Run: uv run --no-sync python tools/bbf_schedule_report.py
"""
from __future__ import annotations

import json
from pathlib import Path

from playtrain_trainers.bbf.config import BBFConfig
from playtrain_trainers.bbf.schedules import (
    discount,
    epsilon,
    is_reset_step,
    reset_steps,
    update_horizon,
)

OUT = Path("results/bbf/schedules")
GRAD_STEPS = [0, 1_000, 2_500, 5_000, 7_500, 10_000, 15_000, 19_999,
              20_000, 25_000, 30_000, 99_999, 100_000, 110_000, 150_000, 199_999]
ENV_STEPS = [0, 1_000, 1_999, 2_000, 2_500, 3_000, 4_000, 4_001, 10_000, 100_000]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    report = {}
    for label, cfg in (("RR=2", BBFConfig()), ("RR=8", BBFConfig(replay_ratio=256))):
        rows = [
            {
                "gradient_step": s,
                "is_reset": is_reset_step(s, cfg),
                "n": update_horizon(s, cfg),
                "gamma": round(discount(s, cfg), 6),
            }
            for s in GRAD_STEPS
        ]
        rs = reset_steps(cfg)
        gpe = cfg.gradient_steps_per_env_step
        report[label] = {
            "gradient_steps_per_env_step": gpe,
            "total_gradient_steps": cfg.total_gradient_steps,
            "reset_steps": rs,
            "reset_steps_in_env_steps": [s // gpe for s in rs],
            "env_steps_per_anneal_cycle": cfg.cycle_steps // gpe,
            # How much of the 100k-env-step run the resets actually span. At
            # the gin's values this is half the run at RR=2 but only an eighth
            # at RR=8 -- see flag F-010.
            "resets_cover_fraction_of_run": round(
                cfg.no_resets_after / gpe / cfg.training_steps, 4
            ),
            "schedule": rows,
        }
    report["epsilon"] = [
        {"env_step": s, "epsilon": round(epsilon(s, BBFConfig()), 6)} for s in ENV_STEPS
    ]
    (OUT / "schedules.json").write_text(json.dumps(report, indent=2) + "\n")

    for label in ("RR=2", "RR=8"):
        r = report[label]
        print(f"\n{label}: {r['gradient_steps_per_env_step']} gradient steps per env "
              f"step, {r['total_gradient_steps']:,} total")
        print(f"  resets at {r['reset_steps']}")
        print(f"  the 10k-gradient-step anneal is "
              f"{r['env_steps_per_anneal_cycle']:,} ENV steps at this replay ratio")
        print(f"  resets in ENV steps: {r['reset_steps_in_env_steps']} "
              f"-> they span {100 * r['resets_cover_fraction_of_run']:.1f}% of the run")
        print(f"  {'grad step':>10} {'reset':>6} {'n':>3} {'gamma':>9}")
        for row in r["schedule"]:
            print(f"  {row['gradient_step']:>10,} {str(row['is_reset']):>6} "
                  f"{row['n']:>3} {row['gamma']:>9.5f}")
    print("\n  epsilon (ENV steps):")
    for row in report["epsilon"]:
        print(f"  {row['env_step']:>10,}  {row['epsilon']:.5f}")
    print(f"\nwrote {OUT / 'schedules.json'}")


if __name__ == "__main__":
    main()
