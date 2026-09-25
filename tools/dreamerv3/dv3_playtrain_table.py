"""U09: DreamerV3 on PlayTrain frostbite beside the reference arms (PROTOCOL section 5 B).

DreamerV3 rows come from `eval_seed*.json` (tools/dreamerv3/dv3_evaluate.py: 100 whole
games per seed on the BBF fixed seed pool) and `metrics_seed*.json` (training-episode
score, last 50k frames before 400k). Reference rows come from the committed
`results/bbf/*/metrics_seed*.json` through `dv3_reference_table.arm`, so every number is
re-derived. Comparisons use `bbf_compare.exact_permutation_p`: unpaired, exact.

Usage::

    python tools/dreamerv3/dv3_playtrain_table.py results/dreamerv3/playtrain_frostbite_u09
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bbf"))
import dv3_gate_summary as G  # noqa: E402
import dv3_reference_table as REF  # noqa: E402
from bbf_compare import exact_permutation_p  # noqa: E402

from playtrain_trainers.bbf.evaluate import bootstrap_ci  # noqa: E402


def dreamer(run_dir: Path, mode: str) -> dict:
    evals = [json.load(open(f)) for f in sorted(glob.glob(str(run_dir / "eval_seed*.json")))]
    aggs = [e[mode]["aggregate"] for e in evals]
    scores = [a["score_mean"] for a in aggs]
    avg = lambda k: float(np.mean([a[k] for a in aggs]))  # noqa: E731
    return {
        "n": len(scores), "mean": float(np.mean(scores)), "sd": float(np.std(scores, ddof=1)),
        "per_seed": [round(x, 1) for x in scores], "ci95": bootstrap_ci(scores),
        "win_rate": avg("win_rate"), "floes": avg("floes_visited_mean"), "normalized": avg("normalized_progress"),
        "scores": scores,
    }


def main() -> None:
    run_dir = Path(sys.argv[1])
    train = G.summarize(run_dir)
    tr = [s["last50k"] for s in train["seeds"]]
    rows = {"DreamerV3 sampled": dreamer(run_dir, "sampled"), "DreamerV3 greedy": dreamer(run_dir, "greedy")}
    refs = {}
    for run_id, label in REF.ARMS:
        if "ALE" in label:
            continue
        a = REF.arm(run_id)
        if a:
            files = sorted(glob.glob(f"results/bbf/{run_id}/metrics_seed*.json"))
            a["scores"] = [json.load(open(f))["summary"]["score_mean"] for f in files]
            a["ci95"] = bootstrap_ci(a["scores"]) if len(a["scores"]) > 1 else (a["mean"], a["mean"])
            refs[label] = a
    print(f"DreamerV3 training-episode score, last 50k frames before 400k: per seed {[round(x, 1) for x in tr]}, "
          f"mean {np.mean(tr):.1f}, CI {tuple(round(x, 1) for x in bootstrap_ci(tr))}")
    print()
    print(f"{'arm':<26} {'n':>2} {'mean':>8} {'sd':>7} {'95% CI (over seeds)':>22} {'win':>5} {'floes':>6} {'norm':>6}  per seed")
    for label, a in {**refs, **rows}.items():
        sd = f"{a['sd']:7.2f}" if a.get("sd") is not None else "      -"
        ci = f"[{a['ci95'][0]:.1f}, {a['ci95'][1]:.1f}]"
        f = lambda v, fmt: format(v, fmt) if v is not None else "-"  # noqa: E731
        print(f"{label:<26} {a['n']:>2} {a['mean']:>8.2f} {sd} {ci:>22} {f(a['win_rate'], '5.2f'):>5} "
              f"{f(a['floes'], '6.1f'):>6} {f(a['normalized'], '6.3f'):>6}  {sorted(a['per_seed'])}")
    print()
    for mode in ("DreamerV3 sampled", "DreamerV3 greedy"):
        d = np.array(rows[mode]["scores"])
        for label, a in refs.items():
            if a["n"] < 2:
                continue
            p, floor, n = exact_permutation_p(np.array(a["scores"]), d)
            print(f"{mode} vs {label:<24}: diff {d.mean() - np.mean(a['scores']):+7.2f}  exact p {p:.4f} (floor {floor:.4f}, {n} splits)")


if __name__ == "__main__":
    main()
