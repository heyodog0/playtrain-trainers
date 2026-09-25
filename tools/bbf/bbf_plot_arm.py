"""Plot one multi-seed BBF arm STRICTLY from its metrics_seed*.json files.

MISSION hard rule 4: the plot is regenerated from the saved files and the
table it draws is printed first, so the two can be compared by eye. Nothing
here recomputes a score -- every number is read back from disk.

Run: uv run --no-sync --with matplotlib python tools/bbf/bbf_plot_arm.py \
        results/bbf/bbf_ale_frostbite_rr2 --title "BBF on ALE Frostbite, RR=2"
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

INK = "#1f2933"
MUTED = "#7b8794"
SEED_COLORS = ["#3f7cac", "#c1666b", "#6a8d73", "#b08968", "#7d6b91",
               "#4a7b9d", "#a2678a", "#5c8a72"]
# PROTOCOL section 3, Table A.1 (the Frostbite row).
PAPER = {"BBF (paper, RR=8)": 2384.8, "SR-SPR": 2584.8, "SPR": 1170.7,
         "human": 4334.7, "random": 65.2}


def load(run_dir: Path) -> list[dict]:
    runs = []
    for f in sorted(glob.glob(str(run_dir / "metrics_seed*.json"))):
        d = json.load(open(f))
        runs.append(d)
    if not runs:
        raise SystemExit(f"no metrics_seed*.json under {run_dir}")
    return runs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--title", default=None)
    ap.add_argument("--reference", action="store_true",
                    help="draw the paper's Frostbite reference lines")
    ap.add_argument("--ceiling", type=float, default=None,
                    help="draw a score ceiling line (260 for PlayTrain frostbite)")
    args = ap.parse_args()

    runs = load(args.run_dir)
    title = args.title or args.run_dir.name

    # --- the table, printed BEFORE anything is drawn --------------------
    print(f"source: {args.run_dir}/metrics_seed*.json  ({len(runs)} seeds)")
    print(f"commit: {runs[0]['commit'][:12]}  unit: {runs[0]['unit']}")
    # win_rate / floes are PlayTrain frostbite's scoring rule and are
    # meaningless elsewhere, so they are shown only for that backend -- older
    # ALE metrics files still contain them and must not be quoted.
    # frostbite's fields exist only where its profile applies (D-044), so
    # key off the fields, not the backend.
    pt = "win_rate" in runs[0]["summary"] and "floes_visited_mean" in runs[0]["summary"]
    head = f"\n{'seed':>8} {'100-ep mean':>12} {'ci95_lo':>9} {'ci95_hi':>9}"
    head += f" {'win':>6} {'floes':>7}" if pt else ""
    print(head + f" {'grad':>8} {'wall_h':>7}")
    finals = []
    for d in runs:
        s = d["summary"]
        finals.append(s["score_mean"])
        row = (f"{d['config']['seed']:>8} {s['score_mean']:>12.2f} "
               f"{s['score_ci95'][0]:>9.1f} {s['score_ci95'][1]:>9.1f}")
        if pt:
            row += f" {s['win_rate']:>6.3f} {s['floes_visited_mean']:>7.2f}"
        print(row + f" {d['gradient_steps']:>8} {d['wall_seconds'] / 3600:>7.2f}")
    mean = float(np.mean(finals))
    sd = float(np.std(finals, ddof=1)) if len(finals) > 1 else 0.0
    # Bootstrap over SEEDS, as PROTOCOL section 5 B asks.
    rng = np.random.default_rng(0)
    boot = rng.choice(np.array(finals), size=(10_000, len(finals)), replace=True).mean(axis=1)
    lo, hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    print(f"\n{len(finals)}-seed mean {mean:.2f}  sd {sd:.2f}  "
          f"95% CI over seeds [{lo:.2f}, {hi:.2f}]")

    # --- the plot -------------------------------------------------------
    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [1.55, 1]}
    )

    for i, d in enumerate(runs):
        curve = d.get("curve") or []
        if not curve:
            continue
        xs = [c["env_step"] for c in curve]
        ys = [c["score_mean"] for c in curve]
        c = SEED_COLORS[i % len(SEED_COLORS)]
        ax.plot(xs, ys, color=c, lw=1.5, marker="o", ms=3,
                label=f"seed {d['config']['seed']}")
        # The final 100-episode eval is the reported number; mark it apart
        # from the 10-episode curve so the two are not read as one series.
        ax.plot([d["config"]["training_steps"]], [d["summary"]["score_mean"]],
                marker="*", ms=13, color=c, mec="white", mew=0.8, zorder=5)
    ax.set_xlabel("env steps (agent steps)")
    ax.set_ylabel("episode score")
    ax.set_title(f"{title}\nlines = 10-ep curve, star = final 100-ep eval",
                 color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=7, ncol=2)

    # Final means per seed, with the reference lines beside them.
    seeds = [str(d["config"]["seed"]) for d in runs]
    colors = [SEED_COLORS[i % len(SEED_COLORS)] for i in range(len(runs))]
    yerr = np.array([
        [d["summary"]["score_mean"] - d["summary"]["score_ci95"][0] for d in runs],
        [d["summary"]["score_ci95"][1] - d["summary"]["score_mean"] for d in runs],
    ])
    ax2.bar(seeds, finals, color=colors, yerr=yerr, capsize=3,
            error_kw={"lw": 1, "ecolor": INK})
    ax2.axhline(mean, color=INK, lw=1.4, ls="-")
    ax2.annotate(f"{len(finals)}-seed mean {mean:.0f}", xy=(0.02, mean),
                 xycoords=("axes fraction", "data"), xytext=(0, 4),
                 textcoords="offset points", color=INK, fontsize=8)
    if args.reference:
        for label, v in PAPER.items():
            ax2.axhline(v, color=MUTED, lw=0.9, ls=":")
            ax2.annotate(f"{label} {v:g}", xy=(0.98, v),
                         xycoords=("axes fraction", "data"), xytext=(0, 3),
                         textcoords="offset points", ha="right",
                         color=MUTED, fontsize=7)
    if args.ceiling is not None:
        ax2.axhline(args.ceiling, color="#c1666b", lw=1.3)
        ax2.annotate(f"ceiling {args.ceiling:g}", xy=(0.98, args.ceiling),
                     xycoords=("axes fraction", "data"), xytext=(0, 3),
                     textcoords="offset points", ha="right",
                     color="#c1666b", fontsize=8)
    ax2.set_ylabel("final 100-episode mean")
    ax2.set_xlabel("seed")
    ax2.set_title("final eval per seed (bars = bootstrap CI95 over episodes)",
                  color=INK, fontsize=9, loc="left")

    for a in (ax, ax2):
        a.spines[["top", "right"]].set_visible(False)
        a.tick_params(colors=MUTED, labelsize=8)
        for sp in a.spines.values():
            sp.set_color(MUTED)

    fig.tight_layout()
    out = args.run_dir / "curves.png"
    fig.savefig(out, dpi=170, facecolor="white")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
