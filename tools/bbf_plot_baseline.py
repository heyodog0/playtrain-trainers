"""Plot a BBF results directory STRICTLY from its metrics.json.

MISSION hard rule 4: the plot is regenerated from the saved file, never from
whatever is in memory at the end of a run, and the table it draws is printed
so the two can be compared. This lab has a history of plots that looked fine
and showed the wrong data.

Run: uv run --no-sync python tools/bbf_plot_baseline.py results/bbf/random_frostbite
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

INK = "#1f2933"
MUTED = "#7b8794"
BAR = "#3f7cac"
CEIL = "#c1666b"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    args = ap.parse_args()

    path = args.run_dir / "metrics.json"
    d = json.loads(path.read_text())
    eps = d["episodes"]
    s = d["summary"]
    scores = np.array([e["score"] for e in eps], dtype=float)
    ceiling = float(s["score_ceiling"])

    # --- the table, printed before the plot is drawn ---------------------
    counts: dict[float, int] = {}
    for v in scores:
        counts[v] = counts.get(v, 0) + 1
    print(f"source: {path}")
    print(f"unit={d['unit']}  run_id={d['run_id']}  commit={d['commit'][:12]}")
    print(f"episodes={len(eps)}  (summary says {s['episodes']})")
    print(f"mean={scores.mean():.2f}  (summary says {s['score_mean']:.2f})")
    print(f"ci95=[{s['score_ci95'][0]:.2f}, {s['score_ci95'][1]:.2f}]")
    # Same guard as elsewhere: these are PlayTrain frostbite's scoring rule
    # and `aggregate` omits them for other backends (D-028).
    if "win_rate" in s:
        print(f"win_rate={s['win_rate']:.3f}  floes={s['floes_visited_mean']:.2f}")
    print("\nscore : episodes")
    for k in sorted(counts):
        print(f"{k:>6.0f} : {counts[k]}")
    # Recompute from the raw episodes; a mismatch means the file disagrees
    # with itself and the plot would be a lie either way.
    assert abs(scores.mean() - s["score_mean"]) < 1e-9, "summary disagrees with episodes"
    assert len(eps) == s["episodes"], "episode count disagrees with summary"

    # --- the plot --------------------------------------------------------
    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(9.5, 3.6), gridspec_kw={"width_ratios": [2.1, 1]}
    )
    ks = sorted(counts)
    ax.bar(ks, [counts[k] for k in ks], width=8, color=BAR, edgecolor="white", linewidth=0.6)
    ax.axvline(scores.mean(), color=INK, lw=1.4, zorder=3)
    ax.annotate(
        f"mean {scores.mean():.1f}",
        xy=(scores.mean(), ax.get_ylim()[1] * 0.92),
        xytext=(6, 0), textcoords="offset points",
        color=INK, fontsize=9, va="top",
    )
    ax.set_xlabel("raw episode score")
    ax.set_ylabel("episodes")
    ax.set_title(f"{d['run_id']}  ({len(eps)} episodes)", color=INK, fontsize=10, loc="left")

    # Where the run sits between the floor it measured and the game's ceiling.
    ax2.barh([0], [scores.mean()], color=BAR, height=0.45)
    ax2.set_ylim(-0.6, 0.6)
    ax2.set_xlim(0, ceiling * 1.18)
    win_min = 220.0
    for x, label, color, dy in (
        (scores.mean(), f"mean {scores.mean():.1f}", INK, 0.30),
        (win_min, "min win 220", MUTED, -0.34),
        (ceiling, f"ceiling {ceiling:.0f}", CEIL, 0.30),
    ):
        if label.startswith(("min", "ceiling")):
            ax2.axvline(x, color=color, lw=1.3, ls=":" if label.startswith("min") else "-")
        # Labels sit to the RIGHT of their line with room reserved by the
        # 1.18x xlim, so none of them fall off the axis (an earlier version
        # placed them left of the line and they rendered outside it).
        ax2.text(x + ceiling * 0.015, dy, label, color=color, fontsize=8, va="center")
    ax2.set_yticks([])
    ax2.set_xlabel("score vs the game's ceiling")
    ax2.set_title("scale", color=INK, fontsize=10, loc="left")

    for a in (ax, ax2):
        a.spines[["top", "right"]].set_visible(False)
        a.tick_params(colors=MUTED, labelsize=8)
        for sp in a.spines.values():
            sp.set_color(MUTED)

    fig.tight_layout()
    out = args.run_dir / "scores.png"
    fig.savefig(out, dpi=170, facecolor="white")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
