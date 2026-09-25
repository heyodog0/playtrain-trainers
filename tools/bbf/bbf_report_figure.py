"""The per-game comparison figure, built only from committed metrics files.

MISSION hard rule 4: regenerated from `results/bbf/*/metrics_seed*.json`, with
the table printed before anything is drawn so the two can be compared.

    uv run --no-sync --with matplotlib python tools/bbf/bbf_report_figure.py --game frostbite
    uv run --no-sync --with matplotlib python tools/bbf/bbf_report_figure.py --all

frostbite writes `comparison.png` (the path REPORT.md has always used); every
other game writes `comparison_<game>.png`. An arm whose run directory does not
exist is skipped and says so, rather than drawn as zero.

Color follows the ARM, never its position: BBF RR=2 is the same blue in every
game's figure whether or not RR=8 was run there. The five hues are slots 1-5
of the dataviz reference palette, validated in display order (CVD and
normal-vision separation pass; the sub-3:1 contrast of three of them is
relieved by the x-axis label on every bar and the printed table).
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

OUT = Path("results/bbf")
INK, MUTED, GRID = "#1f2933", "#6b7280", "#c9ccd1"
RANDOM_GRAY = "#9aa0a6"   # a reference level, not a series: neutral, labeled

# (label, run-id template, color). Order is display order.
ARMS = [
    ("random",             "random_{g}",                             RANDOM_GRAY),
    ("IMPALA 100k",        "ref_impala_{g}_100k_sampled",            "#eda100"),
    ("PPO 100k",           "ref_ppo_{g}_100k_sampled",               "#e87ba4"),
    ("BBF RR=2",           "bbf_playtrain_{g}_rr2_v4",               "#2a78d6"),
    ("BBF RR=2\nreset 10k", "bbf_playtrain_{g}_rr2_v4_reset10k",    "#1baf7a"),
    ("BBF RR=8",           "bbf_playtrain_{g}_rr8_v4",               "#eb6834"),
]

# Only frostbite has a ceiling and a WIN state (D-014, D-044).
FROSTBITE_LINES = {"ceiling": 260.0, "min winning score": 220.0}
GAMES = ["frostbite", "venture", "amidar", "star_gunner"]
MIN_SLOTS = 4   # a game with fewer arms keeps the same bar width, left-aligned


def seeds_of(run_id: str):
    """Per-seed means and a bootstrap CI95 over seeds, or None if absent.

    The random baseline is one 100-episode run, so its CI is over episodes.
    """
    single = OUT / run_id / "metrics.json"
    if single.is_file():
        s = json.loads(single.read_text())["summary"]
        return np.array([s["score_mean"]]), s["score_ci95"]
    files = sorted(glob.glob(str(OUT / run_id / "metrics_seed*.json")))
    if not files:
        return None
    v = np.array([json.loads(Path(f).read_text())["summary"]["score_mean"] for f in files])
    b = np.random.default_rng(0).choice(v, (20000, len(v))).mean(1)
    return v, [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def figure(game: str) -> Path:
    rnd = json.loads((OUT / f"random_{game}" / "metrics.json").read_text())["summary"]["score_mean"]
    lines = FROSTBITE_LINES if game == "frostbite" else {}
    print(f"\nPlayTrain {game}, 100k agent steps (random {rnd:.2f})")
    print(f"{'arm':<22}{'n':>3}{'mean':>10}{'CI95 over seeds':>22}{'over random':>13}")
    data = []
    for label, tmpl, col in ARMS:
        got = seeds_of(tmpl.format(g=game))
        if got is None:
            print(f"{label.replace(chr(10), ' '):<22}  - not run for {game}")
            continue
        v, ci = got
        m = float(v.mean())
        data.append((label, tmpl.format(g=game), m, ci, v, col))
        print(f"{label.replace(chr(10), ' '):<22}{len(v):>3}{m:>10.2f}"
              f"{('[' + format(ci[0], '.1f') + ', ' + format(ci[1], '.1f') + ']'):>22}"
              f"{m - rnd:>+13.2f}")

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.2),
                                  gridspec_kw={"width_ratios": [1.25, 1]})

    # Left: final scores, per-seed dots, on the game's own scale.
    for i, (label, _, m, ci, v, col) in enumerate(data):
        ax.bar(i, m, color=col, width=0.62,
               yerr=[[m - ci[0]], [ci[1] - m]], capsize=3,
               error_kw={"lw": 1, "ecolor": INK})
        if len(v) > 1:
            ax.scatter(np.full(len(v), i) + np.linspace(-.14, .14, len(v)), v,
                       s=16, color="white", edgecolor=INK, linewidth=.7, zorder=4)
    top = max(max(d[3][1], d[4].max()) for d in data)
    pad = 0.015 * max(top, max(lines.values(), default=0))
    for name, y in lines.items():
        ax.axhline(y, color=MUTED, lw=1, ls="-" if name == "ceiling" else ":")
        ax.text(-.45, y + pad, f"{name} {y:.0f}", color=MUTED, fontsize=7.5)
    ax.axhline(rnd, color=MUTED, lw=1, ls="--")
    ax.text(-.45, rnd + pad, f"random {rnd:.1f}", color=MUTED, fontsize=7.5)
    ax.set_xlim(-.6, max(len(data), MIN_SLOTS) - .4)
    ax.set_xticks(range(len(data)))
    ax.set_xticklabels([d[0] for d in data], fontsize=8, color=INK)
    ax.set_ylabel("final 100-episode score", color=INK)
    ax.set_title(f"PlayTrain {game} at 100k agent steps\n"
                 "bars = mean, whiskers = CI95 over seeds, dots = seeds",
                 color=INK, fontsize=9.5, loc="left")

    # Right: learning curves for the BBF arms that exist.
    bbf = [d for d in data if d[0].startswith("BBF")]
    for label, run_id, *_, col in bbf:
        for j, f in enumerate(sorted(glob.glob(str(OUT / run_id / "metrics_seed*.json")))):
            c = json.loads(Path(f).read_text()).get("curve") or []
            if c:
                ax2.plot([x["env_step"] for x in c], [x["score_mean"] for x in c],
                         color=col, lw=1.2, alpha=.6,
                         label=label.replace("\n", " ") if j == 0 else None)
    for name, y in lines.items():
        ax2.axhline(y, color=MUTED, lw=1, ls="-" if name == "ceiling" else ":")
    ax2.axhline(rnd, color=MUTED, lw=1, ls="--")
    ax2.set_xlabel("env steps (agent steps)", color=INK)
    ax2.set_ylabel("10-episode eval score", color=INK)
    ax2.set_title("BBF learning curves, one line per seed\n"
                  "(10-episode evals: shape only, too noisy to compare arms)",
                  color=INK, fontsize=9.5, loc="left")
    if len(bbf) > 1:   # one series is named by the title; no legend box
        ax2.legend(frameon=False, fontsize=8, labelcolor=INK, loc="center left")

    for a in (ax, ax2):
        a.spines[["top", "right"]].set_visible(False)
        a.tick_params(colors=MUTED, labelsize=8)
        a.yaxis.grid(True, color=GRID, lw=.5)
        a.set_axisbelow(True)
        for sp in a.spines.values():
            sp.set_color(GRID)
    fig.tight_layout()
    p = OUT / ("comparison.png" if game == "frostbite" else f"comparison_{game}.png")
    fig.savefig(p, dpi=170, facecolor="white")
    plt.close(fig)
    print(f"wrote {p}")
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="frostbite", choices=GAMES)
    ap.add_argument("--all", action="store_true", help="every game in GAMES")
    args = ap.parse_args()
    for g in (GAMES if args.all else [args.game]):
        figure(g)


if __name__ == "__main__":
    main()
