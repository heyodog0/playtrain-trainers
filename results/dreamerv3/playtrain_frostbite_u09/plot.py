"""Regenerate curves.png from metrics_seed*.json and eval_seed*.json; numbers printed on it."""
import glob, json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
here = Path(__file__).parent
fig, ax = plt.subplots(figsize=(11, 5))
for f in sorted(glob.glob(str(here / "metrics_seed*.json"))):
    d = json.load(open(f)); e = d["episodes"]; seed = d["seed"]
    fr = np.array([x["frames"] for x in e]) / 1e3; sc = np.array([x["score"] for x in e], float)
    k = 25; sm = np.convolve(sc, np.ones(k) / k, mode="valid")
    w = [x["score"] for x in e if 350_000 <= x["frames"] < 400_000]
    ev = json.load(open(here / f"eval_seed{seed}.json"))["sampled"]["aggregate"]["score_mean"]
    ax.plot(fr[k - 1:], sm, lw=1.2, label=f"seed {seed}: train last-50k {np.mean(w):.1f}, eval {ev:.1f}")
ax.axvspan(350, 400, color="grey", alpha=0.15)
for y, lab, ls in [(33.5, "random 33.5", ":"), (41.83, "PPO/IMPALA 41.8", "--"), (150.44, "BBF RR=2 v4 150.4", "-."), (203.46, "BBF RR=8 v4 203.5", "-."), (260, "ceiling 260 (WIN)", "-")]:
    ax.axhline(y, color="k", ls=ls, lw=0.7); ax.text(2, y + 3, lab, fontsize=7)
ax.set_xlabel("frames (k)"); ax.set_ylabel("training episode score (25-episode running mean)")
ax.set_title("DreamerV3 (faithful port) on PlayTrain frostbite, 5 seeds; grey = [350k, 400k) window")
ax.legend(fontsize=7, loc="lower right"); ax.set_ylim(0, 275)
fig.tight_layout(); fig.savefig(here / "curves.png", dpi=110); print("wrote", here / "curves.png")
