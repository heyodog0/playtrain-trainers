"""Regenerate arms.png from summary.json plus the committed reference arms. Numbers printed on it."""
import json, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
here = Path(__file__).parent
sys.path.insert(0, str(here.parents[2] / "tools" / "dreamerv3"))
import dv3_gate_summary as G
s = json.loads((here / "summary.json").read_text())["last50k"]
arms = [
    ("published\n(score file)", G.OFFICIAL_LAST50K),
    ("official e3f0224\noriginal 0-4", G.OFFICIAL_HERE_LAST50K),
    ("official e3f0224\nseeds 5-19", s["seeds"]),
    ("official\nA100", s["a100"]),
    ("official\nfloat32", s["fp32"]),
    ("official\n2023 atari-py", s["ataripy"]),
]
fig, ax = plt.subplots(figsize=(12, 5))
for i, (name, v) in enumerate(arms):
    ax.scatter([i] * len(v), v, s=28, alpha=0.75)
    m = sum(v) / len(v)
    ax.hlines(m, i - 0.3, i + 0.3, color="k")
    ax.text(i, 5300, f"mean {m:.0f}\n{sum(x >= 1000 for x in v)}/{len(v)} >=1000", ha="center", fontsize=8)
ax.axhline(1000, color="grey", ls=":", lw=0.8)
ax.set_xticks(range(len(arms))); ax.set_xticklabels([a[0] for a in arms], fontsize=8)
ax.set_ylabel("last-50k training score, Frostbite"); ax.set_ylim(0, 6000)
ax.set_title("Official DreamerV3 on our cluster vs the published seeds (ataripy seed 7 has no window score, see README)")
fig.tight_layout(); fig.savefig(here / "arms.png", dpi=110)
print("wrote", here / "arms.png")
