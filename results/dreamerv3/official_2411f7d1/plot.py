"""Regenerate curves.png from the committed per-arm JSON. Numbers are printed on it."""
import json, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools" / "dreamerv3"))
import dv3_official_summary as S

here = Path(__file__).parent
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
arms = [("e3f0224 on our cluster (atari100k)", Path(here.parent / "official_jax_frostbite.json")),
        ("2411f7d1 atari_frostbite", here / "atari_frostbite.json"),
        ("2411f7d1 atari100k_frostbite", here / "atari100k_frostbite.json")]
for ax, (name, path) in zip(axes, arms):
    d = json.loads(path.read_text())
    vals = []
    for seed, rec in sorted(d["seeds"].items()):
        f = [x[0] / 1e3 for x in rec["scores"]]; s = [x[1] for x in rec["scores"]]
        v = S.last50k(rec["scores"]); vals.append(v)
        ax.plot(f, s, lw=0.6, alpha=0.8, label=f"seed {seed}: {v:.0f}")
    ax.axvspan(350, 400, color="grey", alpha=0.15)
    ax.axhline(2938.1, color="k", ls="--", lw=0.8)
    ax.set_title(f"{name}\nlast-50k mean {sum(vals)/len(vals):.1f}")
    ax.set_xlabel("frames (k)"); ax.legend(fontsize=7)
axes[0].set_ylabel("training episode score")
fig.text(0.5, -0.02, "dashed: published last-50k 2938.1; grey: gate window [350k, 400k)", ha="center")
fig.tight_layout(); fig.savefig(here / "curves.png", dpi=110, bbox_inches="tight")
print("wrote", here / "curves.png")
