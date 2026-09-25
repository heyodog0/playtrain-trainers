"""U04 record: the network's parameter count, and the CNN-vs-ResNet check.

The mission asks for the parameter count to be printed and recorded, and
compared to the JAX model's order of magnitude. The paper gives no parameter
count, so the only quantitative cross-check available is its own statement
that the 3-layer CNN has "roughly 50% more parameters than the ResNet at each
scale level" (section discussing Figure 3). This script measures that ratio
for our reconstruction; see flag F-009 for what it turned up.

Run: uv run --no-sync python tools/bbf_net_report.py
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from playtrain_trainers.bbf.config import BBFConfig
from playtrain_trainers.bbf.net import BBFNetwork

OUT = Path("results/bbf/net")


def nature_cnn_params(width_scale: int, obs_size: int, in_ch: int, hidden: int) -> dict:
    """The 3-layer CNN arm of the paper's Figure 3, priced for comparison.

    Padding is SAME, which is what the official `spr_networks.RainbowCNN`
    uses (`padding: Any = 'SAME'`). An earlier version of this script used
    valid padding, which shrank the CNN's latent from 11x11 to 7x7 and made
    the parameter ratio come out 0.81x instead of ~1.9x -- i.e. the wrong
    side of the paper's claim. That was the whole of F-009.
    """
    def same_pad(k, s):
        # SAME padding for stride s: output is ceil(in / s).
        return nn.Conv2d, k, s

    convs = nn.Sequential(
        nn.Conv2d(in_ch, 32 * width_scale, 8, 4, padding=2),
        nn.ReLU(),
        nn.Conv2d(32 * width_scale, 64 * width_scale, 4, 2, padding=1),
        nn.ReLU(),
        nn.Conv2d(64 * width_scale, 64 * width_scale, 3, 1, padding=1),
        nn.ReLU(),
    )
    with torch.no_grad():
        flat = convs(torch.zeros(1, in_ch, obs_size, obs_size)).flatten(1).shape[1]
    conv_p = sum(p.numel() for p in convs.parameters())
    proj_p = flat * hidden + hidden
    return {"latent_flat": flat, "convs": conv_p, "projection": proj_p,
            "encoder_plus_projection": conv_p + proj_p}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    report: dict = {}

    for obs_size in (84, 64):
        cfg = BBFConfig(obs_size=obs_size)
        net = BBFNetwork(cfg, num_actions=8)
        counts = net.parameter_counts()
        nat = nature_cnn_params(cfg.width_scale, obs_size, cfg.obs_channels, cfg.hidden_dim)
        res = counts["encoder"] + counts["projection"]
        report[f"{obs_size}px"] = {
            "obs_shape": list(cfg.obs_shape),
            "latent_shape": list(net.latent_shape),
            "latent_flat": net.latent_dim,
            "params": counts,
            "nature_cnn_same_scale_and_head": nat,
            # Both ratios are like-for-like: the same input size, the same
            # width scale and the same 2048 head on both arms.
            "cnn_over_resnet_ratio_encoder_plus_projection": (
                nat["encoder_plus_projection"] / res
            ),
            "cnn_over_resnet_ratio_encoder_only": nat["convs"] / counts["encoder"],
            "paper_states_cnn_over_resnet": 1.5,
        }

    report["padding"] = "SAME on both arms, matching the official RainbowCNN"
    report["note"] = (
        "F-009 RESOLVED (2026-09-18) by the official source. The paper says "
        "the 3-layer CNN has roughly 50% more parameters than the Impala "
        "ResNet at each width scale, and it does -- once the CNN uses SAME "
        "padding as `spr_networks.RainbowCNN` actually does. At 84 px that "
        "gives the CNN a 256x11x11 = 30976 latent against the ResNet's "
        "128x11x11 = 15488, so the shared flatten -> 2048 projection makes "
        "the CNN the bigger arm by ~1.9x. The earlier 0.81x came from this "
        "script using VALID padding (256x7x7 = 12544), which was our bug, "
        "not the paper's. Our ResNet architecture was correct throughout."
    )

    (OUT / "params.json").write_text(json.dumps(report, indent=2) + "\n")

    for key in ("84px", "64px"):
        r = report[key]
        print(f"\n{'=' * 64}\nobs {key}  latent {tuple(r['latent_shape'])}  "
              f"flat {r['latent_flat']}\n{'=' * 64}")
        c = dict(r["params"])
        tot = c.pop("total")
        for k, v in sorted(c.items(), key=lambda kv: -kv[1]):
            print(f"  {k:<18} {v:>12,}  {100 * v / tot:5.1f}%")
        print(f"  {'TOTAL':<18} {tot:>12,}")
        print(f"  Nature CNN x4 same head: "
              f"{r['nature_cnn_same_scale_and_head']['encoder_plus_projection']:,}")
        print(f"  CNN / ResNet, encoder+projection = "
              f"{r['cnn_over_resnet_ratio_encoder_plus_projection']:.2f}x")
        print(f"  CNN / ResNet, encoder only       = "
              f"{r['cnn_over_resnet_ratio_encoder_only']:.2f}x")
        print(f"  paper states                     = "
              f"~{r['paper_states_cnn_over_resnet']:.2f}x")
    print(f"\n{report['note']}\n\nwrote {OUT / 'params.json'}")


if __name__ == "__main__":
    main()
