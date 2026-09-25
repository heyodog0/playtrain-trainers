"""U07 record: what one reset actually does to each parameter group.

The fraction of its own norm a group moves is the cleanest statement of the
shrink-and-perturb split: the two gin-named modules must move half way, and
everything else all the way.

Run: uv run --no-sync python tools/bbf_reset_report.py
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from playtrain_trainers.bbf.config import BBFConfig
from playtrain_trainers.bbf.losses import build_optimizer, build_target
from playtrain_trainers.bbf.net import BBFNetwork
from playtrain_trainers.bbf.resets import reset_network
from playtrain_trainers.bbf.schedules import reset_steps

OUT = Path("results/bbf/resets")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = BBFConfig()
    torch.manual_seed(0)
    net = BBFNetwork(cfg, num_actions=8)
    opt = build_optimizer(net, cfg)
    target = build_target(net)
    # Drive the weights away from initialization so the distances mean
    # something; a reset applied at initialization moves almost nothing.
    with torch.no_grad():
        for p in net.parameters():
            p.mul_(3.0).add_(0.5)

    summary = reset_network(net, cfg, 8, optimizer=opt, target=target)
    groups = {}
    for k, moved in summary["l2_moved_by_group"].items():
        before = summary["l2_before_by_group"][k]
        groups[k] = {
            "l2_before": round(before, 3),
            "l2_moved": round(moved, 3),
            "moved_fraction_of_norm": round(moved / before, 4),
            "rule": (
                "shrink_and_perturb"
                if k in cfg.shrink_perturb_keys
                else "full_reset"
            ),
        }

    record = {
        "shrink_perturb_keys": list(cfg.shrink_perturb_keys),
        "shrink_factor": cfg.shrink_factor,
        "perturb_factor": cfg.perturb_factor,
        "reset_every": cfg.reset_every,
        "no_resets_after": cfg.no_resets_after,
        "reset_steps_rr2": reset_steps(cfg),
        "params_shrunk_and_perturbed": summary["params_shrunk_and_perturbed"],
        "params_fully_reset": summary["params_fully_reset"],
        "optimizer_state_cleared": summary["optimizer_state_cleared"],
        "target_resynced": summary["target_resynced"],
        "groups": groups,
        "note": (
            "predictor moves 0.997 rather than 1.000 because its LayerNorm "
            "gain initializes to exactly 1.0, so part of the old value "
            "coincides with the fresh one. The reset is still complete."
        ),
    }
    (OUT / "resets.json").write_text(json.dumps(record, indent=2) + "\n")

    print(f"shrink-and-perturb keys: {record['shrink_perturb_keys']} "
          f"(shrink {cfg.shrink_factor}, perturb {cfg.perturb_factor})")
    print(f"resets at gradient steps {record['reset_steps_rr2']} (RR=2)")
    print(f"{summary['params_shrunk_and_perturbed']} tensors shrunk-and-perturbed, "
          f"{summary['params_fully_reset']} fully reset")
    print(f"\n{'group':<18}{'L2 before':>12}{'L2 moved':>12}{'moved/before':>14}  rule")
    for k, v in groups.items():
        print(f"{k:<18}{v['l2_before']:>12.2f}{v['l2_moved']:>12.2f}"
              f"{v['moved_fraction_of_norm']:>14.3f}  {v['rule']}")
    print(f"\nwrote {OUT / 'resets.json'}")


if __name__ == "__main__":
    main()
