"""Re-score an existing BBF checkpoint. NO retraining.

Phase 2 rule: a re-scored number must never sit unlabelled beside a trained
one. Every metrics file this writes carries `"rescore": {...}` naming the
checkpoint and what was varied.

Its first use (U17) isolates D-040: the v3 runs evaluate the EMA target,
because that is what `BBFAgent.step` acts with under the gin's
`target_action_selection = True`. Evaluating the ONLINE net instead changes
only what is MEASURED, not what was learned, so the difference is attributable
to that one choice without retraining anything.

Run: uv run --no-sync python tools/bbf/bbf_rescore.py RUN_ID --network online
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from playtrain_trainers.bbf.config import config_from_dict, results_dir, set_global_seeds
from playtrain_trainers.bbf.evaluate import (
    epsilon_greedy,
    evaluate_policy,
    policy_rng,
    write_metrics,
)
from playtrain_trainers.bbf.envs import make_env
from playtrain_trainers.bbf.net import BBFNetwork, resolve_device
from playtrain_trainers.bbf.train import greedy_action, measured_random_baseline


def _num_actions(cfg) -> int:
    """Ask the env, exactly as `train.py` does -- never hardcode it."""
    env = make_env(cfg, seed=cfg.seed, training=False)
    try:
        return int(env.action_space.n)
    finally:
        env.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id", help="the trained run whose checkpoints to re-score")
    ap.add_argument("--network", choices=["online", "target"], required=True)
    ap.add_argument("--out-suffix", default=None,
                    help="results dir suffix; defaults to _rescore_<network>")
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument(
        "--max-noops", type=int, default=None,
        help="override the random no-op starts for the EVAL env only. Dopamine's "
             "AtariPreprocessing has none, and with sticky_actions False that "
             "makes the official eval deterministic bar epsilon 0.001 (F-005).")
    args = ap.parse_args()

    suffix = args.out_suffix or f"_rescore_{args.network}"
    if args.max_noops is not None and args.out_suffix is None:
        suffix += f"_noops{args.max_noops}"
    cfg_files = sorted(Path("results/bbf").glob(f"{args.run_id}/config_seed*.json"))
    if not cfg_files:
        raise SystemExit(f"no configs under results/bbf/{args.run_id}/")

    device = resolve_device("auto")
    means = []
    for cf in cfg_files:
        cfg = config_from_dict(json.loads(cf.read_text()))
        if args.max_noops is not None:
            cfg.max_noops = args.max_noops
        seed = cfg.seed
        ckpt = Path("outputs/bbf") / args.run_id / f"seed{seed}" / "latest.pt"
        if not ckpt.is_file():
            raise SystemExit(f"no checkpoint at {ckpt}")
        set_global_seeds(seed)
        ck = torch.load(ckpt, map_location=device, weights_only=False)

        # The two state dicts are structurally identical; only the weights
        # differ (the target is an EMA of the online net).
        net = BBFNetwork(cfg, _num_actions(cfg))
        net.load_state_dict(ck["net" if args.network == "online" else "target"])
        net = net.to(device).eval()

        eps, summary = evaluate_policy(
            cfg,
            lambda e, s: epsilon_greedy(
                lambda o: greedy_action(net, o, device),
                cfg.epsilon_eval,
                int(e.action_space.n),
                policy_rng(cfg, s),
            ),
            n_episodes=args.episodes,
            random_mean=measured_random_baseline(cfg),
        )
        cfg.run_id = args.run_id + suffix
        out = results_dir(cfg)
        write_metrics(
            out / f"metrics_seed{seed}.json", cfg, eps, summary,
            unit="U17-rescore",
            extra={
                "rescore": {
                    "source_run_id": args.run_id,
                    "checkpoint": str(ckpt),
                    "trained_commit": ck.get("commit", "unknown"),
                    "env_step": ck.get("env_step"),
                    "evaluated_network": args.network,
                    "max_noops_override": args.max_noops,
                    "note": ("NOT a training run. The checkpoint is unchanged; "
                             "only the EVALUATION differs -- the network scored "
                             "(D-040) and/or the no-op starts (F-005)."),
                },
            },
        )
        (out / f"config_seed{seed}.json").write_text(json.dumps(cfg.to_dict(), indent=2) + "\n")
        means.append(summary["score_mean"])
        print(f"  seed {seed:>6}  {args.network:<7} {summary['score_mean']:8.2f}")
    print(f"\n{args.run_id}{suffix}: {len(means)} seeds, mean {np.mean(means):.2f} "
          f"(sd {np.std(means, ddof=1):.2f})")


if __name__ == "__main__":
    main()
