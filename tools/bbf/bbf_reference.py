"""U11: train a PPO or IMPALA reference arm and score it like every other arm.

    uv run --no-sync python tools/bbf/bbf_reference.py \
        --trainer impala --seed 1 --steps 100000

Trains with the EXISTING trainer on the BBF wrapper stack, then throws away
that trainer's own eval and re-scores the final checkpoint with
`bbf.evaluate` on the shared 100-episode seed pool (D-017). Both arms and BBF
therefore differ only in the learning algorithm, not in the env, the budget
or the way the score is computed.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

from playtrain_trainers.bbf.config import BBFConfig, results_dir, set_global_seeds
from playtrain_trainers.bbf.evaluate import (
    epsilon_greedy,
    evaluate_policy,
    policy_rng,
    write_metrics,
)
from playtrain_trainers.bbf.net import resolve_device
from playtrain_trainers.bbf.reference import run_impala, run_ppo

log = logging.getLogger(__name__)


def load_impala_policy(ckpt_path: Path, cfg: BBFConfig, n_actions: int, device,
                       mode: str = "greedy", rng_seed: int = 0):
    """Policy from an IMPALA `final.pt` (`model_state_dict`, ImpalaNet).

    `mode="sample"` draws from the policy instead of taking its argmax. That
    is the right eval for a policy-gradient method whose policy is still
    close to uniform: the argmax of near-equal logits is arbitrary, and both
    reference arms were observed collapsing to a single constant action
    (F-014).
    """
    from playtrain_trainers.impala.net import ImpalaNet

    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    c = ck["config"]
    model = ImpalaNet(
        observation_shape=tuple(c["obs_shape"]),
        num_actions=c["num_actions"] or n_actions,
        features_dim=c.get("features_dim", 256),
        use_lstm=c.get("use_lstm", False),
        net=c.get("net", "impala"),
    ).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    if c.get("use_lstm", False):
        raise SystemExit("LSTM reference arms are not wired up; use a feedforward config")

    gen = torch.Generator(device="cpu").manual_seed(rng_seed)

    @torch.no_grad()
    def act(obs: np.ndarray) -> int:
        # ImpalaNet takes a time-major dict with [T, B, C, H, W]; the eval env
        # already hands back CHW, which is what the net was trained on (PPO
        # and IMPALA both transpose HWC->CHW inside their loops).
        frame = torch.from_numpy(obs).to(device).view((1, 1) + obs.shape)
        out, _ = model({
            "frame": frame,
            "reward": torch.zeros(1, 1, device=device),
            "done": torch.zeros(1, 1, dtype=torch.bool, device=device),
            "last_action": torch.zeros(1, 1, dtype=torch.int64, device=device),
        })
        logits = out["policy_logits"].view(-1)
        if mode == "greedy":
            return int(logits.argmax().item())
        probs = torch.softmax(logits, dim=-1).cpu()
        return int(torch.multinomial(probs, 1, generator=gen).item())

    return act


def load_ppo_policy(ckpt_path: Path, cfg: BBFConfig, n_actions: int, device,
                    mode: str = "greedy", rng_seed: int = 0):
    """Policy from a PPO `final.pt` (`model`, ActorCritic). See F-014 on modes."""
    from playtrain_trainers.policy import ActorCritic

    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    c = ck["config"]
    model = ActorCritic(
        n_actions=n_actions,
        in_channels=cfg.obs_channels,
        input_hw=cfg.obs_size,
        net=c.get("net", "impala"),
        use_lstm=c.get("use_lstm", False),
    ).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    gen = torch.Generator(device="cpu").manual_seed(rng_seed)

    @torch.no_grad()
    def act(obs: np.ndarray) -> int:
        x = torch.from_numpy(obs).to(device).unsqueeze(0)  # [1, C, H, W]
        # ActorCritic.forward returns (Categorical, value), not logits -- so
        # greedy means the argmax of the distribution's own logits, not of the
        # first return value.
        dist, _ = model(x)
        if mode == "greedy":
            return int(dist.logits.view(-1).argmax().item())
        probs = dist.probs.view(-1).cpu()
        return int(torch.multinomial(probs, 1, generator=gen).item())

    return act


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trainer", required=True, choices=["ppo", "impala"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--game", default="frostbite",
                    help="PlayTrain game; the default keeps the U11 run ids")
    ap.add_argument("--steps", type=int, default=100_000,
                    help="AGENT steps; both trainers count in this unit (D-027)")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--n-envs", type=int, default=8, help="PPO only")
    ap.add_argument("--policy", default="greedy", choices=["greedy", "sample"],
                    help="how to act at eval; see F-014")
    ap.add_argument("--skip-train", action="store_true",
                    help="re-score an existing checkpoint without retraining")
    ap.add_argument("--ckpt-run-id", default=None,
                    help="run_id whose checkpoint to load; defaults to --run-id. "
                         "Needed because a re-score writes to a NEW run_id "
                         "(e.g. *_sampled) while reading the ORIGINAL run's "
                         "checkpoint -- deriving both from one id made "
                         "--skip-train look for a checkpoint it had just "
                         "renamed away from.")
    args = ap.parse_args()
    logging.basicConfig(format="[%(levelname)s %(asctime)s] %(message)s", level=logging.INFO)

    run_id = args.run_id or f"ref_{args.trainer}_{args.game}_100k"
    cfg = BBFConfig(run_id=run_id, seed=args.seed, game=args.game,
                    training_steps=args.steps)
    cfg.validate()
    set_global_seeds(args.seed)
    device = resolve_device(cfg.device)

    ckpt_run_id = args.ckpt_run_id or run_id
    out_dir = Path(cfg.output_root).parent / "bbf_ref" / ckpt_run_id / f"seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    if args.skip_train:
        log.info("--skip-train: re-scoring the existing checkpoint only")
        wall = 0.0
    else:
        if args.trainer == "impala":
            run_impala(cfg, args.seed, str(out_dir), args.steps)
        else:
            run_ppo(cfg, args.seed, str(out_dir), args.steps, n_envs=args.n_envs)
        wall = time.time() - t0
        log.info("%s training done in %.1f s", args.trainer, wall)

    ckpt = out_dir / "final.pt"
    if not ckpt.is_file():
        raise SystemExit(f"no checkpoint at {ckpt}")

    # Re-score with the shared evaluator so this arm is commensurable with
    # BBF and the random baseline: same 100 seeds, same whole-game episodes.
    loader = load_impala_policy if args.trainer == "impala" else load_ppo_policy
    n_actions = 8
    policy = loader(ckpt, cfg, n_actions, device, mode=args.policy,
                    rng_seed=args.seed)
    episodes, summary = evaluate_policy(
        cfg,
        lambda e, s: epsilon_greedy(policy, cfg.epsilon_eval, int(e.action_space.n),
                                    policy_rng(cfg, s)),
        random_mean=json.loads(
            (Path(cfg.results_root) / f"random_{args.game}" / "metrics.json").read_text()
        )["summary"]["score_mean"],
        progress_every=25,
    )

    res = results_dir(cfg)
    write_metrics(
        res / f"metrics_seed{args.seed}.json", cfg, episodes, summary,
        unit="U11",
        extra={
            "trainer": args.trainer,
            "eval_policy": args.policy,
            "wall_seconds": round(wall, 1),
            "device": str(device),
            "agent_steps": args.steps,
            "n_envs": args.n_envs if args.trainer == "ppo" else 1,
            "note": (
                "Trained by the existing trainer on the BBF wrapper stack "
                "(D-026) at 100k AGENT steps (D-027), then re-scored by "
                "bbf.evaluate on the shared eval seed pool so the number is "
                "comparable with the BBF arms."
            ),
        },
    )
    (res / f"config_seed{args.seed}.json").write_text(json.dumps(cfg.to_dict(), indent=2) + "\n")
    # frostbite-only fields are absent for other games (D-044); a cosmetic
    # print reading them unconditionally failed 18 finished runs.
    line = (f"\n{args.trainer} @ {args.steps} agent steps, seed {args.seed}: "
            f"mean {summary['score_mean']:.2f} "
            f"CI95 [{summary['score_ci95'][0]:.2f}, {summary['score_ci95'][1]:.2f}]")
    if "win_rate" in summary:
        line += f" win {summary['win_rate']:.3f}"
    if "floes_visited_mean" in summary:
        line += f" floes {summary['floes_visited_mean']:.2f}"
    if "score_over_random" in summary:
        line += f" over_random {summary['score_over_random']:+.2f}"
    print(line)


if __name__ == "__main__":
    main()
