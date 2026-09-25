"""BBF training loop.

    uv run --no-sync python -m playtrain_trainers.bbf.train \
        --config configs/bbf/frostbite_rr2.json

One env, one GPU, `gradient_steps_per_env_step` optimizer steps per env step,
each on its own batch of `batch_size`, each followed by its own EMA target
update -- which is what the official `train` does inside its `lax.scan` over
`batches_to_group` batches (D-039). Two clocks (D-033): resets are checked
once per ENV step against `reset_every` / `no_resets_after`; the n-step and
discount anneals read `cycle_grad_steps`, the GRADIENT steps since the last
reset. The network that acts in the env AND is evaluated is the EMA target
when `target_action_selection` is set, as in `BBFAgent.step` (D-040).

What the run writes:
    outputs/bbf/<run_id>/seed<N>/   tb/, checkpoints, not committed
    results/bbf/<run_id>/           config.json + metrics.json, committed
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from playtrain_trainers.bbf.config import (
    BBFConfig,
    load_config,
    results_dir,
    run_dir,
    set_global_seeds,
)
from playtrain_trainers.bbf.envs import make_env
from playtrain_trainers.bbf.evaluate import (
    epsilon_greedy,
    evaluate_policy,
    git_commit,
    policy_rng,
    write_metrics,
)
from playtrain_trainers.bbf.losses import (
    acts_with_target,
    augment,
    build_optimizer,
    build_target,
    c51_loss,
    c51_target_distribution,
    combined_loss,
    ema_update,
    select_bootstrap_action,
    spr_loss,
    to_float,
)
from playtrain_trainers.bbf.net import BBFNetwork, resolve_device
from playtrain_trainers.bbf.replay import SubsequenceReplayBuffer
from playtrain_trainers.bbf.resets import reset_network
from playtrain_trainers.bbf.schedules import ResetSchedule, discount, epsilon, update_horizon

log = logging.getLogger(__name__)


def measured_random_baseline(cfg: BBFConfig) -> float | None:
    """D-008's measured floor, read back from the U03 metrics file.

    PROTOCOL section 5 B's normalized progress is measured from the PlayTrain
    random baseline, so it needs that number. Reading it from
    `results/bbf/random_<game>/metrics.json` keeps ONE source of truth rather
    than hardcoding 33.50 in a second place and letting the two drift.
    Returns None when the file is absent (so a run still works before the
    baseline has been measured) or for any non-PlayTrain env, where the
    quantity is meaningless.

    Per GAME, not per backend (D-044): the floor is a property of the game and
    its action set, so frostbite's 33.50 says nothing about venture's.
    """
    if cfg.env_backend != "playtrain":
        return None
    path = Path(cfg.results_root) / f"random_{cfg.game}" / "metrics.json"
    try:
        return float(json.loads(path.read_text())["summary"]["score_mean"])
    except Exception:
        log.warning("no random baseline at %s; normalized progress will be omitted", path)
        return None


def newest_frame(obs: np.ndarray, frame_channels: int) -> np.ndarray:
    """The newest frame of a stacked observation, for the replay ring.

    The wrapper stacks oldest-first on the channel axis, so the newest frame
    is the LAST `frame_channels` channels.
    """
    return obs[-frame_channels:].copy()


@torch.no_grad()
def greedy_action(net: BBFNetwork, obs: np.ndarray, device: torch.device) -> int:
    was_training = net.training
    net.eval()
    try:
        x = torch.from_numpy(obs).unsqueeze(0).to(device)
        return int(net.q_values(x).argmax(dim=1).item())
    finally:
        net.train(was_training)


def update_step(
    net: BBFNetwork,
    target: BBFNetwork,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    cfg: BBFConfig,
    gamma: float,
) -> dict[str, float]:
    """One optimizer step on one batch, then one EMA target update."""
    net.train()
    obs = augment(to_float(batch["obs"]), cfg)
    next_obs = augment(to_float(batch["next_obs"]), cfg)

    out = net.forward_with_spr(obs, batch["spr_actions"])

    with torch.no_grad():
        target_next = target(next_obs)
        # Only compute the ONLINE next-state Q when it is actually the
        # selector. At the gin's `target_action_selection = True` it is not,
        # and encoding the next observation twice is ~12% of the update's
        # encoder work for a tensor that is then discarded (D-022).
        online_next_q = (
            target_next["q"]
            if cfg.target_action_selection
            else net.q_values(next_obs)
        )
        a_star = select_bootstrap_action(online_next_q, target_next["q"], cfg)
        target_probs = c51_target_distribution(
            target_next["probs"],
            a_star,
            batch["n_step_return"],
            batch["discount"],
            batch["done"],
            net.support,
        )
    rl_loss, rl_per_sample = c51_loss(out["logits"], batch["action"], target_probs)

    # SPR targets: the EMA network encodes and projects the real future
    # observations. Flattened over (batch, jump) for one forward pass.
    b, j = batch["spr_obs"].shape[:2]
    spr_flat = augment(to_float(batch["spr_obs"].flatten(0, 1)), cfg)
    with torch.no_grad():
        spr_targets = target.target_projections(spr_flat).view(b, j, -1)
    spr, spr_per_sample = spr_loss(out["spr_predictions"], spr_targets, batch["spr_mask"])

    # Importance weights on the C51 + SPR sum, as the official loss_fn (D-035).
    loss = combined_loss(rl_per_sample, spr_per_sample, batch["weights"], cfg.spr_weight)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    ema_update(target, net, cfg.target_update_tau)

    return {
        "loss": float(loss.detach()),
        "rl_loss": float(rl_loss.detach()),
        "spr_loss": float(spr.detach()),
        "per_sample": rl_per_sample.detach(),
    }


def train(cfg: BBFConfig) -> dict[str, Any]:
    """Run the protocol and return the metrics payload."""
    cfg.validate()
    set_global_seeds(cfg.seed)
    device = resolve_device(cfg.device)
    out_dir = run_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = _make_writer(out_dir)

    env = make_env(cfg, seed=cfg.seed, training=True)
    num_actions = int(env.action_space.n)
    frame_channels = 1 if cfg.obs_mode == "grayscale" else 3

    net = BBFNetwork(cfg, num_actions).to(device)
    target = build_target(net)
    optimizer = build_optimizer(net, cfg)
    # Which network acts in the env (D-032). `target` is an EMA of `net`, so
    # this is a reference, not a copy -- it tracks as training proceeds.
    acting_net = target if acts_with_target(cfg) else net
    buf = SubsequenceReplayBuffer(cfg)
    rng = np.random.default_rng(cfg.seed)

    log.info(
        "device=%s actions=%d params=%s obs=%s",
        device, num_actions, f"{net.parameter_counts()['total']:,}", cfg.obs_shape,
    )

    obs, info = env.reset(seed=int(rng.integers(1 << 30)))
    gradient_step = 0
    cycle_grad_steps = 0          # gradient steps since the last reset
    reset_sched = ResetSchedule(cfg)
    pending_episode_start = False  # a truncation ended the last episode
    episode_returns: list[float] = []
    loss_history: list[dict[str, float]] = []
    curve: list[dict[str, Any]] = []
    resets: list[dict[str, Any]] = []
    started = time.time()

    for env_step in range(cfg.training_steps):
        eps = epsilon(env_step, cfg)
        if rng.random() < eps:
            action = int(rng.integers(num_actions))
        else:
            # D-032/D-040: the gin's `target_action_selection = True` makes
            # the BEHAVIOUR policy the EMA target network. This is independent
            # of the Double-DQN bootstrap, which the online net selects.
            action = greedy_action(acting_net, obs, device)

        next_obs, reward, terminated, truncated, info = env.step(action)
        buf.add(
            newest_frame(obs, frame_channels),
            action,
            float(reward),
            bool(terminated),
            episode_start=pending_episode_start,
        )
        pending_episode_start = False
        obs = next_obs
        if terminated or truncated:
            if info.get("real_done", True):
                episode_returns.append(float(info.get("episode_score", 0.0)))
            # A terminal already starts a new episode id in the buffer; a
            # truncation without one has to be flagged or the next episode's
            # frames would be stitched onto this one.
            pending_episode_start = bool(truncated and not terminated)
            obs, info = env.reset(seed=int(rng.integers(1 << 30)))

        # --- learn: one optimizer step per batch (D-039) ---
        if env_step + 1 >= cfg.min_replay_history:
            for _ in range(cfg.gradient_steps_per_env_step):
                n = update_horizon(cycle_grad_steps, cfg)
                gamma = discount(cycle_grad_steps, cfg)
                if not buf.can_sample(cfg.batch_size, n):
                    break
                batch = buf.sample(cfg.batch_size, n, gamma, rng, device)
                stats = update_step(net, target, optimizer, batch, cfg, gamma)
                buf.update_priorities(
                    batch["indices"].cpu().numpy(), stats.pop("per_sample").cpu().numpy()
                )
                gradient_step += 1
                cycle_grad_steps += 1
                stats["n"] = n
                stats["gamma"] = gamma
                loss_history.append({"gradient_step": gradient_step, **stats})

        # --- reset: checked once per ENV step, as `_train_step` does (D-033) ---
        if reset_sched.due(env_step) and reset_sched.fire(env_step):
            summary = reset_network(
                net, cfg, num_actions, optimizer=optimizer, target=target
            )
            summary["env_step"] = env_step
            summary["gradient_step"] = gradient_step
            resets.append(summary)
            cycle_grad_steps = 0
            log.info("reset at env step %d (gradient step %d)", env_step, gradient_step)

        if cfg.log_every and (env_step + 1) % cfg.log_every == 0 and loss_history:
            _log_scalars(writer, env_step + 1, loss_history[-1], episode_returns, eps)

        if cfg.eval_every and (env_step + 1) % cfg.eval_every == 0:
            point = _curve_point(cfg, acting_net, device, env_step + 1, gradient_step)
            curve.append(point)
            log.info("curve @ %d env steps: mean %.2f", env_step + 1, point["score_mean"])
            if writer is not None:
                writer.add_scalar("eval/curve_score_mean", point["score_mean"], env_step + 1)

        if cfg.checkpoint_every and (env_step + 1) % cfg.checkpoint_every == 0:
            _save_checkpoint(out_dir, net, target, optimizer, env_step + 1, gradient_step)

    wall = time.time() - started
    sps = cfg.training_steps / wall if wall > 0 else float("nan")
    log.info("training done: %d env steps in %.1f s (%.1f steps/s)",
             cfg.training_steps, wall, sps)
    env.close()

    # --- the reported number: the final 100-episode eval (PROTOCOL section 2) ---
    # D-040: evaluated with the network that ACTS -- the EMA target under the
    # gin's `target_action_selection = True`, as `BBFAgent.step` does in eval
    # mode. Before v3 the online net was evaluated instead.
    final_eps, final_summary = evaluate_policy(
        cfg,
        lambda e, s: epsilon_greedy(
            lambda o: greedy_action(acting_net, o, device),
            cfg.epsilon_eval,
            int(e.action_space.n),
            policy_rng(cfg, s),
        ),
        random_mean=measured_random_baseline(cfg),
        progress_every=25,
    )

    _save_checkpoint(out_dir, net, target, optimizer, cfg.training_steps, gradient_step)
    res_dir = results_dir(cfg)
    payload = write_metrics(
        res_dir / f"metrics_seed{cfg.seed}.json",
        cfg,
        final_eps,
        final_summary,
        unit="U09" if cfg.env_backend == "ale" else "U10",
        extra={
            "wall_seconds": round(wall, 1),
            "env_steps_per_s": round(sps, 2),
            "gradient_steps": gradient_step,
            "device": str(device),
            "parameter_counts": net.parameter_counts(),
            "train_episode_returns": episode_returns,
            "curve": curve,
            "resets": [
                {"env_step": r["env_step"],
                 "gradient_step": r["gradient_step"],
                 "l2_moved_by_group": r["l2_moved_by_group"],
                 "target_l2_moved_by_group": r.get("target_l2_moved_by_group")}
                for r in resets
            ],
            "acting_network": "target" if acts_with_target(cfg) else "online",
            "loss_history": loss_history[:: max(1, len(loss_history) // 2000)],
            "replay": {"size": len(buf), "total_added": buf.total_added},
        },
    )
    (res_dir / f"config_seed{cfg.seed}.json").write_text(
        json.dumps(cfg.to_dict(), indent=2) + "\n"
    )
    if writer is not None:
        writer.close()
    return payload


def _curve_point(cfg, net, device, env_step, gradient_step) -> dict[str, Any]:
    """A cheap 10-episode eval for the learning curve (D-006)."""
    eps, summary = evaluate_policy(
        cfg,
        lambda e, s: epsilon_greedy(
            lambda o: greedy_action(net, o, device),
            cfg.epsilon_eval,
            int(e.action_space.n),
            policy_rng(cfg, s),
        ),
        n_episodes=cfg.eval_episodes_curve,
    )
    point = {
        "env_step": env_step,
        "gradient_step": gradient_step,
        "episodes": len(eps),
        "score_mean": summary["score_mean"],
        "scores": [e.score for e in eps],
    }
    # win_rate and floes are PlayTrain frostbite's scoring rule, and
    # `aggregate` omits them for other backends (D-028). Reading them
    # unconditionally crashed every ALE curve eval.
    for k in ("win_rate", "floes_visited_mean", "score_over_random"):
        if k in summary:
            point[k] = summary[k]
    return point


def _make_writer(out_dir: Path):
    try:
        from torch.utils.tensorboard import SummaryWriter

        return SummaryWriter(str(out_dir / "tb"))
    except Exception as exc:  # pragma: no cover - tensorboard is optional here
        log.warning("TensorBoard unavailable (%s); continuing without it", exc)
        return None


def _log_scalars(writer, step, last, returns, eps) -> None:
    if writer is None:
        return
    for k in ("loss", "rl_loss", "spr_loss", "n", "gamma"):
        writer.add_scalar(f"train/{k}", last[k], step)
    writer.add_scalar("train/epsilon", eps, step)
    if returns:
        writer.add_scalar("train/episode_score_mean", float(np.mean(returns[-20:])), step)


def _save_checkpoint(out_dir, net, target, optimizer, env_step, gradient_step) -> None:
    torch.save(
        {
            "net": net.state_dict(),
            "target": target.state_dict(),
            "optimizer": optimizer.state_dict(),
            "env_step": env_step,
            "gradient_step": gradient_step,
            "commit": git_commit(),
        },
        out_dir / "latest.pt",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=None, help="override the config seed")
    ap.add_argument("--run-id", default=None, help="override the config run_id")
    args = ap.parse_args()
    logging.basicConfig(
        format="[%(levelname)s %(asctime)s] %(message)s", level=logging.INFO
    )
    cfg = load_config(args.config)
    if args.seed is not None:
        cfg.seed = args.seed
    if args.run_id is not None:
        cfg.run_id = args.run_id
    payload = train(cfg)
    s = payload["summary"]
    line = (
        f"\nfinal {s['episodes']}-episode eval: mean {s['score_mean']:.2f} "
        f"CI95 [{s['score_ci95'][0]:.2f}, {s['score_ci95'][1]:.2f}]"
    )
    # win_rate and floes are PlayTrain frostbite's scoring rule; `aggregate`
    # omits them for other backends (D-028). This print read them
    # unconditionally and crashed the ALE arm AFTER a full 100k-step run had
    # already written its metrics -- a cosmetic line failing a 1.5 h job.
    # Each field independently, because D-044 decoupled them: a game can have
    # a WIN state without frostbite's floe rule. Coupling them is what made
    # this cosmetic line fail a finished 1.5 h job once already.
    if "win_rate" in s:
        line += f" win_rate {s['win_rate']:.3f}"
    if "floes_visited_mean" in s:
        line += f" floes {s['floes_visited_mean']:.2f}"
    if "score_over_random" in s:
        line += f" over_random {s['score_over_random']:+.2f}"
    print(line)


if __name__ == "__main__":
    main()
