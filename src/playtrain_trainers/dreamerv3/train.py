"""The training loop — a port of ``embodied/run/train.py`` and ``dreamerv3/main.py``.

Run it as::

    uv run --no-sync python -m playtrain_trainers.dreamerv3.train \
        --config configs/dreamerv3/ale_frostbite.json --seed 0

Structure, following ``run/train.py``: one env, a driver that steps it with the
SAMPLED policy, ``replay.add`` on every transition, and a ratio accumulator deciding
how many gradient steps to take after each one.

**Three clocks on every line** (MISSION rule 3). ``frames`` counts environment
frames, ``agent_steps`` counts post-repeat decisions, ``grad_steps`` counts gradient
steps. The achieved ratio is asserted at every report against the configured 0.25 --
on the POST-WARMUP clock, because the full-run figure is legitimately ~0.9 % low
(the 1024-step warmup is not credited and there is no catch-up burst; see
``reference/elements_when_ratio.md``). Both are logged.
"""

from __future__ import annotations

import dataclasses

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from playtrain_trainers.dreamerv3 import config as config_
from playtrain_trainers.dreamerv3 import envs as envs_
from playtrain_trainers.dreamerv3 import replay as replay_
from playtrain_trainers.dreamerv3.agent import Agent


class Ratio:
    """``elements.when.Ratio``, byte-for-byte.

    Returns how many gradient steps to take now, carrying the fractional remainder
    exactly so it cannot drift. The first call returns 1.
    """

    def __init__(self, ratio: float):
        self._ratio = ratio
        self._prev: float | None = None

    def __call__(self, step: int) -> int:
        step = int(step)
        if self._ratio == 0:
            return 0
        if self._ratio < 0:
            return 1
        if self._prev is None:
            self._prev = float(step)
            return 1
        repeats = int((step - self._prev) * self._ratio)
        self._prev += repeats / self._ratio
        return repeats


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def host(v: torch.Tensor) -> np.ndarray:
    """To numpy for replay. bf16 entries go to float32, as `embodied/jax/agent.py:402`
    does with `np.float32(x) if x.dtype == bfloat16`."""
    v = v.detach()
    if v.dtype == torch.bfloat16:
        v = v.float()
    return v.cpu().numpy()


def save_recon_png(path: Path, obs_images: np.ndarray, recon: np.ndarray) -> None:
    """A strip of real frames over their reconstructions, for the by-eye check."""
    from PIL import Image

    n = min(8, obs_images.shape[0])
    real = np.concatenate([obs_images[i] for i in range(n)], 1)
    pred = np.clip(recon[:n] * 255, 0, 255).astype(np.uint8)
    pred = np.concatenate([pred[i] for i in range(n)], 1)
    grid = np.concatenate([real, pred], 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid).resize((grid.shape[1] * 3, grid.shape[0] * 3), Image.NEAREST).save(path)


def run(
    cfg: config_.DreamerConfig,
    seed: int,
    outdir: Path,
    steps: int | None = None,
    make_world_model=None,
) -> dict[str, Any]:
    """One seed, start to finish. Returns the metrics record that gets saved."""
    config_.set_global_seeds(seed)
    device = pick_device(cfg.device)
    if device.type != "cuda" and cfg.compute_dtype == "bfloat16":
        # D-012: off CUDA (the local MPS/CPU smoke only) run float32. The smoke checks
        # shapes and finiteness, not cluster numerics.
        cfg = dataclasses.replace(cfg, compute_dtype="float32")
    steps = steps or cfg.run.steps
    outdir.mkdir(parents=True, exist_ok=True)

    env = envs_.make_env(cfg, seed)
    action_dim = env.act_space["action"][3]
    obs_shape = (cfg.env.size[0], cfg.env.size[1], 1 if cfg.env.gray else 3)
    # The loop never names a world model; it passes whatever it was given (U10).
    wm = make_world_model(obs_shape, action_dim, cfg) if make_world_model else None
    agent = Agent(obs_shape, action_dim, cfg, world_model=wm).to(device)

    rp = replay_.Replay(
        length=cfg.sequence_length,
        capacity=int(cfg.replay.size),
        chunksize=cfg.replay.chunksize,
        online=cfg.replay.online,
        seed=seed,
    )

    frames = 0
    agent_steps = 0
    grad_steps = 0
    warmup = cfg.train_warmup_steps
    should_train = Ratio(cfg.train_ratio_per_agent_step)
    carry = agent.init_policy(1)
    episode_rng = np.random.default_rng(seed)

    episodes: list[dict[str, Any]] = []
    log_lines: list[dict[str, Any]] = []
    ep_score = 0.0
    ep_steps = 0
    started = time.time()
    last_report = started
    warmup_seconds: float | None = None
    last_batch: dict[str, torch.Tensor] | None = None
    pending_batch: dict[str, np.ndarray] | None = None  # the Prefetch(amount=1) slot
    pending_updates: dict[str, np.ndarray] | None = None  # agent.train's lagged outs

    # The driver loop, shaped exactly like `embodied/core/driver.py`: ONE `env.step`
    # per iteration, and the observation it returns is always stored -- including the
    # `is_first` one and, critically, the `is_last` one.
    #
    # An earlier version of this loop stored the CURRENT observation and then stepped,
    # which meant the terminal observation was replaced by the reset before the next
    # iteration could store it. No `is_terminal` step ever entered the replay, so the
    # continue head's target was the constant `1 - 1/horizon` for the whole run (its
    # loss sat at exactly H(1 - 1/333) = 0.02044), imagined rollouts never terminated,
    # and the agent was never told that dying is bad. That was the U08 gate failure.
    env.seed_episode(int(episode_rng.integers(1 << 30)))
    acts = {"action": 0, "reset": True}

    while agent_steps < steps:
        obs = env.step(acts)
        torch_obs = {
            # np.ascontiguousarray: the runtime hands back a read-only view.
            "image": torch.as_tensor(np.ascontiguousarray(obs["image"]), device=device).unsqueeze(0),
            "is_first": torch.as_tensor(np.array([obs["is_first"]]), device=device),
        }
        carry, act, entries = agent.policy(carry, torch_obs, mode="train")
        # `driver._step`: actions are masked to zero on the terminal transition, so
        # the stored action never claims the agent acted after the episode ended.
        action = 0 if obs["is_last"] else int(act.item())

        rp.add(
            {
                "image": obs["image"],
                "reward": np.float32(obs["reward"]),
                "is_first": np.bool_(obs["is_first"]),
                "is_last": np.bool_(obs["is_last"]),
                "is_terminal": np.bool_(obs["is_terminal"]),
                "action": np.int32(action),
                # Whatever the world model declared as `entry_keys` (U10), not a
                # hardcoded deter/stoch pair.
                **{k: host(v[0]) for k, v in entries.items()},
            }
        )
        agent_steps += 1
        # The official logger multiplies the step clock by the action repeat, so every
        # driver step counts as `repeat` frames.
        frames += cfg.env.repeat
        ep_score = obs["info"]["score"]
        ep_steps += 1

        # `acts['reset'] = obs['is_last']`: the env resets on the NEXT step, and the
        # agent carry is not reset by hand -- `is_first` masks it inside `observe_step`,
        # which is how the official code does it.
        acts = {"action": action, "reset": bool(obs["is_last"])}

        if obs["is_last"]:
            episodes.append(
                {
                    "score": float(ep_score),
                    "length": ep_steps,
                    "episode_frames": int(obs["info"]["frames"]),
                    "frames": frames,
                    "agent_steps": agent_steps,
                    "grad_steps": grad_steps,
                    "lives": int(obs["info"]["lives"]),
                }
            )
            ep_score, ep_steps = 0.0, 0
            env.seed_episode(int(episode_rng.integers(1 << 30)))

        # `agent.stream` wraps the sampler in `Prefetch(amount=1)` (embodied/jax/agent.py:337,
        # embodied/core/streams.py:32-77): a worker thread holds ONE batch ahead. Its first
        # request blocks only until the replay has an item (`limiters.wait(len(sampler))`),
        # so the first training batch is drawn from a nearly empty buffer; each `next()`
        # hands that batch over and releases the worker, which samples the next one at
        # once -- before the handed-over batch is trained on or written back, and before
        # the following env steps are added. Emulated here with the worker assumed to win
        # that race, which is what the thread does in practice.
        if pending_batch is None and len(rp) > 0:
            pending_batch = rp.sample(cfg.batch_size)

        # The ratio accumulator is consulted only once the replay can fill a batch,
        # exactly as `trainfn` does; there is no catch-up burst for the warmup.
        if len(rp) >= cfg.batch_steps:
            if warmup_seconds is None:
                warmup_seconds = time.time() - started
            for _ in range(should_train(agent_steps)):
                batch, pending_batch = pending_batch, rp.sample(cfg.batch_size)
                data = {k: torch.as_tensor(v, device=device) for k, v in batch.items()}
                _, updates, metrics = agent.train_step({}, data)
                # embodied/jax/agent.py:286-289: `train()` returns the PREVIOUS call's outs
                # (`pending_outs`) and stashes its own, so `trainfn`'s `replay.update`
                # writes back the latents of the step BEFORE this one (D-032).
                if pending_updates is not None:
                    rp.update(pending_updates)
                pending_updates = {k: host(v) for k, v in updates.items()}
                grad_steps += 1
                last_batch = data

                now = time.time()
                if now - last_report >= cfg.run.report_every or grad_steps == 1:
                    last_report = now
                    achieved_full = grad_steps / max(1, agent_steps)
                    achieved_post = grad_steps / max(1, agent_steps - warmup + 1)
                    target = cfg.train_ratio_per_agent_step
                    drift = abs(achieved_post - target) / target
                    # The accumulator carries its remainder exactly, so the grad
                    # count has a closed form: one step at the warmup boundary, then
                    # floor((agent_steps - warmup) * ratio) more. Checking against
                    # THAT catches drift immediately; the relative check below cannot,
                    # because at a handful of gradient steps +-1 step is already >5%.
                    expected = int((agent_steps - warmup) * target) + 1
                    line = {
                        "frames": frames,
                        "agent_steps": agent_steps,
                        "grad_steps": grad_steps,
                        "loss": metrics["loss"],
                        "grad_norm": metrics["grad_norm"],
                        "lr": metrics["lr"],
                        "ratio_post_warmup": achieved_post,
                        "ratio_full_run": achieved_full,
                        "grad_steps_expected": expected,
                        "episodes": len(episodes),
                        "last_score": episodes[-1]["score"] if episodes else None,
                        "sps": agent_steps / max(1e-9, now - started),
                        **{k: v for k, v in metrics.items() if k.startswith("loss/")},
                        # bbf-loop F-014: PPO and IMPALA both "scored" on frostbite
                        # with a policy that had collapsed to one constant action, and
                        # nothing in the loss curves showed it. Entropy and the
                        # advantage magnitude are the two numbers that would have.
                        # ln(18) = 2.890 is uniform over the ALE minimal set;
                        # ln(8) = 2.079 over PlayTrain default8.
                        **{
                            k: metrics[k]
                            for k in ("ent", "adv", "adv_mag", "ret", "val", "weight", "rscale", "logpi", "ret_rate")
                            if k in metrics
                        },
                    }
                    log_lines.append(line)
                    print(json.dumps(line), flush=True)
                    # MISSION rule 3. Exact first, relative second.
                    assert grad_steps == expected, (
                        f"train ratio drifted: {grad_steps} gradient steps but the "
                        f"accumulator law gives {expected} at agent_steps={agent_steps}, "
                        f"warmup={warmup}, ratio={target}, frames={frames}"
                    )
                    if grad_steps >= 20:
                        assert drift <= 0.05, (
                            f"train ratio drifted: achieved {achieved_post:.5f} vs target "
                            f"{target} ({drift:.1%}) at grad_steps={grad_steps}, "
                            f"agent_steps={agent_steps}, frames={frames}"
                        )
                    _save_report_png(agent, data, outdir, grad_steps, device, cfg, seed)
                    # Flush the record every report. A 4.3 h job that is preempted
                    # otherwise loses every episode it collected; the file is small
                    # and the write is once per 300 s.
                    _write_record(
                        outdir, cfg, seed, frames, agent_steps, grad_steps, warmup,
                        started, warmup_seconds, episodes, log_lines, device, partial=True,
                    )

    # Always leave one reconstruction from the END of the run. The official report
    # cadence is 300 s, so a run shorter than that would otherwise only ever save the
    # one from gradient step 1, which shows nothing.
    if last_batch is not None:
        _save_report_png(agent, last_batch, outdir, grad_steps, device, cfg, seed)

    record = _write_record(
        outdir, cfg, seed, frames, agent_steps, grad_steps, warmup, started,
        warmup_seconds, episodes, log_lines, device, partial=False,
    )
    (outdir / f"config_seed{seed}.json").write_text(json.dumps(cfg.to_dict(), indent=2) + "\n")
    torch.save(agent.state_dict(), outdir / f"ckpt_seed{seed}.pt")
    env.close()
    return record


def _save_report_png(agent, data, outdir: Path, grad_steps: int, device, cfg, seed: int = 0) -> None:
    """Reconstructions for the by-eye check required by MISSION's 'verified'."""
    try:
        with torch.no_grad():
            K = cfg.replay_context
            obs = {
                "image": data["image"][:, K:],
                "reward": data["reward"][:, K:],
                "is_first": data["is_first"][:, K:],
                "is_last": data["is_last"][:, K:],
                "is_terminal": data["is_terminal"][:, K:],
            }
            prevact = agent.onehot(data["action"][:, K - 1 : -1])
            ctx = {k: data[k][:, K - 1] for k in agent.wm.entry_keys}
            _, _, repfeat, _, _ = agent.wm.loss(ctx, obs, prevact, agent.scales)
            recon = agent.wm.dec(repfeat).pred()
        save_recon_png(
            outdir / f"recon_seed{seed}_{grad_steps:06d}.png",
            obs["image"][0].detach().cpu().numpy(),
            recon[0].float().detach().cpu().numpy(),
        )
    except Exception as exc:  # pragma: no cover - a plot must never kill a run
        print(json.dumps({"warn": f"recon png failed: {exc}"}), flush=True)


def _write_record(
    outdir: Path, cfg, seed: int, frames: int, agent_steps: int, grad_steps: int,
    warmup: int, started: float, warmup_seconds: float | None, episodes: list,
    log_lines: list, device, partial: bool,
) -> dict[str, Any]:
    """Build and save the metrics record.

    Called on every report as well as at the end, so a job that is preempted or hits
    its time limit still leaves behind every episode it collected. The file is a few
    hundred KB and the write is once per `report_every`.
    """
    elapsed = time.time() - started
    ws = warmup_seconds if warmup_seconds is not None else elapsed
    peak_gb = torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else 0.0
    reserved_gb = torch.cuda.max_memory_reserved() / 1024**3 if device.type == "cuda" else 0.0
    train_steps = max(1, agent_steps - warmup)
    record = {
        "run_id": cfg.run_id,
        "partial": partial,
        "device": str(device),
        "peak_gpu_gb": round(peak_gb, 2),
        "reserved_gpu_gb": round(reserved_gb, 2),
        "seconds_per_1k_agent_steps_overall": round(1000 * elapsed / max(1, agent_steps), 2),
        "seconds_per_grad_step": round((elapsed - ws) / max(1, grad_steps), 4),
        "seconds_per_1k_agent_steps_steady": round(1000 * (elapsed - ws) / train_steps, 2),
        "warmup_seconds": round(ws, 1),
        "seed": seed,
        "env_backend": cfg.env_backend,
        "game": cfg.game,
        "frames": frames,
        "agent_steps": agent_steps,
        "grad_steps": grad_steps,
        "achieved_ratio_post_warmup": grad_steps / max(1, agent_steps - warmup + 1),
        "achieved_ratio_full_run": grad_steps / max(1, agent_steps),
        "target_ratio": cfg.train_ratio_per_agent_step,
        "grad_steps_expected": (
            int((agent_steps - warmup) * cfg.train_ratio_per_agent_step) + 1
            if agent_steps >= warmup
            else 0
        ),
        "wall_seconds": elapsed,
        "episodes": episodes,
        "log": log_lines,
    }
    (outdir / f"metrics_seed{seed}.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None, help="override run.steps (smoke only)")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args(argv)

    cfg = config_.load_config(args.config)
    seed = args.seed if args.seed is not None else cfg.seed
    outdir = Path(args.outdir) if args.outdir else config_.run_dir(cfg, seed)
    record = run(cfg, seed, outdir, steps=args.steps)
    print(
        json.dumps(
            {
                "done": cfg.run_id,
                "seed": seed,
                "frames": record["frames"],
                "agent_steps": record["agent_steps"],
                "grad_steps": record["grad_steps"],
                "ratio_post_warmup": record["achieved_ratio_post_warmup"],
                "episodes": len(record["episodes"]),
                "wall_seconds": round(record["wall_seconds"], 1),
                "outdir": str(outdir),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
