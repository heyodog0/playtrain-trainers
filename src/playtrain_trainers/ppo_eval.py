"""Greedy (argmax) rollout for PPO ActorCritic policies + W&B media logging.

Two pieces:
  - ``greedy_rollout`` runs one deterministic episode with an ActorCritic
    (feedforward OR LSTM, threading recurrent state + a_{t-1}/r_{t-1} when the
    net feeds them). Pure / env-agnostic so it's unit-testable with a fake env.
  - ``log_rollout_to_wandb`` attaches the rollout to the W&B run as a video plus
    summary scalars — so the agent's *behavior* lives on the run page next to
    the training curves, instead of a separate run-card file. No-op if W&B is off.

The trainer calls both once at end-of-run (best-effort). The standalone
"""
from __future__ import annotations

import numpy as np
import torch

WIN_THRESHOLD = 30_000


def _shape_reward(r: float, mode: str) -> float:
    """Reproduce the trainer's reward_clip shaping — used to feed r_{t-1} to a
    core trained with symlog_reward=False (so the cue matches training)."""
    if mode == "sign":
        return float(np.sign(r))
    if mode == "symlog":
        return float(np.sign(r) * np.log1p(abs(r)))
    return float(r)


@torch.no_grad()
def greedy_rollout(model, env, seed: int, max_steps: int, device,
                   reward_clip: str = "none") -> dict:
    """Run one argmax episode and return its frames + return.

    Args:
        model: ``playtrain_trainers.policy.ActorCritic`` (feedforward or LSTM).
        env: a Gymnasium-style single env — ``reset(seed=...) -> (obs, info)``,
            ``step(a) -> (obs, reward, terminated, truncated, info)``, obs HWC uint8.
        seed: env reset seed (the fixed training instance, or the run seed).
        reward_clip: how training shaped the reward (only used to reproduce the
            r_{t-1} cue when the net was trained with symlog_reward=False).

    Returns ``dict(frames=[HWC uint8], total_return, length, won)``.
    """
    obs, _ = env.reset(seed=seed)
    frames = [np.asarray(obs).copy()]
    total = 0.0
    last_action, last_reward = 0, 0.0
    state: tuple = ()
    if getattr(model, "use_lstm", False):
        state = tuple(s.to(device) for s in model.initial_state(batch_size=1))
    feed = getattr(model, "feed_prev_action_reward", False)
    symlog_r = getattr(model, "symlog_reward", True)

    for _ in range(max_steps):
        o = (torch.as_tensor(np.asarray(obs).copy(), device=device)
             .permute(2, 0, 1).unsqueeze(0).contiguous())  # (1, C, H, W)
        done = torch.zeros(1, device=device)
        la = rw = None
        if feed:
            la = torch.tensor([last_action], device=device, dtype=torch.int64)
            fed = last_reward if symlog_r else _shape_reward(last_reward, reward_clip)
            rw = torch.tensor([fed], device=device, dtype=torch.float32)
        z, state = model.get_states(o, state, done, la, rw)
        action = int(model.actor(z).view(-1).argmax().item())
        obs, reward, term, trunc, _ = env.step(action)
        frames.append(np.asarray(obs).copy())
        total += float(reward)
        last_action, last_reward = action, float(reward)
        if term or trunc:
            break

    return {"frames": frames, "total_return": total, "length": len(frames),
            "won": total >= WIN_THRESHOLD}


def log_rollout_to_wandb(wandb_run, rollout: dict, fps: int = 15,
                         key: str = "eval/greedy_rollout") -> None:
    """Attach the rollout to the W&B run: a GIF video + summary scalars
    (greedy_return / greedy_win / greedy_length). No-op when wandb_run is None.

    The GIF is encoded with imageio (a base dep) and handed to W&B as a file
    PATH — ``wandb.Video(path)`` doesn't need moviepy/ffmpeg, unlike passing a
    raw numpy array.
    """
    if wandb_run is None:
        return
    import os
    import tempfile

    import imageio.v2 as imageio
    import wandb

    wandb_run.summary["eval/greedy_return"] = float(rollout["total_return"])
    wandb_run.summary["eval/greedy_win"] = float(rollout["won"])
    wandb_run.summary["eval/greedy_length"] = int(rollout["length"])

    with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as f:
        gif_path = f.name
    try:
        imageio.mimsave(gif_path, rollout["frames"], fps=fps)
        wandb.log({key: wandb.Video(gif_path, format="gif")})
    finally:
        try:
            os.remove(gif_path)
        except OSError:
            pass
