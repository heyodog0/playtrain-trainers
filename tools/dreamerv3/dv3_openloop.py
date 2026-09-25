"""Open-loop prediction check — `agent.report`'s diagnostic, which U07 did not port.

Why this matters more than the reconstruction PNG the trainer already saves. That PNG
is a POSTERIOR reconstruction: at every step the encoder sees the real frame, so it
only tests one-step decoding. The actor, by contrast, trains entirely on 15-step
rollouts of the PRIOR, where no observation is available after the start. A world model
can reconstruct perfectly and still predict garbage two steps out, and nothing the
trainer logs would show it.

`agent.report` does exactly this: observe the first half of a window, then `imagine`
the second half using the RECORDED actions (not the policy), decode both, and stack
true / predicted / error into a video. Here it is as a standalone tool, reporting
per-step error so the degradation curve is a number rather than an impression.

Usage::

    python tools/dreamerv3/dv3_openloop.py \\
        --config configs/dreamerv3/ale_frostbite.json \\
        --ckpt outputs/dreamerv3/ale_frostbite_gate_v3/ckpt_seed3.pt \\
        --out results/dreamerv3/openloop/seed3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from playtrain_trainers.dreamerv3 import config as config_
from playtrain_trainers.dreamerv3 import envs as envs_
from playtrain_trainers.dreamerv3.agent import Agent
from playtrain_trainers.dreamerv3.train import pick_device


def collect_window(agent: Agent, env, length: int, device, seed: int) -> dict:
    """Roll the trained policy for `length` steps, recording frames and actions."""
    rng = np.random.default_rng(seed)
    env.seed_episode(int(rng.integers(1 << 30)))
    acts = {"action": 0, "reset": True}
    carry = agent.init_policy(1)
    images, actions, firsts = [], [], []
    while len(images) < length:
        obs = env.step(acts)
        torch_obs = {
            "image": torch.as_tensor(
                np.ascontiguousarray(obs["image"]), device=device
            ).unsqueeze(0),
            "is_first": torch.as_tensor(np.array([obs["is_first"]]), device=device),
        }
        carry, act, _ = agent.policy(carry, torch_obs, mode="train")
        action = 0 if obs["is_last"] else int(act.item())
        images.append(obs["image"].copy())
        actions.append(action)
        firsts.append(bool(obs["is_first"]))
        acts = {"action": action, "reset": bool(obs["is_last"])}
        if obs["is_last"]:
            env.seed_episode(int(rng.integers(1 << 30)))
    return {
        "image": np.stack(images)[None],  # (1, T, H, W, C)
        "action": np.array(actions)[None],  # (1, T)
        "is_first": np.array(firsts)[None],  # (1, T)
    }


@torch.no_grad()
def open_loop(agent: Agent, window: dict, device) -> dict:
    """`agent.report`'s open-loop half: observe, then imagine with recorded actions."""
    T = window["image"].shape[1]
    half = T // 2
    image = torch.as_tensor(window["image"], device=device)
    action = agent.onehot(torch.as_tensor(window["action"], device=device))
    is_first = torch.as_tensor(window["is_first"], device=device)

    tokens = agent.enc(image[:, :half])
    carry, _, obsfeat = agent.dyn.observe(
        agent.dyn.initial(1, device=device), tokens, action[:, :half], is_first[:, :half]
    )
    # D-028: a tensor policy is scanned along time, so step t uses action[:, half + t].
    _, imgfeat, _ = agent.dyn.imagine(carry, action[:, half:], T - half)

    obs_recon = agent.dec(obsfeat).pred()
    img_recon = agent.dec(imgfeat).pred()
    pred = torch.cat([obs_recon, img_recon], 1)
    truth = image.float() / 255

    # Per-step mean squared error over pixels, so the degradation curve is explicit.
    per_step = ((pred - truth) ** 2).mean(dim=(2, 3, 4))[0].cpu().numpy()
    # A frozen-frame baseline: how well would "nothing changes" have done? Without it
    # a small MSE could just mean the game looks static.
    frozen = truth[:, half - 1 : half].expand(-1, T - half, -1, -1, -1)
    frozen_mse = ((frozen - truth[:, half:]) ** 2).mean(dim=(2, 3, 4))[0].cpu().numpy()

    return {
        "per_step_mse": per_step,
        "observed_mse": float(per_step[:half].mean()),
        "imagined_mse": float(per_step[half:].mean()),
        "frozen_baseline_mse": float(frozen_mse.mean()),
        "pred": pred[0].cpu().numpy(),
        "truth": truth[0].cpu().numpy(),
        "half": half,
    }


def save_strip(path: Path, truth: np.ndarray, pred: np.ndarray, half: int, every: int = 4) -> None:
    """`agent.report`'s video layout, as a still: true / predicted / error, with a
    green-to-red column marker at the point imagination takes over."""
    from PIL import Image

    idx = list(range(0, truth.shape[0], every))
    t = np.clip(truth[idx] * 255, 0, 255).astype(np.uint8)
    p = np.clip(pred[idx] * 255, 0, 255).astype(np.uint8)
    e = ((p.astype(np.int32) - t.astype(np.int32) + 255) // 2).astype(np.uint8)
    rows = [np.concatenate(list(x), 1) for x in (t, p, e)]
    grid = np.concatenate(rows, 0)
    # Mark where imagination starts.
    h, w, _ = grid.shape
    per = w // len(idx)
    for n, i in enumerate(idx):
        colour = (0, 255, 0) if i < half else (255, 0, 0)
        grid[:2, n * per : (n + 1) * per] = colour
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid).resize((w * 3, h * 3), Image.NEAREST).save(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--length", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = config_.load_config(args.config)
    device = pick_device(cfg.device)
    env = envs_.make_env(cfg, args.seed)
    action_dim = env.act_space["action"][3]
    obs_shape = (cfg.env.size[0], cfg.env.size[1], 1 if cfg.env.gray else 3)
    agent = Agent(obs_shape, action_dim, cfg).to(device)
    agent.load_state_dict(torch.load(args.ckpt, map_location=device))
    agent.eval()

    window = collect_window(agent, env, args.length, device, args.seed)
    result = open_loop(agent, window, device)
    env.close()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    save_strip(out / "openloop.png", result["truth"], result["pred"], result["half"])
    summary = {
        "ckpt": args.ckpt,
        "length": args.length,
        "half": result["half"],
        "observed_mse": result["observed_mse"],
        "imagined_mse": result["imagined_mse"],
        "frozen_baseline_mse": result["frozen_baseline_mse"],
        "imagined_vs_frozen": result["imagined_mse"] / max(1e-9, result["frozen_baseline_mse"]),
        "per_step_mse": [float(x) for x in result["per_step_mse"]],
    }
    (out / "openloop.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(json.dumps({k: v for k, v in summary.items() if k != "per_step_mse"}, indent=2))
    print("per-step MSE after imagination starts (first 16):")
    print("  " + "  ".join(f"{x:.4f}" for x in result["per_step_mse"][result["half"] :][:16]))


if __name__ == "__main__":
    main()
