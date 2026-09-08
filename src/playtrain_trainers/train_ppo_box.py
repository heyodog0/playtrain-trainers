"""Continuous-action PPO over a PlayTrain box action space (CleanRL-style).

The minimal, self-contained counterpart to ``train_ppo_clean`` for
``gym.spaces.Box`` spaces (``runtime/action_spaces.json`` — e.g. ``mouse2d``:
pointer x/y + click). Drives ``playtrain.runtime.NativeVecEnv`` directly (the
sync native host is the only box-capable backend today) with a diagonal-Normal
``ActorCritic(continuous=True)`` head. Feedforward only, no LSTM, no
prev-action feed — the full ``train_ppo_clean`` feature set stays
Discrete-only until box training earns the integration.

    python -m playtrain_trainers.train_ppo_box --game aim_trainer \
        --action-space mouse2d --num-envs 32 --total-steps 2000000
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from playtrain_trainers.policy import ActorCritic


@dataclass
class Config:
    game: str = "aim_trainer"
    action_space: str = "mouse2d"
    num_envs: int = 32
    num_steps: int = 128           # rollout length per env
    total_steps: int = 2_000_000
    lr: float = 2.5e-4
    gamma: float = 0.999
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.001
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    update_epochs: int = 3
    num_minibatches: int = 8
    net: str = "impala"
    log_std_init: float = -1.0
    seed: int = 1
    device: str = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")


def main(cfg: Config) -> dict:
    from playtrain.runtime import NativeVecEnv

    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    device = torch.device(cfg.device)

    env = NativeVecEnv(cfg.game, num_envs=cfg.num_envs,
                       action_space=cfg.action_space, autoreset=True)
    assert env.box_channels is not None, "train_ppo_box needs a box action space"
    k = env.action_dim
    N, T = cfg.num_envs, cfg.num_steps

    model = ActorCritic(n_actions=k, net=cfg.net, continuous=True).to(device)
    # std=1.0 saturates [0,1]-range channels (clamped samples are near-uniform,
    # starving the mean's gradient); start tighter so aiming is expressible.
    with torch.no_grad():
        model.log_std.fill_(cfg.log_std_init)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-5)

    obs_buf = torch.zeros(T, N, 3, 64, 64, dtype=torch.uint8, device=device)
    act_buf = torch.zeros(T, N, k, device=device)
    logp_buf = torch.zeros(T, N, device=device)
    rew_buf = torch.zeros(T, N, device=device)
    done_buf = torch.zeros(T, N, device=device)
    val_buf = torch.zeros(T, N, device=device)

    obs = env.reset(seeds=rng.integers(0, 2**31 - 1, N).astype(np.int32))
    next_obs = torch.from_numpy(obs.copy()).permute(0, 3, 1, 2).to(device)
    next_done = torch.zeros(N, device=device)

    ep_ret = np.zeros(N)
    ep_len = np.zeros(N, dtype=np.int64)
    finished_returns: list[float] = []
    global_step = 0
    t_start = time.time()
    updates = max(1, cfg.total_steps // (N * T))

    for update in range(updates):
        for t in range(T):
            obs_buf[t] = next_obs
            done_buf[t] = next_done
            with torch.no_grad():
                dist, value = model(next_obs)
                action = dist.sample()
                logp_buf[t] = dist.log_prob(action)
                val_buf[t] = value
            act_buf[t] = action
            o, r, term, trunc, _ = env.step(action.cpu().numpy())
            global_step += N
            ep_ret += r
            ep_len += 1
            dones = term | trunc
            for i in np.flatnonzero(dones):
                finished_returns.append(float(ep_ret[i]))
                ep_ret[i] = 0.0
                ep_len[i] = 0
            rew_buf[t] = torch.from_numpy(r.copy()).to(device)
            next_obs = torch.from_numpy(o.copy()).permute(0, 3, 1, 2).to(device)
            next_done = torch.from_numpy(dones.astype(np.float32)).to(device)

        # GAE
        with torch.no_grad():
            _, next_value = model(next_obs)
            adv = torch.zeros_like(rew_buf)
            lastgae = 0
            for t in reversed(range(T)):
                nextnonterm = 1.0 - (next_done if t == T - 1 else done_buf[t + 1])
                nextval = next_value if t == T - 1 else val_buf[t + 1]
                delta = rew_buf[t] + cfg.gamma * nextval * nextnonterm - val_buf[t]
                lastgae = delta + cfg.gamma * cfg.gae_lambda * nextnonterm * lastgae
                adv[t] = lastgae
            ret = adv + val_buf

        b_obs = obs_buf.reshape(-1, 3, 64, 64)
        b_act = act_buf.reshape(-1, k)
        b_logp = logp_buf.reshape(-1)
        b_adv = adv.reshape(-1)
        b_ret = ret.reshape(-1)
        b_val = val_buf.reshape(-1)
        inds = np.arange(T * N)
        mb = T * N // cfg.num_minibatches
        for _ in range(cfg.update_epochs):
            rng.shuffle(inds)
            for s in range(0, T * N, mb):
                idx = inds[s:s + mb]
                dist, value = model(b_obs[idx])
                newlogp = dist.log_prob(b_act[idx])
                entropy = dist.entropy().mean()
                ratio = (newlogp - b_logp[idx]).exp()
                madv = (b_adv[idx] - b_adv[idx].mean()) / (b_adv[idx].std() + 1e-8)
                pg = torch.max(-madv * ratio,
                               -madv * ratio.clamp(1 - cfg.clip_coef, 1 + cfg.clip_coef)).mean()
                v_clipped = b_val[idx] + (value - b_val[idx]).clamp(-cfg.clip_coef, cfg.clip_coef)
                v_loss = 0.5 * torch.max((value - b_ret[idx]) ** 2,
                                         (v_clipped - b_ret[idx]) ** 2).mean()
                loss = pg + cfg.vf_coef * v_loss - cfg.ent_coef * entropy
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                opt.step()

        recent = finished_returns[-100:]
        sps = int(global_step / (time.time() - t_start))
        print(f"update {update + 1}/{updates} step {global_step} "
              f"sps {sps} episodes {len(finished_returns)} "
              f"mean_return(100) {np.mean(recent) if recent else float('nan'):.2f} "
              f"std {torch.exp(model.log_std).mean().item():.3f}", flush=True)

    env.close()
    return {"global_step": global_step,
            "episodes": len(finished_returns),
            "mean_return_100": float(np.mean(finished_returns[-100:])) if finished_returns else None}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for f, t in [("game", str), ("action_space", str), ("num_envs", int),
                 ("num_steps", int), ("total_steps", int), ("lr", float),
                 ("net", str), ("seed", int), ("device", str)]:
        p.add_argument(f"--{f.replace('_', '-')}", type=t, default=getattr(Config, f))
    main(Config(**vars(p.parse_args())))
