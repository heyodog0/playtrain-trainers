"""Greedy (argmax) evaluation logged during training.

Training return (the behavior policy, with sampled actions) can look great while
the *deterministic* argmax policy is brittle — a known failure mode on sparse
navigation tasks (and the symptom that motivated the LSTM work). Logging greedy
win-rate / return alongside training return makes that visible live, instead of
only via a post-hoc rollout. Works for feedforward and LSTM nets (the recurrent
state is threaded across each episode, fresh per episode).
"""
from __future__ import annotations

import torch

from playtrain_trainers.impala.environment import _format_frame


@torch.no_grad()
def greedy_eval(
    model,
    gym_env,
    *,
    seeds: list[int],
    device: torch.device,
    max_steps: int,
    win_threshold: float,
    chw_transpose: bool = True,
) -> tuple[float, float]:
    """Run one greedy episode per seed; return (win_rate, mean_return).

    Greedy = argmax over policy_logits each step (independent of the net's
    train/eval flag, so it's safe to call on a model a learner thread is also
    using). The recurrent state starts zeroed each episode and is threaded
    across steps; `done` stays False within an episode (we break on terminal),
    so no mid-episode reset is needed.
    """
    wins = 0
    returns: list[float] = []
    for seed in seeds:
        obs, _ = gym_env.reset(seed=seed)
        core_state = tuple(s.to(device) for s in model.initial_state(batch_size=1))
        last_action = 0
        ep_return = 0.0
        for _ in range(max_steps):
            frame = _format_frame(obs, chw_transpose).to(device)  # [1,1,C,H,W]
            inputs = {
                "frame": frame,
                "reward": torch.zeros(1, 1, device=device),
                "done": torch.zeros(1, 1, dtype=torch.bool, device=device),
                "last_action": torch.tensor([[last_action]], dtype=torch.int64,
                                            device=device),
            }
            out, core_state = model(inputs, core_state)
            action = int(out["policy_logits"].view(-1).argmax())
            obs, reward, terminated, truncated, _ = gym_env.step(action)
            ep_return += float(reward)
            last_action = action
            if terminated or truncated:
                break
        returns.append(ep_return)
        wins += int(ep_return >= win_threshold)
    return wins / len(seeds), sum(returns) / len(returns)


def eval_seeds(fixed_env_seed: int | None, n_episodes: int) -> list[int]:
    """Seeds to evaluate on.

    Fixed-instance runs: eval the memorized instance (deterministic greedy on a
    deterministic env => one episode is sufficient). Procedural runs: a block of
    held-out seeds disjoint from training (cfg.seed*1e6 + actor_index)."""
    if fixed_env_seed is not None:
        return [fixed_env_seed]
    return [9_000_000 + i for i in range(n_episodes)]
