"""The 100-episode evaluation of a trained DreamerV3 seed, on the BBF arms' fixed pool.

PROTOCOL section 4 D-009 / section 5 B: besides the training-episode score, each seed is
scored over 100 whole games with the SAME game seeds every BBF, PPO, IMPALA and random
arm faced (`bbf.evaluate.eval_seeds`: 8,000,000 + i), sampled AND greedy. The
statistics come from `bbf.evaluate.aggregate`, so win rate, floes, normalized
progress and the bootstrap CI are computed exactly as for those arms.

The env is the port's own PlayTrain wrapper (64x64 RGB, the recurrent agent needs its
carry), which plays whole games (lives unused) and takes each game seed through
`seed_episode`. Sampling is reseeded per (run seed, game seed), so any one episode can
be replayed on its own, as `bbf.evaluate.policy_rng` does for the other arms.

Usage::

    python tools/dreamerv3/dv3_evaluate.py outputs/dreamerv3/playtrain_frostbite_u09 --seeds 0 1 2 3 4
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import torch

from playtrain_trainers.bbf.config import BBFConfig
from playtrain_trainers.bbf.evaluate import GAME_PROFILES, EpisodeResult, aggregate
from playtrain_trainers.dreamerv3 import config as C
from playtrain_trainers.dreamerv3 import envs as E
from playtrain_trainers.dreamerv3.agent import Agent

RANDOM_MEAN = 33.50  # PlayTrain frostbite random, bbf-loop D-008


class _StateTracking(E.PlayTrainDreamer):
    """The port's wrapper, also keeping the runtime's `gameState` for the win flag."""

    def _absorb(self, obs, info):
        super()._absorb(obs, info)
        self.game_state = str(info.get("gameState", getattr(self, "game_state", "PLAYING")))


def play(agent: Agent, env, game_seed: int, run_seed: int, mode: str, max_agent_steps: int,
         device: torch.device) -> EpisodeResult:
    torch.manual_seed(int(np.random.default_rng([run_seed, game_seed]).integers(1 << 62)))
    env.seed_episode(game_seed)
    obs = env.step({"action": 0, "reset": True})
    carry = agent.init_policy(1)
    steps, truncated = 0, False
    while True:
        tobs = {
            "image": torch.as_tensor(np.ascontiguousarray(obs["image"]), device=device).unsqueeze(0),
            "is_first": torch.as_tensor(np.array([obs["is_first"]]), device=device),
        }
        carry, act, _ = agent.policy(carry, tobs, mode="eval" if mode == "greedy" else "train")
        if obs["is_last"]:
            break
        if steps >= max_agent_steps:
            truncated = True
            break
        obs = env.step({"action": int(act.item()), "reset": False})
        steps += 1
    info = obs["info"]
    state = getattr(env, "game_state", "UNKNOWN")
    return EpisodeResult(
        seed=game_seed, score=float(info["score"]), agent_steps=steps, game_frames=int(info["frames"]),
        lives_left=int(info["lives"]), game_state=state, won=state == "WIN",
        truncated=truncated or (bool(obs["is_last"]) and state not in ("WIN", "GAMEOVER")),
    )


def evaluate_seed(run_dir: Path, seed: int, episodes: int, device: torch.device) -> dict:
    cfg = C.load_config(run_dir / f"config_seed{seed}.json")
    if device.type != "cuda":
        cfg = dataclasses.replace(cfg, compute_dtype="float32")  # D-012, as train.run
    bbf = BBFConfig()
    pool = [bbf.eval_seed_base + i for i in range(episodes)]
    env = E.make_playtrain_dreamer(cfg.game, seed, cfg)
    env.__class__ = _StateTracking  # same object, plus the gameState it now records
    action_dim = int(env.act_space["action"][3])
    agent = Agent((cfg.env.size[0], cfg.env.size[1], 3), action_dim, cfg).to(device)
    agent.load_state_dict(torch.load(run_dir / f"ckpt_seed{seed}.pt", map_location=device))
    agent.eval()
    out = {"seed": seed, "game_seeds": [pool[0], pool[-1]], "episodes": episodes}
    try:
        for mode in ("sampled", "greedy"):
            t0 = time.time()
            res = [play(agent, env, s, seed, mode, bbf.max_steps_per_episode, device) for s in pool]
            out[mode] = {
                "aggregate": aggregate(res, random_mean=RANDOM_MEAN, profile=GAME_PROFILES["frostbite"]),
                "per_episode": [dataclasses.asdict(r) for r in res],
                "seconds": round(time.time() - t0, 1),
            }
            a = out[mode]["aggregate"]
            print(f"seed {seed} {mode:<7}: mean {a['score_mean']:.2f} CI {a['score_ci95']} win {a.get('win_rate')} "
                  f"floes {a.get('floes_visited_mean')} norm {a.get('normalized_progress')}", flush=True)
    finally:
        env.close()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir")
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--episodes", type=int, default=100)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for seed in args.seeds:
        rec = evaluate_seed(run_dir, seed, args.episodes, device)
        (run_dir / f"eval_seed{seed}.json").write_text(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
