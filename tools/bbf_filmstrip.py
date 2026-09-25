"""Replay one evaluation episode and save it as a filmstrip, for the eye check.

MISSION "what verified means": a baseline unit has one episode inspected by
eye. The episode is replayed from its recorded seed with the same policy, so
what the strip shows is the episode the metrics file scored -- and the printed
score is checked against the one in metrics.json.

Run: uv run --no-sync python tools/bbf_filmstrip.py results/bbf/random_frostbite --pick max
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from playtrain_trainers.bbf.config import config_from_dict
from playtrain_trainers.bbf.envs import make_env
from playtrain_trainers.bbf.evaluate import policy_rng, random_policy


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--pick", default="max", choices=["max", "median", "min", "first"])
    ap.add_argument("--cols", type=int, default=12, help="frames in the strip")
    ap.add_argument("--scale", type=int, default=3)
    args = ap.parse_args()

    d = json.loads((args.run_dir / "metrics.json").read_text())
    cfg = config_from_dict(d["config"])
    eps = d["episodes"]
    scores = [e["score"] for e in eps]
    idx = {
        "max": int(np.argmax(scores)),
        "min": int(np.argmin(scores)),
        "first": 0,
        "median": int(np.argsort(scores)[len(scores) // 2]),
    }[args.pick]
    target = eps[idx]
    print(f"episode {idx}: seed={target['seed']} score={target['score']} "
          f"steps={target['agent_steps']} state={target['game_state']}")

    # The evaluator seeds the policy per episode from the episode's game seed,
    # so this replays the scored episode exactly; the assert below holds it to
    # that.
    rng = policy_rng(cfg, target["seed"])
    env = make_env(cfg, seed=cfg.seed, training=False)
    try:
        policy = random_policy(env.action_space.n, rng)
        obs, info = env.reset(seed=target["seed"])
        frames = [obs[-1].copy()]
        for _ in range(cfg.max_steps_per_episode):
            obs, _, term, trunc, info = env.step(policy(obs))
            frames.append(obs[-1].copy())
            if term or trunc:
                break
    finally:
        env.close()
    print(f"replay: score={info['episode_score']} steps={len(frames) - 1} "
          f"state={info['gameState']} lives={info['lives']}")
    assert float(info["episode_score"]) == float(target["score"]), (
        f"replay scored {info['episode_score']}, metrics.json says {target['score']} "
        "-- the episode is not reproducible from its seed"
    )

    # Even sample across the episode so the strip covers the whole thing.
    pick = np.linspace(0, len(frames) - 1, min(args.cols, len(frames))).round().astype(int)
    strip = np.concatenate([frames[i] for i in pick], axis=1)
    img = Image.fromarray(strip).resize(
        (strip.shape[1] * args.scale, strip.shape[0] * args.scale), Image.NEAREST
    )
    out = args.run_dir / f"episode_{args.pick}.png"
    img.save(out)
    print(f"frames at agent steps {list(pick)}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
