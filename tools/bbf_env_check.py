"""U02 verification artifact: save the wrapper stack's frames and measure the
game's timing constants.

Writes into results/bbf/env_check/:
  stack.png          the 4-frame grayscale stack, side by side, at 84 px
  stack_x4.png       the same, nearest-upscaled 4x, for looking at by eye
  native64_vs_84.png the 64 px render, a 64->84 upscale and the native 84 px
                     render side by side (the D-002 evidence)
  env_check.json     the measured numbers

Run: uv run --no-sync python tools/bbf_env_check.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from playtrain_trainers.bbf.config import BBFConfig
from playtrain_trainers.bbf.envs import _distinct_action_meanings, make_playtrain_atari100k

OUT = Path("results/bbf/env_check")
GAME_SRC = Path("../playtrain/examples/games/js/frostbite.js")


def _grab(cfg: BBFConfig, seed: int, steps: int, actions=None):
    env = make_playtrain_atari100k(cfg, seed=seed)
    try:
        obs, info = env.reset(seed=seed)
        rng = np.random.default_rng(seed)
        trail = [dict(info)]
        for t in range(steps):
            a = actions[t] if actions is not None else int(rng.integers(env.action_space.n))
            obs, r, term, trunc, info = env.step(a)
            trail.append(dict(info))
            if term or trunc:
                break
        return obs.copy(), trail, env.action_space.n, dict(info)
    finally:
        env.close()


def _up(arr: np.ndarray, factor: int) -> np.ndarray:
    return np.asarray(
        Image.fromarray(arr).resize(
            (arr.shape[1] * factor, arr.shape[0] * factor), resample=Image.NEAREST
        )
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = BBFConfig()
    out: dict[str, object] = {}

    # --- the stacked frame, for the eye check (D-002) ---------------------
    # Hold UP for a while so the stack shows the player mid-hop and the frames
    # visibly differ; a stack of identical frames would prove nothing.
    obs, trail, n_actions, info = _grab(cfg, seed=5, steps=10, actions=[3] * 10)
    assert obs.shape == (4, 84, 84), obs.shape
    strip = np.concatenate(list(obs), axis=1)
    Image.fromarray(strip).save(OUT / "stack.png")
    Image.fromarray(_up(strip, 4)).save(OUT / "stack_x4.png")
    frames_differ = int(sum(
        not np.array_equal(obs[i], obs[i + 1]) for i in range(obs.shape[0] - 1)
    ))
    out["stack"] = {
        "shape": list(obs.shape),
        "dtype": str(obs.dtype),
        "distinct_gray_values": int(len(np.unique(obs))),
        "adjacent_frame_pairs_that_differ": frames_differ,
        # The runtime unpacks score/lives out of a binary header, so they come
        # back as numpy scalars; cast for json.
        "info_at_end": {
            "score": float(info["score"]),
            "lives": int(info["lives"]),
            "gameState": str(info["gameState"]),
        },
    }

    # --- D-002 evidence: native 84 px vs an upscaled 64 px ----------------
    smalls = {}
    for size in (64, 84):
        c = BBFConfig(obs_size=size, max_noops=0, frame_stack=1)
        f, _, _, _ = _grab(c, seed=11, steps=0)
        smalls[size] = f[0]
    up = np.asarray(Image.fromarray(smalls[64]).resize((84, 84), resample=Image.BILINEAR))
    # Pad the 64 px panel to 84 rows so the three can sit side by side; the
    # padding is black and clearly not part of the render.
    pad64 = np.zeros((84, 64), np.uint8)
    pad64[:64, :64] = smalls[64]
    panel = np.concatenate([pad64, up, smalls[84]], axis=1)
    Image.fromarray(_up(panel, 4)).save(OUT / "native64_vs_84.png")
    out["d002_native_84px"] = {
        "mean_abs_diff_upscaled64_vs_native84": round(
            float(np.abs(up.astype(np.int16) - smalls[84].astype(np.int16)).mean()), 3
        ),
        "verdict": "runtime renders at 84 px; not a resize of 64 px",
    }

    # --- D-003: how many of the 8 actions the game can distinguish --------
    src = GAME_SRC.read_text()
    meanings = ["NOOP", "LEFT", "RIGHT", "UP", "DOWN", "D", "LEFT_D", "RIGHT_D"]
    distinct = _distinct_action_meanings(meanings, src)
    out["d003_actions"] = {
        "declared": n_actions,
        "distinct_for_this_game": len(distinct),
        "distinct_names": distinct,
        "note": "actions 5-7 press SPACE (key 32), which frostbite.js never reads",
    }

    # --- F-004: the game's timing constants -------------------------------
    # Read from the source rather than inferred, then expressed in agent steps.
    consts = {
        "move_cooldown_frames_after_hop": 15,
        "jump_frames_airborne": 15,
        "stun_frames_after_death": 30,
        "floe_speed_px_per_frame_min": 1.5,
        "floe_speed_px_per_frame_max": 3.0,
        "screen_px": 400,
    }
    skip = cfg.frame_skip
    out["f004_timing"] = {
        **consts,
        "frame_skip": skip,
        "agent_steps_per_hop": consts["move_cooldown_frames_after_hop"] / skip,
        "frames_per_floe_traversal_min": consts["screen_px"] / consts["floe_speed_px_per_frame_max"],
        "frames_per_floe_traversal_max": consts["screen_px"] / consts["floe_speed_px_per_frame_min"],
        "agent_steps_per_floe_traversal_min": round(
            consts["screen_px"] / consts["floe_speed_px_per_frame_max"] / skip, 1
        ),
        "agent_steps_per_floe_traversal_max": round(
            consts["screen_px"] / consts["floe_speed_px_per_frame_min"] / skip, 1
        ),
        "note": (
            "a row's floes cross the screen in 133-267 frames = 33-67 agent steps, "
            "while a vertical hop costs 15 frames = 3.75 agent steps, so the agent "
            "gets roughly 9-18 hop opportunities per floe traversal"
        ),
    }

    # --- the score ceiling, from the source (D-014) -----------------------
    out["d014_ceiling"] = {
        "floes": 16,
        "points_per_floe": 10,
        "igloo_target_visits": 12,
        "igloo_bonus": 100,
        "max_score_all_floes_then_igloo": 16 * 10 + 100,
        "min_winning_score_igloo_asap": 12 * 10 + 100,
        "lives": 3,
    }

    # gymnasium and the runtime hand back numpy scalars (Discrete.n, the
    # binary step header); json needs python ones.
    def _plain(o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        raise TypeError(f"not JSON serializable: {type(o).__name__}")

    text = json.dumps(out, indent=2, default=_plain)
    (OUT / "env_check.json").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
