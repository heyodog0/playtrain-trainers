"""Episode-level evaluation, shared by every arm of the comparison.

One evaluator serves the random baseline (U03), the PPO/IMPALA reference
points (U11) and BBF itself (U08/U10), so the arms cannot end up scored by
subtly different code. What it reports is fixed by PROTOCOL section 5 B:

  score                raw game score, NOT the clipped learning reward
  win_rate             fraction of episodes ending in `WIN`
  floes_visited        (score - 100 * won) / 10, i.e. the +10 events
  normalized_progress  (score - random) / (ceiling - random)

The last three are **frostbite's scoring rule**, not PlayTrain's. Frostbite
caps at 260 and ends the episode on WIN (D-014), so the paper's raw Frostbite
row is unreachable there by construction. venture, amidar and star_gunner all
loop levels with an unbounded score and have no WIN state, so for them the
comparison against the paper's row IS the raw score and only
`score_over_random` is added (D-044). `GAME_PROFILES` decides which applies;
a game absent from it gets the plain, always-meaningful fields.
"""
from __future__ import annotations

import dataclasses
import json
import subprocess
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from playtrain_trainers.bbf.config import BBFConfig

# PlayTrain frostbite: +10 per first visit to each of 16 floes, +100 for the
# igloo, which ends the episode (frostbite.js:143-175).
POINTS_PER_FLOE = 10.0
IGLOO_BONUS = 100.0
SCORE_CEILING = 260.0


@dataclasses.dataclass(frozen=True)
class GameProfile:
    """What a PlayTrain game's score actually means (D-044).

    `ceiling` is a real maximum attainable score, which makes normalized
    progress meaningful; None means the score is unbounded (the game loops
    levels) and normalizing against a ceiling would be inventing one.
    `has_win_state` says whether the game can end in `WIN` rather than only
    `GAMEOVER`. `floe_scoring` is frostbite's +10/+100 rule and nothing else.
    """

    ceiling: float | None = None
    has_win_state: bool = False
    floe_scoring: bool = False


# Only frostbite has been characterised against its source. The other three
# were read from their .js: each increments `level` without bound and reaches
# GAMEOVER only on lives, so no ceiling and no WIN.
GAME_PROFILES: dict[str, GameProfile] = {
    "frostbite": GameProfile(ceiling=SCORE_CEILING, has_win_state=True, floe_scoring=True),
    "venture": GameProfile(),
    "amidar": GameProfile(),
    "star_gunner": GameProfile(),
}


def game_profile(cfg: BBFConfig) -> GameProfile:
    """The scoring profile for this run, or the conservative default.

    Non-PlayTrain backends and unknown games get the plain profile: no
    ceiling, no WIN, no floes. Reporting fewer fields is always safe;
    reporting frostbite's fields for another game is not.
    """
    if cfg.env_backend != "playtrain":
        return GameProfile()
    return GAME_PROFILES.get(cfg.game, GameProfile())

# A policy maps a CHW uint8 observation to an action index.
Policy = Callable[[np.ndarray], int]


@dataclasses.dataclass
class EpisodeResult:
    """One evaluation episode: a whole game, not a life (D-016)."""

    seed: int
    score: float
    agent_steps: int
    game_frames: int
    lives_left: int
    game_state: str
    won: bool
    truncated: bool

    @property
    def floes_visited(self) -> float:
        """The +10 events, backed out of the raw score.

        The igloo bonus is the only non-floe reward in the game, so removing it
        leaves a multiple of 10.
        """
        return (self.score - (IGLOO_BONUS if self.won else 0.0)) / POINTS_PER_FLOE


def random_policy(n_actions: int, rng: np.random.Generator) -> Policy:
    """Uniform over the full declared action set (D-008).

    Note D-003/F-008: on frostbite only 5 of the 8 actions are distinguishable,
    so this is uniform over 8 wire actions but not over 5 effects -- LEFT and
    RIGHT each get 2/8 of the mass and NOOP 2/8, while UP and DOWN get 1/8.
    That is the same distribution any agent on this action set starts from, so
    it is the right floor to measure against.
    """
    return lambda _obs: int(rng.integers(n_actions))


def epsilon_greedy(policy: Policy, epsilon: float, n_actions: int, rng: np.random.Generator) -> Policy:
    """PROTOCOL section 1: `epsilon_eval = 0.001`."""
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must be in [0, 1]")

    def act(obs: np.ndarray) -> int:
        if epsilon > 0.0 and rng.random() < epsilon:
            return int(rng.integers(n_actions))
        return policy(obs)

    return act


def run_episode(env, policy: Policy, seed: int, max_agent_steps: int) -> EpisodeResult:
    """Play one whole game and report it.

    ``max_agent_steps`` is a belt-and-braces stop: the runtime already
    truncates at ``max_steps`` FRAMES (D-012), so this only fires if the two
    disagree, which would itself be a bug worth seeing.
    """
    obs, info = env.reset(seed=seed)
    score = float(info.get("episode_score", info.get("score", 0.0)))
    state = str(info.get("gameState", "PLAYING"))
    lives = int(info.get("lives", -1))
    frames = int(info.get("episodeLength", 0))
    truncated = False
    steps = 0
    for steps in range(1, max_agent_steps + 1):
        obs, _, terminated, truncated, info = env.step(policy(obs))
        score = float(info.get("episode_score", score))
        state = str(info.get("gameState", state))
        lives = int(info.get("lives", lives))
        frames = int(info.get("episodeLength", frames))
        if terminated or truncated:
            break
    else:
        truncated = True
    return EpisodeResult(
        seed=seed,
        score=score,
        agent_steps=steps,
        game_frames=frames,
        lives_left=lives,
        game_state=state,
        won=state == "WIN",
        truncated=bool(truncated),
    )


def eval_seeds(cfg: BBFConfig, n_episodes: int | None = None) -> list[int]:
    """The fixed game-seed pool every arm is scored on (D-017).

    Dopamine just runs 100 episodes; pinning the seeds instead means the
    random baseline, PPO, IMPALA and BBF all face the same 100 floe layouts,
    which removes layout luck from the comparison rather than averaging over
    it separately per arm.
    """
    n = cfg.num_eval_episodes if n_episodes is None else n_episodes
    return [cfg.eval_seed_base + i for i in range(n)]


def bootstrap_ci(
    values: Sequence[float],
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean (PROTOCOL section 5 B)."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("cannot bootstrap an empty sample")
    if arr.size == 1:
        return float(arr[0]), float(arr[0])
    rng = np.random.default_rng(seed)
    means = rng.choice(arr, size=(resamples, arr.size), replace=True).mean(axis=1)
    lo = float(np.percentile(means, 100 * (1 - confidence) / 2))
    hi = float(np.percentile(means, 100 * (1 + confidence) / 2))
    return lo, hi


def aggregate(
    episodes: Sequence[EpisodeResult],
    random_mean: float | None = None,
    ceiling: float = SCORE_CEILING,
    ci_seed: int = 0,
    playtrain_metrics: bool = True,
    profile: GameProfile | None = None,
) -> dict[str, Any]:
    """The PROTOCOL section 5 B numbers, plus the spread.

    `profile` says which of frostbite's fields are meaningful for this game
    (D-044); pass one from `game_profile(cfg)`. Without it the legacy
    `playtrain_metrics` flag selects frostbite's profile or the plain one,
    which is what every pre-D-044 caller meant.

    The frostbite-only fields are `win_rate`, `wins`, `floes_visited_mean`,
    `score_ceiling` and `normalized_progress`. They encode its scoring rule
    (+10 per floe, +100 for the igloo, ending the episode) and are nonsense
    elsewhere: on ALE Frostbite `floes_visited_mean` is just score/10, which
    once reported 238 "floes" out of 16 for a 2382-point seed, and `win_rate`
    is identically 0 because the ALE has no WIN state. For an unbounded game
    the honest normalization is `score_over_random`, which is reported
    instead and is a difference, not a fraction.
    """
    if profile is None:
        profile = GAME_PROFILES["frostbite"] if playtrain_metrics else GameProfile()
    if not episodes:
        raise ValueError("no episodes to aggregate")
    scores = [e.score for e in episodes]
    lo, hi = bootstrap_ci(scores, seed=ci_seed)
    wins = sum(e.won for e in episodes)
    out: dict[str, Any] = {
        "episodes": len(episodes),
        "score_mean": float(np.mean(scores)),
        "score_std": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
        "score_sem": (
            float(np.std(scores, ddof=1) / np.sqrt(len(scores))) if len(scores) > 1 else 0.0
        ),
        "score_ci95": [lo, hi],
        "score_min": float(np.min(scores)),
        "score_median": float(np.median(scores)),
        "score_max": float(np.max(scores)),
        "agent_steps_mean": float(np.mean([e.agent_steps for e in episodes])),
        "game_frames_mean": float(np.mean([e.game_frames for e in episodes])),
        "lives_left_mean": float(np.mean([e.lives_left for e in episodes])),
        "truncated": int(sum(e.truncated for e in episodes)),
        "game_states": {
            s: sum(e.game_state == s for e in episodes)
            for s in sorted({e.game_state for e in episodes})
        },
    }
    if profile.has_win_state:
        out["win_rate"] = wins / len(episodes)
        out["wins"] = int(wins)
    if profile.floe_scoring:
        out["floes_visited_mean"] = float(
            np.mean([e.floes_visited for e in episodes])
        )
    if random_mean is not None:
        out["random_mean"] = random_mean
        # Always meaningful: how far above the measured floor this policy is.
        out["score_over_random"] = out["score_mean"] - random_mean
    if profile.ceiling is not None:
        cap = profile.ceiling if ceiling == SCORE_CEILING else ceiling
        out["score_ceiling"] = cap
        if random_mean is not None:
            denom = cap - random_mean
            out["normalized_progress"] = (
                (out["score_mean"] - random_mean) / denom if denom else float("nan")
            )
    return out


def policy_rng(cfg: BBFConfig, episode_seed: int) -> np.random.Generator:
    """Per-episode generator for anything stochastic in the policy.

    Seeded from ``(cfg.seed, episode_seed)`` so that a single evaluation
    episode can be replayed on its own -- which one generator shared across
    all 100 episodes does not allow, since reproducing episode 15 would mean
    replaying the 14 before it. The run seed is folded in so two runs on the
    same layouts still get different action streams.
    """
    return np.random.default_rng([cfg.seed, episode_seed])


def evaluate_policy(
    cfg: BBFConfig,
    policy_fn: Callable[[Any, int], Policy],
    *,
    n_episodes: int | None = None,
    random_mean: float | None = None,
    progress_every: int = 0,
) -> tuple[list[EpisodeResult], dict[str, Any]]:
    """Score a policy over the fixed eval seed pool, one env, whole games.

    ``policy_fn(env, episode_seed)`` is called once per episode, so a policy
    with any randomness in it can take a per-episode generator and stay
    independently replayable (see ``policy_rng``). The env is built with
    ``training=False``, which is what makes an episode a whole game rather
    than a life (D-016).
    """
    from playtrain_trainers.bbf.envs import make_env

    seeds = eval_seeds(cfg, n_episodes)
    env = make_env(cfg, seed=cfg.seed, training=False)
    try:
        results = []
        for i, s in enumerate(seeds, 1):
            results.append(
                run_episode(env, policy_fn(env, s), s, cfg.max_steps_per_episode)
            )
            if progress_every and i % progress_every == 0:
                print(
                    f"  {i}/{len(seeds)} episodes, "
                    f"running mean {np.mean([r.score for r in results]):.2f}",
                    flush=True,
                )
    finally:
        env.close()
    return results, aggregate(
        results,
        random_mean=random_mean,
        profile=game_profile(cfg),
    )


def git_commit(repo: Path | None = None) -> str:
    """The commit the numbers were produced at, for the results directory."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo or Path(__file__).resolve().parents[3],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception:  # pragma: no cover - only when git is unavailable
        return "unknown"


def write_metrics(
    path: Path,
    cfg: BBFConfig,
    episodes: Sequence[EpisodeResult],
    summary: dict[str, Any],
    *,
    unit: str,
    job_ids: Sequence[str] = (),
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write ``metrics.json`` with everything needed to re-derive the numbers.

    MISSION hard rule 3: nothing is reported that does not live in a file like
    this one, with its config, seeds and commit hash beside it.
    """
    payload: dict[str, Any] = {
        "unit": unit,
        "run_id": cfg.run_id,
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "commit": git_commit(),
        "job_ids": list(job_ids),
        "config": cfg.to_dict(),
        "eval_seeds": [e.seed for e in episodes],
        "summary": summary,
        "episodes": [dataclasses.asdict(e) | {"floes_visited": e.floes_visited} for e in episodes],
    }
    if extra:
        payload |= extra
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return payload
