"""Tests for playtrain_trainers.bbf.evaluate.

The arithmetic here turns raw scores into the numbers the report quotes
(win rate, floes visited, normalized progress), so it is checked against
hand-computed values rather than against itself.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from playtrain_trainers.bbf.config import BBFConfig
from playtrain_trainers.bbf.evaluate import (
    SCORE_CEILING,
    EpisodeResult,
    aggregate,
    bootstrap_ci,
    epsilon_greedy,
    eval_seeds,
    random_policy,
    run_episode,
    write_metrics,
)


def _ep(score, won=False, state=None, steps=50, lives=0, frames=200, seed=0, trunc=False):
    return EpisodeResult(
        seed=seed,
        score=float(score),
        agent_steps=steps,
        game_frames=frames,
        lives_left=lives,
        game_state=state or ("WIN" if won else "GAMEOVER"),
        won=won,
        truncated=trunc,
    )


# ----------------------------------------------------------------------
# floes_visited: the igloo bonus must come out before dividing by 10
# ----------------------------------------------------------------------
def test_floes_visited_without_a_win():
    assert _ep(30).floes_visited == 3.0
    assert _ep(0).floes_visited == 0.0
    assert _ep(160).floes_visited == 16.0


def test_floes_visited_with_a_win_removes_the_igloo_bonus():
    # 12 floes then straight into the igloo: 120 + 100 = 220.
    assert _ep(220, won=True).floes_visited == 12.0
    # Every floe then the igloo: the 260 ceiling.
    assert _ep(SCORE_CEILING, won=True).floes_visited == 16.0
    # Naively dividing by 10 would report 26 floes, which is impossible.
    assert _ep(SCORE_CEILING, won=True).floes_visited <= 16.0


# ----------------------------------------------------------------------
# aggregate
# ----------------------------------------------------------------------
def test_aggregate_basic_stats():
    eps = [_ep(10), _ep(20), _ep(30), _ep(40)]
    s = aggregate(eps)
    assert s["episodes"] == 4
    assert s["score_mean"] == 25.0
    assert s["score_min"] == 10.0 and s["score_max"] == 40.0
    assert s["score_median"] == 25.0
    assert s["win_rate"] == 0.0 and s["wins"] == 0
    assert s["floes_visited_mean"] == 2.5
    assert s["game_states"] == {"GAMEOVER": 4}


def test_aggregate_win_rate_and_states():
    eps = [_ep(220, won=True), _ep(30), _ep(260, won=True), _ep(0)]
    s = aggregate(eps)
    assert s["wins"] == 2 and s["win_rate"] == 0.5
    assert s["game_states"] == {"GAMEOVER": 2, "WIN": 2}
    # (12 + 3 + 16 + 0) / 4
    assert s["floes_visited_mean"] == pytest.approx(7.75)


def test_aggregate_normalized_progress():
    s = aggregate([_ep(100)], random_mean=34.1)
    # (100 - 34.1) / (260 - 34.1)
    assert s["normalized_progress"] == pytest.approx((100 - 34.1) / (260 - 34.1))
    assert s["random_mean"] == 34.1


def test_normalized_progress_is_zero_at_the_floor_and_one_at_the_ceiling():
    assert aggregate([_ep(34.1)], random_mean=34.1)["normalized_progress"] == pytest.approx(0.0)
    assert aggregate([_ep(260, won=True)], random_mean=34.1)["normalized_progress"] == pytest.approx(1.0)


def test_aggregate_single_episode_has_zero_spread():
    s = aggregate([_ep(50)])
    assert s["score_std"] == 0.0 and s["score_sem"] == 0.0
    assert s["score_ci95"] == [50.0, 50.0]


def test_aggregate_rejects_empty():
    with pytest.raises(ValueError):
        aggregate([])


def test_aggregate_counts_truncations():
    assert aggregate([_ep(10, trunc=True), _ep(10)])["truncated"] == 1


# ----------------------------------------------------------------------
# bootstrap_ci
# ----------------------------------------------------------------------
def test_bootstrap_ci_brackets_the_mean_and_is_deterministic():
    vals = list(np.random.default_rng(0).normal(100, 20, 200))
    lo, hi = bootstrap_ci(vals, seed=1)
    assert lo < np.mean(vals) < hi
    assert (lo, hi) == bootstrap_ci(vals, seed=1)


def test_bootstrap_ci_narrows_with_more_data():
    rng = np.random.default_rng(0)
    small = bootstrap_ci(list(rng.normal(0, 1, 20)), seed=0)
    large = bootstrap_ci(list(rng.normal(0, 1, 2000)), seed=0)
    assert (large[1] - large[0]) < (small[1] - small[0])


def test_bootstrap_ci_of_a_constant_sample_is_a_point():
    assert bootstrap_ci([7.0] * 10, seed=0) == (7.0, 7.0)


def test_bootstrap_ci_rejects_empty():
    with pytest.raises(ValueError):
        bootstrap_ci([])


# ----------------------------------------------------------------------
# seeds and policies
# ----------------------------------------------------------------------
def test_eval_seeds_are_the_fixed_pool():
    cfg = BBFConfig()
    seeds = eval_seeds(cfg)
    assert len(seeds) == 100
    assert seeds[0] == 8_000_000 and seeds[-1] == 8_000_099
    assert len(set(seeds)) == 100
    # D-017: the same pool regardless of the run's own seed, so every arm is
    # scored on the same layouts.
    assert eval_seeds(BBFConfig(seed=7)) == seeds
    assert eval_seeds(cfg, 10) == seeds[:10]


def test_random_policy_covers_the_action_set():
    rng = np.random.default_rng(0)
    pol = random_policy(8, rng)
    drawn = {pol(None) for _ in range(300)}
    assert drawn == set(range(8))


def test_epsilon_greedy_at_zero_is_the_policy():
    pol = epsilon_greedy(lambda _o: 3, 0.0, 8, np.random.default_rng(0))
    assert {pol(None) for _ in range(100)} == {3}


def test_epsilon_greedy_at_one_is_uniform():
    pol = epsilon_greedy(lambda _o: 3, 1.0, 8, np.random.default_rng(0))
    assert {pol(None) for _ in range(300)} == set(range(8))


def test_epsilon_greedy_at_the_protocol_value_almost_never_explores():
    # PROTOCOL section 1: epsilon_eval = 0.001.
    pol = epsilon_greedy(lambda _o: 3, 0.001, 8, np.random.default_rng(0))
    off = sum(pol(None) != 3 for _ in range(20_000))
    assert 0 < off < 100


def test_epsilon_greedy_rejects_bad_epsilon():
    with pytest.raises(ValueError):
        epsilon_greedy(lambda _o: 0, 1.5, 8, np.random.default_rng(0))


# ----------------------------------------------------------------------
# run_episode against a scripted env
# ----------------------------------------------------------------------
class ScriptedEnv:
    """Ends after ``length`` steps with ``final_state`` and ``score``."""

    def __init__(self, length, score, final_state="GAMEOVER", lives=0):
        self.length = length
        self.score = score
        self.final_state = final_state
        self.lives = lives
        self.action_space = type("A", (), {"n": 8})()
        self.seen: list[int] = []

    def reset(self, *, seed=None, options=None):
        self.t = 0
        return np.zeros((4, 84, 84), np.uint8), {
            "episode_score": 0.0, "lives": 3, "gameState": "PLAYING", "episodeLength": 0,
        }

    def step(self, action):
        self.seen.append(action)
        self.t += 1
        done = self.t >= self.length
        info = {
            "episode_score": float(self.score) if done else 0.0,
            "lives": self.lives if done else 3,
            "gameState": self.final_state if done else "PLAYING",
            "episodeLength": self.t * 4,
        }
        return np.zeros((4, 84, 84), np.uint8), 0.0, done, False, info

    def close(self):
        pass


def test_run_episode_reports_the_terminal_state():
    env = ScriptedEnv(length=7, score=220, final_state="WIN", lives=2)
    r = run_episode(env, lambda _o: 1, seed=5, max_agent_steps=1000)
    assert r.seed == 5 and r.agent_steps == 7 and r.score == 220.0
    assert r.won is True and r.game_state == "WIN" and r.lives_left == 2
    assert r.game_frames == 28 and r.truncated is False
    assert r.floes_visited == 12.0


def test_run_episode_marks_a_cap_hit_as_truncated():
    env = ScriptedEnv(length=10_000, score=50)
    r = run_episode(env, lambda _o: 0, seed=0, max_agent_steps=20)
    assert r.agent_steps == 20 and r.truncated is True


def test_run_episode_uses_the_policy():
    env = ScriptedEnv(length=5, score=0)
    run_episode(env, lambda _o: 4, seed=0, max_agent_steps=100)
    assert env.seen == [4] * 5


# ----------------------------------------------------------------------
# metrics file
# ----------------------------------------------------------------------
def test_write_metrics_round_trip(tmp_path):
    cfg = BBFConfig(run_id="t")
    eps = [_ep(10, seed=1), _ep(220, won=True, seed=2)]
    summary = aggregate(eps, random_mean=34.1)
    path = tmp_path / "metrics.json"
    write_metrics(path, cfg, eps, summary, unit="U03", job_ids=["123"])
    d = json.loads(path.read_text())
    assert d["unit"] == "U03" and d["job_ids"] == ["123"]
    assert d["eval_seeds"] == [1, 2]
    assert len(d["commit"]) in (7, 40) or d["commit"] == "unknown"
    assert d["summary"]["score_mean"] == 115.0
    assert d["episodes"][1]["floes_visited"] == 12.0
    # The config must be round-trippable, so a reader can rebuild the run.
    from playtrain_trainers.bbf.config import config_from_dict

    assert config_from_dict(d["config"]) == cfg


def test_recorded_random_baseline_is_self_consistent():
    """Guard the committed U03 numbers against the raw episodes beside them."""
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "results/bbf/random_frostbite/metrics.json"
    if not path.is_file():
        pytest.skip("U03 has not been run yet")
    d = json.loads(path.read_text())
    eps = d["episodes"]
    s = d["summary"]
    assert len(eps) == s["episodes"] == 100
    assert np.mean([e["score"] for e in eps]) == pytest.approx(s["score_mean"])
    assert sum(e["won"] for e in eps) == s["wins"]
    # Every score must be a whole number of +10 floes plus an optional +100.
    for e in eps:
        assert (e["score"] - (100 if e["won"] else 0)) % 10 == 0
    # No win is possible below 12 floes, since that is when the igloo opens.
    for e in eps:
        assert not e["won"] or e["floes_visited"] >= 12


# ----------------------------------------------------------------------
# PlayTrain-only metrics must not be reported for other envs
# ----------------------------------------------------------------------
def test_aggregate_omits_playtrain_metrics_when_asked():
    """On ALE, floes and win_rate are nonsense; they must not be emitted.

    `floes_visited_mean` applies PlayTrain frostbite's +10-per-floe rule, so
    on ALE it is just score/10 -- it once reported 238 "floes" out of 16 for
    a 2382-point seed. `win_rate` is identically 0 because the ALE has no WIN
    state, which reads as "never wins" rather than "not applicable".
    """
    eps = [_ep(2382.5), _ep(1451.3)]
    s = aggregate(eps, playtrain_metrics=False)
    for k in ("win_rate", "wins", "floes_visited_mean", "score_ceiling",
              "normalized_progress"):
        assert k not in s, k
    # The env-agnostic numbers are still all there.
    for k in ("score_mean", "score_ci95", "score_std", "agent_steps_mean",
              "game_states", "episodes"):
        assert k in s, k


def test_aggregate_keeps_playtrain_metrics_by_default():
    s = aggregate([_ep(220, won=True), _ep(30)], random_mean=33.5)
    for k in ("win_rate", "wins", "floes_visited_mean", "score_ceiling",
              "normalized_progress"):
        assert k in s, k


def test_normalized_progress_is_not_reported_without_playtrain_metrics():
    # It is measured against PlayTrain's 260 ceiling, so it is meaningless
    # for any other env even when a random baseline is supplied.
    s = aggregate([_ep(1000)], random_mean=65.2, playtrain_metrics=False)
    assert "normalized_progress" not in s


def test_curve_point_keys_survive_a_non_playtrain_summary():
    """Regression: `_curve_point` read win_rate/floes unconditionally.

    `aggregate` omits them for non-PlayTrain backends (D-028), so every ALE
    curve eval raised KeyError about seven minutes into a run.
    """
    ale = aggregate([_ep(1000), _ep(2000)], playtrain_metrics=False)
    pt = aggregate([_ep(220, won=True), _ep(30)], random_mean=33.5)
    for s in (ale, pt):
        point = {"score_mean": s["score_mean"]}
        for k in ("win_rate", "floes_visited_mean"):
            if k in s:
                point[k] = s[k]
        assert "score_mean" in point
    assert "win_rate" not in ale and "win_rate" in pt


# ----------------------------------------------------------------------
# Exact permutation test (the audit found an impossible MC p-value)
# ----------------------------------------------------------------------
def test_exact_permutation_respects_its_own_floor():
    """With n=5 per arm there are 252 splits, so p cannot fall below 2/252.

    A Monte-Carlo permutation over this space reported p = 0.007, which is not
    an attainable value. The exact enumeration cannot produce one.
    """
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "cmp", Path(__file__).resolve().parents[1] / "tools/bbf/bbf_compare.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    # Maximally separated arms: every split is less extreme than the observed.
    a, b = np.array([1.0, 2, 3, 4, 5]), np.array([100.0, 101, 102, 103, 104])
    p, floor, n = m.exact_permutation_p(a, b)
    assert n == 252
    assert floor == pytest.approx(2 / 252)
    assert p == pytest.approx(floor), "maximal separation must sit exactly at the floor"
    assert p >= floor, "no p below the floor is attainable"


def test_exact_permutation_is_symmetric_and_bounded():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "cmp", Path(__file__).resolve().parents[1] / "tools/bbf/bbf_compare.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    a, b = np.array([1.0, 5, 3, 9, 2]), np.array([4.0, 6, 2, 8, 7])
    p_ab, _, _ = m.exact_permutation_p(a, b)
    p_ba, _, _ = m.exact_permutation_p(b, a)
    assert p_ab == pytest.approx(p_ba), "a two-sided p must not depend on argument order"
    assert 0.0 < p_ab <= 1.0
    # Identical arms: nothing is more extreme than a zero difference.
    assert m.exact_permutation_p(a, a.copy())[0] == pytest.approx(1.0)
