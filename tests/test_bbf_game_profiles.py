"""Per-game scoring profiles (D-044).

frostbite's score means something specific: +10 per floe, +100 for the igloo,
a 260 ceiling and a WIN state. venture, amidar and star_gunner loop levels
with an unbounded score and no WIN. Reporting frostbite's fields for them
would be inventing numbers, which is exactly what `floes_visited_mean = 238
out of 16` was on ALE before D-028.
"""
from __future__ import annotations

import pytest

from playtrain_trainers.bbf.config import BBFConfig
from playtrain_trainers.bbf.evaluate import (
    GAME_PROFILES,
    EpisodeResult,
    aggregate,
    game_profile,
)

FROSTBITE_ONLY = ("win_rate", "wins", "floes_visited_mean", "score_ceiling",
                  "normalized_progress")
UNBOUNDED = ("venture", "amidar", "star_gunner")


def _eps(scores, won=False, state="GAMEOVER"):
    return [EpisodeResult(seed=i, score=s, agent_steps=10, game_frames=40,
                          lives_left=0, game_state=state, won=won, truncated=False)
            for i, s in enumerate(scores)]


def _cfg(game, backend="playtrain"):
    return BBFConfig(run_id="t", seed=0, game=game, env_backend=backend)


@pytest.mark.parametrize("game", UNBOUNDED)
def test_unbounded_games_report_no_frostbite_field(game):
    out = aggregate(_eps([100.0, 300.0]), random_mean=20.0,
                    profile=game_profile(_cfg(game)))
    assert not [k for k in FROSTBITE_ONLY if k in out]


@pytest.mark.parametrize("game", UNBOUNDED)
def test_unbounded_games_report_score_over_random(game):
    out = aggregate(_eps([100.0, 300.0]), random_mean=20.0,
                    profile=game_profile(_cfg(game)))
    assert out["score_over_random"] == pytest.approx(180.0)


def test_frostbite_still_reports_its_own_fields_unchanged():
    """The v4 numbers must not move under D-044."""
    eps = _eps([160.0, 260.0], won=True, state="WIN")
    out = aggregate(eps, random_mean=33.5, profile=game_profile(_cfg("frostbite")))
    assert out["win_rate"] == 1.0 and out["wins"] == 2
    assert out["score_ceiling"] == 260.0
    # (210 - 33.5) / (260 - 33.5)
    assert out["normalized_progress"] == pytest.approx(176.5 / 226.5)
    assert out["floes_visited_mean"] == pytest.approx(11.0)


def test_the_legacy_flag_still_selects_frostbite():
    """Callers predating `profile` passed playtrain_metrics=True and meant frostbite."""
    eps = _eps([160.0], won=True, state="WIN")
    assert aggregate(eps, random_mean=33.5, playtrain_metrics=True)["score_ceiling"] == 260.0
    assert "win_rate" not in aggregate(eps, playtrain_metrics=False)


def test_ale_never_gets_a_playtrain_profile():
    p = game_profile(_cfg("frostbite", backend="ale"))
    assert p.ceiling is None and not p.has_win_state and not p.floe_scoring


def test_an_unlisted_game_falls_back_to_the_plain_profile():
    """A new game must under-report, never mis-report."""
    p = game_profile(_cfg("some_game_added_later"))
    assert p.ceiling is None and not p.has_win_state and not p.floe_scoring


def test_every_profiled_game_exists_in_the_catalog():
    from pathlib import Path
    games = Path(__file__).resolve().parents[2] / "playtrain" / "games" / "js"
    missing = [g for g in GAME_PROFILES if not (games / f"{g}.js").is_file()]
    assert not missing, f"profiled but absent from the catalog: {missing}"
