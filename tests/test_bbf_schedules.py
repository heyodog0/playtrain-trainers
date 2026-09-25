"""Tests for playtrain_trainers.bbf.schedules.

The anneal values at 0 / 5k / 10k gradient steps since a reset are checked
against the closed form computed here independently of the module, and the
reset schedule is checked against an independent transcription of the
official `_train_step` / `reset_weights` rule (env-step clock, D-033).
"""
from __future__ import annotations

import math

import pytest

from playtrain_trainers.bbf.config import BBFConfig
from playtrain_trainers.bbf.schedules import (
    ResetSchedule,
    discount,
    epsilon,
    exponential_anneal,
    reset_env_steps,
    update_horizon,
)


def log_lerp(step, period, a, b):
    """Independent reference: log-space interpolation, clipped."""
    frac = max(0.0, min(1.0, (period - step) / period))
    return math.exp(frac * (math.log(a) - math.log(b)) + math.log(b))


# ----------------------------------------------------------------------
# exponential_anneal
# ----------------------------------------------------------------------
def test_anneal_hits_both_endpoints_exactly():
    assert exponential_anneal(0, 100, 10.0, 3.0) == pytest.approx(10.0)
    assert exponential_anneal(100, 100, 10.0, 3.0) == pytest.approx(3.0)


def test_anneal_holds_the_final_value_past_the_period():
    for s in (101, 500, 10_000):
        assert exponential_anneal(s, 100, 10.0, 3.0) == pytest.approx(3.0)


def test_anneal_matches_the_reference_throughout():
    for s in range(0, 101, 7):
        assert exponential_anneal(s, 100, 10.0, 3.0) == pytest.approx(
            log_lerp(s, 100, 10.0, 3.0)
        )


def test_anneal_is_geometric_not_linear():
    """The midpoint is the geometric mean, which is below the arithmetic one."""
    mid = exponential_anneal(50, 100, 10.0, 3.0)
    assert mid == pytest.approx(math.sqrt(10.0 * 3.0))
    assert mid < (10.0 + 3.0) / 2


def test_anneal_works_upward():
    assert exponential_anneal(0, 100, 0.97, 0.997) == pytest.approx(0.97)
    assert exponential_anneal(100, 100, 0.97, 0.997) == pytest.approx(0.997)
    assert 0.97 < exponential_anneal(50, 100, 0.97, 0.997) < 0.997


def test_anneal_rejects_nonpositive_endpoints_and_periods():
    with pytest.raises(ValueError):
        exponential_anneal(0, 0, 10.0, 3.0)
    with pytest.raises(ValueError):
        exponential_anneal(0, 10, 0.0, 3.0)


# ----------------------------------------------------------------------
# The horizon and discount at the asked-for steps
# ----------------------------------------------------------------------
def test_update_horizon_at_the_recorded_steps():
    """`cycle_grad_steps` is GRADIENT steps since the last reset (D-033)."""
    cfg = BBFConfig()  # cycle_steps 10k, 10 -> 3
    assert update_horizon(0, cfg) == 10
    # 5k: geometric midpoint of 10 and 3 is 5.477 -> 5
    assert update_horizon(5_000, cfg) == 5
    assert update_horizon(10_000, cfg) == 3
    assert update_horizon(50_000, cfg) == 3


def test_update_horizon_midpoint_is_the_geometric_mean():
    cfg = BBFConfig()
    raw = exponential_anneal(5_000, cfg.cycle_steps, 10.0, 3.0)
    assert raw == pytest.approx(math.sqrt(30.0))
    assert update_horizon(5_000, cfg) == round(raw)


def test_discount_at_the_recorded_steps():
    cfg = BBFConfig()
    assert discount(0, cfg) == pytest.approx(0.97)
    assert discount(10_000, cfg) == pytest.approx(0.997)
    assert discount(15_000, cfg) == pytest.approx(0.997)
    assert discount(5_000, cfg) == pytest.approx(log_lerp(5_000, 10_000, 0.97, 0.997))


def test_horizon_never_drops_below_one():
    cfg = BBFConfig(max_update_horizon=2, update_horizon=1, cycle_steps=10)
    assert all(update_horizon(s, cfg) >= 1 for s in range(0, 50))


def test_schedules_reject_negative_steps():
    with pytest.raises(ValueError):
        update_horizon(-1, BBFConfig())
    with pytest.raises(ValueError):
        discount(-1, BBFConfig())


# ----------------------------------------------------------------------
# Resets are on the ENV-step clock (D-033)
# ----------------------------------------------------------------------
def official_reset_steps(reset_every, no_resets_after, training_steps, offset=1):
    """Independent transcription of spr_agent.BBFAgent._train_step /
    reset_weights, with `training_steps` ticking once per env step."""
    next_reset = reset_every + offset
    fired = []
    for training_steps_now in range(training_steps):
        if reset_every > 0 and training_steps_now > next_reset:
            interval = reset_every
            next_reset = interval + training_steps_now
            if not next_reset > no_resets_after + offset:
                fired.append(training_steps_now)
    return fired


def test_reset_env_steps_at_rr2_match_the_official_rule():
    cfg = BBFConfig()  # 100k env steps, every 20k, none past 100k
    want = official_reset_steps(20_000, 100_000, 100_000)
    assert reset_env_steps(cfg) == want
    # ~20k, 40k, 60k env steps -- the paper's "every 40k gradient steps" at 2
    # gradient steps per env step. The reset that would be due at ~80k is
    # refused because `reset_weights` requires a full interval before
    # `no_resets_after` ("need at least 20000 before 100000 to recover").
    # NOT 5 resets at 10k..50k.
    assert [round(s, -3) for s in want] == [20_000, 40_000, 60_000]


def test_reset_env_steps_at_rr8_paper_value():
    """The paper's RR=8: still 40k gradient steps, i.e. 5k env steps."""
    cfg = BBFConfig(replay_ratio=256, reset_every=5_000)
    got = reset_env_steps(cfg)
    assert got == official_reset_steps(5_000, 100_000, 100_000)
    assert len(got) == 18
    assert got[0] > 5_000 and 90_000 < got[-1] < 95_000


def test_replay_ratio_does_not_move_the_resets_on_its_own():
    """Same reset_every, different RR: identical env-step schedule."""
    assert reset_env_steps(BBFConfig()) == reset_env_steps(BBFConfig(replay_ratio=256))


def test_no_reset_at_step_zero():
    sched = ResetSchedule(BBFConfig())
    assert not sched.due(0)
    assert not sched.due(20_000)
    assert sched.due(20_002)


def test_reset_schedule_refuses_past_no_resets_after():
    cfg = BBFConfig(reset_every=30_000, no_resets_after=100_000)
    got = reset_env_steps(cfg)
    # 30k and 60k fire; at ~90k the following reset would land at ~120k > 100k,
    # so that one is refused.
    assert [round(s, -3) for s in got] == [30_000, 60_000]
    sched = ResetSchedule(cfg)
    for s in got:
        assert sched.due(s) and sched.fire(s)
    assert sched.count == 2
    assert sched.due(200_000) and not sched.fire(200_000)


def test_reset_every_zero_disables_resets():
    assert reset_env_steps(BBFConfig(reset_every=0)) == []


def test_anneal_fraction_is_25_percent_at_both_replay_ratios():
    """Paper: 'the annealing phase is always 25% of training, regardless of
    the replay ratio' -- 10k gradient steps of a 40k-gradient-step interval."""
    for rr, reset_every in ((64, 20_000), (256, 5_000)):
        cfg = BBFConfig(replay_ratio=rr, reset_every=reset_every)
        interval_grad = reset_every * cfg.gradient_steps_per_env_step
        assert interval_grad == 40_000
        assert cfg.cycle_steps / interval_grad == pytest.approx(0.25)


# ----------------------------------------------------------------------
# Epsilon
# ----------------------------------------------------------------------
def test_epsilon_is_one_during_the_warmup():
    cfg = BBFConfig()  # min_replay_history 2000
    assert epsilon(0, cfg) == 1.0
    assert epsilon(1_999, cfg) == 1.0


def test_epsilon_decays_to_epsilon_train():
    cfg = BBFConfig()  # decay over 2001 steps to 0.0
    assert epsilon(2_000, cfg) == pytest.approx(1.0)
    assert epsilon(2_000 + 2_001, cfg) == pytest.approx(cfg.epsilon_train)
    assert epsilon(100_000, cfg) == pytest.approx(cfg.epsilon_train)
    mid = epsilon(2_000 + 1_000, cfg)
    assert 0.4 < mid < 0.6


def test_epsilon_is_monotone():
    cfg = BBFConfig()
    vals = [epsilon(s, cfg) for s in range(0, 6_000, 50)]
    assert all(a >= b - 1e-12 for a, b in zip(vals, vals[1:], strict=False))
