"""vec_double_buffer must not silently override vec_backend.

act_vec_db (the double-buffered vec worker) hardcodes PingPongVecEnv and never
reads vec_backend, so `vec_double_buffer=True` + `vec_backend="envpool"` would run
PlayTrain's own envs while every log line and the saved config.json claim envpool.
That is the one misconfiguration a benchmark cannot self-detect: the run succeeds,
the numbers look plausible, and the system under test has quietly become its own
baseline.

Tests target check_vec_backend_compatible() directly rather than train(), because
a config that PASSES validation proceeds to spawn actors — a test that called
train() to assert "this combination is allowed" would start a real training run.
"""
from __future__ import annotations

import inspect

import pytest

from playtrain_trainers.impala.train import check_vec_backend_compatible


def test_double_buffer_with_envpool_is_rejected():
    with pytest.raises(ValueError, match="vec_double_buffer"):
        check_vec_backend_compatible("envpool", True)


def test_error_names_the_conflict_and_the_cause():
    """The message must name both settings and the function that ignores one of
    them, or the next person disables the wrong knob."""
    with pytest.raises(ValueError) as exc:
        check_vec_backend_compatible("envpool", True)
    msg = str(exc.value)
    assert "vec_backend" in msg and "envpool" in msg
    assert "act_vec_db" in msg
    # It must also say that turning the flag off is not a free fix.
    assert "single-buffered" in msg


@pytest.mark.parametrize("backend,double_buffer", [
    ("native", True),    # the 1.12M record topology
    ("native", False),   # single-buffered native
    ("envpool", False),  # the baseline A/B, exactly as it was run
])
def test_combinations_actually_used_are_allowed(backend, double_buffer):
    check_vec_backend_compatible(backend, double_buffer)  # must not raise


def test_guard_rejects_any_future_non_native_backend():
    """Written as backend != 'native' rather than == 'envpool', so a third
    backend added later inherits the protection instead of silently skipping it."""
    with pytest.raises(ValueError):
        check_vec_backend_compatible("some_future_backend", True)


def test_train_still_calls_the_guard():
    """Guard against the refactor being orphaned: extracting it from train() is
    only safe while train() still invokes it."""
    from playtrain_trainers.impala import train as train_mod
    src = inspect.getsource(train_mod.train)
    assert "check_vec_backend_compatible(" in src
