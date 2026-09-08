"""envpool_kwargs must reach envpool.make_gymnasium.

This is what makes an ALE row matchable. EnvPool's Atari defaults (84x84 gray,
stack_num=4, frame_skip=4) do not match a PlayTrain replica's 64x64x3 RGB single
frame at fs=1, so without these kwargs an Atari A/B would differ in what the
encoder sees as well as in the environment — the same class of confound that made
the ProcGen row unmatched.

envpool has no macOS wheels, so the import is stubbed; these tests check the
plumbing (config -> env_spec -> EnvPoolVec -> make_gymnasium), not envpool itself.
"""
from __future__ import annotations

import sys
import types

import pytest

ATARI_MATCHED = {"img_height": 64, "img_width": 64, "gray_scale": False,
                 "stack_num": 1, "frame_skip": 1}


@pytest.fixture
def fake_envpool(monkeypatch):
    """Stub envpool, recording exactly what make_gymnasium was called with."""
    calls = []

    class _Env:
        def reset(self):
            return None, {}

        def step(self, a):
            return None, None, None, None, {}

    mod = types.ModuleType("envpool")

    def make_gymnasium(env_id, **kw):
        calls.append((env_id, kw))
        return _Env()

    mod.make_gymnasium = make_gymnasium
    monkeypatch.setitem(sys.modules, "envpool", mod)
    return calls


def test_kwargs_are_forwarded(fake_envpool):
    from playtrain_trainers.impala.envpool_vec import EnvPoolVec
    EnvPoolVec("Pong-v5", num_envs=8, num_threads=4, env_kwargs=ATARI_MATCHED)
    env_id, kw = fake_envpool[0]
    assert env_id == "Pong-v5"
    assert kw["num_envs"] == 8 and kw["num_threads"] == 4
    for k, v in ATARI_MATCHED.items():
        assert kw[k] == v, f"{k} not forwarded: {kw.get(k)!r} != {v!r}"


def test_absent_kwargs_changes_nothing(fake_envpool):
    """ProcGen rows pass nothing; make_gymnasium must see only the base args, so
    the existing 429k measurement is unaffected by this change."""
    from playtrain_trainers.impala.envpool_vec import EnvPoolVec
    EnvPoolVec("BigfishEasy-v0", num_envs=4)
    _env_id, kw = fake_envpool[0]
    assert set(kw) == {"num_envs", "num_threads"}, f"unexpected extras: {kw}"


def test_gray_scale_false_survives_as_false(fake_envpool):
    """Guard against a truthiness bug: gray_scale=False must arrive as False,
    not be dropped by an `if v:` filter — silently reverting ALE to grayscale
    would make the A/B unmatched again while still running."""
    from playtrain_trainers.impala.envpool_vec import EnvPoolVec
    EnvPoolVec("Pong-v5", num_envs=2, env_kwargs={"gray_scale": False})
    _env_id, kw = fake_envpool[0]
    assert "gray_scale" in kw and kw["gray_scale"] is False


def test_config_field_reaches_env_spec():
    """ImpalaConfig.envpool_kwargs must be placed into env_spec, or the worker
    never sees it. Checked by source inspection: building a real env_spec means
    running train()."""
    import inspect

    from playtrain_trainers.impala.train import ImpalaConfig, train
    assert "envpool_kwargs" in {f for f in ImpalaConfig.__dataclass_fields__}
    src = inspect.getsource(train)
    assert "envpool_kwargs=cfg.envpool_kwargs" in src

    from playtrain_trainers.impala import vec_actor
    vsrc = inspect.getsource(vec_actor)
    assert 'env_kwargs=env_spec.get("envpool_kwargs")' in vsrc


def test_hwc_transposes_atari_chw(fake_envpool):
    """With gray_scale=False + stack_num=1 envpool emits (N,3,64,64); act_vec
    needs (N,64,64,3)."""
    import numpy as np

    from playtrain_trainers.impala.envpool_vec import EnvPoolVec
    chw = np.zeros((4, 3, 64, 64), dtype=np.uint8)
    out = EnvPoolVec._hwc(chw)
    assert out.shape == (4, 64, 64, 3)
    # ProcGen already HWC — must pass through untouched
    hwc = np.zeros((4, 64, 64, 3), dtype=np.uint8)
    assert EnvPoolVec._hwc(hwc).shape == (4, 64, 64, 3)
