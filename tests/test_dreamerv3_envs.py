"""U02 — the env stack, against `embodied/envs/atari.py` at the pinned commit.

ale-py is not installed on this Mac (the BBF loop ran its ALE arm only on the
cluster), so :class:`FakeALE` stands in for the emulator: it replays a scripted
sequence of screens and game-over/lives signals. That is not a weaker test of the
port than a real ROM would be -- every line of `atari.py` that this file checks is
control flow around the ALE handle (the repeat loop, when frames are rendered, the
max-pool, the no-op law, the episode cap, the terminal flags), and a fake makes those
observable frame by frame in a way a real emulator does not. The real screen is
inspected by eye in U07's cluster smoke.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from playtrain_trainers.dreamerv3 import config as C
from playtrain_trainers.dreamerv3 import envs as E

ARTIFACTS = Path(__file__).resolve().parents[1] / "results" / "dreamerv3" / "env"


class FakeALE:
    """A scripted stand-in for ``ale_py.ALEInterface``.

    Screen ``t`` is a solid colour whose red channel is ``t``, so the max over any
    set of frames is decidable by hand. ``act`` returns the scripted reward.
    """

    def __init__(
        self,
        dims: tuple[int, int] = (8, 6),
        actionset: tuple[int, ...] = (0, 1, 2, 3),
        rewards: list[float] | None = None,
        game_over_at: int | None = None,
        lives_schedule: dict[int, int] | None = None,
        start_lives: int = 3,
    ):
        self.dims = dims
        self.actionset = tuple(actionset)
        self.rewards = list(rewards or [])
        self.game_over_at = game_over_at
        self.lives_schedule = dict(lives_schedule or {})
        self.start_lives = start_lives
        self.t = 0  # frames acted since the last reset_game
        self.resets = 0
        self.noop_frames = 0
        self.floats: dict[str, float] = {}
        self.ints: dict[bytes, int] = {}
        self._lives = start_lives
        self.acted: list[int] = []

    # --- the ALE surface atari.py uses -------------------------------
    def setLoggerMode(self, mode):  # noqa: N802 - the ALE spelling
        pass

    def setInt(self, key, value):  # noqa: N802
        self.ints[key] = value

    def setFloat(self, key, value):  # noqa: N802
        self.floats[key] = value

    def getLegalActionSet(self):  # noqa: N802
        return tuple(range(18))

    def getMinimalActionSet(self):  # noqa: N802
        return self.actionset

    def getScreenDims(self):  # noqa: N802
        return self.dims

    def getScreenRGB(self, buffer):  # noqa: N802
        buffer[...] = 0
        buffer[..., 0] = min(self.t, 255)

    def act(self, action):
        self.acted.append(action)
        if action == E.ACTION_MEANING.index("NOOP") and self._in_noops:
            self.noop_frames += 1
        self.t += 1
        if self.t in self.lives_schedule:
            self._lives = self.lives_schedule[self.t]
        reward = self.rewards[self.t - 1] if self.t - 1 < len(self.rewards) else 0.0
        return reward

    def game_over(self):
        return self.game_over_at is not None and self.t >= self.game_over_at

    def lives(self):
        return self._lives

    def reset_game(self):
        self.resets += 1
        self.t = 0
        self._lives = self.start_lives

    _in_noops = True


def _cfg(**kw) -> C.DreamerConfig:
    return C.config_from_dict({"preset": "atari100k", **kw})


def _ale_env(cfg=None, seed=0, **fake):
    cfg = cfg or _cfg(env_backend="ale")
    ale = FakeALE(**fake)
    env = E.make_ale_dreamer("frostbite", seed, cfg, ale=ale)
    return env, ale


# ----------------------------------------------------------------------
# The pixel pipeline, checked against a hand computation
# ----------------------------------------------------------------------
def test_max_pool_over_the_last_two_frames_by_hand() -> None:
    """`_obs` takes `np.amax` over the pooling buffers, per pixel, per channel."""
    a = np.zeros((4, 4, 3), np.uint8)
    b = np.zeros((4, 4, 3), np.uint8)
    a[0, 0] = (10, 200, 0)
    b[0, 0] = (90, 5, 7)
    a[1, 1] = (255, 0, 0)
    b[2, 2] = (0, 0, 255)
    out = E.aggregate_and_resize([a, b], (4, 4), "max", "pillow", False)
    # Pillow resizing 4x4 to 4x4 is the identity, so this is the pool alone.
    assert tuple(out[0, 0]) == (90, 200, 7)
    assert tuple(out[1, 1]) == (255, 0, 0)
    assert tuple(out[2, 2]) == (0, 0, 255)
    assert out.dtype == np.uint8


def test_mean_aggregate_by_hand() -> None:
    a = np.full((2, 2, 3), 10, np.uint8)
    b = np.full((2, 2, 3), 20, np.uint8)
    out = E.aggregate_and_resize([a, b], (2, 2), "mean", "pillow", False)
    assert (out == 15).all()


def test_pillow_bilinear_downscale_matches_pillow_directly() -> None:
    """The resize is Pillow BILINEAR on the pooled RGB image, not opencv INTER_AREA
    (which is what gymnasium's AtariPreprocessing, and so the BBF ALE arm, uses)."""
    rng = np.random.default_rng(0)
    screen = rng.integers(0, 256, (210, 160, 3), dtype=np.uint8)
    out = E.aggregate_and_resize([screen], (64, 64), "max", "pillow", False)
    expected = np.array(Image.fromarray(screen).resize((64, 64), Image.BILINEAR))
    assert out.shape == (64, 64, 3)
    assert (out == expected).all()


def test_grayscale_weights() -> None:
    px = np.zeros((1, 1, 3), np.uint8)
    px[0, 0] = (100, 150, 200)
    out = E.aggregate_and_resize([px], (1, 1), "max", "pillow", True)
    expected = int(100 * 0.299 + 150 * 0.587 + 200 * (1 - 0.299 - 0.587))
    assert out.shape == (1, 1, 1)
    assert out[0, 0, 0] == expected


# ----------------------------------------------------------------------
# The step loop
# ----------------------------------------------------------------------
def test_obs_keys_and_dtypes() -> None:
    env, _ = _ale_env()
    obs = env.step({"action": 0, "reset": True})
    assert set(obs) == set(E.OBS_KEYS) | {"info"}
    assert obs["image"].dtype == np.uint8
    assert obs["image"].shape == (64, 64, 3)
    assert obs["reward"].dtype == np.float32
    assert obs["is_first"] is True
    assert obs["is_last"] is False
    assert obs["is_terminal"] is False
    assert set(obs["info"]) == {"score", "lives", "frames"}


def test_first_step_resets_even_without_the_reset_flag() -> None:
    env, ale = _ale_env()
    obs = env.step({"action": 0, "reset": False})
    assert obs["is_first"] is True
    assert ale.resets == 1


def test_repeat_acts_four_times_and_renders_the_last_two() -> None:
    """atari.py: `for repeat in range(4)` acts every frame but calls `_render` only
    when `repeat >= repeat_count - pooling`, i.e. on the 3rd and 4th frames."""
    env, ale = _ale_env()
    env.step({"action": 0, "reset": True})
    ale.t = 0  # ignore the no-op frames; count only the repeat loop
    env.step({"action": 2, "reset": False})
    assert ale.t == 4
    assert ale.acted[-4:] == [env.actionset[2]] * 4
    # Screens t=3 and t=4 were rendered (red = t at the time of the render), so the
    # max is 4. Screens 1 and 2 were acted but never read.
    obs = env._obs(0.0)
    assert obs["image"][0, 0, 0] == 4


def test_reset_fills_every_pooling_buffer_with_the_first_frame() -> None:
    """The pinned commit IS this fix ("Fix Atari frame maxpooling on reset"): without
    it the first observation of an episode maxes against a stale screen from the
    episode before, which on a bright final frame dominates the pool for one step."""
    env, ale = _ale_env(seed=3)
    env.seed_episode(11)
    env.step({"action": 0, "reset": True})
    ale.t = 900  # brighter (clamped to 255) than anything the next episode shows
    env.step({"action": 1, "reset": False})
    assert max(b[0, 0, 0] for b in env.buffers) == 255

    env.seed_episode(12)
    env.step({"action": 0, "reset": True})
    # After the reset the buffers hold ONE frame, copied: the screen at the end of
    # the no-ops, not a max against the stale 255.
    expected = E.noop_count(30, 3, 12, 0)
    assert {int(b[0, 0, 0]) for b in env.buffers} == {expected}
    assert env._obs(0.0)["image"][0, 0, 0] == expected


def test_reward_is_summed_over_the_repeat_and_unclipped() -> None:
    env, ale = _ale_env(rewards=[0.0] * 40)
    env.step({"action": 0, "reset": True})
    base = ale.t
    ale.rewards = [0.0] * base + [10.0, 0.0, -3.0, 100.0]
    obs = env.step({"action": 1, "reset": False})
    assert obs["reward"] == np.float32(107.0)
    assert obs["info"]["score"] == 107.0


def test_clip_reward_takes_the_sign() -> None:
    cfg = C.config_from_dict({"preset": "atari100k", "env_backend": "ale"})
    cfg.env.clip_reward = True
    env, ale = _ale_env(cfg=cfg, rewards=[0.0] * 60)
    env.step({"action": 0, "reset": True})
    ale.rewards = [0.0] * ale.t + [5.0, 0.0, 0.0, 0.0]
    obs = env.step({"action": 1, "reset": False})
    assert obs["reward"] == np.float32(1.0)


def test_game_over_sets_is_last_and_is_terminal_and_breaks_the_repeat() -> None:
    env, ale = _ale_env()
    env.step({"action": 0, "reset": True})
    ale.t = 0
    ale.game_over_at = 2
    obs = env.step({"action": 1, "reset": False})
    assert obs["is_last"] is True
    assert obs["is_terminal"] is True
    assert ale.t == 2  # broke out of the repeat loop early


def test_the_frame_cap_is_counted_in_frames_and_reported_as_terminal() -> None:
    """D-006 and D-008: `duration` counts ALE frames, not agent steps, and the
    truncation at the cap reaches the agent as `is_terminal` because `_obs` returns
    `is_last` for both. This is the official quirk, copied deliberately."""
    cfg = _cfg(env_backend="ale")
    cfg.env.length = 12
    env, ale = _ale_env(cfg=cfg)
    env.step({"action": 0, "reset": True})
    steps = 0
    while True:
        obs = env.step({"action": 1, "reset": False})
        steps += 1
        if obs["is_last"]:
            break
        assert steps < 20
    assert env.duration == 12
    assert steps == 3  # 12 frames / repeat 4
    assert obs["is_last"] is True
    assert obs["is_terminal"] is True  # the quirk: a truncation, reported terminal
    assert not ale.game_over()


def test_lives_are_unused_at_the_frozen_config() -> None:
    """D-004: a life loss is neither a terminal nor a reset."""
    env, ale = _ale_env(lives_schedule={2: 2, 6: 1}, start_lives=3)
    env.step({"action": 0, "reset": True})
    ale.t = 0
    ale._lives = 3
    obs = env.step({"action": 1, "reset": False})
    assert obs["is_last"] is False
    assert obs["is_terminal"] is False
    assert obs["info"]["lives"] == 2


def test_lives_reset_mode_still_works_if_someone_unfreezes_it() -> None:
    cfg = _cfg(env_backend="ale")
    cfg.env.lives = "reset"
    env, ale = _ale_env(cfg=cfg, lives_schedule={2: 2})
    env.step({"action": 0, "reset": True})
    ale.t = 0
    ale._lives = 3
    obs = env.step({"action": 1, "reset": False})
    assert obs["is_last"] is True
    assert obs["is_terminal"] is True


def test_frames_clock_in_info() -> None:
    env, _ = _ale_env()
    env.step({"action": 0, "reset": True})
    for i in range(1, 4):
        obs = env.step({"action": 1, "reset": False})
        assert obs["info"]["frames"] == 4 * i


def test_act_space_is_the_minimal_action_set() -> None:
    env, _ = _ale_env(actionset=(0, 1, 3, 4, 11, 12))
    assert env.act_space["action"][3] == 6
    assert env.obs_space["image"][1] == (64, 64, 3)


# ----------------------------------------------------------------------
# No-ops (D-005)
# ----------------------------------------------------------------------
def test_noop_law_includes_zero_and_the_max() -> None:
    """atari.py draws `rng.integers(noops + 1)`: 0..30 inclusive. Dopamine (and the
    BBF port) draws 1..30, which is a different law."""
    rng = np.random.default_rng(0)
    draws = {E.noop_count(30, 0, None, 0, rng) for _ in range(4000)}
    assert min(draws) == 0
    assert max(draws) == 30


def test_noop_count_is_replayable_from_the_episode_seed() -> None:
    a = E.noop_count(30, 7, 1234, 0)
    b = E.noop_count(30, 7, 1234, 0)
    assert a == b
    assert E.noop_count(30, 7, 1235, 0) != a or E.noop_count(30, 7, 1236, 0) != a
    # A retry draws a different count than the attempt that just failed.
    assert E.noop_count(30, 7, 1234, 1) != a or E.noop_count(30, 7, 1234, 2) != a


def test_noop_count_without_an_episode_seed_needs_a_running_generator() -> None:
    with pytest.raises(ValueError, match="running generator"):
        E.noop_count(30, 0, None, 0)


def test_noops_are_taken_at_reset_and_are_not_agent_steps() -> None:
    env, ale = _ale_env(seed=3)
    env.seed_episode(99)
    expected = E.noop_count(30, 3, 99, 0)
    env.step({"action": 0, "reset": True})
    assert env.last_noops == expected
    assert ale.t == expected
    assert env.duration == 0  # the no-op frames are not on the episode's clock


# ----------------------------------------------------------------------
# PlayTrain backend
# ----------------------------------------------------------------------
class FakePlayTrain:
    """Minimal stand-in for `PlayTrainEnv`: HWC frames, gym-style returns."""

    class _Space:
        n = 8

    def __init__(self, end_at: int | None = None, truncate_at: int | None = None):
        self.action_space = self._Space()
        self.steps = 0
        self.frames = 0
        self.score = 0.0
        self.end_at = end_at
        self.truncate_at = truncate_at
        self.seeds: list[int | None] = []
        self.closed = False

    def _obs(self):
        return np.full((64, 64, 3), min(self.frames, 255), np.uint8)

    def _info(self):
        return {
            "score": self.score,
            "lives": 3,
            "episodeLength": self.frames,
            "gameState": "PLAYING",
        }

    def reset(self, *, seed=None, options=None):
        self.seeds.append(seed)
        self.steps = 0
        self.frames = 0
        self.score = 0.0
        return self._obs(), self._info()

    def step(self, action):
        self.steps += 1
        self.frames += 4  # the runtime applies the frame skip itself
        self.score += 10.0
        term = self.end_at is not None and self.steps >= self.end_at
        trunc = self.truncate_at is not None and self.steps >= self.truncate_at
        return self._obs(), 10.0, term, trunc, self._info()

    def close(self):
        self.closed = True


def test_playtrain_backend_obs_contract() -> None:
    cfg = _cfg(env_backend="playtrain")
    fake = FakePlayTrain()
    env = E.make_playtrain_dreamer("frostbite", 0, cfg, env=fake)
    obs = env.step({"action": 0, "reset": True})
    assert set(obs) == set(E.OBS_KEYS) | {"info"}
    assert obs["image"].shape == (64, 64, 3)
    assert obs["image"].dtype == np.uint8
    assert obs["is_first"] is True
    assert obs["info"]["frames"] == 0
    assert obs["info"]["score"] == 0.0


def test_playtrain_clocks_exclude_the_reset_noops() -> None:
    cfg = _cfg(env_backend="playtrain")
    fake = FakePlayTrain()
    env = E.make_playtrain_dreamer("frostbite", 5, cfg, env=fake)
    env.seed_episode(42)
    n = E.noop_count(30, 5, 42, 0)
    env.step({"action": 0, "reset": True})
    assert env.last_noops == n
    obs = env.step({"action": 1, "reset": False})
    # One agent step after the no-ops: 4 frames and one reward on the episode clock,
    # whatever the no-ops did to the runtime's own counters.
    assert obs["info"]["frames"] == 4
    assert obs["info"]["score"] == 10.0
    assert obs["reward"] == np.float32(10.0)


def test_playtrain_terminal_and_truncation_both_report_is_terminal() -> None:
    # noops 0 so the scripted end lands on an agent step, not inside the reset.
    cfg = _cfg(env_backend="playtrain")
    cfg.env.noops = 0
    for kwargs in ({"end_at": 2}, {"truncate_at": 2}):
        fake = FakePlayTrain(**kwargs)
        env = E.make_playtrain_dreamer("frostbite", 0, cfg, env=fake)
        env.seed_episode(1)
        env.step({"action": 0, "reset": True})
        env.step({"action": 1, "reset": False})
        obs = env.step({"action": 1, "reset": False})
        assert obs["is_last"] is True
        assert obs["is_terminal"] is True  # D-008, copied from atari.py


def test_playtrain_reset_retries_when_the_noops_end_the_episode() -> None:
    cfg = _cfg(env_backend="playtrain")

    class EndsDuringNoops(FakePlayTrain):
        def __init__(self):
            super().__init__()
            self.attempts = 0

        def reset(self, *, seed=None, options=None):
            self.attempts += 1
            return super().reset(seed=seed, options=options)

        def step(self, action):
            obs, r, term, trunc, info = super().step(action)
            if self.attempts == 1:
                term = True
            return obs, r, term, trunc, info

    fake = EndsDuringNoops()
    env = E.make_playtrain_dreamer("frostbite", 0, cfg, env=fake)
    env.seed_episode(7)
    env.step({"action": 0, "reset": True})
    assert fake.attempts >= 2
    # The seed is unchanged across retries, so the episode stays replayable.
    assert set(fake.seeds) == {7}


def test_playtrain_refuses_grayscale_and_non_square() -> None:
    with pytest.raises(ValueError, match="D-002"):
        E.PlayTrainDreamer("frostbite", gray=True, env=FakePlayTrain())
    with pytest.raises(ValueError, match="square"):
        E.PlayTrainDreamer("frostbite", size=(64, 32), env=FakePlayTrain())


def test_playtrain_rejects_a_frame_of_the_wrong_shape() -> None:
    class WrongSize(FakePlayTrain):
        def _obs(self):
            return np.zeros((84, 84, 3), np.uint8)

    env = E.PlayTrainDreamer("frostbite", env=WrongSize())
    with pytest.raises(ValueError, match="expected"):
        env.step({"action": 0, "reset": True})


def test_make_env_dispatches() -> None:
    with pytest.raises(ValueError, match="unknown env_backend"):
        cfg = _cfg()
        object.__setattr__(cfg, "env_backend", "dm")
        E.make_env(cfg, 0)


# ----------------------------------------------------------------------
# Frames on disk, for the by-eye check (MISSION "what verified means")
# ----------------------------------------------------------------------
def test_playtrain_real_frame_png() -> None:
    """Saves a real frostbite frame from the runtime. Skipped if node is missing."""
    cfg = _cfg(env_backend="playtrain")
    try:
        env = E.make_playtrain_dreamer("frostbite", 0, cfg)
    except Exception as exc:  # pragma: no cover - depends on the machine
        pytest.skip(f"PlayTrain runtime unavailable: {exc}")
    try:
        env.seed_episode(1)
        obs = env.step({"action": 0, "reset": True})
        for _ in range(30):
            obs = env.step({"action": 2, "reset": False})
        assert obs["image"].shape == (64, 64, 3)
        assert obs["image"].std() > 1.0  # not a blank screen
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        Image.fromarray(obs["image"]).resize((256, 256), Image.NEAREST).save(
            ARTIFACTS / "playtrain_frostbite_frame.png"
        )
    finally:
        env.close()
