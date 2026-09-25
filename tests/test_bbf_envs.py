"""Tests for the BBF Atari-100k wrapper stack (playtrain_trainers.bbf.envs).

The PlayTrain tests spawn a real Node worker, so they are slower than the rest
of the suite but they are the only way to check the two things most likely to
be silently wrong: that a life-loss terminal does NOT restart the game, and
that the score the report will quote is the game's own score.
"""
from __future__ import annotations

import numpy as np
import pytest

from playtrain_trainers.bbf.config import BBFConfig
from playtrain_trainers.bbf.envs import (
    ChannelFirstWrapper,
    ClipRewardWrapper,
    EpisodicLifeWrapper,
    NoopResetWrapper,
    _distinct_action_meanings,
    make_playtrain_atari100k,
)

pytest.importorskip("gymnasium")
import gymnasium as gym  # noqa: E402


# ----------------------------------------------------------------------
# Wrapper unit tests, on a fake env (no Node worker)
# ----------------------------------------------------------------------
class FakeGame(gym.Env):
    """A scriptable stand-in for PlayTrainEnv.

    ``script`` is a list of ``(reward, lives, terminated, truncated)`` applied
    one per step. ``score`` accumulates the rewards, as the real game's does.
    """

    def __init__(self, script, obs_hwc=(84, 84, 4), n_actions=8):
        self.script = list(script)
        self.observation_space = gym.spaces.Box(0, 255, obs_hwc, np.uint8)
        self.action_space = gym.spaces.Discrete(n_actions)
        self.i = 0
        self.resets = 0
        self.actions: list[int] = []
        self.score = 0.0
        self.lives = 3

    def reset(self, *, seed=None, options=None):
        self.resets += 1
        self.i = 0
        self.score = 0.0
        self.lives = 3
        return self._obs(), {"score": self.score, "lives": self.lives, "gameState": "PLAYING"}

    def _obs(self):
        return np.full(self.observation_space.shape, self.i, dtype=np.uint8)

    def step(self, action):
        self.actions.append(int(action))
        reward, lives, term, trunc = self.script[min(self.i, len(self.script) - 1)]
        self.i += 1
        self.score += reward
        self.lives = lives
        info = {"score": self.score, "lives": lives, "gameState": "PLAYING"}
        return self._obs(), float(reward), term, trunc, info


def test_clip_reward_keeps_raw():
    env = ClipRewardWrapper(FakeGame([(10.0, 3, False, False), (100.0, 3, False, False)]), 1.0)
    env.reset()
    _, r, _, _, info = env.step(0)
    assert r == 1.0 and info["raw_reward"] == 10.0
    _, r, _, _, info = env.step(0)
    assert r == 1.0 and info["raw_reward"] == 100.0


def test_clip_reward_clips_negatives_and_passes_small():
    env = ClipRewardWrapper(FakeGame([(-5.0, 3, False, False), (0.5, 3, False, False)]), 1.0)
    env.reset()
    assert env.step(0)[1] == -1.0
    assert env.step(0)[1] == 0.5


def test_clip_must_be_positive():
    with pytest.raises(ValueError):
        ClipRewardWrapper(FakeGame([]), 0.0)


def test_channel_first_shape_and_order():
    inner = FakeGame([], obs_hwc=(84, 84, 4))
    env = ChannelFirstWrapper(inner)
    assert env.observation_space.shape == (4, 84, 84)
    obs, _ = env.reset()
    assert obs.shape == (4, 84, 84) and obs.dtype == np.uint8
    assert obs.flags["C_CONTIGUOUS"]
    # Frame order must survive the transpose: plane k of CHW is channel k of HWC.
    hwc = np.stack([np.full((84, 84), k, np.uint8) for k in range(4)], axis=-1)
    assert np.array_equal(ChannelFirstWrapper(inner).observation(hwc)[2], np.full((84, 84), 2))


def test_channel_first_handles_rgb_stack():
    env = ChannelFirstWrapper(FakeGame([], obs_hwc=(64, 64, 12)))
    assert env.observation_space.shape == (12, 64, 64)


# --- terminal on life loss (D-004): the one that must not restart ----------
def _life_loss_env():
    script = [
        (10.0, 3, False, False),   # step 1: scores, 3 lives
        (0.0, 2, False, False),    # step 2: LIFE LOST
        (10.0, 2, False, False),   # step 3: still playing, same game
        (0.0, 0, True, False),     # step 4: game over
    ]
    inner = FakeGame(script)
    return inner, EpisodicLifeWrapper(ClipRewardWrapper(inner, 1.0))


def test_life_loss_reports_terminal():
    inner, env = _life_loss_env()
    env.reset()
    _, _, term, _, info = env.step(0)
    assert term is False and info["real_done"] is False
    _, _, term, trunc, info = env.step(0)
    assert term is True, "a life loss must be reported as terminated"
    assert trunc is False
    assert info["real_done"] is False, "the GAME is not over, only the life"


def test_life_loss_reset_does_not_restart_the_game():
    inner, env = _life_loss_env()
    env.reset()
    assert inner.resets == 1
    env.step(0)
    env.step(0)  # life lost -> terminated
    obs, info = env.reset()
    assert inner.resets == 1, "reset after a life loss must NOT restart the game"
    assert info["episode_score"] == 10.0, "the game's score carries across the life"
    # and the game continues from where it was
    _, _, _, _, info = env.step(0)
    assert info["episode_score"] == 20.0


def test_real_done_triggers_a_real_reset():
    inner, env = _life_loss_env()
    env.reset()
    for _ in range(4):
        _, _, term, trunc, info = env.step(0)
    assert info["real_done"] is True
    env.reset()
    assert inner.resets == 2, "after the game really ends, reset must restart it"


def test_episode_score_tracks_the_games_own_score():
    inner, env = _life_loss_env()
    env.reset()
    scores = [env.step(0)[4]["episode_score"] for _ in range(4)]
    assert scores == [10.0, 10.0, 20.0, 20.0]
    # Not the clipped reward sum, which would be 1.0 / 1.0 / 2.0 / 2.0.
    assert scores[-1] == inner.score


def test_life_loss_at_game_over_is_real_done_not_a_life_terminal():
    # Lives hit 0 and the game ends on the same step; real_done must win.
    inner = FakeGame([(0.0, 0, True, False)])
    env = EpisodicLifeWrapper(ClipRewardWrapper(inner, 1.0))
    env.reset()
    _, _, term, _, info = env.step(0)
    assert term is True and info["real_done"] is True
    env.reset()
    assert inner.resets == 2


# --- no-ops (D-005) --------------------------------------------------------
def test_noop_reset_takes_between_one_and_max_noops():
    inner = FakeGame([(0.0, 3, False, False)] * 200)
    env = NoopResetWrapper(inner, max_noops=30)
    env.rng = np.random.default_rng(0)
    counts = set()
    for _ in range(40):
        inner.actions.clear()
        env.reset()
        assert all(a == 0 for a in inner.actions), "no-ops must use action 0"
        counts.add(len(inner.actions))
    assert counts, "no no-ops were taken"
    assert min(counts) >= 1 and max(counts) <= 30
    assert len(counts) > 1, "the no-op count must actually vary"


def test_noop_reset_is_reproducible_for_a_seed():
    def run(seed):
        inner = FakeGame([(0.0, 3, False, False)] * 200)
        env = NoopResetWrapper(inner, max_noops=30)
        env.rng = np.random.default_rng(seed)
        env.reset()
        return len(inner.actions)

    assert run(7) == run(7)


def test_zero_noops_is_allowed_and_steps_nothing():
    inner = FakeGame([(0.0, 3, False, False)] * 10)
    NoopResetWrapper(inner, max_noops=0).reset()
    assert inner.actions == []


def test_noop_reset_retries_if_the_episode_ends():
    # Every step ends the episode, so no number of no-ops can survive.
    inner = FakeGame([(0.0, 3, True, False)] * 50)
    env = NoopResetWrapper(inner, max_noops=30)
    env.rng = np.random.default_rng(0)
    with pytest.raises(RuntimeError, match="not playable"):
        env.reset()


def test_noop_reset_rejects_negative():
    with pytest.raises(ValueError):
        NoopResetWrapper(FakeGame([]), -1)


# --- D-003 bookkeeping -----------------------------------------------------
def test_distinct_action_meanings_collapses_space_actions():
    meanings = ["NOOP", "LEFT", "RIGHT", "UP", "DOWN", "D", "LEFT_D", "RIGHT_D"]
    # A game that never reads SPACE: the _D actions are duplicates.
    assert _distinct_action_meanings(meanings, "keyIsDown(37)") == [
        "NOOP", "LEFT", "RIGHT", "UP", "DOWN"
    ]
    # A game that does read it: all 8 are distinct.
    assert len(_distinct_action_meanings(meanings, "keyIsDown(32)")) == 8


# ----------------------------------------------------------------------
# Real PlayTrain env (spawns a Node worker)
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def pt_cfg():
    return BBFConfig()


def _try_make(cfg, seed=0, **kw):
    try:
        return make_playtrain_atari100k(cfg, seed=seed, **kw)
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"PlayTrain runtime unavailable: {exc}")


def test_playtrain_obs_shape_and_dtype(pt_cfg):
    env = _try_make(pt_cfg)
    try:
        assert env.observation_space.shape == (4, 84, 84)
        obs, info = env.reset(seed=0)
        assert obs.shape == (4, 84, 84)
        assert obs.dtype == np.uint8
        # The game draws flat-filled rectangles, so a grayscale frame has only
        # a handful of distinct values -- that is the game, not a bug.
        assert 2 <= len(np.unique(obs[-1])) <= 32
        assert env.action_space.n == 8  # D-003
        assert info["lives"] == 3
    finally:
        env.close()


def test_playtrain_episode_cap_is_in_frames(pt_cfg):
    # D-012: max_steps on the runtime counts frames, so the cap we ask for is
    # 27000 agent steps * 4.
    env = _try_make(pt_cfg)
    try:
        inner = env
        while isinstance(inner, gym.Wrapper):
            inner = inner.env
        assert inner.max_steps == 108_000
        assert inner.frame_skip == 4
        assert inner.obs_size == 84
        assert inner.obs_mode == "grayscale"
    finally:
        env.close()


def test_playtrain_score_in_info_matches_the_game(pt_cfg):
    env = _try_make(pt_cfg)
    try:
        env.reset(seed=3)
        rng = np.random.default_rng(3)
        raw_sum = 0.0
        for _ in range(400):
            _, r, term, trunc, info = env.step(int(rng.integers(8)))
            raw_sum += info["raw_reward"]
            assert abs(r) <= 1.0, "the learning reward must be clipped (D-011)"
            # info["score"] is the game's own counter; it must equal the sum of
            # the raw score deltas the env reported.
            assert info["score"] == pytest.approx(raw_sum)
            assert info["episode_score"] == pytest.approx(info["score"])
            if term or trunc:
                break
        # Scores only arrive in +10 / +100 units (D-014).
        assert raw_sum % 10 == 0
    finally:
        env.close()


def test_playtrain_life_loss_fires_and_keeps_the_game(pt_cfg):
    """The real check behind D-004, on the real game."""
    env = _try_make(pt_cfg, training=True)
    try:
        env.reset(seed=1)
        rng = np.random.default_rng(1)
        saw_life_terminal = False
        for _ in range(3000):
            _, _, term, trunc, info = env.step(int(rng.integers(8)))
            if term and not info["real_done"]:
                saw_life_terminal = True
                lives_at_terminal = info["lives"]
                score_at_terminal = info["score"]
                # Continuing must not restart the game: lives and score persist.
                _, cont_info = env.reset()
                assert cont_info["lives"] == lives_at_terminal
                assert cont_info["episode_score"] == score_at_terminal
                assert cont_info["gameState"] == "PLAYING"
                break
            if info["real_done"]:
                env.reset(seed=int(rng.integers(10_000)))
        assert saw_life_terminal, "a random policy drowns; the life terminal must fire"
    finally:
        env.close()


def test_playtrain_eval_mode_has_no_life_terminals(pt_cfg):
    """D-016: eval episodes are whole games, so lives are not terminals."""
    env = _try_make(pt_cfg, training=False)
    try:
        env.reset(seed=2)
        rng = np.random.default_rng(2)
        lives_seen = set()
        for _ in range(3000):
            _, _, term, trunc, info = env.step(int(rng.integers(8)))
            lives_seen.add(info["lives"])
            if term or trunc:
                assert info["real_done"] is True
                assert info["gameState"] in ("WIN", "GAMEOVER")
                break
            assert term is False
        assert len(lives_seen) > 1, "the episode should have spanned a life loss"
    finally:
        env.close()


def test_playtrain_noops_vary_the_start_state(pt_cfg):
    """D-005 on the real game: the same game seed gives different starts."""
    starts = []
    for wrapper_seed in (0, 1, 2, 3, 4, 5):
        env = _try_make(pt_cfg, seed=wrapper_seed)
        try:
            obs, _ = env.reset(seed=99)  # SAME game seed every time
            starts.append(obs[-1].copy())
        finally:
            env.close()
    assert any(not np.array_equal(starts[0], s) for s in starts[1:]), (
        "no-ops did not decorrelate the start state"
    )


def test_playtrain_frame_written_for_inspection(pt_cfg, tmp_path):
    """Save the 4-frame stack as a PNG strip so D-002 can be checked by eye."""
    png = pytest.importorskip("PIL.Image")
    env = _try_make(pt_cfg)
    try:
        env.reset(seed=0)
        rng = np.random.default_rng(0)
        for _ in range(12):
            obs, _, term, trunc, _ = env.step(int(rng.integers(8)))
            if term or trunc:
                break
    finally:
        env.close()
    strip = np.concatenate(list(obs), axis=1)  # 4 frames side by side
    assert strip.shape == (84, 84 * 4)
    out = tmp_path / "stack.png"
    png.fromarray(strip).save(out)
    assert out.stat().st_size > 0


def test_playtrain_84px_is_rendered_not_upscaled(pt_cfg):
    """D-002: the runtime renders at 84 px; it does not resize a 64 px frame.

    Both sizes are rendered from the same game seed with no-ops disabled, so
    the two frames show the same game state. If 84 px were a resize of 64 px,
    upscaling the 64 px frame would reproduce the 84 px one closely. It does
    not: the 84 px render puts edges where a resample cannot.
    """
    from playtrain_trainers.bbf.config import BBFConfig as C

    Image = pytest.importorskip("PIL.Image")
    frames = {}
    for size in (64, 84):
        cfg = C(obs_size=size, max_noops=0, frame_stack=1)
        env = _try_make(cfg)
        try:
            obs, _ = env.reset(seed=11)
            frames[size] = obs[0].copy()
        finally:
            env.close()
    assert frames[64].shape == (64, 64) and frames[84].shape == (84, 84)
    for resample in (Image.NEAREST, Image.BILINEAR):
        up = np.asarray(
            Image.fromarray(frames[64]).resize((84, 84), resample=resample)
        )
        mad = float(np.abs(up.astype(np.int16) - frames[84].astype(np.int16)).mean())
        assert mad > 1.0, (
            f"84 px frame is within {mad:.2f} mean abs of a 64->84 "
            f"{resample} upscale -- it may be a resize after all"
        )


# ----------------------------------------------------------------------
# Per-episode reproducibility (the property that makes a replay meaningful)
# ----------------------------------------------------------------------
def test_noop_count_is_a_function_of_the_episode_seed_not_the_history():
    """A shared running generator makes episode N depend on the N-1 before it.

    That is what stopped a scored eval episode from replaying to the same
    score, so the count is derived from (wrapper seed, game seed) instead.
    """
    def counts(order):
        inner = FakeGame([(0.0, 3, False, False)] * 200)
        env = NoopResetWrapper(inner, max_noops=30, seed=4)
        got = {}
        for s in order:
            inner.actions.clear()
            env.reset(seed=s)
            got[s] = len(inner.actions)
        return got

    forward = counts([100, 200, 300])
    # Same seeds, different order: each seed must get the same count.
    assert counts([300, 100, 200]) == forward
    # And visiting one seed alone must match too.
    assert counts([200])[200] == forward[200]


def test_noop_count_still_varies_across_wrapper_seeds():
    # D-005: a repeated game seed must not give a repeated start across runs.
    def count(wrapper_seed):
        inner = FakeGame([(0.0, 3, False, False)] * 200)
        NoopResetWrapper(inner, max_noops=30, seed=wrapper_seed).reset(seed=99)
        return len(inner.actions)

    assert len({count(w) for w in range(12)}) > 1


def test_noop_count_varies_across_episode_seeds():
    def count(game_seed):
        inner = FakeGame([(0.0, 3, False, False)] * 200)
        NoopResetWrapper(inner, max_noops=30, seed=0).reset(seed=game_seed)
        return len(inner.actions)

    assert len({count(s) for s in range(12)}) > 1


def test_noop_retry_draws_a_different_count(pt_cfg):
    # A deterministic derivation must still vary per attempt, or a retry would
    # repeat the count that just failed and loop until it gave up.
    inner = FakeGame([(0.0, 3, True, False)] * 50)
    env = NoopResetWrapper(inner, max_noops=30, seed=0)
    with pytest.raises(RuntimeError, match="not playable"):
        env.reset(seed=7)


def test_playtrain_episode_replays_bit_exact(pt_cfg):
    """The whole stack, twice, on the same seed: identical trajectory."""
    from playtrain_trainers.bbf.evaluate import policy_rng, random_policy, run_episode

    def play():
        env = _try_make(pt_cfg, seed=pt_cfg.seed, training=False)
        try:
            pol = random_policy(env.action_space.n, policy_rng(pt_cfg, 8_000_013))
            return run_episode(env, pol, 8_000_013, 27_000)
        finally:
            env.close()

    a, b = play(), play()
    assert (a.score, a.agent_steps, a.game_state, a.lives_left) == (
        b.score, b.agent_steps, b.game_state, b.lives_left
    )
