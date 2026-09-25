"""Atari-100k wrapper stack for BBF, over PlayTrain games and over ALE.

Both backends are wrapped to the same contract so the agent code never branches
on which one it is talking to:

  observation  uint8, CHW, ``(frame_stack * channels, obs_size, obs_size)``
  action       Discrete
  reward       clipped to ``[-reward_clip, reward_clip]`` for learning
  info         ``score``          raw cumulative game score
               ``raw_reward``     the unclipped reward for this step
               ``episode_score``  raw score of the GAME episode so far
               ``lives``
               ``real_done``      True when the GAME episode ended, as opposed
                                  to a life-loss terminal
               ``gameState``      PLAYING | WIN | GAMEOVER | EXIT (PlayTrain)

The PlayTrain runtime already implements frame skip (summing the score delta
over the skipped frames) and frame stacking, so those two are configured on
``PlayTrainEnv`` rather than re-implemented here. What this module adds is the
Dopamine-side protocol: random no-ops at reset, terminal-on-life-loss that does
not restart the game, reward clipping that preserves the raw score, and the
channel-first layout torch wants.

Deviations this module implements: D-001 (agent step = 4 game frames),
D-002 (84x84 grayscale asked of the runtime, no resize, no max-pool),
D-003 (8-action set), D-004 (life loss from info["lives"]), D-005 (no-ops),
D-011 (reward clip), D-012 (episode cap in frames), D-013 (PlayTrainEnv),
D-016 (eval episodes are whole games, not lives).
"""
from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from playtrain_trainers.bbf.config import BBFConfig

# frostbite.js: the p5 game reads only the arrow keys, so every action that
# merely presses SPACE (key 32) is a duplicate of the one that holds the same
# arrow. Reported by `distinct_action_meanings` for the record (D-003).
_SPACE_ONLY_KEY = 32


def _episode_score(info: dict[str, Any], running: float, reward: float) -> float:
    """Raw score of the game episode so far.

    PlayTrain reports the game's own cumulative ``score`` every step, which is
    exactly the quantity the report needs and cannot drift from accumulating
    per-step rewards (it also covers reward earned during the reset no-ops).
    The ALE arm has no such key, so there the unclipped rewards are summed.
    """
    if "score" in info:
        return float(info["score"])
    return running + float(info.get("raw_reward", reward))


class NoopResetWrapper(gym.Wrapper):
    """Take 1..``max_noops`` no-op steps after each real reset (D-005).

    Dopamine's runner does this to decorrelate the start states of episodes
    that would otherwise be identical under a deterministic game. The no-op
    steps are part of the reset, not of the agent's step budget.

    If the game somehow ends during the no-ops the reset is retried, so the
    caller always gets a live episode back.

    The no-op count is drawn from this wrapper's OWN generator, not from
    ``self.np_random``. A gymnasium wrapper's ``np_random`` delegates down to
    the base env, and ``PlayTrainEnv.reset`` reseeds that from the game seed on
    every reset -- so drawing from it would make the no-op count a function of
    the game seed alone, and two episodes replaying the same seed would get
    the same start. Dopamine's runner likewise keeps its own RNG for this.

    When ``reset`` is given a game seed, the count is derived from BOTH that
    seed and this wrapper's seed rather than from a running generator. A single
    running generator makes the start of episode 15 depend on the 14 episodes
    before it, so no episode can be replayed on its own -- which is how a
    replay of a scored eval episode came out with a different score. Folding
    the wrapper seed in keeps starts decorrelated across runs, which is what
    D-005 is for. With no game seed (a bare ``reset()``) the running generator
    is used, since there is nothing to derive from.
    """

    def __init__(
        self,
        env: gym.Env,
        max_noops: int,
        noop_action: int = 0,
        seed: int | None = None,
    ):
        super().__init__(env)
        if max_noops < 0:
            raise ValueError("max_noops must be >= 0")
        self.max_noops = max_noops
        self.noop_action = noop_action
        self.seed = 0 if seed is None else int(seed)
        self.rng = np.random.default_rng(seed)
        self.last_noops = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        if self.max_noops == 0:
            self.last_noops = 0
            return obs, info
        for attempt in range(8):
            # integers(1, n+1): at least one no-op, as in ALE's NoopResetEnv.
            # `attempt` enters the derivation so a retry draws a different
            # count instead of repeating the one that just failed.
            draw = (
                np.random.default_rng([self.seed, seed, attempt])
                if seed is not None
                else self.rng
            )
            n = int(draw.integers(1, self.max_noops + 1))
            obs, info = self.env.reset(seed=seed if attempt == 0 else None, options=options)
            done = False
            for _ in range(n):
                obs, _, terminated, truncated, info = self.env.step(self.noop_action)
                if terminated or truncated:
                    done = True
                    break
            if not done:
                self.last_noops = n
                return obs, info
        raise RuntimeError(
            f"episode ended during {self.max_noops} no-ops on 8 consecutive resets; "
            "the game is not playable under this protocol"
        )


class EpisodicLifeWrapper(gym.Wrapper):
    """Report a life loss as a terminal without restarting the game (D-004).

    ``info['lives']`` comes straight from the game. When it drops, ``step``
    returns ``terminated=True`` while the underlying game keeps running; the
    next ``reset`` continues that same game instead of starting a new one.
    This is Dopamine/ALE ``EpisodicLifeEnv``, and it is what
    ``AtariPreprocessing.terminal_on_life_loss = True`` means in BBF.gin.

    PlayTrain frostbite respawns the player within the same frame that
    decrements ``lives``, so no extra step is needed to advance past the
    terminal. On the ALE (``info["lives"]`` from ale-py) the game likewise
    just continues; Dopamine's agent sees the terminal and carries on from
    the current screen, which is what handing back ``_last_obs`` does.

    ``info['real_done']`` distinguishes the game ending from a life ending, and
    ``info['episode_score']`` accumulates over the whole game, so an evaluator
    can score whole games while training sees life-episodes.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._lives = 0
        self._real_done = True
        self._last_obs: np.ndarray | None = None
        self._last_info: dict[str, Any] = {}
        self._episode_score = 0.0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if self._real_done or self._last_obs is None:
            obs, info = self.env.reset(seed=seed, options=options)
            # A real reset starts a new game, so the score baseline is the
            # score the game reports right after its no-ops.
            self._episode_score = float(info.get("score", 0.0))
        else:
            # Mid-game: hand back the observation the life-loss terminal was
            # reported with. Passing the reset through would restart the game
            # and throw away the lives the agent still has.
            obs, info = self._last_obs, dict(self._last_info)
        self._lives = int(info.get("lives", 0))
        self._real_done = False
        info = dict(info)
        info["real_done"] = False
        info["episode_score"] = self._episode_score
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._episode_score = _episode_score(info, self._episode_score, reward)
        self._real_done = bool(terminated or truncated)
        lives = int(info.get("lives", self._lives))
        info = dict(info)
        info["real_done"] = self._real_done
        info["episode_score"] = self._episode_score
        if not self._real_done and lives < self._lives:
            terminated = True
        self._lives = lives
        self._last_obs, self._last_info = obs, info
        return obs, reward, terminated, truncated, info


class ClipRewardWrapper(gym.Wrapper):
    """Clip the learning reward, keep the raw one in info (D-011).

    The score deltas are +10 per floe and +100 for the igloo, so without this
    the C51 support would have to span two orders of magnitude for one game.
    ``info['raw_reward']`` and ``info['score']`` keep the reportable numbers.
    """

    def __init__(self, env: gym.Env, clip: float):
        super().__init__(env)
        if clip <= 0:
            raise ValueError("reward clip must be > 0")
        self.clip = float(clip)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info["raw_reward"] = float(reward)
        return obs, float(np.clip(reward, -self.clip, self.clip)), terminated, truncated, info


class ChannelFirstWrapper(gym.ObservationWrapper):
    """HWC uint8 from the runtime -> CHW uint8 for torch.

    The runtime stacks grayscale frames on the last axis ((84,84,4)) and RGB
    frames by concatenating channels ((84,84,12)); both become CHW here, so a
    grayscale stack is (4,84,84) and an RGB stack (12,84,84) with the frames in
    the same oldest-to-newest order.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        h, w, c = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(c, h, w), dtype=np.uint8
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        # ascontiguousarray: the transpose is a view, and the replay buffer
        # stores these, so a contiguous copy is what callers expect.
        return np.ascontiguousarray(observation.transpose(2, 0, 1))


def _distinct_action_meanings(meanings: list[str], game_source: str) -> list[str]:
    """Which of the runtime's actions the game can actually tell apart.

    frostbite.js reads ``keyIsDown(37/38/39/40)`` only, so the SPACE-pressing
    actions collapse onto their arrow-only twins. Used for the record in
    D-003, not to change the action space -- the protocol runs the game's
    declared space.
    """
    reads_space = f"keyIsDown({_SPACE_ONLY_KEY})" in game_source or "keyPressed" in game_source
    if reads_space:
        return list(meanings)
    seen: list[str] = []
    for name in meanings:
        base = name.replace("_D", "") or "NOOP"
        base = "NOOP" if base == "D" else base
        if base not in seen:
            seen.append(base)
    return seen


def make_playtrain_atari100k(
    cfg: BBFConfig,
    seed: int,
    *,
    training: bool = True,
    max_steps_frames: int | None = None,
    channel_first: bool = True,
) -> gym.Env:
    """Build the PlayTrain env with the Atari-100k protocol wrapped around it.

    Parameters
    ----------
    cfg:
        Supplies the frozen protocol values; nothing is tuned here.
    seed:
        Seeds the wrapper RNG (the no-op count). The game seed is passed to
        ``reset(seed=...)`` by the caller, which is what varies episodes.
    training:
        True applies terminal-on-life-loss (D-004). False leaves whole games
        as the episode, which is the unit the reported eval score is in
        (D-016).
    channel_first:
        True (the default) returns CHW, which is what BBF's torch code wants.
        False stops before that wrapper and returns the runtime's native HWC.
        The PPO and IMPALA reference trainers size their networks from an HWC
        observation space and do their own transpose, so U11 asks for HWC --
        that way the reference arms run the SAME wrapper stack (D-026) rather
        than an approximation of it.
    """
    from playtrain.runtime import PlayTrainEnv

    if cfg.env_backend != "playtrain":
        raise ValueError(f"cfg.env_backend is {cfg.env_backend!r}, not 'playtrain'")
    if cfg.max_pool_frames:
        # D-002: the p5 renderer draws every sprite every frame, so there is no
        # flicker to pool away. Pooling here would just blur motion.
        raise ValueError("max_pool_frames is an ALE-only correction; see D-002")

    env: gym.Env = PlayTrainEnv(
        game=cfg.game,
        obs_size=cfg.obs_size,
        obs_mode=cfg.obs_mode,
        frame_stack=cfg.frame_stack,
        frame_skip=cfg.frame_skip,
        # D-012: the runtime counts FRAMES here, not agent steps.
        max_steps=cfg.max_steps_frames if max_steps_frames is None else max_steps_frames,
        action_space=cfg.action_space,
    )
    # Order follows the standard ALE stack: no-ops belong to the real game
    # reset, so NoopResetWrapper sits INSIDE the life-loss wrapper. Were it
    # outside, every life-loss "reset" would also burn 1..30 no-op steps in
    # the middle of a live game. ClipRewardWrapper is innermost so every
    # wrapper above it can read info["raw_reward"].
    env = ClipRewardWrapper(env, cfg.reward_clip)
    env = NoopResetWrapper(env, cfg.max_noops, seed=seed)
    if training and cfg.terminal_on_life_loss:
        env = EpisodicLifeWrapper(env)
    else:
        env = _RealDoneWrapper(env)
    if channel_first:
        env = ChannelFirstWrapper(env)
    # The PPO trainer's make_env notes why a factory must not call reset(): a
    # vec backend resets every env itself, and a reset here would burn the
    # first episode's seed. The only wrapper RNG is the no-op one, seeded above.
    return env


class _RealDoneWrapper(gym.Wrapper):
    """Add the same info keys as EpisodicLifeWrapper, without life terminals.

    Eval runs score whole games, so they skip the life-loss terminal but still
    need ``episode_score`` and ``real_done`` to be present and to mean the same
    thing (D-016).
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._episode_score = 0.0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._episode_score = float(info.get("score", 0.0))
        info = dict(info)
        info["real_done"] = False
        info["episode_score"] = self._episode_score
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._episode_score = _episode_score(info, self._episode_score, reward)
        info = dict(info)
        info["real_done"] = bool(terminated or truncated)
        info["episode_score"] = self._episode_score
        return obs, reward, terminated, truncated, info


def make_ale_atari100k(
    cfg: BBFConfig,
    seed: int,
    *,
    training: bool = True,
) -> gym.Env:
    """The section 5A sanity arm: the same protocol on real ALE Frostbite.

    Uses ``gymnasium``'s ``AtariPreprocessing``, which is the same
    preprocessing Dopamine applies (grayscale, 84x84 resize, frame skip with a
    max-pool over the last two frames, optional life-loss terminal), plus
    ``FrameStackObservation`` for the 4-stack. ale-py is an optional dependency
    (flag F-002); the ImportError names what to install.
    """
    if cfg.env_backend != "ale":
        raise ValueError(f"cfg.env_backend is {cfg.env_backend!r}, not 'ale'")
    try:
        import ale_py  # noqa: F401
        from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "the ALE sanity arm needs ale-py: `uv add --optional ale \"gymnasium[atari]\"`. "
            "See flag F-002 in the bbf-loop STATE.md for the fallback."
        ) from exc

    gym.register_envs(ale_py)
    # frameskip=1 on the base env: AtariPreprocessing owns the skip.
    base = gym.make(cfg.game, frameskip=1, repeat_action_probability=0.0)
    # D-038: `terminal_on_life_loss` is handled by EpisodicLifeWrapper, NOT
    # by AtariPreprocessing. Gymnasium's flag only marks the step terminated;
    # its `reset()` then does a full ALE reset, so training would restart the
    # game at every life loss and never see a second-life state. Dopamine
    # continues the same game, which is what the wrapper below does.
    env: gym.Env = AtariPreprocessing(
        base,
        noop_max=cfg.max_noops,
        frame_skip=cfg.frame_skip,
        screen_size=cfg.obs_size,
        terminal_on_life_loss=False,
        grayscale_obs=cfg.obs_mode == "grayscale",
        scale_obs=False,
    )
    env = FrameStackObservation(env, cfg.frame_stack)
    env = ClipRewardWrapper(env, cfg.reward_clip)
    if training and cfg.terminal_on_life_loss:
        env = EpisodicLifeWrapper(env)
    else:
        env = _RealDoneWrapper(env)
    return env


def make_env(cfg: BBFConfig, seed: int, *, training: bool = True) -> gym.Env:
    """Dispatch on ``cfg.env_backend``."""
    if cfg.env_backend == "playtrain":
        return make_playtrain_atari100k(cfg, seed, training=training)
    if cfg.env_backend == "ale":
        return make_ale_atari100k(cfg, seed, training=training)
    raise ValueError(f"unknown env_backend {cfg.env_backend!r}")
