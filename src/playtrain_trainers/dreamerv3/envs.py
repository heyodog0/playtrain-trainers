"""DreamerV3's environment contract, on ALE and on PlayTrain.

The official code drives ``ale_py.ALEInterface`` itself rather than going through
gymnasium, and its preprocessing (max over the last ``pooling`` frames of the repeat,
Pillow BILINEAR to 64x64, raw reward, ``is_terminal = is_last``) differs from
``gymnasium.wrappers.AtariPreprocessing`` in ways that matter. :class:`AtariDreamer`
is therefore a direct port of ``embodied/envs/atari.py``, NOT a reuse of the BBF ALE
stack, which is built on ``AtariPreprocessing`` for Dopamine's protocol (see D-025).

Both backends present the official observation dict::

    image        uint8 (H, W, 3)   HWC, as the official env emits it
    reward       float32           raw, unclipped
    is_first     bool
    is_last      bool
    is_terminal  bool

plus one key the official env does not have:

    info         {"score", "lives", "frames"}

``info`` is our replacement for the official ``log/<name>`` convention (``logfn`` in
``run/train.py`` picks up obs keys starting with ``log/``; ``make_agent`` strips them
before the agent sees the space). The trainer strips ``info`` the same way. Carrying
``frames`` here is what lets every log line print all three clocks (MISSION rule 3).

Actions are the official dict form, ``{"action": int, "reset": bool}``, so the driver
in ``train.py`` can be a transcription of ``embodied/core/driver.py``'s contract.

Deviations implemented here: D-002 (native 64px RGB on PlayTrain, no max-pool),
D-003 (default8), D-004 (lives unused), D-005 (no-op law and where its randomness
comes from), D-006 (episode cap in frames), D-007 (raw reward), D-008
(``is_terminal = is_last``, copied), D-025 (env seeding).
"""

from __future__ import annotations

import collections
import os
import threading
from typing import Any, Protocol

import numpy as np
from PIL import Image

from playtrain_trainers.dreamerv3.config import DreamerConfig

# atari.py: the ALE action table, in index order.
ACTION_MEANING = (
    "NOOP", "FIRE", "UP", "RIGHT", "LEFT", "DOWN", "UPRIGHT", "UPLEFT",
    "DOWNRIGHT", "DOWNLEFT", "UPFIRE", "RIGHTFIRE", "LEFTFIRE", "DOWNFIRE",
    "UPRIGHTFIRE", "UPLEFTFIRE", "DOWNRIGHTFIRE", "DOWNLEFTFIRE",
)

# atari.py: luminance weights. Unused at the frozen config (gray=False), ported so
# the grayscale path is not a silent hole if someone flips env.gray.
WEIGHTS = np.array([0.299, 0.587, 1 - (0.299 + 0.587)])

OBS_KEYS = ("image", "reward", "is_first", "is_last", "is_terminal")


class DreamerEnv(Protocol):
    """What ``train.py`` may assume of either backend."""

    @property
    def obs_space(self) -> dict[str, Any]: ...

    @property
    def act_space(self) -> dict[str, Any]: ...

    def step(self, action: dict[str, Any]) -> dict[str, Any]: ...

    def close(self) -> None: ...


# ----------------------------------------------------------------------
# No-op law (D-005)
# ----------------------------------------------------------------------
def noop_count(
    max_noops: int,
    wrapper_seed: int,
    episode_seed: int | None,
    attempt: int,
    running: np.random.Generator | None = None,
) -> int:
    """How many no-op frames to burn at a reset.

    The law is the official one: ``rng.integers(noops + 1)``, so the count is drawn
    from ``0..max_noops`` INCLUSIVE and zero no-ops is a legal outcome. (Dopamine, and
    so the BBF port, draws ``1..max_noops``; ours follows DreamerV3.)

    Where the randomness comes from is a deviation (D-005, carried over from
    bbf-loop). The official env draws from one running generator, which makes the
    start of episode 15 a function of the 14 episodes before it, so no episode can be
    replayed on its own. We derive the count from ``(wrapper_seed, episode_seed,
    attempt)`` instead whenever the caller supplies an episode seed, which keeps
    starts decorrelated while leaving every episode independently replayable. With no
    episode seed there is nothing to derive from and the running generator is used,
    exactly as in the official code.
    """
    if max_noops < 0:
        raise ValueError("max_noops must be >= 0")
    if episode_seed is None:
        if running is None:
            raise ValueError("a running generator is required when episode_seed is None")
        draw = running
    else:
        draw = np.random.default_rng([int(wrapper_seed), int(episode_seed), int(attempt)])
    return int(draw.integers(max_noops + 1))


# ----------------------------------------------------------------------
# The pixel pipeline, as a pure function so it can be checked by hand
# ----------------------------------------------------------------------
def aggregate_and_resize(
    buffers: list[np.ndarray] | collections.deque,
    size: tuple[int, int],
    aggregate: str = "max",
    resize: str = "pillow",
    gray: bool = False,
) -> np.ndarray:
    """``atari.py`` ``_obs``'s image half, lifted out verbatim.

    ``buffers`` are the last ``pooling`` raw RGB screens, newest first. The order does
    not matter to ``max``/``mean``, but it is kept so the function reads like the
    original.
    """
    if aggregate == "max":
        image = np.amax(buffers, 0)
    elif aggregate == "mean":
        image = np.mean(buffers, 0).astype(np.uint8)
    else:
        raise ValueError(f"unknown aggregate {aggregate!r}")
    if resize == "opencv":
        import cv2

        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    elif resize == "pillow":
        image = Image.fromarray(image)
        image = image.resize(size, Image.BILINEAR)
        image = np.array(image)
    else:
        raise ValueError(f"unknown resize {resize!r}")
    if gray:
        # atari.py's comment: averaging channels equally would map a fully red object
        # on a fully green background to the same value.
        image = (image * WEIGHTS).sum(-1).astype(image.dtype)[:, :, None]
    return image


# ----------------------------------------------------------------------
# ALE
# ----------------------------------------------------------------------
def load_ale(name: str, seed: int):
    """Build a real ``ALEInterface`` with the ROM loaded.

    ``atari.py`` calls ``ale_py.roms.get_rom_path(name)``; that helper moved between
    ale-py releases (the official code pins 0.9.0, the cluster venv has 0.12.1), so
    the lookup is tried in the order: ``$ALE_ROM_PATH`` (which the official code also
    honours and which wins there too), ``ale_py.roms.get_rom_path``, then the packaged
    ``ale_py/roms/<name>.bin``.
    """
    import ale_py

    ale = ale_py.ALEInterface()
    ale.setLoggerMode(ale_py.LoggerMode.Error)
    # D-026: the pinned code passes `b'random_seed'`; ale-py 0.12.1 (the cluster venv)
    # only accepts `str` and raises TypeError on bytes. 0.9.0, which the official
    # `requirements.txt` pins, accepted both. Same setting either way.
    try:
        ale.setInt("random_seed", seed)
    except TypeError:  # pragma: no cover - only on ale-py < 0.10
        ale.setInt(b"random_seed", seed)
    path = os.environ.get("ALE_ROM_PATH", None)
    if path:
        ale.loadROM(os.path.join(path, f"{name}.bin"))
        return ale
    try:
        import ale_py.roms as roms

        ale.loadROM(roms.get_rom_path(name))
        return ale
    except (ImportError, AttributeError):
        pass
    import pathlib

    rom = pathlib.Path(ale_py.__file__).parent / "roms" / f"{name}.bin"
    if not rom.exists():
        raise FileNotFoundError(
            f"no ROM for {name!r}: set ALE_ROM_PATH or install a ale-py that ships roms"
        )
    ale.loadROM(str(rom))
    return ale


class AtariDreamer:
    """Port of ``embodied/envs/atari.py`` (177 lines) at the pinned commit.

    Every control-flow decision below is the official one, including the ones that
    look like bugs; the fidelity record for each is D-015 in DEVIATIONS.md. The one
    structural change is that the ALE handle is injectable, so the whole step / no-op
    / pooling path can be tested without ale-py installed (it is not available on this
    Mac; the real screen is inspected in U07's cluster smoke).
    """

    LOCK = threading.Lock()

    def __init__(
        self,
        name: str,
        repeat: int = 4,
        size: tuple[int, int] = (64, 64),
        gray: bool = False,
        noops: int = 30,
        lives: str = "unused",
        sticky: bool = False,
        actions: str = "needed",
        length: int = 108000,
        pooling: int = 2,
        aggregate: str = "max",
        resize: str = "pillow",
        autostart: bool = False,
        clip_reward: bool = False,
        seed: int | None = None,
        ale: Any = None,
    ):
        assert lives in ("unused", "discount", "reset"), lives
        assert actions in ("all", "needed"), actions
        assert resize in ("opencv", "pillow"), resize
        assert aggregate in ("max", "mean"), aggregate
        assert pooling >= 1, pooling
        assert repeat >= 1, repeat
        if name == "james_bond":
            name = "jamesbond"

        self.repeat = repeat
        self.size = size
        self.gray = gray
        self.noops = noops
        self.lives = lives
        self.sticky = sticky
        self.length = length
        self.pooling = pooling
        self.aggregate = aggregate
        self.resize = resize
        self.autostart = autostart
        self.clip_reward = clip_reward
        # D-025: the official `make_env` never sets `use_seed` for the atari100k
        # suite, so the reference runs build the ALE with `seed=None` and draw the
        # emulator's `random_seed` from OS entropy. We always seed it.
        self.seed = 0 if seed is None else int(seed)
        self.rng = np.random.default_rng(self.seed)

        with self.LOCK:
            self.ale = ale if ale is not None else load_ale(
                name, int(self.rng.integers(0, 2**31))
            )
        self.ale.setFloat("repeat_action_probability", 0.25 if sticky else 0.0)
        self.actionset = {
            "all": self.ale.getLegalActionSet,
            "needed": self.ale.getMinimalActionSet,
        }[actions]()

        # atari.py names these W, H and allocates (W, H, 3); whatever order
        # getScreenDims returns is the order getScreenRGB writes, so the buffers
        # match the screen either way. Copied as is.
        W, H = self.ale.getScreenDims()
        self.buffers = collections.deque(
            [np.zeros((W, H, 3), np.uint8) for _ in range(self.pooling)],
            maxlen=self.pooling,
        )
        self.prevlives = None
        self.duration = None
        self.done = True
        self.episode_seed: int | None = None
        self.last_noops = 0
        self.score = 0.0

    @property
    def obs_space(self) -> dict[str, Any]:
        channels = 1 if self.gray else 3
        return {
            "image": ("uint8", (*self.size, channels)),
            "reward": ("float32", ()),
            "is_first": ("bool", ()),
            "is_last": ("bool", ()),
            "is_terminal": ("bool", ()),
        }

    @property
    def act_space(self) -> dict[str, Any]:
        return {
            "action": ("int32", (), 0, len(self.actionset)),
            "reset": ("bool", ()),
        }

    def seed_episode(self, episode_seed: int | None) -> None:
        """Set the seed the NEXT reset derives its no-op count from (D-005)."""
        self.episode_seed = None if episode_seed is None else int(episode_seed)

    def step(self, action: dict[str, Any]) -> dict[str, Any]:
        if action["reset"] or self.done:
            self._reset()
            self.prevlives = self.ale.lives()
            self.duration = 0
            self.done = False
            self.score = 0.0
            return self._obs(0.0, is_first=True)
        reward = 0.0
        terminal = False
        last = False
        assert 0 <= action["action"] < len(self.actionset), action["action"]
        act = self.actionset[action["action"]]
        for repeat in range(self.repeat):
            reward += self.ale.act(act)
            self.duration += 1
            if repeat >= self.repeat - self.pooling:
                self._render()
            if self.ale.game_over():
                terminal = True
                last = True
            if self.duration >= self.length:
                last = True
            lives = self.ale.lives()
            if self.lives == "discount" and 0 < lives < self.prevlives:
                terminal = True
            if self.lives == "reset" and 0 < lives < self.prevlives:
                terminal = True
                last = True
            self.prevlives = lives
            if terminal or last:
                break
        self.done = last
        self.score += reward
        return self._obs(reward, is_last=last, is_terminal=terminal)

    def _reset(self) -> None:
        with self.LOCK:
            self.ale.reset_game()
        n = noop_count(self.noops, self.seed, self.episode_seed, 0, self.rng)
        self.last_noops = n
        for _ in range(n):
            self.ale.act(ACTION_MEANING.index("NOOP"))
            if self.ale.game_over():
                with self.LOCK:
                    self.ale.reset_game()
        if self.autostart and ACTION_MEANING.index("FIRE") in self.actionset:
            self.ale.act(ACTION_MEANING.index("FIRE"))
            if self.ale.game_over():
                with self.LOCK:
                    self.ale.reset_game()
            self.ale.act(ACTION_MEANING.index("UP"))
            if self.ale.game_over():
                with self.LOCK:
                    self.ale.reset_game()
        self._render()
        # e3f0224, the pinned commit itself ("Fix Atari frame maxpooling on reset"):
        # every pooling buffer starts as a copy of the first rendered frame, so the
        # first observation does not max against a stale screen from the last episode.
        for i, dst in enumerate(self.buffers):
            if i > 0:
                np.copyto(dst, self.buffers[0])

    def _render(self) -> None:
        # Rotate the ring, then fill slot 0: buffers[0] is always the newest frame.
        self.buffers.appendleft(self.buffers.pop())
        self.ale.getScreenRGB(self.buffers[0])

    def _obs(self, reward, is_first=False, is_last=False, is_terminal=False) -> dict[str, Any]:
        if self.clip_reward:
            reward = np.sign(reward)
        image = aggregate_and_resize(
            self.buffers, self.size, self.aggregate, self.resize, self.gray
        )
        return dict(
            image=image,
            reward=np.float32(reward),
            is_first=is_first,
            is_last=is_last,
            # D-008, copied: the official `_obs` ignores its own `is_terminal`
            # argument and reports `is_last`, so a truncation at the 108k-frame cap
            # is handed to the agent as a terminal. Kept, with the argument left in
            # place exactly as upstream has it.
            is_terminal=is_last,
            info={
                "score": float(self.score),
                "lives": int(self.ale.lives()),
                "frames": int(self.duration or 0),
            },
        )

    def close(self) -> None:  # pragma: no cover - ALE has no close
        pass


# ----------------------------------------------------------------------
# PlayTrain
# ----------------------------------------------------------------------
class PlayTrainDreamer:
    """The same contract over ``PlayTrainEnv``.

    The runtime renders natively at 64x64 RGB and applies the frame skip itself
    (summing the score delta over the skipped frames), so the two pieces of ALE
    preprocessing that exist for emulator artefacts are not applied: there is no
    resize (D-002, the native render is already the target size) and no max-pool over
    the last two frames (D-002, a p5 game draws every sprite every frame, so there is
    no flicker to pool away; pooling would only blur motion). The remaining protocol
    -- raw reward, no frame stack, no life terminals, random no-ops, the cap in frames
    -- is the official one.
    """

    def __init__(
        self,
        game: str,
        size: tuple[int, int] = (64, 64),
        gray: bool = False,
        repeat: int = 4,
        noops: int = 30,
        length: int = 108000,
        clip_reward: bool = False,
        seed: int | None = None,
        action_space: str | list | None = None,
        env: Any = None,
    ):
        if gray:
            raise ValueError("the frozen config is RGB (env.gray False); D-002")
        if size[0] != size[1]:
            raise ValueError(f"the runtime renders square frames, got {size}")
        self.size = size
        self.gray = gray
        self.repeat = repeat
        self.noops = noops
        self.length = length
        self.clip_reward = clip_reward
        self.seed = 0 if seed is None else int(seed)
        self.rng = np.random.default_rng(self.seed)

        if env is not None:
            self.env = env
        else:
            from playtrain.runtime import PlayTrainEnv

            self.env = PlayTrainEnv(
                game=game,
                obs_size=size[0],
                obs_mode="rgb",
                frame_stack=1,
                frame_skip=repeat,
                # D-006: the runtime counts FRAMES here, like atari.py's `length`.
                max_steps=length,
                action_space=action_space,
            )
        self.done = True
        self.episode_seed: int | None = None
        self.last_noops = 0
        self.score = 0.0
        self.frames = 0
        self._last_obs: np.ndarray | None = None
        self._lives = 0
        self._raw_score = 0.0
        self._raw_frames = 0
        self._score_offset = 0.0
        self._frame_offset = 0

    @property
    def obs_space(self) -> dict[str, Any]:
        return {
            "image": ("uint8", (*self.size, 1 if self.gray else 3)),
            "reward": ("float32", ()),
            "is_first": ("bool", ()),
            "is_last": ("bool", ()),
            "is_terminal": ("bool", ()),
        }

    @property
    def act_space(self) -> dict[str, Any]:
        return {
            "action": ("int32", (), 0, int(self.env.action_space.n)),
            "reset": ("bool", ()),
        }

    def seed_episode(self, episode_seed: int | None) -> None:
        self.episode_seed = None if episode_seed is None else int(episode_seed)

    def step(self, action: dict[str, Any]) -> dict[str, Any]:
        if action["reset"] or self.done:
            self._reset()
            self.done = False
            return self._obs(0.0, is_first=True)
        assert 0 <= action["action"] < int(self.env.action_space.n), action["action"]
        obs, reward, terminated, truncated, info = self.env.step(int(action["action"]))
        self._absorb(obs, info)
        last = bool(terminated or truncated)
        self.done = last
        # The runtime reports the game's own cumulative score, which covers reward
        # earned during the reset no-ops and cannot drift from a running sum.
        return self._obs(float(reward), is_last=last, is_terminal=bool(terminated))

    def _reset(self) -> None:
        for attempt in range(8):
            # The episode seed is unchanged on a retry -- the same seed must always
            # replay the same episode (D-005) -- but `attempt` enters the no-op
            # derivation, so a retry draws a different count rather than repeating
            # the one that just ended the game.
            obs, info = self.env.reset(seed=self.episode_seed)
            self._absorb(obs, info)
            n = noop_count(self.noops, self.seed, self.episode_seed, attempt, self.rng)
            ended = False
            for _ in range(n):
                obs, _reward, terminated, truncated, info = self.env.step(0)
                self._absorb(obs, info)
                if terminated or truncated:
                    ended = True
                    break
            if not ended:
                self.last_noops = n
                # The official env counts the no-op frames against neither the score
                # nor the episode length; PlayTrain's own counters do include them, so
                # the episode's clocks are rebased here.
                self.score = 0.0
                self._score_offset = float(info.get("score", 0.0))
                self._frame_offset = int(info.get("episodeLength", 0))
                self.frames = 0
                return
        raise RuntimeError(
            f"the episode ended during {self.noops} no-ops on 8 consecutive resets; "
            "the game is not playable under this protocol"
        )

    def _absorb(self, obs: np.ndarray, info: dict[str, Any]) -> None:
        self._last_obs = obs
        self._lives = int(info.get("lives", 0) or 0)
        self._raw_score = float(info.get("score", 0.0))
        self._raw_frames = int(info.get("episodeLength", 0))

    def _obs(self, reward, is_first=False, is_last=False, is_terminal=False) -> dict[str, Any]:
        if self.clip_reward:
            reward = np.sign(reward)
        image = self._last_obs
        assert image is not None
        if image.shape != (*self.size, 3):
            raise ValueError(f"expected {(*self.size, 3)} from the runtime, got {image.shape}")
        if not is_first:
            self.score = self._raw_score - self._score_offset
            self.frames = self._raw_frames - self._frame_offset
        return dict(
            image=np.asarray(image, np.uint8),
            reward=np.float32(reward),
            is_first=is_first,
            is_last=is_last,
            # D-008, copied from atari.py: the cap is reported as a terminal.
            is_terminal=is_last,
            info={
                "score": float(self.score),
                "lives": int(self._lives),
                "frames": int(self.frames),
            },
        )

    def close(self) -> None:
        self.env.close()


# ----------------------------------------------------------------------
# Constructors
# ----------------------------------------------------------------------
def make_ale_dreamer(game: str, seed: int, cfg: DreamerConfig, ale: Any = None) -> AtariDreamer:
    """The PROTOCOL section 5 A gate environment, section 3 exactly."""
    return AtariDreamer(
        game,
        repeat=cfg.env.repeat,
        size=tuple(cfg.env.size),
        gray=cfg.env.gray,
        noops=cfg.env.noops,
        lives=cfg.env.lives,
        sticky=cfg.env.sticky,
        actions=cfg.env.actions,
        length=cfg.env.length,
        pooling=cfg.env.pooling,
        aggregate=cfg.env.aggregate,
        resize=cfg.env.resize,
        autostart=cfg.env.autostart,
        clip_reward=cfg.env.clip_reward,
        seed=seed,
        ale=ale,
    )


def make_playtrain_dreamer(
    game: str, seed: int, cfg: DreamerConfig, env: Any = None
) -> PlayTrainDreamer:
    """The PlayTrain arm, PROTOCOL section 4."""
    return PlayTrainDreamer(
        game,
        size=tuple(cfg.env.size),
        gray=cfg.env.gray,
        repeat=cfg.env.repeat,
        noops=cfg.env.noops,
        length=cfg.env.length,
        clip_reward=cfg.env.clip_reward,
        seed=seed,
        env=env,
    )


def make_env(cfg: DreamerConfig, seed: int) -> DreamerEnv:
    """Dispatch on ``cfg.env_backend``."""
    if cfg.env_backend == "ale":
        return make_ale_dreamer(cfg.game, seed, cfg)
    if cfg.env_backend == "playtrain":
        return make_playtrain_dreamer(cfg.game, seed, cfg)
    raise ValueError(f"unknown env_backend {cfg.env_backend!r}")
