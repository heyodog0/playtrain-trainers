"""Single-env adapter emitting time-major dicts (port of torchbeast/core/environment.py).

Differences from upstream:
  - Gymnasium API (reset returns (obs, info), step returns 5-tuple) instead
    of legacy gym. `done = terminated or truncated`.
  - Optional HWC→CHW transpose for image obs (our minigrid_env emits HWC;
    the IMPALA-CNN expects CHW). Toggle via `chw_transpose` ctor arg.

Output dict shape contract (matches monobeast.Environment so torchbeast's
Net/learn code is drop-in):

    frame:          [1, 1, *obs_shape]   (T=1, B=1 leading dims)
    reward:         [1, 1]   float32
    done:           [1, 1]   bool
    episode_return: [1, 1]   float32  (running, reset to 0 after done)
    episode_step:   [1, 1]   int32    (running, reset to 0 after done)
    last_action:    [1, 1]   int64
"""
from __future__ import annotations

import numpy as np
import torch


def _format_frame(frame: np.ndarray, chw_transpose: bool) -> torch.Tensor:
    """np HWC uint8 -> torch [1, 1, C, H, W] uint8 (or [1, 1, *shape] for non-image).

    node-gym returns frames backed by a read-only mmap; calling
    torch.from_numpy on a non-writable array prints a UserWarning ('not
    writable, undefined behavior on writes'). We never write into this
    tensor — only copy_() into the shared-mem buffer downstream — but
    PyTorch can't see that. np.array(..., copy=True) detaches from the
    mmap so the warning goes away. ~12KB copy/step, negligible vs the
    full forward+env step cost.
    """
    arr = np.array(frame, copy=True)
    t = torch.from_numpy(arr)
    if chw_transpose and t.ndim == 3:
        t = t.permute(2, 0, 1).contiguous()  # HWC -> CHW
    return t.view((1, 1) + tuple(t.shape))


class Environment:
    """Wraps a single gymnasium env. Auto-resets on done; the post-reset
    frame appears at the same timestep as `done=True` (matches torchbeast)."""

    def __init__(self, gym_env, chw_transpose: bool = True,
                 initial_seed: int | None = None,
                 fixed_seed: int | None = None):
        self.gym_env = gym_env
        self.chw_transpose = chw_transpose
        # Two distinct seeding modes:
        #
        # fixed_seed:
        #   If set, EVERY reset (initial + auto-reset on done) uses this
        #   exact seed. Mirrors PPO's fixed_env_seed — memorize-one-instance
        #   training. This is what makes the IMPALA-vs-PPO comparison fair
        #   (otherwise IMPALA trains on a procedural distribution while
        #   PPO trains on a single instance).
        #
        # initial_seed:
        #   Used ONLY for the first reset. Subsequent resets either advance
        #   the env's seeded RNG (gymnasium) or pick a fresh random seed
        #   (node_gym, where seed=None means JS-side randomSeed). Used
        #   when you want each actor to start from a deterministic point
        #   but explore the procedural distribution thereafter.
        #
        # If both are set, fixed_seed wins.
        self._fixed_seed = fixed_seed
        self._initial_seed = initial_seed
        self.episode_return: torch.Tensor | None = None
        self.episode_step: torch.Tensor | None = None

    def _reset_seed(self, *, is_initial: bool) -> int | None:
        if self._fixed_seed is not None:
            return self._fixed_seed
        return self._initial_seed if is_initial else None

    def initial(self) -> dict:
        obs, _info = self.gym_env.reset(seed=self._reset_seed(is_initial=True))
        self.episode_return = torch.zeros(1, 1)
        self.episode_step = torch.zeros(1, 1, dtype=torch.int32)
        return dict(
            frame=_format_frame(obs, self.chw_transpose),
            reward=torch.zeros(1, 1),
            done=torch.ones(1, 1, dtype=torch.bool),
            episode_return=self.episode_return,
            episode_step=self.episode_step,
            last_action=torch.zeros(1, 1, dtype=torch.int64),
        )

    def step(self, action: torch.Tensor) -> dict:
        a = int(action.item())
        obs, reward, terminated, truncated, _info = self.gym_env.step(a)
        done = bool(terminated or truncated)

        # In-place updates (matches monobeast). The previous `self.x = self.x +
        # 1` pattern allocated a fresh tensor every step (80/rollout × 4 actors
        # = 320 tiny new tensors per rollout) which pumps PyTorch's caching
        # allocator high-water mark unnecessarily.
        self.episode_step += 1
        self.episode_return += float(reward)
        episode_return = self.episode_return
        episode_step = self.episode_step

        if done:
            # Auto-reset. If fixed_seed is set, use it so every episode is
            # the same instance (mirrors PPO fixed_env_seed). Otherwise
            # seed=None and the env picks a fresh seed.
            obs, _info = self.gym_env.reset(seed=self._reset_seed(is_initial=False))
            # Zero in-place rather than reallocating, so the snapshot reference
            # taken above keeps pointing at the final return (not the zeroed
            # value). Reusing the tensor also keeps the allocator quiet.
            self.episode_return = torch.zeros(1, 1)
            self.episode_step = torch.zeros(1, 1, dtype=torch.int32)

        return dict(
            frame=_format_frame(obs, self.chw_transpose),
            reward=torch.tensor(float(reward)).view(1, 1),
            done=torch.tensor(done).view(1, 1),
            episode_return=episode_return,
            episode_step=episode_step,
            last_action=action.view(1, 1),
        )

    def close(self) -> None:
        self.gym_env.close()
