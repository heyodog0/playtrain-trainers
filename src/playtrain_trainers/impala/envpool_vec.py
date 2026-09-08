"""EnvPool adapter with the NativeVecEnv step/reset surface, so act_vec can
drive real ProcGen (or any EnvPool env) through the FULL vec topology — the
1-to-1 layer-2 comparison row (same trainer, same workers/GPUs/batching;
only the env implementation differs).

Scope: throughput benchmarking. Autoreset is EnvPool's own; the playtrain
seed policies (formula/fixed/pool) don't apply — reset seeds are ignored.
frame_skip/render_skip are meaningless here (procgen has no such knobs;
run fs=1 configs). Single-buffer only (PingPong is native-only).
"""
from __future__ import annotations

import numpy as np


class EnvPoolVec:
    """envpool.make_gymnasium wrapped to look like NativeVecEnv."""

    def __init__(self, env_id: str, num_envs: int, num_threads: int = 0,
                 env_kwargs: dict | None = None, **_ignored):
        """env_kwargs is forwarded verbatim to envpool.make_gymnasium.

        This is what makes an ALE row matchable. EnvPool's Atari defaults are
        84x84 grayscale, stack_num=4, frame_skip=4 — none of which match a
        PlayTrain replica's 64x64x3 RGB single frame at fs=1. Passing
        {"img_height": 64, "img_width": 64, "gray_scale": False,
         "stack_num": 1, "frame_skip": 1} makes ALE emit the identical
        observation format, so an Atari A/B differs only in the environment
        rather than also in what the encoder sees. ProcGen ignores these (it is
        64x64x3 natively), so the same field serves both backends.
        """
        import envpool
        self.env_kwargs = dict(env_kwargs or {})
        self.env = envpool.make_gymnasium(
            env_id, num_envs=num_envs, num_threads=num_threads,
            **self.env_kwargs)
        self.num_envs = num_envs
        self.num_threads = num_threads

    @staticmethod
    def _hwc(obs: np.ndarray) -> np.ndarray:
        # act_vec expects (N, H, W, C) uint8. EnvPool procgen emits HWC
        # already; Atari-style CHW gets transposed.
        if obs.ndim == 4 and obs.shape[1] in (1, 3, 4) \
                and obs.shape[-1] not in (1, 3, 4):
            obs = np.ascontiguousarray(obs.transpose(0, 2, 3, 1))
        return obs

    def reset(self, seeds=None):
        del seeds  # EnvPool seeds at construction; policies not applicable
        obs, _info = self.env.reset()
        return self._hwc(obs)

    def step(self, actions: np.ndarray):
        obs, rew, term, trunc, _info = self.env.step(
            np.asarray(actions, dtype=np.int32))
        return (self._hwc(obs), rew.astype(np.float32),
                term.astype(bool), trunc.astype(bool), None)

    def close(self) -> None:
        # envpool 0.8.4 + gymnasium>=1.0: env.close() trips a missing
        # `.closed` attribute; the process teardown reclaims everything.
        pass
