"""PROTOCOL section 5 C: PPO and IMPALA at the same 100k-agent-step budget.

The point of these arms is to show BBF's sample efficiency against something
on the SAME game with the SAME observations, so three things are held fixed
and one is deliberately not:

  fixed   the wrapper stack -- `make_reference_env` is the BBF stack in its
          HWC form (84x84 grayscale x4, frame_skip 4, random no-ops,
          terminal-on-life-loss, reward clipped to [-1,1]), which both
          trainers accept because they size their nets from an HWC
          observation space and transpose internally
  fixed   the budget -- 100_000 AGENT STEPS, see D-027 on why that is the
          right number for each trainer's own counter
  fixed   the scoring -- every arm is scored by `bbf.evaluate` on the same
          100-episode seed pool (D-017), NOT by each trainer's own eval, so
          the numbers in the report are commensurable
  free    the algorithm's own hyperparameters, which stay at whatever the
          existing configs use; retuning PPO or IMPALA for a 100k budget is
          not in scope and would be a different experiment

Injection. IMPALA's `train()` already takes an `env_fn`, so it needs nothing
special. PPO's `train()` builds its venv internally from a module-level
`make_env` that hardcodes `PlayTrainEnv(game=..., frame_skip=...)` and ignores
obs size, obs mode and frame stacking entirely -- so with PPO's own plumbing
it would see 64x64 RGB unstacked against BBF's 84x84 grayscale x4. It is
therefore overridden at runtime, in this process only (D-026). The shared
trainer file is not edited: it is the code behind published numbers.
"""
from __future__ import annotations

import logging
from typing import Any

import gymnasium as gym

from playtrain_trainers.bbf.config import BBFConfig
from playtrain_trainers.bbf.envs import make_playtrain_atari100k

log = logging.getLogger(__name__)


def make_reference_env(cfg: BBFConfig, seed: int, *, training: bool = True) -> gym.Env:
    """The BBF wrapper stack in HWC, for the reference trainers."""
    return make_playtrain_atari100k(cfg, seed, training=training, channel_first=False)


# ----------------------------------------------------------------------
# IMPALA
# ----------------------------------------------------------------------
def impala_env_fn(cfg: BBFConfig, base_seed: int):
    """An `env_fn(actor_index) -> (env, initial_seed)` for IMPALA's `train()`.

    Each actor gets its own game-seed stream, mirroring `_default_env_fn`'s
    stride so two actors never replay the same episode in lockstep.
    """
    def _fn(actor_index: int):
        seed = base_seed * 1_000_000 + actor_index
        return make_reference_env(cfg, seed, training=True), seed

    return _fn


def run_impala(cfg: BBFConfig, seed: int, log_dir: str, total_steps: int) -> dict[str, Any]:
    """Train IMPALA on the BBF wrapper stack at `total_steps` AGENT steps."""
    from playtrain_trainers.impala.train import ImpalaConfig, train

    icfg = ImpalaConfig(
        game=cfg.game,
        env_backend="playtrain",
        # D-027: IMPALA's counter is `step += unroll_length * batch_size`, one
        # unit per env DECISION, so total_steps is agent steps directly.
        total_steps=total_steps,
        seed=seed,
        log_dir=log_dir,
        # The wrapper stack owns frame skip and stacking, so IMPALA's own
        # knobs must stay at their no-op defaults or they would compound.
        frame_skip=1,
        frame_stack=1,
        obs_shape=(cfg.frame_stack, cfg.obs_size, cfg.obs_size),
        num_actions=None,  # taken from the env spec
        device="auto",
    )
    return train(icfg, env_fn=impala_env_fn(cfg, seed))


# ----------------------------------------------------------------------
# PPO
# ----------------------------------------------------------------------
def install_ppo_env_override(cfg: BBFConfig) -> None:
    """Point PPO's module-level `make_env` at the BBF wrapper stack.

    A runtime override rather than an edit: `train_ppo_clean.py` is the code
    behind existing published numbers, and its `make_env` signature is fixed
    by `build_venv`, so the replacement accepts and ignores the arguments it
    does not need. Confined to this process (D-026).
    """
    import playtrain_trainers.train_ppo_clean as ppo

    def _make_env(game, seed, backend="playtrain", fixed_env_seed=None,
                  frame_skip=1, train_seeds=None):
        def _thunk():
            # No reset() in the thunk -- SubprocVecEnv resets after
            # construction, and an extra tick here desynchronizes frameCount
            # (the note in PPO's own make_env explains the consequence).
            return make_reference_env(cfg, seed, training=True)

        return _thunk

    ppo.make_env = _make_env
    log.info("PPO make_env overridden with the BBF wrapper stack (D-026)")


def run_ppo(cfg: BBFConfig, seed: int, log_dir: str, total_steps: int,
            n_envs: int = 8) -> None:
    """Train PPO on the BBF wrapper stack at `total_steps` AGENT steps."""
    import playtrain_trainers.train_ppo_clean as ppo

    install_ppo_env_override(cfg)
    pcfg = ppo.Config(
        game=cfg.game,
        env_backend="playtrain",
        vec_backend="subproc",
        # D-027: PPO's counter is `global_step += total_envs`, one unit per
        # env decision summed over envs, so this is agent steps directly --
        # the same unit and the same budget BBF got.
        total_timesteps=total_steps,
        n_envs=n_envs,
        seed=seed,
        log_dir=log_dir,
        # Owned by the wrapper stack; left at defaults so nothing compounds.
        frame_skip=1,
        obs_size=cfg.obs_size,
        device="auto",
    )
    return ppo.train(pcfg)
