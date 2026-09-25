"""BBF's schedules: the receding update horizon, the discount, epsilon, resets.

Two different clocks drive these, and the official `spr_agent.py` is explicit
about which is which (D-033):

* **Resets run on ENV steps.** `_train_step` checks
  `self.training_steps > self.next_reset` and then does
  `self.training_steps += 1`, once per env step. So the gin's
  `reset_every = 20_000` and `no_resets_after = 100_000` are agent steps:
  at RR=2 the 20k-env-step interval is the paper's "every 40k gradient
  steps", and 100k is the whole Atari-100k budget (the gin's own comment is
  "need to change if training longer").

* **The anneals run on GRADIENT steps since the last reset.** The agent keeps
  `cycle_grad_steps`, adds the number of batches per update, zeroes it on a
  reset, and feeds it to `update_horizon_scheduler` / `gamma_scheduler`:

      update horizon  n: 10 -> 3
      discount    gamma: 0.97 -> 0.997

  both over `cycle_steps = 10_000` gradient steps. At RR=2 that is 5k env
  steps of a 20k reset interval, at RR=8 1.25k of 5k -- the paper's "always
  25% of training, regardless of the replay ratio".

An earlier version of this module put the resets on the gradient clock too,
which halved the reset interval at RR=2 and stopped all resets at the
midpoint of every run. Every result before v3 carries that.

The interpolation is exponential -- linear in log space -- which is how
Dopamine's `exponential_decay_scheduler` in the BBF agent does it.
"""
from __future__ import annotations

import math

from playtrain_trainers.bbf.config import BBFConfig

# spr_agent.BBFAgent.__init__: reset_offset=1. The first reset is due when
# training_steps exceeds reset_every + reset_offset.
RESET_OFFSET = 1


def exponential_anneal(
    step: int, decay_period: int, initial_value: float, final_value: float
) -> float:
    """Interpolate `initial_value` -> `final_value` linearly in log space.

    Mirrors Dopamine's `exponential_decay_scheduler`: the fraction of the
    period remaining is clipped to [0, 1], so the value is exactly
    `initial_value` at step 0 and exactly `final_value` from `decay_period`
    onward. Works in either direction; the discount anneals upward.
    """
    if decay_period < 1:
        raise ValueError("decay_period must be >= 1")
    if initial_value <= 0 or final_value <= 0:
        raise ValueError("exponential interpolation needs positive endpoints")
    start, end = math.log(initial_value), math.log(final_value)
    remaining = max(0.0, min(1.0, (decay_period - step) / decay_period))
    return math.exp(remaining * (start - end) + end)


def update_horizon(cycle_grad_steps: int, cfg: BBFConfig) -> int:
    """The n-step horizon after `cycle_grad_steps` gradient steps since a reset.

    Rounded to an integer because the replay buffer accumulates a whole
    number of transitions; the underlying schedule is continuous. Matches
    `int(onp.round(n_schedule(x) * max_update_horizon))` in the official agent.
    """
    if cycle_grad_steps < 0:
        raise ValueError("cycle_grad_steps must be >= 0")
    value = exponential_anneal(
        cycle_grad_steps, cfg.cycle_steps, cfg.max_update_horizon, cfg.update_horizon
    )
    return max(1, int(round(value)))


def discount(cycle_grad_steps: int, cfg: BBFConfig) -> float:
    """The discount after `cycle_grad_steps` gradient steps since a reset."""
    if cycle_grad_steps < 0:
        raise ValueError("cycle_grad_steps must be >= 0")
    return exponential_anneal(cycle_grad_steps, cfg.cycle_steps, cfg.min_gamma, cfg.gamma)


class ResetSchedule:
    """When to reset, in ENV steps, exactly as `spr_agent.BBFAgent` does it.

    `due(env_step)` is the official `training_steps > next_reset`, evaluated
    once per env step with `env_step` the 0-based count of env steps already
    taken (the agent's `training_steps` before its increment). `fire(env_step)`
    is `reset_weights`: it moves `next_reset` forward by `reset_every` and
    returns whether the reset is allowed -- it is refused when the NEXT reset
    would land past `no_resets_after` ("need at least `interval` before
    `no_resets_after` to recover"). At the gin's 20k / 100k on a 100k run that
    is 3 resets, at ~20k, 40k and 60k env steps; at the paper's RR=8 value of
    5k it is 18, the last at ~90k.

    Step 0 is never a reset: `next_reset` starts at `reset_every + 1`.
    """

    def __init__(self, cfg: BBFConfig):
        self.reset_every = int(cfg.reset_every)
        self.no_resets_after = int(cfg.no_resets_after)
        self.next_reset = self.reset_every + RESET_OFFSET
        self.count = 0

    def due(self, env_step: int) -> bool:
        return self.reset_every > 0 and env_step > self.next_reset

    def fire(self, env_step: int) -> bool:
        self.next_reset = env_step + self.reset_every
        if self.next_reset > self.no_resets_after + RESET_OFFSET:
            return False
        self.count += 1
        return True


def reset_env_steps(cfg: BBFConfig, training_steps: int | None = None) -> list[int]:
    """Every env step at which a reset fires over a run, for the record.

    Replays the schedule rather than deriving a closed form, so it cannot
    drift from what the training loop actually does.
    """
    total = cfg.training_steps if training_steps is None else training_steps
    sched = ResetSchedule(cfg)
    out = []
    for step in range(total):
        if sched.due(step) and sched.fire(step):
            out.append(step)
    return out


def epsilon(env_step: int, cfg: BBFConfig) -> float:
    """Exploration epsilon at this ENV step (not gradient step).

    Dopamine's `linearly_decaying_epsilon`: 1.0 while the buffer fills, then a
    linear decay to `epsilon_train` over `epsilon_decay_period` steps. The gin
    sets `epsilon_train = 0.0` and `epsilon_decay_period = 2001`, so BBF is
    fully greedy from about 4k env steps on -- exploration comes from the
    random warmup and the resets, not from epsilon. PROTOCOL section 1 lists
    only the 0.0, which is why this is spelled out here.
    """
    if env_step < cfg.min_replay_history:
        return 1.0
    decayed = env_step - cfg.min_replay_history
    if cfg.epsilon_decay_period < 1:
        return cfg.epsilon_train
    frac = min(1.0, decayed / cfg.epsilon_decay_period)
    return 1.0 + frac * (cfg.epsilon_train - 1.0)
