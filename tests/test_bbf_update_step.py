"""`update_step` end to end: which parameters a real update actually moves.

D-043 lived here. The SPR term was detached at the call site, so the transition
model and the predictor -- which receive gradient through NOTHING ELSE -- were
frozen for every v3 run, and BBF trained without its representation-learning
half. No test exercised `update_step`, and the loss-level tests all backwarded
`spr_loss(...)[0]`, which is differentiable. These tests close that gap by
asserting on the gradient each module receives from one real update.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from playtrain_trainers.bbf.config import BBFConfig
from playtrain_trainers.bbf.losses import build_optimizer, build_target
from playtrain_trainers.bbf.net import BBFNetwork
from playtrain_trainers.bbf.train import update_step

# Only the SPR loss reaches these; if they are ever frozen, SPR is off.
SPR_ONLY = ("transition_model", "predictor")


def _cfg(**over) -> BBFConfig:
    # width_scale 1 rather than the gin's 4: same modules, ~16x less CPU.
    base = dict(run_id="t", seed=0, device="cpu", width_scale=1)
    return BBFConfig(**(base | over))


def _batch(cfg: BBFConfig, num_actions: int, batch: int = 2) -> dict[str, torch.Tensor]:
    rng = np.random.default_rng(0)
    j = cfg.jumps
    u8 = lambda *s: torch.from_numpy(rng.integers(0, 255, s, dtype=np.uint8))  # noqa: E731
    return {
        "obs": u8(batch, *cfg.obs_shape),
        "next_obs": u8(batch, *cfg.obs_shape),
        "spr_obs": u8(batch, j, *cfg.obs_shape),
        "spr_actions": torch.from_numpy(rng.integers(0, num_actions, (batch, j))),
        "spr_mask": torch.ones(batch, j, dtype=torch.bool),
        "action": torch.from_numpy(rng.integers(0, num_actions, batch)),
        "n_step_return": torch.rand(batch),
        "discount": torch.full((batch,), 0.99),
        # dtypes as `replay.sample` produces them: bool done/mask, f32 weights.
        "done": torch.zeros(batch, dtype=torch.bool),
        "weights": torch.ones(batch),
    }


def _grads(cfg, num_actions=6):
    net = BBFNetwork(cfg, num_actions)
    target = build_target(net)
    opt = build_optimizer(net, cfg)
    stats = update_step(net, target, opt, _batch(cfg, num_actions), cfg, gamma=0.99)
    return stats, {n: p.grad for n, p in net.named_parameters()}


def test_every_trainable_parameter_receives_gradient():
    _, grads = _grads(_cfg())
    dead = [n for n, g in grads.items() if g is None or not torch.isfinite(g).all()
            or g.abs().sum() == 0]
    assert not dead, f"no usable gradient into: {dead}"


@pytest.mark.parametrize("module", SPR_ONLY)
def test_the_spr_only_modules_are_trained(module):
    """These move only if the SPR term is in the backward graph (D-043)."""
    _, grads = _grads(_cfg())
    got = [g.abs().sum() for n, g in grads.items() if n.startswith(module)]
    assert got, f"no parameters named {module}"
    assert all(g > 0 for g in got)


def test_zero_spr_weight_is_the_only_way_to_freeze_them():
    """The v3 bug must not be reachable except by asking for it."""
    _, grads = _grads(_cfg(spr_weight=0.0))
    frozen = [n for n, g in grads.items()
              if n.startswith(SPR_ONLY) and (g is None or g.abs().sum() == 0)]
    assert len(frozen) == len([n for n in grads if n.startswith(SPR_ONLY)])


def test_update_step_reports_a_finite_spr_loss():
    stats, _ = _grads(_cfg())
    assert np.isfinite(stats["spr_loss"]) and stats["spr_loss"] > 0
    assert np.isfinite(stats["loss"]) and np.isfinite(stats["rl_loss"])
