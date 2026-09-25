"""Periodic network resets: BBF's "harder resets".

The gin specifies:

    reset_every        = 20_000   ENV steps (D-033)
    no_resets_after    = 100_000  ENV steps
    shrink_perturb_keys = "encoder,transition_model"
    shrink_factor      = 0.5
    perturb_factor     = 0.5

and the paper's "Harder resets" section explains what makes them harder than
SR-SPR's: SR-SPR perturbed the convolutional layers only 20% of the way
towards a random target and fully reset the later layers; BBF moves the
encoder and transition model HALF the way.

So there are two behaviours, by parameter name:

    encoder.*, transition_model.*   theta <- 0.5 * theta + 0.5 * theta_fresh
    everything else                 theta <- theta_fresh

"theta_fresh" is a newly constructed network's initialization, so the reset
draws from exactly the same init scheme the run started from.

What the official `jit_reset` does with the rest of the state (D-034):

* **Optimizer state** is rebuilt from scratch (`optimizer.init(online_params)`
  with nothing copied over at BBF's settings). Cleared here.
* **The EMA target** gets the SAME treatment as the online network, from its
  OWN independent random draw (`reset_target=True`, `target_random_params`
  from a second PRNG key): its encoder and transition model are moved half
  way toward a fresh init, and its heads are fully re-randomized. It is NOT
  copied from the online network. An earlier version of this module
  re-synced target := online; every result before v3 carries that.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from playtrain_trainers.bbf.config import BBFConfig


def matches_key(param_name: str, keys: tuple[str, ...]) -> bool:
    """Does `param_name` belong to one of the shrink-and-perturb modules?

    Matches the module itself or anything beneath it, and only on a full path
    component -- so a key of "encoder" catches "encoder.stages.0.conv.weight"
    but would not catch a hypothetical "encoder_head.weight".
    """
    return any(param_name == k or param_name.startswith(k + ".") for k in keys)


@torch.no_grad()
def shrink_and_perturb_(
    param: torch.Tensor, fresh: torch.Tensor, shrink: float, perturb: float
) -> None:
    """In-place `param <- shrink * param + perturb * fresh`."""
    param.mul_(shrink).add_(fresh, alpha=perturb)


@torch.no_grad()
def _reset_params_(net: nn.Module, fresh: nn.Module, cfg: BBFConfig) -> dict[str, Any]:
    """Apply the shrink-and-perturb / full-reset rule to `net` from `fresh`."""
    fresh_params = dict(fresh.named_parameters())
    moved: dict[str, float] = {}
    kept_norm: dict[str, float] = {}
    n_shrunk = n_reset = 0
    for name, p in net.named_parameters():
        f = fresh_params[name]
        before = p.detach().clone()
        if matches_key(name, tuple(cfg.shrink_perturb_keys)):
            shrink_and_perturb_(p.data, f.data, cfg.shrink_factor, cfg.perturb_factor)
            n_shrunk += 1
        else:
            p.data.copy_(f.data)
            n_reset += 1
        group = name.split(".")[0]
        moved[group] = moved.get(group, 0.0) + float((p.data - before).pow(2).sum())
        kept_norm[group] = kept_norm.get(group, 0.0) + float(before.pow(2).sum())
    return {
        "params_shrunk_and_perturbed": n_shrunk,
        "params_fully_reset": n_reset,
        "l2_moved_by_group": {k: v**0.5 for k, v in moved.items()},
        "l2_before_by_group": {k: v**0.5 for k, v in kept_norm.items()},
    }


@torch.no_grad()
def reset_network(
    net: nn.Module,
    cfg: BBFConfig,
    num_actions: int,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    target: nn.Module | None = None,
) -> dict[str, Any]:
    """Apply one reset in place; returns a summary of what changed.

    The fresh initializations come from constructing new networks of the same
    config, so they use the run's own init scheme. Construction draws from the
    global torch RNG, which `set_global_seeds` seeds -- so a reset is
    reproducible for a given seed and step. The online network and the
    target each get their OWN fresh draw, as in `jit_reset`.

    Buffers are left alone: the only one is the C51 support, a constant grid
    that a fresh construction would reproduce identically.
    """
    from playtrain_trainers.bbf.net import BBFNetwork

    device = next(net.parameters()).device
    fresh = BBFNetwork(cfg, num_actions).to(device)
    summary = _reset_params_(net, fresh, cfg)

    if optimizer is not None:
        clear_optimizer_state(optimizer)

    if target is not None:
        fresh_t = BBFNetwork(cfg, num_actions).to(device)
        t_summary = _reset_params_(target, fresh_t, cfg)
        summary["target_l2_moved_by_group"] = t_summary["l2_moved_by_group"]

    summary["optimizer_state_cleared"] = optimizer is not None
    summary["target_reset"] = target is not None
    return summary


def clear_optimizer_state(optimizer: torch.optim.Optimizer) -> None:
    """Drop Adam's moment estimates, as `optimizer.init` in `jit_reset` does."""
    optimizer.state.clear()
