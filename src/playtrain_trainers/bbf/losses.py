"""BBF's losses, augmentation, target updates and optimizer.

Pieces, and the one place each can go quietly wrong:

* **C51 projection** -- the categorical Bellman target has to be redistributed
  onto the fixed atom grid. The boundary cases (a target landing exactly on an
  atom, or clamped to v_min/v_max) are where implementations differ, so the
  vectorized version here is checked against a brute-force loop in the tests.
* **Target action selection** -- the gin sets BOTH `double_dqn = True` and
  `target_action_selection = True`, which pull in opposite directions (D-022).
* **SPR loss** -- BOTH sides L2-normalized, the target detached, the squared
  distance averaged over the jumps (not the cosine summed: that is 2.5x the
  official gradient at K = 5), and the importance weights applied to the
  C51 + SPR sum, not to C51 alone (D-035).
* **Augmentation** -- the shift must be the same for every frame in a stack
  (otherwise it destroys the motion cue the stack exists to provide) and
  independent across batch elements.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from playtrain_trainers.bbf.config import BBFConfig


# ----------------------------------------------------------------------
# C51
# ----------------------------------------------------------------------
def project_distribution(
    target_support: torch.Tensor,
    weights: torch.Tensor,
    support: torch.Tensor,
) -> torch.Tensor:
    """Project a distribution on `target_support` onto the fixed `support`.

    Args:
        target_support: [B, K] the shifted atom locations, `r + gamma^n * z`.
        weights: [B, K] the probability on each of those locations.
        support: [A] the fixed, evenly spaced atom grid.

    Returns [B, A]. Each target atom's mass is split between the two grid
    atoms it falls between, in inverse proportion to its distance from each --
    the standard C51 projection (Bellemare et al. 2017, Algorithm 1). Targets
    outside the grid are clamped onto its endpoints, which is what makes
    `v_max` a real modelling choice rather than a formality.
    """
    if target_support.shape != weights.shape:
        raise ValueError("target_support and weights must have the same shape")
    n_atoms = support.numel()
    v_min, v_max = support[0], support[-1]
    delta = (v_max - v_min) / (n_atoms - 1)

    clamped = target_support.clamp(min=v_min, max=v_max)
    # Position of each target atom in units of grid spacing.
    b = (clamped - v_min) / delta
    lower = b.floor().clamp(0, n_atoms - 1)
    upper = b.ceil().clamp(0, n_atoms - 1)

    # When a target lands exactly on an atom, floor == ceil and the naive
    # split (upper - b) and (b - lower) are both zero, which would delete the
    # mass. Give it all to the lower index in that case.
    exact = (upper == lower).float()
    w_lower = (upper - b) + exact
    w_upper = b - lower

    # Sized by the GRID, not by the number of target atoms: the two are equal
    # in the usual C51 step, but a projection of K target atoms onto A grid
    # atoms with K != A would otherwise silently return a K-wide tensor.
    out = torch.zeros(
        target_support.shape[0], n_atoms, dtype=weights.dtype, device=weights.device
    )
    out.scatter_add_(1, lower.long(), weights * w_lower)
    out.scatter_add_(1, upper.long(), weights * w_upper)
    return out


def c51_target_distribution(
    next_probs: torch.Tensor,
    next_action: torch.Tensor,
    n_step_return: torch.Tensor,
    discount: torch.Tensor,
    done: torch.Tensor,
    support: torch.Tensor,
) -> torch.Tensor:
    """The projected Bellman target for C51.

    `next_probs` is [B, A, atoms] from whichever network evaluates the
    bootstrap; `next_action` [B] is the action whichever network selects it
    with (see `select_bootstrap_action`). Where `done`, the bootstrap is
    dropped and the whole target mass sits on the return itself.
    """
    b = next_probs.shape[0]
    probs = next_probs.gather(
        1, next_action.view(b, 1, 1).expand(b, 1, next_probs.shape[-1])
    ).squeeze(1)
    # Zero the discount on terminal transitions: the target becomes a point
    # mass at the accumulated return.
    disc = discount * (~done).to(discount.dtype)
    target_support = n_step_return.unsqueeze(1) + disc.unsqueeze(1) * support.unsqueeze(0)
    return project_distribution(target_support, probs, support)


def select_bootstrap_action(
    online_next_q: torch.Tensor,
    target_next_q: torch.Tensor,
    cfg: BBFConfig,
) -> torch.Tensor:
    """Which network's argmax picks the bootstrap action (D-032).

    `double_dqn = True` means the ONLINE network selects and the target
    evaluates -- standard Double DQN. The official `spr_agent.py` is explicit:

        select_dist = online_dist if double_dqn else target_dist
        # Double DQN uses the current network for the action selection

    `target_action_selection` is a SEPARATE, orthogonal flag that decides
    which network acts in the ENVIRONMENT (see `acting_network` below), not
    which one selects the bootstrap. An earlier version of this function had
    both backwards; see D-022 (superseded) and D-032.
    """
    q = online_next_q if cfg.double_dqn else target_next_q
    return q.argmax(dim=1)


def acts_with_target(cfg: BBFConfig) -> bool:
    """Does the BEHAVIOUR policy come from the EMA target network? (D-032)

    The gin sets `target_action_selection = True`, so BBF acts in the
    environment using the target network, while learning with a Double-DQN
    bootstrap. Both flags are honored, and they are independent.
    """
    return bool(cfg.target_action_selection)


def c51_loss(
    logits: torch.Tensor,
    action: torch.Tensor,
    target_probs: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cross-entropy between the predicted and target atom distributions.

    Returns `(loss, per_sample_loss)`. `loss` is the (optionally weighted)
    batch mean; `per_sample_loss` is the DIFFERENTIABLE unweighted per-sample
    cross-entropy, so the training step can combine it with the SPR term and
    weight the sum (`combined_loss`). Priorities are set from its detached
    value -- weighting it as well would let importance weights feed back into
    the priorities they came from.
    """
    b = logits.shape[0]
    chosen = logits.gather(
        1, action.view(b, 1, 1).expand(b, 1, logits.shape[-1])
    ).squeeze(1)
    log_probs = F.log_softmax(chosen, dim=-1)
    per_sample = -(target_probs.detach() * log_probs).sum(dim=-1)
    loss = per_sample if weights is None else per_sample * weights
    return loss.mean(), per_sample


# ----------------------------------------------------------------------
# SPR
# ----------------------------------------------------------------------
def spr_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """SPR loss in the official form: mean over jumps of the squared distance
    between L2-normalized prediction and target (D-035).

    `predictions` and `targets` are [B, K, D]; `mask` [B, K] is False where
    the episode had already ended, so no target exists. Returns
    `(loss, per_sample)` with `per_sample` [B].

    `spr_agent.py`:

        spr_loss = jnp.power(spr_predictions - spr_targets, 2).sum(-1)
        spr_loss = (spr_loss * same_traj_mask.transpose(1, 0)).mean(0)

    For unit vectors ||p - t||^2 = 2 - 2 cos, so this is the negative cosine
    up to an affine map -- but the MEAN over the K jumps (masked entries
    still count in the denominator) and the factor 2 fix the gradient scale.
    An earlier version summed -cos over jumps, which at K = 5 was 2.5x the
    official SPR gradient; every result before v3 carries that.
    """
    if predictions.shape != targets.shape:
        raise ValueError(
            f"predictions {tuple(predictions.shape)} != targets {tuple(targets.shape)}"
        )
    pred = F.normalize(predictions, dim=-1, eps=1e-8)
    # The target is the EMA network's output and must not carry gradient; the
    # asymmetry is what stops the loss collapsing to a constant.
    tgt = F.normalize(targets.detach(), dim=-1, eps=1e-8)
    sq = (pred - tgt).pow(2).sum(dim=-1)  # [B, K]
    if mask is not None:
        sq = sq * mask.to(sq.dtype)
    per_sample = sq.mean(dim=1)
    # DIFFERENTIABLE, like `c51_loss`'s: `combined_loss` backprops through it,
    # so detaching here silently removes SPR from the gradient entirely
    # (D-043). The scalar is for logging only.
    return per_sample.mean(), per_sample


def combined_loss(
    rl_per_sample: torch.Tensor,
    spr_per_sample: torch.Tensor,
    weights: torch.Tensor,
    spr_weight: float,
) -> torch.Tensor:
    """`mean(loss_weights * (dqn_loss + spr_weight * spr_loss))` (D-035).

    The official agent applies the prioritized-replay importance weights to
    the WHOLE per-sample loss, SPR term included. An earlier version weighted
    only the C51 term.
    """
    return (weights * (rl_per_sample + spr_weight * spr_per_sample)).mean()


# ----------------------------------------------------------------------
# Augmentation
# ----------------------------------------------------------------------
def random_shift(obs: torch.Tensor, pad: int, generator: torch.Generator | None = None) -> torch.Tensor:
    """DrQ random shift: replicate-pad by `pad`, then crop back at random.

    One offset per batch element, shared by every channel -- the stack's
    frames must move together or the shift destroys the motion information
    the stack is there to carry.
    """
    if pad < 1:
        return obs
    b, c, h, w = obs.shape
    padded = F.pad(obs, (pad, pad, pad, pad), mode="replicate")
    offs = torch.randint(
        0, 2 * pad + 1, (b, 2), device=obs.device, generator=generator
    )
    out = torch.empty_like(obs)
    for i in range(b):
        dy, dx = int(offs[i, 0]), int(offs[i, 1])
        out[i] = padded[i, :, dy : dy + h, dx : dx + w]
    return out


def random_intensity(
    obs: torch.Tensor, scale: float, generator: torch.Generator | None = None
) -> torch.Tensor:
    """SPR's intensity jitter: scale each image by 1 + s*clip(N(0,1), -2, 2)."""
    if scale <= 0:
        return obs
    b = obs.shape[0]
    r = torch.randn((b, 1, 1, 1), device=obs.device, generator=generator)
    return obs * (1.0 + scale * r.clamp(-2.0, 2.0))


def augment(
    obs: torch.Tensor, cfg: BBFConfig, generator: torch.Generator | None = None
) -> torch.Tensor:
    """The gin's `data_augmentation = True`: random shift then intensity.

    Takes and returns FLOAT observations in [0, 1]. Intensity jitter is
    multiplicative and would be meaningless on uint8, and the replicate pad
    needs a float tensor anyway, so normalization happens before this.
    """
    if not cfg.data_augmentation:
        return obs
    out = random_shift(obs, cfg.aug_shift_pad, generator)
    return random_intensity(out, cfg.aug_intensity_scale, generator)


def to_float(obs: torch.Tensor) -> torch.Tensor:
    """uint8 [0,255] -> float [0,1], leaving an already-float tensor alone."""
    return obs.float() / 255.0 if obs.dtype == torch.uint8 else obs


# ----------------------------------------------------------------------
# Target network
# ----------------------------------------------------------------------
@torch.no_grad()
def ema_update(target: nn.Module, online: nn.Module, tau: float) -> None:
    """`target <- (1 - tau) * target + tau * online`, the gin's tau = 0.005.

    Buffers are COPIED rather than blended: the C51 support is a buffer, and
    an exponential average of a constant grid with itself is the same grid,
    but a copy keeps that true for any future non-float buffer too.
    """
    if not 0.0 <= tau <= 1.0:
        raise ValueError("tau must be in [0, 1]")
    for t, o in zip(target.parameters(), online.parameters(), strict=True):
        t.mul_(1.0 - tau).add_(o.detach(), alpha=tau)
    for t, o in zip(target.buffers(), online.buffers(), strict=True):
        t.copy_(o)


def build_target(online: nn.Module) -> nn.Module:
    """A frozen deep copy of the online network to act as the EMA target."""
    import copy

    target = copy.deepcopy(online)
    for p in target.parameters():
        p.requires_grad_(False)
    target.eval()
    return target


# ----------------------------------------------------------------------
# Optimizer
# ----------------------------------------------------------------------
def build_optimizer(net: nn.Module, cfg: BBFConfig) -> torch.optim.Optimizer:
    """AdamW with a separate encoder learning rate and no decay on biases.

    `create_scaling_optimizer` in the official agent builds `optax.adamw` with
    `mask = tree_map(lambda x: x.ndim != 1, p)` (its `decay_bias=False`
    default), so weight decay never touches a rank-1 parameter -- every bias
    here. It is applied through `optax.masked` twice, once for the
    encoder/transition-model keys and once for the rest, which is what the
    two learning rates are for (D-036).

    So four groups: {encoder, other} x {decay, no_decay}. The "encoder" lr
    covers the transition model too, exactly as `encoder_keys` does.
    `weight_decay = 0.1` with AdamW's decoupled decay is Dopamine's
    `create_scaling_optimizer`.
    """
    groups: dict[str, list[torch.Tensor]] = {
        "encoder": [], "encoder_no_decay": [], "other": [], "other_no_decay": []
    }
    for name, p in net.named_parameters():
        if not p.requires_grad:
            continue
        enc = name.startswith(("encoder.", "transition_model."))
        key = "encoder" if enc else "other"
        if p.ndim == 1:
            key += "_no_decay"
        groups[key].append(p)
    param_groups = []
    for key, params in groups.items():
        if not params:
            continue
        param_groups.append({
            "params": params,
            "name": key,
            "lr": cfg.encoder_learning_rate if key.startswith("encoder") else cfg.learning_rate,
            "weight_decay": 0.0 if key.endswith("_no_decay") else cfg.weight_decay,
        })
    return torch.optim.AdamW(
        param_groups, lr=cfg.learning_rate, eps=cfg.adam_eps, weight_decay=cfg.weight_decay
    )
