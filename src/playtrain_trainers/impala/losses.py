"""IMPALA loss functions (baseline / entropy / policy gradient).

Ports the three loss helpers from torchbeast/monobeast.py (and
polybeast_learner.py — identical math). Used by the V-trace learner step:

    vt = vtrace.from_logits(...)
    pg_loss     = compute_policy_gradient_loss(target_logits, actions, vt.pg_advantages)
    baseline_loss = baseline_cost * compute_baseline_loss(vt.vs - baseline)
    entropy_loss  = entropy_cost  * compute_entropy_loss(target_logits)
    total = pg_loss + baseline_loss + entropy_loss

All three return scalars (sum-reduced, not mean) — matching torchbeast.
The cost multipliers (`baseline_cost`, `entropy_cost`) live in the
trainer, not here.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_baseline_loss(advantages: torch.Tensor) -> torch.Tensor:
    """0.5 * sum(advantages**2). Advantages = vs - V(x_s) (V-trace target minus
    learner baseline). Gradient flows through `advantages`, so the caller
    must NOT detach the baseline before subtraction."""
    return 0.5 * torch.sum(advantages ** 2)


def compute_entropy_loss(logits: torch.Tensor) -> torch.Tensor:
    """Negative entropy of the policy (so minimizing this loss INCREASES entropy
    after the entropy_cost sign convention in monobeast). Returns sum_t,b,a
    p(a)*log p(a), which is negative-entropy summed over T*B."""
    policy = F.softmax(logits, dim=-1)
    log_policy = F.log_softmax(logits, dim=-1)
    return torch.sum(policy * log_policy)


def compute_policy_gradient_loss(
    logits: torch.Tensor,
    actions: torch.Tensor,
    advantages: torch.Tensor,
) -> torch.Tensor:
    """Surrogate PG loss: sum_t,b CE(logits, action) * stop_grad(advantage).

    Shapes: logits [T, B, A], actions [T, B], advantages [T, B].
    Advantages are detached so gradient only flows through `logits` — this is
    the standard score-function estimator with V-trace's pg_advantages target.
    """
    cross_entropy = F.nll_loss(
        F.log_softmax(torch.flatten(logits, 0, 1), dim=-1),
        target=torch.flatten(actions, 0, 1),
        reduction="none",
    )
    cross_entropy = cross_entropy.view_as(advantages)
    return torch.sum(cross_entropy * advantages.detach())
