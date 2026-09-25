"""The optimizer — a port of ``agent._make_opt`` and ``embodied/jax/opt.py``.

The chain, in ``_make_opt``'s order::

    clip_by_agc(0.3, pmin=1e-3)      adaptive gradient clipping, per parameter
    scale_by_rms(0.999, eps=1e-20)   divide by the bias-corrected RMS
    scale_by_momentum(0.9)           THEN take the bias-corrected momentum
    scale_by_learning_rate(sched)    lr 4e-5, linear warmup over 1000 steps

The order of the middle two is what makes this **LaProp**, not Adam. Adam takes the
momentum of the gradient and divides by the RMS of the gradient; here the gradient is
normalized by its RMS FIRST and the momentum is taken of the already-normalized
update. Swapping them changes the effective step size whenever the gradient scale is
moving, which is exactly the regime early training is in.

``optax.scale_by_learning_rate`` multiplies by ``-lr``, so the chain produces the
update to ADD to the parameters; the sign lives in the transform, not in the apply.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch


def warmup_schedule(step: int, lr: float, warmup: int) -> float:
    """``optax.join_schedules([linear(0, lr, warmup), const(lr)], [warmup])``.

    ``step`` is the number of updates ALREADY applied, so the first update runs at
    lr 0 and the schedule reaches full lr at step ``warmup``.
    """
    if not warmup:
        return lr
    if step >= warmup:
        return lr
    return lr * step / warmup


def agc_scale(param: torch.Tensor, update: torch.Tensor, clip: float, pmin: float = 1e-3) -> float:
    """``opt.clip_by_agc``'s per-parameter factor.

    ``upper = clip * max(pmin, ||param||)`` and the update is scaled by
    ``1 / max(1, ||update|| / upper)`` -- so it is never scaled UP, only down, and the
    bound is relative to the parameter's own norm. ``pmin`` keeps a zero-initialized
    parameter (``outscale: 0.0``) from having a bound of zero, which would clip its
    update to nothing and freeze it forever.
    """
    unorm = torch.linalg.vector_norm(update.flatten(), 2)
    pnorm = torch.linalg.vector_norm(param.flatten(), 2)
    upper = clip * torch.clamp(pnorm, min=pmin)
    return float(1.0 / torch.clamp(unorm / upper, min=1.0))


class LaProp(torch.optim.Optimizer):
    """``_make_opt``'s chain as a single torch optimizer.

    State per parameter: ``nu`` (the RMS accumulator) and ``mu`` (the momentum), both
    float32 regardless of the parameter dtype, plus one shared step counter used for
    both bias corrections and for the learning-rate schedule.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 4e-5,
        agc: float = 0.3,
        pmin: float = 1e-3,
        eps: float = 1e-20,
        beta1: float = 0.9,
        beta2: float = 0.999,
        momentum: bool = True,
        wd: float = 0.0,
        warmup: int = 1000,
        schedule: str = "const",
        anneal: int = 0,
    ):
        if schedule != "const":
            raise NotImplementedError(f"only schedule='const' is ported, got {schedule!r}")
        assert anneal > 0 or schedule == "const"
        if wd:
            raise NotImplementedError("wd is 0.0 at the frozen config; the wdregex mask is not ported")
        defaults = dict(
            lr=lr, agc=agc, pmin=pmin, eps=eps, beta1=beta1, beta2=beta2,
            momentum=momentum, warmup=warmup,
        )
        super().__init__(params, defaults)
        self._step = 0

    @property
    def step_count(self) -> int:
        return self._step

    def current_lr(self) -> float:
        group = self.param_groups[0]
        return warmup_schedule(self._step, group["lr"], group["warmup"])

    @torch.no_grad()
    def step(self, closure=None):  # noqa: D102
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # The bias corrections use the step number AFTER the increment, matching
        # optax's `safe_int32_increment` at the top of each update_fn.
        t = self._step + 1
        for group in self.param_groups:
            beta1, beta2 = group["beta1"], group["beta2"]
            eps, agc, pmin = group["eps"], group["agc"], group["pmin"]
            lr = warmup_schedule(self._step, group["lr"], group["warmup"])
            bc2 = 1 - beta2**t
            bc1 = 1 - beta1**t
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.detach().float()
                state = self.state[p]
                if not state:
                    state["nu"] = torch.zeros_like(g)
                    state["mu"] = torch.zeros_like(g)

                # 1. adaptive gradient clipping
                if agc:
                    g = g * agc_scale(p.detach().float(), g, agc, pmin)

                # 2. scale_by_rms: divide by the bias-corrected root mean square
                nu = state["nu"]
                nu.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                g = g / (torch.sqrt(nu / bc2) + eps)

                # 3. scale_by_momentum: momentum OF THE NORMALIZED update
                if group["momentum"]:
                    mu = state["mu"]
                    mu.mul_(beta1).add_(g, alpha=1 - beta1)
                    g = mu / bc1

                # 4. scale_by_learning_rate (which carries the minus sign)
                p.add_((-lr * g).to(p.dtype))

        self._step = t
        return loss


def global_norm(params: Iterable[torch.nn.Parameter]) -> float:
    """``optax.global_norm`` over the gradients, for the logged ``grad_norm``."""
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += float(torch.square(p.grad.detach().float()).sum())
    return math.sqrt(total)


def rms(tensors: Iterable[torch.Tensor]) -> float:
    """``nets.rms``: sqrt(sum of squares / total count) over a whole tree."""
    sumsq, count = 0.0, 0
    for t in tensors:
        if t is None:
            continue
        sumsq += float(torch.square(t.detach().float()).sum())
        count += t.numel()
    return math.sqrt(sumsq / count) if count else 0.0
