"""Actor-critic and imagination — a port of ``imag_loss``, ``repl_loss``,
``lambda_return`` and ``embodied/jax/utils.py``'s ``Normalize`` and ``SlowModel``.

Four things here are easy to get backwards, and each has a test:

* **``slowtar: False``.** The slow critic is a REGULARIZER
  (``slowreg * value.loss(sg(slowvalue.pred()))``), not the bootstrap target. In
  DreamerV2 it was the target, so the habit is wrong here.
* **``disc = 1`` in imagination.** ``contdisc: True`` puts the discount inside the
  continue prediction, whose TARGET was already scaled by ``1 - 1/horizon`` in the
  world-model loss. The replay critic is the exception and uses ``1 - 1/333``.
* **the advantage is not de-meaned.** ``adv = (ret - tarval[:, :-1]) / rscale`` uses
  the percentile SCALE only; ``roffset`` is computed and used for a metric, never for
  the advantage.
* **``lambda_return`` ignores its ``val`` argument.** Only ``boot`` is read. The
  parameter exists so the shape assertion covers it; in imagination both are
  ``tarval``, so it makes no difference there, but the replay critic passes different
  tensors for the two and only ``boot`` matters.
"""

from __future__ import annotations

import copy

import torch
from torch import nn

from playtrain_trainers.dreamerv3 import outs


# ----------------------------------------------------------------------
# Normalizers (utils.Normalize)
# ----------------------------------------------------------------------
class Normalize(nn.Module):
    """``utils.Normalize`` with ``impl`` in ``('none', 'perc', 'meanstd')``.

    At the frozen config ``retnorm`` is ``perc`` with ``rate 0.01``, ``limit 1.0``,
    percentiles 5 and 95 and ``debias: False``; ``valnorm`` and ``advnorm`` are
    ``none``, which returns ``(0.0, 1.0)`` and makes their call sites identities.

    ``limit: 1.0`` on the return normalizer means the scale is ``max(1, hi - lo)``:
    returns SMALLER than one are never scaled up. Without it, an agent that has found
    nothing yet would divide a near-zero spread into a huge advantage.
    """

    def __init__(
        self,
        impl: str = "perc",
        rate: float = 0.01,
        limit: float = 1.0,
        perclo: float = 5.0,
        perchi: float = 95.0,
        debias: bool = False,
    ):
        super().__init__()
        self.impl = impl
        self.rate = rate
        self.limit = limit
        self.perclo = perclo
        self.perchi = perchi
        self.debias = debias
        if impl not in ("none", "perc", "meanstd"):
            raise NotImplementedError(impl)
        if impl != "none":
            if debias:
                self.register_buffer("corr", torch.zeros(()))
            if impl == "perc":
                self.register_buffer("lo", torch.zeros(()))
                self.register_buffer("hi", torch.zeros(()))
            else:
                self.register_buffer("mean", torch.zeros(()))
                self.register_buffer("sqrs", torch.zeros(()))

    def forward(self, x: torch.Tensor, update: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        if update:
            self.update(x)
        return self.stats()

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        x = x.detach().float()
        if self.impl == "none":
            return
        if self.impl == "perc":
            self._ema("lo", torch.quantile(x, self.perclo / 100))
            self._ema("hi", torch.quantile(x, self.perchi / 100))
        else:
            self._ema("mean", x.mean())
            self._ema("sqrs", torch.square(x).mean())
        if self.debias:
            self._ema("corr", torch.ones(()))

    def _ema(self, name: str, value: torch.Tensor) -> None:
        buf = getattr(self, name)
        buf.copy_((1 - self.rate) * buf + self.rate * value.to(buf.device))

    def stats(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.impl == "none":
            return torch.zeros(()), torch.ones(())
        corr = 1.0
        if self.debias:
            corr = 1.0 / torch.clamp(self.corr, min=self.rate)
        if self.impl == "perc":
            lo, hi = self.lo * corr, self.hi * corr
            return lo.detach(), torch.clamp(hi - lo, min=self.limit).detach()
        mean = self.mean * corr
        std = torch.sqrt(torch.relu(self.sqrs * corr - mean**2))
        return mean.detach(), torch.clamp(std, min=self.limit).detach()


# ----------------------------------------------------------------------
# Slow critic (utils.SlowModel)
# ----------------------------------------------------------------------
class SlowModel(nn.Module):
    """``utils.SlowModel``: an EMA copy of ``source``, updated every ``every`` calls.

    ``mix = rate if count % every == 0 else 0`` and
    ``dst = mix * src + (1 - mix) * dst``. At ``rate 0.02, every 1`` the copy moves
    2% of the way to the live critic on every gradient step. The copy starts as an
    exact clone of the source, not from a fresh initialization.
    """

    def __init__(self, source: nn.Module, rate: float = 0.02, every: int = 1):
        super().__init__()
        assert rate == 1 or rate < 0.5, rate
        self.source = source
        self.model = copy.deepcopy(source)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.rate = rate
        self.every = every
        self.register_buffer("count", torch.zeros((), dtype=torch.long))

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    @torch.no_grad()
    def update(self) -> None:
        mix = self.rate if int(self.count) % self.every == 0 else 0.0
        if mix:
            for dst, src in zip(self.model.parameters(), self.source.parameters()):
                dst.mul_(1 - mix).add_(src.detach(), alpha=mix)
            for dst, src in zip(self.model.buffers(), self.source.buffers()):
                if dst.dtype.is_floating_point:
                    dst.mul_(1 - mix).add_(src.detach(), alpha=mix)
        self.count += 1


# ----------------------------------------------------------------------
# lambda_return
# ----------------------------------------------------------------------
def lambda_return(
    last: torch.Tensor,
    term: torch.Tensor,
    rew: torch.Tensor,
    val: torch.Tensor,
    boot: torch.Tensor,
    disc: float,
    lam: float,
) -> torch.Tensor:
    """``agent.lambda_return``, verbatim.

    Inputs are ``(B, T)``; the output is ``(B, T - 1)``. The recursion::

        live   = (1 - term)[:, 1:] * disc
        cont   = (1 - last)[:, 1:] * lam
        interm = rew[:, 1:] + (1 - cont) * live * boot[:, 1:]
        ret_t  = interm_t + live_t * cont_t * ret_{t+1}

    with ``ret_{T-1}`` seeded from ``boot[:, -1]``. ``term`` kills the bootstrap AND
    the recursion (a terminal state is worth nothing after it); ``last`` only kills
    the recursion, so a truncated episode still bootstraps from its last value.

    ``val`` is accepted and never read -- upstream passes it so the shape assertion
    covers it. Kept in the signature so the call sites read like the original.
    """
    shapes = {x.shape for x in (last, term, rew, val, boot)}
    assert len(shapes) == 1, shapes
    rets = [boot[:, -1]]
    live = (1 - term[:, 1:].float()) * disc
    cont = (1 - last[:, 1:].float()) * lam
    interm = rew[:, 1:] + (1 - cont) * live * boot[:, 1:]
    for t in reversed(range(live.shape[1])):
        rets.append(interm[:, t] + live[:, t] * cont[:, t] * rets[-1])
    return torch.stack(list(reversed(rets))[:-1], 1)


# ----------------------------------------------------------------------
# imag_loss / repl_loss
# ----------------------------------------------------------------------
def imag_loss(
    act: torch.Tensor,
    rew: torch.Tensor,
    con: torch.Tensor,
    policy: outs.Output,
    value: outs.Output,
    slowvalue: outs.Output,
    retnorm: Normalize,
    valnorm: Normalize,
    advnorm: Normalize,
    update: bool = True,
    contdisc: bool = True,
    slowtar: bool = False,
    horizon: int = 333,
    lam: float = 0.95,
    actent: float = 3e-4,
    slowreg: float = 1.0,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, float]]:
    """``agent.imag_loss``. All inputs are ``(B*K, H+1)`` over the imagined rollout."""
    losses: dict[str, torch.Tensor] = {}
    metrics: dict[str, float] = {}

    voffset, vscale = valnorm.stats()
    val = value.pred() * vscale + voffset
    slowval = slowvalue.pred() * vscale + voffset
    tarval = slowval if slowtar else val
    # contdisc: the discount already lives in the continue prediction, whose target
    # was scaled by (1 - 1/horizon) in the world-model loss. Applying it again here
    # would square it.
    disc = 1.0 if contdisc else 1 - 1 / horizon
    weight = torch.cumprod(disc * con, 1) / disc
    last = torch.zeros_like(con)
    term = 1 - con
    ret = lambda_return(last, term, rew, tarval, tarval, disc, lam)

    roffset, rscale = retnorm(ret, update)
    # The advantage is SCALED but not de-meaned: roffset exists for the metric below.
    adv = (ret - tarval[:, :-1]) / rscale
    aoffset, ascale = advnorm(adv, update)
    adv_normed = (adv - aoffset) / ascale
    logpi = policy.logp(act.detach())[:, :-1]
    ent = policy.entropy()[:, :-1]
    metrics["rscale"] = float(rscale.detach())
    metrics["roffset"] = float(roffset.detach())
    metrics["logpi"] = float(logpi.mean().detach())
    losses["policy"] = weight[:, :-1].detach() * -(
        logpi * adv_normed.detach() + actent * ent
    )

    voffset, vscale = valnorm(ret, update)
    tar_normed = (ret - voffset) / vscale
    tar_padded = torch.cat([tar_normed, 0 * tar_normed[:, -1:]], 1)
    losses["value"] = weight[:, :-1].detach() * (
        value.loss(tar_padded.detach())
        + slowreg * value.loss(slowvalue.pred().detach())
    )[:, :-1]

    ret_normed = ((ret - roffset) / rscale).detach()
    def f(t: torch.Tensor) -> float:
        return float(t.detach())

    metrics.update(
        adv=f(adv.mean()),
        adv_std=f(adv.std()),
        adv_mag=f(adv.abs().mean()),
        rew=f(rew.mean()),
        con=f(con.mean()),
        ret=f(ret_normed.mean()),
        val=f(val.mean()),
        tar=f(tar_normed.mean()),
        weight=f(weight.mean()),
        slowval=f(slowval.mean()),
        ret_min=f(ret_normed.min()),
        ret_max=f(ret_normed.max()),
        ret_rate=f((ret_normed.abs() >= 1.0).float().mean()),
        ent=f(ent.mean()),
    )
    return losses, {"ret": ret}, metrics


def repl_loss(
    last: torch.Tensor,
    term: torch.Tensor,
    rew: torch.Tensor,
    boot: torch.Tensor,
    value: outs.Output,
    slowvalue: outs.Output,
    valnorm: Normalize,
    update: bool = True,
    slowreg: float = 1.0,
    slowtar: bool = False,
    horizon: int = 333,
    lam: float = 0.95,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, float]]:
    """``agent.repl_loss``: the critic trained on the REPLAYED sequence.

    The λ-return runs over real replayed rewards but bootstraps from ``boot`` -- the
    imagined return at each step -- so the critic is anchored to real data while
    still being told what the world model thinks happens next. Here ``disc`` is
    ``1 - 1/horizon`` explicitly, because these steps have no continue prediction
    carrying it.

    The weight is ``~last`` and is NOT detached upstream: it comes from the data, so
    there is nothing to detach.
    """
    voffset, vscale = valnorm.stats()
    val = value.pred() * vscale + voffset
    slowval = slowvalue.pred() * vscale + voffset
    tarval = slowval if slowtar else val
    disc = 1 - 1 / horizon
    weight = (~last).float() if last.dtype == torch.bool else (1 - last).float()
    ret = lambda_return(last.float(), term.float(), rew, tarval, boot, disc, lam)

    voffset, vscale = valnorm(ret, update)
    ret_normed = (ret - voffset) / vscale
    ret_padded = torch.cat([ret_normed, 0 * ret_normed[:, -1:]], 1)
    losses = {
        "repval": weight[:, :-1]
        * (
            value.loss(ret_padded.detach())
            + slowreg * value.loss(slowvalue.pred().detach())
        )[:, :-1]
    }
    return losses, {"ret": ret}, {}


def imagination_starts(entries: dict[str, torch.Tensor], nlast: int) -> dict[str, torch.Tensor]:
    """``RSSM.starts``: flatten the last ``nlast`` steps of every sequence into the
    batch axis, so imagination runs ``B * K`` rollouts in parallel.

    At ``imag_last: 0`` upstream sets ``K = T``, i.e. EVERY replayed step is a start:
    16 sequences x 64 steps = 1024 rollouts of 15 steps each per gradient step.
    """
    B = next(iter(entries.values())).shape[0]
    return {k: v[:, -nlast:].reshape(B * nlast, *v.shape[2:]) for k, v in entries.items()}


def assemble_imagined(
    repfeat: dict[str, torch.Tensor],
    imgfeat: dict[str, torch.Tensor],
    K: int,
    ac_grads: bool = False,
) -> dict[str, torch.Tensor]:
    """``agent.loss``: prepend the replayed start feature to the imagined rollout.

    The result is ``H + 1`` features per rollout: the real step the rollout started
    from, then the ``H`` imagined ones. Both halves are detached at ``ac_grads:
    False`` -- the actor-critic gradient does NOT reach the world model.
    """
    # Any key: the feature dict is opaque to the actor-critic (U10).
    B = next(iter(repfeat.values())).shape[0]
    first = {k: v[:, -K:].reshape(B * K, 1, *v.shape[2:]) for k, v in repfeat.items()}
    out = {}
    for k in imgfeat:
        head = first[k] if ac_grads else first[k].detach()
        out[k] = torch.cat([head, imgfeat[k].detach()], 1)
    return out


def reduce_imagination_losses(
    losses: dict[str, torch.Tensor], B: int, K: int
) -> dict[str, torch.Tensor]:
    """``losses.update({k: v.mean(1).reshape((B, K)) for k, v in los.items()})``.

    The imagination losses are ``(B*K, H)``; they are averaged over the H horizon
    steps and reshaped back to ``(B, K)`` so every world-model term and every
    actor-critic term is ``(B, T)`` before the final mean. MEAN over the horizon, not
    sum -- a sum would scale the actor loss by 15.
    """
    return {k: v.mean(1).reshape(B, K) for k, v in losses.items()}
