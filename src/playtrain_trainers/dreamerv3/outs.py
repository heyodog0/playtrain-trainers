"""Output distributions — a port of ``embodied/jax/outs.py``.

Ported: ``Output``, ``Agg``, ``MSE``, ``Binary``, ``Categorical``, ``OneHot``,
``TwoHot``. Not ported (unreachable at the frozen atari100k config): ``Huber``,
``Normal``, ``Frozen``, ``Concat`` -- see D-016.

Everything computes in float32 regardless of the compute dtype, as upstream does
(``f32(...)`` in every constructor), because a bf16 softmax over 255 bins loses the
tails that the two-hot target lands in.
"""

from __future__ import annotations

from typing import Callable

import torch


class Output:
    """``outs.Output``: ``loss`` is the negative log-likelihood of a detached target."""

    def pred(self) -> torch.Tensor:
        raise NotImplementedError

    def loss(self, target: torch.Tensor) -> torch.Tensor:
        return -self.logp(target.detach())

    def sample(self, generator: torch.Generator | None = None) -> torch.Tensor:
        raise NotImplementedError

    def logp(self, event: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def prob(self, event: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.logp(event))

    def entropy(self) -> torch.Tensor:
        raise NotImplementedError

    def kl(self, other: "Output") -> torch.Tensor:
        raise NotImplementedError


class Agg(Output):
    """``outs.Agg``: reduce the last ``dims`` axes of loss/logp/entropy/kl.

    This is where the image loss becomes a per-step scalar (sum over H, W, C) and
    where the 32 categoricals of the latent become one distribution (sum over the
    stoch axis). The reduction is ``sum`` everywhere the frozen config reaches.
    """

    def __init__(self, output: Output, dims: int, agg: Callable = torch.sum):
        self.output = output
        self.axes = tuple(-i for i in range(1, dims + 1))
        self.agg = agg

    def pred(self) -> torch.Tensor:
        return self.output.pred()

    def loss(self, target: torch.Tensor) -> torch.Tensor:
        return self.agg(self.output.loss(target), self.axes)

    def sample(self, generator: torch.Generator | None = None) -> torch.Tensor:
        return self.output.sample(generator)

    def logp(self, event: torch.Tensor) -> torch.Tensor:
        return self.output.logp(event).sum(self.axes)

    def prob(self, event: torch.Tensor) -> torch.Tensor:
        return self.output.prob(event).sum(self.axes)

    def entropy(self) -> torch.Tensor:
        return self.agg(self.output.entropy(), self.axes)

    def kl(self, other: "Output") -> torch.Tensor:
        assert isinstance(other, Agg), other
        return self.agg(self.output.kl(other.output), self.axes)


class MSE(Output):
    """``outs.MSE``: squared error, NOT halved and NOT averaged.

    The image reconstruction loss is this, summed over H/W/C by an ``Agg(.., 3)``.
    """

    def __init__(self, mean: torch.Tensor, squash: Callable | None = None):
        self.mean = mean.float()
        self.squash = squash or (lambda x: x)

    def pred(self) -> torch.Tensor:
        return self.mean

    def loss(self, target: torch.Tensor) -> torch.Tensor:
        assert target.dtype.is_floating_point, target.dtype
        assert self.mean.shape == target.shape, (self.mean.shape, target.shape)
        return torch.square(self.mean - self.squash(target.float()).detach())


class Binary(Output):
    """``outs.Binary``: a logit, with ``pred`` thresholded at zero."""

    def __init__(self, logit: torch.Tensor):
        self.logit = logit.float()

    def pred(self) -> torch.Tensor:
        return self.logit > 0

    def prob1(self) -> torch.Tensor:
        """Not in the original; the continue head needs the probability itself."""
        return torch.sigmoid(self.logit)

    def logp(self, event: torch.Tensor) -> torch.Tensor:
        event = event.float()
        logp = torch.nn.functional.logsigmoid(self.logit)
        lognotp = torch.nn.functional.logsigmoid(-self.logit)
        return event * logp + (1 - event) * lognotp

    def sample(self, generator: torch.Generator | None = None) -> torch.Tensor:
        prob = torch.sigmoid(self.logit)
        return torch.bernoulli(prob, generator=generator).bool()


class Categorical(Output):
    """``outs.Categorical``, with the unimix applied in PROBABILITY space.

    ``logits = log((1 - u) * softmax(logits) + u / n)``: the mixture is taken over
    the normalized probabilities and logged back, so the stored logits are already
    normalized. Mixing in logit space instead would be a different distribution.
    """

    def __init__(self, logits: torch.Tensor, unimix: float = 0.0):
        logits = logits.float()
        if unimix:
            probs = torch.softmax(logits, -1)
            uniform = torch.ones_like(probs) / probs.shape[-1]
            probs = (1 - unimix) * probs + unimix * uniform
            logits = torch.log(probs)
        self.logits = logits

    def pred(self) -> torch.Tensor:
        return torch.argmax(self.logits, -1)

    def sample(self, generator: torch.Generator | None = None) -> torch.Tensor:
        # jax.random.categorical is Gumbel-max over the logits; torch.multinomial on
        # the softmax is the same distribution.
        flat = self.logits.reshape(-1, self.logits.shape[-1])
        idx = torch.multinomial(torch.softmax(flat, -1), 1, generator=generator)
        return idx.reshape(self.logits.shape[:-1])

    def logp(self, event: torch.Tensor) -> torch.Tensor:
        onehot = torch.nn.functional.one_hot(event.long(), self.logits.shape[-1])
        return (torch.log_softmax(self.logits, -1) * onehot).sum(-1)

    def entropy(self) -> torch.Tensor:
        logprob = torch.log_softmax(self.logits, -1)
        prob = torch.softmax(self.logits, -1)
        return -(prob * logprob).sum(-1)

    def kl(self, other: "Categorical") -> torch.Tensor:
        logprob = torch.log_softmax(self.logits, -1)
        logother = torch.log_softmax(other.logits, -1)
        prob = torch.softmax(self.logits, -1)
        return (prob * (logprob - logother)).sum(-1)


class OneHot(Output):
    """``outs.OneHot``: a categorical whose samples are straight-through one-hots.

    ``sample`` returns ``sg(onehot) + probs - sg(probs)``: the forward value is the
    hard one-hot, the gradient is the softmax's. This is what lets the world-model
    loss flow through the sampled latent.
    """

    def __init__(self, logits: torch.Tensor, unimix: float = 0.0):
        self.dist = Categorical(logits, unimix)

    def pred(self) -> torch.Tensor:
        return self._onehot_with_grad(self.dist.pred())

    def sample(self, generator: torch.Generator | None = None) -> torch.Tensor:
        return self._onehot_with_grad(self.dist.sample(generator))

    def logp(self, event: torch.Tensor) -> torch.Tensor:
        return (torch.log_softmax(self.dist.logits, -1) * event).sum(-1)

    def entropy(self) -> torch.Tensor:
        return self.dist.entropy()

    def kl(self, other: "OneHot") -> torch.Tensor:
        return self.dist.kl(other.dist)

    def _onehot_with_grad(self, index: torch.Tensor) -> torch.Tensor:
        value = torch.nn.functional.one_hot(
            index.long(), self.dist.logits.shape[-1]
        ).float()
        probs = torch.softmax(self.dist.logits, -1)
        return value.detach() + (probs - probs.detach())


def twohot_bins(bins: int, device=None) -> torch.Tensor:
    """``heads.Head.symexp_twohot``'s bin edges.

    ``symexp(linspace(-20, 0, .))`` mirrored about zero: the bins are dense near zero
    and reach ``+-(e^20 - 1)`` at the ends, which is why the reward head needs no
    clipping. At the frozen ``bins: 255`` the odd branch runs: 128 points from -20 to
    0, then the first 127 mirrored and reversed.
    """
    if bins % 2 == 1:
        half = torch.linspace(-20, 0, (bins - 1) // 2 + 1, dtype=torch.float32, device=device)
        half = symexp_t(half)
        return torch.cat([half, -half[:-1].flip(0)], 0)
    half = torch.linspace(-20, 0, bins // 2, dtype=torch.float32, device=device)
    half = symexp_t(half)
    return torch.cat([half, -half.flip(0)], 0)


def symexp_t(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.expm1(torch.abs(x))


class TwoHot(Output):
    """``outs.TwoHot``: a categorical over fixed bins, read out as a weighted mean.

    Two details are copied because they are load-bearing:

    * ``pred`` sums symmetrically (the left half reversed and added to the right half)
      so a uniform distribution over symmetric bins reads out as exactly zero. The
      naive left-to-right sum does not, and both output heads start at zero by
      construction (``outscale: 0.0``).
    * ``loss`` puts the target's mass on the two bins bracketing it, weighted by
      distance, and takes the cross-entropy against the log-softmax -- computed from
      the logits with a logsumexp, not from ``self.probs``.
    """

    def __init__(
        self,
        logits: torch.Tensor,
        bins: torch.Tensor,
        squash: Callable | None = None,
        unsquash: Callable | None = None,
    ):
        logits = logits.float()
        assert logits.shape[-1] == len(bins), (logits.shape, len(bins))
        assert bins.dtype == torch.float32, bins.dtype
        self.logits = logits
        self.probs = torch.softmax(logits, -1)
        self.bins = bins.to(logits.device)
        self.squash = squash or (lambda x: x)
        self.unsquash = unsquash or (lambda x: x)

    def pred(self) -> torch.Tensor:
        n = self.logits.shape[-1]
        if n % 2 == 1:
            m = (n - 1) // 2
            p1, p2, p3 = self.probs[..., :m], self.probs[..., m : m + 1], self.probs[..., m + 1 :]
            b1, b2, b3 = self.bins[..., :m], self.bins[..., m : m + 1], self.bins[..., m + 1 :]
            wavg = (p2 * b2).sum(-1) + ((p1 * b1).flip(-1) + (p3 * b3)).sum(-1)
            return self.unsquash(wavg)
        p1, p2 = self.probs[..., : n // 2], self.probs[..., n // 2 :]
        b1, b2 = self.bins[..., : n // 2], self.bins[..., n // 2 :]
        wavg = ((p1 * b1).flip(-1) + (p2 * b2)).sum(-1)
        return self.unsquash(wavg)

    def loss(self, target: torch.Tensor) -> torch.Tensor:
        assert target.dtype == torch.float32, target.dtype
        target = self.squash(target).detach()
        nbins = len(self.bins)
        below = (self.bins <= target[..., None]).to(torch.int64).sum(-1) - 1
        above = nbins - (self.bins > target[..., None]).to(torch.int64).sum(-1)
        below = below.clamp(0, nbins - 1)
        above = above.clamp(0, nbins - 1)
        equal = below == above
        dist_to_below = torch.where(equal, torch.ones_like(target), (self.bins[below] - target).abs())
        dist_to_above = torch.where(equal, torch.ones_like(target), (self.bins[above] - target).abs())
        total = dist_to_below + dist_to_above
        weight_below = dist_to_above / total
        weight_above = dist_to_below / total
        onehot_below = torch.nn.functional.one_hot(below, nbins).float()
        onehot_above = torch.nn.functional.one_hot(above, nbins).float()
        tgt = onehot_below * weight_below[..., None] + onehot_above * weight_above[..., None]
        log_pred = self.logits - torch.logsumexp(self.logits, -1, keepdim=True)
        return -(tgt * log_pred).sum(-1)
