"""The world-model seam — MISSION deliverable 2.

The actor-critic, the replay and the training loop talk to the world model only
through :class:`WorldModel`. Everything DreamerV3-specific about the world model —
the convolutional encoder, the block-diagonal GRU, the categorical latent, the
reconstruction decoder — lives behind it in :class:`RSSMWorldModel`. Swapping what
the model predicts, or how it factors its latent, means writing another class with
these seven methods and changing one line in ``Agent``.

What the interface deliberately does NOT expose, because nothing outside depends on
it: tokens, the encoder at all, the decoder at all, the KL structure, or the shape of
the latent. ``feat`` is an opaque dict; only :meth:`feat2tensor` turns it into
something the heads can consume, and only :attr:`entry_keys` names the parts that
must survive a round trip through replay.

See ``WORLD_MODEL_INTERFACE.md`` for the contract in prose, including what each
method promises and where a factored latent or a different prediction target would
plug in.
"""

from __future__ import annotations

from typing import Any, Callable, Protocol, runtime_checkable

import torch
from torch import nn

from playtrain_trainers.dreamerv3 import heads as heads_
from playtrain_trainers.dreamerv3 import losses as losses_
from playtrain_trainers.dreamerv3 import outs
from playtrain_trainers.dreamerv3 import rssm as rssm_
from playtrain_trainers.dreamerv3.config import DreamerConfig

Carry = dict[str, torch.Tensor]
Feat = dict[str, torch.Tensor]


@runtime_checkable
class WorldModel(Protocol):
    """What the rest of the agent may assume about a world model.

    Seven members. A replacement implements these and nothing else is required.
    """

    #: Width of ``feat2tensor(feat)``; the heads are built against it.
    feat_dim: int
    #: Keys of ``entries`` that replay stores per step and hands back as a carry.
    entry_keys: tuple[str, ...]

    def initial(self, batch_size: int, device: torch.device | None = ...) -> Carry:
        """A zeroed carry for ``batch_size`` independent streams."""
        ...

    def observe_step(
        self, carry: Carry, obs: dict[str, torch.Tensor], prevact: torch.Tensor,
        is_first: torch.Tensor,
    ) -> tuple[Carry, Feat]:
        """One ACTING step. ``is_first`` resets the carry and the incoming action."""
        ...

    def loss(
        self, carry: Carry, obs: dict[str, torch.Tensor], prevact: torch.Tensor,
        scales: dict[str, float],
    ) -> tuple[Carry, Feat, Feat, dict[str, torch.Tensor], dict[str, float]]:
        """One TRAINING pass over a replayed batch.

        Returns ``(carry, entries, feat, losses, metrics)``. Every loss is ``(B, T)``
        before reduction, and its keys must be a subset of ``scales``.
        """
        ...

    def imagine(
        self, carry: Carry, policy: Callable[[Feat], torch.Tensor] | torch.Tensor,
        length: int,
    ) -> tuple[Carry, Feat, torch.Tensor]:
        """Roll the model forward without observations, under ``policy``."""
        ...

    def predict_reward(self, feat_tensor: torch.Tensor) -> outs.Output:
        """Reward prediction at the given features."""
        ...

    def predict_continue(self, feat_tensor: torch.Tensor) -> outs.Output:
        """Continue (non-terminal) prediction at the given features."""
        ...

    def feat2tensor(self, feat: Feat) -> torch.Tensor:
        """Flatten a feature dict into the vector the heads consume."""
        ...


class RSSMWorldModel(nn.Module):
    """DreamerV3's world model: encoder, RSSM, decoder, reward and continue heads.

    This is the faithful port (U02-U05); the class exists to put a named boundary
    around it, not to change it. Behaviour is identical to calling the pieces
    directly.
    """

    def __init__(self, obs_shape: tuple[int, int, int], action_dim: int, cfg: DreamerConfig):
        super().__init__()
        a = cfg.agent
        self.cfg = cfg
        self.entry_keys = ("deter", "stoch")

        self.enc = rssm_.Encoder(
            obs_shape, depth=a.enc.depth, mults=a.enc.mults, kernel=a.enc.kernel,
            act=a.enc.act, norm=a.enc.norm, winit=a.enc.winit,
        )
        self.dyn = rssm_.RSSM(
            action_dim, self.enc.outdim, deter=a.rssm.deter, hidden=a.rssm.hidden,
            stoch=a.rssm.stoch, classes=a.rssm.classes, blocks=a.rssm.blocks,
            imglayers=a.rssm.imglayers, obslayers=a.rssm.obslayers,
            dynlayers=a.rssm.dynlayers, act=a.rssm.act, norm=a.rssm.norm,
            unimix=a.rssm.unimix, outscale=a.rssm.outscale, winit=a.rssm.winit,
            absolute=a.rssm.absolute, free_nats=a.rssm.free_nats,
        )
        self.dec = rssm_.Decoder(
            obs_shape, a.rssm.deter, a.rssm.stoch, a.rssm.classes,
            depth=a.dec.depth, mults=a.dec.mults, kernel=a.dec.kernel,
            units=a.dec.units, act=a.dec.act, norm=a.dec.norm,
            outscale=a.dec.outscale, winit=a.dec.winit, bspace=a.dec.bspace,
        )
        self.feat_dim = self.dyn.feat_dim
        self.rew = heads_.MLPHead(
            self.feat_dim, a.rewhead.output, a.rewhead.layers, a.rewhead.units,
            act=a.rewhead.act, norm=a.rewhead.norm, outscale=a.rewhead.outscale,
            winit=a.rewhead.winit, bins=a.rewhead.bins,
        )
        self.con = heads_.MLPHead(
            self.feat_dim, a.conhead.output, a.conhead.layers, a.conhead.units,
            act=a.conhead.act, norm=a.conhead.norm, outscale=a.conhead.outscale,
            winit=a.conhead.winit,
        )

    # -- the interface -------------------------------------------------
    def initial(self, batch_size: int, device: torch.device | None = None) -> Carry:
        return self.dyn.initial(batch_size, device=device)

    def feat2tensor(self, feat: Feat) -> torch.Tensor:
        return heads_.feat2tensor(feat)

    def observe_step(self, carry, obs, prevact, is_first):
        tokens = self.enc(obs["image"])
        return self.dyn.observe_step(carry, tokens, prevact, is_first)

    def imagine(self, carry, policy, length):
        return self.dyn.imagine(carry, policy, length)

    def predict_reward(self, feat_tensor: torch.Tensor) -> outs.Output:
        return self.rew(feat_tensor)

    def predict_continue(self, feat_tensor: torch.Tensor) -> outs.Output:
        return self.con(feat_tensor)

    def loss(self, carry, obs, prevact, scales):
        tokens = self.enc(obs["image"])
        carry, entries, kl, feat, metrics = self.dyn.loss(
            carry, tokens, prevact, obs["is_first"]
        )
        recon = self.dec(feat)
        a = self.cfg.agent
        wm = losses_.world_model_loss(
            feat, kl, recon, self.rew, self.con, obs, scales,
            horizon=a.horizon, contdisc=a.contdisc, reward_grad=a.reward_grad,
        )
        metrics.update(wm.metrics)
        return carry, entries, feat, wm.terms, metrics

    def loss_keys(self) -> tuple[str, ...]:
        """Which loss keys :meth:`loss` returns, so the agent can size its scales."""
        return ("rew", "con", "image", "dyn", "rep")


def scales_for(cfg: DreamerConfig, model: Any) -> dict[str, float]:
    """The loss-scale table for whatever keys ``model`` actually produces.

    A world model that predicts something else declares different `loss_keys`, and
    the table follows it rather than assuming DreamerV3's five.
    """
    a = cfg.agent.loss_scales
    table = {"rew": a.rew, "con": a.con, "dyn": a.dyn, "rep": a.rep}
    for key in model.loss_keys():
        table.setdefault(key, a.rec)
    return {k: table[k] for k in model.loss_keys()}
