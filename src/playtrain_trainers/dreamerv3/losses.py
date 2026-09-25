"""World-model losses — a port of ``agent.loss``'s first half and ``rssm.loss``.

The five world-model terms, each shape ``(B, T)`` before any reduction::

    rec    Agg(MSE, 3, sum) of the decoder against `image / 255`
    rew    symexp two-hot cross-entropy against the raw reward
    con    binary cross-entropy against `(~is_terminal) * (1 - 1/horizon)`
    dyn    KL(sg(post) || prior), floored at free_nats
    rep    KL(post || sg(prior)), floored at free_nats

and the total is ``sum(losses[k].mean() * scales[k])``. The ``rec`` scale is spread
over the decoder's output keys (``agent.__init__``: ``scales.update({k: rec for k in
dec_space})``), so on Atari it lands on the single key ``image``.

The imagination and replay-critic terms (``imag_loss``, ``repl_loss``) belong to U06
and are not here; this module is what the world model alone needs.

Three details that are easy to get wrong and are tested:

* ``reward_grad: True`` means the reward loss is NOT detached from the latent
  (``sg(..., skip=True)`` returns its argument unchanged), so the reward head shapes
  the representation. Reading ``sg`` as "always stop" inverts this.
* ``contdisc: True`` scales the continue TARGET by ``1 - 1/horizon``, so the target is
  0.997 rather than 1.0 for a non-terminal step. The discount lives in the continue
  prediction rather than in the return recursion.
* the per-key loss is asserted to be ``(B, T)`` BEFORE the mean, which is what stops a
  loss that accidentally kept a trailing axis from being quietly averaged away.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from playtrain_trainers.dreamerv3 import heads as heads_
from playtrain_trainers.dreamerv3 import outs


def continue_target(is_terminal: torch.Tensor, horizon: int, contdisc: bool = True) -> torch.Tensor:
    """``agent.loss``: ``con = f32(~is_terminal)``, then ``con *= 1 - 1/horizon``.

    At ``horizon: 333`` a non-terminal step is trained towards 0.997, not 1.0. The
    continue head therefore predicts "not terminal AND discount", which is why
    ``imag_loss`` runs with ``disc = 1``: applying the discount again would square it.
    """
    con = (~is_terminal).float()
    if contdisc:
        con = con * (1 - 1 / horizon)
    return con


def image_target(image: torch.Tensor) -> torch.Tensor:
    """``f32(value) / 255`` — matches the decoder's sigmoid output range."""
    assert image.dtype == torch.uint8, image.dtype
    return image.float() / 255


@dataclass
class WorldModelLosses:
    """The five terms, each ``(B, T)``, plus the scaled total and the metrics."""

    terms: dict[str, torch.Tensor]
    total: torch.Tensor
    metrics: dict[str, float]


def scales_for(loss_scales: Any, decoder_keys: tuple[str, ...] = ("image",)) -> dict[str, float]:
    """``agent.__init__``: pop ``rec`` and spread it over the decoder's output keys."""
    scales = {
        "rew": loss_scales.rew,
        "con": loss_scales.con,
        "dyn": loss_scales.dyn,
        "rep": loss_scales.rep,
    }
    for key in decoder_keys:
        scales[key] = loss_scales.rec
    return scales


def world_model_loss(
    repfeat: dict[str, torch.Tensor],
    kl_losses: dict[str, torch.Tensor],
    recon: outs.Output,
    rew_head: torch.nn.Module,
    con_head: torch.nn.Module,
    obs: dict[str, torch.Tensor],
    scales: dict[str, float],
    horizon: int = 333,
    contdisc: bool = True,
    reward_grad: bool = True,
    image_key: str = "image",
) -> WorldModelLosses:
    """``agent.loss``'s world-model half.

    ``kl_losses`` and ``recon`` come from ``RSSM.loss`` and ``Decoder`` respectively,
    so this function owns only the three prediction terms and the composition.
    """
    B, T = obs["is_first"].shape
    feat = heads_.feat2tensor(repfeat)
    losses: dict[str, torch.Tensor] = {}

    # reward_grad: True -> the input is NOT detached (sg(..., skip=True)).
    rew_in = feat if reward_grad else feat.detach()
    losses["rew"] = rew_head(rew_in).loss(obs["reward"].float())

    con = continue_target(obs["is_terminal"], horizon, contdisc)
    losses["con"] = con_head(feat).loss(con)

    losses[image_key] = recon.loss(image_target(obs[image_key]))

    losses["dyn"] = kl_losses["dyn"]
    losses["rep"] = kl_losses["rep"]

    shapes = {k: tuple(v.shape) for k, v in losses.items()}
    assert all(x == (B, T) for x in shapes.values()), ((B, T), shapes)
    assert set(losses) == set(scales), (sorted(losses), sorted(scales))

    total = sum(v.mean() * scales[k] for k, v in losses.items())
    metrics = {f"loss/{k}": float(v.mean().detach()) for k, v in losses.items()}
    return WorldModelLosses(losses, total, metrics)
