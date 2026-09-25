"""Prediction heads — a port of ``embodied/jax/heads.py``.

Ported: ``MLPHead`` and the three ``Head`` output types the frozen config reaches
(``symexp_twohot`` for the reward and the critic, ``binary`` for the continue head,
``categorical`` for a discrete policy). ``onehot``, ``mse``, ``huber``,
``symlog_mse``, ``bounded_normal``, ``normal_logstd`` and ``DictHead`` are not
ported -- D-016 lists them.

``feat2tensor`` (``agent.py``) is the input: ``concat([deter, stoch.flatten()], -1)``,
in that order.
"""

from __future__ import annotations

import torch
from torch import nn

from playtrain_trainers.dreamerv3 import nets as nn_
from playtrain_trainers.dreamerv3 import outs


def feat2tensor(feat: dict[str, torch.Tensor]) -> torch.Tensor:
    """``agent.py``'s ``feat2tensor``: deter first, then the flattened latent."""
    # agent.py:51-53, both halves through `nn.cast`.
    stoch = nn_.cast(feat["stoch"])
    return torch.cat([nn_.cast(feat["deter"]), stoch.reshape(*stoch.shape[:-2], -1)], -1)


class MLPHead(nn.Module):
    """``heads.MLPHead``: an ``MLP`` trunk and one output layer.

    ``outscale`` applies to the OUTPUT layer only. The frozen config sets it to 0.0
    for the reward head and the critic, so both predict exactly zero at
    initialization (which is also why ``TwoHot.pred``'s symmetric sum matters), and
    to 1.0 for the continue head and 0.01 for the policy.
    """

    def __init__(
        self,
        insize: int,
        output: str,
        layers: int = 1,
        units: int = 1024,
        act: str = "silu",
        norm: str = "rms",
        outscale: float = 1.0,
        winit: str = "trunc_normal_in",
        bins: int = 255,
        shape: tuple[int, ...] = (),
        classes: int | None = None,
        unimix: float = 0.0,
        generator: torch.Generator | None = None,
    ):
        super().__init__()
        self.impl = output
        self.bins = bins
        self.shape = tuple(shape)
        self.classes = classes
        # Kept so the head still reports what the config asked for, but deliberately
        # NOT applied to the categorical output; see D-027 in `forward`.
        self.unimix = unimix
        self.mlp = nn_.MLP(
            insize, layers, units, act_name=act, norm=norm, winit=winit, generator=generator
        )
        kw = dict(winit=winit, outscale=outscale, generator=generator)
        if output == "symexp_twohot":
            self.out = nn_.Linear(units, (*self.shape, bins), **kw)
            self.register_buffer("bin_values", outs.twohot_bins(bins), persistent=False)
        elif output == "binary":
            self.out = nn_.Linear(units, self.shape or 1, **kw)
        elif output == "categorical":
            assert classes is not None
            self.out = nn_.Linear(units, (*self.shape, classes), **kw)
        else:
            raise NotImplementedError(output)

    def forward(self, x: torch.Tensor) -> outs.Output:
        x = self.mlp(x)
        x = self.out(x)
        if self.impl == "symexp_twohot":
            out: outs.Output = outs.TwoHot(x, self.bin_values)
        elif self.impl == "binary":
            # The space is scalar, so the trailing size-1 axis is squeezed away; the
            # official Linear is given `space.shape == ()` and produces no axis.
            out = outs.Binary(x.squeeze(-1) if not self.shape else x)
        else:
            # D-027: NO unimix here. The official `Head.categorical` builds
            # `outs.Categorical(logits)` with no unimix argument; `Head.unimix` is
            # consumed only by `Head.onehot`, which a discrete policy
            # (`policy_dist_disc: categorical`) never reaches. So `agent.policy.unimix:
            # 0.01` is dead in the pinned code path, even though the paper's
            # hyperparameter table lists "Actor unimix 1%". PROTOCOL section 3: the code
            # wins. The latent's unimix is a different thing and IS applied
            # (`rssm._dist` -> `OneHot(logits, unimix)`).
            out = outs.Categorical(x)
        if self.shape:
            out = outs.Agg(out, len(self.shape), torch.sum)
        return out
