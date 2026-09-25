"""Encoder, RSSM and Decoder — a port of ``dreamerv3/rssm.py``.

Only the image path is ported: the frozen atari100k config has one observation key
(``image``, uint8 64x64x3) and no vector keys, so the encoder's MLP branch, the
decoder's ``DictHead`` branch and ``DictConcat``'s discrete/masking machinery are
unreachable. D-016 records each omission.

Tensor layout follows the original: images are NHWC and the batch/time axes are
flattened before the convolutions, not carried through them.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from playtrain_trainers.dreamerv3 import nets as nn_
from playtrain_trainers.dreamerv3 import outs


class Encoder(nn.Module):
    """``rssm.Encoder``, image branch.

    Per stage: a stride-1 ``kernel``x``kernel`` convolution, then a 2x2 MAX-pool done
    as a reshape (``strided: False``), then norm, then the activation. At 64 px and
    ``mults (2, 3, 4, 4)`` the spatial size goes 64 -> 32 -> 16 -> 8 -> 4 and the
    channels go 128 -> 192 -> 256 -> 256, so the token vector is 4*4*256 = 4096.
    """

    def __init__(
        self,
        obs_shape: tuple[int, int, int],  # HWC
        depth: int = 64,
        mults: tuple[int, ...] = (2, 3, 4, 4),
        kernel: int = 5,
        act: str = "silu",
        norm: str = "rms",
        winit: str = "trunc_normal_in",
        outer: bool = False,
        strided: bool = False,
        generator: torch.Generator | None = None,
    ):
        super().__init__()
        if outer or strided:
            raise NotImplementedError("the frozen config has outer=False, strided=False")
        self.obs_shape = obs_shape
        self.act = nn_.act(act)
        self.depths = tuple(depth * m for m in mults)
        convs, norms = [], []
        insize = obs_shape[-1]
        for d in self.depths:
            convs.append(nn_.Conv2D(insize, d, kernel, winit=winit, generator=generator))
            norms.append(nn_.Norm(norm, d))
            insize = d
        self.convs = nn.ModuleList(convs)
        self.norms = nn.ModuleList(norms)
        h, w = obs_shape[0], obs_shape[1]
        for _ in self.depths:
            h, w = h // 2, w // 2
        assert 3 <= h <= 16 and 3 <= w <= 16, (h, w)
        self.minres = (h, w)
        self.outdim = h * w * self.depths[-1]

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """``image``: uint8 (..., H, W, C). Returns tokens (..., outdim)."""
        assert image.dtype == torch.uint8, image.dtype
        bshape = image.shape[:-3]
        # rssm.py:230, `nn.cast(imgs, force=True) / 255 - 0.5`: the scaling itself runs in
        # the compute dtype.
        x = nn_.cast(image, force=True) / 255 - 0.5
        x = x.reshape(-1, *image.shape[-3:])
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x)
            B, H, W, C = x.shape
            # The 2x2 max-pool, written as the original writes it.
            x = x.reshape(B, H // 2, 2, W // 2, 2, C).amax((2, 4))
            x = self.act(norm(x))
        x = x.reshape(x.shape[0], -1)
        return x.reshape(*bshape, x.shape[-1])


class RSSM(nn.Module):
    """``rssm.RSSM``: the block-diagonal GRU over ``deter``, plus prior and posterior.

    The recurrence (``_core``) is not a standard GRU. Three separate
    Linear->Norm->act projections of ``deter``, ``stoch`` and the action are
    concatenated, broadcast to every block, concatenated with the block-split
    ``deter`` itself, run through ``dynlayers`` BlockLinears and finally a BlockLinear
    to ``3 * deter`` gates. The gates are then::

        reset  = sigmoid(reset)
        cand   = tanh(reset * cand)
        update = sigmoid(update - 1)
        deter  = update * cand + (1 - update) * deter

    Three things differ from the textbook GRU and from ``nets.GRU`` in the same
    repository, and all three are copied: the candidate is ``tanh(reset * cand)``
    rather than ``tanh(W [reset * h, x])``, the update bias is folded in as
    ``update - 1`` (so the gate starts closed and the state is sticky), and the action
    is scaled by ``action / sg(max(1, |action|))`` before it is projected.
    """

    def __init__(
        self,
        action_dim: int,
        token_dim: int,
        deter: int = 8192,
        hidden: int = 1024,
        stoch: int = 32,
        classes: int = 64,
        blocks: int = 8,
        imglayers: int = 2,
        obslayers: int = 1,
        dynlayers: int = 1,
        act: str = "silu",
        norm: str = "rms",
        unimix: float = 0.01,
        outscale: float = 1.0,
        winit: str = "trunc_normal_in",
        absolute: bool = False,
        free_nats: float = 1.0,
        generator: torch.Generator | None = None,
    ):
        super().__init__()
        assert deter % blocks == 0
        self.deter = deter
        self.hidden = hidden
        self.stoch = stoch
        self.classes = classes
        self.blocks = blocks
        self.unimix = unimix
        self.absolute = absolute
        self.free_nats = free_nats
        self.action_dim = action_dim
        self.token_dim = token_dim
        self.act = nn_.act(act)
        kw = dict(winit=winit, generator=generator)

        # _core
        self.dynin0 = nn_.Linear(deter, hidden, **kw)
        self.dynin0norm = nn_.Norm(norm, hidden)
        self.dynin1 = nn_.Linear(stoch * classes, hidden, **kw)
        self.dynin1norm = nn_.Norm(norm, hidden)
        self.dynin2 = nn_.Linear(action_dim, hidden, **kw)
        self.dynin2norm = nn_.Norm(norm, hidden)
        # Per block: deter/blocks of the state, plus all three projections.
        core_in = deter + 3 * hidden * blocks
        dynhid, dynhidnorm = [], []
        size = core_in
        for _ in range(dynlayers):
            dynhid.append(nn_.BlockLinear(size, deter, blocks, **kw))
            dynhidnorm.append(nn_.Norm(norm, deter))
            size = deter
        self.dynhid = nn.ModuleList(dynhid)
        self.dynhidnorm = nn.ModuleList(dynhidnorm)
        self.dyngru = nn_.BlockLinear(size, 3 * deter, blocks, **kw)

        # posterior
        obs_in = token_dim if absolute else deter + token_dim
        obs, obsnorm = [], []
        size = obs_in
        for _ in range(obslayers):
            obs.append(nn_.Linear(size, hidden, **kw))
            obsnorm.append(nn_.Norm(norm, hidden))
            size = hidden
        self.obs = nn.ModuleList(obs)
        self.obsnorm = nn.ModuleList(obsnorm)
        self.obslogit = nn_.Linear(size, stoch * classes, outscale=outscale, **kw)

        # prior
        prior, priornorm = [], []
        size = deter
        for _ in range(imglayers):
            prior.append(nn_.Linear(size, hidden, **kw))
            priornorm.append(nn_.Norm(norm, hidden))
            size = hidden
        self.prior = nn.ModuleList(prior)
        self.priornorm = nn.ModuleList(priornorm)
        self.priorlogit = nn_.Linear(size, stoch * classes, outscale=outscale, **kw)

    # ------------------------------------------------------------------
    @property
    def feat_dim(self) -> int:
        return self.deter + self.stoch * self.classes

    def initial(self, bsize: int, device=None) -> dict[str, torch.Tensor]:
        # rssm.py:46, `nn.cast(dict(deter=zeros(f32), stoch=zeros(f32)))`.
        return nn_.cast({
            "deter": torch.zeros(bsize, self.deter, device=device, dtype=torch.float32),
            "stoch": torch.zeros(bsize, self.stoch, self.classes, device=device, dtype=torch.float32),
        })

    def truncate(self, entries: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        assert entries["deter"].ndim == 3, entries["deter"].shape
        return {k: v[:, -1] for k, v in entries.items()}

    def starts(self, entries: dict[str, torch.Tensor], nlast: int) -> dict[str, torch.Tensor]:
        B = entries["deter"].shape[0]
        return {k: v[:, -nlast:].reshape(B * nlast, *v.shape[2:]) for k, v in entries.items()}

    # ------------------------------------------------------------------
    def _core(
        self, deter: torch.Tensor, stoch: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        stoch = stoch.reshape(stoch.shape[0], -1)
        # The action is squashed by its own magnitude, with the denominator detached.
        # For a one-hot action this is the identity; it exists for continuous spaces.
        action = action / torch.clamp(action.abs(), min=1.0).detach()
        g = self.blocks
        x0 = self.act(self.dynin0norm(self.dynin0(deter)))
        x1 = self.act(self.dynin1norm(self.dynin1(stoch)))
        x2 = self.act(self.dynin2norm(self.dynin2(action)))
        x = torch.cat([x0, x1, x2], -1)
        # (..., 3H) -> (..., g, 3H): the same projections go to every block.
        x = x.unsqueeze(-2).expand(*x.shape[:-1], g, x.shape[-1])
        dg = deter.reshape(*deter.shape[:-1], g, deter.shape[-1] // g)
        x = torch.cat([dg, x], -1).reshape(*deter.shape[:-1], -1)
        for lin, norm in zip(self.dynhid, self.dynhidnorm):
            x = self.act(norm(lin(x)))
        x = self.dyngru(x)
        # The split is PER BLOCK: reshape to (g, 3 * deter/g), split, flatten back.
        xg = x.reshape(*x.shape[:-1], g, 3 * self.deter // g)
        reset, cand, update = torch.chunk(xg, 3, -1)
        def flat(t):
            return t.reshape(*t.shape[:-2], -1)

        reset, cand, update = flat(reset), flat(cand), flat(update)
        reset = torch.sigmoid(reset)
        cand = torch.tanh(reset * cand)
        update = torch.sigmoid(update - 1)
        return update * cand + (1 - update) * deter

    def _prior_logit(self, deter: torch.Tensor) -> torch.Tensor:
        x = deter
        for lin, norm in zip(self.prior, self.priornorm):
            x = self.act(norm(lin(x)))
        return self._reshape_logit(self.priorlogit(x))

    def _post_logit(self, deter: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        x = tokens if self.absolute else torch.cat([deter, tokens], -1)
        for lin, norm in zip(self.obs, self.obsnorm):
            x = self.act(norm(lin(x)))
        return self._reshape_logit(self.obslogit(x))

    def _reshape_logit(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(*x.shape[:-1], self.stoch, self.classes)

    def dist(self, logits: torch.Tensor) -> outs.Agg:
        """``_dist``: unimixed one-hots, aggregated over the stoch axis by SUM."""
        return outs.Agg(outs.OneHot(logits, self.unimix), 1, torch.sum)

    # ------------------------------------------------------------------
    def observe_step(
        self,
        carry: dict[str, torch.Tensor],
        tokens: torch.Tensor,
        action: torch.Tensor,
        reset: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """One posterior step. ``reset`` zeroes the carry AND the incoming action."""
        # rssm.py:62, `carry, tokens, action = nn.cast((carry, tokens, action))`.
        carry, tokens, action = nn_.cast((carry, tokens, action))
        keep = (~reset).to(tokens.dtype).reshape(-1, 1)
        deter = carry["deter"] * keep
        stoch = carry["stoch"] * keep.reshape(-1, 1, 1)
        action = action * keep
        deter = self._core(deter, stoch, action)
        tokens = tokens.reshape(*deter.shape[:-1], -1)
        logit = self._post_logit(deter, tokens)
        stoch = nn_.cast(self.dist(logit).sample(generator))  # rssm.py:87
        carry = {"deter": deter, "stoch": stoch}
        feat = {"deter": deter, "stoch": stoch, "logit": logit}
        return carry, feat

    def observe(
        self,
        carry: dict[str, torch.Tensor],
        tokens: torch.Tensor,
        action: torch.Tensor,
        reset: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Scan ``observe_step`` over the time axis (axis 1), as ``nj.scan`` does."""
        T = tokens.shape[1]
        feats: list[dict[str, torch.Tensor]] = []
        for t in range(T):
            carry, feat = self.observe_step(
                carry, tokens[:, t], action[:, t], reset[:, t], generator
            )
            feats.append(feat)
        stacked = {k: torch.stack([f[k] for f in feats], 1) for k in feats[0]}
        entries = {"deter": stacked["deter"], "stoch": stacked["stoch"]}
        return carry, entries, stacked

    def imagine_step(
        self,
        carry: dict[str, torch.Tensor],
        action: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """One prior step: no tokens, no reset, latent sampled from the prior."""
        # rssm.py:95, the action embedding comes from `DictConcat`, whose discrete branch
        # is `one_hot(..., dtype=COMPUTE_DTYPE)`. The returned action keeps its own dtype.
        deter = self._core(carry["deter"], carry["stoch"], nn_.cast(action))
        logit = self._prior_logit(deter)
        stoch = nn_.cast(self.dist(logit).sample(generator))  # rssm.py:100
        carry = nn_.cast({"deter": deter, "stoch": stoch})  # rssm.py:101
        feat = nn_.cast({"deter": deter, "stoch": stoch, "logit": logit})  # rssm.py:102
        return carry, feat

    def imagine(
        self,
        carry: dict[str, torch.Tensor],
        policy,
        length: int,
        generator: torch.Generator | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]:
        """Roll the prior forward ``length`` steps under ``policy``.

        ``policy`` is called on a DETACHED carry, exactly as ``imagine`` does
        (``policy(sg(carry))``): the actor's gradient comes from the return, not from
        differentiating the rollout back into the state it started from.
        """
        # rssm.py:110/114, the scan starts from `nn.cast(carry)` and a tensor policy is
        # `nn.cast(policy)`.
        carry = nn_.cast(carry)
        if not callable(policy):
            policy = nn_.cast(policy)
        feats, actions = [], []
        for t in range(length):
            if callable(policy):
                action = policy({k: v.detach() for k, v in carry.items()})
            else:
                # D-028: a TENSOR policy is a per-step action sequence, and upstream
                # scans it along the time axis (`nj.scan(..., policy, axis=1)`), so
                # step t uses `policy[:, t]`. Passing the whole array every step --
                # which this branch used to do -- is only invisible because training
                # always supplies a callable; the open-loop report is the one caller
                # that feeds recorded actions.
                action = policy[:, t]
            carry, feat = self.imagine_step(carry, action, generator)
            feats.append(feat)
            actions.append(action)
        stacked = {k: torch.stack([f[k] for f in feats], 1) for k in feats[0]}
        return carry, stacked, torch.stack(actions, 1)

    def loss(
        self,
        carry: dict[str, torch.Tensor],
        tokens: torch.Tensor,
        action: torch.Tensor,
        reset: torch.Tensor,
        generator: torch.Generator | None = None,
    ):
        """``rssm.loss``: the dyn/rep KL pair with free nats, plus the entropies.

        ``dyn = KL(sg(post) || prior)`` trains the prior towards the posterior;
        ``rep = KL(post || sg(prior))`` trains the posterior towards the prior. Each
        is clipped BELOW at ``free_nats`` per step -- ``max(kl, 1.0)``, so the loss is
        flat until the KL exceeds one nat, and the clip applies to the aggregated
        32-categorical KL, not per categorical.
        """
        carry, entries, feat = self.observe(carry, tokens, action, reset, generator)
        prior = self._prior_logit(feat["deter"])
        post = feat["logit"]
        dyn = self.dist(post.detach()).kl(self.dist(prior))
        rep = self.dist(post).kl(self.dist(prior.detach()))
        if self.free_nats:
            dyn = torch.clamp(dyn, min=self.free_nats)
            rep = torch.clamp(rep, min=self.free_nats)
        losses = {"dyn": dyn, "rep": rep}
        metrics = {
            "dyn_ent": self.dist(prior).entropy().mean(),
            "rep_ent": self.dist(post).entropy().mean(),
        }
        return carry, entries, losses, feat, metrics


class Decoder(nn.Module):
    """``rssm.Decoder``, image branch with ``bspace: 8``.

    The spatial seed is built from ``deter`` and ``stoch`` on SEPARATE paths and added:
    ``deter`` goes through one BlockLinear straight to the ``minres x minres x C``
    volume (rearranged so each block owns a slice of the channels), ``stoch`` through
    two dense layers. Upsampling is nearest-neighbour (``repeat`` twice) followed by a
    stride-1 convolution, not a transposed convolution. The output passes through a
    sigmoid, so the reconstruction target is the image in [0, 1].
    """

    def __init__(
        self,
        obs_shape: tuple[int, int, int],  # HWC
        deter: int,
        stoch: int,
        classes: int,
        depth: int = 64,
        mults: tuple[int, ...] = (2, 3, 4, 4),
        kernel: int = 5,
        units: int = 1024,
        act: str = "silu",
        norm: str = "rms",
        outscale: float = 1.0,
        winit: str = "trunc_normal_in",
        bspace: int = 8,
        outer: bool = False,
        strided: bool = False,
        generator: torch.Generator | None = None,
    ):
        super().__init__()
        if outer or strided or not bspace:
            raise NotImplementedError("the frozen config has bspace=8, outer/strided False")
        assert deter % bspace == 0
        self.obs_shape = obs_shape
        self.act = nn_.act(act)
        self.depths = tuple(depth * m for m in mults)
        self.imgdep = obs_shape[-1]
        factor = 2 ** len(self.depths)
        self.minres = (obs_shape[0] // factor, obs_shape[1] // factor)
        assert 3 <= self.minres[0] <= 16 and 3 <= self.minres[1] <= 16, self.minres
        shape = (*self.minres, self.depths[-1])
        u = math.prod(shape)
        kw = dict(winit=winit, generator=generator)
        self.bspace = bspace
        self.sp0 = nn_.BlockLinear(deter, u, bspace, **kw)
        self.sp1 = nn_.Linear(stoch * classes, 2 * units, **kw)
        self.sp1norm = nn_.Norm(norm, 2 * units)
        self.sp2 = nn_.Linear(2 * units, shape, **kw)
        self.spnorm = nn_.Norm(norm, shape[-1])

        convs, norms = [], []
        insize = self.depths[-1]
        for d in reversed(self.depths[:-1]):
            convs.append(nn_.Conv2D(insize, d, kernel, **kw))
            norms.append(nn_.Norm(norm, d))
            insize = d
        self.convs = nn.ModuleList(convs)
        self.norms = nn.ModuleList(norms)
        self.imgout = nn_.Conv2D(insize, self.imgdep, kernel, outscale=outscale, **kw)

    def forward(self, feat: dict[str, torch.Tensor]) -> outs.Agg:
        # rssm.py:293/316, `nn.cast(feat['stoch'])`, `nn.cast((feat['deter'], feat['stoch']))`.
        deter, stoch = nn_.cast((feat["deter"], feat["stoch"]))
        bshape = deter.shape[:-1]
        g = self.bspace
        h, w = self.minres
        x0 = deter.reshape(-1, deter.shape[-1])
        x1 = stoch.reshape(math.prod(bshape), -1)
        x0 = self.sp0(x0)
        # '... (g h w c) -> ... h w (g c)': each block owns a channel slice.
        c = x0.shape[-1] // (g * h * w)
        x0 = x0.reshape(-1, g, h, w, c).permute(0, 2, 3, 1, 4).reshape(-1, h, w, g * c)
        x1 = self.act(self.sp1norm(self.sp1(x1)))
        x1 = self.sp2(x1)
        x = self.act(self.spnorm(x0 + x1))
        for conv, norm in zip(self.convs, self.norms):
            x = x.repeat_interleave(2, -2).repeat_interleave(2, -3)
            x = self.act(norm(conv(x)))
        x = x.repeat_interleave(2, -2).repeat_interleave(2, -3)
        x = self.imgout(x)
        x = torch.sigmoid(x)
        x = x.reshape(*bshape, *x.shape[1:])
        return outs.Agg(outs.MSE(x), 3, torch.sum)
