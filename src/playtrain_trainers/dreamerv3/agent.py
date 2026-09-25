"""The agent — a port of ``dreamerv3/agent.py``'s ``Agent`` class.

Wires the pieces from U03-U06 into one module with three entry points:

``policy(carry, obs)``   one acting step; returns the sampled action and the latent
                         entries that go into the replay alongside the transition
``loss(carry, obs, prevact)``  the full training loss: world model, imagination,
                         actor-critic and the replay critic
``train(carry, data)``   one gradient step over a replayed batch, including the
                         replay-context split, the slow-critic EMA and the entries to
                         write back

``opt_modules`` (upstream's ``agent.modules``) is ``[dyn, enc, dec, rew, con, pol, val]`` and ONE optimizer spans
all of them (``ac_grads: False`` keeps the actor-critic gradient out of the world
model by detaching the imagined features, not by splitting the optimizer).
``slowval`` is an EMA copy and is deliberately not optimized.
"""

from __future__ import annotations

import copy
from typing import Any

import torch
from torch import nn

from playtrain_trainers.dreamerv3 import ac as ac_
from playtrain_trainers.dreamerv3 import heads as heads_
from playtrain_trainers.dreamerv3 import nets as nn_
from playtrain_trainers.dreamerv3 import opt as opt_
from playtrain_trainers.dreamerv3 import world_model as wm_
from playtrain_trainers.dreamerv3.config import DreamerConfig


class Agent(nn.Module):
    """``agent.Agent`` for a single discrete action space and one image key."""

    def __init__(
        self,
        obs_shape: tuple[int, int, int],
        action_dim: int,
        cfg: DreamerConfig,
        world_model: wm_.WorldModel | None = None,
    ):
        super().__init__()
        # `jax.compute_dtype` (D-012): the official code casts explicitly at fixed sites,
        # so the dtype is a property of the model build, not of a framework autocast.
        nn_.set_compute_dtype(cfg.compute_dtype)
        self.cfg = cfg
        self.action_dim = action_dim
        self.obs_shape = obs_shape
        a = cfg.agent

        # MISSION deliverable 2: everything DreamerV3-specific about the world model
        # lives behind `WorldModel`. Swapping what it predicts, or how it factors its
        # latent, means another class with those seven methods and this one line.
        self.wm = world_model if world_model is not None else wm_.RSSMWorldModel(
            obs_shape, action_dim, cfg
        )
        feat = self.wm.feat_dim  # width the actor and critic are built against

        # The actor-critic depends ONLY on `feat_dim` and the reward/continue
        # predictions -- nothing about how the world model produced them.
        self.pol = heads_.MLPHead(
            feat, a.policy_dist_disc, a.policy.layers, a.policy.units, act=a.policy.act,
            norm=a.policy.norm, outscale=a.policy.outscale, winit=a.policy.winit,
            classes=action_dim, unimix=a.policy.unimix,
        )
        self.val = heads_.MLPHead(
            feat, a.value.output, a.value.layers, a.value.units, act=a.value.act,
            norm=a.value.norm, outscale=a.value.outscale, winit=a.value.winit,
            bins=a.value.bins,
        )
        self.slowval = ac_.SlowModel(self.val, rate=a.slowvalue.rate, every=a.slowvalue.every)

        def mk(c):
            return ac_.Normalize(c.impl, c.rate, c.limit, c.perclo, c.perchi, c.debias)

        self.retnorm = mk(a.retnorm)
        self.valnorm = mk(a.valnorm)
        self.advnorm = mk(a.advnorm)

        # Named `opt_modules`, not `modules`: `nn.Module.modules()` is a method.
        self.opt_modules = nn.ModuleList([self.wm, self.pol, self.val])
        self.scales = wm_.scales_for(cfg, self.wm)
        self.opt = opt_.LaProp(
            self.opt_modules.parameters(), lr=a.opt.lr, agc=a.opt.agc, eps=a.opt.eps,
            beta1=a.opt.beta1, beta2=a.opt.beta2, momentum=a.opt.momentum, wd=a.opt.wd,
            warmup=a.opt.warmup, schedule=a.opt.schedule, anneal=a.opt.anneal,
        )

        # The official agent acts with its own copy of the policy-key params
        # (`^(enc|dyn|dec|pol)/`, embodied/jax/agent.py:160-170) and refreshes it LAZILY:
        # `train()` stashes those params as they were BEFORE the update (only if no stash
        # is pending), and `policy()` computes its action with the current copy and only
        # THEN swaps the stash in (agent.py:240-250, 276-282). Acting therefore lags
        # training by about one update. Kept off the module tree on purpose: not in the
        # state_dict, not seen by the optimizer, not moved by `.to()` (built lazily on the
        # agent's device at the first `policy()` call, before any update, as upstream
        # copies `params` at init).
        object.__setattr__(self, "_acting", None)
        object.__setattr__(self, "_pending", None)

    # ------------------------------------------------------------------
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def init_policy(self, batch_size: int) -> dict[str, Any]:
        dev = self.device()
        return {
            "dyn": self.wm.initial(batch_size, device=dev),
            "prevact": torch.zeros(batch_size, self.action_dim, device=dev),
        }

    def onehot(self, index: torch.Tensor) -> torch.Tensor:
        return nn.functional.one_hot(index.long(), self.action_dim).float()

    @torch.no_grad()
    def policy(
        self, carry: dict[str, Any], obs: dict[str, torch.Tensor], mode: str = "train"
    ) -> tuple[dict[str, Any], torch.Tensor, dict[str, torch.Tensor]]:
        """One acting step. Always SAMPLES in train mode (``sample(policy)``).

        Uses the acting copy of the params, then applies a pending sync -- in that order,
        as the official ``policy()`` does.
        """
        wm, pol = self._acting_modules()
        dyn_carry, feat = wm.observe_step(
            carry["dyn"], obs, carry["prevact"], obs["is_first"]
        )
        dist = pol(wm.feat2tensor(feat))
        act = dist.pred() if mode == "eval" else dist.sample()
        carry = {"dyn": dyn_carry, "prevact": self.onehot(act)}
        # The entries stored alongside the transition, so a replayed window can start
        # from the latent the model had at collection time (replay_context).
        entries = {k: feat[k] for k in wm.entry_keys}
        if self._pending is not None:
            wm.load_state_dict(self._pending[0])
            pol.load_state_dict(self._pending[1])
            object.__setattr__(self, "_pending", None)
        return carry, act, entries

    def _acting_modules(self):
        if self._acting is None:
            acting = (copy.deepcopy(self.wm), copy.deepcopy(self.pol))
            for m in acting:
                m.requires_grad_(False)
            object.__setattr__(self, "_acting", acting)
        return self._acting

    # ------------------------------------------------------------------
    def loss(
        self, carry: dict[str, Any], obs: dict[str, torch.Tensor], prevact: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """``agent.loss``: world model, then imagination, then the replay critic."""
        cfg = self.cfg.agent
        B, T = obs["is_first"].shape

        dyn_carry, entries, repfeat, wm_terms, metrics = self.wm.loss(
            carry["dyn"], obs, prevact, self.scales
        )
        losses = dict(wm_terms)

        # Imagination: imag_last 0 -> K = T, every replayed step starts a rollout.
        K = min(cfg.imag_last or T, T)
        H = cfg.imag_length
        starts = ac_.imagination_starts(entries, K)

        def policyfn(feat):
            return self.onehot(self.pol(self.wm.feat2tensor(feat)).sample())

        _, imgfeat, imgprevact = self.wm.imagine(starts, policyfn, H)
        imgfeat = ac_.assemble_imagined(repfeat, imgfeat, K, ac_grads=cfg.ac_grads)
        inp = self.wm.feat2tensor(imgfeat)
        lastact = policyfn({k: v[:, -1] for k, v in imgfeat.items()}).unsqueeze(1)
        imgact = torch.cat([imgprevact, lastact], 1).argmax(-1)

        il, iout, imets = ac_.imag_loss(
            imgact,
            self.wm.predict_reward(inp).pred(),
            self.wm.predict_continue(inp).prob1(),
            self.pol(inp),
            self.val(inp),
            self.slowval(inp),
            self.retnorm, self.valnorm, self.advnorm,
            update=self.training, contdisc=cfg.contdisc, horizon=cfg.horizon,
            slowtar=cfg.imag_loss.slowtar, lam=cfg.imag_loss.lam,
            actent=cfg.imag_loss.actent, slowreg=cfg.imag_loss.slowreg,
        )
        losses.update(ac_.reduce_imagination_losses(il, B, K))
        metrics.update(imets)

        # Replay critic: real rewards, bootstrapped by the imagined return.
        if cfg.repval_loss:
            feat = repfeat if cfg.repval_grad else {k: v.detach() for k, v in repfeat.items()}
            boot = iout["ret"][:, 0].reshape(B, K)
            feat = {k: v[:, -K:] for k, v in feat.items()}
            last = obs["is_last"][:, -K:]
            term = obs["is_terminal"][:, -K:].float()
            rew = obs["reward"][:, -K:].float()
            rinp = self.wm.feat2tensor(feat)
            rl, _, _ = ac_.repl_loss(
                last, term, rew, boot, self.val(rinp), self.slowval(rinp), self.valnorm,
                update=self.training, horizon=cfg.horizon,
                slowtar=cfg.repl_loss.slowtar, lam=cfg.repl_loss.lam,
                slowreg=cfg.repl_loss.slowreg,
            )
            losses.update(rl)

        scales = dict(self.scales)
        scales.update({"policy": cfg.loss_scales.policy, "value": cfg.loss_scales.value})
        if cfg.repval_loss:
            scales["repval"] = cfg.loss_scales.repval
        assert set(losses) == set(scales), (sorted(losses), sorted(scales))
        total = sum(v.mean() * scales[k] for k, v in losses.items())
        metrics.update({f"loss/{k}": float(v.mean().detach()) for k, v in losses.items()})
        aux = {
            "carry": {"dyn": dyn_carry},
            "entries": entries,
            "repfeat": repfeat,
            "metrics": metrics,
        }
        return total, aux

    # ------------------------------------------------------------------
    def train_step(
        self, carry: dict[str, Any], data: dict[str, torch.Tensor]
    ) -> tuple[dict[str, Any], dict[str, torch.Tensor], dict[str, float]]:
        """``agent.train``: one gradient step over a replayed batch.

        The replay-context split happens first, so the carry comes from the stored
        latent rather than from a zeroed state, and the 64 trained steps are paired
        with the actions that produced them.
        """
        K = self.cfg.replay_context
        ctx_carry = {k: data[k][:, K - 1] for k in self.wm.entry_keys}
        obs = {
            "image": data["image"][:, K:],
            "reward": data["reward"][:, K:],
            "is_first": data["is_first"][:, K:],
            "is_last": data["is_last"][:, K:],
            "is_terminal": data["is_terminal"][:, K:],
        }
        prevact = self.onehot(data["action"][:, K - 1 : -1])
        stepid = data["stepid"][:, K:]

        if self._pending is None:
            # `allo`, taken before `_train` runs: the policy keys as they were BEFORE
            # this update (embodied/jax/agent.py:268, 276-282).
            object.__setattr__(self, "_pending", (
                {k: v.detach().clone() for k, v in self.wm.state_dict().items()},
                {k: v.detach().clone() for k, v in self.pol.state_dict().items()},
            ))
        self.opt.zero_grad(set_to_none=True)
        total, aux = self.loss({"dyn": ctx_carry}, obs, prevact)
        total.backward()
        metrics = dict(aux["metrics"])
        metrics["grad_norm"] = opt_.global_norm(self.opt_modules.parameters())
        metrics["lr"] = self.opt.current_lr()
        metrics["loss"] = float(total.detach())
        self.opt.step()
        self.slowval.update()

        entries = aux["entries"]
        updates = {"stepid": stepid, **{k: v.detach() for k, v in entries.items()}}
        B, T = obs["is_first"].shape
        assert all(v.shape[:2] == (B, T) for v in updates.values()), {
            k: tuple(v.shape) for k, v in updates.items()
        }
        carry = {"dyn": aux["carry"]["dyn"]}
        return carry, updates, metrics
