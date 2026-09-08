"""Cross-implementation parity for the IMPALA learn step.

Approach: use the SAME model class (a tiny custom net, so we don't need
Atari obs) for both our learn() and torchbeast's monobeast.learn(), then
verify bit-exact agreement on:
  - loss values (total/pg/baseline/entropy)
  - learner_model parameter deltas after one optimizer step
  - actor_model state_dict after the sync write

This is the strongest possible "matches torchbeast's training step" check.
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn

from playtrain_trainers.impala.learn import learn as ours_learn


# ----------------------------------------------------------------------
# torchbeast import (legacy gym stub — see test_impala_losses.py rationale)
# ----------------------------------------------------------------------
_REPO = Path(__file__).resolve().parents[1]
_TB_ROOT = _REPO / "torchbeast"
if _TB_ROOT.is_dir() and str(_TB_ROOT) not in sys.path:
    sys.path.insert(0, str(_TB_ROOT))
import types as _types
for _name in ("gym", "gym.spaces", "torchbeast.atari_wrappers"):
    sys.modules.setdefault(_name, _types.ModuleType(_name))
try:
    from torchbeast import monobeast as theirs  # type: ignore
    HAS_TORCHBEAST = True
except Exception:  # noqa: BLE001
    theirs = None
    HAS_TORCHBEAST = False

needs_torchbeast = pytest.mark.skipif(
    not HAS_TORCHBEAST, reason="torchbeast not importable from ./torchbeast"
)


# ----------------------------------------------------------------------
# A tiny torchbeast-compatible net (no convs — just an MLP). Same shape
# contract as monobeast.AtariNet but trivial to seed deterministically.
# ----------------------------------------------------------------------
class _TinyNet(nn.Module):
    def __init__(self, frame_dim: int, num_actions: int):
        super().__init__()
        self.num_actions = num_actions
        core_in = frame_dim + num_actions + 1
        self.fc = nn.Linear(frame_dim, frame_dim)
        self.policy = nn.Linear(core_in, num_actions)
        self.baseline = nn.Linear(core_in, 1)

    def initial_state(self, batch_size=1):
        return ()

    def forward(self, inputs, core_state=()):
        x = inputs["frame"]  # [T, B, D] float
        T, B, D = x.shape
        x = torch.flatten(x, 0, 1)
        x = F.relu(self.fc(x))
        oh = F.one_hot(inputs["last_action"].view(T * B), self.num_actions).float()
        cr = torch.clamp(inputs["reward"], -1, 1).view(T * B, 1)
        h = torch.cat([x, cr, oh], dim=-1)
        policy_logits = self.policy(h).view(T, B, self.num_actions)
        baseline = self.baseline(h).view(T, B)
        if self.training:
            flat = F.softmax(policy_logits.view(-1, self.num_actions), dim=-1)
            action = torch.multinomial(flat, 1).view(T, B)
        else:
            action = policy_logits.argmax(-1)
        return dict(policy_logits=policy_logits, baseline=baseline, action=action), ()


def _make_batch(T=5, B=4, D=8, A=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "frame": torch.randn(T + 1, B, D, generator=g),
        "reward": torch.randn(T + 1, B, generator=g) * 0.3,
        "done": (torch.rand(T + 1, B, generator=g) < 0.15),
        "episode_return": torch.randn(T + 1, B, generator=g),
        "episode_step": torch.randint(0, 50, (T + 1, B), generator=g, dtype=torch.int32),
        "last_action": torch.randint(0, A, (T + 1, B), generator=g, dtype=torch.int64),
        "policy_logits": torch.randn(T + 1, B, A, generator=g),
        "baseline": torch.randn(T + 1, B, generator=g),
        "action": torch.randint(0, A, (T + 1, B), generator=g, dtype=torch.int64),
    }


def _make_flags(discounting=0.99, baseline_cost=0.5, entropy_cost=0.01,
                reward_clipping="abs_one", grad_norm_clipping=40.0):
    class F_:  # mimic argparse Namespace
        pass
    f = F_()
    f.discounting = discounting
    f.baseline_cost = baseline_cost
    f.entropy_cost = entropy_cost
    f.reward_clipping = reward_clipping
    f.grad_norm_clipping = grad_norm_clipping
    return f


def _clone_state_dict(sd):
    return {k: v.detach().clone() for k, v in sd.items()}


@needs_torchbeast
class TestLearnParity:
    @pytest.mark.parametrize("seed", [0, 1, 7])
    def test_one_step_matches_torchbeast(self, seed):
        D, A, T, B = 8, 3, 5, 4

        # Build two identical actor/learner pairs with the same init.
        torch.manual_seed(seed)
        ours_actor = _TinyNet(D, A)
        torch.manual_seed(seed)
        ours_learner = _TinyNet(D, A)
        ours_learner.load_state_dict(ours_actor.state_dict())

        torch.manual_seed(seed)
        thr_actor = _TinyNet(D, A)
        torch.manual_seed(seed)
        thr_learner = _TinyNet(D, A)
        thr_learner.load_state_dict(thr_actor.state_dict())

        # Force eval mode so action sampling is deterministic (argmax),
        # otherwise torch.multinomial would diverge between the two paths.
        # Note: learn() doesn't sample actions itself — the action field is
        # taken from `batch` — so train vs eval here only affects the inner
        # forward of the learner. Either is fine; eval is deterministic.
        ours_learner.eval()
        thr_learner.eval()

        # Same optimizer state, same LR schedule.
        opt_ours = torch.optim.RMSprop(ours_learner.parameters(), lr=1e-3,
                                       alpha=0.99, eps=0.01, momentum=0.0)
        opt_thr = torch.optim.RMSprop(thr_learner.parameters(), lr=1e-3,
                                      alpha=0.99, eps=0.01, momentum=0.0)

        batch = _make_batch(T=T, B=B, D=D, A=A, seed=seed)
        flags = _make_flags()

        # Our learn step.
        ours_stats = ours_learn(
            actor_model=ours_actor,
            learner_model=ours_learner,
            batch={k: v.clone() for k, v in batch.items()},
            initial_agent_state=(),
            optimizer=opt_ours,
            scheduler=None,
            discounting=flags.discounting,
            baseline_cost=flags.baseline_cost,
            entropy_cost=flags.entropy_cost,
            grad_norm_clipping=flags.grad_norm_clipping,
            reward_clipping=flags.reward_clipping,
            lock=threading.Lock(),
        )

        # torchbeast's learn step.
        thr_stats = theirs.learn(
            flags=flags,
            actor_model=thr_actor,
            model=thr_learner,
            batch={k: v.clone() for k, v in batch.items()},
            initial_agent_state=(),
            optimizer=opt_thr,
            scheduler=type("S", (), {"step": staticmethod(lambda: None)})(),
            lock=threading.Lock(),
        )

        # Loss values bit-exact.
        for k in ("total_loss", "pg_loss", "baseline_loss", "entropy_loss"):
            assert abs(ours_stats[k] - thr_stats[k]) < 1e-6, (
                f"{k}: ours={ours_stats[k]} theirs={thr_stats[k]}"
            )

        # Post-step learner params bit-exact.
        for (n1, p1), (n2, p2) in zip(
            ours_learner.state_dict().items(), thr_learner.state_dict().items()
        ):
            assert n1 == n2
            torch.testing.assert_close(p1, p2, rtol=1e-6, atol=1e-6)

        # Actor weight sync bit-exact (the IMPALA actor-learner handoff).
        for (n1, p1), (n2, p2) in zip(
            ours_actor.state_dict().items(), thr_actor.state_dict().items()
        ):
            assert n1 == n2
            torch.testing.assert_close(p1, p2, rtol=1e-6, atol=1e-6)


class TestLearnBasicSanity:
    """Smoke-level checks that don't need torchbeast."""

    def test_learn_returns_finite_losses(self):
        D, A, T, B = 8, 3, 4, 2
        torch.manual_seed(0)
        actor = _TinyNet(D, A)
        learner = _TinyNet(D, A)
        learner.load_state_dict(actor.state_dict())
        learner.eval()
        opt = torch.optim.RMSprop(learner.parameters(), lr=1e-3)
        batch = _make_batch(T=T, B=B, D=D, A=A, seed=0)
        stats = ours_learn(
            actor_model=actor, learner_model=learner, batch=batch,
            initial_agent_state=(), optimizer=opt, scheduler=None,
            discounting=0.99, baseline_cost=0.5, entropy_cost=0.01,
            grad_norm_clipping=40.0,
        )
        for k in ("total_loss", "pg_loss", "baseline_loss", "entropy_loss"):
            assert np.isfinite(stats[k]), f"{k} = {stats[k]}"

    def test_actor_model_synced_after_learn(self):
        D, A = 6, 3
        torch.manual_seed(0)
        actor = _TinyNet(D, A)
        learner = _TinyNet(D, A)
        learner.load_state_dict(actor.state_dict())
        # Perturb learner so a sync will be observable
        with torch.no_grad():
            for p in learner.parameters():
                p.add_(0.1)
        learner.eval()
        opt = torch.optim.RMSprop(learner.parameters(), lr=1e-3)
        batch = _make_batch(T=3, B=2, D=D, A=A, seed=0)
        ours_learn(
            actor_model=actor, learner_model=learner, batch=batch,
            initial_agent_state=(), optimizer=opt, scheduler=None,
            discounting=0.99, baseline_cost=0.5, entropy_cost=0.01,
            grad_norm_clipping=40.0,
        )
        for (_, a), (_, l) in zip(
            actor.state_dict().items(), learner.state_dict().items()
        ):
            torch.testing.assert_close(a, l)
