"""Parity tests for playtrain_trainers.impala.losses against torchbeast/monobeast.

Two layers:
  1. Closed-form numpy/analytic ground-truth checks ported from
     torchbeast/tests/polybeast_loss_functions_test.py (loss values and
     gradients).
  2. Cross-implementation parity: import torchbeast's monobeast loss
     functions directly and assert bit-exact agreement on random inputs.
     Skipped if torchbeast isn't importable.

Run:
    uv run pytest tests/test_impala_losses.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from playtrain_trainers.impala import losses as ours


# ----------------------------------------------------------------------
# Import upstream torchbeast loss functions from the sibling clone.
# We pull from monobeast.py — polybeast_learner.py has identical math
# but requires the C++ nest extension to import.
# ----------------------------------------------------------------------
_REPO = Path(__file__).resolve().parents[1]
_TB_ROOT = _REPO / "torchbeast"
if _TB_ROOT.is_dir() and str(_TB_ROOT) not in sys.path:
    sys.path.insert(0, str(_TB_ROOT))
# monobeast.py top-imports legacy `gym` (via atari_wrappers) which we don't
# install. Stub both so we can reach the pure-torch loss functions further down.
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


def _softmax(logits):
    return np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)


def _assert_allclose(actual, desired):
    np.testing.assert_allclose(actual, desired, rtol=1e-6, atol=1e-5)


# ======================================================================
# Layer 1: closed-form ground truth (ported from torchbeast tests)
# ======================================================================

class TestBaselineLoss:
    advantages = np.array([1.4, 3.43, 5.2, 0.33])

    def test_value(self):
        truth = 0.5 * np.sum(self.advantages ** 2)
        got = ours.compute_baseline_loss(torch.from_numpy(self.advantages))
        _assert_allclose(truth, got)

    def test_grad(self):
        x = torch.from_numpy(self.advantages).requires_grad_()
        ours.compute_baseline_loss(x).backward()
        # d/dx (0.5 x^2) = x
        _assert_allclose(x.grad, self.advantages)


class TestEntropyLoss:
    logits = np.array([0.0012, 0.321, 0.523, 0.109, 0.416])

    def test_value(self):
        sm = _softmax(self.logits)
        truth = np.sum(sm * np.log(sm))  # = -H, the "loss" sign convention
        got = ours.compute_entropy_loss(torch.from_numpy(self.logits))
        _assert_allclose(truth, got)

    def test_grad(self):
        x = torch.from_numpy(self.logits).requires_grad_()
        ours.compute_entropy_loss(x).backward()
        # d/d_logits (sum sm log sm) computed via chain rule
        sm = _softmax(self.logits)
        sm_grad = np.expand_dims(sm, 0).T * (np.eye(sm.size) - sm)
        expected = np.matmul(
            np.ones_like(self.logits), np.matmul(np.diag(1 + np.log(sm)), sm_grad)
        )
        _assert_allclose(x.grad, expected)


class TestPolicyGradientLoss:
    logits = np.array(
        [
            [[0.206, 0.738, 0.125, 0.484, 0.332],
             [0.168, 0.504, 0.523, 0.496, 0.626],
             [0.236, 0.186, 0.627, 0.441, 0.533]],
            [[0.015, 0.904, 0.583, 0.651, 0.855],
             [0.811, 0.292, 0.061, 0.597, 0.590],
             [0.999, 0.504, 0.464, 0.077, 0.143]],
        ]
    )
    actions = np.array([[3, 0, 1], [4, 2, 2]])
    advantages = np.array([[1.4, 0.31, 0.75], [2.1, 1.5, 0.03]])

    def test_value(self):
        T, B, N = self.logits.shape
        labels = F.one_hot(torch.from_numpy(self.actions), num_classes=N).numpy()
        ce = -labels * np.log(_softmax(self.logits))
        truth = np.sum(ce * self.advantages.reshape(T, B, 1))

        got = ours.compute_policy_gradient_loss(
            torch.from_numpy(self.logits),
            torch.from_numpy(self.actions),
            torch.from_numpy(self.advantages),
        )
        _assert_allclose(truth, got.item())

    def test_grad_through_logits(self):
        T, B, N = self.logits.shape
        x = torch.from_numpy(self.logits).requires_grad_()
        ours.compute_policy_gradient_loss(
            x, torch.from_numpy(self.actions), torch.from_numpy(self.advantages)
        ).backward()
        # d/d_logits CE = (softmax - one_hot), then * advantages broadcast
        labels = F.one_hot(torch.from_numpy(self.actions), num_classes=N).numpy()
        expected = (_softmax(self.logits) - labels) * self.advantages.reshape(T, B, 1)
        _assert_allclose(x.grad, expected)

    def test_grad_does_not_flow_through_advantages(self):
        x = torch.from_numpy(self.logits).requires_grad_()
        a = torch.from_numpy(self.advantages).requires_grad_()
        ours.compute_policy_gradient_loss(x, torch.from_numpy(self.actions), a).backward()
        assert x.grad is not None
        assert a.grad is None  # advantages.detach() inside the loss


# ======================================================================
# Layer 2: cross-implementation parity vs upstream torchbeast
# ======================================================================

@needs_torchbeast
class TestParityWithTorchbeast:
    @pytest.mark.parametrize("seed", [0, 1, 2, 17])
    def test_baseline_loss_parity(self, seed):
        rng = np.random.default_rng(seed)
        adv = torch.from_numpy(rng.standard_normal((6, 4)).astype(np.float32))
        torch.testing.assert_close(
            ours.compute_baseline_loss(adv),
            theirs.compute_baseline_loss(adv),
            rtol=0, atol=0,
        )

    @pytest.mark.parametrize("seed", [0, 1, 2, 17])
    def test_entropy_loss_parity(self, seed):
        rng = np.random.default_rng(seed)
        logits = torch.from_numpy(rng.standard_normal((6, 4, 7)).astype(np.float32))
        torch.testing.assert_close(
            ours.compute_entropy_loss(logits),
            theirs.compute_entropy_loss(logits),
            rtol=0, atol=0,
        )

    @pytest.mark.parametrize("seed", [0, 1, 2, 17])
    def test_policy_gradient_loss_parity(self, seed):
        rng = np.random.default_rng(seed)
        T, B, A = 6, 4, 7
        logits = torch.from_numpy(rng.standard_normal((T, B, A)).astype(np.float32))
        actions = torch.from_numpy(rng.integers(0, A, size=(T, B), dtype=np.int64))
        advantages = torch.from_numpy(rng.standard_normal((T, B)).astype(np.float32))
        torch.testing.assert_close(
            ours.compute_policy_gradient_loss(logits, actions, advantages),
            theirs.compute_policy_gradient_loss(logits, actions, advantages),
            rtol=0, atol=0,
        )
