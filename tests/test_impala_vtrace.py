"""Parity tests for playtrain_trainers.impala.vtrace against torchbeast/core/vtrace.

Two layers of verification:
  1. Ported torchbeast unit tests (numpy ground-truth comparisons) running
     against playtrain_trainers.impala.vtrace — locks down the math against an
     independent reference implementation.
  2. Cross-implementation parity tests that import torchbeast.core.vtrace
     directly (from the sibling clone) and assert bit-exact agreement on
     randomized inputs. Strongest possible "matches torchbeast" check;
     skipped automatically if torchbeast isn't importable.

Run:
    uv run pytest tests/test_impala_vtrace.py -v
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from playtrain_trainers.impala import vtrace as ours


# ----------------------------------------------------------------------
# Try to import the upstream torchbeast.core.vtrace for parity checks.
# The clone lives at ./torchbeast/torchbeast/core/vtrace.py; only torch
# is needed (no C++ extensions) so we can sys.path-inject it directly.
# ----------------------------------------------------------------------
_REPO = Path(__file__).resolve().parents[1]
_TB_ROOT = _REPO / "torchbeast"
if _TB_ROOT.is_dir() and str(_TB_ROOT) not in sys.path:
    sys.path.insert(0, str(_TB_ROOT))
try:
    from torchbeast.core import vtrace as theirs  # type: ignore
    HAS_TORCHBEAST = True
except Exception:  # noqa: BLE001
    theirs = None
    HAS_TORCHBEAST = False


needs_torchbeast = pytest.mark.skipif(
    not HAS_TORCHBEAST, reason="torchbeast not importable from ./torchbeast"
)


# ----------------------------------------------------------------------
# Helpers (ported from torchbeast/tests/vtrace_test.py)
# ----------------------------------------------------------------------
def _shaped_arange(*shape):
    return np.arange(np.prod(shape), dtype=np.float32).reshape(*shape)


def _softmax(logits):
    return np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)


def _ground_truth_calculation(
    discounts, log_rhos, rewards, values, bootstrap_value,
    clip_rho_threshold, clip_pg_rho_threshold,
):
    """Closed-form V-trace in numpy (copied verbatim from torchbeast tests).

    v_s = V(x_s)
         + sum_{t=s}^{T-1} gamma^{t-s}
             * prod_{i=s}^{t-1} c_i
             * rho_t (r_t + gamma V(x_{t+1}) - V(x_t))
    """
    vs = []
    seq_len = len(discounts)
    rhos = np.exp(log_rhos)
    cs = np.minimum(rhos, 1.0)
    clipped_rhos = rhos
    if clip_rho_threshold:
        clipped_rhos = np.minimum(rhos, clip_rho_threshold)
    clipped_pg_rhos = rhos
    if clip_pg_rho_threshold:
        clipped_pg_rhos = np.minimum(rhos, clip_pg_rho_threshold)

    values_t_plus_1 = np.concatenate([values, bootstrap_value[None, :]], axis=0)
    for s in range(seq_len):
        v_s = np.copy(values[s])
        for t in range(s, seq_len):
            v_s += (
                np.prod(discounts[s:t], axis=0)
                * np.prod(cs[s:t], axis=0)
                * clipped_rhos[t]
                * (rewards[t] + discounts[t] * values_t_plus_1[t + 1] - values[t])
            )
        vs.append(v_s)
    vs = np.stack(vs, axis=0)
    pg_advantages = clipped_pg_rhos * (
        rewards
        + discounts * np.concatenate([vs[1:], bootstrap_value[None, :]], axis=0)
        - values
    )
    return ours.VTraceReturns(vs=vs, pg_advantages=pg_advantages)


def _assert_allclose(actual, desired):
    np.testing.assert_allclose(actual, desired, rtol=1e-6, atol=1e-5)


# ======================================================================
# Layer 1: ported torchbeast tests, running against our implementation
# ======================================================================

class TestActionLogProbs:
    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_action_log_probs(self, batch_size):
        seq_len, num_actions = 7, 3
        policy_logits = _shaped_arange(seq_len, batch_size, num_actions) + 10
        actions = np.random.randint(
            0, num_actions, size=(seq_len, batch_size), dtype=np.int64
        )

        got = ours.action_log_probs(
            torch.from_numpy(policy_logits), torch.from_numpy(actions)
        )

        action_index_mask = actions[..., None] == np.arange(num_actions)
        truth = np.log(_softmax(policy_logits))[action_index_mask].reshape(
            *policy_logits.shape[:-1]
        )
        _assert_allclose(truth, got)


class TestVtrace:
    @pytest.mark.parametrize("batch_size", [1, 5])
    def test_vtrace_matches_numpy_ground_truth(self, batch_size):
        seq_len = 5
        log_rhos = _shaped_arange(seq_len, batch_size) / (batch_size * seq_len)
        log_rhos = 5 * (log_rhos - 0.5)  # rho ∈ ~[0.08, 12.2)
        kwargs = {
            "log_rhos": log_rhos,
            "discounts": np.array(
                [[0.9 / (b + 1) for b in range(batch_size)] for _ in range(seq_len)],
                dtype=np.float32,
            ),
            "rewards": _shaped_arange(seq_len, batch_size),
            "values": _shaped_arange(seq_len, batch_size) / batch_size,
            "bootstrap_value": _shaped_arange(batch_size) + 1.0,
            "clip_rho_threshold": 3.7,
            "clip_pg_rho_threshold": 2.2,
        }
        truth = _ground_truth_calculation(**kwargs)

        tensors = {k: torch.tensor(v) for k, v in kwargs.items()}
        got = ours.from_importance_weights(**tensors)

        _assert_allclose(truth.vs, got.vs)
        _assert_allclose(truth.pg_advantages, got.pg_advantages)

    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_from_logits_matches_from_importance_weights(self, batch_size):
        seq_len, num_actions = 5, 3
        kwargs = {
            "behavior_policy_logits": _shaped_arange(seq_len, batch_size, num_actions),
            "target_policy_logits": _shaped_arange(seq_len, batch_size, num_actions),
            "actions": np.random.randint(
                0, num_actions - 1, size=(seq_len, batch_size)
            ),
            "discounts": np.array(
                [[0.9 / (b + 1) for b in range(batch_size)] for _ in range(seq_len)],
                dtype=np.float32,
            ),
            "rewards": _shaped_arange(seq_len, batch_size),
            "values": _shaped_arange(seq_len, batch_size) / batch_size,
            "bootstrap_value": _shaped_arange(batch_size) + 1.0,
        }
        tensors = {k: torch.from_numpy(v) for k, v in kwargs.items()}

        from_logits_out = ours.from_logits(
            clip_rho_threshold=None, clip_pg_rho_threshold=None, **tensors
        )

        target_lp = ours.action_log_probs(
            tensors["target_policy_logits"], tensors["actions"]
        )
        behavior_lp = ours.action_log_probs(
            tensors["behavior_policy_logits"], tensors["actions"]
        )
        from_iw = ours.from_importance_weights(
            log_rhos=target_lp - behavior_lp,
            discounts=tensors["discounts"],
            rewards=tensors["rewards"],
            values=tensors["values"],
            bootstrap_value=tensors["bootstrap_value"],
            clip_rho_threshold=None,
            clip_pg_rho_threshold=None,
        )

        _assert_allclose(from_iw.vs, from_logits_out.vs)
        _assert_allclose(from_iw.pg_advantages, from_logits_out.pg_advantages)
        _assert_allclose(behavior_lp, from_logits_out.behavior_action_log_probs)
        _assert_allclose(target_lp, from_logits_out.target_action_log_probs)

    def test_higher_rank_inputs(self):
        T, B = 3, 2
        kwargs = {
            "log_rhos": torch.zeros(T, B, 1),
            "discounts": torch.zeros(T, B, 1),
            "rewards": torch.zeros(T, B, 42),
            "values": torch.zeros(T, B, 42),
            "bootstrap_value": torch.zeros(B, 42),
        }
        out = ours.from_importance_weights(**kwargs)
        assert tuple(out.vs.shape) == (T, B, 42)


# ======================================================================
# Layer 2: cross-implementation parity vs upstream torchbeast
# ======================================================================

@needs_torchbeast
class TestParityWithTorchbeast:
    """Bit-exact agreement on randomized inputs. If any of these fail, our
    port has drifted from the upstream V-trace implementation."""

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 17])
    @pytest.mark.parametrize("clip", [(1.0, 1.0), (None, None), (3.7, 2.2)])
    def test_from_importance_weights_parity(self, seed, clip):
        rng = np.random.default_rng(seed)
        T, B = 8, 4
        log_rhos = rng.uniform(-2.5, 2.5, size=(T, B)).astype(np.float32)
        discounts = rng.uniform(0.0, 0.999, size=(T, B)).astype(np.float32)
        rewards = rng.standard_normal((T, B)).astype(np.float32)
        values = rng.standard_normal((T, B)).astype(np.float32)
        bootstrap_value = rng.standard_normal(B).astype(np.float32)

        kw = {
            "log_rhos": torch.from_numpy(log_rhos),
            "discounts": torch.from_numpy(discounts),
            "rewards": torch.from_numpy(rewards),
            "values": torch.from_numpy(values),
            "bootstrap_value": torch.from_numpy(bootstrap_value),
            "clip_rho_threshold": clip[0],
            "clip_pg_rho_threshold": clip[1],
        }
        o = ours.from_importance_weights(**kw)
        t = theirs.from_importance_weights(**kw)

        torch.testing.assert_close(o.vs, t.vs, rtol=0, atol=0)
        torch.testing.assert_close(o.pg_advantages, t.pg_advantages, rtol=0, atol=0)

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_from_logits_parity(self, seed):
        rng = np.random.default_rng(seed)
        T, B, A = 6, 3, 5
        behavior_logits = rng.standard_normal((T, B, A)).astype(np.float32)
        target_logits = rng.standard_normal((T, B, A)).astype(np.float32)
        actions = rng.integers(0, A, size=(T, B), dtype=np.int64)
        discounts = rng.uniform(0.0, 0.999, size=(T, B)).astype(np.float32)
        rewards = rng.standard_normal((T, B)).astype(np.float32)
        values = rng.standard_normal((T, B)).astype(np.float32)
        bootstrap_value = rng.standard_normal(B).astype(np.float32)

        kw = {
            "behavior_policy_logits": torch.from_numpy(behavior_logits),
            "target_policy_logits": torch.from_numpy(target_logits),
            "actions": torch.from_numpy(actions),
            "discounts": torch.from_numpy(discounts),
            "rewards": torch.from_numpy(rewards),
            "values": torch.from_numpy(values),
            "bootstrap_value": torch.from_numpy(bootstrap_value),
            "clip_rho_threshold": 1.0,
            "clip_pg_rho_threshold": 1.0,
        }
        o = ours.from_logits(**kw)
        t = theirs.from_logits(**kw)

        torch.testing.assert_close(o.vs, t.vs, rtol=0, atol=0)
        torch.testing.assert_close(o.pg_advantages, t.pg_advantages, rtol=0, atol=0)
        torch.testing.assert_close(o.log_rhos, t.log_rhos, rtol=0, atol=0)
        torch.testing.assert_close(
            o.behavior_action_log_probs, t.behavior_action_log_probs, rtol=0, atol=0
        )
        torch.testing.assert_close(
            o.target_action_log_probs, t.target_action_log_probs, rtol=0, atol=0
        )

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_action_log_probs_parity(self, seed):
        rng = np.random.default_rng(seed)
        T, B, A = 7, 4, 6
        logits = rng.standard_normal((T, B, A)).astype(np.float32)
        actions = rng.integers(0, A, size=(T, B), dtype=np.int64)

        o = ours.action_log_probs(torch.from_numpy(logits), torch.from_numpy(actions))
        t = theirs.action_log_probs(torch.from_numpy(logits), torch.from_numpy(actions))

        torch.testing.assert_close(o, t, rtol=0, atol=0)
