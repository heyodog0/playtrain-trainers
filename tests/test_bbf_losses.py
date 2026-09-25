"""Tests for playtrain_trainers.bbf.losses.

The C51 projection is checked against a brute-force reference written
independently below, over random inputs and over the boundary cases where
implementations usually diverge: a target landing exactly on an atom, and
targets clamped past v_min / v_max.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from torch import nn

from playtrain_trainers.bbf.config import BBFConfig
from playtrain_trainers.bbf.losses import (
    augment,
    build_optimizer,
    build_target,
    c51_loss,
    c51_target_distribution,
    ema_update,
    project_distribution,
    random_intensity,
    random_shift,
    select_bootstrap_action,
    spr_loss,
    to_float,
)
from playtrain_trainers.bbf.net import BBFNetwork


# ----------------------------------------------------------------------
# Brute-force C51 projection reference
# ----------------------------------------------------------------------
def brute_force_project(target_support, weights, support):
    """Loop-and-scalar reference for the C51 projection.

    Deliberately written from the algorithm as stated (Bellemare et al. 2017,
    Algorithm 1) rather than by refactoring the implementation.
    """
    ts = np.asarray(target_support, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    z = np.asarray(support, dtype=np.float64)
    n = len(z)
    v_min, v_max = z[0], z[-1]
    delta = (v_max - v_min) / (n - 1)
    # Sized by the grid, not by the target atoms -- the same trap the
    # implementation fell into.
    out = np.zeros((ts.shape[0], n), dtype=np.float64)
    for i in range(ts.shape[0]):
        for j in range(ts.shape[1]):
            tz = min(v_max, max(v_min, ts[i, j]))
            b = (tz - v_min) / delta
            lo = int(math.floor(b))
            hi = int(math.ceil(b))
            lo = min(max(lo, 0), n - 1)
            hi = min(max(hi, 0), n - 1)
            if lo == hi:
                out[i, lo] += w[i, j]
            else:
                out[i, lo] += w[i, j] * (hi - b)
                out[i, hi] += w[i, j] * (b - lo)
    return out


def _support(n=51, vmax=10.0):
    return torch.linspace(-vmax, vmax, n, dtype=torch.float64)


# ----------------------------------------------------------------------
# project_distribution
# ----------------------------------------------------------------------
def test_projection_matches_brute_force_on_random_input():
    rng = np.random.default_rng(0)
    z = _support()
    for _ in range(20):
        ts = torch.tensor(rng.uniform(-15, 15, (7, 51)), dtype=torch.float64)
        w = torch.tensor(rng.dirichlet(np.ones(51), 7), dtype=torch.float64)
        got = project_distribution(ts, w, z).numpy()
        want = brute_force_project(ts, w, z)
        assert np.allclose(got, want, atol=1e-10), np.abs(got - want).max()


def test_projection_preserves_total_mass():
    rng = np.random.default_rng(1)
    z = _support()
    w = torch.tensor(rng.dirichlet(np.ones(51), 5), dtype=torch.float64)
    ts = torch.tensor(rng.uniform(-30, 30, (5, 51)), dtype=torch.float64)
    out = project_distribution(ts, w, z)
    assert torch.allclose(out.sum(1), w.sum(1), atol=1e-10)


def test_projection_when_a_target_lands_exactly_on_an_atom():
    """floor == ceil: the naive split is 0/0 and would delete the mass."""
    z = _support(n=5, vmax=2.0)  # [-2, -1, 0, 1, 2]
    ts = torch.tensor([[0.0, 1.0, -2.0, 2.0, -1.0]], dtype=torch.float64)
    w = torch.tensor([[0.2, 0.2, 0.2, 0.2, 0.2]], dtype=torch.float64)
    out = project_distribution(ts, w, z)
    assert out.sum().item() == pytest.approx(1.0)
    # Every atom gets exactly its own 0.2 back.
    assert torch.allclose(out, torch.full((1, 5), 0.2, dtype=torch.float64), atol=1e-12)
    assert np.allclose(out.numpy(), brute_force_project(ts, w, z))


def test_projection_clamps_targets_outside_the_support():
    z = _support(n=5, vmax=2.0)
    ts = torch.tensor([[-99.0, 99.0]], dtype=torch.float64)
    w = torch.tensor([[0.3, 0.7]], dtype=torch.float64)
    out = project_distribution(ts, w, z)
    assert out[0, 0].item() == pytest.approx(0.3)   # all of it on v_min
    assert out[0, -1].item() == pytest.approx(0.7)  # all of it on v_max
    assert out[0, 1:-1].abs().sum().item() == pytest.approx(0.0)


def test_projection_splits_a_midpoint_evenly():
    z = _support(n=5, vmax=2.0)  # spacing 1.0
    ts = torch.tensor([[0.5]], dtype=torch.float64)
    w = torch.tensor([[1.0]], dtype=torch.float64)
    out = project_distribution(ts, w, z)
    assert out[0, 2].item() == pytest.approx(0.5)  # atom 0.0
    assert out[0, 3].item() == pytest.approx(0.5)  # atom 1.0


def test_projection_output_is_sized_by_the_grid_not_the_targets():
    """K target atoms projected onto A grid atoms must return A columns."""
    z = _support(n=7, vmax=3.0)
    ts = torch.tensor([[0.0, 1.0]], dtype=torch.float64)  # K = 2, A = 7
    w = torch.tensor([[0.5, 0.5]], dtype=torch.float64)
    out = project_distribution(ts, w, z)
    assert out.shape == (1, 7)
    assert out.sum().item() == pytest.approx(1.0)
    assert np.allclose(out.numpy(), brute_force_project(ts, w, z))


def test_projection_rejects_mismatched_shapes():
    z = _support()
    with pytest.raises(ValueError):
        project_distribution(torch.zeros(2, 51), torch.zeros(3, 51), z)


# ----------------------------------------------------------------------
# The Bellman target
# ----------------------------------------------------------------------
def test_target_distribution_on_a_terminal_is_a_point_mass_at_the_return():
    z = torch.linspace(-10, 10, 51)
    probs = torch.rand(1, 3, 51)
    probs = probs / probs.sum(-1, keepdim=True)
    out = c51_target_distribution(
        probs,
        torch.tensor([1]),
        n_step_return=torch.tensor([1.0]),
        discount=torch.tensor([0.9]),
        done=torch.tensor([True]),
        support=z,
    )
    # The bootstrap is dropped, so all mass sits at 1.0. Spacing is 0.4, so
    # 1.0 falls between atom 27 (0.8) and atom 28 (1.2), split by distance.
    assert out.shape == (1, 51)
    assert out.sum().item() == pytest.approx(1.0, abs=1e-5)
    assert out[0, 27:29].sum().item() == pytest.approx(1.0, abs=1e-5)
    assert out[0, 27].item() == pytest.approx(0.5, abs=1e-5)
    assert out[0, 28].item() == pytest.approx(0.5, abs=1e-5)


def test_target_distribution_uses_the_selected_action_only():
    z = torch.linspace(-10, 10, 51)
    probs = torch.zeros(1, 3, 51)
    probs[0, 0, 0] = 1.0   # action 0: mass at -10
    probs[0, 1, -1] = 1.0  # action 1: mass at +10
    probs[0, 2, 25] = 1.0
    a = c51_target_distribution(probs, torch.tensor([1]), torch.tensor([0.0]),
                                torch.tensor([1.0]), torch.tensor([False]), z)
    assert a[0, -1].item() == pytest.approx(1.0)
    b = c51_target_distribution(probs, torch.tensor([0]), torch.tensor([0.0]),
                                torch.tensor([1.0]), torch.tensor([False]), z)
    assert b[0, 0].item() == pytest.approx(1.0)


def test_target_distribution_shifts_by_the_return():
    z = torch.linspace(-10, 10, 51)  # spacing 0.4
    probs = torch.zeros(1, 2, 51)
    probs[0, 0, 25] = 1.0  # mass at 0.0
    out = c51_target_distribution(probs, torch.tensor([0]), torch.tensor([0.4]),
                                  torch.tensor([1.0]), torch.tensor([False]), z)
    # 0 + 1.0*0.0 + 0.4 = 0.4 = atom 26 exactly.
    assert out[0, 26].item() == pytest.approx(1.0, abs=1e-5)


def test_bootstrap_action_selection_follows_double_dqn():
    """Superseded D-022; see D-032 and the two tests at the end of this file."""
    online = torch.tensor([[0.0, 5.0, 0.0]])
    target = torch.tensor([[9.0, 0.0, 0.0]])
    assert select_bootstrap_action(online, target, BBFConfig()).item() == 1


# ----------------------------------------------------------------------
# c51_loss
# ----------------------------------------------------------------------
def test_c51_loss_is_zero_when_prediction_matches_a_one_hot_target():
    logits = torch.zeros(1, 2, 4)
    logits[0, 0] = torch.tensor([50.0, -50.0, -50.0, -50.0])
    target = torch.zeros(1, 4)
    target[0, 0] = 1.0
    loss, per = c51_loss(logits, torch.tensor([0]), target)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)
    assert per.shape == (1,)


def test_c51_loss_is_the_cross_entropy():
    logits = torch.randn(3, 2, 5)
    target = torch.softmax(torch.randn(3, 5), dim=-1)
    action = torch.tensor([0, 1, 0])
    loss, per = c51_loss(logits, action, target)
    chosen = torch.stack([logits[i, action[i]] for i in range(3)])
    want = -(target * torch.log_softmax(chosen, -1)).sum(-1)
    assert torch.allclose(per, want, atol=1e-6)
    assert loss.item() == pytest.approx(want.mean().item(), abs=1e-6)


def test_c51_per_sample_loss_is_unweighted():
    """Priorities come from this, so importance weights must not enter it."""
    logits = torch.randn(4, 2, 5)
    target = torch.softmax(torch.randn(4, 5), dim=-1)
    a = torch.zeros(4, dtype=torch.long)
    _, per_plain = c51_loss(logits, a, target)
    w = torch.tensor([0.1, 1.0, 0.5, 2.0])
    loss_w, per_w = c51_loss(logits, a, target, weights=w)
    assert torch.allclose(per_plain, per_w)
    assert loss_w.item() == pytest.approx((per_plain * w).mean().item(), abs=1e-6)


def test_c51_loss_gradient_flows_to_the_chosen_action_only():
    logits = torch.randn(1, 3, 5, requires_grad=True)
    target = torch.softmax(torch.randn(1, 5), dim=-1)
    c51_loss(logits, torch.tensor([1]), target)[0].backward()
    assert logits.grad[0, 1].abs().sum() > 0
    assert logits.grad[0, 0].abs().sum() == 0
    assert logits.grad[0, 2].abs().sum() == 0


# ----------------------------------------------------------------------
# SPR loss
# ----------------------------------------------------------------------
def test_spr_loss_is_zero_when_directions_agree():
    pred = torch.randn(4, 3, 16)
    loss, per = spr_loss(pred, pred.clone())
    assert loss.item() == pytest.approx(0.0, abs=1e-5)
    assert torch.allclose(per, torch.zeros(4), atol=1e-5)


def test_spr_loss_is_scale_invariant():
    pred = torch.randn(2, 2, 8)
    a, _ = spr_loss(pred, pred * 7.0)
    b, _ = spr_loss(pred, pred)
    assert a.item() == pytest.approx(b.item(), abs=1e-5)


def test_spr_loss_is_four_when_directions_oppose():
    """||p - t||^2 = 2 - 2cos = 4 for antiparallel unit vectors."""
    pred = torch.randn(3, 1, 8)
    loss, _ = spr_loss(pred, -pred)
    assert loss.item() == pytest.approx(4.0, abs=1e-5)


def test_spr_loss_is_the_official_mean_over_jumps(monkeypatch):
    """Transcription of spr_agent.loss_fn: normalize both, squared distance
    summed over features, times the mask, MEAN over the K jumps (masked
    jumps still in the denominator)."""
    torch.manual_seed(0)
    pred, tgt = torch.randn(4, 5, 16), torch.randn(4, 5, 16)
    mask = torch.rand(4, 5) > 0.3
    p = pred / pred.norm(dim=-1, keepdim=True)
    t = tgt / tgt.norm(dim=-1, keepdim=True)
    want = ((p - t).pow(2).sum(-1) * mask).mean(1)
    _, per = spr_loss(pred, tgt, mask)
    assert torch.allclose(per, want, atol=1e-5)


def test_spr_loss_mask_excludes_jumps_past_an_episode_end():
    pred = torch.randn(2, 3, 8)
    tgt = -pred  # every valid jump contributes 4
    mask = torch.tensor([[True, True, False], [True, False, False]])
    _, per = spr_loss(pred, tgt, mask)
    # Masked entries contribute 0 but still divide: 8/3 and 4/3.
    assert per[0].item() == pytest.approx(8.0 / 3.0, abs=1e-5)
    assert per[1].item() == pytest.approx(4.0 / 3.0, abs=1e-5)


def test_spr_loss_masked_targets_cannot_affect_the_result():
    pred = torch.randn(1, 2, 8)
    tgt = pred.clone()
    mask = torch.tensor([[True, False]])
    _, a = spr_loss(pred, tgt, mask)
    tgt2 = tgt.clone()
    tgt2[0, 1] = torch.randn(8) * 100  # garbage in the masked slot
    _, b = spr_loss(pred, tgt2, mask)
    assert a.item() == pytest.approx(b.item(), abs=1e-6)


def test_spr_gradient_is_two_over_k_times_the_summed_cosine_gradient():
    """Why the form matters: at K = 5 the summed -cos loss had 2.5x the
    gradient of the official one."""
    torch.manual_seed(1)
    pred = torch.randn(3, 5, 8, requires_grad=True)
    tgt = torch.randn(3, 5, 8)
    spr_loss(pred, tgt)[0].backward()
    g_official = pred.grad.clone()
    pred.grad = None
    p = torch.nn.functional.normalize(pred, dim=-1)
    t = torch.nn.functional.normalize(tgt, dim=-1)
    (-(p * t).sum(-1).sum(1)).mean().backward()
    g_summed_cos = pred.grad
    assert torch.allclose(g_official, g_summed_cos * (2.0 / 5.0), atol=1e-6)


def test_combined_loss_weights_the_sum_not_just_c51():
    from playtrain_trainers.bbf.losses import combined_loss

    rl = torch.tensor([1.0, 2.0])
    spr = torch.tensor([0.5, 0.25])
    w = torch.tensor([1.0, 0.5])
    got = combined_loss(rl, spr, w, spr_weight=5.0)
    want = ((1.0 + 5 * 0.5) * 1.0 + (2.0 + 5 * 0.25) * 0.5) / 2
    assert got.item() == pytest.approx(want)


def test_spr_target_is_detached():
    pred = torch.randn(2, 2, 8, requires_grad=True)
    tgt = torch.randn(2, 2, 8, requires_grad=True)
    spr_loss(pred, tgt)[0].backward()
    assert pred.grad is not None
    assert tgt.grad is None, "gradient must not flow into the EMA target"


def test_spr_loss_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        spr_loss(torch.randn(2, 3, 8), torch.randn(2, 2, 8))


# ----------------------------------------------------------------------
# Augmentation
# ----------------------------------------------------------------------
def test_random_shift_preserves_shape_and_dtype():
    obs = torch.rand(6, 4, 84, 84)
    out = random_shift(obs, 4)
    assert out.shape == obs.shape and out.dtype == obs.dtype


def test_random_shift_is_the_same_for_every_frame_in_a_stack():
    """A per-frame shift would destroy the motion cue the stack provides."""
    # A single bright pixel at the same place in all 4 frames must stay
    # aligned across the stack after shifting.
    obs = torch.zeros(1, 4, 21, 21)
    obs[0, :, 10, 10] = 1.0
    torch.manual_seed(0)
    for _ in range(20):
        out = random_shift(obs, 4)
        pos = [tuple(torch.nonzero(out[0, c])[0].tolist()) for c in range(4)]
        assert len(set(pos)) == 1, pos


def test_random_shift_differs_across_batch_elements():
    obs = torch.zeros(32, 1, 21, 21)
    obs[:, 0, 10, 10] = 1.0
    torch.manual_seed(0)
    out = random_shift(obs, 4)
    positions = {tuple(torch.nonzero(out[i, 0])[0].tolist()) for i in range(32)}
    assert len(positions) > 3, positions


def test_random_shift_offsets_stay_within_the_pad():
    obs = torch.zeros(64, 1, 21, 21)
    obs[:, 0, 10, 10] = 1.0
    torch.manual_seed(1)
    out = random_shift(obs, 4)
    for i in range(64):
        nz = torch.nonzero(out[i, 0])
        if nz.numel() == 0:
            continue  # shifted off the edge, which replicate-pad can do
        y, x = nz[0].tolist()
        assert abs(y - 10) <= 4 and abs(x - 10) <= 4


def test_random_shift_with_zero_pad_is_identity():
    obs = torch.rand(2, 4, 8, 8)
    assert torch.equal(random_shift(obs, 0), obs)


def test_random_intensity_scales_each_image_uniformly():
    obs = torch.ones(8, 4, 5, 5)
    torch.manual_seed(0)
    out = random_intensity(obs, 0.05)
    # One scalar per image, so every pixel of an image shares a value.
    for i in range(8):
        assert out[i].std().item() == pytest.approx(0.0, abs=1e-6)
    # And the images do not all share the same one.
    assert len({round(out[i].mean().item(), 6) for i in range(8)}) > 3


def test_random_intensity_stays_within_two_sigma():
    obs = torch.ones(256, 1, 3, 3)
    torch.manual_seed(0)
    out = random_intensity(obs, 0.05)
    assert out.min() >= 1.0 - 2 * 0.05 - 1e-6
    assert out.max() <= 1.0 + 2 * 0.05 + 1e-6


def test_random_intensity_with_zero_scale_is_identity():
    obs = torch.rand(2, 1, 4, 4)
    assert torch.equal(random_intensity(obs, 0.0), obs)


def test_augment_is_off_when_the_config_says_so():
    obs = torch.rand(2, 4, 21, 21)
    assert torch.equal(augment(obs, BBFConfig(data_augmentation=False)), obs)


def test_augment_changes_the_observation_when_on():
    obs = torch.rand(8, 4, 21, 21)
    torch.manual_seed(0)
    out = augment(obs, BBFConfig())
    assert out.shape == obs.shape
    assert not torch.allclose(out, obs)


def test_to_float_normalizes_uint8_only():
    u8 = torch.full((1, 1, 2, 2), 255, dtype=torch.uint8)
    assert to_float(u8).max().item() == pytest.approx(1.0)
    f = torch.rand(1, 1, 2, 2)
    assert torch.equal(to_float(f), f)


# ----------------------------------------------------------------------
# EMA target
# ----------------------------------------------------------------------
def test_build_target_is_a_frozen_copy():
    online = nn.Linear(4, 3)
    target = build_target(online)
    assert torch.equal(target.weight, online.weight)
    assert not any(p.requires_grad for p in target.parameters())
    assert not target.training
    # A copy, not a reference.
    with torch.no_grad():
        online.weight.fill_(9.0)
    assert not torch.equal(target.weight, online.weight)


def test_ema_update_blends_at_tau():
    online, target = nn.Linear(3, 2), nn.Linear(3, 2)
    with torch.no_grad():
        online.weight.fill_(1.0)
        target.weight.fill_(0.0)
    ema_update(target, online, 0.005)
    assert target.weight[0, 0].item() == pytest.approx(0.005)
    ema_update(target, online, 0.005)
    assert target.weight[0, 0].item() == pytest.approx(0.005 * 2 - 0.005**2)


def test_ema_update_at_tau_one_copies():
    online, target = nn.Linear(3, 2), nn.Linear(3, 2)
    ema_update(target, online, 1.0)
    assert torch.allclose(target.weight, online.weight)


def test_ema_update_at_tau_zero_is_a_noop():
    online, target = nn.Linear(3, 2), nn.Linear(3, 2)
    before = target.weight.clone()
    ema_update(target, online, 0.0)
    assert torch.equal(target.weight, before)


def test_ema_converges_towards_the_online_net():
    online, target = nn.Linear(3, 2), nn.Linear(3, 2)
    with torch.no_grad():
        online.weight.fill_(1.0)
        target.weight.fill_(0.0)
    for _ in range(2000):
        ema_update(target, online, 0.005)
    assert target.weight[0, 0].item() > 0.99


def test_ema_copies_buffers_rather_than_blending():
    net = BBFNetwork(BBFConfig(width_scale=1, hidden_dim=16), num_actions=3)
    target = build_target(net)
    with torch.no_grad():
        net.support.fill_(7.0)
    ema_update(target, net, 0.005)
    assert torch.allclose(target.support, net.support)


def test_ema_rejects_a_bad_tau():
    online, target = nn.Linear(2, 2), nn.Linear(2, 2)
    with pytest.raises(ValueError):
        ema_update(target, online, 1.5)


# ----------------------------------------------------------------------
# Optimizer
# ----------------------------------------------------------------------
def test_optimizer_separates_the_encoder_learning_rate():
    cfg = BBFConfig(learning_rate=1e-4, encoder_learning_rate=5e-5)
    net = BBFNetwork(cfg, num_actions=4)
    opt = build_optimizer(net, cfg)
    groups = {g["name"]: g for g in opt.param_groups}
    assert groups["encoder"]["lr"] == 5e-5
    assert groups["encoder_no_decay"]["lr"] == 5e-5
    assert groups["other"]["lr"] == 1e-4
    assert groups["other_no_decay"]["lr"] == 1e-4


def test_optimizer_covers_every_trainable_parameter():
    cfg = BBFConfig(width_scale=1, hidden_dim=32)
    net = BBFNetwork(cfg, num_actions=4)
    opt = build_optimizer(net, cfg)
    in_opt = sum(p.numel() for g in opt.param_groups for p in g["params"])
    assert in_opt == sum(p.numel() for p in net.parameters() if p.requires_grad)


def test_optimizer_uses_the_gin_hyperparameters():
    cfg = BBFConfig()
    opt = build_optimizer(BBFNetwork(BBFConfig(width_scale=1, hidden_dim=16), 3), cfg)
    assert isinstance(opt, torch.optim.AdamW)
    for g in opt.param_groups:
        assert g["eps"] == pytest.approx(1.5e-4)
        want_wd = 0.0 if g["name"].endswith("_no_decay") else 0.1
        assert g["weight_decay"] == pytest.approx(want_wd), g["name"]


def test_weight_decay_skips_every_rank_one_parameter():
    """create_scaling_optimizer: mask = tree_map(lambda x: x.ndim != 1, p)."""
    cfg = BBFConfig(width_scale=1, hidden_dim=16)
    net = BBFNetwork(cfg, num_actions=3)
    opt = build_optimizer(net, cfg)
    for g in opt.param_groups:
        for p in g["params"]:
            if p.ndim == 1:
                assert g["weight_decay"] == 0.0
            else:
                assert g["weight_decay"] == pytest.approx(0.1)


def test_encoder_group_is_exactly_the_encoder():
    cfg = BBFConfig(width_scale=1, hidden_dim=16)
    net = BBFNetwork(cfg, num_actions=3)
    opt = build_optimizer(net, cfg)
    enc = {id(p) for g in opt.param_groups if g["name"].startswith("encoder") for p in g["params"]}
    # `encoder_keys = {"encoder", "transition_model"}` in the official agent.
    assert enc == {id(p) for p in net.encoder.parameters()} | {
        id(p) for p in net.transition_model.parameters()
    }


# ----------------------------------------------------------------------
# D-032: bootstrap selection and acting are ORTHOGONAL
# ----------------------------------------------------------------------
def test_double_dqn_means_the_online_net_selects_the_bootstrap():
    """The official spr_agent.py: `select_dist = online_dist if double_dqn`."""
    online = torch.tensor([[0.0, 5.0, 0.0]])
    target = torch.tensor([[9.0, 0.0, 0.0]])
    # gin default double_dqn = True -> ONLINE selects (standard Double DQN).
    assert select_bootstrap_action(online, target, BBFConfig()).item() == 1
    # double_dqn off -> the target selects (vanilla DQN target).
    assert select_bootstrap_action(
        online, target, BBFConfig(double_dqn=False)
    ).item() == 0


def test_target_action_selection_does_not_touch_the_bootstrap():
    """It governs the BEHAVIOUR policy, not the Bellman target (D-032).

    An earlier version had this backwards, letting `target_action_selection`
    override `double_dqn` for the bootstrap.
    """
    online = torch.tensor([[0.0, 5.0, 0.0]])
    target = torch.tensor([[9.0, 0.0, 0.0]])
    for flag in (True, False):
        cfg = BBFConfig(target_action_selection=flag)
        assert select_bootstrap_action(online, target, cfg).item() == 1


def test_acts_with_target_reads_the_gin_flag():
    from playtrain_trainers.bbf.losses import acts_with_target

    assert acts_with_target(BBFConfig()) is True          # gin: True
    assert acts_with_target(BBFConfig(target_action_selection=False)) is False


def test_spr_reaches_the_predictions_through_combined_loss():
    """The REAL training path, not `spr_loss(...)[0]` (D-043).

    `train.py` backprops `combined_loss(rl_per_sample, spr_per_sample, ...)`.
    When `spr_loss` returned a detached `per_sample`, that call graph carried
    no SPR term at all and every v3 run trained without the auxiliary loss,
    while the tests above -- which backward the scalar first return -- passed.
    """
    from playtrain_trainers.bbf.losses import combined_loss

    pred = torch.randn(4, 3, 8, requires_grad=True)
    rl = torch.randn(4, requires_grad=True)
    _, spr_per = spr_loss(pred, torch.randn(4, 3, 8))
    combined_loss(rl, spr_per, torch.ones(4), spr_weight=5.0).backward()
    assert pred.grad is not None and pred.grad.abs().sum() > 0


def test_spr_weight_scales_the_gradient_into_the_predictions():
    """A zero weight must be the only way to switch SPR off."""
    from playtrain_trainers.bbf.losses import combined_loss

    def grad(w):
        pred = torch.randn(4, 3, 8, generator=torch.Generator().manual_seed(0))
        pred.requires_grad_(True)
        tgt = torch.randn(4, 3, 8, generator=torch.Generator().manual_seed(1))
        _, spr_per = spr_loss(pred, tgt)
        combined_loss(torch.zeros(4), spr_per, torch.ones(4), spr_weight=w).backward()
        return pred.grad.abs().sum()

    assert grad(0.0) == 0.0
    assert torch.isclose(grad(5.0), 5.0 * grad(1.0), rtol=1e-5)
