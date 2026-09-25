"""Tests for playtrain_trainers.bbf.resets.

The central check the mission asks for is parameter distance before and after,
on a toy net: with shrink = perturb = 0.5 a shrink-and-perturb parameter must
land exactly halfway between its old value and a fresh initialization, while a
fully reset parameter must retain no trace of its old value.
"""
from __future__ import annotations

import pytest
import torch

from playtrain_trainers.bbf.config import BBFConfig
from playtrain_trainers.bbf.losses import build_optimizer, build_target
from playtrain_trainers.bbf.net import BBFNetwork
from playtrain_trainers.bbf.resets import (
    clear_optimizer_state,
    matches_key,
    reset_network,
    shrink_and_perturb_,
)

A = 4


def toy_cfg(**kw):
    """Small enough to construct repeatedly in a test."""
    base = dict(width_scale=1, hidden_dim=32, obs_size=64, num_atoms=11)
    base.update(kw)
    return BBFConfig(**base)


# ----------------------------------------------------------------------
# Key matching
# ----------------------------------------------------------------------
def test_matches_key_on_path_components():
    keys = ("encoder", "transition_model")
    assert matches_key("encoder.stages.0.conv.weight", keys)
    assert matches_key("transition_model.conv1.bias", keys)
    assert matches_key("encoder", keys)
    assert not matches_key("projection.weight", keys)
    assert not matches_key("predictor.0.weight", keys)
    assert not matches_key("value_head.bias", keys)


def test_matches_key_does_not_match_a_prefix_of_a_longer_name():
    # "encoder" must not swallow "encoder_head".
    assert not matches_key("encoder_head.weight", ("encoder",))


def test_the_gin_keys_are_what_the_network_exposes():
    """The gin names modules by string, so the names have to actually exist."""
    net = BBFNetwork(toy_cfg(), A)
    names = {n.split(".")[0] for n, _ in net.named_parameters()}
    for key in BBFConfig().shrink_perturb_keys:
        assert key in names, f"gin key {key!r} matches no module"


# ----------------------------------------------------------------------
# shrink_and_perturb_
# ----------------------------------------------------------------------
def test_shrink_and_perturb_is_the_stated_convex_combination():
    p = torch.full((3,), 2.0)
    f = torch.full((3,), 10.0)
    shrink_and_perturb_(p, f, 0.5, 0.5)
    assert torch.allclose(p, torch.full((3,), 6.0))


def test_shrink_one_perturb_zero_is_a_noop():
    p = torch.randn(5)
    before = p.clone()
    shrink_and_perturb_(p, torch.randn(5), 1.0, 0.0)
    assert torch.equal(p, before)


def test_shrink_zero_perturb_one_is_a_full_reset():
    p = torch.randn(5)
    f = torch.randn(5)
    shrink_and_perturb_(p, f, 0.0, 1.0)
    assert torch.equal(p, f)


# ----------------------------------------------------------------------
# The distance check, on the toy net
# ----------------------------------------------------------------------
def test_shrink_perturb_params_land_halfway_to_a_fresh_init():
    """The mission's parameter-distance check.

    With shrink = perturb = 0.5, new = 0.5*old + 0.5*fresh, so
    (new - old) = 0.5*(fresh - old) and therefore
    ||new - old|| == ||new - fresh||: the parameter sits exactly at the
    midpoint of the segment from its old value to the fresh one.
    """
    cfg = toy_cfg()
    torch.manual_seed(0)
    net = BBFNetwork(cfg, A)
    # Move the weights well away from initialization so "halfway back" is a
    # meaningful distance rather than noise.
    with torch.no_grad():
        for p in net.parameters():
            p.mul_(4.0).add_(1.0)
    old = {n: p.detach().clone() for n, p in net.named_parameters()}

    torch.manual_seed(123)
    fresh_ref = BBFNetwork(cfg, A)
    fresh = {n: p.detach().clone() for n, p in fresh_ref.named_parameters()}

    torch.manual_seed(123)  # so reset_network draws the SAME fresh init
    reset_network(net, cfg, A)

    for n, p in net.named_parameters():
        if n.startswith(("encoder.", "transition_model.")):
            want = 0.5 * old[n] + 0.5 * fresh[n]
            assert torch.allclose(p, want, atol=1e-6), n
            d_old = (p - old[n]).norm().item()
            d_fresh = (p - fresh[n]).norm().item()
            assert d_old == pytest.approx(d_fresh, rel=1e-5), n
        else:
            assert torch.allclose(p, fresh[n], atol=1e-6), n


def test_fully_reset_params_keep_no_trace_of_the_old_value():
    """"No trace" stated properly: the result is INDEPENDENT of the old value.

    A magnitude test does not work here -- a LayerNorm gain's fresh init is
    exactly 1.0, so "small and zero-centred" is false for it while the reset
    is perfectly correct. Resetting two differently-initialized networks with
    the same fresh-init seed is the real check.
    """
    cfg = toy_cfg()

    def reset_from(fill_value):
        torch.manual_seed(1)
        net = BBFNetwork(cfg, A)
        with torch.no_grad():
            for q in net.parameters():
                q.fill_(fill_value)
        torch.manual_seed(999)  # identical fresh init in both runs
        reset_network(net, cfg, A)
        return {n: q.detach().clone() for n, q in net.named_parameters()}

    a, b = reset_from(7.0), reset_from(-3.0)
    for n in a:
        if n.startswith(("encoder.", "transition_model.")):
            # Half the old value survives, so these MUST differ.
            assert not torch.allclose(a[n], b[n]), n
            # And by exactly half the gap between the two old values.
            assert torch.allclose(
                a[n] - b[n], torch.full_like(a[n], 0.5 * (7.0 - -3.0)), atol=1e-6
            ), n
        else:
            # Fully reset: identical regardless of what was there before.
            assert torch.equal(a[n], b[n]), n


def test_reset_summary_counts_and_distances():
    cfg = toy_cfg()
    net = BBFNetwork(cfg, A)
    with torch.no_grad():
        for p in net.parameters():
            p.add_(3.0)
    s = reset_network(net, cfg, A)
    n_sp = sum(
        1 for n, _ in net.named_parameters()
        if n.startswith(("encoder.", "transition_model."))
    )
    n_all = sum(1 for _ in net.named_parameters())
    assert s["params_shrunk_and_perturbed"] == n_sp
    assert s["params_fully_reset"] == n_all - n_sp
    assert set(s["l2_moved_by_group"]) == {
        "encoder", "projection", "value_head", "advantage_head",
        "transition_model", "predictor",
    }
    assert all(v > 0 for v in s["l2_moved_by_group"].values())
    assert s["optimizer_state_cleared"] is False and s["target_reset"] is False


def test_encoder_moves_less_than_a_fully_reset_head():
    """Relative to what it had, the shrink-and-perturb group moves less."""
    cfg = toy_cfg()
    net = BBFNetwork(cfg, A)
    with torch.no_grad():
        for p in net.parameters():
            p.add_(2.0)
    s = reset_network(net, cfg, A)
    enc_frac = s["l2_moved_by_group"]["encoder"] / s["l2_before_by_group"]["encoder"]
    proj_frac = s["l2_moved_by_group"]["projection"] / s["l2_before_by_group"]["projection"]
    assert enc_frac < proj_frac


def test_reset_is_reproducible_for_a_seed():
    cfg = toy_cfg()

    def run():
        torch.manual_seed(5)
        net = BBFNetwork(cfg, A)
        torch.manual_seed(77)
        reset_network(net, cfg, A)
        return torch.cat([p.detach().flatten() for p in net.parameters()])

    assert torch.equal(run(), run())


def test_reset_keeps_the_network_usable():
    cfg = toy_cfg()
    net = BBFNetwork(cfg, A)
    reset_network(net, cfg, A)
    out = net(torch.randint(0, 256, (2, 4, 64, 64), dtype=torch.uint8))
    assert out["q"].shape == (2, A)
    assert torch.isfinite(out["q"]).all()
    assert torch.allclose(out["probs"].sum(-1), torch.ones(2, A), atol=1e-5)


def test_reset_leaves_the_c51_support_alone():
    cfg = toy_cfg()
    net = BBFNetwork(cfg, A)
    before = net.support.clone()
    reset_network(net, cfg, A)
    assert torch.equal(net.support, before)


# ----------------------------------------------------------------------
# Optimizer state and the EMA target (D-023)
# ----------------------------------------------------------------------
def test_reset_clears_optimizer_state():
    cfg = toy_cfg()
    net = BBFNetwork(cfg, A)
    opt = build_optimizer(net, cfg)
    net(torch.randint(0, 256, (2, 4, 64, 64), dtype=torch.uint8))["q"].sum().backward()
    opt.step()
    assert len(opt.state) > 0, "the optimizer should have moments to clear"
    reset_network(net, cfg, A, optimizer=opt)
    assert len(opt.state) == 0


def test_clear_optimizer_state_directly():
    cfg = toy_cfg()
    net = BBFNetwork(cfg, A)
    opt = build_optimizer(net, cfg)
    net(torch.randint(0, 256, (2, 4, 64, 64), dtype=torch.uint8))["q"].sum().backward()
    opt.step()
    clear_optimizer_state(opt)
    assert len(opt.state) == 0


def test_optimizer_still_steps_after_a_reset():
    cfg = toy_cfg()
    net = BBFNetwork(cfg, A)
    opt = build_optimizer(net, cfg)
    reset_network(net, cfg, A, optimizer=opt)
    before = net.value_head.weight.detach().clone()
    net(torch.randint(0, 256, (2, 4, 64, 64), dtype=torch.uint8))["q"].sum().backward()
    opt.step()
    assert not torch.equal(net.value_head.weight, before)


def test_reset_applies_the_same_rule_to_the_target_from_its_own_draw():
    """`jit_reset` with reset_target=True (D-034): the target's encoder and
    transition model move half way toward a fresh init drawn with a SEPARATE
    PRNG key, and its other params are fully re-randomized. It is NOT a copy
    of the reset online network."""
    cfg = toy_cfg()
    torch.manual_seed(0)
    net = BBFNetwork(cfg, A)
    target = build_target(net)
    with torch.no_grad():
        for p in target.parameters():
            p.add_(5.0)  # make target differ from online
    old_t = {n: p.detach().clone() for n, p in target.named_parameters()}

    # Reproduce the two fresh draws reset_network makes, in order.
    torch.manual_seed(42)
    fresh_online = {n: p.detach().clone() for n, p in BBFNetwork(cfg, A).named_parameters()}
    fresh_target = {n: p.detach().clone() for n, p in BBFNetwork(cfg, A).named_parameters()}
    torch.manual_seed(42)
    s = reset_network(net, cfg, A, target=target)
    assert s["target_reset"] is True
    assert "target_l2_moved_by_group" in s

    for n, p in target.named_parameters():
        if n.startswith(("encoder.", "transition_model.")):
            assert torch.allclose(p, 0.5 * old_t[n] + 0.5 * fresh_target[n], atol=1e-6), n
        else:
            assert torch.allclose(p, fresh_target[n], atol=1e-6), n
    # And the online net used the FIRST draw, so the two are not equal.
    for (n, t), (_, o) in zip(target.named_parameters(), net.named_parameters(), strict=True):
        if not n.startswith(("encoder.", "transition_model.")):
            assert torch.allclose(o, fresh_online[n], atol=1e-6), n
            if t.ndim > 1:  # biases are zero in every fresh draw (D-037)
                assert not torch.equal(t, o), n


def test_target_stays_frozen_after_a_reset():
    cfg = toy_cfg()
    net = BBFNetwork(cfg, A)
    target = build_target(net)
    reset_network(net, cfg, A, target=target)
    assert not any(p.requires_grad for p in target.parameters())


def test_target_without_reset_is_left_alone():
    cfg = toy_cfg()
    net = BBFNetwork(cfg, A)
    target = build_target(net)
    before = torch.cat([p.detach().flatten().clone() for p in target.parameters()])
    reset_network(net, cfg, A)  # no target passed
    after = torch.cat([p.detach().flatten() for p in target.parameters()])
    assert torch.equal(before, after)


