"""Tests for playtrain_trainers.bbf.net.

Two jobs: pin the shapes the loss and replay code will assume, and pin the
architecture facts that ARE specified (15 conv layers, width x4, 51 atoms,
2048 hidden) so a later edit cannot quietly change what "BBF" means here.
"""
from __future__ import annotations

import pytest
import torch
from torch import nn

from playtrain_trainers.bbf.config import BBFConfig
from playtrain_trainers.bbf.net import (
    BASE_DEPTHS,
    BBFEncoder,
    BBFNetwork,
    ResidualBlock,
    TransitionModel,
    build_network,
    renormalize,
    resolve_device,
)

A = 8


@pytest.fixture(scope="module")
def net():
    return BBFNetwork(BBFConfig(), num_actions=A)


# ----------------------------------------------------------------------
# renormalize (D-019)
# ----------------------------------------------------------------------
def test_renormalize_maps_each_sample_to_unit_range():
    x = torch.randn(5, 3, 7, 7) * 10 + 4
    y = renormalize(x)
    assert y.shape == x.shape
    flat = y.flatten(1)
    assert torch.allclose(flat.min(dim=-1).values, torch.zeros(5), atol=1e-5)
    assert torch.allclose(flat.max(dim=-1).values, torch.ones(5), atol=1e-4)


def test_renormalize_is_per_sample_not_per_batch():
    # Two samples on wildly different scales must BOTH end up in [0, 1].
    x = torch.stack([torch.zeros(2, 3, 3), torch.full((2, 3, 3), 500.0)])
    x[0, 0, 0, 0] = -1.0
    x[1, 0, 0, 0] = 1000.0
    y = renormalize(x).flatten(1)
    assert y.min() >= 0.0 and y.max() <= 1.0 + 1e-4
    assert y[0].max() == pytest.approx(1.0, abs=1e-4)
    assert y[1].max() == pytest.approx(1.0, abs=1e-4)


def test_renormalize_survives_a_constant_latent():
    # A freshly reset network can emit a constant latent; without the epsilon
    # floor this divides by zero and poisons the first update with NaNs.
    y = renormalize(torch.full((3, 4, 5, 5), 2.0))
    assert torch.isfinite(y).all()


def test_renormalize_preserves_order_within_a_sample():
    x = torch.randn(2, 3, 4, 4)
    y = renormalize(x)
    for i in range(2):
        assert torch.equal(x[i].flatten().argsort(), y[i].flatten().argsort())


# ----------------------------------------------------------------------
# Encoder: the specified architecture
# ----------------------------------------------------------------------
def test_encoder_is_fifteen_conv_layers():
    """The paper calls Impala-CNN "a 15-layer ResNet"."""
    enc = BBFEncoder(in_channels=4, width_scale=4, num_blocks=2)
    convs = [m for m in enc.modules() if isinstance(m, nn.Conv2d)]
    # 3 stages x (1 stem conv + 2 blocks x 2 convs) = 15
    assert len(convs) == 15


def test_encoder_widths_are_base_depths_times_scale():
    assert BASE_DEPTHS == (16, 32, 32)
    assert BBFEncoder(4, width_scale=4).depths == (64, 128, 128)
    assert BBFEncoder(4, width_scale=1).depths == (16, 32, 32)
    assert BBFEncoder(4, width_scale=2).depths == (32, 64, 64)


def test_encoder_latent_shape_at_the_protocol_input():
    enc = BBFEncoder(4, width_scale=4, num_blocks=2)
    # 84 -> 42 -> 21 -> 11 through three stride-2 pools.
    assert enc.latent_shape((4, 84, 84)) == (128, 11, 11)
    # The native PlayTrain size lands on 8x8; see F-009.
    assert enc.latent_shape((4, 64, 64)) == (128, 8, 8)


def test_encoder_output_is_renormalized_when_asked():
    enc = BBFEncoder(4, width_scale=1, num_blocks=1, do_renormalize=True)
    out = enc(torch.randint(0, 256, (3, 4, 84, 84), dtype=torch.uint8))
    flat = out.flatten(1)
    assert torch.allclose(flat.min(dim=-1).values, torch.zeros(3), atol=1e-5)


def test_encoder_without_renormalize_is_not_bounded():
    enc = BBFEncoder(4, width_scale=1, num_blocks=1, do_renormalize=False)
    out = enc(torch.randint(0, 256, (3, 4, 84, 84), dtype=torch.uint8))
    assert out.min() >= 0.0  # it is a relu output
    assert not torch.allclose(out.flatten(1).max(dim=-1).values, torch.ones(3))


def test_encoder_accepts_uint8_and_float():
    enc = BBFEncoder(4, width_scale=1, num_blocks=1)
    u8 = torch.randint(0, 256, (2, 4, 84, 84), dtype=torch.uint8)
    a = enc(u8)
    b = enc(u8.float() / 255.0)
    assert torch.allclose(a, b, atol=1e-5)


def test_residual_block_is_identity_at_zero_weights():
    blk = ResidualBlock(6)
    with torch.no_grad():
        for p in blk.parameters():
            p.zero_()
    x = torch.randn(2, 6, 5, 5)
    assert torch.allclose(blk(x), x)


# ----------------------------------------------------------------------
# Transition model (D-020)
# ----------------------------------------------------------------------
def test_transition_model_preserves_the_latent_shape():
    tm = TransitionModel(channels=128, num_actions=A)
    latent = torch.rand(3, 128, 11, 11)
    out = tm(latent, torch.tensor([0, 3, 7]))
    assert out.shape == latent.shape


def test_transition_model_can_be_applied_repeatedly():
    tm = TransitionModel(channels=32, num_actions=A)
    latent = torch.rand(2, 32, 8, 8)
    for _ in range(5):
        latent = tm(latent, torch.tensor([1, 2]))
    assert latent.shape == (2, 32, 8, 8)
    assert torch.isfinite(latent).all()


def test_transition_model_output_depends_on_the_action():
    tm = TransitionModel(channels=16, num_actions=A)
    latent = torch.rand(1, 16, 6, 6)
    outs = [tm(latent, torch.tensor([a])) for a in range(A)]
    # Every action must move the latent somewhere different, or the SPR loss
    # carries no information about the action taken.
    for i in range(1, A):
        assert not torch.allclose(outs[0], outs[i], atol=1e-6)


def test_transition_model_output_is_renormalized():
    tm = TransitionModel(channels=16, num_actions=A, do_renormalize=True)
    out = tm(torch.rand(4, 16, 6, 6), torch.tensor([0, 1, 2, 3])).flatten(1)
    assert torch.allclose(out.min(dim=-1).values, torch.zeros(4), atol=1e-5)


def test_transition_model_rejects_an_out_of_range_action():
    tm = TransitionModel(channels=8, num_actions=4)
    with pytest.raises(RuntimeError):
        tm(torch.rand(1, 8, 4, 4), torch.tensor([9]))


# ----------------------------------------------------------------------
# Full network: shapes
# ----------------------------------------------------------------------
def test_forward_shapes(net):
    obs = torch.randint(0, 256, (6, 4, 84, 84), dtype=torch.uint8)
    out = net(obs)
    assert out["latent"].shape == (6, 128, 11, 11)
    assert out["features"].shape == (6, 2048)
    assert out["logits"].shape == (6, A, 51)
    assert out["probs"].shape == (6, A, 51)
    assert out["log_probs"].shape == (6, A, 51)
    assert out["q"].shape == (6, A)


def test_probs_are_a_distribution_over_atoms(net):
    out = net(torch.randint(0, 256, (4, 4, 84, 84), dtype=torch.uint8))
    assert torch.allclose(out["probs"].sum(-1), torch.ones(4, A), atol=1e-5)
    assert (out["probs"] >= 0).all()
    assert torch.allclose(out["log_probs"].exp(), out["probs"], atol=1e-6)


def test_q_is_the_expectation_over_the_support(net):
    out = net(torch.randint(0, 256, (3, 4, 84, 84), dtype=torch.uint8))
    manual = (out["probs"] * net.support).sum(-1)
    assert torch.allclose(out["q"], manual, atol=1e-6)


def test_q_is_inside_the_support_range(net):
    out = net(torch.randint(0, 256, (8, 4, 84, 84), dtype=torch.uint8))
    assert (out["q"] >= net.cfg.v_min).all() and (out["q"] <= net.cfg.v_max).all()


def test_support_is_the_c51_grid(net):
    assert net.support.shape == (51,)
    assert net.support[0].item() == pytest.approx(-10.0)
    assert net.support[-1].item() == pytest.approx(10.0)
    diffs = net.support.diff()
    assert torch.allclose(diffs, diffs[0].expand_as(diffs), atol=1e-6)
    # The support must travel with the module, so it has to be a buffer.
    assert "support" in dict(net.named_buffers())


def test_dueling_advantage_is_mean_zero_over_actions(net):
    """The dueling combination must remove the advantage's action-mean."""
    feats = torch.randn(5, 2048)
    logits, _ = net.heads(feats)
    h = torch.relu(feats)
    value = net.value_head(h).view(-1, 1, 51)
    # logits - value should be the centered advantage, so its action-mean is 0.
    centered = logits - value
    assert torch.allclose(centered.mean(dim=1), torch.zeros(5, 51), atol=1e-5)


def test_adding_a_constant_to_the_advantage_head_does_not_change_logits(net):
    """The identifiability the dueling form is there to fix."""
    feats = torch.randn(3, 2048)
    before, _ = net.heads(feats)
    with torch.no_grad():
        net.advantage_head.bias += 2.5
    after, _ = net.heads(feats)
    with torch.no_grad():
        net.advantage_head.bias -= 2.5
    assert torch.allclose(before, after, atol=1e-4)


# ----------------------------------------------------------------------
# SPR paths
# ----------------------------------------------------------------------
def test_spr_predictions_shape(net):
    obs = torch.randint(0, 256, (4, 4, 84, 84), dtype=torch.uint8)
    actions = torch.randint(0, A, (4, 5))
    out = net.spr_predictions(obs, actions)
    # jumps = 5
    assert out.shape == (4, 5, 2048)


def test_spr_predictions_respect_the_jump_count(net):
    obs = torch.randint(0, 256, (2, 4, 84, 84), dtype=torch.uint8)
    for k in (1, 3, 5):
        assert net.spr_predictions(obs, torch.zeros(2, k, dtype=torch.long)).shape[1] == k


def test_spr_predictions_reject_a_flat_action_tensor(net):
    obs = torch.randint(0, 256, (2, 4, 84, 84), dtype=torch.uint8)
    with pytest.raises(ValueError, match=r"\[B, K\]"):
        net.spr_predictions(obs, torch.zeros(2, dtype=torch.long))


def test_target_projections_shape_and_no_grad(net):
    obs = torch.randint(0, 256, (4, 4, 84, 84), dtype=torch.uint8)
    out = net.target_projections(obs)
    assert out.shape == (4, 2048)
    assert not out.requires_grad, "the SPR target must not carry gradient"


def test_projection_is_shared_between_the_q_head_and_spr(net):
    """SPR must shape the representation the Q head reads, not a private one."""
    obs = torch.randint(0, 256, (2, 4, 84, 84), dtype=torch.uint8)
    latent = net.encoder(obs)
    assert torch.allclose(net.project(latent), net(obs)["features"], atol=1e-5)
    # There is exactly one projection Linear in the module tree.
    assert sum(1 for n, _ in net.named_modules() if n == "projection") == 1


def test_spr_gradient_reaches_the_encoder_and_transition_model(net):
    obs = torch.randint(0, 256, (2, 4, 84, 84), dtype=torch.uint8)
    net.zero_grad(set_to_none=True)
    net.spr_predictions(obs, torch.zeros(2, 5, dtype=torch.long)).sum().backward()
    assert net.encoder.stages[0].conv.weight.grad is not None
    assert net.transition_model.conv1.weight.grad is not None
    assert net.predictor.weight.grad is not None  # D-031: one Linear
    net.zero_grad(set_to_none=True)


# ----------------------------------------------------------------------
# Parameter accounting (recorded for U04)
# ----------------------------------------------------------------------
def test_parameter_groups_partition_the_model(net):
    c = net.parameter_counts()
    assert c["total"] == sum(v for k, v in c.items() if k != "total")


def test_parameter_counts_are_the_recorded_ones(net):
    """Pin the count so a silent architecture change is visible in the diff."""
    c = net.parameter_counts()
    assert c["encoder"] == 1_552_192
    assert c["projection"] == 31_721_472
    assert c["value_head"] == 104_499        # 2048 * 51 + 51
    assert c["advantage_head"] == 835_992    # 2048 * 8 * 51 + 408
    assert c["transition_model"] == 304_384
    # D-031: the predictor is ONE Linear (2048*2048 + 2048), as in the
    # official spr_networks.py. It was 8,396,800 before that audit.
    assert c["predictor"] == 4_196_352
    assert c["total"] == 38_714_891


def test_the_projection_dominates_the_parameter_count(net):
    c = net.parameter_counts()
    # 74% of the model is the flatten -> 2048 dense layer. This is the number
    # behind F-009: it is what makes our reconstruction disagree with the
    # paper's stated CNN-vs-ResNet parameter ratio.
    assert c["projection"] / c["total"] > 0.7


def test_num_atoms_and_hidden_dim_follow_the_config():
    cfg = BBFConfig(num_atoms=11, hidden_dim=64)
    n = BBFNetwork(cfg, num_actions=3)
    out = n(torch.randint(0, 256, (2, 4, 84, 84), dtype=torch.uint8))
    assert out["logits"].shape == (2, 3, 11)
    assert out["features"].shape == (2, 64)


def test_width_scale_one_is_much_smaller():
    small = BBFNetwork(BBFConfig(width_scale=1), num_actions=A).parameter_counts()
    big = BBFNetwork(BBFConfig(width_scale=4), num_actions=A).parameter_counts()
    assert small["encoder"] * 8 < big["encoder"]


# ----------------------------------------------------------------------
# Devices
# ----------------------------------------------------------------------
def test_resolve_device_honors_an_explicit_spec():
    assert resolve_device("cpu").type == "cpu"


def test_build_network_on_cpu():
    n, dev = build_network(BBFConfig(), A, device="cpu")
    assert dev.type == "cpu"
    assert n.support.device.type == "cpu"


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS not available"
)
def test_forward_on_mps():
    n, dev = build_network(BBFConfig(), A, device="mps")
    assert dev.type == "mps"
    obs = torch.randint(0, 256, (2, 4, 84, 84), dtype=torch.uint8, device=dev)
    out = n(obs)
    assert out["q"].shape == (2, A)
    assert torch.isfinite(out["q"]).all()
    # The SPR path must work on MPS too; the training smoke test runs here.
    spr = n.spr_predictions(obs, torch.zeros(2, 5, dtype=torch.long, device=dev))
    assert spr.shape == (2, 5, 2048) and torch.isfinite(spr).all()


# ----------------------------------------------------------------------
# The shared-encoder path used by the training step
# ----------------------------------------------------------------------
def test_forward_with_spr_matches_the_separate_calls(net):
    obs = torch.randint(0, 256, (3, 4, 84, 84), dtype=torch.uint8)
    actions = torch.randint(0, A, (3, 5))
    net.eval()
    with torch.no_grad():
        joint = net.forward_with_spr(obs, actions)
        plain = net(obs)
        spr = net.spr_predictions(obs, actions)
    assert torch.allclose(joint["q"], plain["q"], atol=1e-5)
    assert torch.allclose(joint["logits"], plain["logits"], atol=1e-5)
    assert torch.allclose(joint["spr_predictions"], spr, atol=1e-5)


def test_forward_with_spr_encodes_once(net):
    """The point of the shared path: one encoder call, not two."""
    calls = []
    real = net.encoder.forward

    def counting(x):
        calls.append(1)
        return real(x)

    net.encoder.forward = counting
    try:
        net.forward_with_spr(
            torch.randint(0, 256, (2, 4, 84, 84), dtype=torch.uint8),
            torch.zeros(2, 5, dtype=torch.long),
        )
    finally:
        net.encoder.forward = real
    assert len(calls) == 1


def test_forward_with_spr_rejects_a_flat_action_tensor(net):
    with pytest.raises(ValueError, match=r"\[B, K\]"):
        net.forward_with_spr(
            torch.randint(0, 256, (2, 4, 84, 84), dtype=torch.uint8),
            torch.zeros(2, dtype=torch.long),
        )
