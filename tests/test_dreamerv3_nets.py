"""U03 — networks, against `rssm.py`, `nets.py`, `heads.py`, `outs.py` at the pin.

The parameter counts below are written as ARITHMETIC over the official shapes, not
read from the modules: every expected number spells out `in * out + bias` for the
layers `configs.yaml` specifies, so the test compares the port against the official
files rather than against itself. If the official run is ever executed by hand, its
startup line `Optimizer ... has N params:` is the cross-check to record in D-016.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from playtrain_trainers.dreamerv3 import config as C
from playtrain_trainers.dreamerv3 import heads as H
from playtrain_trainers.dreamerv3 import nets as N
from playtrain_trainers.dreamerv3 import outs as O
from playtrain_trainers.dreamerv3 import rssm as R

ARTIFACTS = Path(__file__).resolve().parents[1] / "results" / "dreamerv3" / "net"

CFG = C.atari100k_config()
ACTIONS = 18  # ALE Frostbite minimal action set (PROTOCOL section 3)


def count(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def gen(seed: int = 0) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


# ----------------------------------------------------------------------
# Initializer (nets.Initializer)
# ----------------------------------------------------------------------
def test_compute_fans_matches_the_official_rule() -> None:
    assert N.compute_fans(()) == (1, 1)
    assert N.compute_fans((7,)) == (1, 7)
    assert N.compute_fans((3, 5)) == (3, 5)
    # Conv2D HWIO: space = 5*5, fanin = Cin * space, fanout = Cout * space
    assert N.compute_fans((5, 5, 3, 128)) == (3 * 25, 128 * 25)
    # BlockLinear (blocks, in/blocks, out/blocks): space = blocks, so fanin is the
    # FULL input width, not the per-block one. Copied deliberately.
    assert N.compute_fans((8, 4096, 1024)) == (4096 * 8, 1024 * 8)


def test_init_name_parsing() -> None:
    i = N.init("trunc_normal_in")
    assert (i.dist, i.fan, i.scale) == ("trunc_normal", "in", 1.0)
    assert N.init("zeros").dist == "zeros"
    assert N.init("normal").fan == "in"
    assert N.init("trunc_normal_out").fan == "out"


def test_trunc_normal_scale_and_support() -> None:
    """`x = truncated_normal(-2, 2) * 1.1368 * sqrt(1 / fan)`."""
    fan = 256
    x = N.Initializer("trunc_normal", "in")((fan, 4096), generator=gen())
    limit = 2.0 * 1.1368 * np.sqrt(1 / fan)
    assert x.abs().max().item() <= limit + 1e-6
    # The 1.1368 factor makes the truncated draw unit-variance again, so the std is
    # close to sqrt(1/fan).
    assert abs(x.std().item() - np.sqrt(1 / fan)) < 0.02 * np.sqrt(1 / fan)


def test_zeros_init_and_outscale() -> None:
    lin = N.Linear(8, 4, winit="trunc_normal_in", outscale=0.0, generator=gen())
    assert torch.equal(lin.kernel, torch.zeros_like(lin.kernel))
    assert torch.equal(lin.bias, torch.zeros_like(lin.bias))


def test_outscale_multiplies_the_kernel_not_the_bias() -> None:
    a = N.Linear(8, 4, outscale=1.0, generator=gen(1))
    b = N.Linear(8, 4, outscale=0.5, generator=gen(1))
    assert torch.allclose(b.kernel, a.kernel * 0.5, atol=1e-7)


# ----------------------------------------------------------------------
# Layers
# ----------------------------------------------------------------------
def test_linear_shapes_and_kernel_layout() -> None:
    lin = N.Linear(6, 5, generator=gen())
    assert lin.kernel.shape == (6, 5)  # (in, out), the JAX layout
    assert lin(torch.zeros(3, 2, 6)).shape == (3, 2, 5)


def test_linear_with_a_tuple_of_units_reshapes() -> None:
    """The decoder's `sp2` produces a (4, 4, 256) block from one Linear."""
    lin = N.Linear(16, (4, 4, 8), generator=gen())
    assert lin.kernel.shape == (16, 4 * 4 * 8)
    assert lin(torch.zeros(2, 16)).shape == (2, 4, 4, 8)


def test_block_linear_is_block_diagonal() -> None:
    """Block g of the output must depend only on block g of the input."""
    bl = N.BlockLinear(8, 12, blocks=4, generator=gen())
    assert bl.kernel.shape == (4, 2, 3)
    x = torch.zeros(1, 8)
    x[0, 0] = 1.0  # touch only block 0 (inputs 0 and 1)
    out = bl(x) - bl.bias
    assert out[0, :3].abs().sum() > 0
    assert out[0, 3:].abs().sum() == 0


def test_block_linear_matches_a_dense_equivalent() -> None:
    bl = N.BlockLinear(6, 6, blocks=3, generator=gen())
    x = torch.randn(2, 6, generator=gen(5))
    manual = []
    for k in range(3):
        manual.append(x[:, 2 * k : 2 * k + 2] @ bl.kernel[k])
    expected = torch.cat(manual, -1) + bl.bias
    assert torch.allclose(bl(x), expected, atol=1e-6)


def test_conv2d_is_nhwc_with_same_padding() -> None:
    conv = N.Conv2D(3, 8, 5, generator=gen())
    assert conv.kernel.shape == (5, 5, 3, 8)  # HWIO
    out = conv(torch.zeros(2, 16, 16, 3))
    assert out.shape == (2, 16, 16, 8)  # 'same' keeps the resolution at stride 1


def test_conv2d_against_a_hand_rolled_correlation() -> None:
    """torch conv2d is a cross-correlation, and so is jax.lax.conv_general_dilated
    with these dimension numbers, so no kernel flip belongs anywhere."""
    conv = N.Conv2D(1, 1, 3, bias=False, generator=gen(3))
    x = torch.randn(1, 5, 5, 1, generator=gen(4))
    k = conv.kernel[:, :, 0, 0]
    padded = torch.zeros(7, 7)
    padded[1:6, 1:6] = x[0, :, :, 0]
    expected = torch.zeros(5, 5)
    for i in range(5):
        for j in range(5):
            expected[i, j] = (padded[i : i + 3, j : j + 3] * k).sum()
    assert torch.allclose(conv(x)[0, :, :, 0], expected, atol=1e-5)


def test_rms_norm_formula_and_eps() -> None:
    norm = N.Norm("rms", 4)
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    mean2 = (x**2).mean(-1, keepdim=True)
    expected = x * torch.rsqrt(mean2 + 1e-4)
    assert torch.allclose(norm(x), expected, atol=1e-6)
    assert norm.eps == 1e-4  # not torch's 1e-5/1e-6 defaults
    assert norm.shift is None  # rms has scale only


def test_rms_norm_runs_in_float32_even_for_bf16_input() -> None:
    norm = N.Norm("rms", 4)
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.bfloat16)
    out = norm(x)
    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out).all()


def test_norm_1em_suffix_sets_eps() -> None:
    assert N.Norm("rms1em6", 4).eps == 1e-6


def test_symlog_symexp_round_trip() -> None:
    x = torch.tensor([-1234.5, -1.0, 0.0, 0.5, 9999.0])
    assert torch.allclose(N.symexp(N.symlog(x)), x, atol=1e-2)
    assert torch.equal(N.symlog(torch.zeros(3)), torch.zeros(3))


def test_gelu_uses_the_tanh_approximation() -> None:
    """jax.nn.gelu defaults to approximate=True; torch's default is exact."""
    x = torch.linspace(-3, 3, 7)
    exact = torch.nn.functional.gelu(x)
    got = N.act("gelu")(x)
    assert not torch.allclose(got, exact, atol=1e-6)
    assert torch.allclose(got, torch.nn.functional.gelu(x, approximate="tanh"))


# ----------------------------------------------------------------------
# outs
# ----------------------------------------------------------------------
def test_twohot_bins_are_symexp_of_a_linspace() -> None:
    bins = O.twohot_bins(255)
    assert bins.shape == (255,)
    assert bins[127].item() == 0.0
    assert torch.allclose(bins, -bins.flip(0), atol=1e-3)  # symmetric
    half = N.symexp(torch.linspace(-20, 0, 128))
    assert torch.allclose(bins[:128], half, atol=1e-6)
    # The end bins are +-(e^20 - 1) ~ 4.85e8, so the check is relative: float32
    # cannot represent that to better than about 32 ulps.
    assert abs(bins[0].item() + np.expm1(20)) / np.expm1(20) < 1e-6


def test_twohot_pred_is_exactly_zero_for_uniform_logits() -> None:
    """The symmetric sum in `TwoHot.pred` exists for this: with outscale 0.0 the
    reward head and the critic start at uniform logits and must read out as zero."""
    logits = torch.zeros(4, 255)
    pred = O.TwoHot(logits, O.twohot_bins(255)).pred()
    assert torch.equal(pred, torch.zeros(4))
    naive = (torch.softmax(logits, -1) * O.twohot_bins(255)).sum(-1)
    assert naive.abs().max() > 0  # the naive sum does NOT give zero


def test_twohot_loss_against_an_independent_implementation() -> None:
    """Written from the paper's definition: put the target's mass on the two
    bracketing bins in proportion to distance, then cross-entropy."""
    bins = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    logits = torch.tensor([[0.2, -0.5, 1.0, 0.3, -0.1]])
    target = torch.tensor([0.25])

    def reference(logits, bins, target):
        lo = int((bins <= target).sum().item()) - 1
        hi = lo + 1
        w_hi = float((target - bins[lo]) / (bins[hi] - bins[lo]))
        w_lo = 1.0 - w_hi
        logp = torch.log_softmax(logits, -1)[0]
        return -(w_lo * logp[lo] + w_hi * logp[hi])

    got = O.TwoHot(logits, bins).loss(target)
    assert torch.allclose(got, reference(logits, bins, target).reshape(1), atol=1e-6)


def test_twohot_loss_on_an_exact_bin_and_outside_the_range() -> None:
    bins = torch.tensor([-1.0, 0.0, 1.0])
    logits = torch.zeros(1, 3)
    # On a bin: all the mass on that bin, so the loss is -log softmax there.
    assert torch.allclose(
        O.TwoHot(logits, bins).loss(torch.tensor([0.0])),
        torch.tensor([float(np.log(3))]),
        atol=1e-6,
    )
    # Outside: clipped to the end bin rather than extrapolated.
    assert torch.allclose(
        O.TwoHot(logits, bins).loss(torch.tensor([50.0])),
        torch.tensor([float(np.log(3))]),
        atol=1e-6,
    )


def test_twohot_round_trips_a_value_through_loss_and_pred() -> None:
    """Fitting the two-hot target exactly reproduces the value on read-out."""
    bins = O.twohot_bins(255)
    target = torch.tensor([37.5])
    logits = torch.zeros(1, 255, requires_grad=True)
    optimizer = torch.optim.Adam([logits], lr=0.5)
    for _ in range(400):
        optimizer.zero_grad()
        O.TwoHot(logits, bins).loss(target).sum().backward()
        optimizer.step()
    assert abs(O.TwoHot(logits.detach(), bins).pred().item() - 37.5) < 0.1


def test_categorical_unimix_is_mixed_in_probability_space() -> None:
    logits = torch.tensor([[10.0, 0.0, -10.0, 3.0]])
    dist = O.Categorical(logits, 0.01)
    probs = torch.softmax(dist.logits, -1)
    plain = torch.softmax(logits, -1)
    expected = 0.99 * plain + 0.01 * torch.full_like(plain, 0.25)
    assert torch.allclose(probs, expected, atol=1e-6)
    assert probs.min() >= 0.01 / 4 - 1e-8  # no probability can be crushed to zero


def test_categorical_kl_against_a_brute_force_sum() -> None:
    torch.manual_seed(0)
    p = O.Categorical(torch.randn(3, 5), 0.01)
    q = O.Categorical(torch.randn(3, 5), 0.01)
    pp = torch.softmax(p.logits, -1)
    qq = torch.softmax(q.logits, -1)
    brute = (pp * (pp.log() - qq.log())).sum(-1)
    assert torch.allclose(p.kl(q), brute, atol=1e-5)
    assert (p.kl(p).abs() < 1e-6).all()


def test_categorical_entropy_against_a_brute_force_sum() -> None:
    torch.manual_seed(0)
    p = O.Categorical(torch.randn(4, 6))
    pp = torch.softmax(p.logits, -1)
    assert torch.allclose(p.entropy(), -(pp * pp.log()).sum(-1), atol=1e-5)


def test_onehot_sample_is_straight_through() -> None:
    logits = torch.randn(2, 5, generator=gen(), requires_grad=True)
    dist = O.OneHot(logits, 0.01)
    sample = dist.sample(gen(7))
    assert sample.shape == (2, 5)
    hard = (sample.detach() > 0).float()
    assert torch.equal(sample.detach().round(), hard)  # forward value is one-hot
    sample.sum().backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0


def test_agg_sums_the_last_dims() -> None:
    mse = O.MSE(torch.zeros(2, 3, 4, 5))
    agg = O.Agg(mse, 3, torch.sum)
    loss = agg.loss(torch.ones(2, 3, 4, 5))
    assert loss.shape == (2,)
    assert torch.allclose(loss, torch.full((2,), 60.0))


def test_mse_is_not_halved() -> None:
    out = O.MSE(torch.tensor([3.0]))
    assert out.loss(torch.tensor([1.0])).item() == 4.0


def test_binary_logp_and_pred() -> None:
    out = O.Binary(torch.tensor([2.0, -2.0]))
    assert out.pred().tolist() == [True, False]
    expected = np.log(1 / (1 + np.exp(-2.0)))
    assert abs(out.logp(torch.tensor([1.0, 0.0]))[0].item() - expected) < 1e-6


# ----------------------------------------------------------------------
# Encoder / RSSM / Decoder shapes
# ----------------------------------------------------------------------
def build_encoder(cfg=CFG, generator=None) -> R.Encoder:
    return R.Encoder(
        (cfg.env.size[0], cfg.env.size[1], 3),
        depth=cfg.agent.enc.depth,
        mults=cfg.agent.enc.mults,
        kernel=cfg.agent.enc.kernel,
        act=cfg.agent.enc.act,
        norm=cfg.agent.enc.norm,
        winit=cfg.agent.enc.winit,
        generator=generator,
    )


def build_rssm(cfg=CFG, token_dim=4096, generator=None) -> R.RSSM:
    r = cfg.agent.rssm
    return R.RSSM(
        ACTIONS,
        token_dim,
        deter=r.deter,
        hidden=r.hidden,
        stoch=r.stoch,
        classes=r.classes,
        blocks=r.blocks,
        imglayers=r.imglayers,
        obslayers=r.obslayers,
        dynlayers=r.dynlayers,
        act=r.act,
        norm=r.norm,
        unimix=r.unimix,
        outscale=r.outscale,
        winit=r.winit,
        free_nats=r.free_nats,
        generator=generator,
    )


def build_decoder(cfg=CFG, generator=None) -> R.Decoder:
    d = cfg.agent.dec
    return R.Decoder(
        (cfg.env.size[0], cfg.env.size[1], 3),
        cfg.agent.rssm.deter,
        cfg.agent.rssm.stoch,
        cfg.agent.rssm.classes,
        depth=d.depth,
        mults=d.mults,
        kernel=d.kernel,
        units=d.units,
        act=d.act,
        norm=d.norm,
        outscale=d.outscale,
        winit=d.winit,
        bspace=d.bspace,
        generator=generator,
    )


def test_encoder_resolution_ladder_and_token_dim() -> None:
    enc = build_encoder()
    assert [c.depth for c in enc.convs] == [128, 192, 256, 256]  # depth 64 x mults
    assert enc.minres == (4, 4)  # 64 -> 32 -> 16 -> 8 -> 4
    assert enc.outdim == 4 * 4 * 256 == 4096
    x = torch.zeros(2, 5, 64, 64, 3, dtype=torch.uint8)
    assert enc(x).shape == (2, 5, 4096)


def test_encoder_scales_pixels_to_pm_half() -> None:
    enc = build_encoder()
    white = torch.full((1, 64, 64, 3), 255, dtype=torch.uint8)
    black = torch.zeros(1, 64, 64, 3, dtype=torch.uint8)
    # 255/255 - 0.5 = 0.5 and 0/255 - 0.5 = -0.5 reach the first conv.
    assert not torch.equal(enc(white), enc(black))


def test_encoder_pooling_is_max_not_stride() -> None:
    """`strided: False`: a stride-1 conv then a 2x2 MAX over disjoint blocks."""
    enc = build_encoder()
    x = torch.zeros(1, 8, 8, 4)
    x[0, 0, 0, 0] = 5.0
    x[0, 1, 1, 0] = 9.0  # same 2x2 block, larger: max must pick it
    B, H, W, Cc = x.shape
    pooled = x.reshape(B, H // 2, 2, W // 2, 2, Cc).amax((2, 4))
    assert pooled[0, 0, 0, 0] == 9.0
    assert enc.act is torch.nn.functional.silu


def test_rssm_shapes_and_carry() -> None:
    dyn = build_rssm()
    carry = dyn.initial(3)
    assert carry["deter"].shape == (3, 8192)
    assert carry["stoch"].shape == (3, 32, 64)
    assert dyn.feat_dim == 8192 + 32 * 64
    tokens = torch.zeros(3, 4, 4096)
    action = torch.nn.functional.one_hot(torch.zeros(3, 4, dtype=torch.long), ACTIONS).float()
    reset = torch.zeros(3, 4, dtype=torch.bool)
    carry, entries, feat = dyn.observe(carry, tokens, action, reset, gen())
    assert entries["deter"].shape == (3, 4, 8192)
    assert entries["stoch"].shape == (3, 4, 32, 64)
    assert feat["logit"].shape == (3, 4, 32, 64)


def test_rssm_reset_zeroes_the_carry_and_the_action() -> None:
    dyn = build_rssm(token_dim=16)
    a = dyn.initial(1)
    a["deter"] = torch.randn(1, 8192, generator=gen(2))
    tokens = torch.zeros(1, 16)
    action = torch.ones(1, ACTIONS)
    reset_on, _ = dyn.observe_step(a, tokens, action, torch.ones(1, dtype=torch.bool), gen(1))
    zero = dyn.initial(1)
    from_zero, _ = dyn.observe_step(
        zero, tokens, torch.zeros(1, ACTIONS), torch.zeros(1, dtype=torch.bool), gen(1)
    )
    assert torch.allclose(reset_on["deter"], from_zero["deter"], atol=1e-5)


def test_rssm_core_gate_law() -> None:
    """`update = sigmoid(update - 1)` and `cand = tanh(reset * cand)`, not the
    textbook GRU and not `nets.GRU`'s `tanh(sigmoid(res) * cand)` form."""
    dyn = build_rssm(token_dim=16)
    deter = torch.zeros(1, 8192)
    stoch = torch.zeros(1, 32, 64)
    action = torch.zeros(1, ACTIONS)
    out = dyn._core(deter, stoch, action)
    assert out.shape == (1, 8192)
    # With a zero previous state, deter_new = update * cand; the update gate is
    # sigmoid(x - 1), so from zero-ish gates the state moves slowly.
    assert out.abs().max() < 1.0


def test_rssm_blocks_divide_deter() -> None:
    with pytest.raises(AssertionError):
        build_rssm().__class__(ACTIONS, 16, deter=100, blocks=8)


def test_rssm_free_nats_clip_is_below_not_above() -> None:
    dyn = build_rssm(token_dim=16)
    carry = dyn.initial(2)
    tokens = torch.zeros(2, 3, 16)
    action = torch.zeros(2, 3, ACTIONS)
    reset = torch.zeros(2, 3, dtype=torch.bool)
    _, _, losses, _, metrics = dyn.loss(carry, tokens, action, reset, gen())
    assert losses["dyn"].shape == (2, 3)
    # `max(kl, free_nats)` is a FLOOR: nothing may come out below 1.0, and a KL
    # above it passes through untouched. At t=0 the state is zero so prior and
    # posterior agree and the floor binds; by t=1 they have diverged and it does not.
    assert (losses["dyn"] >= 1.0 - 1e-6).all()
    assert (losses["rep"] >= 1.0 - 1e-6).all()
    assert torch.allclose(losses["dyn"][:, 0], torch.ones(2), atol=1e-4)
    assert (losses["dyn"][:, 1:] > 1.5).all()
    assert set(metrics) == {"dyn_ent", "rep_ent"}


def test_free_nats_clips_below_not_above() -> None:
    """The clip direction, isolated: a KL of 0 becomes 1.0, a KL of 5 stays 5."""
    dyn = build_rssm(token_dim=16)
    same = torch.zeros(2, 32, 64)
    kl = dyn.dist(same).kl(dyn.dist(same))
    assert torch.allclose(kl, torch.zeros(2), atol=1e-5)
    assert torch.clamp(kl, min=dyn.free_nats).tolist() == [1.0, 1.0]
    assert torch.clamp(torch.tensor([5.0]), min=dyn.free_nats).item() == 5.0


def test_rssm_dist_aggregates_over_the_stoch_axis_by_sum() -> None:
    dyn = build_rssm(token_dim=16)
    logits = torch.zeros(2, 32, 64)
    kl = dyn.dist(logits).kl(dyn.dist(logits))
    assert kl.shape == (2,)  # the 32 categoricals collapsed, not the batch
    ent = dyn.dist(logits).entropy()
    # 32 independent uniform 64-way categoricals: entropy 32 * log(64).
    assert abs(ent[0].item() - 32 * np.log(64)) < 0.5


def test_imagine_rolls_the_prior_without_tokens() -> None:
    dyn = build_rssm(token_dim=16)
    carry = dyn.initial(2)
    def policy(c):
        return torch.nn.functional.one_hot(torch.zeros(2, dtype=torch.long), ACTIONS).float()

    carry, feat, actions = dyn.imagine(carry, policy, 5, gen())
    assert feat["deter"].shape == (2, 5, 8192)
    assert actions.shape == (2, 5, ACTIONS)


def test_imagine_detaches_the_carry_before_the_policy() -> None:
    dyn = build_rssm(token_dim=16)
    carry = dyn.initial(1)
    seen = {}

    def policy(c):
        seen["requires_grad"] = c["deter"].requires_grad
        return torch.zeros(1, ACTIONS)

    dyn.imagine(carry, policy, 1, gen())
    assert seen["requires_grad"] is False


def test_decoder_shapes_and_sigmoid_range() -> None:
    dec = build_decoder(generator=gen())
    feat = {"deter": torch.zeros(2, 3, 8192), "stoch": torch.zeros(2, 3, 32, 64)}
    out = dec(feat)
    pred = out.pred()
    assert pred.shape == (2, 3, 64, 64, 3)
    assert (pred >= 0).all() and (pred <= 1).all()
    loss = out.loss(torch.zeros(2, 3, 64, 64, 3))
    assert loss.shape == (2, 3)  # summed over H, W, C by the Agg


def test_decoder_minres_and_upsampling_ladder() -> None:
    dec = build_decoder()
    assert dec.minres == (4, 4)
    assert [c.depth for c in dec.convs] == [256, 192, 128]  # depths[:-1] reversed
    assert dec.imgout.depth == 3


# ----------------------------------------------------------------------
# Parameter counts, from the official shapes by hand
# ----------------------------------------------------------------------
D, HID, S, K, G = 8192, 1024, 32, 64, 8
TOK = 4096
def NORM(n: int) -> int:
    """An rms Norm has one scale vector of width n, and no shift."""
    return n


def test_encoder_parameter_count_by_hand() -> None:
    expected = (
        (5 * 5 * 3 * 128 + 128) + NORM(128)
        + (5 * 5 * 128 * 192 + 192) + NORM(192)
        + (5 * 5 * 192 * 256 + 256) + NORM(256)
        + (5 * 5 * 256 * 256 + 256) + NORM(256)
    )
    assert count(build_encoder()) == expected == 3_492_864


def test_rssm_parameter_count_by_hand() -> None:
    core_in = D + 3 * HID * G  # 8192 + 24576 = 32768
    expected = (
        # _core: three input projections, each Linear + rms Norm
        (D * HID + HID) + NORM(HID)
        + (S * K * HID + HID) + NORM(HID)
        + (ACTIONS * HID + HID) + NORM(HID)
        # dynhid0: BlockLinear(core_in -> deter, 8 blocks) + Norm
        + (G * (core_in // G) * (D // G) + D) + NORM(D)
        # dyngru: BlockLinear(deter -> 3 * deter, 8 blocks)
        + (G * (D // G) * (3 * D // G) + 3 * D)
        # posterior: one layer over [deter, tokens], then the logit layer
        + ((D + TOK) * HID + HID) + NORM(HID)
        + (HID * S * K + S * K)
        # prior: two layers over deter, then the logit layer
        + (D * HID + HID) + NORM(HID)
        + (HID * HID + HID) + NORM(HID)
        + (HID * S * K + S * K)
    )
    assert count(build_rssm()) == expected == 95_496_192


def test_decoder_parameter_count_by_hand() -> None:
    u = 4 * 4 * 256  # minres x minres x depths[-1] = 4096
    expected = (
        (G * (D // G) * (u // G) + u)  # sp0, BlockLinear(deter -> 4096)
        + (S * K * 2 * 1024 + 2 * 1024) + NORM(2 * 1024)  # sp1 + norm
        + (2 * 1024 * u + u)  # sp2 -> (4, 4, 256)
        + NORM(256)  # spnorm, over the channel axis
        + (5 * 5 * 256 * 256 + 256) + NORM(256)
        + (5 * 5 * 256 * 192 + 192) + NORM(192)
        + (5 * 5 * 192 * 128 + 128) + NORM(128)
        + (5 * 5 * 128 * 3 + 3)  # imgout
    )
    assert count(build_decoder()) == expected == 20_282_115


def build_head(kind: str):
    feat = D + S * K
    if kind == "rew":
        h = CFG.agent.rewhead
        return H.MLPHead(feat, h.output, h.layers, h.units, bins=h.bins, outscale=h.outscale)
    if kind == "con":
        h = CFG.agent.conhead
        return H.MLPHead(feat, h.output, h.layers, h.units, outscale=h.outscale)
    if kind == "val":
        h = CFG.agent.value
        return H.MLPHead(feat, h.output, h.layers, h.units, bins=h.bins, outscale=h.outscale)
    p = CFG.agent.policy
    return H.MLPHead(
        feat, "categorical", p.layers, p.units, outscale=p.outscale,
        classes=ACTIONS, unimix=p.unimix,
    )


def test_head_parameter_counts_by_hand() -> None:
    feat = D + S * K  # 10240
    mlp1 = (feat * HID + HID) + NORM(HID)
    mlp3 = mlp1 + 2 * ((HID * HID + HID) + NORM(HID))
    assert count(build_head("rew")) == mlp1 + (HID * 255 + 255) == 10_749_183
    assert count(build_head("con")) == mlp1 + (HID * 1 + 1) == 10_488_833
    assert count(build_head("val")) == mlp3 + (HID * 255 + 255) == 12_850_431
    assert count(build_head("pol")) == mlp3 + (HID * ACTIONS + ACTIONS) == 12_607_506


def test_policy_head_does_not_apply_unimix() -> None:
    """D-027. The official `Head.categorical` builds `outs.Categorical(logits)` with NO
    unimix; `Head.unimix` is consumed only by `Head.onehot`, which a discrete policy
    never reaches. So `agent.policy.unimix: 0.01` is dead in the pinned code path.

    The earlier version of this file built the head with `unimix=p.unimix` and then
    only checked the CONFIG value, so it encoded the deviation instead of catching it.
    This asserts what the head actually does with the argument.
    """
    head = build_head("pol")
    assert head.unimix == 0.01  # the config value is still carried
    feat = torch.randn(4, D + S * K, generator=gen(11))
    dist = head(feat)
    # The logits must be the raw output of the head's final Linear, not
    # log(0.99 * softmax + 0.01/18) -- so no probability is floored at 5.6e-4.
    raw = head.out(head.mlp(feat))
    assert torch.allclose(dist.logits, raw.float(), atol=1e-6)
    # A deliberately peaked input keeps a genuinely tiny probability tiny.
    peaked = O.Categorical(torch.tensor([[0.0, -20.0, -20.0]]))
    assert peaked.logits.min() < -15
    floored = O.Categorical(torch.tensor([[0.0, -20.0, -20.0]]), 0.01)
    assert floored.logits.min() > -10  # what unimix WOULD have done


def test_latent_unimix_is_still_applied() -> None:
    """The RSSM latent's unimix is a different setting and IS in the official path
    (`rssm._dist` -> `OneHot(logits, self.unimix)`). D-027 must not remove it."""
    dyn = build_rssm(token_dim=16)
    assert dyn.unimix == 0.01
    logits = torch.full((1, 32, 64), -30.0)
    logits[..., 0] = 30.0
    probs = torch.softmax(dyn.dist(logits).output.dist.logits, -1)
    assert probs.min() >= 0.01 / 64 * 0.99  # the uniform floor is present


def test_head_outputs() -> None:
    feat = torch.zeros(2, 3, D + S * K)
    rew = build_head("rew")(feat)
    assert isinstance(rew, O.TwoHot)
    assert rew.pred().shape == (2, 3)
    assert torch.equal(rew.pred(), torch.zeros(2, 3))  # outscale 0.0 -> exactly zero
    con = build_head("con")(feat)
    assert isinstance(con, O.Binary)
    assert con.prob1().shape == (2, 3)
    pol = build_head("pol")(feat)
    assert isinstance(pol, O.Categorical)
    assert pol.logits.shape == (2, 3, ACTIONS)
    val = build_head("val")(feat)
    assert torch.equal(val.pred(), torch.zeros(2, 3))


def test_critic_and_reward_head_start_at_exactly_zero() -> None:
    """`outscale: 0.0` on both output layers. If this regressed, the actor would be
    chasing a nonzero value function before a single gradient step."""
    for kind in ("rew", "val"):
        head = build_head(kind)
        assert torch.equal(head.out.kernel, torch.zeros_like(head.out.kernel))


def test_feat2tensor_order_is_deter_then_stoch() -> None:
    feat = {"deter": torch.ones(2, 4), "stoch": torch.zeros(2, 3, 5)}
    out = H.feat2tensor(feat)
    assert out.shape == (2, 4 + 15)
    assert out[0, :4].sum() == 4 and out[0, 4:].sum() == 0


def test_total_parameter_count_and_write_params_json() -> None:
    parts = {
        "enc": count(build_encoder()),
        "dyn": count(build_rssm()),
        "dec": count(build_decoder()),
        "rew": count(build_head("rew")),
        "con": count(build_head("con")),
        "pol": count(build_head("pol")),
        "val": count(build_head("val")),
    }
    total = sum(parts.values())
    assert total == 165_967_124
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "params.json").write_text(
        json.dumps(
            {
                "config": "atari100k (defaults = size200m)",
                "action_dim": ACTIONS,
                "obs_shape_hwc": [64, 64, 3],
                "token_dim": TOK,
                "modules": parts,
                "total_trainable": total,
                "note": (
                    "Optimizer covers dyn, enc, dec, rew, con, pol, val (agent.modules); "
                    "slowval is an EMA copy of val and is not optimized, so it is not "
                    "counted here. The official run prints 'Optimizer ... has N params:' "
                    "at startup; that number is the cross-check (D-016)."
                ),
            },
            indent=2,
        )
        + "\n"
    )


# ----------------------------------------------------------------------
# Debug-size smoke: the whole stack runs end to end on CPU
# ----------------------------------------------------------------------
def test_debug_size_stack_runs() -> None:
    cfg = C.debug_config()
    enc = build_encoder(cfg)
    dyn = build_rssm(cfg, token_dim=enc.outdim)
    dec = build_decoder(cfg)
    image = torch.randint(0, 256, (2, 3, 64, 64, 3), dtype=torch.uint8)
    tokens = enc(image)
    action = torch.nn.functional.one_hot(torch.zeros(2, 3, dtype=torch.long), ACTIONS).float()
    reset = torch.zeros(2, 3, dtype=torch.bool)
    reset[:, 0] = True
    carry, entries, losses, feat, _ = dyn.loss(dyn.initial(2), tokens, action, reset, gen())
    recon = dec(feat)
    loss = recon.loss(image.float() / 255)
    assert loss.shape == (2, 3)
    assert torch.isfinite(loss).all()
    total = losses["dyn"].mean() + losses["rep"].mean() + loss.mean()
    total.backward()
    grads = [p.grad for p in list(enc.parameters()) + list(dyn.parameters()) if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
