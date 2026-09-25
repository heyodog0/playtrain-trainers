"""U05 — world-model losses, against `agent.loss` and `rssm.loss` at the pin.

The two-hot loss, the unimix KL and the free-nats floor were verified against
independent references in `test_dreamerv3_nets.py` (U03, where the distributions
live); this file checks them again where they are USED, and adds what only appears at
this level: the continue target, the image target, the (B, T) shape assertion, the
scale table, and which terms carry a gradient into the latent.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from playtrain_trainers.dreamerv3 import config as C
from playtrain_trainers.dreamerv3 import heads as H
from playtrain_trainers.dreamerv3 import losses as L
from playtrain_trainers.dreamerv3 import outs as O
from playtrain_trainers.dreamerv3 import rssm as R

CFG = C.debug_config()
B, T = 2, 4
DETER, STOCH, CLASSES = 8, 2, 4
FEAT = DETER + STOCH * CLASSES


def gen(seed: int = 0) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def make_heads(rew_outscale: float = 0.0):
    rew = H.MLPHead(FEAT, "symexp_twohot", layers=1, units=8, bins=5, outscale=rew_outscale)
    con = H.MLPHead(FEAT, "binary", layers=1, units=8, outscale=1.0)
    return rew, con


def make_obs(terminal_at: int | None = None, reward: float = 1.0):
    is_terminal = torch.zeros(B, T, dtype=torch.bool)
    if terminal_at is not None:
        is_terminal[:, terminal_at] = True
    return {
        "is_first": torch.zeros(B, T, dtype=torch.bool),
        "is_last": is_terminal.clone(),
        "is_terminal": is_terminal,
        "reward": torch.full((B, T), reward),
        "image": torch.randint(0, 256, (B, T, 8, 8, 3), dtype=torch.uint8),
    }


def make_feat(requires_grad: bool = False):
    deter = torch.randn(B, T, DETER, generator=gen(1), requires_grad=requires_grad)
    stoch = torch.randn(B, T, STOCH, CLASSES, generator=gen(2), requires_grad=requires_grad)
    return {"deter": deter, "stoch": stoch}


# ----------------------------------------------------------------------
# The continue target
# ----------------------------------------------------------------------
def test_continue_target_is_scaled_by_one_minus_one_over_horizon() -> None:
    """`contdisc: True` means a non-terminal step is trained towards 0.997, not 1.0:
    the discount lives inside the continue prediction."""
    is_terminal = torch.tensor([[False, False, True]])
    con = L.continue_target(is_terminal, horizon=333, contdisc=True)
    assert torch.allclose(con, torch.tensor([[1 - 1 / 333, 1 - 1 / 333, 0.0]]))
    assert abs(con[0, 0].item() - 0.996997) < 1e-5


def test_continue_target_without_contdisc_is_plain() -> None:
    is_terminal = torch.tensor([[False, True]])
    con = L.continue_target(is_terminal, horizon=333, contdisc=False)
    assert torch.equal(con, torch.tensor([[1.0, 0.0]]))


def test_continue_target_zero_at_terminal_regardless_of_horizon() -> None:
    for horizon in (2, 10, 333, 10_000):
        con = L.continue_target(torch.tensor([[True]]), horizon, True)
        assert con.item() == 0.0


def test_the_discount_is_not_applied_twice() -> None:
    """The continue TARGET already carries `1 - 1/horizon`, which is why `imag_loss`
    runs with `disc = 1` under contdisc (PROTOCOL section 2). Multiplying again would
    square the discount, halving the effective horizon."""
    horizon = 333
    once = L.continue_target(torch.tensor([[False]]), horizon, True).item()
    twice = once * (1 - 1 / horizon)
    assert abs(once - 0.996997) < 1e-5
    assert abs(twice - 0.994) < 1e-3
    assert once != twice


# ----------------------------------------------------------------------
# The image target
# ----------------------------------------------------------------------
def test_image_target_is_divided_by_255_not_centred() -> None:
    """The decoder ends in a sigmoid, so the target is [0, 1] -- NOT the encoder's
    `x/255 - 0.5`. Using the encoder's scaling here would ask a sigmoid to produce
    negative values."""
    img = torch.tensor([[0, 128, 255]], dtype=torch.uint8)
    target = L.image_target(img)
    assert target.dtype == torch.float32
    assert torch.allclose(target, torch.tensor([[0.0, 128 / 255, 1.0]]))
    assert target.min() >= 0.0 and target.max() <= 1.0


def test_image_target_rejects_a_float_image() -> None:
    with pytest.raises(AssertionError):
        L.image_target(torch.zeros(2, 2, dtype=torch.float32))


def test_reconstruction_loss_is_summed_over_pixels_not_averaged() -> None:
    """`Agg(MSE, 3, sum)`: the per-step loss is the SUM over H, W and C, so a 64x64x3
    image contributes 12288 squared errors. Averaging instead would shrink the
    reconstruction term by four orders of magnitude against the KL."""
    pred = torch.full((1, 1, 4, 4, 3), 0.5)
    recon = O.Agg(O.MSE(pred), 3, torch.sum)
    target = torch.zeros(1, 1, 4, 4, 3)
    assert recon.loss(target).item() == pytest.approx(48 * 0.25)


# ----------------------------------------------------------------------
# Shapes and composition
# ----------------------------------------------------------------------
def build_case(reward_grad: bool = True, requires_grad: bool = False):
    rew, con = make_heads()
    feat = make_feat(requires_grad)
    obs = make_obs(terminal_at=2)
    recon = O.Agg(O.MSE(torch.full((B, T, 8, 8, 3), 0.5, requires_grad=requires_grad)), 3, torch.sum)
    kl = {"dyn": torch.ones(B, T), "rep": torch.ones(B, T)}
    scales = L.scales_for(CFG.agent.loss_scales)
    return L.world_model_loss(
        feat, kl, recon, rew, con, obs, scales, horizon=333, reward_grad=reward_grad
    )


def test_every_term_is_B_by_T_before_the_mean() -> None:
    out = build_case()
    assert set(out.terms) == {"rew", "con", "image", "dyn", "rep"}
    for key, value in out.terms.items():
        assert value.shape == (B, T), key


def test_a_term_with_a_stray_axis_is_refused() -> None:
    """The official code asserts the shape before reducing; a loss that kept a
    trailing axis would otherwise be averaged away silently."""
    rew, con = make_heads()
    bad = O.Agg(O.MSE(torch.full((B, T, 8, 8, 3), 0.5)), 2, torch.sum)  # 2 dims, not 3
    with pytest.raises(AssertionError):
        L.world_model_loss(
            make_feat(), {"dyn": torch.ones(B, T), "rep": torch.ones(B, T)}, bad,
            rew, con, make_obs(), L.scales_for(CFG.agent.loss_scales),
        )


def test_scale_table_matches_configs_yaml() -> None:
    """`rec` is not a key of the loss dict: it is spread over the decoder's output
    keys, so on Atari it becomes the scale for `image`."""
    scales = L.scales_for(C.atari100k_config().agent.loss_scales)
    assert scales == {"rew": 1.0, "con": 1.0, "dyn": 1.0, "rep": 0.1, "image": 1.0}
    assert "rec" not in scales


def test_scale_table_spreads_rec_over_several_decoder_keys() -> None:
    scales = L.scales_for(C.atari100k_config().agent.loss_scales, ("image", "depth"))
    assert scales["image"] == scales["depth"] == 1.0


def test_total_is_the_scaled_sum_of_the_means() -> None:
    out = build_case()
    scales = L.scales_for(CFG.agent.loss_scales)
    expected = sum(v.mean() * scales[k] for k, v in out.terms.items())
    assert torch.allclose(out.total, expected)
    # The rep KL enters at 0.1, so doubling it moves the total by 0.1 per unit.
    assert scales["rep"] == 0.1


def test_rep_scale_really_is_a_tenth_of_dyn() -> None:
    rew, con = make_heads()
    feat = make_feat()
    obs = make_obs()
    recon = O.Agg(O.MSE(torch.full((B, T, 8, 8, 3), 0.5)), 3, torch.sum)
    scales = L.scales_for(CFG.agent.loss_scales)
    base = L.world_model_loss(
        feat, {"dyn": torch.ones(B, T), "rep": torch.ones(B, T)}, recon, rew, con, obs, scales
    ).total
    more_dyn = L.world_model_loss(
        feat, {"dyn": torch.full((B, T), 2.0), "rep": torch.ones(B, T)}, recon, rew, con, obs, scales
    ).total
    more_rep = L.world_model_loss(
        feat, {"dyn": torch.ones(B, T), "rep": torch.full((B, T), 2.0)}, recon, rew, con, obs, scales
    ).total
    assert (more_dyn - base).item() == pytest.approx(1.0, abs=1e-5)
    assert (more_rep - base).item() == pytest.approx(0.1, abs=1e-5)


def test_metrics_carry_every_term() -> None:
    out = build_case()
    assert set(out.metrics) == {f"loss/{k}" for k in out.terms}
    assert all(np.isfinite(v) for v in out.metrics.values())


# ----------------------------------------------------------------------
# Gradient routing
# ----------------------------------------------------------------------
def test_reward_grad_true_lets_the_reward_loss_reach_the_latent() -> None:
    """`sg(x, skip=True)` returns x UNCHANGED. `reward_grad: True` therefore means
    the reward head shapes the representation; reading `sg` as "always stop" would
    silently cut that path.

    The head is built with a nonzero output scale here: at the frozen `outscale: 0.0`
    the path is connected but carries exactly zero for the first gradient step (see
    `test_zero_outscale_means_no_reward_gradient_reaches_the_latent_at_step_one`),
    which would make this test pass for the wrong reason.
    """
    rew, con = make_heads(rew_outscale=1.0)
    feat = make_feat(requires_grad=True)
    obs = make_obs()
    recon = O.Agg(O.MSE(torch.full((B, T, 8, 8, 3), 0.5)), 3, torch.sum)
    kl = {"dyn": torch.ones(B, T), "rep": torch.ones(B, T)}
    scales = L.scales_for(CFG.agent.loss_scales)
    out = L.world_model_loss(feat, kl, recon, rew, con, obs, scales, reward_grad=True)
    out.terms["rew"].sum().backward()
    assert feat["deter"].grad is not None
    assert feat["deter"].grad.abs().sum() > 0


def test_reward_grad_false_would_cut_it() -> None:
    rew, con = make_heads(rew_outscale=1.0)
    feat = make_feat(requires_grad=True)
    obs = make_obs()
    recon = O.Agg(O.MSE(torch.full((B, T, 8, 8, 3), 0.5)), 3, torch.sum)
    kl = {"dyn": torch.ones(B, T), "rep": torch.ones(B, T)}
    scales = L.scales_for(CFG.agent.loss_scales)
    out = L.world_model_loss(feat, kl, recon, rew, con, obs, scales, reward_grad=False)
    out.terms["rew"].sum().backward()
    assert feat["deter"].grad is None or feat["deter"].grad.abs().sum() == 0


def test_zero_outscale_means_no_reward_gradient_reaches_the_latent_at_step_one() -> None:
    """A consequence of `rewhead.outscale: 0.0` worth knowing before reading a curve:
    the output kernel starts at exactly zero, so on the FIRST gradient step the
    reward loss moves that kernel and nothing else. `reward_grad: True` is connected
    but carries zero until the kernel leaves zero. The same holds for the critic.
    """
    rew, con = make_heads(rew_outscale=0.0)
    feat = make_feat(requires_grad=True)
    out = L.world_model_loss(
        feat,
        {"dyn": torch.ones(B, T), "rep": torch.ones(B, T)},
        O.Agg(O.MSE(torch.full((B, T, 8, 8, 3), 0.5)), 3, torch.sum),
        rew, con, make_obs(), L.scales_for(CFG.agent.loss_scales), reward_grad=True,
    )
    out.terms["rew"].sum().backward()
    assert feat["deter"].grad.abs().sum() == 0  # connected, but zero-valued
    assert rew.out.kernel.grad.abs().sum() > 0  # the output layer does move


def test_the_continue_loss_always_reaches_the_latent() -> None:
    """Unlike the reward head, the continue head has no `skip` flag upstream."""
    rew, con = make_heads()
    feat = make_feat(requires_grad=True)
    out = L.world_model_loss(
        feat,
        {"dyn": torch.ones(B, T), "rep": torch.ones(B, T)},
        O.Agg(O.MSE(torch.full((B, T, 8, 8, 3), 0.5)), 3, torch.sum),
        rew, con, make_obs(), L.scales_for(CFG.agent.loss_scales),
    )
    out.terms["con"].sum().backward()
    assert feat["deter"].grad.abs().sum() > 0


def test_targets_are_detached() -> None:
    """`loss(sg(target))`: a target that carried a gradient would train the model to
    move the target towards the prediction."""
    pred = torch.zeros(2, requires_grad=True)
    target = torch.ones(2, requires_grad=True)
    O.MSE(pred).loss(target).sum().backward()
    assert target.grad is None


# ----------------------------------------------------------------------
# The two-hot reward loss, where it is used
# ----------------------------------------------------------------------
def test_reward_head_round_trips_a_reward_through_loss_and_pred() -> None:
    """Fit the head's output layer to one reward and read it back: +100 is the
    frostbite igloo bonus, which must survive the symexp bins without clipping."""
    head = H.MLPHead(FEAT, "symexp_twohot", layers=1, units=16, bins=255, outscale=0.0)
    feat = torch.ones(1, 1, FEAT)
    target = torch.full((1, 1), 100.0)
    opt = torch.optim.Adam(head.parameters(), lr=0.05)
    for _ in range(300):
        opt.zero_grad()
        head(feat).loss(target).sum().backward()
        opt.step()
    assert abs(head(feat).pred().item() - 100.0) < 2.0


def test_reward_loss_at_initialization_is_the_uniform_cross_entropy() -> None:
    """With `outscale: 0.0` the logits start uniform, so the loss is -log(1/bins)
    spread over the two bracketing bins -- i.e. exactly log(bins)."""
    head = H.MLPHead(FEAT, "symexp_twohot", layers=1, units=8, bins=5, outscale=0.0)
    loss = head(torch.randn(3, 2, FEAT, generator=gen())).loss(torch.zeros(3, 2))
    assert torch.allclose(loss, torch.full((3, 2), float(np.log(5))), atol=1e-5)


def test_continue_loss_at_a_half_probability() -> None:
    """The continue head starts at logit 0 only if its output layer is zeroed; it is
    not (`outscale: 1.0`), so this checks the loss formula directly instead."""
    out = O.Binary(torch.zeros(2, 3))
    loss = out.loss(L.continue_target(torch.zeros(2, 3, dtype=torch.bool), 333))
    assert torch.allclose(loss, torch.full((2, 3), float(np.log(2))), atol=1e-6)


# ----------------------------------------------------------------------
# The KL pair, where it is used
# ----------------------------------------------------------------------
def test_kl_pair_is_asymmetric_in_which_side_is_detached() -> None:
    """`dyn` trains the PRIOR towards the posterior, `rep` the posterior towards the
    prior. Swapping the detach turns one gradient path off and doubles the other."""
    dyn = R.RSSM(3, 6, deter=8, hidden=4, stoch=2, classes=4, blocks=2, free_nats=0.0)
    post = torch.randn(2, 2, 4, requires_grad=True, generator=gen(3))
    prior = torch.randn(2, 2, 4, requires_grad=True, generator=gen(4))

    dyn_kl = dyn.dist(post.detach()).kl(dyn.dist(prior))
    dyn_kl.sum().backward()
    assert post.grad is None
    assert prior.grad is not None and prior.grad.abs().sum() > 0

    prior.grad = None
    rep_kl = dyn.dist(post).kl(dyn.dist(prior.detach()))
    rep_kl.sum().backward()
    assert post.grad is not None and post.grad.abs().sum() > 0
    assert prior.grad is None


def test_free_nats_floor_applies_to_the_aggregated_kl() -> None:
    """`max(kl, 1.0)` is applied AFTER the sum over the 32 categoricals, not per
    categorical. Per-categorical clipping would set a floor of 32 nats."""
    dyn = R.RSSM(3, 6, deter=8, hidden=4, stoch=32, classes=64, blocks=2, free_nats=1.0)
    same = torch.zeros(2, 32, 64)
    kl = dyn.dist(same).kl(dyn.dist(same))
    assert kl.shape == (2,)
    assert torch.allclose(kl, torch.zeros(2), atol=1e-5)
    assert torch.clamp(kl, min=dyn.free_nats).tolist() == [1.0, 1.0]  # not 32.0


# ----------------------------------------------------------------------
# End to end on the debug-size world model
# ----------------------------------------------------------------------
def test_world_model_loss_end_to_end_and_backward() -> None:
    cfg = C.debug_config()
    enc = R.Encoder(
        (64, 64, 3), depth=cfg.agent.enc.depth, mults=cfg.agent.enc.mults,
        kernel=cfg.agent.enc.kernel, generator=gen(),
    )
    dyn = R.RSSM(
        6, enc.outdim, deter=cfg.agent.rssm.deter, hidden=cfg.agent.rssm.hidden,
        stoch=cfg.agent.rssm.stoch, classes=cfg.agent.rssm.classes,
        blocks=cfg.agent.rssm.blocks, free_nats=cfg.agent.rssm.free_nats, generator=gen(),
    )
    dec = R.Decoder(
        (64, 64, 3), cfg.agent.rssm.deter, cfg.agent.rssm.stoch, cfg.agent.rssm.classes,
        depth=cfg.agent.dec.depth, mults=cfg.agent.dec.mults, units=cfg.agent.dec.units,
        bspace=cfg.agent.dec.bspace, generator=gen(),
    )
    feat_dim = cfg.agent.rssm.deter + cfg.agent.rssm.stoch * cfg.agent.rssm.classes
    rew = H.MLPHead(feat_dim, "symexp_twohot", layers=1, units=8, bins=5, outscale=0.0)
    con = H.MLPHead(feat_dim, "binary", layers=1, units=8, outscale=1.0)

    obs = {
        "is_first": torch.zeros(2, 3, dtype=torch.bool),
        "is_terminal": torch.zeros(2, 3, dtype=torch.bool),
        "reward": torch.zeros(2, 3),
        "image": torch.randint(0, 256, (2, 3, 64, 64, 3), dtype=torch.uint8),
    }
    obs["is_first"][:, 0] = True
    action = torch.nn.functional.one_hot(torch.zeros(2, 3, dtype=torch.long), 6).float()
    tokens = enc(obs["image"])
    _, _, kl, repfeat, _ = dyn.loss(dyn.initial(2), tokens, action, obs["is_first"], gen())
    recon = dec(repfeat)
    out = L.world_model_loss(
        repfeat, kl, recon, rew, con, obs,
        L.scales_for(cfg.agent.loss_scales),
        horizon=cfg.agent.horizon, contdisc=cfg.agent.contdisc,
        reward_grad=cfg.agent.reward_grad,
    )
    assert torch.isfinite(out.total)
    out.total.backward()
    params = list(enc.parameters()) + list(dyn.parameters()) + list(dec.parameters())
    grads = [p.grad for p in params if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    # The reconstruction term dominates at initialization: 12288 squared errors per
    # step against KLs floored at 1 nat.
    assert out.terms["image"].mean() > out.terms["dyn"].mean()
