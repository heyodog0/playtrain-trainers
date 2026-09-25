"""U06 — imagination, actor-critic and optimizer, against `imag_loss`, `repl_loss`,
`lambda_return`, `_make_opt`, `embodied/jax/opt.py` and `embodied/jax/utils.py`.

`lambda_return` is checked against a brute-force recursion written from the
definition, on random inputs including terminals and truncations; the optimizer chain
is checked step by step against hand arithmetic, including that the RMS comes BEFORE
the momentum.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from playtrain_trainers.dreamerv3 import ac as A
from playtrain_trainers.dreamerv3 import config as C
from playtrain_trainers.dreamerv3 import opt as OPT
from playtrain_trainers.dreamerv3 import outs as O


def gen(seed: int = 0) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


# ----------------------------------------------------------------------
# lambda_return
# ----------------------------------------------------------------------
def brute_force_lambda_return(last, term, rew, boot, disc, lam):
    """Written from the recursion in `agent.lambda_return`, one scalar at a time.

    ret_t = rew_{t+1} + (1 - cont_t) * live_t * boot_{t+1}
                      + live_t * cont_t * ret_{t+1}
    with live_t = (1 - term_{t+1}) * disc, cont_t = (1 - last_{t+1}) * lam,
    and the recursion seeded at ret_{T-1} = boot_{T-1}.
    """
    B, T = rew.shape
    out = np.zeros((B, T - 1), np.float64)
    for b in range(B):
        nxt = float(boot[b, T - 1])
        for t in reversed(range(T - 1)):
            live = (1 - float(term[b, t + 1])) * disc
            cont = (1 - float(last[b, t + 1])) * lam
            ret = float(rew[b, t + 1]) + (1 - cont) * live * float(boot[b, t + 1])
            ret += live * cont * nxt
            out[b, t] = ret
            nxt = ret
    return out


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_lambda_return_against_brute_force(seed: int) -> None:
    g = gen(seed)
    B, T = 3, 7
    rew = torch.randn(B, T, generator=g)
    boot = torch.randn(B, T, generator=g)
    term = (torch.rand(B, T, generator=g) < 0.2).float()
    last = (torch.rand(B, T, generator=g) < 0.2).float()
    got = A.lambda_return(last, term, rew, boot, boot, 0.997, 0.95)
    want = brute_force_lambda_return(last, term, rew, boot, 0.997, 0.95)
    assert got.shape == (B, T - 1)
    assert np.allclose(got.numpy(), want, atol=1e-5)


def test_lambda_return_with_no_terminals_is_the_standard_recursion() -> None:
    B, T = 1, 5
    rew = torch.ones(B, T)
    boot = torch.zeros(B, T)
    zeros = torch.zeros(B, T)
    got = A.lambda_return(zeros, zeros, rew, boot, boot, 1.0, 1.0)
    # disc = lam = 1, no bootstrap: ret_t is the sum of the remaining rewards.
    assert got[0].tolist() == [4.0, 3.0, 2.0, 1.0]


def test_terminal_kills_both_the_bootstrap_and_the_recursion() -> None:
    B, T = 1, 4
    rew = torch.tensor([[0.0, 1.0, 1.0, 1.0]])
    boot = torch.full((B, T), 100.0)
    last = torch.zeros(B, T)
    term = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
    got = A.lambda_return(last, term, rew, boot, boot, 1.0, 1.0)
    # At t=1 the next state is terminal: live=0, so no bootstrap and no tail.
    assert got[0, 1].item() == 1.0


def test_truncation_keeps_the_bootstrap_but_cuts_the_tail() -> None:
    """`last` without `term` is a truncation: the value function still bootstraps,
    which is the difference between "the episode ended" and "we stopped looking"."""
    B, T = 1, 4
    rew = torch.zeros(B, T)
    boot = torch.full((B, T), 7.0)
    term = torch.zeros(B, T)
    last = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
    got = A.lambda_return(last, term, rew, boot, boot, 1.0, 0.95)
    # At t=1 the next step is `last`: cont = 0, so ret = (1-0)*live*boot = 7.
    assert got[0, 1].item() == pytest.approx(7.0)


def test_lambda_return_ignores_its_val_argument() -> None:
    """Upstream passes `val` only so the shape assertion covers it; the body reads
    `boot`. Copied, and pinned here so a future reader does not 'fix' it."""
    B, T = 2, 5
    rew = torch.randn(B, T, generator=gen())
    boot = torch.randn(B, T, generator=gen(1))
    zeros = torch.zeros(B, T)
    a = A.lambda_return(zeros, zeros, rew, torch.zeros(B, T), boot, 0.99, 0.95)
    b = A.lambda_return(zeros, zeros, rew, torch.full((B, T), 999.0), boot, 0.99, 0.95)
    assert torch.equal(a, b)


def test_lambda_return_requires_equal_shapes() -> None:
    with pytest.raises(AssertionError):
        A.lambda_return(
            torch.zeros(2, 3), torch.zeros(2, 3), torch.zeros(2, 3),
            torch.zeros(2, 3), torch.zeros(2, 4), 1.0, 1.0,
        )


# ----------------------------------------------------------------------
# Normalize
# ----------------------------------------------------------------------
def test_perc_normalizer_tracks_the_5th_and_95th_percentiles() -> None:
    norm = A.Normalize("perc", rate=0.5, limit=1e-8, perclo=5.0, perchi=95.0, debias=False)
    x = torch.linspace(0, 100, 1001)
    for _ in range(40):  # let the EMA converge
        norm(x, update=True)
    lo, scale = norm.stats()
    assert abs(float(lo) - 5.0) < 0.5
    assert abs(float(scale) - 90.0) < 1.0


def test_perc_limit_floors_the_scale_at_one() -> None:
    """`limit: 1.0` means returns smaller than 1 are never scaled UP. Without it an
    agent that has found nothing yet divides a near-zero spread into a huge
    advantage."""
    norm = A.Normalize("perc", rate=1.0, limit=1.0, debias=False)
    norm(torch.linspace(0.0, 0.001, 100), update=True)
    _, scale = norm.stats()
    assert float(scale) == 1.0


def test_none_normalizer_is_the_identity() -> None:
    norm = A.Normalize("none")
    offset, scale = norm(torch.randn(50, generator=gen()), update=True)
    assert float(offset) == 0.0 and float(scale) == 1.0


def test_normalizer_ema_rate() -> None:
    norm = A.Normalize("perc", rate=0.01, limit=1e-8, debias=False)
    norm(torch.full((100,), 10.0), update=True)
    # One update from zero at rate 0.01 moves 1% of the way.
    assert abs(float(norm.hi) - 0.1) < 1e-4


def test_normalizer_stats_are_detached() -> None:
    norm = A.Normalize("perc", rate=0.5, limit=1e-8, debias=False)
    x = torch.linspace(0, 10, 100, requires_grad=True)
    lo, scale = norm(x, update=True)
    assert not lo.requires_grad and not scale.requires_grad


def test_debias_correction_when_enabled() -> None:
    """`debias: False` at the frozen config, but the branch is ported: the
    correction divides by the EMA of 1.0, undoing the cold-start bias."""
    norm = A.Normalize("perc", rate=0.1, limit=1e-8, debias=True)
    norm(torch.full((100,), 10.0), update=True)
    lo, _ = norm.stats()
    assert abs(float(norm.corr) - 0.1) < 1e-6
    assert abs(float(lo) - 10.0) < 1e-4  # corrected back to the true value


# ----------------------------------------------------------------------
# SlowModel
# ----------------------------------------------------------------------
def make_linear(value: float) -> torch.nn.Module:
    lin = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        lin.weight.fill_(value)
    return lin


def test_slow_model_starts_as_an_exact_copy() -> None:
    src = make_linear(3.0)
    slow = A.SlowModel(src, rate=0.02, every=1)
    assert torch.equal(next(slow.model.parameters()), next(src.parameters()))


def test_slow_model_ema_rate_is_two_percent() -> None:
    src = make_linear(0.0)
    slow = A.SlowModel(src, rate=0.02, every=1)
    with torch.no_grad():
        next(src.parameters()).fill_(1.0)
    slow.update()
    assert float(next(slow.model.parameters())[0, 0]) == pytest.approx(0.02)
    slow.update()
    assert float(next(slow.model.parameters())[0, 0]) == pytest.approx(0.02 + 0.98 * 0.02)


def test_slow_model_every_skips_updates() -> None:
    src = make_linear(0.0)
    slow = A.SlowModel(src, rate=0.4, every=3)  # the official assert rejects exactly 0.5
    with torch.no_grad():
        next(src.parameters()).fill_(1.0)
    slow.update()  # count 0 -> mixes
    first = float(next(slow.model.parameters())[0, 0])
    slow.update()  # count 1 -> skipped
    slow.update()  # count 2 -> skipped
    assert float(next(slow.model.parameters())[0, 0]) == first
    slow.update()  # count 3 -> mixes
    assert float(next(slow.model.parameters())[0, 0]) > first


def test_slow_model_parameters_do_not_require_grad() -> None:
    slow = A.SlowModel(make_linear(1.0))
    assert not any(p.requires_grad for p in slow.model.parameters())


# ----------------------------------------------------------------------
# The optimizer chain
# ----------------------------------------------------------------------
def test_agc_clips_an_oversize_gradient_exactly() -> None:
    """`upper = clip * max(pmin, ||param||)`; the update is scaled by
    `1 / max(1, ||update|| / upper)`."""
    param = torch.full((4,), 0.5)  # norm 1.0
    update = torch.full((4,), 5.0)  # norm 10.0
    scale = OPT.agc_scale(param, update, clip=0.3, pmin=1e-3)
    assert scale == pytest.approx(0.3 * 1.0 / 10.0)


def test_agc_never_scales_up() -> None:
    param = torch.full((4,), 5.0)
    tiny = torch.full((4,), 1e-6)
    assert OPT.agc_scale(param, tiny, clip=0.3) == pytest.approx(1.0)


def test_agc_pmin_protects_a_zero_initialized_parameter() -> None:
    """With `outscale: 0.0` the reward head and the critic start at exactly zero. A
    bound of `clip * ||param||` would be zero, clipping their updates to nothing and
    freezing them forever; `pmin` is what stops that."""
    zero = torch.zeros(4)
    update = torch.full((4,), 1.0)  # norm 2.0
    scale = OPT.agc_scale(zero, update, clip=0.3, pmin=1e-3)
    assert scale > 0
    assert scale == pytest.approx(0.3 * 1e-3 / 2.0)


def test_warmup_schedule() -> None:
    assert OPT.warmup_schedule(0, 4e-5, 1000) == 0.0
    assert OPT.warmup_schedule(500, 4e-5, 1000) == pytest.approx(2e-5)
    assert OPT.warmup_schedule(1000, 4e-5, 1000) == pytest.approx(4e-5)
    assert OPT.warmup_schedule(50_000, 4e-5, 1000) == pytest.approx(4e-5)
    assert OPT.warmup_schedule(0, 4e-5, 0) == pytest.approx(4e-5)


def test_first_update_runs_at_zero_learning_rate() -> None:
    p = torch.nn.Parameter(torch.ones(3))
    opt = OPT.LaProp([p], lr=1e-2, warmup=10, agc=0.0)
    p.grad = torch.ones(3)
    opt.step()
    assert torch.equal(p.detach(), torch.ones(3))  # lr was 0 on step 0
    p.grad = torch.ones(3)
    opt.step()
    assert not torch.equal(p.detach(), torch.ones(3))


def test_chain_order_is_rms_then_momentum_by_hand() -> None:
    """LaProp, not Adam. With beta1 = beta2 = 0 the two orders differ: RMS-then-
    momentum gives a unit-magnitude step, momentum-then-RMS gives the same here, so
    the check uses a second step where the histories diverge.
    """
    p = torch.nn.Parameter(torch.zeros(1))
    opt = OPT.LaProp([p], lr=1.0, warmup=0, agc=0.0, beta1=0.9, beta2=0.999, eps=0.0)

    # Step 1: nu = (1-b2) g^2, nu_hat = nu / (1 - b2) = g^2, so g/sqrt(nu_hat) = 1.
    # Then mu = (1-b1) * 1, mu_hat = mu / (1 - b1) = 1. Update = -lr * 1.
    p.grad = torch.tensor([3.0])
    opt.step()
    assert float(p.detach()) == pytest.approx(-1.0, abs=1e-5)

    # Step 2 with a different gradient: work the same arithmetic through by hand.
    b1, b2 = 0.9, 0.999
    nu = (1 - b2) * 9.0
    mu = (1 - b1) * 1.0
    g = 1.0
    nu = b2 * nu + (1 - b2) * g * g
    normed = g / np.sqrt(nu / (1 - b2**2))
    mu = b1 * mu + (1 - b1) * normed
    expected = -1.0 - 1.0 * (mu / (1 - b1**2))
    p.grad = torch.tensor([1.0])
    opt.step()
    assert float(p.detach()) == pytest.approx(expected, abs=1e-5)


def test_adam_would_give_a_different_answer() -> None:
    """Guard against the chain silently becoming Adam: same hyperparameters, same
    gradients, different trajectory."""
    p1 = torch.nn.Parameter(torch.zeros(1))
    p2 = torch.nn.Parameter(torch.zeros(1))
    laprop = OPT.LaProp([p1], lr=0.1, warmup=0, agc=0.0, beta1=0.9, beta2=0.999, eps=1e-20)
    adam = torch.optim.Adam([p2], lr=0.1, betas=(0.9, 0.999), eps=1e-20)
    for g in (3.0, -1.0, 0.5, 2.0):
        p1.grad = torch.tensor([g])
        p2.grad = torch.tensor([g])
        laprop.step()
        adam.step()
    assert abs(float(p1.detach()) - float(p2.detach())) > 1e-3


def test_optimizer_state_is_float32_for_a_bf16_parameter() -> None:
    p = torch.nn.Parameter(torch.zeros(4, dtype=torch.bfloat16))
    opt = OPT.LaProp([p], lr=1e-3, warmup=0)
    p.grad = torch.ones(4, dtype=torch.bfloat16)
    opt.step()
    state = opt.state[p]
    assert state["nu"].dtype == torch.float32
    assert state["mu"].dtype == torch.float32
    assert p.dtype == torch.bfloat16


def test_optimizer_refuses_unported_options() -> None:
    p = torch.nn.Parameter(torch.zeros(1))
    with pytest.raises(NotImplementedError, match="schedule"):
        OPT.LaProp([p], schedule="cosine", anneal=10)
    with pytest.raises(NotImplementedError, match="wd"):
        OPT.LaProp([p], wd=0.1)


def test_optimizer_matches_the_frozen_config() -> None:
    cfg = C.atari100k_config().agent.opt
    p = torch.nn.Parameter(torch.zeros(1))
    opt = OPT.LaProp(
        [p], lr=cfg.lr, agc=cfg.agc, eps=cfg.eps, beta1=cfg.beta1, beta2=cfg.beta2,
        momentum=cfg.momentum, wd=cfg.wd, warmup=cfg.warmup, schedule=cfg.schedule,
        anneal=cfg.anneal,
    )
    group = opt.param_groups[0]
    assert (group["lr"], group["agc"], group["eps"]) == (4e-5, 0.3, 1e-20)
    assert (group["beta1"], group["beta2"], group["warmup"]) == (0.9, 0.999, 1000)


def test_global_norm_and_rms_helpers() -> None:
    p = torch.nn.Parameter(torch.zeros(4))
    p.grad = torch.tensor([3.0, 4.0, 0.0, 0.0])
    assert OPT.global_norm([p]) == pytest.approx(5.0)
    assert OPT.rms([torch.tensor([3.0, 4.0])]) == pytest.approx(np.sqrt(25 / 2))


# ----------------------------------------------------------------------
# imag_loss
# ----------------------------------------------------------------------
def _fresh_norms():
    """A fresh (retnorm, valnorm, advnorm) triple; the normalizers carry EMA state,
    so two calls in one test must not share them."""
    return (
        A.Normalize("perc", limit=1.0, debias=False),
        A.Normalize("none"),
        A.Normalize("none"),
    )


def make_ac_case(BK: int = 4, H: int = 3, bins: int = 5):
    torch.manual_seed(0)
    # `act` is the INDEX tensor the policy sampled, not a one-hot: `imag_loss` calls
    # `policy.logp(sg(act))` and `Categorical.logp` one-hots it itself. (The RSSM core
    # is the place that wants a one-hot; upstream `DictConcat` converts it there.)
    act = torch.randint(0, 3, (BK, H + 1), generator=gen(5))
    rew = torch.randn(BK, H + 1, generator=gen(1))
    con = torch.full((BK, H + 1), 0.997)
    policy = O.Categorical(torch.randn(BK, H + 1, 3, generator=gen(2), requires_grad=True), 0.01)
    bins_t = O.twohot_bins(bins)
    value = O.TwoHot(torch.randn(BK, H + 1, bins, generator=gen(3), requires_grad=True), bins_t)
    slowvalue = O.TwoHot(torch.randn(BK, H + 1, bins, generator=gen(4)), bins_t)
    return act, rew, con, policy, value, slowvalue


def test_imag_loss_shapes_and_horizon_reduction() -> None:
    act, rew, con, policy, value, slowvalue = make_ac_case(BK=6, H=3)
    losses, outs_, metrics = A.imag_loss(
        act, rew, con, policy, value, slowvalue,
        A.Normalize("perc", limit=1.0, debias=False), A.Normalize("none"), A.Normalize("none"),
    )
    assert losses["policy"].shape == (6, 3)  # H, not H+1
    assert losses["value"].shape == (6, 3)
    assert outs_["ret"].shape == (6, 3)
    reduced = A.reduce_imagination_losses(losses, B=2, K=3)
    assert reduced["policy"].shape == (2, 3)


def test_imag_loss_uses_disc_one_under_contdisc() -> None:
    """The discount is already in the continue prediction; `weight = cumprod(con)`
    with disc = 1. With contdisc off it would be `cumprod(0.997 * con) / 0.997`."""
    act, rew, con, policy, value, slowvalue = make_ac_case()
    _, _, m_on = A.imag_loss(
        act, rew, con, policy, value, slowvalue,
        A.Normalize("perc", limit=1.0, debias=False), A.Normalize("none"), A.Normalize("none"),
        contdisc=True,
    )
    _, _, m_off = A.imag_loss(
        act, rew, con, policy, value, slowvalue,
        A.Normalize("perc", limit=1.0, debias=False), A.Normalize("none"), A.Normalize("none"),
        contdisc=False,
    )
    assert m_on["weight"] != m_off["weight"]


def test_imagination_weight_is_the_cumulative_continue_probability() -> None:
    act, rew, con, policy, value, slowvalue = make_ac_case(BK=2, H=3)
    con = torch.tensor([[1.0, 0.5, 0.5, 1.0], [1.0, 1.0, 1.0, 1.0]])
    _, _, metrics = A.imag_loss(
        act, rew, con, policy, value, slowvalue,
        A.Normalize("perc", limit=1.0, debias=False), A.Normalize("none"), A.Normalize("none"),
    )
    expected = torch.cumprod(con, 1).mean()
    assert metrics["weight"] == pytest.approx(float(expected), abs=1e-6)


def test_slow_critic_is_a_regularizer_not_the_bootstrap() -> None:
    """`slowtar: False`. Changing the slow critic must move the VALUE loss (it is a
    regularization target) but not the return (which bootstraps from the live
    critic). Getting this backwards is the DreamerV2 habit."""
    act, rew, con, policy, value, slowvalue = make_ac_case()
    norms = _fresh_norms
    l1, o1, _ = A.imag_loss(act, rew, con, policy, value, slowvalue, *norms(), slowtar=False)
    other = O.TwoHot(torch.randn(*slowvalue.logits.shape, generator=gen(9)), slowvalue.bins)
    l2, o2, _ = A.imag_loss(act, rew, con, policy, value, other, *norms(), slowtar=False)
    assert torch.allclose(o1["ret"], o2["ret"])  # the return did not move
    assert not torch.allclose(l1["value"], l2["value"])  # the value loss did


def test_slowtar_true_would_move_the_return() -> None:
    act, rew, con, policy, value, slowvalue = make_ac_case()
    norms = _fresh_norms
    _, o1, _ = A.imag_loss(act, rew, con, policy, value, slowvalue, *norms(), slowtar=True)
    other = O.TwoHot(torch.randn(*slowvalue.logits.shape, generator=gen(9)), slowvalue.bins)
    _, o2, _ = A.imag_loss(act, rew, con, policy, value, other, *norms(), slowtar=True)
    assert not torch.allclose(o1["ret"], o2["ret"])


def test_advantage_is_scaled_but_not_de_meaned() -> None:
    """`adv = (ret - tarval[:, :-1]) / rscale`: the percentile OFFSET is computed and
    used only for a metric. Subtracting it would change the policy gradient."""
    act, rew, con, policy, value, slowvalue = make_ac_case(BK=8, H=4)
    retnorm = A.Normalize("perc", rate=1.0, limit=1.0, debias=False)
    losses, outs_, metrics = A.imag_loss(
        act, rew, con, policy, value, slowvalue,
        retnorm, A.Normalize("none"), A.Normalize("none"),
    )
    lo, scale = retnorm.stats()
    val = value.pred()
    expected_adv = (outs_["ret"] - val[:, :-1]) / scale
    assert metrics["adv"] == pytest.approx(float(expected_adv.mean().detach()), abs=1e-5)
    assert float(lo) != 0.0  # the offset exists and is simply not used here


def test_policy_loss_gradient_flows_to_the_policy_only() -> None:
    """`Categorical` with a unimix rebuilds its logits, so the leaf to check is the
    tensor handed in, not `policy.logits`."""
    act, rew, con, _, value, slowvalue = make_ac_case()
    raw = torch.randn(*act.shape, 3, generator=gen(2), requires_grad=True)
    policy = O.Categorical(raw, 0.01)
    value_raw = value.logits  # a leaf here: TwoHot stores what it was given
    losses, _, _ = A.imag_loss(
        act, rew, con, policy, value, slowvalue,
        A.Normalize("perc", limit=1.0, debias=False), A.Normalize("none"), A.Normalize("none"),
    )
    losses["policy"].sum().backward(retain_graph=True)
    assert raw.grad is not None and raw.grad.abs().sum() > 0
    assert value_raw.grad is None  # the actor loss does not train the critic


def test_entropy_bonus_sign_and_scale() -> None:
    """`-(logpi * adv + actent * H)`: raising the entropy LOWERS the loss."""
    act, rew, con, policy, value, slowvalue = make_ac_case()
    norms = _fresh_norms
    low, _, _ = A.imag_loss(act, rew, con, policy, value, slowvalue, *norms(), actent=0.0)
    high, _, _ = A.imag_loss(act, rew, con, policy, value, slowvalue, *norms(), actent=1.0)
    assert high["policy"].sum() < low["policy"].sum()


# ----------------------------------------------------------------------
# repl_loss
# ----------------------------------------------------------------------
def test_repl_loss_uses_the_explicit_discount_and_bootstraps_from_imagination() -> None:
    BK, T, bins = 3, 5, 5
    last = torch.zeros(BK, T, dtype=torch.bool)
    term = torch.zeros(BK, T)
    rew = torch.randn(BK, T, generator=gen(1))
    boot = torch.randn(BK, T, generator=gen(2))
    bins_t = O.twohot_bins(bins)
    value = O.TwoHot(torch.randn(BK, T, bins, generator=gen(3), requires_grad=True), bins_t)
    slowvalue = O.TwoHot(torch.randn(BK, T, bins, generator=gen(4)), bins_t)
    losses, outs_, _ = A.repl_loss(
        last, term, rew, boot, value, slowvalue, A.Normalize("none"), horizon=333
    )
    assert losses["repval"].shape == (BK, T - 1)
    want = A.lambda_return(
        last.float(), term, rew, value.pred(), boot, 1 - 1 / 333, 0.95
    )
    assert torch.allclose(outs_["ret"], want, atol=1e-6)


def test_repl_loss_weight_zeroes_the_last_step_of_an_episode() -> None:
    BK, T, bins = 2, 4, 5
    last = torch.zeros(BK, T, dtype=torch.bool)
    last[0, 1] = True
    bins_t = O.twohot_bins(bins)
    value = O.TwoHot(torch.zeros(BK, T, bins, requires_grad=True), bins_t)
    slowvalue = O.TwoHot(torch.zeros(BK, T, bins), bins_t)
    losses, _, _ = A.repl_loss(
        last, torch.zeros(BK, T), torch.zeros(BK, T), torch.zeros(BK, T),
        value, slowvalue, A.Normalize("none"),
    )
    assert losses["repval"][0, 1].item() == 0.0
    assert losses["repval"][1, 1].item() != 0.0


# ----------------------------------------------------------------------
# Imagination assembly
# ----------------------------------------------------------------------
def test_imagination_starts_flattens_every_replayed_step() -> None:
    """`imag_last: 0` -> K = T: every replayed step starts a rollout, so a 16x64
    batch produces 1024 rollouts."""
    entries = {"deter": torch.randn(2, 5, 8, generator=gen()), "stoch": torch.randn(2, 5, 3, 4, generator=gen(1))}
    starts = A.imagination_starts(entries, nlast=5)
    assert starts["deter"].shape == (10, 8)
    assert starts["stoch"].shape == (10, 3, 4)
    assert torch.equal(starts["deter"][0], entries["deter"][0, 0])
    assert torch.equal(starts["deter"][5], entries["deter"][1, 0])


def test_assemble_imagined_prepends_the_real_start_and_detaches() -> None:
    B, K, H = 2, 3, 4
    repfeat = {
        "deter": torch.randn(B, K, 8, generator=gen(), requires_grad=True),
        "stoch": torch.randn(B, K, 2, 4, generator=gen(1), requires_grad=True),
    }
    imgfeat = {
        "deter": torch.randn(B * K, H, 8, generator=gen(2), requires_grad=True),
        "stoch": torch.randn(B * K, H, 2, 4, generator=gen(3), requires_grad=True),
    }
    out = A.assemble_imagined(repfeat, imgfeat, K, ac_grads=False)
    assert out["deter"].shape == (B * K, H + 1, 8)
    assert not out["deter"].requires_grad  # ac_grads False: no path to the world model
    assert torch.equal(out["deter"][0, 0], repfeat["deter"][0, 0].detach())


def test_ac_grads_true_would_open_the_path() -> None:
    B, K, H = 1, 2, 3
    repfeat = {"deter": torch.randn(B, K, 4, generator=gen(), requires_grad=True)}
    imgfeat = {"deter": torch.randn(B * K, H, 4, generator=gen(1), requires_grad=True)}
    out = A.assemble_imagined(repfeat, imgfeat, K, ac_grads=True)
    assert out["deter"].requires_grad


def test_horizon_losses_are_averaged_not_summed() -> None:
    """`v.mean(1)` over the H horizon steps. A sum would scale the actor loss by 15
    against every world-model term."""
    losses = {"policy": torch.ones(6, 15)}
    reduced = A.reduce_imagination_losses(losses, B=2, K=3)
    assert reduced["policy"].shape == (2, 3)
    assert torch.equal(reduced["policy"], torch.ones(2, 3))  # mean of ones, not 15


# ----------------------------------------------------------------------
# Frozen config wiring
# ----------------------------------------------------------------------
def test_frozen_ac_values() -> None:
    cfg = C.atari100k_config().agent
    assert cfg.imag_loss.slowtar is False
    assert cfg.repl_loss.slowtar is False
    assert cfg.imag_loss.lam == cfg.repl_loss.lam == 0.95
    assert cfg.imag_loss.actent == 3e-4
    assert cfg.slowvalue.rate == 0.02 and cfg.slowvalue.every == 1
    assert cfg.retnorm.impl == "perc" and cfg.retnorm.limit == 1.0
    assert cfg.valnorm.impl == cfg.advnorm.impl == "none"
    assert cfg.imag_length == 15 and cfg.imag_last == 0
    assert cfg.horizon == 333 and cfg.contdisc is True
    assert cfg.ac_grads is False


# ----------------------------------------------------------------------
# D-028: a tensor policy is a per-step action sequence
# ----------------------------------------------------------------------
def test_imagine_with_a_tensor_policy_indexes_along_time() -> None:
    """Upstream scans a non-callable policy along the time axis
    (`nj.scan(..., nn.cast(policy), length, axis=1)`), so step t uses `policy[:, t]`.

    Ours used to pass the WHOLE array at every step. Training never hit the branch --
    it always supplies a callable -- so the bug was invisible until the open-loop
    report, which is the one caller that feeds recorded actions.
    """
    from playtrain_trainers.dreamerv3 import rssm as R

    A, H = 4, 3
    dyn = R.RSSM(A, 6, deter=8, hidden=4, stoch=2, classes=4, blocks=2)
    carry = dyn.initial(2)
    # A distinct one-hot action per step, so using the wrong one is detectable.
    acts = torch.zeros(2, H, A)
    for t in range(H):
        acts[:, t, t % A] = 1.0
    _, feat, used = dyn.imagine(carry, acts, H, gen())
    assert used.shape == (2, H, A)
    assert torch.equal(used, acts)  # each step got its own action, in order
    assert feat["deter"].shape == (2, H, 8)


def test_imagine_tensor_policy_differs_from_a_constant_action() -> None:
    """Guard the regression directly: a time-varying action sequence must not give
    the same rollout as holding one action fixed."""
    from playtrain_trainers.dreamerv3 import rssm as R

    A, H = 4, 3
    dyn = R.RSSM(A, 6, deter=8, hidden=4, stoch=2, classes=4, blocks=2)
    varying = torch.zeros(1, H, A)
    for t in range(H):
        varying[:, t, t % A] = 1.0
    fixed = torch.zeros(1, H, A)
    fixed[:, :, 0] = 1.0
    _, f1, _ = dyn.imagine(dyn.initial(1), varying, H, gen(1))
    _, f2, _ = dyn.imagine(dyn.initial(1), fixed, H, gen(1))
    assert not torch.allclose(f1["deter"], f2["deter"])
