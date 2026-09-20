"""The `core` option, and the promise that adding it left the LSTM untouched.

The LSTM regression replays `tests/data/lstm_golden.pt`, frozen from the
pre-change revision by `tools/make_lstm_golden.py`, and demands bit-identical
output. Any drift in the LSTM path is a bug, not a tolerance to widen.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from playtrain_trainers.impala.fwp import CORE_CLASSES, CompFWPCore, DeltaNetCore
from playtrain_trainers.impala.net import CORES, FWP_CORES, ImpalaNet, resolve_core

GOLDEN = Path(__file__).parent / "data" / "lstm_golden.pt"


@pytest.fixture(scope="module")
def golden():
    if not GOLDEN.exists():  # pragma: no cover - regenerate with the tool
        pytest.skip(f"{GOLDEN} missing; run tools/make_lstm_golden.py")
    return torch.load(GOLDEN, weights_only=False)


def build_from_golden(golden, **kwargs) -> ImpalaNet:
    model = ImpalaNet(**golden["spec"], **kwargs)
    model.load_state_dict(golden["state_dict"])
    model.eval()
    return model


def assert_matches_golden(model: ImpalaNet, golden) -> None:
    with torch.no_grad():
        for name, case in golden["cases"].items():
            out, state = model(case["inputs"], case["core_state_in"])
            for key, want in case["outputs"].items():
                assert torch.equal(out[key], want), f"{name}/{key} drifted"
            assert len(state) == len(case["core_state_out"])
            for i, (got, want) in enumerate(zip(state, case["core_state_out"])):
                assert torch.equal(got, want), f"{name}/core_state[{i}] drifted"


def test_core_lstm_is_bit_identical_to_the_frozen_baseline(golden):
    assert_matches_golden(build_from_golden(golden, core="lstm"), golden)


def test_legacy_use_lstm_flag_is_bit_identical_too(golden):
    """Existing configs say use_lstm=True and must be completely unaffected."""
    assert_matches_golden(build_from_golden(golden, use_lstm=True), golden)


def test_the_two_spellings_produce_the_same_module(golden):
    by_flag = build_from_golden(golden, use_lstm=True)
    by_core = build_from_golden(golden, core="lstm")
    assert by_flag.core_kind == by_core.core_kind == "lstm"
    assert by_flag.use_lstm and by_core.use_lstm
    assert set(by_flag.state_dict()) == set(by_core.state_dict())


@pytest.mark.parametrize(
    ("core", "use_lstm", "expected"),
    [
        ("", False, "ff"),
        ("", True, "lstm"),
        (None, False, "ff"),
        ("ff", False, "ff"),
        ("lstm", False, "lstm"),
        ("lstm", True, "lstm"),
        ("deltanet", False, "deltanet"),
        ("compfwp", False, "compfwp"),
    ],
)
def test_resolve_core(core, use_lstm, expected):
    assert resolve_core(core, use_lstm) == expected


def test_resolve_core_rejects_contradictions_and_typos():
    with pytest.raises(ValueError, match="contradicts"):
        resolve_core("deltanet", True)
    with pytest.raises(ValueError, match="core must be one of"):
        resolve_core("lstmm", False)


def test_feedforward_is_unchanged_under_either_spelling():
    torch.manual_seed(0)
    a = ImpalaNet((3, 32, 32), 5, features_dim=32)
    torch.manual_seed(0)
    b = ImpalaNet((3, 32, 32), 5, features_dim=32, core="ff")
    assert a.core_kind == b.core_kind == "ff"
    assert not a.use_lstm and not b.use_lstm
    assert a.initial_state(4) == () == b.initial_state(4)

    inputs = {
        "frame": torch.randint(0, 256, (2, 3, 3, 32, 32), dtype=torch.uint8),
        "reward": torch.zeros(2, 3),
        "done": torch.zeros(2, 3, dtype=torch.bool),
        "last_action": torch.zeros(2, 3, dtype=torch.int64),
    }
    a.eval(), b.eval()
    with torch.no_grad():
        assert torch.equal(a(inputs)[0]["baseline"], b(inputs)[0]["baseline"])


def test_initial_state_contract_tuple_of_tensors_batch_at_dim_1():
    """Buffers, actors and the learner assume exactly this and nothing more."""
    model = ImpalaNet((3, 32, 32), 5, features_dim=32, core="lstm")
    state = model.initial_state(batch_size=7)
    assert isinstance(state, tuple) and state
    for tensor in state:
        assert isinstance(tensor, torch.Tensor)
        assert tensor.ndim == 3
        assert tensor.shape[1] == 7
        assert not tensor.any()  # zero state


def test_golden_records_the_same_initial_state_shapes(golden):
    model = build_from_golden(golden, core="lstm")
    shapes = [tuple(s.shape) for s in model.initial_state(5)]
    assert shapes == [tuple(s) for s in golden["initial_state_shapes"]]


@pytest.mark.parametrize("core", FWP_CORES)
def test_every_fast_weight_core_validates_as_config(core):
    assert core in CORES
    assert resolve_core(core, False) == core


def test_every_declared_fast_weight_core_has_a_module():
    """A core the config accepts but nothing implements must not slip through."""
    assert set(FWP_CORES) == set(CORE_CLASSES)


def test_an_unknown_core_kind_fails_loudly():
    from playtrain_trainers.impala.fwp import build_fwp_core

    with pytest.raises(NotImplementedError, match="nonesuch"):
        build_fwp_core("nonesuch", 32, 16, 4)


# --------------------------------------------------------------- FWP cores

FWP_SPEC = {"observation_shape": (3, 32, 32), "num_actions": 5, "features_dim": 32}
FWP_KW = {"fwp_dim": 16, "fwp_heads": 4}


def build_core(core: str, **over) -> ImpalaNet:
    torch.manual_seed(0)
    model = ImpalaNet(**FWP_SPEC, core=core, **{**FWP_KW, **over})
    model.eval()  # argmax action, so single-step and batched runs compare
    return model


def fwp_inputs(T: int, B: int, done_at=(), seed: int = 0) -> dict:
    gen = torch.Generator().manual_seed(seed)
    done = torch.zeros(T, B, dtype=torch.bool)
    for t, b in done_at:
        done[t, b] = True
    return {
        "frame": torch.randint(
            0, 256, (T, B, *FWP_SPEC["observation_shape"]), dtype=torch.uint8, generator=gen
        ),
        "reward": torch.randn(T, B, generator=gen),
        "done": done,
        "last_action": torch.zeros(T, B, dtype=torch.int64),
    }


def slice_step(inputs: dict, t: int) -> dict:
    return {k: v[t : t + 1] for k, v in inputs.items()}


BUILT_CORES = ["deltanet", "compfwp"]


@pytest.mark.parametrize("core", BUILT_CORES)
def test_initial_state_is_a_one_tuple_with_batch_at_dim_1(core):
    model = build_core(core)
    state = model.initial_state(batch_size=6)
    assert isinstance(state, tuple) and len(state) == 1
    (S,) = state
    assert S.shape == (1, 6, FWP_KW["fwp_dim"] ** 2)
    assert not S.any()


@pytest.mark.parametrize("core", BUILT_CORES)
def test_t_step_forward_equals_t_single_steps(core):
    """The learner's T>1 unroll and the actor's T=1 path must agree."""
    model = build_core(core)
    T, B = 6, 3
    inputs = fwp_inputs(T, B, seed=1)

    with torch.no_grad():
        batched, batched_state = model(inputs, model.initial_state(B))

        state = model.initial_state(B)
        steps = []
        for t in range(T):
            out, state = model(slice_step(inputs, t), state)
            steps.append(out)

    for key in ("policy_logits", "baseline"):
        stacked = torch.cat([s[key] for s in steps], dim=0)
        torch.testing.assert_close(batched[key], stacked, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(batched_state[0], state[0], atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("core", BUILT_CORES)
def test_t_step_equals_single_steps_across_episode_boundaries(core):
    model = build_core(core)
    T, B = 6, 3
    inputs = fwp_inputs(T, B, done_at=((0, 1), (2, 0), (4, 2)), seed=2)

    with torch.no_grad():
        batched, batched_state = model(inputs, model.initial_state(B))
        state = model.initial_state(B)
        steps = []
        for t in range(T):
            out, state = model(slice_step(inputs, t), state)
            steps.append(out)

    stacked = torch.cat([s["baseline"] for s in steps], dim=0)
    torch.testing.assert_close(batched["baseline"], stacked, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(batched_state[0], state[0], atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("core", BUILT_CORES)
def test_state_carried_across_two_unrolls_equals_one_long_unroll(core):
    model = build_core(core)
    T, B = 8, 3
    inputs = fwp_inputs(T, B, done_at=((3, 1),), seed=3)

    def window(lo, hi):
        return {k: v[lo:hi] for k, v in inputs.items()}

    with torch.no_grad():
        long_out, long_state = model(inputs, model.initial_state(B))
        first, mid = model(window(0, 5), model.initial_state(B))
        second, split_state = model(window(5, 8), mid)

    joined = torch.cat([first["baseline"], second["baseline"]], dim=0)
    torch.testing.assert_close(long_out["baseline"], joined, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(long_state[0], split_state[0], atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("core", BUILT_CORES)
def test_state_is_zeroed_entering_a_done_step(core):
    """done[t] marks frame[t] as a fresh episode, so the state going in is zero."""
    model = build_core(core)
    B = 2
    inputs = fwp_inputs(1, B, done_at=((0, 0),), seed=4)

    dirty = (torch.randn(1, B, FWP_KW["fwp_dim"] ** 2),)
    with torch.no_grad():
        _, after_dirty = model(inputs, dirty)
        clean = (dirty[0].clone(),)
        clean[0][:, 0] = 0.0
        _, after_clean = model(inputs, clean)

    # env 0 reset, so its outgoing state cannot depend on what was there before
    torch.testing.assert_close(after_dirty[0][:, 0], after_clean[0][:, 0])
    assert not torch.allclose(after_dirty[0][:, 1], torch.zeros(()))


@pytest.mark.parametrize("core", BUILT_CORES)
def test_a_done_on_every_env_makes_the_incoming_state_irrelevant(core):
    model = build_core(core)
    B = 3
    inputs = fwp_inputs(1, B, done_at=tuple((0, b) for b in range(B)), seed=5)
    with torch.no_grad():
        from_zero, state_zero = model(inputs, model.initial_state(B))
        from_junk, state_junk = model(inputs, (torch.randn(1, B, FWP_KW["fwp_dim"] ** 2),))
    torch.testing.assert_close(from_zero["baseline"], from_junk["baseline"])
    torch.testing.assert_close(state_zero[0], state_junk[0])


@pytest.mark.parametrize("core", BUILT_CORES)
def test_state_stays_flat_and_batched_at_dim_1_through_a_forward(core):
    model = build_core(core)
    T, B = 4, 5
    with torch.no_grad():
        _, state = model(fwp_inputs(T, B, seed=6), model.initial_state(B))
    assert isinstance(state, tuple) and len(state) == 1
    assert state[0].shape == (1, B, FWP_KW["fwp_dim"] ** 2)


@pytest.mark.parametrize("core", BUILT_CORES)
def test_parameter_count(core, capsys):
    """Printed for the record; the assertion is only that the core adds some."""
    with_core = build_core(core)
    torch.manual_seed(0)
    ff = ImpalaNet(**FWP_SPEC)
    n_core = sum(p.numel() for p in with_core.parameters())
    n_ff = sum(p.numel() for p in ff.parameters())
    with capsys.disabled():
        print(f"\n  {core}: {n_core} params ({n_core - n_ff} in the core)")
    assert n_core > n_ff


def test_default_fwp_state_fits_the_memory_budget():
    """fwp_dim=128 is chosen so 64 envs' state stays near 4 MB in fp32."""
    model = ImpalaNet(**FWP_SPEC, core="deltanet")
    (state,) = model.initial_state(64)
    assert state.shape == (1, 64, 128 * 128)
    assert state.element_size() * state.numel() / 1e6 == pytest.approx(4.19, abs=0.01)


# ------------------------------------------------- CompFWP and its ablations


def test_compfwp_indep_indep_delta_is_deltanet_numerically():
    """The ablation the review's table hangs on: no competition, plain delta."""
    torch.manual_seed(0)
    delta = DeltaNetCore(32, fwp_dim=16, n_heads=4)
    comp = CompFWPCore(32, fwp_dim=16, n_heads=4, read="indep", error="indep")
    missing, unexpected = comp.load_state_dict(delta.state_dict(), strict=False)
    assert not missing and not unexpected

    x = torch.randn(5, 3, 32)
    notdone = torch.ones(5, 3)
    with torch.no_grad():
        a, sa = delta(x, notdone, delta.initial_state(3))
        b, sb = comp(x, notdone, comp.initial_state(3))
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    torch.testing.assert_close(sa[0], sb[0], rtol=0, atol=0)


def test_the_joint_variant_actually_differs():
    """Otherwise the equivalence above would be testing nothing."""
    torch.manual_seed(0)
    delta = DeltaNetCore(32, fwp_dim=16, n_heads=4)
    comp = CompFWPCore(32, fwp_dim=16, n_heads=4)
    comp.load_state_dict(delta.state_dict(), strict=False)
    x = torch.randn(5, 3, 32)
    notdone = torch.ones(5, 3)
    with torch.no_grad():
        a, _ = delta(x, notdone, delta.initial_state(3))
        b, _ = comp(x, notdone, comp.initial_state(3))
    assert not torch.allclose(a, b, atol=1e-5)


def test_compfwp_ablation_flags_reach_the_core_through_impalanet():
    model = ImpalaNet(
        **FWP_SPEC, core="compfwp", **FWP_KW,
        fwp_read="indep", fwp_error="indep", fwp_write="additive",
    )
    assert model.core.variant == "indep/indep/additive"
    assert model.core.set_block is None and model.core.W_p is None


def test_compfwp_rejects_an_impossible_flag_combination():
    with pytest.raises(ValueError, match="no joint read"):
        CompFWPCore(32, read="indep", error="joint")
    with pytest.raises(ValueError, match="write must be"):
        CompFWPCore(32, write="nope")


def test_set_block_is_permutation_equivariant():
    """No positional information: the rows must be interchangeable."""
    torch.manual_seed(1)
    from playtrain_trainers.impala.fwp import SetBlock

    block = SetBlock(16, n_heads=4)
    rows = torch.randn(3, 6, 16)
    perm = torch.randperm(6)
    with torch.no_grad():
        a = block(rows)[:, perm]
        b = block(rows[:, perm])
    torch.testing.assert_close(a, b, atol=1e-5, rtol=0)


def test_the_write_key_is_one_of_the_competing_rows():
    """The joint prediction has to be taken at the key, not pooled from the rest."""
    torch.manual_seed(0)
    comp = CompFWPCore(32, fwp_dim=16, n_heads=4)
    state = torch.randn(2, 16, 16)
    q = torch.randn(2, 4, 16)
    k = torch.nn.functional.normalize(torch.randn(2, 16), dim=-1)
    with torch.no_grad():
        rows = comp._compete(state, q, k)
    assert rows.shape == (2, 5, 16)  # M rows plus the write key
    with torch.no_grad():
        assert comp.read(state, q, k).shape == (2, 4, 16)


def test_additive_write_ignores_the_error_term():
    torch.manual_seed(0)
    comp = CompFWPCore(32, fwp_dim=16, n_heads=4, write="additive")
    state = torch.randn(2, 16, 16)
    k = torch.nn.functional.normalize(torch.randn(2, 16), dim=-1)
    v = torch.randn(2, 16)
    beta = torch.rand(2, 1)
    q = torch.randn(2, 4, 16)
    with torch.no_grad():
        got = comp.write(state, k, v, beta, q)
    want = state + beta.unsqueeze(-1) * v.unsqueeze(-1) * k.unsqueeze(-2)
    torch.testing.assert_close(got, want)


# ------------------------------------------- long-horizon state boundedness
#
# The smoke at 50M died here: trainer episodes run to max_decisions=5000 and
# the state only resets on done, so a write rule that is not contractive along
# the write key overflows long before the episode ends. Stage 0's 40-step
# episodes cannot surface it, which is exactly why these live at this level.

LONG_HORIZON = 2000


def run_state_norm(core, steps: int = LONG_HORIZON, B: int = 2, seed: int = 0) -> float:
    torch.manual_seed(seed)
    core.eval()
    state = core.initial_state(B)[0].reshape(B, core.fwp_dim, core.fwp_dim)
    with torch.no_grad():
        for _ in range(steps):
            _, state = core.step(torch.randn(B, 64), state)
    assert torch.isfinite(state).all(), "state went non-finite"
    return state.norm().item()


@pytest.mark.parametrize("core_cls", [DeltaNetCore, CompFWPCore])
def test_state_stays_bounded_over_a_trainer_length_episode(core_cls):
    torch.manual_seed(0)
    norm = run_state_norm(core_cls(64, fwp_dim=16, n_heads=4))
    assert norm < 1e3, f"{core_cls.kind} state reached {norm:.4g} after {LONG_HORIZON} steps"


@pytest.mark.parametrize("gain", [0.5, 1.0, 3.0])
def test_compfwp_stays_bounded_once_w_p_is_trained_away_from_zero(gain):
    """Boundedness must be structural, not an artefact of the zero init."""
    torch.manual_seed(0)
    core = CompFWPCore(64, fwp_dim=16, n_heads=4)
    torch.nn.init.orthogonal_(core.W_p.weight, gain=gain)
    norm = run_state_norm(core)
    assert norm < 1e3, f"W_p gain {gain} diverged to {norm:.4g}"


def test_the_correction_form_is_what_keeps_it_bounded():
    """The regression this guards: replacing the prediction instead of
    correcting it puts W_p @ S k inside the error, and that compounds."""

    class ReplacingCompFWP(CompFWPCore):
        def write(self, state, k, v, beta, q):
            pred = self.W_p(self._compete(state, q, k)[:, -1])
            return state + beta.unsqueeze(-1) * (v - pred).unsqueeze(-1) * k.unsqueeze(-2)

    torch.manual_seed(0)
    bad = ReplacingCompFWP(64, fwp_dim=16, n_heads=4)
    torch.nn.init.orthogonal_(bad.W_p.weight, gain=1.0)
    torch.manual_seed(0)
    good = CompFWPCore(64, fwp_dim=16, n_heads=4)
    torch.nn.init.orthogonal_(good.W_p.weight, gain=1.0)

    with torch.no_grad():
        bad_state = bad.initial_state(2)[0].reshape(2, 16, 16)
        good_state = good.initial_state(2)[0].reshape(2, 16, 16)
        torch.manual_seed(1)
        xs = [torch.randn(2, 64) for _ in range(LONG_HORIZON)]
        for x in xs:
            _, bad_state = bad.step(x, bad_state)
            if not torch.isfinite(bad_state).all():
                break
        for x in xs:
            _, good_state = good.step(x, good_state)

    bad_norm = bad_state.norm().item()
    assert not (bad_norm < 1e3), "the replacing form should blow up; the guard is stale"
    assert good_state.norm().item() < 1e3


def test_zero_initialised_w_p_makes_the_write_path_exactly_the_delta_rule():
    """At init only the read differs, so the contraction is exact to start."""
    torch.manual_seed(0)
    delta = DeltaNetCore(64, fwp_dim=16, n_heads=4)
    torch.manual_seed(0)
    comp = CompFWPCore(64, fwp_dim=16, n_heads=4)
    assert not comp.W_p.weight.any()

    state = torch.randn(3, 16, 16)
    k = torch.nn.functional.normalize(torch.randn(3, 16), dim=-1)
    v = torch.randn(3, 16)
    beta = torch.rand(3, 1)
    q = torch.randn(3, 4, 16)
    with torch.no_grad():
        torch.testing.assert_close(
            comp.write(state, k, v, beta, q), delta.write(state, k, v, beta, q)
        )


def test_w_p_still_receives_gradient_despite_the_zero_init():
    torch.manual_seed(0)
    core = CompFWPCore(64, fwp_dim=16, n_heads=4)
    out, _ = core(torch.randn(4, 2, 64), torch.ones(4, 2), core.initial_state(2))
    out.sum().backward()
    assert core.W_p.weight.grad is not None
    assert core.W_p.weight.grad.abs().sum() > 0


# ------------------------------------------------- state-norm instrumentation
#
# The trace the collapse probe reads. Measured in the learner from the state
# entering the unroll, so the compiled forward is untouched, and kept as
# detached tensors so no per-step host sync is added.

from playtrain_trainers.impala.fwp import matrix_state_norms  # noqa: E402


def test_matrix_state_norms_are_per_env_frobenius_norms():
    state = torch.zeros(1, 3, 16)
    state[0, 0] = 3.0 / 4.0  # 16 entries of 0.75 -> norm 3
    state[0, 1] = 1.0 / 4.0  # -> norm 1
    mean, mx = matrix_state_norms((state,))
    assert mx.item() == pytest.approx(3.0)
    assert mean.item() == pytest.approx((3.0 + 1.0 + 0.0) / 3)
    assert not mean.requires_grad and not mx.requires_grad


def test_matrix_state_norms_is_none_without_state():
    assert matrix_state_norms(()) is None


def test_state_norm_reaches_learn_stats_for_fwp_cores_only():
    """Absent for lstm and ff: there is no matrix state to measure."""
    from playtrain_trainers.impala.learn import learn

    keys = ("fwp_state_norm_mean", "fwp_state_norm_max")
    for core, expected in (("compfwp", True), ("deltanet", True), ("lstm", False), ("ff", False)):
        torch.manual_seed(0)
        model = ImpalaNet(**FWP_SPEC, core=core, **FWP_KW)
        T, B = 3, 2
        batch = fwp_inputs(T + 1, B, seed=1)
        batch["episode_return"] = torch.zeros(T + 1, B)
        batch["policy_logits"] = torch.zeros(T + 1, B, FWP_SPEC["num_actions"])
        batch["action"] = torch.zeros(T + 1, B, dtype=torch.int64)
        batch["baseline"] = torch.zeros(T + 1, B)
        stats = learn(
            actor_model=None,
            learner_model=model,
            batch=batch,
            initial_agent_state=model.initial_state(B),
            optimizer=torch.optim.SGD(model.parameters(), lr=1e-4),
            scheduler=None,
            discounting=0.99,
            baseline_cost=0.5,
            entropy_cost=0.01,
            grad_norm_clipping=40.0,
        )
        present = all(k in stats for k in keys)
        assert present is expected, f"{core}: state-norm keys present={present}"
        if present:
            assert stats["fwp_state_norm_mean"].item() == 0.0  # fresh state


# ------------------------------------------------------------- state decay
#
# P.4's intervention: bound ||S|| directly, since P.3 showed episode length
# does not. Default 0.0, so a run that does not ask for it is unchanged.


@pytest.mark.parametrize("core_cls", [DeltaNetCore, CompFWPCore])
def test_decay_zero_is_bit_identical_to_no_decay(core_cls):
    """The default must not perturb anything that already ran."""
    torch.manual_seed(0)
    plain = core_cls(64, fwp_dim=16, n_heads=4)
    torch.manual_seed(0)
    explicit = core_cls(64, fwp_dim=16, n_heads=4, decay=0.0)
    explicit.load_state_dict(plain.state_dict())

    x = torch.randn(6, 3, 64)
    notdone = torch.ones(6, 3)
    with torch.no_grad():
        a, sa = plain(x, notdone, plain.initial_state(3))
        b, sb = explicit(x, notdone, explicit.initial_state(3))
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    torch.testing.assert_close(sa[0], sb[0], rtol=0, atol=0)


@pytest.mark.parametrize("core_cls", [DeltaNetCore, CompFWPCore])
def test_decay_lowers_the_settled_state_norm(core_cls):
    """More forgetting must mean a smaller state, monotonically."""
    norms = {}
    for decay in (0.0, 1e-3, 1e-2):
        torch.manual_seed(0)
        core = core_cls(64, fwp_dim=16, n_heads=4, decay=decay)
        norms[decay] = run_state_norm(core, steps=LONG_HORIZON)
    assert norms[1e-2] < norms[1e-3] < norms[0.0], norms


@pytest.mark.parametrize("core_cls", [DeltaNetCore, CompFWPCore])
def test_decay_keeps_the_state_bounded_even_with_a_trained_w_p(core_cls):
    torch.manual_seed(0)
    core = core_cls(64, fwp_dim=16, n_heads=4, decay=1e-2)
    if isinstance(core, CompFWPCore):
        torch.nn.init.orthogonal_(core.W_p.weight, gain=3.0)
    assert run_state_norm(core, steps=LONG_HORIZON) < 1e3


def test_decay_reaches_the_core_through_impalanet_and_defaults_off():
    assert ImpalaNet(**FWP_SPEC, core="compfwp", **FWP_KW).core.decay == 0.0
    model = ImpalaNet(**FWP_SPEC, core="deltanet", **FWP_KW, fwp_decay=5e-3)
    assert model.core.decay == pytest.approx(5e-3)


@pytest.mark.parametrize("bad", [-0.1, 1.0, 1.5])
def test_decay_outside_zero_to_one_is_rejected(bad):
    with pytest.raises(ValueError, match="decay must be"):
        DeltaNetCore(64, fwp_dim=16, n_heads=4, decay=bad)


# ------------------------------------------------------- Q.2 diagnostics
#
# Grad norm before clipping and the scale of the core's output, for every
# core including the LSTM — the LSTM is the stable reference these are meant
# to be compared against, so leaving it out would defeat the purpose.


@pytest.mark.parametrize("core", ["lstm", "deltanet", "compfwp", "ff"])
def test_grad_norm_and_core_scale_reach_learn_stats_for_every_core(core):
    from playtrain_trainers.impala.learn import learn

    torch.manual_seed(0)
    model = ImpalaNet(**FWP_SPEC, core=core, **FWP_KW)
    T, B = 3, 2
    batch = fwp_inputs(T + 1, B, seed=1)
    batch["episode_return"] = torch.zeros(T + 1, B)
    batch["policy_logits"] = torch.zeros(T + 1, B, FWP_SPEC["num_actions"])
    batch["action"] = torch.zeros(T + 1, B, dtype=torch.int64)
    batch["baseline"] = torch.zeros(T + 1, B)
    stats = learn(
        actor_model=None, learner_model=model, batch=batch,
        initial_agent_state=model.initial_state(B),
        optimizer=torch.optim.SGD(model.parameters(), lr=1e-4),
        scheduler=None, discounting=0.99, baseline_cost=0.5,
        entropy_cost=0.01, grad_norm_clipping=40.0,
    )
    for key in ("grad_norm", "core_out_scale"):
        assert key in stats, f"{core}: {key} missing"
        assert torch.isfinite(stats[key]).all()
    assert stats["grad_norm"].item() > 0
    assert stats["core_out_scale"].item() > 0


def test_core_out_scale_is_not_in_the_state_dict():
    """Non-persistent: weight sync, checkpoints and the golden must not see it."""
    model = ImpalaNet(**FWP_SPEC, core="compfwp", **FWP_KW)
    assert "core_out_scale" not in model.state_dict()
    assert hasattr(model, "core_out_scale")


def test_core_out_scale_tracks_the_actual_core_output():
    torch.manual_seed(0)
    model = ImpalaNet(**FWP_SPEC, core="deltanet", **FWP_KW)
    model.eval()
    batch = fwp_inputs(4, 2, seed=3)
    with torch.no_grad():
        model(batch, model.initial_state(2))
    recorded = model.core_out_scale.item()
    assert recorded > 0 and np.isfinite(recorded)

    # a second, different batch must move it
    with torch.no_grad():
        model(fwp_inputs(4, 2, seed=9), model.initial_state(2))
    assert model.core_out_scale.item() != recorded


# ----------------------------------------------- R.1b per-group attribution


def test_grad_group_assigns_every_parameter_of_every_core():
    """No parameter may fall into 'other', or the attribution has a hole."""
    from playtrain_trainers.impala.learn import GRAD_GROUPS, grad_group

    for core in ("lstm", "deltanet", "compfwp", "ff"):
        model = ImpalaNet(**FWP_SPEC, core=core, **FWP_KW)
        for name, _ in model.named_parameters():
            g = grad_group(name)
            assert g in GRAD_GROUPS, f"{core}: {name} -> {g}"


def test_grad_group_norms_partition_the_total():
    """The per-group norms must reconstruct the total, or they mislead."""
    from playtrain_trainers.impala.learn import grad_group_norms

    torch.manual_seed(0)
    model = ImpalaNet(**FWP_SPEC, core="compfwp", **FWP_KW)
    out, _ = model(fwp_inputs(4, 2, seed=1), model.initial_state(2))
    out["baseline"].pow(2).mean().backward()

    groups = grad_group_norms(model)
    assert groups, "no groups recorded"
    combined = sum(v.item() ** 2 for v in groups.values()) ** 0.5
    total = sum(p.grad.pow(2).sum().item() for p in model.parameters()
                if p.grad is not None) ** 0.5
    assert combined == pytest.approx(total, rel=1e-5)


def test_grad_group_logging_is_off_by_default():
    from playtrain_trainers.impala.learn import learn

    torch.manual_seed(0)
    model = ImpalaNet(**FWP_SPEC, core="compfwp", **FWP_KW)
    T, B = 3, 2
    batch = fwp_inputs(T + 1, B, seed=1)
    batch["episode_return"] = torch.zeros(T + 1, B)
    batch["policy_logits"] = torch.zeros(T + 1, B, FWP_SPEC["num_actions"])
    batch["action"] = torch.zeros(T + 1, B, dtype=torch.int64)
    batch["baseline"] = torch.zeros(T + 1, B)
    common = dict(
        actor_model=None, learner_model=model, batch=batch,
        initial_agent_state=model.initial_state(B),
        optimizer=torch.optim.SGD(model.parameters(), lr=0.0),
        scheduler=None, discounting=0.99, baseline_cost=0.5,
        entropy_cost=0.01, grad_norm_clipping=40.0,
    )
    assert not [k for k in learn(**common) if k.startswith("gradgrp/")]
    on = [k for k in learn(**common, log_grad_groups=True) if k.startswith("gradgrp/")]
    assert len(on) >= 5, on


# --------------------------------------------------- S.1 V-trace diagnostics


def _learn_batch(model, T=3, B=2):
    batch = fwp_inputs(T + 1, B, seed=1)
    batch["episode_return"] = torch.zeros(T + 1, B)
    batch["policy_logits"] = torch.randn(T + 1, B, FWP_SPEC["num_actions"])
    batch["action"] = torch.zeros(T + 1, B, dtype=torch.int64)
    batch["baseline"] = torch.zeros(T + 1, B)
    return dict(
        actor_model=None, learner_model=model, batch=batch,
        initial_agent_state=model.initial_state(B),
        optimizer=torch.optim.SGD(model.parameters(), lr=0.0),
        scheduler=None, discounting=0.99, baseline_cost=0.5,
        entropy_cost=0.01, grad_norm_clipping=40.0,
    )


@pytest.mark.parametrize("core", ["lstm", "compfwp"])
def test_vtrace_stats_present_only_when_asked_and_finite(core):
    from playtrain_trainers.impala.learn import learn

    torch.manual_seed(0)
    model = ImpalaNet(**FWP_SPEC, core=core, **FWP_KW)
    off = learn(**_learn_batch(model))
    assert not [k for k in off if k.startswith("vtrace/")]
    on = learn(**_learn_batch(model), log_vtrace=True)
    keys = {k for k in on if k.startswith("vtrace/")}
    assert keys == {"vtrace/adv_abs_mean", "vtrace/adv_std", "vtrace/adv_abs_max",
                    "vtrace/log_rho_abs_mean", "vtrace/rho_clip_frac", "vtrace/td_abs_mean"}
    for k in keys:
        assert torch.isfinite(on[k]).all(), k
        assert not on[k].requires_grad, k
    assert 0.0 <= on["vtrace/rho_clip_frac"].item() <= 1.0


# ------------------------------------------------ S.2 unthrottling the block

CANDIDATE = dict(fwp_w_o_gain=1.0, fwp_read_norm=True, fwp_w_p_init=0.02)


@pytest.mark.parametrize("core", ["deltanet", "compfwp"])
def test_track_b_defaults_are_bit_identical_to_before(core):
    """Flags off must reproduce every run to date exactly."""
    torch.manual_seed(0)
    a = ImpalaNet(**FWP_SPEC, core=core, **FWP_KW)
    torch.manual_seed(0)
    b = ImpalaNet(**FWP_SPEC, core=core, **FWP_KW,
                  fwp_w_o_gain=0.1, fwp_read_norm=False, fwp_w_p_init=0.0)
    assert set(a.state_dict()) == set(b.state_dict())
    for k in a.state_dict():
        torch.testing.assert_close(a.state_dict()[k], b.state_dict()[k], rtol=0, atol=0)
    inputs = fwp_inputs(4, 2, seed=1)
    a.eval(); b.eval()
    with torch.no_grad():
        oa, sa = a(inputs, a.initial_state(2))
        ob, sb = b(inputs, b.initial_state(2))
    torch.testing.assert_close(oa["baseline"], ob["baseline"], rtol=0, atol=0)
    torch.testing.assert_close(sa[0], sb[0], rtol=0, atol=0)


def test_deltanet_equivalence_survives_with_flags_off():
    torch.manual_seed(0)
    delta = DeltaNetCore(32, fwp_dim=16, n_heads=4)
    comp = CompFWPCore(32, fwp_dim=16, n_heads=4, read="indep", error="indep")
    missing, unexpected = comp.load_state_dict(delta.state_dict(), strict=False)
    assert not missing and not unexpected
    x = torch.randn(5, 3, 32); nd = torch.ones(5, 3)
    with torch.no_grad():
        a, sa = delta(x, nd, delta.initial_state(3))
        b, sb = comp(x, nd, comp.initial_state(3))
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    torch.testing.assert_close(sa[0], sb[0], rtol=0, atol=0)


def test_read_norm_leaves_the_write_path_untouched():
    """The contraction argument rests on the write; read_norm must not touch it."""
    torch.manual_seed(0)
    plain = CompFWPCore(32, fwp_dim=16, n_heads=4)
    torch.manual_seed(0)
    normed = CompFWPCore(32, fwp_dim=16, n_heads=4, read_norm=True)
    # read_norm adds LayerNorm params; everything else identical
    sd = {k: v for k, v in normed.state_dict().items() if not k.startswith("read_norm")}
    assert set(sd) == set(plain.state_dict())
    x = torch.randn(6, 3, 32); nd = torch.ones(6, 3)
    with torch.no_grad():
        _, s_plain = plain(x, nd, plain.initial_state(3))
        _, s_norm = normed(x, nd, normed.initial_state(3))
    torch.testing.assert_close(s_plain[0], s_norm[0], rtol=0, atol=0)


def test_candidate_state_stays_bounded():
    torch.manual_seed(0)
    core = CompFWPCore(64, fwp_dim=16, n_heads=4, w_o_gain=1.0, read_norm=True, w_p_init=0.02)
    assert run_state_norm(core, steps=LONG_HORIZON) < 1e3


def test_candidate_flags_reach_the_core():
    m = ImpalaNet(**FWP_SPEC, core="compfwp", **FWP_KW, **CANDIDATE)
    assert m.core.read_norm is not None
    assert m.core.W_p.weight.abs().sum() > 0
    assert m.core.W_o.weight.norm() > ImpalaNet(**FWP_SPEC, core="compfwp", **FWP_KW).core.W_o.weight.norm() * 5


# --------------------------------------------- T.1 the reference core port
#
# Irie et al.'s RL DeltaNet (IDSIA/recurrent-fwp), ported as deltanet_ref.
# features_dim 32 in FWP_SPEC, so heads*dim_head must be 32.

from playtrain_trainers.impala.fwp import RefDeltaNetCore, elu_p1_sum_norm  # noqa: E402

REF_KW = {"fwp_ref_heads": 4, "fwp_ref_dim_head": 8}


def build_ref() -> ImpalaNet:
    torch.manual_seed(0)
    m = ImpalaNet(**FWP_SPEC, core="deltanet_ref", **REF_KW)
    m.eval()
    return m


def test_ref_feature_map_is_nonnegative_and_sums_to_one():
    """The property the whole comparison rests on: W q is a convex combination."""
    x = torch.randn(5, 4, 8) * 3
    y = elu_p1_sum_norm(x)
    assert (y >= 0).all()
    torch.testing.assert_close(y.sum(-1), torch.ones(5, 4), atol=1e-4, rtol=0)
    # and it really is applied to both q and k inside the core
    core = RefDeltaNetCore(32, n_heads=4, dim_head=8)
    q, k, v, beta = core.project(torch.randn(3, 32))
    for t in (q, k):
        assert (t >= 0).all()
        torch.testing.assert_close(t.sum(-1), torch.ones(3, 4), atol=1e-4, rtol=0)
    assert ((beta > 0) & (beta < 1)).all() and beta.shape == (3, 4, 1)


def test_ref_rejects_mismatched_head_geometry():
    with pytest.raises(ValueError, match="n_heads\\*dim_head"):
        RefDeltaNetCore(32, n_heads=4, dim_head=9)


def test_ref_initial_state_is_a_one_tuple_batch_at_dim_1():
    m = build_ref()
    state = m.initial_state(6)
    assert isinstance(state, tuple) and len(state) == 1
    assert state[0].shape == (1, 6, 4 * 8 * 8) and not state[0].any()


def test_ref_t_step_equals_single_steps_with_boundaries():
    m = build_ref()
    T, B = 6, 3
    inputs = fwp_inputs(T, B, done_at=((0, 1), (2, 0), (4, 2)), seed=2)
    with torch.no_grad():
        batched, bstate = m(inputs, m.initial_state(B))
        state = m.initial_state(B); steps = []
        for t in range(T):
            out, state = m(slice_step(inputs, t), state); steps.append(out)
    for key in ("policy_logits", "baseline"):
        torch.testing.assert_close(batched[key], torch.cat([s[key] for s in steps]), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(bstate[0], state[0], atol=1e-5, rtol=1e-5)


def test_ref_carry_across_unrolls_equals_one_long_unroll():
    m = build_ref()
    T, B = 8, 3
    inputs = fwp_inputs(T, B, done_at=((3, 1),), seed=3)
    window = lambda lo, hi: {k: v[lo:hi] for k, v in inputs.items()}  # noqa: E731
    with torch.no_grad():
        long_out, long_state = m(inputs, m.initial_state(B))
        first, mid = m(window(0, 5), m.initial_state(B))
        second, split_state = m(window(5, 8), mid)
    torch.testing.assert_close(long_out["baseline"], torch.cat([first["baseline"], second["baseline"]]), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(long_state[0], split_state[0], atol=1e-5, rtol=1e-5)


def test_ref_state_is_zeroed_entering_a_done_step():
    m = build_ref()
    B = 2
    inputs = fwp_inputs(1, B, done_at=((0, 0),), seed=4)
    dirty = (torch.randn(1, B, 4 * 8 * 8),)
    clean = (dirty[0].clone(),); clean[0][:, 0] = 0.0
    with torch.no_grad():
        _, after_dirty = m(inputs, dirty)
        _, after_clean = m(inputs, clean)
    torch.testing.assert_close(after_dirty[0][:, 0], after_clean[0][:, 0])


def test_ref_read_is_after_write_and_bounded_by_written_values():
    """With sum-normalised q the read is a convex combination of written rows,
    so a single write of v followed by a read must return something inside
    v's range — the bound that the unnormalised bilinear read does not have."""
    core = RefDeltaNetCore(32, n_heads=4, dim_head=8).eval()
    W = torch.zeros(1, 4, 8, 8)
    x = torch.randn(1, 32)
    with torch.no_grad():
        q, k, v, beta = core.project(x)
        _, W = core.step(x, W)
        out = torch.einsum("bhij,bhj->bhi", W, q)
    # after one write from zero, W q = (beta*v) * (k . q); |k.q| <= 1 since both on the simplex
    assert (out.abs() <= (beta * v).abs() + 1e-6).all()


def test_ref_state_stays_bounded_over_a_trainer_length_episode():
    torch.manual_seed(0)
    core = RefDeltaNetCore(64, n_heads=4, dim_head=16).eval()
    W = core.initial_state(2)[0].reshape(2, 4, 16, 16)
    with torch.no_grad():
        for _ in range(LONG_HORIZON):
            _, W = core.step(torch.randn(2, 64), W)
    assert torch.isfinite(W).all() and W.norm().item() < 1e3


def test_ref_parameter_count(capsys):
    m = build_ref()
    n = sum(p.numel() for p in m.parameters())
    n_core = sum(p.numel() for p in m.core.parameters())
    with capsys.disabled():
        print(f"\n  deltanet_ref: {n} params ({n_core} in the core)")
    assert n_core > 0
