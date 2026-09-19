"""The `core` option, and the promise that adding it left the LSTM untouched.

The LSTM regression replays `tests/data/lstm_golden.pt`, frozen from the
pre-change revision by `tools/make_lstm_golden.py`, and demands bit-identical
output. Any drift in the LSTM path is a bug, not a tolerance to widen.
"""

from __future__ import annotations

from pathlib import Path

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
