"""The `core` option, and the promise that adding it left the LSTM untouched.

The LSTM regression replays `tests/data/lstm_golden.pt`, frozen from the
pre-change revision by `tools/make_lstm_golden.py`, and demands bit-identical
output. Any drift in the LSTM path is a bug, not a tolerance to widen.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

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
def test_fast_weight_cores_are_accepted_but_not_yet_built(core):
    """They validate as config now; the modules land in the next items."""
    assert core in CORES
    assert resolve_core(core, False) == core
    with pytest.raises(NotImplementedError, match=core):
        ImpalaNet((3, 32, 32), 5, features_dim=32, core=core)
