"""Freeze the LSTM core's forward output, before any `core` refactor touches it.

Run this on the pinned pre-change revision. `tests/test_fwp_core.py` replays the
saved cases and asserts bit-identical results, which is what makes "the LSTM
path stays byte-identical" a checkable claim rather than an intention.

    uv run python tools/make_lstm_golden.py

The file carries the model's state_dict, so the comparison does not depend on
initialisation RNG staying put across torch versions. Cases cover both forward
paths: T=1 (the actor's single masked cell step) and T>1 (the segmented unroll),
each with and without episode boundaries, plus a carried non-zero state.
"""

from __future__ import annotations

from pathlib import Path

import torch

from playtrain_trainers.impala.net import ImpalaNet

OUT = Path(__file__).resolve().parents[1] / "tests" / "data" / "lstm_golden.pt"
SPEC = {
    "observation_shape": (3, 64, 64),
    "num_actions": 7,
    "features_dim": 256,
    "net": "impala",
}
INIT_SEED = 1234


def make_inputs(T: int, B: int, done_at: list[tuple[int, int]], gen) -> dict:
    shape = (T, B, *SPEC["observation_shape"])
    done = torch.zeros(T, B, dtype=torch.bool)
    for t, b in done_at:
        done[t, b] = True
    return {
        "frame": torch.randint(0, 256, shape, dtype=torch.uint8, generator=gen),
        "reward": torch.randn(T, B, generator=gen),
        "done": done,
        "last_action": torch.randint(
            0, SPEC["num_actions"], (T, B), dtype=torch.int64, generator=gen
        ),
    }


def main() -> Path:
    torch.manual_seed(INIT_SEED)
    model = ImpalaNet(**SPEC, use_lstm=True)
    model.eval()  # argmax action: sampling would make the golden unreproducible

    gen = torch.Generator().manual_seed(99)
    cases = {}

    # T=1, fresh state, no boundary — the actor's ordinary step.
    cases["t1_clean"] = (make_inputs(1, 4, [], gen), model.initial_state(4))
    # T=1 with a carried non-zero state and one env resetting.
    carried = tuple(torch.randn(1, 4, SPEC["features_dim"], generator=gen) for _ in range(2))
    cases["t1_done_carried"] = (make_inputs(1, 4, [(0, 2)], gen), carried)
    # T>1 with no boundary: one segment, still masked entering the unroll.
    cases["t5_clean"] = (make_inputs(5, 3, [], gen), model.initial_state(3))
    # T>1 with boundaries at two different timesteps: three segments.
    cases["t5_segmented"] = (make_inputs(5, 3, [(1, 0), (3, 2)], gen), model.initial_state(3))
    # T>1, boundary on the first row (the buffer's carry frame) and a carried state.
    carried3 = tuple(torch.randn(1, 3, SPEC["features_dim"], generator=gen) for _ in range(2))
    cases["t5_done_row0"] = (make_inputs(5, 3, [(0, 1), (2, 0)], gen), carried3)

    golden = {}
    with torch.no_grad():
        for name, (inputs, state) in cases.items():
            out, new_state = model(inputs, state)
            golden[name] = {
                "inputs": inputs,
                "core_state_in": tuple(state),
                "outputs": out,
                "core_state_out": tuple(new_state),
            }

    payload = {
        "spec": SPEC,
        "init_seed": INIT_SEED,
        "torch_version": torch.__version__,
        "state_dict": model.state_dict(),
        "initial_state_shapes": [tuple(s.shape) for s in model.initial_state(5)],
        "cases": golden,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, OUT)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.2f} MB), cases: {sorted(golden)}")
    return OUT


if __name__ == "__main__":
    main()
