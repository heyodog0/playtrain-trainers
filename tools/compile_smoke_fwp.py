"""Check the fast-weight cores survive torch.compile without recompiling per step.

The learner compiles in place with ``Module.compile(mode=...)`` and then runs
the same T x B unroll every step, so what matters is not whether the first call
is slow — it always is — but whether dynamo settles: no new frames, no new
graph breaks, and a flat per-iteration time after the first.

    uv run python tools/compile_smoke_fwp.py --cores deltanet,compfwp -T 100 -B 8

Exits non-zero if any core keeps recompiling.
"""

from __future__ import annotations

import argparse
import json
import time

import torch
import torch._dynamo

from playtrain_trainers.impala.net import ImpalaNet

SPEC = {"observation_shape": (3, 64, 64), "num_actions": 7, "features_dim": 256}


def make_inputs(T: int, B: int, seed: int = 0) -> dict:
    gen = torch.Generator().manual_seed(seed)
    done = torch.zeros(T, B, dtype=torch.bool)
    done[T // 3, 0] = True  # one episode boundary, so the reset path is exercised
    done[2 * T // 3, B - 1] = True
    return {
        "frame": torch.randint(
            0, 256, (T, B, *SPEC["observation_shape"]), dtype=torch.uint8, generator=gen
        ),
        "reward": torch.randn(T, B, generator=gen),
        "done": done,
        "last_action": torch.zeros(T, B, dtype=torch.int64),
    }


def counter_snapshot() -> dict:
    c = torch._dynamo.utils.counters
    return {
        "frames_ok": int(c["frames"].get("ok", 0)),
        "frames_total": int(c["frames"].get("total", 0)),
        "graph_breaks": int(sum(c["graph_break"].values())),
        "unique_graph_breaks": len(c["graph_break"]),
    }


def smoke(core: str, T: int, B: int, iters: int, mode: str, fwp_dim: int, flags: dict | None = None) -> dict:
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()

    torch.manual_seed(0)
    model = ImpalaNet(**SPEC, core=core, **({} if core == "deltanet_ref" else {"fwp_dim": fwp_dim, **(flags or {})}))
    model.train()
    model.compile(mode=mode)

    inputs = make_inputs(T, B)
    state = model.initial_state(B)

    times, snaps = [], []
    for _ in range(iters):
        started = time.time()
        out, _ = model(inputs, state)
        out["baseline"].sum().backward()
        model.zero_grad(set_to_none=True)
        times.append(time.time() - started)
        snaps.append(counter_snapshot())

    after_warmup = snaps[-1]["frames_ok"] - snaps[0]["frames_ok"]
    breaks_after_warmup = snaps[-1]["graph_breaks"] - snaps[0]["graph_breaks"]
    steady = times[1:]
    result = {
        "core": core,
        "compile_mode": mode,
        "T": T,
        "B": B,
        "first_iter_s": round(times[0], 2),
        "steady_iter_s": [round(t, 3) for t in steady],
        "speedup_after_warmup": round(times[0] / max(min(steady), 1e-9), 1),
        "graph_breaks_total": snaps[-1]["graph_breaks"],
        "unique_graph_break_sites": snaps[-1]["unique_graph_breaks"],
        "frames_compiled": snaps[-1]["frames_ok"],
        "new_frames_after_first_iter": after_warmup,
        "new_graph_breaks_after_first_iter": breaks_after_warmup,
        "break_reasons": dict(torch._dynamo.utils.counters["graph_break"]),
    }
    # The acceptance condition: dynamo has settled. Nothing new gets compiled
    # after the first iteration, so there is no recompile-per-step storm.
    result["settled"] = after_warmup == 0 and breaks_after_warmup == 0
    return result


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cores", default="deltanet,compfwp")
    p.add_argument("-T", type=int, default=100)
    p.add_argument("-B", type=int, default=8)
    p.add_argument("--iters", type=int, default=3)
    p.add_argument("--mode", default="default")
    p.add_argument("--fwp-dim", type=int, default=128)
    p.add_argument("--flags", default="", help="comma-separated ImpalaNet kwargs, e.g. fwp_feature_map=elu_sumnorm,fwp_multihead=1")
    a = p.parse_args(argv)

    flags = {}
    for kv in filter(None, a.flags.split(",")):
        key, val = kv.split("=", 1)
        flags[key] = (val.lower() in ("1", "true")) if key in ("fwp_multihead", "fwp_gate", "fwp_out_norm", "fwp_out_gate") else float(val) if key in ("fwp_key_scale", "fwp_beta_max", "fwp_decay", "fwp_w_p_init", "fwp_w_o_gain") else val

    results = []
    for core in a.cores.split(","):
        print(f"--- {core} (T={a.T}, B={a.B}, mode={a.mode}) ...", flush=True)
        res = smoke(core, a.T, a.B, a.iters, a.mode, a.fwp_dim, flags)
        results.append(res)
        print(json.dumps(res, indent=2), flush=True)

    ok = all(r["settled"] for r in results)
    print("\nSETTLED" if ok else "\nNOT SETTLED", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
