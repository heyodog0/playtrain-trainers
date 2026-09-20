"""Where do the fast-weight cores' enormous gradients actually come from?

Training runs show CompFWP demanding gradient norms of 1e3 to 1e7 against a
clip of 40, where the LSTM asks for 370 — so every update is clipped and the
optimiser only ever sees a direction. This attributes that to parameter
groups, on a laptop, with no cluster time.

It drives the real ``learn()`` loss rather than a surrogate, so the numbers
connect to the training runs: clipping is set absurdly high so the recorded
gradients are pre-clip, and the learning rate is zero so the parameters do not
move between measurements.

    uv run python tools/grad_forensics.py
    uv run python tools/grad_forensics.py --unroll 1,10,25,50,100
    uv run python tools/grad_forensics.py --fwp-dim 32,64,128
"""

from __future__ import annotations

import argparse
from collections import OrderedDict

import torch

from playtrain_trainers.impala.learn import learn
from playtrain_trainers.impala.net import ImpalaNet

SPEC = {"observation_shape": (3, 64, 64), "num_actions": 8, "features_dim": 256}
NO_CLIP = 1e12


def make_batch(T: int, B: int, seed: int = 0) -> dict:
    gen = torch.Generator().manual_seed(seed)
    done = torch.zeros(T + 1, B, dtype=torch.bool)
    done[T // 3, 0] = True
    done[2 * T // 3, B - 1] = True
    A = SPEC["num_actions"]
    return {
        "frame": torch.randint(0, 256, (T + 1, B, *SPEC["observation_shape"]),
                               dtype=torch.uint8, generator=gen),
        "reward": torch.randn(T + 1, B, generator=gen),
        "done": done,
        "last_action": torch.randint(0, A, (T + 1, B), dtype=torch.int64, generator=gen),
        "episode_return": torch.randn(T + 1, B, generator=gen).abs() * 1000,
        "policy_logits": torch.randn(T + 1, B, A, generator=gen),
        "action": torch.randint(0, A, (T + 1, B), dtype=torch.int64, generator=gen),
        "baseline": torch.randn(T + 1, B, generator=gen),
    }


def group_of(name: str) -> str:
    """Bucket a parameter by the component it belongs to."""
    if name.startswith("encoder"):
        return "encoder"
    if name.startswith("core.set_block"):
        return "core.set_block"
    for part in ("W_k", "W_v", "W_q", "W_o", "W_p", "w_b"):
        if f"core.{part}" in name:
            return f"core.{part}"
    if name.startswith("core"):
        return "core.rnn"
    if name.startswith(("policy", "baseline")):
        return "heads"
    if name.startswith("mix"):
        return "mix"
    return "other"


def load_trained(model: torch.nn.Module, path: str) -> str:
    """Load a run's final.pt, whatever shape the checkpoint happens to be."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("model_state_dict", "state_dict", "model"):
        if isinstance(blob, dict) and key in blob and isinstance(blob[key], dict):
            blob = blob[key]
            break
    blob = {k.replace("_orig_mod.", ""): v for k, v in blob.items()}
    missing, unexpected = model.load_state_dict(blob, strict=False)
    return f"loaded (missing {len(missing)}, unexpected {len(unexpected)})"


def grad_norms(core: str, T: int, B: int, fwp_dim: int, seed: int = 0,
               ckpt: str | None = None, flags: dict | None = None, steps: int = 0) -> dict:
    torch.manual_seed(seed)
    kwargs = {"fwp_dim": fwp_dim, **(flags or {})} if core in ("deltanet", "compfwp") else {}
    model = ImpalaNet(**SPEC, core=core, **kwargs)
    if ckpt:
        load_trained(model, ckpt)
    init = {k: v.detach().clone() for k, v in model.state_dict().items() if v.is_floating_point()}
    if steps:
        # Displacement probe: N real learn() steps with the trainer's optimiser
        # on fresh random batches. Random targets, so the direction is noise —
        # what this measures is how much gradient reaches each group, x lr,
        # which is exactly the throttling question.
        opt = torch.optim.RMSprop(model.parameters(), lr=1e-4, alpha=0.99, eps=1e-5)
        for i in range(steps):
            learn(actor_model=None, learner_model=model, batch=make_batch(T, B, seed + 1 + i),
                  initial_agent_state=model.initial_state(B), optimizer=opt, scheduler=None,
                  discounting=0.99, baseline_cost=0.5, entropy_cost=0.01, grad_norm_clipping=40.0)
        disp: dict[str, list] = {}
        for k, v in model.state_dict().items():
            if k in init:
                g = group_of(k)
                d, n = (v.float() - init[k].float()).pow(2).sum().item(), init[k].float().pow(2).sum().item()
                disp.setdefault(g, [0.0, 0.0]); disp[g][0] += d; disp[g][1] += n
        displacement = {g: (d / n) ** 0.5 if n > 0 else float("inf") for g, (d, n) in disp.items()}
    else:
        displacement = None
    batch = make_batch(T, B, seed)
    learn(
        actor_model=None, learner_model=model, batch=batch,
        initial_agent_state=model.initial_state(B),
        optimizer=torch.optim.SGD(model.parameters(), lr=0.0),  # no movement
        scheduler=None, discounting=0.99, baseline_cost=0.5, entropy_cost=0.01,
        grad_norm_clipping=NO_CLIP,  # so .grad is pre-clip
    )
    groups: dict[str, float] = {}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        groups[group_of(name)] = groups.get(group_of(name), 0.0) + float(p.grad.pow(2).sum())
    total = sum(groups.values()) ** 0.5
    return {"total": total, "displacement": displacement,
            "groups": OrderedDict(sorted(((k, v ** 0.5) for k, v in groups.items()),
                                         key=lambda kv: -kv[1]))}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cores", default="lstm,deltanet,compfwp")
    p.add_argument("-T", "--unroll", default="100")
    p.add_argument("-B", type=int, default=8)
    p.add_argument("--fwp-dim", default="128")
    p.add_argument("--seeds", default="0")
    p.add_argument("--flags", default="", help="k=v,... passed to ImpalaNet for fwp cores")
    p.add_argument("--steps", type=int, default=0, help="learn() steps before measuring displacement")
    p.add_argument("--ckpt", default=None,
                   help="core=path,core=path — measure trained weights instead of init")
    a = p.parse_args(argv)

    Ts = [int(x) for x in a.unroll.split(",")]
    dims = [int(x) for x in a.fwp_dim.split(",")]
    seeds = [int(x) for x in a.seeds.split(",")]

    for T in Ts:
        for dim in dims:
            tag = f"T={T} B={a.B} fwp_dim={dim}"
            print(f"\n=== {tag} " + "=" * max(0, 56 - len(tag)))
            totals = {}
            ckpts = dict(x.split("=", 1) for x in a.ckpt.split(",")) if a.ckpt else {}
            for core in a.cores.split(","):
                label = core
                if core in ckpts:
                    label = f"{core} [trained]"
                flags = {}
                for kv in filter(None, a.flags.split(",")):
                    k, v = kv.split("=", 1)
                    flags[k] = (v.lower() == "true") if v.lower() in ("true", "false") else float(v)
                per_seed = [grad_norms(core, T, a.B, dim, s, ckpts.get(core), flags, a.steps) for s in seeds]
                total = sum(r["total"] for r in per_seed) / len(per_seed)
                totals[core] = total
                print(f"\n{label}: total grad norm {total:,.1f}")
                merged: dict[str, float] = {}
                for r in per_seed:
                    for k, v in r["groups"].items():
                        merged[k] = merged.get(k, 0.0) + v / len(per_seed)
                dispm: dict[str, float] = {}
                for r in per_seed:
                    for k, v in (r["displacement"] or {}).items():
                        dispm[k] = dispm.get(k, 0.0) + v / len(per_seed)
                for k, v in sorted(merged.items(), key=lambda kv: -kv[1]):
                    share = 100 * v**2 / max(total**2, 1e-30)
                    dtxt = f"   disp {dispm[k]:.4f}" if k in dispm else ""
                    print(f"    {k:18s} {v:>14,.1f}   {share:5.1f}% of total{dtxt}")
            if "lstm" in totals:
                print()
                for core, t in totals.items():
                    if core != "lstm":
                        print(f"  {core} / lstm = {t / totals['lstm']:.1f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
