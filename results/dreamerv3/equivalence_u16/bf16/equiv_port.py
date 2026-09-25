"""U15c': the port side of the same-batch equivalence test.

Loads the OFFICIAL initial params (params0.npz from equiv_official.py) into the port,
feeds the identical batch, replaces Categorical.sample with argmax exactly as the
official side does, and runs the same three train steps in float32 on CPU. Then
compares every loss term and the gradient norm against the official step{1,2,3}.json.

Usage: python equiv_port.py OFFICIAL_OUT_DIR
"""
import json
import pathlib
import re
import sys

import numpy as np
import torch

import os
DEVICE = os.environ.get("DEVICE", "cpu")
torch.set_num_threads(16)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.use_deterministic_algorithms(True)

from playtrain_trainers.dreamerv3 import config as C
from playtrain_trainers.dreamerv3 import outs as O
from playtrain_trainers.dreamerv3.agent import Agent

OFF = pathlib.Path(sys.argv[1])


def det_sample(self, generator=None):
    return torch.argmax(self.logits, -1)


O.Categorical.sample = det_sample


def official_name(port: str) -> str | None:
    """Port state_dict key -> official ninjax key. None for keys that are aliases."""
    if port.startswith(("opt_modules.", "slowval.source.")):
        return None  # the same tensors registered a second time
    rules = [
        (r"^wm\.enc\.convs\.(\d)\.(kernel|bias)$", lambda m: f"enc/cnn{m[1]}/{m[2]}"),
        (r"^wm\.enc\.norms\.(\d)\.scale$", lambda m: f"enc/cnn{m[1]}norm/scale"),
        (r"^wm\.dyn\.(dynin\d|dyngru|obslogit|priorlogit)\.(kernel|bias)$", lambda m: f"dyn/{m[1]}/{m[2]}"),
        (r"^wm\.dyn\.(dynin\d)norm\.scale$", lambda m: f"dyn/{m[1]}norm/scale"),
        (r"^wm\.dyn\.(dynhid|obs|prior)\.(\d)\.(kernel|bias)$", lambda m: f"dyn/{m[1]}{m[2]}/{m[3]}"),
        (r"^wm\.dyn\.(dynhid|obs|prior)norm\.(\d)\.scale$", lambda m: f"dyn/{m[1]}{m[2]}norm/scale"),
        (r"^wm\.dec\.(sp\d|imgout)\.(kernel|bias)$", lambda m: f"dec/{m[1]}/{m[2]}"),
        (r"^wm\.dec\.(sp1norm|spnorm)\.scale$", lambda m: f"dec/{m[1]}/scale"),
        # the decoder's conv stack is indexed in reverse relative to the port
        (r"^wm\.dec\.convs\.(\d)\.(kernel|bias)$", lambda m: f"dec/conv{2 - int(m[1])}/{m[2]}"),
        (r"^wm\.dec\.norms\.(\d)\.scale$", lambda m: f"dec/conv{2 - int(m[1])}norm/scale"),
        (r"^wm\.(rew|con)\.mlp\.lins\.(\d)\.(kernel|bias)$", lambda m: f"{m[1]}/mlp/linear{m[2]}/{m[3]}"),
        (r"^wm\.(rew|con)\.mlp\.norms\.(\d)\.scale$", lambda m: f"{m[1]}/mlp/norm{m[2]}/scale"),
        (r"^wm\.rew\.out\.(kernel|bias)$", lambda m: f"rew/head/logits/{m[1]}"),
        (r"^wm\.con\.out\.(kernel|bias)$", lambda m: f"con/head/logit/{m[1]}"),
        (r"^(pol|val)\.mlp\.lins\.(\d)\.(kernel|bias)$", lambda m: f"{m[1]}/mlp/linear{m[2]}/{m[3]}"),
        (r"^(pol|val)\.mlp\.norms\.(\d)\.scale$", lambda m: f"{m[1]}/mlp/norm{m[2]}/scale"),
        (r"^slowval\.model\.mlp\.lins\.(\d)\.(kernel|bias)$", lambda m: f"slowval/mlp/linear{m[1]}/{m[2]}"),
        (r"^slowval\.model\.mlp\.norms\.(\d)\.scale$", lambda m: f"slowval/mlp/norm{m[1]}/scale"),
        (r"^pol\.out\.(kernel|bias)$", lambda m: f"pol/head/action/logits/{m[1]}"),
        (r"^val\.out\.(kernel|bias)$", lambda m: f"val/head/logits/{m[1]}"),
        (r"^slowval\.model\.out\.(kernel|bias)$", lambda m: f"slowval/head/logits/{m[1]}"),
        (r"^retnorm\.(lo|hi)$", lambda m: f"retnorm/{m[1]}/value"),
        (r"^slowval\.count$", lambda m: "slowval_count/value"),
    ]
    for pat, fn in rules:
        m = re.match(pat, port)
        if m:
            return fn(m)
    raise KeyError(f"no mapping for port key {port}")


import os
raw = {"preset": "atari100k", "batch_size": 2, "batch_length": 8, "compute_dtype": os.environ.get("DTYPE", "float32"), "device": DEVICE}
if os.environ.get("WARMUP") is not None:
    raw["agent"] = {"opt": {"warmup": int(os.environ["WARMUP"])}}
cfg = C.config_from_dict(raw)
print("warmup", cfg.agent.opt.warmup, "dtype", cfg.compute_dtype)
torch.manual_seed(0)
agent = Agent((64, 64, 3), 18, cfg).to(DEVICE)
params0 = np.load(OFF / "params0.npz")
sd = agent.state_dict()
used, loaded = set(), {}
for key, tensor in sd.items():
    name = official_name(key)
    if name is None:
        continue
    src = params0[name]
    assert tuple(src.shape) == tuple(tensor.shape), (key, name, src.shape, tuple(tensor.shape))
    loaded[key] = torch.as_tensor(src).to(tensor.dtype).to(tensor.device)
    used.add(name)
missing = [k for k in params0.files if k not in used and not k.startswith("opt/")]
assert not missing, f"official params not loaded: {missing}"
agent.load_state_dict({**sd, **loaded})
print("port device", agent.device())
# The aliases must now hold the loaded values too.
assert torch.equal(agent.slowval.source.out.kernel, agent.val.out.kernel)
print("loaded", len(loaded), "tensors,", sum(v.numel() for v in loaded.values()), "numbers; unmapped official:",
      sorted({k.split('/')[0] for k in params0.files if k not in used}))

batch = np.load(OFF / "batch.npz")
data = {
    "image": torch.as_tensor(batch["image"]),
    "reward": torch.as_tensor(batch["reward"]),
    "is_first": torch.as_tensor(batch["is_first"]),
    "is_last": torch.as_tensor(batch["is_last"]),
    "is_terminal": torch.as_tensor(batch["is_terminal"]),
    "action": torch.as_tensor(batch["action"]).long(),
    "stepid": torch.as_tensor(batch["stepid"]),
    "deter": torch.as_tensor(batch["dyn/deter"]),
    "stoch": torch.as_tensor(batch["dyn/stoch"]),
}
data = {k: v.to(agent.device()) for k, v in data.items()}

rows = []
for step in (1, 2, 3):
    off = json.loads((OFF / f"step{step}.json").read_text())
    carry = agent.init_policy(2) if hasattr(agent, "init_policy") else {}
    _, _, mets = agent.train_step({}, data)
    pfile = OFF / f"params{step}.npz"
    if pfile.exists():
        offp = np.load(pfile)
        worst, rels, signs = [], [], []
        for key, tensor in agent.state_dict().items():
            name = official_name(key)
            if name is None or name not in offp.files or not tensor.is_floating_point():
                continue
            p0 = torch.as_tensor(params0[name]).double()
            d_off = torch.as_tensor(offp[name]).double() - p0
            d_port = tensor.detach().cpu().double() - p0
            if d_off.norm() == 0 and d_port.norm() == 0:
                continue
            rel = float((d_port - d_off).norm() / max(float(d_off.norm()), 1e-30))
            agree = float(((d_port.sign() == d_off.sign()) | (d_off.abs() < 1e-12)).double().mean())
            rels.append(rel); signs.append(agree); worst.append((rel, name, float(d_off.norm()), float(d_port.norm()), agree))
        worst.sort(reverse=True)
        print(f"step {step} PARAM UPDATES: {len(rels)} tensors changed; median rel diff {np.median(rels):.2e}, max {max(rels):.2e}; min sign agreement {min(signs):.4f}")
        for rel, name, a, b, agree in worst[:8]:
            print(f"    {name:<36} |d_off| {a:.4e} |d_port| {b:.4e} rel {rel:.2e} sign-agree {agree:.4f}")
    ours = {k: float(v.detach()) if torch.is_tensor(v) else float(v) for k, v in mets.items() if np.ndim(v) == 0}
    if os.environ.get("PORT_OUT"):
        pathlib.Path(os.environ["PORT_OUT"]).mkdir(parents=True, exist_ok=True)
        (pathlib.Path(os.environ["PORT_OUT"]) / f"step{step}.json").write_text(json.dumps(ours, indent=1, sort_keys=True))
    if step == 1:
        common = sorted(set(off) & set(ours))
        print("ALL COMMON METRICS, step 1 (official | port | rel):")
        for k in common:
            a, b = off[k], ours[k]
            print(f"  {k:<28} {a:>14.6g} {b:>14.6g}  {abs(a - b) / max(abs(a), 1e-8):.2e}")
        print("official-only:", sorted(set(off) - set(ours)))
        print("port-only:", sorted(set(ours) - set(off)))
    for k in sorted(off):
        if not k.startswith("loss/") and k != "opt/grad_norm":
            continue
        pk = "grad_norm" if k == "opt/grad_norm" else k
        if pk not in ours:
            rows.append((step, k, off[k], None, None))
            continue
        a, b = off[k], ours[pk]
        rel = abs(a - b) / max(abs(a), 1e-8)
        rows.append((step, k, a, b, rel))
for step, k, a, b, rel in rows:
    flag = "" if rel is not None and rel < 1e-4 else "   <-- DIFFERS" if rel is not None else "   <-- MISSING in port"
    print(f"step {step} {k:<16} official {a:>14.6f}  port {b if b is None else f'{b:>14.6f}'}  rel {rel if rel is None else f'{rel:.2e}'}{flag}")
json.dump([dict(step=s, key=k, official=a, port=b, rel=r) for s, k, a, b, r in rows],
          open(OFF.parent / "equiv_compare.json", "w"), indent=1)
print("DONE")
