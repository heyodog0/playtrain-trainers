"""FLOPs of every matmul/conv in one port train_step (forward + backward), by dtype."""
import os, sys, collections, numpy as np, torch
from math import prod
from torch.utils._python_dispatch import TorchDispatchMode
from playtrain_trainers.dreamerv3 import config as C
from playtrain_trainers.dreamerv3.agent import Agent
aten = torch.ops.aten
tot, n = collections.Counter(), collections.Counter()
class Count(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        f = func.overloadpacket
        def add(kind, dts, fl):
            k = (kind, dts); tot[k] += fl; n[k] += 1
        if f in (aten.mm, aten.bmm):
            a, b = args[0], args[1]
            add('dot', f'{a.dtype}x{b.dtype}->{out.dtype}', 2 * prod(out.shape) * a.shape[-1])
        elif f in (aten.addmm, aten.baddbmm):
            a, b = args[1], args[2]
            add('dot', f'{a.dtype}x{b.dtype}->{out.dtype}', 2 * prod(out.shape) * a.shape[-1])
        elif f is aten.convolution:
            x, w = args[0], args[1]
            add('convolution', f'{x.dtype}x{w.dtype}->{out.dtype}', 2 * prod(out.shape) * prod(w.shape) // w.shape[0])
        elif f is aten.convolution_backward:
            g, x, w = args[0], args[1], args[2]
            add('convolution', f'{g.dtype}x{w.dtype}->bwd', 2 * 2 * prod(g.shape) * prod(w.shape) // w.shape[0])
        return out
cfg = C.config_from_dict({"preset": "atari100k", "batch_size": 2, "batch_length": 8,
                          "compute_dtype": os.environ.get("DTYPE", "bfloat16"), "device": "cuda"})
agent = Agent((64, 64, 3), 18, cfg).to("cuda")
b = np.load(sys.argv[1])
data = {"image": b["image"], "reward": b["reward"], "is_first": b["is_first"], "is_last": b["is_last"],
        "is_terminal": b["is_terminal"], "action": b["action"].astype(np.int64), "stepid": b["stepid"],
        "deter": b["dyn/deter"], "stoch": b["dyn/stoch"]}
data = {k: torch.as_tensor(v).cuda() for k, v in data.items()}
with Count():
    agent.train_step({}, data)
T = sum(tot.values())
for k in sorted(tot, key=lambda k: -tot[k]):
    print(f'{k[0]:<12} {k[1]:<44} n {n[k]:>5}  GFLOP {tot[k]/1e9:>10.3f}  {100*tot[k]/T:6.2f}%')
print('TOTAL GFLOP', T/1e9)
