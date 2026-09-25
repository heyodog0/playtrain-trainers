"""U16e port side (CPU, fp32): port LaProp, Normalize and SlowModel on the same streams."""
import os, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from playtrain_trainers.dreamerv3 import opt as OPT, ac as AC, nets as NN
import long_streams as S
torch.set_num_threads(8); N = int(sys.argv[2]); off = np.load(sys.argv[1])
def rel(a, b): return float(np.abs(a - b).max() / (np.abs(b).max() + 1e-30))
params = {k: torch.nn.Parameter(torch.as_tensor(v)) for k, v in S.init_params().items()}
opt = OPT.LaProp(list(params.values()), lr=4e-5, agc=0.3, eps=1e-20, beta1=0.9, beta2=0.999, momentum=True, warmup=1000)
worst = 0.0
for t in range(N):
    g = S.grads(t)
    for k, p in params.items(): p.grad = torch.as_tensor(g[k])
    opt.step()
    if (t + 1) % 500 == 0:
        e = max(rel(params[k].detach().numpy(), off[f'opt_{k}_{t + 1}']) for k in params)
        d = max(rel(params[k].detach().numpy() - S.init_params()[k], off[f'opt_{k}_{t + 1}'] - S.init_params()[k]) for k in params)
        worst = max(worst, d)
        if (t + 1) % 2500 == 0: print(f'optimizer step {t + 1:>5}: params rel {e:.1e}  displacement-from-init rel {d:.1e}')
print(f'optimizer worst displacement rel over {N} steps: {worst:.1e}')
norm = AC.Normalize('perc', 0.01, 1.0, 5.0, 95.0, False); offs, scs = [], []
for t in range(N):
    o, s = norm(torch.as_tensor(S.returns(t)), True); offs.append(float(o)); scs.append(float(s))
offs, scs = np.array(offs), np.array(scs)
print(f"retnorm offset max abs err {np.abs(offs - off['norm_offset']).max():.1e} (scale of offset {np.abs(off['norm_offset']).max():.1f}); scale max rel err {np.max(np.abs(scs - off['norm_scale']) / off['norm_scale']):.1e}; final offset {offs[-1]:.4f}/{off['norm_offset'][-1]:.4f} scale {scs[-1]:.4f}/{off['norm_scale'][-1]:.4f}")
for rate, every in [(0.02, 1), (0.1, 3)]:
    src = NN.Linear(8, 16); slow = AC.SlowModel(src, rate=rate, every=every); w = 0.0
    for t in range(N):
        s_ = S.source(t)
        with torch.no_grad(): src.kernel.copy_(torch.as_tensor(s_['kernel'])); src.bias.copy_(torch.as_tensor(s_['bias']))
        slow.update()
        if (t + 1) % 500 == 0:
            w = max(w, rel(slow.model.kernel.numpy(), off[f'slow_{rate}_{every}_kernel_{t + 1}']), rel(slow.model.bias.numpy(), off[f'slow_{rate}_{every}_bias_{t + 1}']))
    print(f'slow model rate {rate} every {every}: worst rel err over {N} steps {w:.1e}')
