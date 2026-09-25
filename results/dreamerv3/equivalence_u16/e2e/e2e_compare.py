import json, sys, numpy as np
o = json.load(open(sys.argv[1] + '/e2e.json')); p = json.load(open(sys.argv[2] + '/e2e.json'))
To, Tp = np.array(o['trans']), np.array(p['trans'])
n = min(len(To), len(Tp)); diff = np.where(np.any(To[:n] != Tp[:n], 1))[0]
print(f'transitions: official {len(To)} port {len(Tp)}; identical through step {diff[0] if len(diff) else n} of {n}' + (f'; first difference at step {diff[0]}: official {To[diff[0]].tolist()} port {Tp[diff[0]].tolist()}' if len(diff) else ''))
print(f'episodes: official {int(To[:, 3].sum())} port {int(Tp[:, 3].sum())}')
print(f'batches sampled: official {len(o["batches"])} port {len(p["batches"])}')
exact = ('image', 'action', 'reward', 'is_first', 'is_last', 'is_terminal')
for i, (a, b) in enumerate(zip(o['batches'], p['batches'])):
    ok = [k for k in exact if a[k] == b[k]]
    da, db = np.array(a['deter_slice']), np.array(b['deter_slice'])
    rel = np.abs(da - db).max() / (np.abs(da).max() + 1e-12)
    sm = int((np.array(a['stoch_argmax']) != np.array(b['stoch_argmax'])).sum())
    if i < 6 or len(ok) < len(exact) or rel > 1e-3 or i % 10 == 0:
        print(f'  batch {i:>3}: exact keys equal {len(ok)}/{len(exact)}{"" if len(ok)==len(exact) else " missing " + str([k for k in exact if k not in ok])}  entries deter rel {rel:.1e}  stoch argmax mismatches {sm}')
keys = ['loss/image', 'loss/dyn', 'loss/con', 'loss/rew', 'loss/value', 'loss/repval', 'loss/policy']
print(f'train steps with metrics: official {len(o["mets"])} port {len(p["mets"])}')
for i, (a, b) in enumerate(zip(o['mets'], p['mets'])):
    rels = {k: abs(a[k] - b[k]) / max(abs(a[k]), 1e-8) for k in keys if k in a and k in b}
    gn = abs(a['opt/grad_norm'] - b['grad_norm']) / a['opt/grad_norm']
    worst = max(rels, key=rels.get)
    if i < 5 or i % 10 == 0 or i == len(o['mets']) - 1 or max(rels.values()) > 1e-2:
        print(f'  train {i:>3}: worst {worst} rel {rels[worst]:.1e}  image rel {rels["loss/image"]:.1e}  grad_norm rel {gn:.1e}')
