import json, pathlib, statistics as st
E = pathlib.Path('/n/netscratch/gershman_lab/Everyone/truong/dv3-equiv/sweep')
keys = ['loss/image', 'loss/dyn', 'loss/con', 'loss/rew', 'loss/value', 'loss/repval', 'loss/policy']
rows = {k: {'port_fp32': [], 'off_bf16': [], 'port_bf16': [], 'port_vs_off_bf16': []} for k in keys + ['grad_norm']}
for sd in sorted(E.glob('s*')):
    for step in (1,):
        try:
            of = json.load(open(sd / 'off_fp32' / f'step{step}.json'))
            ob = json.load(open(sd / 'off_bf16' / f'step{step}.json'))
            pf = json.load(open(sd / 'port_fp32' / f'step{step}.json'))
            pb = json.load(open(sd / 'port_bf16' / f'step{step}.json'))
        except FileNotFoundError as e:
            print('missing', e); continue
        for k in keys + ['grad_norm']:
            ok = 'opt/grad_norm' if k == 'grad_norm' else k
            ref = of[ok]; r = lambda x: abs(x - ref) / max(abs(ref), 1e-8)
            rows[k]['port_fp32'].append(r(pf[k])); rows[k]['off_bf16'].append(r(ob[ok])); rows[k]['port_bf16'].append(r(pb[k]))
            rows[k]['port_vs_off_bf16'].append(abs(pb[k] - ob[ok]) / max(abs(ob[ok]), 1e-8))
print(f"{'metric':<12} {'|port32-off32|':>16} {'|off16-off32|':>16} {'|port16-off32|':>16} {'|port16-off16|':>16}   (step 1, mean over seeds)")
for k, d in rows.items():
    if not d['port_fp32']: continue
    m = {a: st.mean(v) for a, v in d.items()}
    print(f"{k:<12} {m['port_fp32']:>16.2e} {m['off_bf16']:>16.2e} {m['port_bf16']:>16.2e} {m['port_vs_off_bf16']:>16.2e}   n={len(d['port_fp32'])}")
