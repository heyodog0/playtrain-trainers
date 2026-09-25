"""Scale for the bf16 acting comparison, all on the SAME sequence (acting_float32's)."""
import sys, numpy as np
E = sys.argv[1]
of = np.load(f'{E}/acting_float32/acting.npz'); ob = np.load(f'{E}/acting_seqA_bf16/acting.npz')
pf = np.load(f'{E}/acting_float32/port_acting.npz'); pb = np.load(f'{E}/acting_seqA_bf16/port_acting.npz')
def rel(a, b): return np.abs(a - b).max(-1) / (np.abs(b).max(-1) + 1e-12)
def mm(a, b): return (a.argmax(-1) != b.argmax(-1)).sum(-1)
for name, a, b in [('off_bf16 vs off_fp32', ob, of), ('port_bf16 vs port_fp32', pb, pf), ('port_bf16 vs off_bf16', pb, ob), ('port_fp32 vs off_fp32', pf, of)]:
    r = rel(a['deter'], b['deter']); m = mm(a['stoch'], b['stoch'])
    print(f'{name:<24} actions equal {int((a["acts"] == b["acts"]).sum()):>2}/40  deter rel step1 {r[1]:.2e} step2 {r[2]:.2e} step5 {r[5]:.2e} mean(1-19) {r[1:20].mean():.2e}  stoch mismatches steps1-5 {m[1:6].tolist()}')
