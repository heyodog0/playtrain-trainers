"""U16b: chi-square of each side's draws against the exact probabilities; exact probs and
straight-through comparisons."""
import sys, numpy as np
from scipy import stats
o, p = np.load(sys.argv[1]), np.load(sys.argv[2])
def chi(idx, probs):
    K = probs.shape[-1]; rows = probs.reshape(-1, K).astype(np.float64); rows = rows / rows.sum(-1, keepdims=True); idx = idx.reshape(idx.shape[0], -1); ps = []
    for r in range(rows.shape[0]):
        obs = np.bincount(idx[:, r], minlength=K).astype(float); exp = rows[r] * idx.shape[0]
        keep = exp >= 5; ob = np.append(obs[keep], obs[~keep].sum()); ex = np.append(exp[keep], exp[~keep].sum())
        if ex[-1] < 1e-9: ob, ex = ob[:-1], ex[:-1]
        ex = ex * ob.sum() / ex.sum(); ex[-1] += ob.sum() - ex.sum()
        ps.append(stats.chisquare(ob, ex).pvalue)
    return np.array(ps)
print('probs max|port-off|: policy %.2e latent %.2e' % (np.abs(p['pol_probs'] - o['pol_probs']).max(), np.abs(p['lat_probs'] - o['lat_probs']).max()))
for name, idx, probs in [('policy port', p['pol_idx'], o['pol_probs']), ('policy official', o['pol_idx'], o['pol_probs']),
                         ('latent port', p['lat_idx'], o['lat_probs']), ('latent official', o['lat_idx'], o['lat_probs'])]:
    ps = chi(idx, probs)
    print(f'{name:<16} draws {idx.shape[0]:>6} rows {len(ps):>4}  chi2 p: min {ps.min():.3f} median {np.median(ps):.3f} frac<0.01 {np.mean(ps < 0.01):.3f}  KS(p~U) {stats.kstest(ps, "uniform").pvalue:.3f}')
print('straight-through value max|port-off| %.2e  grad max|port-off| %.2e (grad scale %.2e)' % (
    np.abs(p['st_value'] - o['st_value']).max(), np.abs(p['st_grad'] - o['st_grad']).max(), np.abs(o['st_grad']).max()))
