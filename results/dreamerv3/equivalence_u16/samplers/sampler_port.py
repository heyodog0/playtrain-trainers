"""U16b port side: the port's real samplers (on DEVICE) on the official's logits; chi-square vs exact
probs for both sides; exact comparison of probs and straight-through gradients."""
import os, sys, numpy as np, torch
from playtrain_trainers.dreamerv3 import outs
DEV = os.environ.get('DEVICE', 'cuda')
z = np.load(sys.argv[1]); S = 200_000
torch.manual_seed(0)
pol = torch.as_tensor(z['pol_logits'], device=DEV); lat = torch.as_tensor(z['lat_logits'], device=DEV)
# exact probabilities computed by each side
dp = outs.Categorical(pol); dl = outs.OneHot(lat, 0.01)
pp = torch.softmax(dp.logits, -1).cpu().numpy(); lp = torch.softmax(dl.dist.logits, -1).cpu().numpy()
print('probs max|port-off|: policy %.2e latent %.2e' % (np.abs(pp - z['pol_probs']).max(), np.abs(lp - z['lat_probs']).max()))
# port samples: expand to (S, ...) and draw once with the real sample()
pidx = outs.Categorical(pol.expand(S, *pol.shape)).sample().cpu().numpy()
loh = outs.OneHot(lat.expand(S // 10, *lat.shape), 0.01).sample()
assert torch.allclose(loh.sum(-1), torch.ones_like(loh.sum(-1)))
lidx = loh.argmax(-1).cpu().numpy()
l = torch.tensor(z['lat_logits'], requires_grad=True)
v = outs.OneHot(l, 0.01).pred(); (torch.as_tensor(z['w']) * v).sum().backward()
np.savez(sys.argv[2], pol_probs=pp, lat_probs=lp, pol_idx=pidx, lat_idx=lidx,
         st_value=v.detach().numpy(), st_grad=l.grad.numpy())
print('saved port samples')
