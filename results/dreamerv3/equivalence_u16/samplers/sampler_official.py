"""U16b official side: real jax samplers on fixed logits; exact probs; straight-through grads."""
import os, sys, numpy as np
os.environ['JAX_PLATFORMS'] = 'cpu'
sys.path.insert(0, '/n/holylabs/gershman_lab/Users/rtruong/dreamerv3-official-run')
import jax, jax.numpy as jnp
jax.config.update('jax_default_matmul_precision', 'highest')
from embodied.jax import outs
OUT = sys.argv[1]; S = 200_000
rng = np.random.default_rng(0)
pol = (rng.normal(size=(4, 18)) * 2).astype(np.float32)
lat = (rng.normal(size=(4, 32, 64)) * 3).astype(np.float32)
w = rng.normal(size=(4, 32, 64)).astype(np.float32)
res = {'pol_logits': pol, 'lat_logits': lat, 'w': w}
# policy: Categorical, no unimix (D-027); latent: OneHot with unimix 0.01, as rssm._dist
dp = outs.Categorical(jnp.asarray(pol))
res['pol_probs'] = np.asarray(jax.nn.softmax(dp.logits, -1))
res['pol_idx'] = np.asarray(dp.sample(jax.random.PRNGKey(1), (S,)))            # (S, 4)
dl = outs.OneHot(jnp.asarray(lat), 0.01)
res['lat_probs'] = np.asarray(jax.nn.softmax(dl.dist.logits, -1))
oh = np.asarray(dl.sample(jax.random.PRNGKey(2), (S // 10,)))                  # (S/10, 4, 32, 64)
res['lat_idx'] = oh.argmax(-1)
assert np.allclose(oh.sum(-1), 1)
# straight-through gradient of sum(w * onehot) wrt logits, through pred (deterministic index)
g = jax.grad(lambda l: (jnp.asarray(w) * outs.OneHot(l, 0.01).pred()).sum())(jnp.asarray(lat))
res['st_grad'] = np.asarray(g)
res['st_value'] = np.asarray(outs.OneHot(jnp.asarray(lat), 0.01).pred())
np.savez(OUT, **res); print('saved', {k: v.shape for k, v in res.items()})
