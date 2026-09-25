"""How big is the jit TwoHot.pred bias on the GPU the official runs used, and does it
persist once the head's distribution is no longer uniform? Exact reference: float64."""
import sys
sys.path.insert(0, '/n/holylabs/gershman_lab/Users/rtruong/dreamerv3-official-run')
import jax, jax.numpy as jnp, numpy as np
from embodied.jax import outs, nets
print('devices', jax.devices())
half = nets.symexp(jnp.linspace(-20, 0, 128, dtype=jnp.float32))
bins = jnp.concatenate([half, -half[:-1][::-1]], 0)
b64 = np.asarray(bins, np.float64)
jpred = jax.jit(lambda l: outs.TwoHot(l, bins).pred())
epred = lambda l: outs.TwoHot(l, bins).pred()
rng = np.random.default_rng(0)
print('zero logits: jit', float(jpred(jnp.zeros((1, 255)))[0]), 'eager', float(epred(jnp.zeros((1, 255)))[0]))
for std in [0.0, 0.01, 0.1, 0.5, 1.0, 2.0, 4.0]:
  for center in [0.0, 3.0]:
    # centre>0: mass concentrated near the middle bins (small predicted values), as a trained head would be
    idx = np.abs(np.arange(255) - 127)
    l = rng.normal(0, std, (4096, 255)) - center * idx / 127.0 * 10
    l = l.astype(np.float32)
    p64 = np.exp(l - l.max(-1, keepdims=True)); p64 /= p64.sum(-1, keepdims=True)
    exact = (p64 * b64).sum(-1)
    j = np.asarray(jpred(jnp.asarray(l)), np.float64)
    print(f'std {std:<4} center {center}: exact mean {exact.mean():+.4g}  jit-exact mean {np.mean(j - exact):+.4g}  |jit-exact| mean {np.mean(np.abs(j - exact)):.4g}  max mass on |bin|>1e4: {p64[:, np.abs(b64) > 1e4].sum(-1).mean():.3g}')
