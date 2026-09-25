import os, sys; os.environ['JAX_PLATFORMS']='cpu'
sys.path.insert(0, '/n/holylabs/gershman_lab/Users/rtruong/dreamerv3-official-run')
import jax, jax.numpy as jnp, numpy as np
jax.config.update('jax_default_matmul_precision','highest')
from embodied.jax import outs, nets
z = np.load('/n/netscratch/gershman_lab/Everyone/truong/dv3-equiv/official/params0.npz')
for k in ['rew/head/logits/kernel','rew/head/logits/bias','val/head/logits/kernel','val/head/logits/bias']:
    print(k, 'absmax', float(np.abs(z[k]).max()))
half = nets.symexp(jnp.linspace(-20, 0, 128, dtype=jnp.float32))
bins = jnp.concatenate([half, -half[:-1][::-1]], 0)
print('bins symmetric exactly:', bool(jnp.all(bins == -bins[::-1])), 'mid', float(bins[127]))
logits = jnp.zeros((4, 255), jnp.float32)
print('eager pred', outs.TwoHot(logits, bins).pred())
print('jit pred  ', jax.jit(lambda l: outs.TwoHot(l, bins).pred())(logits))
print('naive sum ', (jax.nn.softmax(logits) * bins).sum(-1))
p = jax.nn.softmax(logits)
print('probs all equal:', bool(jnp.all(p == p[:, :1])), float(p[0,0]))
