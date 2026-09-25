"""U16e official side (CPU, fp32): the official optax chain from Agent._make_opt, the official
Normalize (perc, the frozen retnorm config) and the official SlowModel, over 10k-step streams."""
import os, sys, numpy as np
os.environ['JAX_PLATFORMS'] = 'cpu'
TREE = '/n/holylabs/gershman_lab/Users/rtruong/dreamerv3-official-run'
sys.path.insert(0, TREE); sys.path.insert(1, TREE + '/dreamerv3'); sys.path.insert(0, os.path.dirname(__file__))
import jax, jax.numpy as jnp, ninjax as nj, optax
jax.config.update('jax_default_matmul_precision', 'highest')
from embodied.jax import nets, utils
nets.COMPUTE_DTYPE = jnp.float32  # fp32 test; Linear asserts its input is COMPUTE_DTYPE
from dreamerv3.agent import Agent
import long_streams as S
N = int(sys.argv[2]); out = {}
# optimizer: the frozen agent.opt config
chain = Agent._make_opt(None, lr=4e-5, agc=0.3, eps=1e-20, beta1=0.9, beta2=0.999, momentum=True,
                        wd=0.0, schedule='const', warmup=1000, anneal=0)
params = {k: jnp.asarray(v) for k, v in S.init_params().items()}; state = chain.init(params)
@jax.jit
def opt_step(params, state, g):
    u, state = chain.update(g, state, params); return optax.apply_updates(params, u), state
for t in range(N):
    params, state = opt_step(params, state, {k: jnp.asarray(v) for k, v in S.grads(t).items()})
    if (t + 1) % 500 == 0:
        for k, v in params.items(): out[f'opt_{k}_{t + 1}'] = np.asarray(v)
# retnorm
norm = utils.Normalize('perc', rate=0.01, limit=1.0, perclo=5.0, perchi=95.0, debias=False, name='retnorm')
f = nj.pure(lambda x: norm(x, True)); st = {}; offs, scs = [], []
for t in range(N):
    st, (o, s) = f(st, jnp.asarray(S.returns(t)), create=True, modify=True)
    offs.append(float(o)); scs.append(float(s))
out['norm_offset'] = np.array(offs); out['norm_scale'] = np.array(scs)
# slow model, two settings
for rate, every in [(0.02, 1), (0.1, 3)]:
    src = nets.Linear(16, name='src'); slow = utils.SlowModel(nets.Linear(16, name='slow'), source=src, rate=rate, every=every)
    def first(x): src(x); return slow(x)
    st, _ = nj.pure(first)({}, jnp.zeros((1, 8)), create=True, modify=True, seed=jax.random.PRNGKey(0))
    upd = nj.pure(lambda: slow.update()); ks = []
    for t in range(N):
        s_ = S.source(t); st = {**st, 'src/kernel': jnp.asarray(s_['kernel']), 'src/bias': jnp.asarray(s_['bias'])}
        st, _ = upd(st, modify=True, create=True)
        if (t + 1) % 500 == 0:
            out[f'slow_{rate}_{every}_kernel_{t + 1}'] = np.asarray(st['slow/kernel']); out[f'slow_{rate}_{every}_bias_{t + 1}'] = np.asarray(st['slow/bias'])
np.savez(sys.argv[1], **out); print('official saved', len(out), 'arrays over', N, 'steps')
