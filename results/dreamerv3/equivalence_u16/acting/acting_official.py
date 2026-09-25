"""U16c official side: agent.policy stepped over a recorded observation sequence (with a reset
mid-sequence), from params loaded from INPUTS, argmax sampling. Saves actions and entries per step."""
import os, sys, pathlib, numpy as np
TREE = pathlib.Path('/n/holylabs/gershman_lab/Users/rtruong/dreamerv3-official-run')
sys.path.insert(0, str(TREE)); sys.path.insert(1, str(TREE / 'dreamerv3'))
import jax, jax.numpy as jnp
jax.config.update('jax_default_matmul_precision', 'highest')
import embodied.jax.outs as outs
outs.Categorical.sample = lambda self, seed, shape=(): jnp.argmax(self.logits, -1)
import elements, ruamel.yaml as yaml
from dreamerv3 import main as M
OUT = pathlib.Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
DTYPE = os.environ.get('DTYPE', 'float32'); INPUTS = pathlib.Path(os.environ['INPUTS'])
configs = yaml.YAML(typ='safe').load((TREE / 'dreamerv3' / 'configs.yaml').read_text())
config = elements.Config(configs['defaults']).update(configs['atari100k']).update({
    'task': 'atari100k_frostbite', 'seed': 0, 'logdir': str(OUT / 'logdir'), 'batch_size': 2, 'batch_length': 8,
    'report_length': 8, 'jax.platform': 'cuda', 'jax.compute_dtype': DTYPE, 'jax.prealloc': False})
agent = M.make_agent(config)
jax.config.update('jax_transfer_guard', 'allow')
src = np.load(INPUTS / 'params0.npz')
for k in list(agent.params):
  if k in src.files and not k.startswith('opt/'):
    agent.params[k] = jax.device_put(jnp.asarray(src[k]), agent.params[k].sharding)
for k in list(agent.policy_params):
  agent.policy_params[k] = jax.device_put(jnp.asarray(src[k]), agent.policy_params[k].sharding)
# A recorded sequence: 40 steps of real frames from random actions, with an env reset at step 20.
env = M.make_env(config, 0); rng = np.random.default_rng(7)
nact = int(env.act_space['action'].high); seq = []
obs = env.step({'action': np.int32(0), 'reset': True})
for t in range(40):
  if t == 20:
    obs = env.step({'action': np.int32(0), 'reset': True})
  seq.append({k: np.asarray(v) for k, v in obs.items() if not k.startswith('log/')})
  obs = env.step({'action': np.int32(rng.integers(nact)), 'reset': False})
env.close()
if os.environ.get('SEQ'):
  z = np.load(os.environ['SEQ']); keys = [k[4:] for k in z.files if k.startswith('obs_')]
  seq = [{k: z['obs_' + k][t] for k in keys} for t in range(len(z['acts']))]
assert seq[0]['is_first'] and seq[20]['is_first'] and not seq[5]['is_first']
carry = agent.init_policy(1)
acts, deters, stochs = [], [], []
for o in seq:
  batch = {k: v[None] for k, v in o.items()}
  carry, act, out = agent.policy(carry, batch, mode='train')
  acts.append(int(np.asarray(act['action'])[0]))
  deters.append(np.asarray(out['dyn/deter'], np.float32)[0]); stochs.append(np.asarray(out['dyn/stoch'], np.float32)[0])
np.savez(OUT / 'acting.npz', acts=np.array(acts), deter=np.stack(deters), stoch=np.stack(stochs),
         **{f'obs_{k}': np.stack([o[k] for o in seq]) for k in seq[0]})
print('official acts', acts)
