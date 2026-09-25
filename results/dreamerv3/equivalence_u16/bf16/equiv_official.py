"""U15c': the official side of the same-batch equivalence test.

Runs in the pinned official tree and venv (dreamerv3-official-run, jax 0.4.33,
ale_py 0.9.0), on CPU in float32 at 'highest' matmul precision, with every
Categorical.sample replaced by argmax. That one patch reaches all three sampling
sites -- the RSSM latents (OneHot.sample -> dist.sample), the imagination policy and
the acting policy (agent.sample -> Categorical.sample) -- so a train step is
deterministic given params and batch.

Writes to OUT:
  params0.npz      flat official params before any update, keyed by ninjax path
  batch.npz        the exact batch fed to train (real frostbite frames, random actions)
  step{1,2,3}.json scalar metrics returned by each train step
  params1.npz      params after the first update
  spaces.json      the batch spaces the agent expects
"""
import json
import os
import pathlib
import sys

TREE = pathlib.Path(sys.argv[1])
OUT = pathlib.Path(sys.argv[2])
PLATFORM = sys.argv[3] if len(sys.argv) > 3 else 'cpu'
if PLATFORM == 'cpu':
  os.environ['JAX_PLATFORMS'] = 'cpu'
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(TREE))
sys.path.insert(1, str(TREE / 'dreamerv3'))

import jax
jax.config.update('jax_default_matmul_precision', 'highest')
import jax.numpy as jnp
import numpy as np

import embodied.jax.outs as outs


def det_sample(self, seed, shape=()):
  assert shape == (), shape
  return jnp.argmax(self.logits, -1)


outs.Categorical.sample = det_sample

import jax.stages
if os.environ.get('DUMP_HLO'):
  # Save the unoptimized HLO of every program the agent compiles (U16a dtype audit).
  _orig_compile = jax.stages.Lowered.compile
  _count = [0]
  def _compile(self, *a, **k):
    _count[0] += 1
    (OUT / f'program{_count[0]}.hlo.txt').write_text(self.as_text(dialect='hlo'))
    return _orig_compile(self, *a, **k)
  jax.stages.Lowered.compile = _compile

import elements
import ruamel.yaml as yaml
from dreamerv3 import main as M

B, L = 2, 9  # data length; the official batch_length EXCLUDES replay_context=1, so it is L - 1 = 8
configs = yaml.YAML(typ='safe').load((TREE / 'dreamerv3' / 'configs.yaml').read_text())
config = elements.Config(configs['defaults'])
config = config.update(configs['atari100k'])
config = config.update({
    'task': 'atari100k_frostbite', 'seed': int(os.environ.get('SEED', 0)), 'logdir': str(OUT / 'logdir'),
    'batch_size': B, 'batch_length': L - 1, 'report_length': L - 1,
    'jax.platform': PLATFORM, 'jax.compute_dtype': os.environ.get('DTYPE', 'float32'), 'jax.prealloc': False,
    'jax.jit': True,
})
if os.environ.get('WARMUP') is not None:
  config = config.update({'agent.opt.warmup': int(os.environ['WARMUP'])})
print('warmup', config.agent.opt.warmup)
agent = M.make_agent(config)
# The official code sets jax_transfer_guard 'disallow' on accelerators; this script reads
# params and metrics back to the host on purpose.
jax.config.update('jax_transfer_guard', 'allow')
spaces = {k: [str(v.dtype), list(v.shape)] for k, v in agent.spaces.items()}
(OUT / 'spaces.json').write_text(json.dumps(spaces, indent=1))
print('spaces', spaces)

INPUTS = os.environ.get('INPUTS')
if INPUTS:
  # Fixed inputs from an earlier run: its initial model params (optimizer state stays as
  # freshly initialized, i.e. zeros) and, below, its batch.
  src = np.load(pathlib.Path(INPUTS) / 'params0.npz')
  for k in list(agent.params):
    if k in src.files and not k.startswith('opt/'):
      assert src[k].shape == agent.params[k].shape, k
      agent.params[k] = jax.device_put(jnp.asarray(src[k]), agent.params[k].sharding)
  print('loaded params0 and batch from', INPUTS)
params0 = {k: np.asarray(v) for k, v in agent.params.items()}
np.savez(OUT / 'params0.npz', **params0)
print('params', len(params0), sum(v.size for v in params0.values()))

# A real batch: two random-action frostbite rollouts, each starting at a reset.
env = M.make_env(config, 0)
rng = np.random.default_rng(0)
nact = int(env.act_space['action'].high)
seqs = []
for b in range(B):
  obs = env.step({'action': np.int32(0), 'reset': True})
  seq = []
  for t in range(L):
    act = np.int32(rng.integers(nact))
    seq.append({**{k: v for k, v in obs.items() if not k.startswith('log/')}, 'action': act})
    obs = env.step({'action': act, 'reset': False})
  seqs.append(seq)
env.close()
batch = {}
for k in seqs[0][0]:
  batch[k] = np.stack([np.stack([np.asarray(s[t][k]) for t in range(L)]) for s in seqs])
for k, (dtype, shape) in spaces.items():
  if k not in batch:
    batch[k] = np.zeros((B, L, *shape), dtype)  # stepid, consec, replay entries
for k, (dtype, shape) in spaces.items():
  batch[k] = batch[k].astype(dtype)
  assert batch[k].shape == (B, L, *shape), (k, batch[k].shape, shape)
if INPUTS:
  batch = dict(np.load(pathlib.Path(INPUTS) / 'batch.npz'))
np.savez(OUT / 'batch.npz', **batch)
print('batch', {k: (v.dtype, v.shape) for k, v in batch.items()})

carry = agent.init_train(B)
for step in (1, 2, 3):
  data = {k: jnp.asarray(v) for k, v in batch.items()}
  allo = {k: v for k, v in agent.params.items() if k in agent.policy_keys}
  dona = {k: v for k, v in agent.params.items() if k not in agent.policy_keys}
  seed = agent._seeds(0, agent.train_mirrored)
  agent.params, carry, _, mets = agent._train(dona, allo, seed, carry, data)
  mets = {k: float(np.asarray(v)) for k, v in mets.items() if np.asarray(v).size == 1}
  (OUT / f'step{step}.json').write_text(json.dumps(mets, indent=1, sort_keys=True))
  print('step', step, {k: v for k, v in mets.items() if k.startswith(('loss/', 'opt/'))})
  if step in (1, 3):
    np.savez(OUT / f'params{step}.npz', **{k: np.asarray(v) for k, v in agent.params.items() if not k.startswith('opt/')})
  carry = agent.init_train(B)  # same starting carry each step: only params change
print('DONE')
