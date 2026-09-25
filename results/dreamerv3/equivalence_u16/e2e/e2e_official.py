"""U16d end to end, official side: the official Driver/Replay/make_stream/agent.stream with its
train loop copied verbatim from embodied/run/train.py. Deterministic: argmax sampling, ALE seed 0
(the port always seeds, D-025), replay seed 0 (the default), no report stream, and Prefetch made
synchronous with the same idealization the port emulates (first batch at the first insertion,
refilled on every hand-over)."""
import os, sys, json, pathlib, hashlib, numpy as np
from functools import partial as bind
TREE = pathlib.Path('/n/holylabs/gershman_lab/Users/rtruong/dreamerv3-official-run')
sys.path.insert(0, str(TREE)); sys.path.insert(1, str(TREE / 'dreamerv3'))
import jax, jax.numpy as jnp
jax.config.update('jax_default_matmul_precision', 'highest')
import embodied, elements
import embodied.jax.outs as outs
outs.Categorical.sample = lambda self, seed, shape=(): jnp.argmax(self.logits, -1)
import embodied.envs.atari as atari
_init = atari.Atari.__init__
def _seeded(self, *a, **k):
  k['seed'] = 0 if k.get('seed') is None else k['seed']; _init(self, *a, **k)
atari.Atari.__init__ = _seeded
OUT = pathlib.Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True); N = int(sys.argv[2])
DTYPE = os.environ.get('DTYPE', 'float32')
batches = []
def fingerprint(d):
  r = {k: hashlib.sha1(np.ascontiguousarray(d[k]).tobytes()).hexdigest()[:16]
       for k in ('image', 'action', 'reward', 'is_first', 'is_last', 'is_terminal')}
  r['deter_slice'] = np.asarray(d['dyn/deter'], np.float32)[:, :, :32].tolist()
  r['stoch_argmax'] = np.asarray(d['dyn/stoch']).argmax(-1)[:, :, :8].tolist()
  return r
class SyncPrefetch:
  def __init__(self, source, transform=None, amount=1):
    self.source = iter(source) if hasattr(source, '__iter__') else source()
    self.transform = transform or (lambda x: x); self.slot = None
  def __iter__(self): return self
  def fill(self):
    if self.slot is None:
      raw = next(self.source); batches.append(fingerprint(raw)); self.slot = self.transform(raw)
  def __next__(self):
    self.fill(); d, self.slot = self.slot, None; self.fill(); return d
embodied.streams.Prefetch = SyncPrefetch
import ruamel.yaml as yaml
from dreamerv3 import main as M
configs = yaml.YAML(typ='safe').load((TREE / 'dreamerv3' / 'configs.yaml').read_text())
config = elements.Config(configs['defaults']).update(configs['atari100k']).update({
    'task': 'atari100k_frostbite', 'seed': 0, 'logdir': str(OUT / 'logdir'),
    'jax.platform': 'cuda', 'jax.compute_dtype': DTYPE, 'run.steps': N})
agent = M.make_agent(config); jax.config.update('jax_transfer_guard', 'allow')
np.savez(OUT / 'params0.npz', **{k: np.asarray(v) for k, v in agent.params.items() if not k.startswith('opt/')})
replay = M.make_replay(config, 'replay')
args = config.run; B, L = config.batch_size, config.batch_length
stream_train = iter(agent.stream(M.make_stream(config, replay, 'train')))
step = elements.Counter()
should_train = elements.when.Ratio(args.train_ratio / (B * L))
trans_log, mets_log, train_at = [], [], []
def record(tran, worker):
  trans_log.append([int(tran['action']), float(tran['reward']), bool(tran['is_first']), bool(tran['is_last'])])
def fill(tran, worker):
  if len(replay.sampler) > 0:
    stream_train.fill()
carry_train = [agent.init_train(B)]
def trainfn(tran, worker):  # verbatim from embodied/run/train.py
  if len(replay) < B * L:
    return
  for _ in range(should_train(step)):
    batch = next(stream_train)
    carry_train[0], outs_, mets = agent.train(carry_train[0], batch)
    train_at.append(int(step))
    if 'replay' in outs_:
      replay.update(outs_['replay'])
    if mets:
      mets_log.append({k: float(np.asarray(v)) for k, v in mets.items() if np.asarray(v).size == 1})
driver = embodied.Driver([bind(M.make_env, config, 0)], parallel=False)
driver.on_step(lambda tran, _: step.increment())
driver.on_step(replay.add)
driver.on_step(fill)
driver.on_step(record)
driver.on_step(trainfn)
policy = lambda *a: agent.policy(*a, mode='train')
driver.reset(agent.init_policy)
while step < N:
  driver(policy, steps=1)
if agent.pending_mets:
  m = agent._take_outs(agent.pending_mets)
  mets_log.append({k: float(np.asarray(v)) for k, v in m.items() if np.asarray(v).size == 1})
json.dump(dict(trans=trans_log, mets=mets_log, train_at=train_at, batches=batches), open(OUT / 'e2e.json', 'w'))
print('official: steps', int(step), 'trains', len(train_at), 'batches sampled', len(batches), 'mets', len(mets_log))
