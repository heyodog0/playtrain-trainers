"""U16d end to end, port side: train.run unchanged, deterministic in the same way as the official
harness: argmax sampling, env seed 0 with no-ops from the running generator (seed_episode is D-005's
reproducibility deviation, off here), replay seed 0, fp32 or bf16 on the H100."""
import os, sys, json, pathlib, hashlib, numpy as np, torch
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
from playtrain_trainers.dreamerv3 import config as C, outs as O, replay as RP, envs as EN, train as T
from playtrain_trainers.dreamerv3.agent import Agent
O.Categorical.sample = lambda self, generator=None: torch.argmax(self.logits, -1)
EN.AtariDreamer.seed_episode = lambda self, episode_seed: None
OUT = pathlib.Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True); N = int(sys.argv[2])
batches, trans_log, mets_log = [], [], []
def fingerprint(d):
    r = {k: hashlib.sha1(np.ascontiguousarray(d[k]).tobytes()).hexdigest()[:16]
         for k in ('image', 'action', 'reward', 'is_first', 'is_last', 'is_terminal')}
    r['deter_slice'] = np.asarray(d['deter'], np.float32)[:, :, :32].tolist()
    r['stoch_argmax'] = np.asarray(d['stoch']).argmax(-1)[:, :, :8].tolist()
    return r
_sample, _add, _train = RP.Replay.sample, RP.Replay.add, Agent.train_step
def sample(self, batch, mode='train'):
    d = _sample(self, batch, mode); batches.append(fingerprint(d)); return d
def add(self, step, worker=0):
    trans_log.append([int(step['action']), float(step['reward']), bool(step['is_first']), bool(step['is_last'])])
    return _add(self, step, worker)
def train_step(self, carry, data):
    c, u, m = _train(self, carry, data)
    mets_log.append({k: float(v) for k, v in m.items() if np.ndim(v) == 0}); return c, u, m
RP.Replay.sample, RP.Replay.add, Agent.train_step = sample, add, train_step
# identical initial params: load the official run's params0 right after construction
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from equiv_port_names import official_name
_init = Agent.__init__
def init(self, *a, **k):
    _init(self, *a, **k)
    src = np.load(os.environ['OFF_PARAMS']); sd = self.state_dict(); new = {}
    for key, t in sd.items():
        n = official_name(key)
        if n is not None:
            assert tuple(src[n].shape) == tuple(t.shape), (key, n)
            new[key] = torch.as_tensor(src[n]).to(t.dtype)
    self.load_state_dict({**sd, **new}); print('port: loaded', len(new), 'official tensors')
Agent.__init__ = init
cfg = C.config_from_dict({"preset": "atari100k", "env_backend": "ale", "game": "frostbite",
                          "compute_dtype": os.environ.get("DTYPE", "float32"), "device": "cuda"})
rec = T.run(cfg, seed=0, outdir=OUT / 'run', steps=N)
# the ALE loader in the port honours ALE_ROM_PATH or the packaged ROM; both are md5 4ca73eb9 (U15a)
json.dump(dict(trans=trans_log, mets=mets_log, batches=batches), open(OUT / 'e2e.json', 'w'))
print('port: steps', rec['agent_steps'], 'trains', rec['grad_steps'], 'batches sampled', len(batches))
