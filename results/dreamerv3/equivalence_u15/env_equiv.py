"""U15: the official atari.py (ale012 tree, one-line setInt diff) and the port's
AtariDreamer, same seed and same action stream, compared byte for byte every step.
Both on ale_py 0.12.1 with the atari100k env settings."""
import importlib.util, sys, types
import numpy as np
sys.path.insert(0, '/n/holylabs/gershman_lab/Users/rtruong/dreamerv3-official-ale012')
from embodied.envs.atari import Atari as Official
# Load the port's envs.py without importing the torch-dependent package __init__.
PORT = '/n/netscratch/gershman_lab/Everyone/truong/dv3-equiv/port_src_8be1a32/playtrain_trainers/dreamerv3'
for name in ('playtrain_trainers', 'playtrain_trainers.dreamerv3'):
    sys.modules.setdefault(name, types.ModuleType(name))
def load(mod, path):
    spec = importlib.util.spec_from_file_location(mod, path); m = importlib.util.module_from_spec(spec)
    sys.modules[mod] = m; spec.loader.exec_module(m); return m
load('playtrain_trainers.dreamerv3.config', f'{PORT}/config.py')
E = load('playtrain_trainers.dreamerv3.envs', f'{PORT}/envs.py')
kw = dict(repeat=4, size=(64, 64), gray=False, noops=30, lives='unused', sticky=False,
          actions='needed', length=108000, pooling=2, aggregate='max', resize='pillow',
          autostart=False, clip_reward=False)
total_steps, total_eps, bad = 0, 0, []
for seed in range(5):
    a, b = Official('frostbite', seed=seed, **kw), E.AtariDreamer('frostbite', seed=seed, **kw)
    rng = np.random.default_rng(100 + seed)
    oa, ob = a.step({'action': 0, 'reset': True}), b.step({'action': 0, 'reset': True})
    for t in range(3000):
        for k in ('image', 'reward', 'is_first', 'is_last', 'is_terminal'):
            if not np.array_equal(np.asarray(oa[k]), np.asarray(ob[k])):
                bad.append((seed, t, k)); break
        if bad and bad[-1][0] == seed:
            break
        total_eps += int(bool(oa['is_last']))
        act = int(rng.integers(len(a.actionset)))
        reset = bool(oa['is_last'])
        oa, ob = a.step({'action': act, 'reset': reset}), b.step({'action': act, 'reset': reset})
        total_steps += 1
print('steps compared', total_steps, 'episodes', total_eps, 'mismatches', bad[:10])
print('IDENTICAL' if not bad else 'DIFFERS')
