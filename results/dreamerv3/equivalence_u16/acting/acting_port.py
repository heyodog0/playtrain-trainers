"""U16c port side: agent.policy over the official's recorded sequence, same params, argmax."""
import os, sys, pathlib, re, numpy as np, torch
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
from playtrain_trainers.dreamerv3 import config as C, outs as O
from playtrain_trainers.dreamerv3.agent import Agent
O.Categorical.sample = lambda self, generator=None: torch.argmax(self.logits, -1)
sys.path.insert(0, os.path.dirname(__file__))
from equiv_port_names import official_name
OFF = pathlib.Path(sys.argv[1]); INPUTS = pathlib.Path(os.environ['INPUTS']); DTYPE = os.environ.get('DTYPE', 'float32')
cfg = C.config_from_dict({"preset": "atari100k", "batch_size": 2, "batch_length": 8, "compute_dtype": DTYPE, "device": "cuda"})
agent = Agent((64, 64, 3), 18, cfg).to('cuda')
src = np.load(INPUTS / 'params0.npz'); sd = agent.state_dict(); new = {}
for k, t in sd.items():
    n = official_name(k)
    if n is not None:
        new[k] = torch.as_tensor(src[n]).to(t.dtype).to(t.device)
agent.load_state_dict({**sd, **new})
z = np.load(OFF / 'acting.npz')
carry = agent.init_policy(1); acts, deters, stochs = [], [], []
for t in range(len(z['acts'])):
    obs = {'image': torch.as_tensor(z['obs_image'][t][None]).cuda(), 'is_first': torch.as_tensor(z['obs_is_first'][t][None]).cuda()}
    carry, act, ent = agent.policy(carry, obs, mode='train')
    acts.append(int(act[0])); deters.append(ent['deter'][0].float().cpu().numpy()); stochs.append(ent['stoch'][0].float().cpu().numpy())
acts, deters, stochs = np.array(acts), np.stack(deters), np.stack(stochs)
print('port acts    ', acts.tolist())
print('official acts', z['acts'].tolist())
print('actions equal:', int((acts == z['acts']).sum()), '/', len(acts))
dd = np.abs(deters - z['deter']).max(-1) / (np.abs(z['deter']).max(-1) + 1e-12)
sd_ = (stochs.argmax(-1) != z['stoch'].argmax(-1)).mean((-1))
print('deter rel max err per step:', ' '.join(f'{x:.1e}' for x in dd))
print('stoch categorical mismatches per step (of 32):', ' '.join(str(int(round(x * 32))) for x in sd_))
np.savez(OFF / 'port_acting.npz', acts=acts, deter=deters, stoch=stochs)
