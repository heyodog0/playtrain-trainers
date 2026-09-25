#!/bin/bash
#SBATCH --job-name=dv3_ale012_venv
#SBATCH -p test
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 0-01:00
#SBATCH -o /n/holylabs/gershman_lab/Users/rtruong/dreamerv3-official-ale012/venv_%j.out
# U15a: official e3f0224 on ale-py 0.12.1 (the port's ALE). Only source edit: setInt key
# bytes -> str, which 0.12 requires (D-026). Everything else as the pinned venv.
set -x
SRC=/n/holylabs/gershman_lab/Users/rtruong/dreamerv3-official-run
TREE=/n/holylabs/gershman_lab/Users/rtruong/dreamerv3-official-ale012
export PATH=$HOME/.local/bin:$PATH
cd "$TREE"
rsync -a --exclude .venv --exclude logdir --exclude logs --exclude 'venv_*' --exclude '*.sbatch' "$SRC/" "$TREE/"
sed -i "s/self.ale.setInt(b'random_seed', /self.ale.setInt('random_seed', /" embodied/envs/atari.py
diff "$SRC/embodied/envs/atari.py" embodied/envs/atari.py
[ -x "$TREE/.venv/bin/python" ] || uv venv --python 3.11 "$TREE/.venv"
uv pip install --python "$TREE/.venv/bin/python" --index-strategy unsafe-best-match \
  "jax[cuda12]==0.4.33" "ale_py==0.12.1" \
  "elements>=3.19.1" "ninjax>=3.5.1" "portal>=3.5.0" "scope>=0.4.4" "granular>=0.20.3" \
  "optax" "einops" "chex" "jaxtyping" "colored_traceback" "numpy<2" "ruamel.yaml" \
  "opencv-python-headless" "tqdm" "av" "cloudpickle" "psutil" "rich"
"$TREE/.venv/bin/python" - <<'PY'
import sys, numpy as np, ale_py, hashlib
sys.path.insert(0, '.')
import ale_py.roms as roms
print('ale_py', ale_py.__version__, hashlib.md5(open(roms.get_rom_path('frostbite'), 'rb').read()).hexdigest())
from embodied.envs.atari import Atari
env = Atari('frostbite', size=(64, 64), gray=False, sticky=False, actions='needed', noops=30, resize='pillow')
rng = np.random.default_rng(0); tot = 0; eps = []; env.step({'action': 0, 'reset': True})
for i in range(6000):
  o = env.step({'action': int(rng.integers(env.act_space['action'].high)), 'reset': False}); tot += o['reward']
  if o['is_last']:
    eps.append(float(tot)); tot = 0; env.step({'action': 0, 'reset': True})
print('image', o['image'].shape, 'random episodes', eps)
PY
rc=$?; echo "EXIT=$rc"; exit $rc
