#!/bin/bash
#SBATCH --job-name=dv3_oldvenv
#SBATCH --partition=shared
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 0-01:30
#SBATCH -o /n/holylabs/gershman_lab/Users/rtruong/dreamerv3-official-2411f7d1/venv_%j.out
set -x
TREE=/n/holylabs/gershman_lab/Users/rtruong/dreamerv3-official-2411f7d1
export PATH=$HOME/.local/bin:$PATH
cd "$TREE"
# 2024-era pins for the April-2024 code: its requirements are unpinned
# (jax[cuda12_pip], tensorflow_probability, tensorflow-cpu), so pick versions
# contemporary with the commit. ale-py 0.8.1 for the old ROM API + bytes setInt.
uv venv --python 3.11 "$TREE/.venv"
uv pip install --python "$TREE/.venv/bin/python" \
  --index-strategy unsafe-best-match \
  --find-links https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
  "jax[cuda12]==0.4.26" "tensorflow-cpu==2.16.1" "tensorflow-probability==0.24.0" \
  "chex==0.1.86" "einops" "optax==0.2.2" "numpy<2" "cloudpickle" "colored" "gputil" \
  "msgpack" "psutil" "ruamel.yaml" "pyzmq" "ale-py==0.8.1" "pillow" "opencv-python-headless"
"$TREE/.venv/bin/python" -c "
import jax, ale_py, tensorflow_probability, optax
print('jax', jax.__version__, 'ale_py', ale_py.__version__)
"
rc=$?; echo "EXIT=$rc"; exit $rc
