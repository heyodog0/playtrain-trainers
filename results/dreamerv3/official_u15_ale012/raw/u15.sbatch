#!/bin/bash -l
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err
#SBATCH -N 1
#SBATCH -c 23
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH -t 0-08:00
# U15a: official e3f0224 on ale_py 0.12.1 (setInt bytes->str only). Partition/account on the sbatch line.
# Usage: sbatch -p P -A A --array=S --job-name=dv3_u14_<arm> u14.sbatch <arm> [extra main.py flags...]
set -x
TREE=/n/holylabs/gershman_lab/Users/rtruong/dreamerv3-official-ale012
cd "$TREE"
ARM="$1"; shift
SEED="${SLURM_ARRAY_TASK_ID:-0}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$TREE/.venv/bin/python" -c "import jax, ale_py; print('jax', jax.__version__, 'ale_py', ale_py.__version__, jax.devices())"
"$TREE/.venv/bin/python" -u dreamerv3/main.py \
  --configs atari100k \
  --task atari100k_frostbite \
  --seed "$SEED" \
  --logdir "$TREE/logdir/u15_${ARM}/frostbite_seed${SEED}" "$@"
rc=$?; echo "EXIT=$rc"; exit $rc
