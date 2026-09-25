#!/bin/bash -l
#SBATCH -J dv3_old
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err
#SBATCH -p kempner_h100
#SBATCH -A kempner_gershman_lab
#SBATCH -N 1
#SBATCH -c 23
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH -t 0-08:00
# U13: the official code at the scores commit 2411f7d1, unmodified.
# Usage: sbatch --array=0-4 --job-name=dv3_old_A oldrun.sbatch atari_frostbite
set -x
TREE=/n/holylabs/gershman_lab/Users/rtruong/dreamerv3-official-2411f7d1
cd "$TREE"
TASK="$1"
SEED="${SLURM_ARRAY_TASK_ID:-0}"
# Same frostbite.bin the pinned run used (ale-py 0.9.0 bundles ROMs); the old
# atari.py honours ALE_ROM_PATH and loads <path>/<name>.bin.
export ALE_ROM_PATH=$(dirname $(find /n/holylabs/gershman_lab/Users/rtruong/dreamerv3-official-run/.venv -name frostbite.bin | head -1))
echo "ALE_ROM_PATH=$ALE_ROM_PATH"
nvidia-smi --query-gpu=name --format=csv,noheader
"$TREE/.venv/bin/python" -c "import jax; print('jax', jax.__version__, jax.devices())"
"$TREE/.venv/bin/python" -u dreamerv3/main.py \
  --configs atari100k \
  --task "$TASK" \
  --seed "$SEED" \
  --logdir "$TREE/logdir/${TASK}_seed${SEED}"
rc=$?; echo "EXIT=$rc"; exit $rc
