#!/bin/bash -l
#SBATCH -J dv3_eval
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err
#SBATCH -p kempner_h100
#SBATCH -A kempner_gershman_lab
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -t 0-04:00
# U09: 100-episode sampled + greedy eval of one seed on the BBF fixed game-seed pool.
set -x
source /n/holylabs/gershman_lab/Users/rtruong/playtrain-trainers-dreamerv3/bin/cluster-env.sh
export DV3_TREE=/n/holylabs/gershman_lab/Users/rtruong/playtrain-trainers-dreamerv3-u16
export PYTHONPATH=$DV3_TREE/src:$DV3_PT/src
cd "$DV3_TREE"
SEED="${SLURM_ARRAY_TASK_ID:-0}"
"$DV3_PY" -u tools/dreamerv3/dv3_evaluate.py "outputs/dreamerv3/$RUN" --seeds "$SEED" --episodes "${EPISODES:-100}"
rc=$?; echo "EXIT=$rc"; exit $rc
