#!/bin/bash -l
#SBATCH -J bbf_randbase
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err
#SBATCH -p shared
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 0-02:00
#
# The measured random floor for one PlayTrain game (D-008, D-044), 100
# episodes. CPU only -- a uniform policy needs no GPU, and `shared` does not
# spend the lab's H100 allocation.
#
#   sbatch --array=0-2 tools/bbf/bbf_random_baselines.sbatch venture amidar star_gunner

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
mkdir -p logs
source ./bin/cluster-env.sh

GAMES=("$@")
GAME=${GAMES[${SLURM_ARRAY_TASK_ID:-0}]}

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} game=$GAME"
"$BBF_PY" tools/bbf_random_baseline.py --game "$GAME"
rc=$?
echo "EXIT=$rc"
exit $rc
