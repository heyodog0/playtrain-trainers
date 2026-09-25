#!/bin/bash -l
#SBATCH -J bbf_rescore
#SBATCH -o logs/%x_%A.out
#SBATCH -e logs/%x_%A.err
#SBATCH -p kempner_h100
#SBATCH -A kempner_gershman_lab
#SBATCH -N 1
#SBATCH -c 23
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH -t 0-04:00
#
# Re-score a finished run's checkpoints. NOT a training run: it loads
# latest.pt and changes only which network is evaluated (D-040).
#
#   sbatch tools/bbf/bbf_rescore.sbatch <run_id> {online|target}
#
# All 5 seeds run in ONE task, not an array -- a 100-episode eval is minutes,
# so the queue wait dominates. 4h is generous cover for the slowest node.

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
mkdir -p logs
source ./bin/cluster-env.sh

RUN_ID=${1:?usage: sbatch bbf_rescore.sbatch RUN_ID NETWORK [EXTRA...]}
NETWORK=${2:?usage: sbatch bbf_rescore.sbatch RUN_ID NETWORK [EXTRA...]}
shift 2   # anything further is passed through, e.g. --max-noops 0

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} run_id=$RUN_ID network=$NETWORK extra=$*"
nvidia-smi --query-gpu=name --format=csv,noheader || true

"$BBF_PY" tools/bbf/bbf_rescore.py "$RUN_ID" --network "$NETWORK" "$@"
rc=$?
echo "EXIT=$rc"
exit $rc
