#!/bin/bash -l
#SBATCH -J bbf_smoke_many
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err
#SBATCH -p kempner_h100
#SBATCH -A kempner_gershman_lab
#SBATCH -N 1
#SBATCH -c 23
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH -t 0-01:00
#
# One 2k-step smoke per config, as an array. The gate before spending ~55
# GPU-h on the phase 3 arms: it exercises warmup, prioritized sampling, the
# C51 + SPR update, the anneals, a reset, curve evals and the final eval, on
# each new game and backend.
#
#   sbatch --array=0-5 tools/bbf/bbf_smoke_many.sbatch cfgA.json cfgB.json ...

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
mkdir -p logs
source ./bin/cluster-env.sh

CFGS=("$@")
CFG=${CFGS[${SLURM_ARRAY_TASK_ID:-0}]}

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} config=$CFG"
nvidia-smi --query-gpu=name --format=csv,noheader || true
"$BBF_PY" -m playtrain_trainers.bbf.train --config "$CFG"
rc=$?
echo "EXIT=$rc"
exit $rc
