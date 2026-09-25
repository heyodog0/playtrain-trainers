#!/bin/bash -l
#SBATCH -J bbf_run
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err
#SBATCH -p kempner_h100
#SBATCH -A kempner_gershman_lab
#SBATCH -N 1
#SBATCH -c 23
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH -t 0-08:00
#
# One BBF seed per array task. BBF at RR=2 is one env and one GPU, so seeds
# are independent array tasks rather than anything distributed.
#
#   sbatch --array=0-4 tools/bbf/bbf_run.sbatch configs/bbf/<cfg>.json <run_id>
#
# The array index picks the seed through `seed_for(base, index)` so the seeds
# are spread rather than 0..4 (see bbf/config.py on why consecutive small
# integers are a poor seed family). 8h is ~4x the 1.83h/seed the U08 smoke
# measured, which leaves room for a slow node.
#
# Uses .venv/bin/python directly, NOT `uv run`: a sync would resolve playtrain
# from git and try to compile the native host (D-025).

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
mkdir -p logs
source ./bin/cluster-env.sh

CFG=${1:?usage: sbatch --array=0-4 bbf_run.sbatch CONFIG [RUN_ID]}
RUN_ID=${2:-}
IDX=${SLURM_ARRAY_TASK_ID:-0}

SEED=$("$BBF_PY" -c "from playtrain_trainers.bbf.config import seed_for; print(seed_for(0, $IDX))")

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} array_task=$IDX seed=$SEED cfg=$CFG"
nvidia-smi --query-gpu=name --format=csv,noheader || true

ARGS=(--config "$CFG" --seed "$SEED")
[ -n "$RUN_ID" ] && ARGS+=(--run-id "$RUN_ID")
"$BBF_PY" -m playtrain_trainers.bbf.train "${ARGS[@]}"
# Propagate the real status: `echo "EXIT=$?"` made the SCRIPT exit 0,
# so sacct reported COMPLETED 0:0 for runs that never started.
rc=$?
echo "EXIT=$rc"
exit $rc
