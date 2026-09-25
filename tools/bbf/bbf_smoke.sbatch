#!/bin/bash -l
#SBATCH -J bbf_smoke
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH -p kempner_h100
#SBATCH -A kempner_gershman_lab
#SBATCH -N 1
#SBATCH -c 23
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH -t 0-01:00
#
# U08 smoke test: 2000 env steps of BBF end to end on PlayTrain frostbite,
# on one H100. Not a protocol run -- configs/bbf/smoke.json deliberately
# shrinks min_replay_history, reset_every and the eval sizes so the whole
# path (warmup, prioritized sampling, C51 + SPR update, anneals, a reset,
# curve evals, checkpointing, final eval) executes inside 2k steps.
#
# 23c/1gpu is exactly the partition's per-GPU cap.
#
# Uses .venv/bin/python directly, NOT `uv run`: a sync against this tree
# would resolve playtrain from git and try to compile the native host
# (D-025), and `uv sync` in a live tree has killed running jobs before.

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
mkdir -p logs

source ./bin/cluster-env.sh

echo "host=$(hostname) job=${SLURM_JOB_ID:-none}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
"$BBF_PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

CFG=${1:-configs/bbf/smoke.json}
echo "config=$CFG"
"$BBF_PY" -m playtrain_trainers.bbf.train --config "$CFG"
# Propagate the real status: `echo "EXIT=$?"` made the SCRIPT exit 0,
# so sacct reported COMPLETED 0:0 for runs that never started.
rc=$?
echo "EXIT=$rc"
exit $rc
