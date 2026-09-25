#!/bin/bash -l
#SBATCH -J bbf_prof
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH -p kempner_h100
#SBATCH -A kempner_gershman_lab
#SBATCH -N 1
#SBATCH -c 23
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH -t 0-00:30
#
# Prices each component of one BBF optimizer step. See tools/bbf/bbf_profile.py.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
mkdir -p logs
source ./bin/cluster-env.sh
"$BBF_PY" tools/bbf/bbf_profile.py
