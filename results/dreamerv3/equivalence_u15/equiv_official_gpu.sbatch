#!/bin/bash
#SBATCH --job-name=dv3_equiv_offgpu
#SBATCH -p kempner_h100
#SBATCH -A kempner_gershman_lab
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH -t 0-00:40
#SBATCH -o /n/netscratch/gershman_lab/Everyone/truong/dv3-equiv/official_gpu_%j.out
TREE=/n/holylabs/gershman_lab/Users/rtruong/dreamerv3-official-run
cd /n/netscratch/gershman_lab/Everyone/truong/dv3-equiv
nvidia-smi --query-gpu=name --format=csv,noheader
"$TREE/.venv/bin/python" -u equiv_official.py "$TREE" /n/netscratch/gershman_lab/Everyone/truong/dv3-equiv/${OUT_OVERRIDE:-official_gpu} cuda 2>&1 | grep -v -i warning
