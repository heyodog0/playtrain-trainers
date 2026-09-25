#!/bin/bash
#SBATCH --job-name=dv3_twohot_gpu
#SBATCH -p kempner_h100
#SBATCH -A kempner_gershman_lab
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -t 0-00:20
#SBATCH -o /n/netscratch/gershman_lab/Everyone/truong/dv3-equiv/twohot_gpu_%j.out
nvidia-smi --query-gpu=name --format=csv,noheader
/n/holylabs/gershman_lab/Users/rtruong/dreamerv3-official-run/.venv/bin/python -u /n/netscratch/gershman_lab/Everyone/truong/dv3-equiv/twohot_probe_gpu.py 2>&1 | grep -v -i warning
