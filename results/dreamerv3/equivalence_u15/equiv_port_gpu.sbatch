#!/bin/bash
#SBATCH --job-name=dv3_equiv_portgpu
#SBATCH -p kempner_h100
#SBATCH -A kempner_gershman_lab
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -t 0-00:40
#SBATCH -o /n/netscratch/gershman_lab/Everyone/truong/dv3-equiv/port_gpu_%j.out
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH=/n/netscratch/gershman_lab/Everyone/truong/dv3-equiv/port_src_8be1a32
export DEVICE=cuda
nvidia-smi --query-gpu=name --format=csv,noheader
/n/holylabs/gershman_lab/Users/rtruong/playtrain-trainers-dreamerv3/.venv/bin/python -u /n/netscratch/gershman_lab/Everyone/truong/dv3-equiv/equiv_port.py "$OFF_DIR"
