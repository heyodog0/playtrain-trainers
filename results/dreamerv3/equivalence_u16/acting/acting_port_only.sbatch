#!/bin/bash
#SBATCH --job-name=dv3_acting_port
#SBATCH -p kempner_h100
#SBATCH -A kempner_gershman_lab
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -t 0-00:20
#SBATCH -o /n/netscratch/gershman_lab/Everyone/truong/dv3-equiv/acting_port_%j.out
E=/n/netscratch/gershman_lab/Everyone/truong/dv3-equiv
export INPUTS=$E/official_w0
PYTHONPATH=$E/port_src_bf16 /n/holylabs/gershman_lab/Users/rtruong/playtrain-trainers-dreamerv3/.venv/bin/python -u $E/acting_port.py $E/acting_$DTYPE
