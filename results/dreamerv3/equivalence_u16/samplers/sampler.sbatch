#!/bin/bash
#SBATCH --job-name=dv3_sampler
#SBATCH -p kempner_h100
#SBATCH -A kempner_gershman_lab
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -t 0-00:30
#SBATCH -o /n/netscratch/gershman_lab/Everyone/truong/dv3-equiv/sampler_%j.out
E=/n/netscratch/gershman_lab/Everyone/truong/dv3-equiv
/n/holylabs/gershman_lab/Users/rtruong/dreamerv3-official-run/.venv/bin/python -u $E/sampler_official.py $E/sampler_official.npz 2>&1 | grep -v -i warn
nvidia-smi --query-gpu=name --format=csv,noheader
PYTHONPATH=$E/port_src_bf16 /n/holylabs/gershman_lab/Users/rtruong/playtrain-trainers-dreamerv3/.venv/bin/python -u $E/sampler_port.py $E/sampler_official.npz $E/sampler_port.npz
/n/holylabs/gershman_lab/Users/rtruong/dreamerv3-official-run/.venv/bin/python -u $E/sampler_stats.py $E/sampler_official.npz $E/sampler_port.npz
