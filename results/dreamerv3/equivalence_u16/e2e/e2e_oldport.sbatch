#!/bin/bash
#SBATCH --job-name=dv3_e2e_old
#SBATCH -p kempner_h100
#SBATCH -A kempner_gershman_lab
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH -t 0-01:00
#SBATCH -o /n/netscratch/gershman_lab/Everyone/truong/dv3-equiv/e2e_old_%j.out
# Control: the port BEFORE D-030/031/032 (8be1a32) through the same harness, same official run.
E=/n/netscratch/gershman_lab/Everyone/truong/dv3-equiv
OFF_PARAMS=$E/e2e_off_float32/params0.npz DTYPE=float32 PYTHONPATH=$E/port_src_8be1a32 /n/holylabs/gershman_lab/Users/rtruong/playtrain-trainers-dreamerv3/.venv/bin/python -u $E/e2e_port.py $E/e2e_oldport_float32 1330 2>&1 | grep -E "^port|Error"
/n/holylabs/gershman_lab/Users/rtruong/dreamerv3-official-run/.venv/bin/python $E/e2e_compare.py $E/e2e_off_float32 $E/e2e_oldport_float32 | head -14
