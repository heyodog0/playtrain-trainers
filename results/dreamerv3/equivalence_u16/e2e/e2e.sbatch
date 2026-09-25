#!/bin/bash
#SBATCH --job-name=dv3_e2e
#SBATCH -p kempner_h100
#SBATCH -A kempner_gershman_lab
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH -t 0-02:00
#SBATCH -o /n/netscratch/gershman_lab/Everyone/truong/dv3-equiv/e2e_%j.out
E=/n/netscratch/gershman_lab/Everyone/truong/dv3-equiv
N=${N:-1330}
export ALE_ROM_PATH=$(dirname $(find /n/holylabs/gershman_lab/Users/rtruong/dreamerv3-official-run/.venv -name frostbite.bin | head -1))
/n/holylabs/gershman_lab/Users/rtruong/dreamerv3-official-run/.venv/bin/python -u $E/e2e_official.py $E/e2e_off_$DTYPE $N 2>&1 | grep -E "^official|Error|error" | head
OFF_PARAMS=$E/e2e_off_$DTYPE/params0.npz PYTHONPATH=$E/port_src_d032 /n/holylabs/gershman_lab/Users/rtruong/playtrain-trainers-dreamerv3/.venv/bin/python -u $E/e2e_port.py $E/e2e_port_$DTYPE $N 2>&1 | grep -E "^port|Error|error" | head
/n/holylabs/gershman_lab/Users/rtruong/dreamerv3-official-run/.venv/bin/python $E/e2e_compare.py $E/e2e_off_$DTYPE $E/e2e_port_$DTYPE
