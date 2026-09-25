#!/bin/bash
#SBATCH --job-name=dv3_acting_seqA
#SBATCH -p kempner_h100
#SBATCH -A kempner_gershman_lab
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH -t 0-00:40
#SBATCH -o /n/netscratch/gershman_lab/Everyone/truong/dv3-equiv/acting_seqA_%j.out
E=/n/netscratch/gershman_lab/Everyone/truong/dv3-equiv
export INPUTS=$E/official_w0 SEQ=$E/acting_float32/acting.npz DTYPE=bfloat16
/n/holylabs/gershman_lab/Users/rtruong/dreamerv3-official-run/.venv/bin/python -u $E/acting_official.py $E/acting_seqA_bf16 2>&1 | grep -E "official acts|Error"
PYTHONPATH=$E/port_src_bf16 /n/holylabs/gershman_lab/Users/rtruong/playtrain-trainers-dreamerv3/.venv/bin/python -u $E/acting_port.py $E/acting_seqA_bf16 | grep "actions equal"
/n/holylabs/gershman_lab/Users/rtruong/dreamerv3-official-run/.venv/bin/python $E/acting_scale.py $E
