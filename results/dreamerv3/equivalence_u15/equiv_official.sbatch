#!/bin/bash
#SBATCH --job-name=dv3_equiv_off
#SBATCH -p test
#SBATCH -c 16
#SBATCH --mem=96G
#SBATCH -t 0-02:00
#SBATCH -o /n/netscratch/gershman_lab/Everyone/truong/dv3-equiv/official_%j.out
set -x
TREE=/n/holylabs/gershman_lab/Users/rtruong/dreamerv3-official-run
OUT=/n/netscratch/gershman_lab/Everyone/truong/dv3-equiv/official
cd /n/netscratch/gershman_lab/Everyone/truong/dv3-equiv
"$TREE/.venv/bin/python" -u equiv_official.py "$TREE" "${OUT_OVERRIDE:-$OUT}"
rc=$?; echo "EXIT=$rc"; exit $rc
