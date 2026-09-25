#!/bin/bash
#SBATCH --job-name=dv3_equiv_port
#SBATCH -p test
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 0-02:00
#SBATCH -o /n/netscratch/gershman_lab/Everyone/truong/dv3-equiv/port_%j.out
set -x
TREE=/n/holylabs/gershman_lab/Users/rtruong/playtrain-trainers-dreamerv3
# Current port source (commit 8be1a32) on PYTHONPATH, ahead of the tree's own install;
# the tree itself is left as the provenance of the v3 seeds.
export PYTHONPATH=/n/netscratch/gershman_lab/Everyone/truong/dv3-equiv/port_src_8be1a32
"$TREE/.venv/bin/python" -c "import playtrain_trainers.dreamerv3.world_model as w; print(w.__file__)"
"$TREE/.venv/bin/python" -u /n/netscratch/gershman_lab/Everyone/truong/dv3-equiv/equiv_port.py ${OFF_DIR:-/n/netscratch/gershman_lab/Everyone/truong/dv3-equiv/official}
rc=$?; echo "EXIT=$rc"; exit $rc
