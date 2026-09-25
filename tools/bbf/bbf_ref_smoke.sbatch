#!/bin/bash -l
#SBATCH -J bbf_refsmoke
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH -p kempner_h100
#SBATCH -A kempner_gershman_lab
#SBATCH -N 1
#SBATCH -c 23
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH -t 0-00:30
#
# U11 wiring smoke: 2000 agent steps of each reference trainer on the BBF
# wrapper stack, so a mis-wired env or checkpoint loader is caught before
# six real jobs burn six GPU-hours. NOT a result.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
mkdir -p logs
source ./bin/cluster-env.sh

# F-013's check: sb3 was installed --no-deps precisely so it could not move
# gymnasium out from under the ALE arm and these tests. Prove it here rather
# than assume it.
echo "================ test suite ================"
"$BBF_PY" -m pytest tests/test_bbf_config.py tests/test_bbf_envs.py \
  tests/test_bbf_evaluate.py tests/test_bbf_net.py tests/test_bbf_replay.py \
  tests/test_bbf_schedules.py tests/test_bbf_losses.py tests/test_bbf_resets.py \
  -q 2>&1 | tail -4

for T in impala ppo; do
  echo "================ $T ================"
  "$BBF_PY" tools/bbf/bbf_reference.py --trainer "$T" --seed 1 --steps 2000 \
      --run-id "ref_smoke_$T" --n-envs 4 2>&1 | tail -25
  echo "rc=$?"
done
