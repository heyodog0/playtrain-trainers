#!/bin/bash -l
#SBATCH -J bbf_ref
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err
#SBATCH -p kempner_h100
#SBATCH -A kempner_gershman_lab
#SBATCH -N 1
#SBATCH -c 23
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH -t 0-02:00
#
# U11: one reference seed per array task.
#   sbatch --array=0-2 tools/bbf/bbf_ref_run.sbatch impala
#   sbatch --array=0-2 tools/bbf/bbf_ref_run.sbatch ppo
#   sbatch --array=0-2 tools/bbf/bbf_ref_run.sbatch impala greedy venture
#
# The third argument is the PlayTrain game (default frostbite, which keeps the
# U11 run ids unchanged).
#
# A second argument of `sample` re-scores the ALREADY-TRAINED checkpoint by
# sampling from the policy instead of taking its argmax, into a _sampled
# run_id. See F-014: both arms' greedy policies collapsed to one constant
# action, which is not a fair reading of a near-uniform stochastic policy.
#
# Seeds come from the same `seed_for(0, i)` family as the BBF arms, so the
# reference seeds are the first three of BBF's five rather than an unrelated
# set. 2h is generous: the smoke measured PPO at ~857 sps and IMPALA at 2000
# steps in 7.7 s, so training is minutes; the 100-episode eval dominates.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
mkdir -p logs
source ./bin/cluster-env.sh

# No braces in this message: a `}` inside ${1:?...} closes the expansion
# early, which silently made TRAINER="impala}" and every task exit 2.
TRAINER=${1:?usage: sbatch --array=0-2 bbf_ref_run.sbatch impala OR ppo}
IDX=${SLURM_ARRAY_TASK_ID:-0}
SEED=$("$BBF_PY" -c "from playtrain_trainers.bbf.config import seed_for; print(seed_for(0, $IDX))")

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} task=$IDX trainer=$TRAINER seed=$SEED args=$*"
POLICY=${2:-greedy}
GAME=${3:-frostbite}
EXTRA=""
# Read the ORIGINAL run's checkpoint, write to a new run_id.
[ "$POLICY" = "sample" ] && EXTRA="--skip-train \
    --run-id ref_${TRAINER}_${GAME}_100k_sampled \
    --ckpt-run-id ref_${TRAINER}_${GAME}_100k"

"$BBF_PY" tools/bbf/bbf_reference.py --trainer "$TRAINER" --seed "$SEED" \
    --game "$GAME" --steps 100000 --policy "$POLICY" $EXTRA
# Propagate the real status: `echo "EXIT=$?"` made the SCRIPT exit 0,
# so sacct reported COMPLETED 0:0 for runs that never started.
rc=$?
echo "EXIT=$rc"
exit $rc
