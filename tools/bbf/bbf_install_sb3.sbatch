#!/bin/bash -l
#SBATCH -J bbf_sb3
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH -p shared

#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 0-00:25
#
# U11 needs `stable_baselines3` because train_ppo_clean imports SubprocVecEnv
# at module level (F-013). Run from a COMPUTE node: the login node was
# contended enough that uv hung with no output and exited 124.
#
# --no-deps is deliberate: sb3 pins gymnasium, and this venv is on 1.3.0,
# which the ALE arm (ale-py 0.12.1) and all 311 BBF tests depend on. sb3's
# own runtime needs are supplied explicitly instead.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
mkdir -p logs
source ./bin/cluster-env.sh
export UV_LINK_MODE=copy

echo "host=$(hostname)"
pgrep -u "$USER" -af "uv " || echo "no stray uv processes"

timeout 600 uv pip install --python .venv/bin/python --no-deps \
    stable-baselines3 cloudpickle pandas
echo "uv rc=$?"

"$BBF_PY" - <<'PY'
import gymnasium, numpy, torch
print("gymnasium", gymnasium.__version__, "| numpy", numpy.__version__, "| torch", torch.__version__)
try:
    import stable_baselines3
    from stable_baselines3.common.vec_env import SubprocVecEnv
    print("sb3", stable_baselines3.__version__, "SubprocVecEnv OK")
except Exception as e:
    print("sb3 FAILED:", type(e).__name__, e)
try:
    import playtrain_trainers.train_ppo_clean as ppo
    print("train_ppo_clean imports OK")
except Exception as e:
    print("train_ppo_clean FAILED:", type(e).__name__, e)
import ale_py
print("ale_py", ale_py.__version__)
PY
