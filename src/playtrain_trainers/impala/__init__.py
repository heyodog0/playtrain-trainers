"""IMPALA (Espeholt et al. 2018) — V-trace off-policy actor-critic.

This subpackage ports the deterministic pieces of FAIR's torchbeast
(https://github.com/facebookresearch/torchbeast) into this repo so we can
run IMPALA against the same MiniGrid / node-gym backends the PPO trainer
uses. The math (V-trace + loss functions) is bit-exact against torchbeast;
see tests/test_impala_vtrace.py and tests/test_impala_losses.py for the
parity checks.

The async actor-learner architecture is a separate concern from this
module (see train_impala.py — currently a stub).
"""
from playtrain_trainers.impala import buffers, environment, learn, losses, net, vtrace

__all__ = ["buffers", "environment", "learn", "losses", "net", "vtrace"]
