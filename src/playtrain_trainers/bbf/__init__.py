"""BBF (Schwarzer et al. 2023) on PlayTrain, as a PyTorch reimplementation.

The frozen protocol this package implements lives in
``playtrain-internal/bbf-loop/PROTOCOL.md``; every PlayTrain-specific choice is
numbered in ``playtrain-internal/bbf-loop/DEVIATIONS.md``.
"""
from __future__ import annotations

from playtrain_trainers.bbf.net import BBFNetwork, build_network
from playtrain_trainers.bbf.replay import SubsequenceReplayBuffer
from playtrain_trainers.bbf.envs import (
    make_ale_atari100k,
    make_env,
    make_playtrain_atari100k,
)
from playtrain_trainers.bbf.config import (
    BBFConfig,
    config_from_dict,
    load_config,
    results_dir,
    run_dir,
    seed_for,
    set_global_seeds,
)

__all__ = [
    "BBFConfig",
    "BBFNetwork",
    "SubsequenceReplayBuffer",
    "build_network",
    "make_ale_atari100k",
    "make_env",
    "make_playtrain_atari100k",
    "config_from_dict",
    "load_config",
    "results_dir",
    "run_dir",
    "seed_for",
    "set_global_seeds",
]
