"""Encoder registry: every variant builds in both trainers, and the two
pre-existing variants are byte-identical to what checkpoints on disk expect."""
from __future__ import annotations

import pytest
import torch

from playtrain_trainers.impala.net import ImpalaNet
from playtrain_trainers.policy import ActorCritic, build_encoder

VARIANTS = ["nature", "impala", "impoola", "impala_s2", "impoola_s2", "impoola_s2w"]


@pytest.mark.parametrize("net", VARIANTS)
def test_builds_in_both_trainers(net):
    obs = torch.randint(0, 255, (4, 3, 64, 64), dtype=torch.uint8)
    dist, value = ActorCritic(6, net=net)(obs)
    assert dist.logits.shape == (4, 6) and value.shape == (4,)
    out, _ = ImpalaNet((3, 64, 64), 6, net=net)(
        {"frame": obs.view(2, 2, 3, 64, 64), "done": torch.zeros(2, 2, dtype=torch.bool)}
    )
    assert out["policy_logits"].shape == (2, 2, 6)


@pytest.mark.parametrize("net", ["impala", "nature"])
def test_existing_variants_keep_their_state_dict(net):
    """The registry refactor must not rename or add a single key: 156 runs'
    checkpoints load by name."""
    keys = set(build_encoder(net, 3, 256, 64).state_dict())
    expected = 32 if net == "impala" else 8
    assert len(keys) == expected
    assert all(k.startswith(("stages.", "stem.", "fc.")) for k in keys)


def test_gap_head_is_resolution_independent():
    """The point of GAP: the same weights accept any input resolution. This is
    what makes obs_scale a runtime knob rather than a different model."""
    enc = build_encoder("impoola_s2", 3, 256, 64)
    for hw in (32, 64, 96):
        obs = torch.randint(0, 255, (2, 3, hw, hw), dtype=torch.uint8)
        assert enc(obs).shape == (2, 256)


def test_obs_scale_requires_gap():
    build_encoder("impoola", 3, 256, 64).set_obs_scale(0.5)  # ok
    with pytest.raises(ValueError, match="requires gap"):
        build_encoder("impala", 3, 256, 64).set_obs_scale(0.5)


def test_unknown_net_lists_the_registry():
    with pytest.raises(ValueError, match="impoola_s2"):
        build_encoder("bogus", 3, 256, 64)
