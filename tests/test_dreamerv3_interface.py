"""U10 — the world-model seam, proved with a stub.

MISSION deliverable 2 asks for the world model to be one bounded module behind a
documented interface, with replay, actor-critic and the training loop depending only
on that interface. The proof is this file: a stub world model with tiny MLP dynamics
and NO DECODER AT ALL trains for 100 gradient steps through the real `train.py` code
path. If any of the agent, the actor-critic, the replay or the loop had reached past
the interface for something DreamerV3-specific -- tokens, the RSSM carry, the KL
terms, the reconstruction -- the stub could not run.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from playtrain_trainers.dreamerv3 import config as C
from playtrain_trainers.dreamerv3 import outs as O
from playtrain_trainers.dreamerv3 import train as T
from playtrain_trainers.dreamerv3 import world_model as WM
from playtrain_trainers.dreamerv3.agent import Agent


class StubWorldModel(nn.Module):
    """A deliberately un-DreamerV3 world model.

    Deterministic MLP dynamics over a flat latent, a mean-squared "prediction" loss
    against the next latent instead of a reconstruction, no decoder, no categorical
    latent, no KL. It exists to exercise the interface, not to work well.
    """

    def __init__(self, obs_shape: tuple[int, int, int], action_dim: int, latent: int = 32):
        super().__init__()
        self.feat_dim = latent
        self.entry_keys = ("latent",)
        self.action_dim = action_dim
        pixels = int(np.prod(obs_shape))
        self.encode = nn.Sequential(nn.Linear(pixels, 64), nn.SiLU(), nn.Linear(64, latent))
        self.step = nn.Sequential(nn.Linear(latent + action_dim, 64), nn.SiLU(), nn.Linear(64, latent))
        self.rewhead = nn.Linear(latent, 1)
        self.conhead = nn.Linear(latent, 1)

    # -- the seven members of the interface ----------------------------
    def initial(self, batch_size, device=None):
        return {"latent": torch.zeros(batch_size, self.feat_dim, device=device)}

    def feat2tensor(self, feat):
        return feat["latent"]

    def observe_step(self, carry, obs, prevact, is_first):
        keep = (~is_first).float().reshape(-1, 1)
        latent = carry["latent"] * keep
        flat = obs["image"].reshape(obs["image"].shape[0], -1).float() / 255
        latent = self.encode(flat) + self.step(torch.cat([latent, prevact * keep], -1))
        return {"latent": latent}, {"latent": latent}

    def imagine(self, carry, policy, length):
        feats, actions = [], []
        latent = carry["latent"]
        for t in range(length):
            action = policy({k: v.detach() for k, v in {"latent": latent}.items()}) if callable(policy) else policy[:, t]
            latent = self.step(torch.cat([latent, action], -1))
            feats.append(latent)
            actions.append(action)
        return {"latent": latent}, {"latent": torch.stack(feats, 1)}, torch.stack(actions, 1)

    def predict_reward(self, feat_tensor):
        return O.MSE(self.rewhead(feat_tensor).squeeze(-1))

    def predict_continue(self, feat_tensor):
        return O.Binary(self.conhead(feat_tensor).squeeze(-1))

    def loss(self, carry, obs, prevact, scales):
        B, T = obs["is_first"].shape
        latents = []
        cur = carry
        for t in range(T):
            cur, feat = self.observe_step(
                cur, {"image": obs["image"][:, t]}, prevact[:, t], obs["is_first"][:, t]
            )
            latents.append(feat["latent"])
        feat = {"latent": torch.stack(latents, 1)}
        inp = self.feat2tensor(feat)
        losses = {
            "rew": self.predict_reward(inp).loss(obs["reward"].float()),
            "con": self.predict_continue(inp).loss((~obs["is_terminal"]).float()),
            # Instead of a reconstruction: predict the NEXT latent. A different target
            # entirely, which is the point of the exercise.
            "pred": torch.cat(
                [
                    ((inp[:, 1:] - inp[:, :-1].detach()) ** 2).mean(-1),
                    torch.zeros(B, 1, device=inp.device),
                ],
                1,
            ),
        }
        return cur, {"latent": feat["latent"]}, feat, losses, {}

    def loss_keys(self):
        return ("rew", "con", "pred")


def make_stub_agent(cfg: C.DreamerConfig, action_dim: int = 6) -> Agent:
    stub = StubWorldModel((64, 64, 3), action_dim)
    return Agent((64, 64, 3), action_dim, cfg, world_model=stub)


# ----------------------------------------------------------------------
# The contract
# ----------------------------------------------------------------------
def test_the_stub_satisfies_the_protocol() -> None:
    stub = StubWorldModel((64, 64, 3), 6)
    assert isinstance(stub, WM.WorldModel)


def test_the_real_world_model_satisfies_the_protocol() -> None:
    cfg = C.debug_config()
    real = WM.RSSMWorldModel((64, 64, 3), 6, cfg)
    assert isinstance(real, WM.WorldModel)
    assert real.entry_keys == ("deter", "stoch")


def test_scales_follow_the_model_not_dreamerv3() -> None:
    """A world model that predicts something else declares different `loss_keys`,
    and the scale table follows it rather than assuming the five DreamerV3 terms."""
    cfg = C.debug_config()
    real_scales = WM.scales_for(cfg, WM.RSSMWorldModel((64, 64, 3), 6, cfg))
    assert set(real_scales) == {"rew", "con", "image", "dyn", "rep"}
    stub_scales = WM.scales_for(cfg, StubWorldModel((64, 64, 3), 6))
    assert set(stub_scales) == {"rew", "con", "pred"}
    assert "dyn" not in stub_scales  # no KL: the stub has no stochastic latent


def test_agent_builds_its_heads_from_feat_dim_alone() -> None:
    cfg = C.debug_config()
    agent = make_stub_agent(cfg)
    assert agent.wm.feat_dim == 32
    assert agent.pol.mlp.lins[0].insize == 32
    assert agent.val.mlp.lins[0].insize == 32


def test_agent_stores_the_entry_keys_the_model_declares() -> None:
    """Replay round-trips whatever `entry_keys` names -- `latent` here, not
    `deter`/`stoch`."""
    cfg = C.debug_config()
    agent = make_stub_agent(cfg)
    obs = {
        "image": torch.randint(0, 256, (1, 64, 64, 3), dtype=torch.uint8),
        "is_first": torch.ones(1, dtype=torch.bool),
    }
    _, _, entries = agent.policy(agent.init_policy(1), obs)
    assert set(entries) == {"latent"}


# ----------------------------------------------------------------------
# The proof: a stub trains through the real code path
# ----------------------------------------------------------------------
def test_stub_trains_100_gradient_steps_through_the_real_agent() -> None:
    """The MISSION deliverable: 100 gradient steps with a stub world model, through
    the real `Agent.train_step` -- imagination, actor-critic, replay context and all."""
    from playtrain_trainers.dreamerv3 import replay as RP

    cfg = C.debug_config()
    agent = make_stub_agent(cfg)
    rp = RP.Replay(length=cfg.sequence_length, capacity=5000, chunksize=128, online=True)
    for i in range(400):
        rp.add(
            {
                "image": np.full((64, 64, 3), i % 256, np.uint8),
                "reward": np.float32(i % 5),
                "is_first": np.bool_(i % 97 == 0),
                "is_last": np.bool_(i % 97 == 96),
                "is_terminal": np.bool_(i % 97 == 96),
                "action": np.int32(i % 6),
                "latent": np.zeros(agent.wm.feat_dim, np.float32),
            }
        )
    before = [p.detach().clone() for p in agent.opt_modules.parameters()]
    for _ in range(100):
        data = {k: torch.as_tensor(v) for k, v in rp.sample(cfg.batch_size).items()}
        _, updates, metrics = agent.train_step({}, data)
        rp.update({k: v.detach().cpu().numpy() for k, v in updates.items()})
    assert agent.opt.step_count == 100
    assert np.isfinite(metrics["loss"])
    assert set(updates) == {"stepid", "latent"}
    assert any(
        not torch.equal(a, b) for a, b in zip(before, agent.opt_modules.parameters())
    )
    # The actor-critic terms are present and finite even though the world model has
    # no decoder, no KL and a different prediction target.
    for key in ("loss/policy", "loss/value", "loss/repval", "loss/pred"):
        assert key in metrics and np.isfinite(metrics[key]), key
    assert "loss/image" not in metrics  # no reconstruction anywhere


def test_stub_runs_the_whole_training_loop(tmp_path: Path) -> None:
    """End to end through `train.py` itself: env, driver, replay, ratio accumulator,
    metrics. Nothing in the loop may assume an RSSM."""
    cfg = C.config_from_dict({"preset": "debug", "env_backend": "playtrain", "game": "frostbite"})
    try:
        record = T.run(
            cfg, seed=0, outdir=tmp_path, steps=150,
            make_world_model=lambda obs_shape, action_dim, c: StubWorldModel(obs_shape, action_dim),
        )
    except FileNotFoundError as exc:  # pragma: no cover - depends on the machine
        # Narrow on purpose: an earlier version caught bare Exception here and turned
        # a real device bug in the stub into a silent skip.
        pytest.skip(f"playtrain runtime unavailable: {exc}")
    assert record["agent_steps"] == 150
    assert record["grad_steps"] == record["grad_steps_expected"]
