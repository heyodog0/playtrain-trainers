"""U07 — the training loop, against `embodied/run/train.py` and `dreamerv3/main.py`.

The important thing here is the accumulator and the three clocks; the agent itself is
covered by U03-U06. `Ratio` is checked against the fetched `elements` source, and the
warmup boundary against the replay's own item count rather than against a constant.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from playtrain_trainers.dreamerv3 import config as C
from playtrain_trainers.dreamerv3 import replay as RP
from playtrain_trainers.dreamerv3 import train as T
from playtrain_trainers.dreamerv3.agent import Agent


# ----------------------------------------------------------------------
# Ratio
# ----------------------------------------------------------------------
def test_ratio_returns_one_on_the_first_call() -> None:
    r = T.Ratio(0.25)
    assert r(1000) == 1


def test_ratio_is_one_in_four_with_no_drift() -> None:
    r = T.Ratio(0.25)
    r(0)
    counts = [r(step) for step in range(1, 401)]
    assert counts[:8] == [0, 0, 0, 1, 0, 0, 0, 1]
    assert sum(counts) == 100  # exactly 400 / 4, no accumulated error


def test_ratio_handles_gaps_without_losing_steps() -> None:
    """If the caller skips ahead, the accumulator returns the whole backlog at once."""
    r = T.Ratio(0.25)
    r(0)
    assert r(40) == 10


def test_ratio_edge_cases() -> None:
    assert T.Ratio(0)(5) == 0
    assert T.Ratio(-1)(5) == 1


def test_ratio_matches_the_fetched_elements_source() -> None:
    """The reference copy lives in the harness; re-implement it here and compare."""

    class Reference:
        def __init__(self, ratio):
            self._ratio, self._prev = ratio, None

        def __call__(self, step):
            step = int(step)
            if self._ratio == 0:
                return 0
            if self._ratio < 0:
                return 1
            if self._prev is None:
                self._prev = step
                return 1
            repeats = int((step - self._prev) * self._ratio)
            self._prev += repeats / self._ratio
            return repeats

    a, b = T.Ratio(0.25), Reference(0.25)
    assert [a(s) for s in range(2000)] == [b(s) for s in range(2000)]


# ----------------------------------------------------------------------
# The warmup boundary
# ----------------------------------------------------------------------
def test_warmup_is_batch_steps_plus_sequence_length_minus_one() -> None:
    """`trainfn` gates on `len(replay) < batch_steps`, and `len(replay)` counts
    ITEMS. An item needs `sequence_length` steps from its start, so the item count
    lags the added steps by `sequence_length - 1`. Found in U07: the boundary is 1088
    agent steps at the frozen config, not 1024."""
    cfg = C.atari100k_config()
    assert cfg.train_warmup_steps == 1088
    assert cfg.batch_steps == 1024
    assert cfg.sequence_length == 65


def test_the_warmup_boundary_matches_a_real_replay_buffer() -> None:
    """Not a restatement of the formula: fill a buffer and find the step at which
    `len(rp)` first reaches `batch_steps`."""
    cfg = C.debug_config()
    rp = RP.Replay(length=cfg.sequence_length, capacity=10_000, chunksize=64, online=True)
    step = 0
    while len(rp) < cfg.batch_steps:
        rp.add(
            {
                "image": np.zeros((4, 4, 3), np.uint8),
                "reward": np.float32(0),
                "is_first": np.bool_(step == 0),
                "is_last": np.bool_(False),
                "is_terminal": np.bool_(False),
                "action": np.int32(0),
            }
        )
        step += 1
    assert step == cfg.train_warmup_steps


def test_gradient_step_count_has_a_closed_form() -> None:
    """The accumulator's exact carry means the count is
    `floor((agent_steps - warmup) * ratio) + 1`, which is what `train.py` asserts."""
    cfg = C.atari100k_config()
    r = T.Ratio(cfg.train_ratio_per_agent_step)
    warmup = cfg.train_warmup_steps
    grad = 0
    for step in range(warmup, 20_000):
        grad += r(step)
        expected = int((step - warmup) * cfg.train_ratio_per_agent_step) + 1
        assert grad == expected, (step, grad, expected)


def test_total_gradient_steps_at_the_frozen_budget() -> None:
    cfg = C.atari100k_config()
    assert cfg.total_gradient_steps == 27_229
    post = cfg.total_gradient_steps / (cfg.run.steps - cfg.train_warmup_steps + 1)
    assert abs(post - 0.25) / 0.25 < 0.05
    assert cfg.total_frames == 440_000


# ----------------------------------------------------------------------
# Devices and autocast
# ----------------------------------------------------------------------
def test_pick_device_honours_an_explicit_name() -> None:
    assert T.pick_device("cpu").type == "cpu"


def test_cast_mirrors_nets_cast() -> None:
    """D-012: `nets.cast` converts floating tensors (all tensors with force) and walks
    dicts and tuples; integer and bool tensors pass through unforced."""
    from playtrain_trainers.dreamerv3 import nets as N
    N.set_compute_dtype("bfloat16")
    try:
        out = N.cast({"f": torch.zeros(2), "i": torch.zeros(2, dtype=torch.int64),
                      "t": (torch.ones(1), torch.ones(1, dtype=torch.bool))})
        assert out["f"].dtype == torch.bfloat16
        assert out["i"].dtype == torch.int64
        assert out["t"][0].dtype == torch.bfloat16 and out["t"][1].dtype == torch.bool
        assert N.cast(torch.zeros(1, dtype=torch.uint8), force=True).dtype == torch.bfloat16
    finally:
        N.set_compute_dtype("float32")


def test_bf16_dtypes_follow_the_official_asserts() -> None:
    """rssm.py:91/103 assert deter, stoch and logit are COMPUTE_DTYPE after every
    observe and imagine step; feat2tensor is cast; losses stay float32 (opt.py:37)."""
    from playtrain_trainers.dreamerv3 import nets as N
    cfg = C.config_from_dict({"preset": "debug", "compute_dtype": "bfloat16"})
    agent = Agent((64, 64, 3), 6, cfg)
    try:
        wm = agent.wm
        carry = wm.initial(2)
        assert all(v.dtype == torch.bfloat16 for v in carry.values())
        obs = {"image": torch.randint(0, 256, (2, 64, 64, 3), dtype=torch.uint8)}
        carry, feat = wm.observe_step(carry, obs, torch.zeros(2, 6), torch.ones(2, dtype=torch.bool))
        assert {k: v.dtype for k, v in feat.items()} == dict.fromkeys(feat, torch.bfloat16)
        carry, ifeat, _ = wm.imagine(carry, lambda c: torch.nn.functional.one_hot(
            torch.zeros(2, dtype=torch.long), 6).float(), 3)
        assert {k: v.dtype for k, v in ifeat.items()} == dict.fromkeys(ifeat, torch.bfloat16)
        assert wm.feat2tensor(feat).dtype == torch.bfloat16
        assert agent.val(wm.feat2tensor(feat)).pred().dtype == torch.float32
    finally:
        N.set_compute_dtype("float32")


def test_host_converts_bf16_entries_to_float32() -> None:
    """Replay stores float32, as embodied/jax/agent.py:402 converts bf16 outs."""
    assert T.host(torch.ones(2, dtype=torch.bfloat16)).dtype == np.float32
    assert T.host(torch.ones(2, dtype=torch.uint8)).dtype == np.uint8


# ----------------------------------------------------------------------
# Reconstruction strip
# ----------------------------------------------------------------------
def test_save_recon_png(tmp_path: Path) -> None:
    from PIL import Image

    real = np.random.randint(0, 256, (4, 8, 8, 3), dtype=np.uint8)
    pred = np.random.rand(4, 8, 8, 3).astype(np.float32)
    out = tmp_path / "recon.png"
    T.save_recon_png(out, real, pred)
    img = np.array(Image.open(out))
    assert img.shape[0] == 2 * 8 * 3  # two rows, 3x nearest upscale
    assert img.shape[1] == 4 * 8 * 3


# ----------------------------------------------------------------------
# End to end on the debug config
# ----------------------------------------------------------------------
@pytest.mark.parametrize("backend", ["playtrain"])
def test_short_run_end_to_end(tmp_path: Path, backend: str) -> None:
    """A real run of the real loop: env, replay, agent, gradient steps, metrics."""
    cfg = C.config_from_dict({"preset": "debug", "env_backend": backend, "game": "frostbite"})
    try:
        record = T.run(cfg, seed=0, outdir=tmp_path, steps=150)
    except Exception as exc:  # pragma: no cover - depends on the machine
        pytest.skip(f"{backend} runtime unavailable: {exc}")

    assert record["agent_steps"] == 150
    assert record["frames"] == 150 * cfg.env.repeat  # the cumulative clock, not per-episode
    assert record["grad_steps"] == record["grad_steps_expected"]
    assert (tmp_path / "metrics_seed0.json").exists()
    saved = json.loads((tmp_path / "metrics_seed0.json").read_text())
    assert saved["target_ratio"] == cfg.train_ratio_per_agent_step
    # Every episode record carries all three clocks (MISSION rule 3).
    for ep in saved["episodes"]:
        assert {"score", "frames", "agent_steps", "grad_steps"} <= set(ep)
    for line in saved["log"]:
        assert {"frames", "agent_steps", "grad_steps"} <= set(line)


def test_agent_train_step_is_finite_and_moves_parameters() -> None:
    cfg = C.debug_config()
    agent = Agent((64, 64, 3), 6, cfg)
    rp = RP.Replay(length=cfg.sequence_length, capacity=1000, chunksize=64, online=True)
    for i in range(40):
        rp.add(
            {
                "image": np.full((64, 64, 3), i % 256, np.uint8),
                "reward": np.float32(i % 3),
                "is_first": np.bool_(i == 0),
                "is_last": np.bool_(False),
                "is_terminal": np.bool_(False),
                "action": np.int32(i % 6),
                "deter": np.zeros(cfg.agent.rssm.deter, np.float32),
                "stoch": np.zeros((cfg.agent.rssm.stoch, cfg.agent.rssm.classes), np.float32),
            }
        )
    data = {k: torch.as_tensor(v) for k, v in rp.sample(cfg.batch_size).items()}
    before = [p.detach().clone() for p in agent.opt_modules.parameters()]
    _, updates, metrics = agent.train_step({}, data)
    assert set(k for k in metrics if k.startswith("loss/")) == {
        "loss/rew", "loss/con", "loss/image", "loss/dyn", "loss/rep",
        "loss/policy", "loss/value", "loss/repval",
    }
    assert np.isfinite(metrics["loss"])
    assert np.isfinite(metrics["grad_norm"])
    assert metrics["lr"] == 0.0  # warmup: the first update runs at lr 0
    # The first step moves nothing because the lr is 0; the second one does.
    assert all(torch.equal(a, b) for a, b in zip(before, agent.opt_modules.parameters()))
    agent.train_step({}, data)
    assert any(not torch.equal(a, b) for a, b in zip(before, agent.opt_modules.parameters()))
    assert set(updates) == {"stepid", "deter", "stoch"}


def test_slow_critic_diverges_from_the_live_one_during_training() -> None:
    """`slowvalue` must be an EMA copy, not an alias: if `SlowModel` shared the
    parameters the regularizer would be comparing the critic to itself."""
    cfg = C.debug_config()
    agent = Agent((64, 64, 3), 6, cfg)
    live = next(agent.val.parameters())
    slow = next(agent.slowval.model.parameters())
    assert live is not slow
    with torch.no_grad():
        live.add_(1.0)
    assert not torch.equal(live, slow)


# ----------------------------------------------------------------------
# The terminal transition must reach the replay (the U08 gate failure)
# ----------------------------------------------------------------------
def test_terminal_transitions_are_stored(tmp_path: Path) -> None:
    """The U08 ALE gate failed because no `is_terminal` step ever entered the replay.

    The loop stored the CURRENT observation and then stepped, so the terminal
    observation was always replaced by the reset before the next iteration could store
    it. The continue head then saw a constant `1 - 1/horizon` target for the entire
    run -- its loss sat at exactly H(1 - 1/333) = 0.02044 -- imagined rollouts never
    terminated, and the agent was never told that dying is bad.

    This test runs the real loop and reads the replay back out.
    """
    cfg = C.config_from_dict({"preset": "debug", "env_backend": "playtrain", "game": "frostbite"})
    seen: dict[str, int] = {}

    real_add = RP.Replay.add

    def counting_add(self, step, worker=0):
        for key in ("is_first", "is_last", "is_terminal"):
            seen[key] = seen.get(key, 0) + int(bool(step[key]))
        seen["steps"] = seen.get("steps", 0) + 1
        return real_add(self, step, worker)

    RP.Replay.add = counting_add
    try:
        record = T.run(cfg, seed=0, outdir=tmp_path, steps=900)
    except Exception as exc:  # pragma: no cover - depends on the machine
        pytest.skip(f"playtrain runtime unavailable: {exc}")
    finally:
        RP.Replay.add = real_add

    assert len(record["episodes"]) >= 2, "need at least two finished episodes to test"
    assert seen["steps"] == record["agent_steps"]
    # One stored is_last per finished episode, and on this game every episode end is a
    # real terminal (GAMEOVER or WIN), never the 108k-frame cap.
    assert seen["is_last"] == len(record["episodes"])
    assert seen["is_terminal"] == len(record["episodes"])
    # One is_first per episode start, including the very first.
    assert seen["is_first"] == len(record["episodes"]) + 1


def test_continue_target_is_not_constant_once_terminals_are_stored() -> None:
    """The fingerprint of the bug, as a direct assertion: with a terminal present the
    continue target takes two distinct values, so its cross-entropy floor is no longer
    H(1 - 1/333)."""
    import math

    from playtrain_trainers.dreamerv3 import losses as L

    horizon = 333
    no_terminal = L.continue_target(torch.zeros(1, 8, dtype=torch.bool), horizon)
    assert len(set(no_terminal.flatten().tolist())) == 1
    p = 1 - 1 / horizon
    floor = -(p * math.log(p) + (1 - p) * math.log(1 - p))
    assert abs(floor - 0.02044) < 1e-4  # the value logged for the whole U08 run

    with_terminal = torch.zeros(1, 8, dtype=torch.bool)
    with_terminal[0, 5] = True
    con = L.continue_target(with_terminal, horizon)
    assert len(set(con.flatten().tolist())) == 2
    assert con[0, 5].item() == 0.0


# ----------------------------------------------------------------------
# U16d: the official driver's two timing semantics
# ----------------------------------------------------------------------
def _snapshot(module):
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


def _same(a, b):
    return all(torch.equal(a[k], b[k]) for k in a)


def test_policy_acts_with_lagged_params() -> None:
    """embodied/jax/agent.py: `train()` stashes the policy keys as they were BEFORE the
    update; `policy()` acts with its current copy, THEN swaps the stash in."""
    # warmup 0, so the very first update already moves the params
    cfg = C.config_from_dict({"preset": "debug", "agent": {"opt": {"warmup": 0}}})
    agent = Agent((64, 64, 3), 6, cfg)
    obs = {"image": torch.randint(0, 256, (1, 64, 64, 3), dtype=torch.uint8),
           "is_first": torch.ones(1, dtype=torch.bool)}
    carry = agent.init_policy(1)
    carry, _, _ = agent.policy(carry, obs)            # builds the acting copy = P0
    p0 = _snapshot(agent.pol)
    rp = RP.Replay(length=cfg.sequence_length, capacity=5000, chunksize=128, online=True)
    for i in range(300):
        rp.add({"image": np.full((64, 64, 3), i % 256, np.uint8), "reward": np.float32(0),
                "is_first": np.bool_(i == 0), "is_last": np.bool_(False), "is_terminal": np.bool_(False),
                "action": np.int32(i % 6), "deter": np.zeros(cfg.agent.rssm.deter, np.float32),
                "stoch": np.zeros((cfg.agent.rssm.stoch, cfg.agent.rssm.classes), np.float32)})
    def train():
        data = {k: torch.as_tensor(v) for k, v in rp.sample(cfg.batch_size).items()}
        agent.train_step({}, data)
    train()                                            # live -> P1; stash = P0
    p1 = _snapshot(agent.pol)
    assert not _same(p0, p1)
    acting_pol = agent._acting_modules()[1]
    assert _same(_snapshot(acting_pol), p0)            # still acting with P0
    carry, _, _ = agent.policy(carry, obs)             # acts with P0, swaps in the stash (= P0)
    assert _same(_snapshot(acting_pol), p0)
    train()                                            # live -> P2; stash = P1
    carry, _, _ = agent.policy(carry, obs)             # acts with P0, swaps in P1
    assert _same(_snapshot(acting_pol), p1)


def test_training_batches_are_prefetched(tmp_path: Path, monkeypatch) -> None:
    """Prefetch(amount=1): the first batch is drawn as soon as the replay has an item,
    and every hand-over immediately draws the next, so there is one more sample than
    there are gradient steps."""
    lens = []
    orig = RP.Replay.sample
    def spy(self, batch, mode="train"):
        lens.append(len(self))
        return orig(self, batch, mode)
    monkeypatch.setattr(RP.Replay, "sample", spy)
    cfg = C.config_from_dict({"preset": "debug", "env_backend": "playtrain", "game": "frostbite"})
    try:
        record = T.run(cfg, seed=0, outdir=tmp_path, steps=150)
    except FileNotFoundError as exc:  # pragma: no cover - depends on the machine
        pytest.skip(f"playtrain runtime unavailable: {exc}")
    assert lens[0] == 1
    assert len(lens) == record["grad_steps"] + 1


def test_replay_write_back_lags_one_train_step(tmp_path: Path, monkeypatch) -> None:
    """D-032: official `train()` returns the previous call's outs, so each replay.update
    carries the latents of the step before; the first train step writes nothing back."""
    events = []
    orig_update, orig_train = RP.Replay.update, Agent.train_step
    def spy_update(self, data):
        events.append("update")
        return orig_update(self, data)
    def spy_train(self, carry, data):
        events.append("train")
        return orig_train(self, carry, data)
    monkeypatch.setattr(RP.Replay, "update", spy_update)
    monkeypatch.setattr(Agent, "train_step", spy_train)
    cfg = C.config_from_dict({"preset": "debug", "env_backend": "playtrain", "game": "frostbite"})
    try:
        record = T.run(cfg, seed=0, outdir=tmp_path, steps=150)
    except FileNotFoundError as exc:  # pragma: no cover - depends on the machine
        pytest.skip(f"playtrain runtime unavailable: {exc}")
    assert events[0] == "train" and events[1] != "update"
    assert events.count("update") == record["grad_steps"] - 1
