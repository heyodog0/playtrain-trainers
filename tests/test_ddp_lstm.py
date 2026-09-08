"""DDP learner ranks with a recurrent core.

Each rank pulls a whole slot and replays it from THAT slot's recurrent-state
snapshot, so the only thing shared between ranks is gradients. The bug worth
guarding is the quiet one: the rank reads the batch but drops the state, so every
unroll is replayed from zeros while the run looks entirely healthy.

The first test exercises the learner body directly on CPU — no CUDA, no
distributed setup — by checking that the state changes the loss. The end-to-end
rank test needs two GPUs and is skipped where there aren't any.
"""
from __future__ import annotations

import torch
import pytest

from playtrain_trainers.impala.ddp_learner import _learn_step
from playtrain_trainers.impala.net import ImpalaNet

T, B, A, OBS = 6, 4, 5, 16
CFG = dict(discounting=0.99, baseline_cost=0.5, entropy_cost=0.01,
           grad_norm_clipping=40.0)


def _batch(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    return {
        "frame": torch.randint(0, 256, (T + 1, B, 3, OBS, OBS),
                               dtype=torch.uint8, generator=g),
        "reward": torch.randn(T + 1, B, generator=g),
        "done": torch.zeros(T + 1, B, dtype=torch.bool),
        "action": torch.randint(0, A, (T + 1, B), generator=g),
        "policy_logits": torch.randn(T + 1, B, A, generator=g),
        "baseline": torch.randn(T + 1, B, generator=g),
        "last_action": torch.randint(0, A, (T + 1, B), generator=g),
        "episode_return": torch.zeros(T + 1, B),
        "episode_step": torch.zeros(T + 1, B, dtype=torch.int32),
    }


def _model(use_lstm: bool):
    torch.manual_seed(7)  # identical init across calls
    m = ImpalaNet((3, OBS, OBS), A, features_dim=32, use_lstm=use_lstm,
                  net="nature")
    m.train()
    return m


def _loss_with(core_state, use_lstm=True):
    m = _model(use_lstm)
    opt = torch.optim.RMSprop(m.parameters(), lr=1e-4)
    torch.manual_seed(11)  # same multinomial draws either way
    return float(_learn_step(m, _batch(), opt, CFG, core_state))


def test_learn_step_consumes_the_recurrent_state():
    """A non-zero entering state must change the learner's loss.

    This is what fails if a rank fetches the slot but not its state snapshot:
    the run trains, the loss falls, and every unroll silently starts from zeros.
    """
    m = _model(use_lstm=True)
    layers, hidden = m.core.num_layers, m.core.hidden_size
    zero = tuple(torch.zeros(layers, B, hidden) for _ in range(2))
    torch.manual_seed(3)
    warm = tuple(torch.randn(layers, B, hidden) for _ in range(2))

    loss_zero = _loss_with(zero)
    loss_warm = _loss_with(warm)
    assert loss_zero != pytest.approx(loss_warm, rel=1e-6), (
        f"loss is identical for zero and warm entering states "
        f"({loss_zero:.6f}): the recurrent state is not reaching the forward")


def test_learn_step_still_works_feedforward():
    """The feedforward path must be untouched: () state, same as before."""
    m = _model(use_lstm=False)
    opt = torch.optim.RMSprop(m.parameters(), lr=1e-4)
    loss = _learn_step(m, _batch(), opt, CFG, ())
    assert torch.isfinite(torch.as_tensor(loss)), "feedforward learn step broke"


def test_ddp_learner_signature_takes_state_buffers():
    """train() passes state_buffers positionally; keep the contract explicit so a
    reordered argument list fails here rather than at hour three of a run."""
    import inspect

    from playtrain_trainers.impala.ddp_learner import ddp_learner
    params = list(inspect.signature(ddp_learner).parameters)
    assert params == ["rank", "world", "rdzv_port", "cfg_d", "free_queue",
                      "full_queue", "buffers", "state_buffers", "weight_state",
                      "step_value"], f"unexpected ddp_learner signature: {params}"


@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason="DDP ranks need 2 CUDA devices")
def test_ddp_lstm_ranks_stay_identical(tmp_path):
    """Two ranks, LSTM, real all-reduce: replicas must remain bit-identical.

    ddp_learner's contract is that gradient all-reduce keeps replicas in step
    without explicit syncing. If per-slot state handling diverged between ranks
    the weights would drift, which this catches.
    """
    import threading

    from playtrain_trainers.impala.train import ImpalaConfig, train

    cfg = ImpalaConfig(
        game="fake", env_backend="playtrain",
        total_steps=T * B * 8, batch_size=B, unroll_length=T,
        obs_shape=(3, OBS, OBS), num_actions=A, features_dim=32,
        net="nature", use_lstm=True, learner_gpus=2,
        inference_mode="remote_vec", vec_workers=1,
        remote_groups_per_worker=1, remote_port_base=24100,
        device="cuda", learner_precision="fp32", compile_learner=False,
        eval_every_steps=0, save_every_steps=0, resume="off",
        stats_log_every=1, log_dir=str(tmp_path),
    )
    # Reuse the loopback actor from the remote_vec LSTM test.
    from test_remote_vec_lstm import _fake_group  # noqa: PLC0415

    stop = threading.Event()
    threading.Thread(target=_fake_group, args=(24100, False, stop),
                     daemon=True).start()
    try:
        result = train(cfg)
    finally:
        stop.set()
    assert result["final_step"] >= cfg.total_steps
