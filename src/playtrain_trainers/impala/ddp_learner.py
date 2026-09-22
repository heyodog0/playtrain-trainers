"""Data-parallel learner ranks for IMPALA (single-run multi-GPU learning).

Each rank is a PROCESS owning one GPU and one model replica wrapped in
torch.nn.parallel.DistributedDataParallel. Ranks independently pull whole
(T+1, B) slots off the shared full_queue (data parallelism over rollouts —
no batch splitting), run the same V-trace learn step, and DDP all-reduces
gradients during backward, so replicas stay bit-identical without any
explicit sync. Rank 0 additionally publishes weights to the seqlock
weight_state (workers see one consistent policy) and writes final.pt.

Why processes, not threads: torch DDP forbids concurrent backward on one
module from multiple threads, and separate processes also dodge the GIL
serialization the learner threads pay under bf16+compile.

Effective learner batch per gradient step = world_size x batch_size
(learning-rate implications are the caller's concern; for throughput
benches it is irrelevant). Step accounting: each rank adds T*B to the
shared step counter per learn step, so the monitor's sps= line reports
TOTAL frames consumed across ranks.

vec/remote_vec modes only (one slot == one learner batch, no stacking).

LSTM: each rank replays its own slot from that slot's own recurrent-state
snapshot — state is per-slot, (num_layers, B, hidden), so ranks share
nothing but gradients and no cross-rank state coupling exists. Two things
make this work without special handling: the data-dependent LSTM
segmentation is already excluded from torch.compile (see net.
_segmented_lstm), and DDP is built without static_graph, so a
per-batch-varying number of LSTM sub-calls is fine — the parameter set is
identical every iteration, which is all the reducer requires.
"""
from __future__ import annotations

import datetime
import logging
import os
import time
import traceback
from collections import deque

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from playtrain_trainers.impala import losses, vtrace
from playtrain_trainers.impala.net import ImpalaNet
from playtrain_trainers.impala.vec_actor import publish_weights


def _learn_step(model, batch, optimizer, cfg_d, core_state=()) -> torch.Tensor:
    """learn()'s body, DDP-safe (no lock, no actor_model).
    Forward MUST go through the DDP wrapper so backward all-reduces.
    `core_state` is the slot's recurrent state entering the unroll; () in
    feedforward mode, in which case the net ignores it."""
    outputs, _ = model(batch, core_state)
    bootstrap_value = outputs["baseline"][-1]
    batch = {k: t[1:] for k, t in batch.items()}
    outputs = {k: t[:-1] for k, t in outputs.items()}
    rewards = torch.clamp(batch["reward"], -1.0, 1.0)
    discounts = (~batch["done"]).float() * cfg_d["discounting"]
    vt = vtrace.from_logits(
        behavior_policy_logits=batch["policy_logits"],
        target_policy_logits=outputs["policy_logits"],
        actions=batch["action"],
        discounts=discounts,
        rewards=rewards,
        values=outputs["baseline"],
        bootstrap_value=bootstrap_value,
    )
    pg_loss = losses.compute_policy_gradient_loss(
        outputs["policy_logits"], batch["action"], vt.pg_advantages)
    baseline_loss = cfg_d["baseline_cost"] * losses.compute_baseline_loss(
        vt.vs - outputs["baseline"])
    entropy_loss = cfg_d["entropy_cost"] * losses.compute_entropy_loss(
        outputs["policy_logits"])
    total_loss = pg_loss + baseline_loss + entropy_loss
    optimizer.zero_grad()
    total_loss.backward()  # <- DDP all-reduce happens here
    nn.utils.clip_grad_norm_(model.parameters(), cfg_d["grad_norm_clipping"])
    optimizer.step()
    return total_loss.detach()


def _acquire_device(rank: int, attempts: int = 12, delay: float = 2.0) -> None:
    """Take cuda:{rank}, retrying past the MPS server's start-up race.

    `nvidia-cuda-mps-control -d` starts the CONTROL daemon; the per-GPU MPS
    server spawns lazily when the first client connects. The learner ranks are
    spawned at the same moment as the vec workers, so two processes can race to
    be the first client on a device and the loser gets

        CUDA error: CUDA-capable device(s) is/are busy or unavailable
        (cudaErrorDevicesUnavailable)

    on its very first CUDA call. Measured ~30% of launches with 2 DDP learners;
    the 1-learner config never hits it because nothing races for cuda:0. The
    window is short, so a bounded retry closes it. Allocating a tensor is part
    of the check: set_device alone is lazy and can succeed before the context
    is really usable.
    """
    last = None
    for i in range(attempts):
        try:
            torch.cuda.set_device(rank)
            torch.zeros(1, device=f"cuda:{rank}")
            if i:
                logging.info("cuda:%d acquired after %d retries", rank, i)
            return
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(delay)
    raise RuntimeError(
        f"could not acquire cuda:{rank} after {attempts} attempts "
        f"({attempts * delay:.0f}s). Under MPS this usually means the server "
        f"never came up for that device; check CUDA_MPS_PIPE_DIRECTORY and "
        f"that nvidia-cuda-mps-control is running. Last error: {last}")


def ddp_learner(
    rank: int,
    world: int,
    rdzv_port: int,
    cfg_d: dict,
    free_queue,
    full_queue,
    buffers,
    state_buffers,
    weight_state: dict,
    step_value,
) -> None:
    """One learner rank. cfg_d: plain-dict subset of ImpalaConfig
    (obs_shape, num_actions, features_dim, net, unroll_length, batch_size,
    total_steps, loss/optim hyperparams, learner_precision, compile_mode,
    frame_hwc, stats_log_every). state_buffers: per-slot (h, c) snapshots
    from create_vec_state_buffers; a list of () tuples in feedforward mode."""
    try:
        logging.basicConfig(
            level=logging.INFO,
            format=f"[ddp-{rank} %(asctime)s] %(message)s")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(rdzv_port))
        _acquire_device(rank)
        torch.distributed.init_process_group(
            "nccl", rank=rank, world_size=world,
            timeout=datetime.timedelta(minutes=15))
        device = torch.device(f"cuda:{rank}")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        model = ImpalaNet(tuple(cfg_d["obs_shape"]), cfg_d["num_actions"],
                          features_dim=cfg_d["features_dim"], use_lstm=False,
                          channels_last=True,
                          net=cfg_d["net"],
                          core=cfg_d.get("core", ""),
                          fwp_dim=int(cfg_d.get("fwp_dim", 128)),
                          fwp_heads=int(cfg_d.get("fwp_heads", 8)),
                          fwp_read=str(cfg_d.get("fwp_read", "joint")),
                          fwp_error=str(cfg_d.get("fwp_error", "joint")),
                          fwp_write=str(cfg_d.get("fwp_write", "delta")),
                          fwp_decay=float(cfg_d.get("fwp_decay", 0.0)),
                          fwp_w_o_gain=float(cfg_d.get("fwp_w_o_gain", 0.1)),
                          fwp_read_norm=bool(cfg_d.get("fwp_read_norm", False)),
                          fwp_w_p_init=float(cfg_d.get("fwp_w_p_init", 0.0)),
                          fwp_ref_heads=int(cfg_d.get("fwp_ref_heads", 4)),
                          fwp_ref_dim_head=int(cfg_d.get("fwp_ref_dim_head", 64)),
                          fwp_feature_map=str(cfg_d.get("fwp_feature_map", "l2k")),
                          fwp_multihead=bool(cfg_d.get("fwp_multihead", False)),
                          fwp_key_scale=float(cfg_d.get("fwp_key_scale", 1.0)),
                          fwp_beta_max=float(cfg_d.get("fwp_beta_max", 1.0)),
                          fwp_gate=bool(cfg_d.get("fwp_gate", False)),
                          fwp_out_norm=bool(cfg_d.get("fwp_out_norm", False)),
                          fwp_out_gate=bool(cfg_d.get("fwp_out_gate", False))).to(device)
        model = model.to(memory_format=torch.channels_last)
        # Identical init across ranks: rank 0's weights are the reference
        # (they were already published to weight_state by train()).
        sd = {n: t.to(device) for n, t in
              zip(weight_state["names"], weight_state["tensors"])}
        model.load_state_dict(sd)
        if cfg_d["compile_mode"] != "off":
            model.compile(mode=cfg_d["compile_mode"])
        ddp_model = DDP(model, device_ids=[rank])
        from playtrain_trainers.impala.train import optimizer_param_groups
        optimizer = torch.optim.RMSprop(
            optimizer_param_groups(model, cfg_d["learning_rate"], float(cfg_d.get("core_lr_mult", 1.0))),
            lr=cfg_d["learning_rate"], momentum=0.0,
            eps=cfg_d["rmsprop_epsilon"], alpha=cfg_d["rmsprop_alpha"])
        autocast = (torch.autocast("cuda", torch.bfloat16)
                    if cfg_d["learner_precision"] == "bf16"
                    else torch.autocast("cuda", enabled=False))
        T, B = cfg_d["unroll_length"], cfg_d["batch_size"]
        logging.info("rank %d/%d up on cuda:%d", rank, world, rank)

        # Rank 0 logs episode returns from the batches it consumes to its own
        # event file in the run's tb dir (TensorBoard merges event files per
        # run dir). The main process only sees sps/mem in DDP mode; without
        # this, DDP runs have NO learning curve. Rank 0's batches are an
        # unbiased ~1/world sample of all trajectories, so the mean matters,
        # not the count.
        writer = None
        # Must be at least the environment count. With 12 workers x 512 envs
        # (double-buffered 2x256) there are 6,144 environments, so a 512-slot
        # window is 12x undersampled. It matters for any game whose episodes end
        # in sync: maze has no failure state, so every unsolved episode lasts
        # exactly max_steps, and ~6,144 truncations land in one logging interval
        # and overwrite the whole window with zeros. That produced a 12.3M-period
        # sawtooth plus an opening plateau in the curve, neither of which was in
        # the agent's behaviour.
        recent_returns: deque = deque(maxlen=8192)
        if rank == 0 and cfg_d.get("log_dir"):
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(
                log_dir=os.path.join(cfg_d["log_dir"], "tb"),
                filename_suffix=".ddp0")

        learn_steps = 0
        while step_value.value < cfg_d["total_steps"]:
            index = full_queue.get()
            if index is None:
                break
            batch = {k: buffers[k][index].to(device, non_blocking=True)
                     for k in buffers}
            # The state ENTERING this unroll, written by the rollout worker.
            # Empty in feedforward mode, so this is a no-op there.
            core_state = tuple(t_.to(device, non_blocking=True)
                               for t_ in state_buffers[index])
            torch.cuda.current_stream().synchronize()
            free_queue.put(index)
            if cfg_d.get("frame_hwc"):
                batch["frame"] = batch["frame"].permute(0, 1, 4, 2, 3)
            with autocast:
                loss = _learn_step(ddp_model, batch, optimizer, cfg_d,
                                   core_state)
            learn_steps += 1
            with step_value.get_lock():
                step_value.value += T * B
            # Rank 0 is the ONLY publisher, so the seqlock's single-writer
            # requirement holds without any lock (a spawned mp.Lock also
            # can't be created inline in Process args — Linux unlinks named
            # semaphores at creation and the child's rebuild 404s).
            if rank == 0 and learn_steps % cfg_d["sync_every"] == 0:
                publish_weights(weight_state, model)
            if writer is not None:
                dones = batch["done"]
                if dones.any():
                    recent_returns.extend(
                        batch["episode_return"][dones].float().cpu().tolist())
                if learn_steps % cfg_d["stats_log_every"] == 0 and recent_returns:
                    writer.add_scalar(
                        "charts/ep_return_mean",
                        sum(recent_returns) / len(recent_returns),
                        step_value.value)
            if rank == 0 and learn_steps % (10 * cfg_d["stats_log_every"]) == 0:
                logging.info("learn_steps=%d step=%d loss=%.1f",
                             learn_steps, step_value.value, float(loss))
        if writer is not None:
            writer.close()
        torch.distributed.destroy_process_group()
        logging.info("rank %d done (%d learn steps)", rank, learn_steps)
    except KeyboardInterrupt:
        pass
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        raise
