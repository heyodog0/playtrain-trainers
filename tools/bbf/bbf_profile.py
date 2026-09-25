"""Where the BBF update's time actually goes, measured on a GPU.

The loop's measured 15.2 env steps/s looks slow for an H100, so this prices
each component of one optimizer step. The answer is the SPR target path: at
`jumps = 5` the EMA encoder runs on `batch * jumps` observations, which is
most of the update's encoder work.

Run under sbatch (needs a GPU): tools/bbf/bbf_profile.sbatch
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from playtrain_trainers.bbf.config import BBFConfig
from playtrain_trainers.bbf.losses import build_optimizer, build_target
from playtrain_trainers.bbf.net import BBFNetwork
from playtrain_trainers.bbf.replay import SubsequenceReplayBuffer
from playtrain_trainers.bbf.train import update_step

OUT = Path("results/bbf/profile")


def timed(fn, k=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(k):
        fn()
    torch.cuda.synchronize()
    return 1000 * (time.time() - t0) / k


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("this profile needs a GPU")
    OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    gpu = torch.cuda.get_device_name(0)
    cfg = BBFConfig()
    net = BBFNetwork(cfg, 8).to(dev)
    tgt = build_target(net)
    opt = build_optimizer(net, cfg)

    buf = SubsequenceReplayBuffer(cfg)
    frame = np.zeros(buf.frame_shape, np.uint8)
    for i in range(3000):
        buf.add(frame, i % 8, 1.0 if i % 50 == 0 else 0.0, i % 300 == 299)
    rng = np.random.default_rng(0)
    batch = buf.sample(cfg.batch_size * cfg.batches_to_group, 3, cfg.gamma, rng, dev)

    b = batch["obs"].shape[0]
    obs_f = batch["obs"].float() / 255.0
    spr_flat = batch["spr_obs"].flatten(0, 1).float() / 255.0
    n_spr = spr_flat.shape[0]

    rows = {
        f"encoder_fwd_batch{b}": timed(lambda: net.encoder(obs_f)),
        f"encoder_fwd_batch{n_spr}_spr_target": timed(lambda: tgt.encoder(spr_flat)),
        f"target_projections_batch{n_spr}": timed(lambda: tgt.target_projections(spr_flat)),
        f"forward_with_spr_batch{b}": timed(lambda: net.forward_with_spr(obs_f, batch["spr_actions"])),
        "replay_sample": timed(
            lambda: buf.sample(cfg.batch_size * cfg.batches_to_group, 3, cfg.gamma, rng, dev),
            k=10,
        ),
    }
    full = timed(lambda: update_step(net, tgt, opt, batch, cfg, cfg.gamma), k=15)
    rows["FULL_update_step"] = full

    # What TF32 would buy. Measured, not guessed. It changes numerics, so this
    # is information for the human (F-011), not a protocol change: the gin
    # pins half_precision=False and PROTOCOL sections 1-3 are frozen.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    tf32 = timed(lambda: update_step(net, tgt, opt, batch, cfg, cfg.gamma), k=15)
    rows["FULL_update_step_tf32"] = tf32

    record = {
        "gpu": gpu,
        "batch_after_grouping": b,
        "jumps": cfg.jumps,
        "spr_target_batch": n_spr,
        "params": net.parameter_counts()["total"],
        "precision": "fp32 (gin half_precision=False)",
        "ms": {k: round(v, 3) for k, v in rows.items()},
        "updates_per_s_fp32": round(1000 / full, 2),
        "updates_per_s_tf32": round(1000 / tf32, 2),
        "tf32_speedup": round(full / tf32, 3),
        "projected_hours_100k_steps_rr2_fp32": round(100_000 / (1000 / full) / 3600, 2),
        "projected_hours_100k_steps_rr2_tf32": round(100_000 / (1000 / tf32) / 3600, 2),
    }
    (OUT / "profile.json").write_text(json.dumps(record, indent=2) + "\n")

    print(f"gpu {gpu}")
    print(f"batch {b} after grouping, jumps {cfg.jumps}, SPR target batch {n_spr}")
    print(f"precision: {record['precision']}\n")
    w = max(len(k) for k in rows)
    for k, v in rows.items():
        share = f"{100 * v / full:5.1f}% of the update" if k != "FULL_update_step" else ""
        print(f"  {k:<{w}}  {v:8.2f} ms  {share}")
    print(f"\n  fp32 {record['updates_per_s_fp32']:.1f} updates/s "
          f"-> {record['projected_hours_100k_steps_rr2_fp32']} h for 100k steps at RR=2")
    print(f"  tf32 {record['updates_per_s_tf32']:.1f} updates/s "
          f"-> {record['projected_hours_100k_steps_rr2_tf32']} h  "
          f"({record['tf32_speedup']:.2f}x)")
    print(f"\nwrote {OUT / 'profile.json'}")


if __name__ == "__main__":
    main()
