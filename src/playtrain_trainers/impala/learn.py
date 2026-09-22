"""V-trace learner step. Pure function — easy to unit-test against torchbeast.

The signature mirrors monobeast.learn(): same batch shape, same loss
breakdown, same gradient/optimizer call sequence. Side effects:
  - Mutates `learner_model` via optimizer.step().
  - Mutates `actor_model` via state_dict load (so freshly-spawned rollouts
    use the just-updated weights — this is the IMPALA actor-learner sync
    point). Pass None in central_gpu mode, where actors never forward on
    the shared CPU model and the sync would be a dead full-net GPU->CPU
    transfer per gradient step.

Returned stats are DETACHED DEVICE TENSORS (episode_returns may be empty).
No .item()/.cpu() happens here — the caller chooses when to pay the
CUDA-sync + GIL cost (train.py does it on a stats_log_every cadence).
"""
from __future__ import annotations

import threading

import torch
from torch import nn

from playtrain_trainers.impala import losses, vtrace
from playtrain_trainers.impala.fwp import FWP_CORES, matrix_state_norms


#: Which component a parameter belongs to, for gradient attribution.
GRAD_GROUPS = ("encoder", "core.set_block", "core.W_k", "core.W_v", "core.W_q",
               "core.W_o", "core.W_p", "core.w_b", "core.rnn", "heads", "mix")


def grad_group(name: str) -> str:
    if name.startswith("encoder"):
        return "encoder"
    if name.startswith("core.set_block"):
        return "core.set_block"
    for part in ("W_k", "W_v", "W_q", "W_o", "W_p", "w_b"):
        if f"core.{part}" in name:
            return f"core.{part}"
    if name.startswith("core"):
        return "core.rnn"
    if name.startswith(("policy", "baseline")):
        return "heads"
    if name.startswith("mix"):
        return "mix"
    return "other"


def grad_group_norms(model) -> dict:
    """Pre-clip gradient norm per component, as detached tensors.

    R.1a established that this cannot be reproduced on synthetic batches, so
    it has to be measured here, inside a real run. Off by default: it adds a
    reduction per parameter on every step, on top of the one clip_grad_norm_
    already does.
    """
    sums: dict = {}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g = grad_group(name)
        sq = p.grad.detach().pow(2).sum()
        sums[g] = sq if g not in sums else sums[g] + sq
    return {f"gradgrp/{k}": v.sqrt() for k, v in sums.items()}


def learn(
    *,
    actor_model: nn.Module | None,
    learner_model: nn.Module,
    batch: dict,
    initial_agent_state,
    optimizer: torch.optim.Optimizer,
    scheduler,
    discounting: float,
    baseline_cost: float,
    entropy_cost: float,
    grad_norm_clipping: float,
    reward_clipping: str = "abs_one",
    win_bonus: float | None = None,
    win_bonus_threshold: float = 10_000.0,
    win_bonus_slope: float | None = None,
    win_bonus_max: float | None = None,
    use_popart: bool = False,
    popart_beta: float = 3e-4,
    lock: threading.Lock | None = None,
    log_grad_groups: bool = False,
    log_vtrace: bool = False,
) -> dict:
    """One V-trace gradient step on a (T+1, B, ...) batch."""
    lock = lock or threading.Lock()
    with lock:
        learner_outputs, _ = learner_model(batch, initial_agent_state)

        # Final V slice bootstraps; trim leading-aligned action vs. obs.
        bootstrap_value = learner_outputs["baseline"][-1]
        batch = {k: t[1:] for k, t in batch.items()}
        learner_outputs = {k: t[:-1] for k, t in learner_outputs.items()}

        # PopArt: the (torch.compiled) forward emits a NORMALIZED value; convert
        # to raw reward scale HERE, in eager, so V-trace's returns and bootstrap
        # are in reward units. Reading the running buffers outside the compiled
        # graph is what avoids the per-step recompile storm that reading them
        # inside forward triggers. With use_popart=False this is the identity.
        if use_popart:
            # Clone the buffers: `values` feeds the loss, so the multiply saves
            # sigma for v_tilde's backward; update_popart_stats copy_()s the
            # buffers in-place, which would trip autograd's version check if we
            # used the live buffers here.
            sigma_old = learner_model.popart_sigma.detach().clone()
            mu_old = learner_model.popart_mu.detach().clone()
            values = learner_outputs["baseline"] * sigma_old + mu_old
            bootstrap_value = bootstrap_value * sigma_old + mu_old
        else:
            values = learner_outputs["baseline"]

        rewards = batch["reward"]
        if use_popart:
            # PopArt normalizes the VALUE TARGETS instead of the rewards, so
            # feed RAW, ordering-preserving rewards (win >> pickup) untouched.
            # reward_clipping is ignored in this mode.
            clipped_rewards = rewards
        elif reward_clipping == "abs_one":
            clipped_rewards = torch.clamp(rewards, -1.0, 1.0)
            if win_bonus is not None:
                # HYBRID CLIP: keep +-1 on the dense sub-goal breadcrumbs (good
                # for exploration), but give the terminal WIN a larger fixed
                # value so there's a gradient to CONSOLIDATE winning. The win is
                # the only reward event above win_bonus_threshold (sub-goals are
                # <=1000, win is +50000+), so a magnitude test isolates it.
                if win_bonus_slope is not None:
                    # GRADED win: map the terminal reward's magnitude into a
                    # bounded modest range [win_bonus, win_bonus_max], preserving
                    # order. A faster / higher-lives win (bigger raw terminal) ->
                    # bigger learner signal, so the policy learns EFFICIENCY, while
                    # the range stays small (no symlog-style scale detuning) and
                    # sub-goals stay +-1 (exploration behavior unchanged).
                    graded = win_bonus + win_bonus_slope * (rewards - win_bonus_threshold)
                    if win_bonus_max is not None:
                        graded = torch.clamp(graded, win_bonus, win_bonus_max)
                    win_val = graded
                else:
                    # fixed win (original hybrid clip)
                    win_val = torch.full_like(rewards, win_bonus)
                clipped_rewards = torch.where(
                    rewards >= win_bonus_threshold,
                    win_val,
                    clipped_rewards,
                )
        elif reward_clipping == "symlog":
            # Sign-preserving log squash sign(x)*log1p(|x|) (same transform as
            # PPO's reward_clip="symlog" / playtrain_trainers.policy.symlog). Unlike abs_one
            # it preserves the *ordering* of the reward ladder — a +50000 win stays
            # strictly larger than a +500 pickup (10.8 vs 6.2) instead of collapsing
            # both to +1 — while taming the 100x magnitude so vtrace value targets
            # stay stable. The fix for the abs_one farming pathology (job 18565971/89).
            clipped_rewards = torch.sign(rewards) * torch.log1p(rewards.abs())
        elif reward_clipping == "none":
            clipped_rewards = rewards
        else:
            raise ValueError(f"unknown reward_clipping={reward_clipping!r}")

        discounts = (~batch["done"]).float() * discounting

        vt = vtrace.from_logits(
            behavior_policy_logits=batch["policy_logits"],
            target_policy_logits=learner_outputs["policy_logits"],
            actions=batch["action"],
            discounts=discounts,
            rewards=clipped_rewards,
            values=values,
            bootstrap_value=bootstrap_value,
        )

        pg_advantages = vt.pg_advantages
        value_error = vt.vs - values
        if use_popart:
            # Canonical PopArt ordering: update (mu, sigma) from THIS batch's
            # V-trace targets and POP-rescale the head BEFORE the loss, so the
            # value regression is normalized from step 1. (Updating AFTER the
            # SGD step left step 1 with sigma=1 against ~50k-scale targets -> a
            # huge gradient -> NaN weights -> the next forward's multinomial()
            # threw in the learner thread and hung the whole run.) Only the
            # BUFFER stats move here (autograd-safe); the head-weight rescale
            # (POP) mutates params and must wait until after backward.
            popart_old = learner_model.update_popart_stats(vt.vs, beta=popart_beta)
            popart_sigma = learner_model.popart_sigma
            # Both V-trace outputs are raw reward units; divide by the UPDATED
            # sigma so the pg and value scales stay O(1). mu cancels in
            # value_error = (vs - values)/sigma; grad flows to v_tilde via values.
            pg_advantages = pg_advantages / popart_sigma
            value_error = value_error / popart_sigma
        # V-trace diagnostics. The gradient forensics located the excess in
        # the policy term, pg = -A * log pi, with entropy only 1.2x — so the
        # advantages A are the unmeasured suspect. |log rho| is the behaviour/
        # target divergence directly; the clip fraction says how often V-trace
        # is truncating it. Detached, no host sync, off by default.
        if log_vtrace:
            rhos = torch.exp(vt.log_rhos.detach())
            adv = pg_advantages.detach()
            stats_vtrace = {
                "vtrace/adv_abs_mean": adv.abs().mean(),
                "vtrace/adv_std": adv.std(),
                "vtrace/adv_abs_max": adv.abs().max(),
                "vtrace/log_rho_abs_mean": vt.log_rhos.detach().abs().mean(),
                "vtrace/rho_clip_frac": (rhos > 1.0).float().mean(),
                "vtrace/td_abs_mean": (vt.vs - values).detach().abs().mean(),
            }
            # fwp-gate G.1: |log rho| by position within the unroll, 10 buckets
            # (fewer when T < 10). If the recurrent state's sensitivity to the
            # parameter change accumulates along the unroll (no forgetting),
            # the mismatch grows with position; a gated state stays flat.
            abs_lr = vt.log_rhos.detach().abs()  # [T, B]
            for b, chunk in enumerate(torch.tensor_split(abs_lr, min(10, abs_lr.shape[0]), dim=0)):
                stats_vtrace[f"vtrace/log_rho_abs_by_pos/{b:02d}"] = chunk.mean()
        pg_loss = losses.compute_policy_gradient_loss(
            learner_outputs["policy_logits"], batch["action"], pg_advantages
        )
        baseline_loss = baseline_cost * losses.compute_baseline_loss(value_error)
        entropy_loss = entropy_cost * losses.compute_entropy_loss(
            learner_outputs["policy_logits"]
        )
        total_loss = pg_loss + baseline_loss + entropy_loss

        # Stats stay as detached device tensors — no .item()/.cpu() here. Each
        # sync stalls the CUDA stream AND holds the GIL (starving the inference
        # thread); the caller decides when to pay that cost (train.py logs on a
        # cadence, accumulating episode_returns across learn steps so nothing
        # is dropped).
        episode_returns = batch["episode_return"][batch["done"]]
        stats = dict(
            episode_returns=episode_returns.detach(),
            total_loss=total_loss.detach(),
            pg_loss=pg_loss.detach(),
            baseline_loss=baseline_loss.detach(),
            entropy_loss=entropy_loss.detach(),
        )

        # Fast-weight state size entering this unroll. Measured here, before
        # the compiled forward is involved, so the graph is untouched; kept as
        # detached tensors for the same no-host-sync reason as the losses.
        if getattr(learner_model, "core_kind", "ff") in FWP_CORES:
            norms = matrix_state_norms(initial_agent_state)
            if norms is not None:
                stats["fwp_state_norm_mean"], stats["fwp_state_norm_max"] = norms
            # fwp-gate G.2: the learned retention alpha of this unroll, stashed
            # by the core's forward (detached; the compiled forward replays the
            # attribute write onto the original module).
            alpha = getattr(getattr(learner_model, "core", None), "last_alpha", None)
            if alpha is not None:
                stats["fwp_alpha_mean"] = alpha.mean()
                stats["fwp_alpha_min"] = alpha.min()
            gate = getattr(getattr(learner_model, "core", None), "last_gate", None)
            if gate is not None:
                stats["fwp_gate_mean"] = gate.mean()
                stats["fwp_gate_min"] = gate.min()

        # NOTE: no per-step finite guard here — it read total_loss on the host
        # (a CUDA sync) every step, which holds the GIL and starves the inference
        # thread, collapsing 1-GPU MPS concurrency from ~100k back to the ~61k
        # time-slicing wall. The before-loss PopArt stats update prevents the
        # cold-start blow-up that motivated the guard (validated: 2M+3M steps,
        # 0 NaN), and grad-norm clipping bounds updates. Re-add as a PERIODIC
        # check (every N steps) if a non-finite ever recurs — never per-step.
        optimizer.zero_grad()
        total_loss.backward()
        # The return value is the total norm BEFORE clipping, which is the
        # diagnostic — how hard the update wanted to pull, not how hard it was
        # allowed to. Free: the reduction already happened inside the clip.
        if log_grad_groups:
            stats.update(grad_group_norms(learner_model))
        if log_vtrace:
            stats.update(stats_vtrace)
        stats["grad_norm"] = nn.utils.clip_grad_norm_(
            learner_model.parameters(), grad_norm_clipping
        ).detach()
        # learn() is shared with other nets (the torchbeast-comparison
        # _TinyNet among them), so this is opt-in rather than assumed.
        scale = getattr(learner_model, "core_out_scale", None)
        if scale is not None:
            stats["core_out_scale"] = scale.detach().clone()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        # POP half of PopArt: now that backward is done, rescale the baseline
        # head so its un-normalized output is preserved across this step's
        # (mu, sigma) change (paired with update_popart_stats above).
        if use_popart:
            learner_model.popart_rescale_head(*popart_old)

        # Push updated weights to the shared actor copy. None in central_gpu
        # mode: actors never forward on the shared CPU model there, so the
        # full GPU->CPU state_dict transfer per gradient step is dead work
        # (and it holds the GIL, starving the inference-server thread).
        if actor_model is not None:
            actor_model.load_state_dict(learner_model.state_dict())
        return stats
