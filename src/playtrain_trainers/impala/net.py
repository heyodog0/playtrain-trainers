"""IMPALA agent net: IMPALA-CNN encoder + optional LSTM core + heads.

Mirrors torchbeast/monobeast.AtariNet's I/O contract so the learn/actor code
matches torchbeast's drop-in. Differences:
  - Encoder is our IMPALA-CNN (ProcGen-paper depths [16,32,32]), not the
    Atari NatureCNN-style 3-layer conv stack.

The core between the CNN features and the heads is selected by `core`:
`"ff"`, `"lstm"`, `"deltanet"` or `"compfwp"`. `use_lstm=True` is the old
spelling of `core="lstm"` and still works, so every existing config runs
unchanged; passing both is allowed only when they agree.

Two of those modes are described below; the fast-weight cores are documented
at their own classes.

  use_lstm=False (default) — **purely Markov feedforward**.
    policy/baseline heads see CNN features only. last_action and reward are
    NOT concatenated. Reason: without a recurrent core, feeding last_action
    makes the policy semi-Markov — it conditions on the action history a
    stochastically-sampled training actor took, which makes the argmax
    projection of the trained policy brittle. Deterministic argmax rollouts
    of policies trained with last_action conditioning often fail to win even
    at the training layout, because the policy expects a *distribution* of
    past actions at each state, not the locked-in argmax chain. Dropping
    last_action keeps the policy argmax-stable when returns saturate. See
    feedback_impala_argmax.md for the diagnostic on the v5_door_6x6 runs.

  use_lstm=True — **recurrent core** (this is the LSTM variant).
    An nn.LSTM sits between the CNN features and the heads. The hidden state
    is threaded across timesteps within an unroll and, via the carried
    core_state, across unroll segments. It is RESET TO ZERO at episode
    boundaries: done[t] marks frame[t] as the first observation of a fresh
    episode (the env auto-resets on done and emits the post-reset frame at
    the same timestep, see environment.py), so the state going *into* step t
    is zeroed whenever done[t] is true. This mirrors monobeast exactly.

    NOTE: last_action/reward conditioning is still NOT wired in here even in
    LSTM mode. The recurrent core is what would make that conditioning
    principled (monobeast concatenates them), so re-adding it is a natural
    follow-up — but it changes the forward contract and is kept as a separate
    step. The core currently sees CNN features only.

Forward contract:
    inputs = {
        "frame":       [T, B, C, H, W] uint8,
        "reward":      [T, B] float,            # accepted but unused by net
        "done":        [T, B] bool,             # used for state reset (LSTM)
        "last_action": [T, B] int64,            # accepted but unused by net
    }
    returns (
        dict(
            policy_logits=[T, B, A] float,
            baseline=    [T, B]    float,
            action=      [T, B]    int64,
        ),
        core_state,  # () when use_lstm=False; (h, c) tuple when use_lstm=True
    )
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from playtrain_trainers.policy import build_encoder

#: Recurrent cores ImpalaNet can be built with. "ff" is the Markov default.
CORES = ("ff", "lstm", "deltanet", "compfwp")
#: The fast-weight cores, whose state is a flattened matrix rather than (h, c).
FWP_CORES = ("deltanet", "compfwp")


def resolve_core(core: str | None, use_lstm: bool) -> str:
    """Reconcile the new `core` option with the legacy `use_lstm` flag.

    `core` unset falls back to `use_lstm`, so old configs keep their meaning.
    When both are given they must agree — silently letting one win would make
    a stale `use_lstm=True` quietly downgrade a fast-weight run to an LSTM.
    """
    if not core:
        return "lstm" if use_lstm else "ff"
    if core not in CORES:
        raise ValueError(f"core must be one of {CORES}, got {core!r}")
    if use_lstm and core != "lstm":
        raise ValueError(
            f"use_lstm=True contradicts core={core!r}; drop use_lstm from the config"
        )
    return core


@torch.compiler.disable
def _segmented_lstm(core, core_input, done, notdone, core_state, T):
    """Segment-wise LSTM unroll (see forward()). Kept OUT of torch.compile:
    the segmentation is data-dependent (a host-side list of split points that
    changes every batch), which graph-breaks dynamo and triggers a
    recompile-per-pattern storm until it falls back to eager anyway. The
    heavy lifting here is cuDNN's fused multi-step kernel — inductor adds
    nothing; the CNN encoder and heads around it stay compiled."""
    starts = torch.nonzero(done.any(dim=1)).flatten().cpu().tolist()
    if not starts or starts[0] != 0:
        # Always mask entering the unroll: row 0 can carry dones (the
        # buffer's carry frame), and the initial-state mask at t=0 is part
        # of the monobeast contract.
        starts = [0] + starts
    outs = []
    for i, s0 in enumerate(starts):
        s1 = starts[i + 1] if i + 1 < len(starts) else T
        nd = notdone[s0].view(1, -1, 1)
        core_state = tuple(nd * s for s in core_state)
        out, core_state = core(core_input[s0:s1], core_state)
        outs.append(out)
    return torch.flatten(torch.cat(outs), 0, 1), core_state


class ImpalaNet(nn.Module):
    """IMPALA agent. Feedforward (Markov) by default; set use_lstm=True for a
    recurrent LSTM core between the CNN features and the policy/baseline heads.
    See module docstring."""

    def __init__(
        self,
        observation_shape: tuple[int, int, int],
        num_actions: int,
        features_dim: int = 256,
        use_lstm: bool = False,
        channels_last: bool = False,
        use_popart: bool = False,
        net: str = "impala",
        core: str | None = None,
        fwp_dim: int = 128,
    ):
        super().__init__()
        c, h, w = observation_shape
        self.observation_shape = observation_shape
        self.num_actions = num_actions
        # `use_lstm` stays a real attribute because ppo_eval, vec_actor and the
        # model_spec dicts all read it; it is now derived from the core.
        self.core_kind = resolve_core(core, use_lstm)
        self.use_lstm = self.core_kind == "lstm"
        self.fwp_dim = fwp_dim
        # Encoder choice — see policy.build_encoder for the registry. Every
        # model instance in a run (shared/learner/inference/eval/worker) must
        # use the same value: the state_dicts differ.
        self.net = net
        # PopArt (van Hasselt et al. 2016): the baseline head predicts a
        # NORMALIZED value; the un-normalized value is v = sigma*out + mu with
        # running (mu, sigma) of the V-trace targets. Lets the learner train on
        # raw, ordering-preserving rewards (win >> pickup) while the value
        # regression stays O(1) — decouples reward scale from stability.
        # Buffers are registered UNCONDITIONALLY (persistent) so every model
        # instance has an identical state_dict for the actor/inference weight
        # sync, even when this instance runs with use_popart=False.
        self.use_popart = use_popart
        # NHWC memory format for the conv stem (tensor cores prefer it in
        # reduced precision). The flag only converts the INPUT tensor; the
        # caller converts the parameters via
        # `.to(memory_format=torch.channels_last)`. Numerics unchanged —
        # same kernels' math, different memory layout.
        self.channels_last = channels_last

        self.encoder = build_encoder(net, in_channels=c,
                                     features_dim=features_dim, input_hw=h)
        core_in = features_dim
        if self.core_kind == "lstm":
            # Hidden size == feature size (matches monobeast AtariNet).
            self.core = nn.LSTM(features_dim, features_dim, num_layers=1)
        elif self.core_kind in FWP_CORES:
            raise NotImplementedError(
                f"core={self.core_kind!r} is accepted by the config but not yet "
                "built; it lands in the fast-weight core work"
            )
        self.policy = nn.Linear(core_in, num_actions)
        self.baseline = nn.Linear(core_in, 1)
        nn.init.orthogonal_(self.policy.weight, gain=0.01)
        nn.init.zeros_(self.policy.bias)
        nn.init.orthogonal_(self.baseline.weight, gain=1.0)
        nn.init.zeros_(self.baseline.bias)
        # PopArt running stats of the value targets. mu=0, sigma=1 => identity
        # (v == raw head output), so with use_popart=False the net is an exact
        # no-op vs the pre-PopArt behavior. count drives fast initial adaptation.
        self.register_buffer("popart_mu", torch.zeros(1))
        self.register_buffer("popart_sigma", torch.ones(1))
        self.register_buffer("popart_count", torch.zeros(1))

    @torch.no_grad()
    @torch.compiler.disable
    def update_popart_stats(self, targets: torch.Tensor, beta: float = 3e-4):
        """ART half of PopArt: shift running (mu, sigma) toward the value-target
        distribution. Touches ONLY buffers (not params, not autograd-graph
        tensors), so it is safe to call BEFORE the loss/backward — which is
        required so sigma reflects the batch from step 1 (else step 1 regresses
        on raw ~50k targets with sigma=1 and explodes). Count-based step early
        (exact running mean) decaying to a floor of `beta`. Returns the OLD
        (mu, sigma) for the paired popart_rescale_head() call. No-op (returns
        None) when use_popart is False."""
        if not self.use_popart:
            return None
        mu_old, sigma_old = self.popart_mu.clone(), self.popart_sigma.clone()
        self.popart_count += 1.0
        b = torch.clamp(1.0 / self.popart_count, min=beta)  # fast start, then EMA
        t = targets.detach().flatten().float()
        new_mu = (1.0 - b) * mu_old + b * t.mean()
        new_nu = (1.0 - b) * (sigma_old ** 2 + mu_old ** 2) + b * (t ** 2).mean()
        new_sigma = torch.clamp(torch.sqrt(torch.clamp(new_nu - new_mu ** 2, min=1e-4)),
                                1e-4, 1e6)
        self.popart_mu.copy_(new_mu)
        self.popart_sigma.copy_(new_sigma)
        return mu_old, sigma_old

    @torch.no_grad()
    @torch.compiler.disable
    def popart_rescale_head(self, mu_old, sigma_old):
        """POP half: rescale the baseline head's (W, b) so the UN-normalized
        output v = sigma*out + mu is preserved across the (mu, sigma) change in
        update_popart_stats — otherwise moving the normalization corrupts what
        the head learned. MUST run AFTER backward: it mutates params in-place,
        which would break the autograd graph if done before. W' =
        (sigma_old/sigma_new) W; b' = (sigma_old*b + mu_old - mu_new)/sigma_new."""
        if not self.use_popart or mu_old is None:
            return
        ratio = sigma_old / self.popart_sigma
        self.baseline.weight.mul_(ratio)
        self.baseline.bias.mul_(ratio)
        self.baseline.bias.add_(((mu_old - self.popart_mu) / self.popart_sigma).squeeze())

    @torch.no_grad()
    @torch.compiler.disable
    def update_popart(self, targets: torch.Tensor, beta: float = 3e-4):
        """Convenience: full PopArt update (stats + head rescale) in one call.
        Used where there's no backward in between (tests). The learner splits
        the two halves around its optimizer step."""
        old = self.update_popart_stats(targets, beta)
        self.popart_rescale_head(*old) if old is not None else None

    def initial_state(self, batch_size: int = 1):
        """Zero recurrent state, as a tuple of tensors with batch at dim 1.

        Empty tuple for the feedforward core; an (h, c) pair of
        [num_layers, batch_size, hidden] tensors for the LSTM. Every core keeps
        to that contract — a tuple of tensors batched at dim 1 — because the
        buffers, actors and learner assume nothing else about the shape.
        """
        if self.core_kind == "ff":
            return tuple()  # no recurrent state
        return tuple(
            torch.zeros(self.core.num_layers, batch_size, self.core.hidden_size)
            for _ in range(2)
        )

    def forward(self, inputs: dict, core_state=()):
        x = inputs["frame"]  # [T, B, C, H, W] uint8
        T, B, *_ = x.shape
        x = torch.flatten(x, 0, 1)  # [T*B, C, H, W]
        if self.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        core_input = self.encoder(x)  # [T*B, features_dim]

        if self.use_lstm:
            # Episode-boundary handling: done[t] => frame[t] starts a fresh
            # episode => zero the state entering step t (monobeast semantics).
            # The naive implementation is a Python loop over T single-step
            # LSTM calls, masking the state EVERY step — but between episode
            # boundaries the mask is all-ones (a float multiply by 1.0 =
            # exact no-op), so masking is only NEEDED at timesteps where ANY
            # env in the batch resets. Those are sparse (~1 per episode-length
            # frames), so we split the unroll into SEGMENTS at those
            # timesteps and run each segment as ONE fused multi-step LSTM
            # call. Same math, ~T/num_segments fewer launch-bound micro-steps
            # (the T=100 loop was ~104ms of the learner's 208ms per step).
            core_input = core_input.view(T, B, -1)
            done = inputs["done"]  # [T, B]
            notdone = (~done).float()
            if T == 1:
                # Actor inference: one masked cell step, no host sync.
                nd = notdone[0].view(1, -1, 1)
                core_state = tuple(nd * s for s in core_state)
                core_output, core_state = self.core(core_input, core_state)
                core_output = torch.flatten(core_output, 0, 1)
            else:
                core_output, core_state = _segmented_lstm(
                    self.core, core_input, done, notdone, core_state, T)
        else:
            core_output = core_input  # [T*B, features_dim]

        policy_logits = self.policy(core_output)
        # With use_popart, this head output is the NORMALIZED value (v_tilde);
        # un-normalization to raw scale (v = sigma*v_tilde + mu) is done in the
        # EAGER learner (learn.py), NOT here. Reading the running popart buffers
        # inside this (torch.compiled) forward would invalidate dynamo's guards
        # every step — they mutate each update — triggering a recompile storm.
        # Keeping forward popart-agnostic makes the compiled graph stable.
        baseline = self.baseline(core_output).squeeze(-1)

        if self.training:
            action = torch.multinomial(
                F.softmax(policy_logits, dim=-1), num_samples=1
            ).view(T * B)
        else:
            action = torch.argmax(policy_logits, dim=-1)

        return (
            dict(
                policy_logits=policy_logits.view(T, B, self.num_actions),
                baseline=baseline.view(T, B),
                action=action.view(T, B),
            ),
            core_state,
        )
