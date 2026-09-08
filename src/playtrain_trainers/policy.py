"""IMPALA-CNN feature extractor + actor-critic head.

The CNN matches the OpenAI ProcGen paper's `build_impala_cnn` exactly:
3 IMPALA stages with depths [16, 32, 32], each stage = Conv -> MaxPool ->
2x ResidualBlock, followed by ReLU + Linear -> 256-dim features. SB3's
default NatureCNN is materially weaker on procedural benchmarks; using
the IMPALA stem is what makes results paper-comparable.

The actor and critic heads are linear layers off the shared features.
Both PPO (this repo) and IMPALA (later) reuse this module.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical, Independent, Normal
from torch.nn import functional as F


def symlog(x: torch.Tensor) -> torch.Tensor:
    """Sign-preserving log squash: ``sign(x) * log1p(|x|)``. Bounded-*growth*
    (not bounded-*range* like clamp), so it preserves reward magnitude ordering
    while keeping the env's ±500…±80000 rewards to a sane LSTM-input scale
    (~±11). Used to squash the r_{t-1} fed into the recurrent core — a hard
    clamp(-1, 1) would collapse everything past the value-function's symlog
    shaping back to a pure sign bit."""
    return torch.sign(x) * torch.log1p(x.abs())


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(x)
        h = self.conv1(h)
        h = F.relu(h)
        h = self.conv2(h)
        return x + h


class _ImpalaStage(nn.Module):
    def __init__(self, in_c: int, out_c: int, stride: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, 3, stride, 1)
        self.pool = nn.MaxPool2d(3, 2, 1)
        self.res1 = _ResidualBlock(out_c)
        self.res2 = _ResidualBlock(out_c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.res2(self.res1(self.pool(self.conv(x))))


class ImpalaCNN(nn.Module):
    """3-stage IMPALA stem -> 256-dim features. Input: uint8 (B, C, H, W).

    Two orthogonal knobs, both off by default (defaults are byte-identical to
    the ProcGen-paper stem):

    ``gap=True`` — Impoola (arXiv 2503.05546): global average pool instead of
    flatten before the dense layer. The flatten holds 84% of the parameters
    (2048x256) for 1.7% of the arithmetic, so this is ~free in wall-clock but
    cuts params 0.62M -> 0.11M. Two independent papers (Impoola; "Mind the
    GAP!", arXiv 2505.17749) name this junction as the limiting component.

    ``stem_stride=2`` — stage 0's conv strides, so every downstream tensor is
    half-resolution in each dim: 30.6 -> 7.7 MMACs and, measured on an H100 at
    batch 3072, 0.31x the fwd+bwd time. Spatial reduction is the ONLY knob that
    converted to throughput in the sweep; cutting channels (3.7x less
    arithmetic) and depthwise-separable convs (2.4x less) each bought nothing,
    because at this size the stem is launch-bound rather than compute-bound.
    The corollary is that channel width is nearly free while resolution is not.

    The two compose deliberately: GAP makes the head resolution-independent, so
    the stride change is confined to the stem. With flatten, ``stem_stride=2``
    silently rewrites the FC input dim (2048 -> 512), which is a different model
    rather than the same model run cheaper.

    ``set_obs_scale`` requires ``gap`` for the same reason — with a
    resolution-independent head, input resolution becomes a runtime knob rather
    than an architectural commitment.
    """

    def __init__(self, in_channels: int = 3, features_dim: int = 256,
                 depths: tuple[int, ...] = (16, 32, 32), input_hw: int = 64,
                 gap: bool = False, stem_stride: int = 1) -> None:
        super().__init__()
        stages, c = [], in_channels
        for i, d in enumerate(depths):
            stages.append(_ImpalaStage(c, d, stride=stem_stride if i == 0 else 1))
            c = d
        self.stages = nn.Sequential(*stages)
        self.gap = gap
        if gap:
            # Resolution-independent: the dense layer sees one scalar per
            # channel, so no dummy forward is needed to size it.
            self.fc = nn.Linear(depths[-1], features_dim)
        else:
            with torch.no_grad():
                dummy = torch.zeros(1, in_channels, input_hw, input_hw)
                flat = self.stages(dummy).flatten(1).shape[1]
            self.fc = nn.Linear(flat, features_dim)
        self.features_dim = features_dim
        self.obs_scale = 1.0

    def set_obs_scale(self, scale: float) -> None:
        """Bilinearly resize the input by ``scale`` before the stem (1.0 = off).

        Only legal with ``gap``: with flatten the FC input dim is tied to the
        input resolution, so this would be a shape error rather than a cheaper
        forward. Exposed as a mutable attribute, not a constructor argument, so
        a trainer can schedule it across a run (progressive resizing) without
        rebuilding the model or invalidating a checkpoint.
        """
        if scale != 1.0 and not self.gap:
            raise ValueError(
                "set_obs_scale requires gap=True: with the flatten head the FC "
                "input dim is tied to the input resolution."
            )
        self.obs_scale = float(scale)

    def forward(self, obs_uint8: torch.Tensor) -> torch.Tensor:
        x = obs_uint8.float() / 255.0
        if self.obs_scale != 1.0:
            x = F.interpolate(x, scale_factor=self.obs_scale,
                              mode="bilinear", align_corners=False)
        x = self.stages(x)
        x = F.relu(x)
        x = x.mean(dim=(2, 3)) if self.gap else x.flatten(1)
        return F.relu(self.fc(x))


class NatureCNN(nn.Module):
    """Atari NatureCNN stem (DQN 2015) -> features_dim. The tiny-net
    baseline for the throughput push: ~6-10x cheaper per frame than
    ImpalaCNN because the stride-4 first conv collapses the full-resolution
    stage-1 activation traffic that dominates the ImpalaCNN learner profile
    (H100 kernel profile: conv = 68% of update time, mostly stage-1 wgrad).
    Measured JAX-twin learner ceilings at 64x64: V-trace 1.06M frames/s vs
    ImpalaCNN's 153-177k. Padding is SAME-style (not Nature's valid-pad at
    84px) so 64/48px inputs divide cleanly: 64 -> 16 -> 8 -> 8.

    Caveat from the procgen paper: ImpalaCNN materially outperforms this
    stem on procedurally-diverse distributions — validate learning quality
    per suite before using it for science runs. Input: uint8 (B, C, H, W).
    """

    def __init__(self, in_channels: int = 3, features_dim: int = 256,
                 input_hw: int = 64) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, 4, 2), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, input_hw, input_hw)
            flat = self.stem(dummy).flatten(1).shape[1]
        self.fc = nn.Linear(flat, features_dim)
        self.features_dim = features_dim

    def forward(self, obs_uint8: torch.Tensor) -> torch.Tensor:
        x = obs_uint8.float() / 255.0
        return F.relu(self.fc(self.stem(x).flatten(1)))


# Encoder registry. Both trainers select an encoder by this string (PPO via
# ActorCritic, IMPALA via impala.net.ImpalaNet), so it lives here once rather
# than as parallel if/elif chains that can drift apart. Every model instance in
# a run (shared/learner/inference/eval/worker) must pass the same value: the
# state_dicts differ.
#
#   impala      ProcGen-paper stem. The science baseline.
#   nature      DQN stem. The throughput baseline.
#   impoola     impala + GAP head (Impoola). Same speed, 5.6x fewer params.
#   impala_s2   impala + strided stem. Isolates the spatial change from GAP.
#   impoola_s2  both. The intended fast encoder.
#   impoola_s2w both, at double width. Spends the spatial saving back on
#               channels, which the sweep measured as nearly free -- same
#               arithmetic as the baseline stem on a 4x smaller grid.
_IMPALA_VARIANTS: dict[str, dict] = {
    "impala": {},
    "impoola": {"gap": True},
    "impala_s2": {"stem_stride": 2},
    "impoola_s2": {"gap": True, "stem_stride": 2},
    "impoola_s2w": {"gap": True, "stem_stride": 2, "depths": (32, 64, 64)},
}


def build_encoder(net: str, in_channels: int, features_dim: int,
                  input_hw: int) -> nn.Module:
    """Encoder factory shared by both trainers. See ``_IMPALA_VARIANTS``."""
    if net == "nature":
        return NatureCNN(in_channels, features_dim, input_hw=input_hw)
    if net in _IMPALA_VARIANTS:
        return ImpalaCNN(in_channels, features_dim, input_hw=input_hw,
                         **_IMPALA_VARIANTS[net])
    known = "|".join(["nature", *_IMPALA_VARIANTS])
    raise ValueError(f"unknown net={net!r} (expected one of {known})")


class ActorCritic(nn.Module):
    """IMPALA-CNN trunk + categorical actor head + scalar critic head.

    Feedforward (Markov) by default. Set ``use_lstm=True`` for a recurrent LSTM
    core between the CNN features and the actor/critic heads — the PPO analogue
    of ``playtrain_trainers.impala.net.ImpalaNet``'s recurrent core, with the *identical*
    done-reset unroll semantics so behavior matches the validated IMPALA path:

      - The hidden state is threaded across timesteps within a rollout/replay
        and, via the carried ``lstm_state``, across rollout segments.
      - It is RESET TO ZERO at episode boundaries: ``done[t]`` marks ``frame[t]``
        as the first observation of a fresh episode (the env auto-resets on done
        and emits the post-reset frame at the same step — see how the PPO trainer
        writes ``dones_buf[step] = next_done``), so the state going *into* step t
        is zeroed whenever ``done[t]`` is true. This mirrors monobeast/CleanRL.

    Optional ``feed_prev_action_reward`` (LSTM only) concatenates the previous
    action (one-hot) and previous reward into the LSTM input, as IMPALA /
    monobeast / R2D2 do — at step t the core sees ``[cnn(frame_t),
    symlog(r_{t-1}), one_hot(a_{t-1})]``. The reward channel is squashed with
    ``symlog`` (not monobeast's hard clamp(-1,1), which would collapse the env's
    ±80000-scale rewards to a sign bit and undo any symlog reward shaping); feed
    the *raw* reward here, so this cue is independent of the trainer's
    reward_clip. Set ``symlog_reward=False`` to skip that squash and concat the
    reward verbatim — then the caller must feed an already-bounded reward (the
    trainer feeds its reward_clip output in that mode, so bounding follows the
    reward_clip you chose). This conditioning is principled ONLY with a recurrent
    core; the bare feedforward version made argmax rollouts brittle (see
    memory/feedback_impala_argmax.md), so it is rejected unless ``use_lstm``.

    The feedforward path (``forward``/``act``) is byte-identical to the pre-LSTM
    module so existing PPO configs are unchanged. Recurrent training uses the
    state-threaded methods below (``initial_state``, ``act_recurrent``,
    ``get_value``, ``get_action_and_value``), following CleanRL's
    ``ppo_atari_lstm`` reference: minibatch over whole env-columns and replay the
    LSTM through each env's full sequence from the stored per-rollout state. When
    ``feed_prev_action_reward`` is on, those methods also take ``last_action`` and
    ``reward`` (the a_{t-1}/r_{t-1} aligned with each frame).
    """

    def __init__(self, n_actions: int, in_channels: int = 3,
                 features_dim: int = 256, input_hw: int = 64,
                 use_lstm: bool = False,
                 feed_prev_action_reward: bool = False,
                 symlog_reward: bool = True,
                 net: str = "impala",
                 continuous: bool = False) -> None:
        super().__init__()
        if feed_prev_action_reward and not use_lstm:
            raise ValueError(
                "feed_prev_action_reward requires use_lstm=True: conditioning on "
                "action history without a recurrent core makes argmax rollouts "
                "brittle (see memory/feedback_impala_argmax.md)."
            )
        if continuous and (use_lstm or feed_prev_action_reward):
            raise ValueError(
                "continuous=True currently supports the feedforward path only "
                "(the LSTM's prev-action one-hot plumbing assumes Discrete)."
            )
        self.encoder = build_encoder(net, in_channels, features_dim, input_hw)
        self.use_lstm = use_lstm
        self.feed_prev_action_reward = feed_prev_action_reward
        # When feeding r_{t-1}: True squashes it with symlog (caller feeds the
        # raw reward); False concats it verbatim (caller is expected to feed an
        # already-bounded reward, e.g. the trainer's reward_clip output).
        self.symlog_reward = symlog_reward
        self.n_actions = n_actions
        if use_lstm:
            # Hidden size == feature size (matches ImpalaNet / monobeast AtariNet).
            # Input grows by (n_actions + 1) when prev action/reward are fed.
            # Left at PyTorch default init to match the validated IMPALA core.
            core_in = features_dim + (n_actions + 1 if feed_prev_action_reward else 0)
            self.lstm = nn.LSTM(core_in, features_dim, num_layers=1)
        self.continuous = continuous
        # Continuous (Box) head: actor outputs per-dim means; a state-independent
        # learned log-std (CleanRL continuous-PPO convention). Actions are
        # unsquashed Normal samples — PlayTrain's box envs clamp to the channel
        # range when quantizing, the standard clip-at-env approach.
        self.actor = nn.Linear(features_dim, n_actions)
        if continuous:
            self.log_std = nn.Parameter(torch.zeros(n_actions))
        self.critic = nn.Linear(features_dim, 1)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.zeros_(self.actor.bias)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    def _dist(self, z: torch.Tensor):
        """Policy distribution over features: Categorical (Discrete) or a
        diagonal Normal (Box; Independent sums log-prob/entropy over dims, so
        downstream shapes match the discrete path)."""
        if self.continuous:
            mean = self.actor(z)
            return Independent(Normal(mean, torch.exp(self.log_std)), 1)
        return Categorical(logits=self.actor(z))

    def initial_state(self, batch_size: int = 1) -> tuple:
        """Zero recurrent state. Empty tuple in feedforward mode; an ``(h, c)``
        pair of ``[num_layers, batch_size, hidden]`` tensors in LSTM mode. The
        tensors live on the module's device."""
        if not self.use_lstm:
            return ()
        device = self.actor.weight.device
        return tuple(
            torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size,
                        device=device)
            for _ in range(2)
        )

    def features(self, obs_uint8: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs_uint8)

    def get_states(self, obs_uint8: torch.Tensor, lstm_state: tuple,
                   done: torch.Tensor, last_action: torch.Tensor | None = None,
                   reward: torch.Tensor | None = None) -> tuple[torch.Tensor, tuple]:
        """Encode obs and (in LSTM mode) replay the recurrent core with
        per-step done-resets.

        Args:
            obs_uint8: ``[T*B, C, H, W]`` — a flattened sequence of T steps over
                B envs, env-major within each timestep (the natural ravel of a
                ``[T, B, ...]`` rollout buffer). In feedforward mode any
                ``[N, C, H, W]`` batch works (T*B == N, B inferred from state).
            lstm_state: ``()`` in FF mode; ``(h, c)`` with batch dim B in LSTM.
            done: ``[T*B]`` (or ``[T, B]``) float/bool; ``done[t]`` zeroes the
                state entering step t.
            last_action: ``[T*B]`` int64 — a_{t-1} aligned with each frame. Only
                consumed when ``feed_prev_action_reward``; defaults to zeros.
            reward: ``[T*B]`` float — raw r_{t-1} aligned with each frame. Only
                consumed when ``feed_prev_action_reward``; squashed with symlog.

        Returns ``(features[T*B, features_dim], new_lstm_state)``.
        """
        z = self.encoder(obs_uint8)  # [T*B, features_dim]
        if not self.use_lstm:
            return z, ()
        B = lstm_state[0].shape[1]
        if self.feed_prev_action_reward:
            # Concat one-hot(a_{t-1}) and clamped r_{t-1} onto the CNN features.
            # NOT zeroed at episode boundaries (done resets the hidden state,
            # which is what severs the cross-episode dependency — monobeast).
            tb = z.shape[0]
            if last_action is None:
                last_action = torch.zeros(tb, dtype=torch.long, device=z.device)
            if reward is None:
                reward = torch.zeros(tb, device=z.device)
            a_oh = F.one_hot(last_action.reshape(tb).long(), self.n_actions).float()
            r = reward.reshape(tb, 1).float()
            if self.symlog_reward:
                r = symlog(r)  # bounded-growth, keeps magnitude (caller feeds raw)
            z = torch.cat([z, r, a_oh], dim=-1)  # [T*B, features+1+n_actions]
        z = z.view(-1, B, self.lstm.input_size)  # [T, B, core_in]
        # done[t] zeroes the recurrent state ENTERING step t. The naive unroll is
        # a Python loop calling self.lstm once per timestep (see _unroll_steps) —
        # correct, but on GPU it's launch/sync-bound (the A100 sits ~75% idle),
        # and PPO pays it n_epochs*n_minibatches times per update. So fuse: when a
        # window has no episode boundary, run the whole sequence in ONE cuDNN
        # call; otherwise split each env at its done boundaries into contiguous
        # segments and run one fused call per segment. Both are equivalent to the
        # per-step loop (guarded by test_ppo_lstm_fused). One .cpu() sync for the
        # boundary mask; the per-step path had a sync-per-step anyway.
        reset_cpu = (done.view(-1, B) != 0).cpu().numpy()  # [T, B] bool
        if not reset_cpu.any():
            core_output, lstm_state = self.lstm(z, lstm_state)
        else:
            core_output, lstm_state = self._unroll_segmented(z, lstm_state, reset_cpu)
        z = torch.flatten(core_output, 0, 1)  # [T*B, features_dim]
        return z, lstm_state

    def _unroll_steps(self, z: torch.Tensor, lstm_state: tuple,
                      done: torch.Tensor) -> tuple[torch.Tensor, tuple]:
        """Reference per-step unroll (monobeast/CleanRL): [T,B,core_in] -> (
        [T,B,H], state); ``done[t]`` zeroes the state entering step t. This is the
        correctness ORACLE the fused path is tested against — not used in the hot
        path, kept so the equivalence guard never drifts."""
        done = done.view(z.shape[0], z.shape[1]).float()
        outputs = []
        for inp, d in zip(z.unbind(0), done.unbind(0)):
            nd = (1.0 - d).view(1, -1, 1)        # [1, B, 1] broadcast over (L,B,H)
            lstm_state = (nd * lstm_state[0], nd * lstm_state[1])
            out, lstm_state = self.lstm(inp.unsqueeze(0), lstm_state)
            outputs.append(out)
        return torch.cat(outputs), lstm_state

    def _unroll_segmented(self, z: torch.Tensor, lstm_state: tuple,
                          reset_cpu) -> tuple[torch.Tensor, tuple]:
        """Fused unroll for windows that contain episode boundaries. Per env,
        split the sequence at done positions into contiguous reset-free segments
        and run nn.LSTM ONCE per segment (state zeroed at each boundary, carried
        within a segment). Exactly equals ``_unroll_steps`` but issues one fused
        cuDNN call per segment instead of one per timestep. ``reset_cpu`` is a
        ``[T, B]`` bool ndarray (already on the host, so the inner loop does no
        GPU↔CPU sync)."""
        T, B, _ = z.shape
        H = self.lstm.hidden_size
        h0, c0 = lstm_state
        out = z.new_empty(T, B, H)
        fh, fc = h0.new_empty(h0.shape), c0.new_empty(c0.shape)
        for b in range(B):
            h = h0[:, b:b + 1].contiguous()
            c = c0[:, b:b + 1].contiguous()
            t = 0
            while t < T:
                if reset_cpu[t, b]:
                    h = torch.zeros_like(h)
                    c = torch.zeros_like(c)
                e = t + 1
                while e < T and not reset_cpu[e, b]:
                    e += 1
                seg_out, (h, c) = self.lstm(z[t:e, b:b + 1].contiguous(), (h, c))
                out[t:e, b:b + 1] = seg_out
                t = e
            fh[:, b:b + 1] = h
            fc[:, b:b + 1] = c
        return out, (fh, fc)

    def forward(self, obs_uint8: torch.Tensor) -> tuple[Categorical, torch.Tensor]:
        """Feedforward forward pass. LSTM mode must use the state-threaded
        methods (``get_action_and_value`` / ``get_value``) instead, since a bare
        obs carries no recurrent state or episode boundaries."""
        if self.use_lstm:
            raise RuntimeError(
                "ActorCritic.forward() is feedforward-only; in use_lstm mode "
                "call get_action_and_value()/get_value() with (lstm_state, done)."
            )
        z = self.encoder(obs_uint8)
        return self._dist(z), self.critic(z).squeeze(-1)

    @torch.no_grad()
    def act(self, obs_uint8: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Feedforward rollout step. Use ``act_recurrent`` in LSTM mode."""
        dist, value = self.forward(obs_uint8)
        action = dist.sample()
        return action, dist.log_prob(action), value

    def get_value(self, obs_uint8: torch.Tensor, lstm_state: tuple,
                  done: torch.Tensor, last_action: torch.Tensor | None = None,
                  reward: torch.Tensor | None = None) -> torch.Tensor:
        """Value of each step in the sequence (state-threaded). Works in both
        modes; ``lstm_state``/``done``/``last_action``/``reward`` are ignored in
        FF mode (and the latter two unless ``feed_prev_action_reward``)."""
        z, _ = self.get_states(obs_uint8, lstm_state, done, last_action, reward)
        return self.critic(z).squeeze(-1)

    def get_action_and_value(
        self,
        obs_uint8: torch.Tensor,
        lstm_state: tuple,
        done: torch.Tensor,
        action: torch.Tensor | None = None,
        last_action: torch.Tensor | None = None,
        reward: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple]:
        """State-threaded actor+critic over a sequence. Returns
        ``(action, log_prob, entropy, value, new_lstm_state)``. If ``action`` is
        given (the PPO update path) it is scored rather than resampled; otherwise
        it is sampled. ``last_action``/``reward`` are the a_{t-1}/r_{t-1} fed to
        the core when ``feed_prev_action_reward``. Works in both modes."""
        z, new_state = self.get_states(obs_uint8, lstm_state, done, last_action, reward)
        dist = self._dist(z)
        if action is None:
            action = dist.sample()
        value = self.critic(z).squeeze(-1)
        return action, dist.log_prob(action), dist.entropy(), value, new_state

    @torch.no_grad()
    def act_recurrent(
        self,
        obs_uint8: torch.Tensor,
        lstm_state: tuple,
        done: torch.Tensor,
        last_action: torch.Tensor | None = None,
        reward: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple]:
        """Single recurrent rollout step. ``obs_uint8`` is ``[B, C, H, W]``,
        ``done`` is ``[B]``; ``last_action``/``reward`` are ``[B]`` (a_{t-1}/r_{t-1}
        when ``feed_prev_action_reward``). Returns
        ``(action, log_prob, value, new_lstm_state)`` — the recurrent analogue
        of ``act``."""
        action, log_prob, _entropy, value, new_state = self.get_action_and_value(
            obs_uint8, lstm_state, done, None, last_action, reward
        )
        return action, log_prob, value, new_state
