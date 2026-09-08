"""Random Network Distillation (Burda et al. 2018) — intrinsic reward bonus.

Single-stream simplification of the original paper: intrinsic reward is added
to extrinsic reward before GAE, with one shared value head. The paper's full
two-head variant (separate intrinsic value, intrinsic discount) is a future
upgrade — we want a baseline number first.

Why RND as the *first* exploration add-on, despite known weaknesses on
procedurally-generated envs: it is the simplest possible intrinsic-reward
signal (~150 LOC), it makes a clean baseline number on sparse-reward tasks,
and the gap between RND and RIDE/E3B is itself the motivation for escalating.

Usage from train_ppo_clean.py:
    rnd = RND(in_channels=C, input_hw=H).to(device)
    rnd.build_obs_rms((C, H, W), device=device)
    rnd_opt = torch.optim.Adam(rnd.predictor.parameters(), lr=cfg.rnd_lr, eps=1e-5)
    rfilter = RewardForwardFilter(n_envs, gamma_int, device=device)

    # rollout: collect intrinsic per step
    intr[step] = rnd.intrinsic_reward(next_obs)

    # post-rollout: normalize, add to extrinsic, run GAE on the sum
    intr_norm = rfilter.update(intr)
    total_rewards = extrinsic + coef * intr_norm

    # update: train predictor on a fraction of rollout obs
    loss = rnd.predictor_loss(obs_minibatch); loss.backward(); rnd_opt.step()
"""
from __future__ import annotations

import torch
from torch import nn


# ----------------------------------------------------------------------
# Running statistics
# ----------------------------------------------------------------------
class RunningMeanStd:
    """Welford running mean/var, on-device. Matches the SB3 / Burda convention.

    Used twice: once for observation normalization (shape = obs shape, updated
    on every rollout's worth of obs), once for intrinsic-return normalization
    (scalar, updated via RewardForwardFilter).
    """
    def __init__(self, shape, device, epsilon: float = 1e-4):
        self.mean = torch.zeros(shape, device=device, dtype=torch.float32)
        self.var = torch.ones(shape, device=device, dtype=torch.float32)
        self.count = float(epsilon)

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        flat = x.reshape(-1, *self.mean.shape).float() if self.mean.ndim else x.reshape(-1).float()
        b_mean = flat.mean(dim=0)
        b_var = flat.var(dim=0, unbiased=False)
        b_count = flat.shape[0]
        delta = b_mean - self.mean
        tot = self.count + b_count
        new_mean = self.mean + delta * (b_count / tot)
        m_a = self.var * self.count
        m_b = b_var * b_count
        m2 = m_a + m_b + delta.pow(2) * (self.count * b_count / tot)
        self.mean = new_mean
        self.var = m2 / tot
        self.count = tot

    @property
    def std(self) -> torch.Tensor:
        return self.var.sqrt().clamp_min(1e-6)


# ----------------------------------------------------------------------
# RND networks
# ----------------------------------------------------------------------
class _NatureCNN(nn.Module):
    """Atari-style CNN backbone used in the Burda RND paper. Distinct from
    the IMPALA-CNN policy backbone — RND uses its own (smaller, frozen-target)
    embedding network so that the policy gradients don't feed back into the
    novelty signal."""
    def __init__(self, in_channels: int, input_hw: int, out_dim: int = 512,
                 deeper: bool = False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4), nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),          nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),          nn.LeakyReLU(),
        )
        with torch.no_grad():
            flat = self.conv(torch.zeros(1, in_channels, input_hw, input_hw)).flatten(1).shape[1]
        layers: list[nn.Module] = [nn.Flatten(), nn.Linear(flat, out_dim)]
        if deeper:
            # Predictor is deeper than target — prevents trivial fitting and
            # gives the predictor capacity to actually distill the random target.
            layers += [nn.LeakyReLU(), nn.Linear(out_dim, out_dim),
                       nn.LeakyReLU(), nn.Linear(out_dim, out_dim)]
        self.head = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.conv(x))


class RND(nn.Module):
    """Frozen random target net + trainable predictor net. Intrinsic reward
    is the squared error between their embeddings on a given obs."""
    def __init__(self, in_channels: int, input_hw: int, embed_dim: int = 512):
        super().__init__()
        self.target = _NatureCNN(in_channels, input_hw, embed_dim, deeper=False)
        self.predictor = _NatureCNN(in_channels, input_hw, embed_dim, deeper=True)
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.obs_rms: RunningMeanStd | None = None

    def build_obs_rms(self, obs_shape, device) -> None:
        self.obs_rms = RunningMeanStd(obs_shape, device=device)

    def _normalize(self, obs: torch.Tensor) -> torch.Tensor:
        x = obs.float()
        if obs.dtype == torch.uint8:
            x = x / 255.0
        if self.obs_rms is not None:
            x = (x - self.obs_rms.mean) / self.obs_rms.std
        return x.clamp(-5.0, 5.0)

    @torch.no_grad()
    def intrinsic_reward(self, obs: torch.Tensor) -> torch.Tensor:
        """Per-sample squared error, summed over the embedding dim. Shape: (B,)."""
        x = self._normalize(obs)
        return (self.predictor(x) - self.target(x)).pow(2).sum(dim=-1)

    def predictor_loss(self, obs: torch.Tensor) -> torch.Tensor:
        """MSE between predictor and (frozen) target embeddings — the training
        signal for the predictor."""
        x = self._normalize(obs)
        with torch.no_grad():
            t = self.target(x)
        return (self.predictor(x) - t).pow(2).sum(dim=-1).mean()


# ----------------------------------------------------------------------
# NovelD (Zhang et al., NeurIPS 2021)
# ----------------------------------------------------------------------
class NovelD:
    """NovelD exploration bonus, built on an RND novelty source.

        bonus_t = [ N(s_{t+1}) - alpha * N(s_t) ]_+  *  1[first episodic visit of s_{t+1}]

    where N(s) is the RND prediction error (reuses the existing ``RND`` module,
    so it slots into the same single-stream / RewardForwardFilter / predictor-
    training machinery as plain RND). The bonus only fires at the *frontier* of
    explored space (novelty increasing) and only the first time a state is seen
    in an episode — this is what lets it solve hard chained MiniGrid tasks
    (KeyCorridor) where flat RND collapses.

    Episodic first-visit is keyed on a hash of the observation. For MiniGrid the
    full-grid render (with the carrying patch) is a bijection with the state, so
    the hash is an *exact* episodic state id. On continuous/pixel envs where no
    two frames repeat, the gate becomes a no-op (every state 'novel') and NovelD
    degrades to a pure RND novelty-difference — still valid, just ungated.

    Lives on the trainer side; call once per rollout step *after* env.step():
        nd.reset(N(obs0), obs0_np)                # before the rollout
        bonus = nd.bonus(N(next_obs), next_obs_np, done_np)   # each step
    """

    def __init__(self, n_envs: int, device, alpha: float = 0.5):
        self.n_envs = n_envs
        self.device = device
        self.alpha = alpha
        self.nov_prev = torch.zeros(n_envs, device=device)
        self.visited: list[set[int]] = [set() for _ in range(n_envs)]

    @staticmethod
    def _keys(obs_np) -> list[int]:
        # obs_np: (N, H, W, C) uint8 — hash each env's frame to an exact state id
        return [hash(obs_np[i].tobytes()) for i in range(obs_np.shape[0])]

    def reset(self, novelty_init: torch.Tensor, obs_init_np) -> None:
        self.nov_prev = novelty_init.detach().clone()
        keys = self._keys(obs_init_np)
        for i in range(self.n_envs):
            self.visited[i] = {keys[i]}

    @torch.no_grad()
    def bonus(self, novelty_next: torch.Tensor, obs_next_np, done_np) -> torch.Tensor:
        keys = self._keys(obs_next_np)
        first_visit = torch.zeros(self.n_envs, device=self.device)
        for i in range(self.n_envs):
            if keys[i] not in self.visited[i]:
                first_visit[i] = 1.0
                self.visited[i].add(keys[i])
        done = torch.as_tensor(done_np, dtype=torch.float32, device=self.device)
        # zero across episode boundaries — s_{t+1} is then a fresh reset obs
        b = torch.clamp(novelty_next - self.alpha * self.nov_prev, min=0.0)
        b = b * first_visit * (1.0 - done)
        # advance novelty; on done, s_{t+1} starts the new episode's memory
        self.nov_prev = novelty_next.detach().clone()
        for i in range(self.n_envs):
            if bool(done_np[i]):
                self.visited[i] = {keys[i]}
        return b


# ----------------------------------------------------------------------
# Reward normalization
# ----------------------------------------------------------------------
class RewardForwardFilter:
    """Tracks discounted forward returns of the intrinsic stream and divides
    raw intrinsic rewards by their running std. Matches the Burda RND code:
    https://github.com/openai/random-network-distillation/blob/master/utils.py

    Without this normalization the intrinsic-reward magnitude drifts (typically
    upward early in training as the predictor fails on novel obs, then downward
    as it learns) and the coef hyperparam becomes uninterpretable.
    """
    def __init__(self, n_envs: int, gamma: float, device):
        self.gamma = gamma
        self.rewems = torch.zeros(n_envs, device=device)
        self.ret_rms = RunningMeanStd((), device=device)

    @torch.no_grad()
    def update(self, intr_rewards: torch.Tensor) -> torch.Tensor:
        """intr_rewards: (T, N). Returns intr_rewards / running_std(returns)."""
        T = intr_rewards.shape[0]
        forward_returns = torch.zeros_like(intr_rewards)
        rew = self.rewems
        for t in range(T):
            rew = rew * self.gamma + intr_rewards[t]
            forward_returns[t] = rew
        self.rewems = rew
        self.ret_rms.update(forward_returns.flatten())
        return intr_rewards / self.ret_rms.std
