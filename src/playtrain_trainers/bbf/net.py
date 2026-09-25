"""BBF network: Impala-CNN encoder, dueling C51 heads, SPR transition model.

What is SPECIFIED by the material this loop holds (PROTOCOL sections 1-2, the
paper text, BBF.gin) and what is RECONSTRUCTED are kept apart, because the
report's credibility rests on the distinction:

SPECIFIED
  encoder      Impala-CNN, "a 15-layer ResNet" (paper section "Base agent"),
               `num_blocks = 2` per stage, every layer's width scaled 4x, so
               the ProcGen depths (16, 32, 32) become (64, 128, 128)
  renormalize  True
  hidden_dim   2048 dense layer before the dueling heads
  heads        dueling, C51 with 51 atoms, no noisy nets
  SPR          `spr_weight = 5`, `jumps = 5`, an EMA target encoder

RECONSTRUCTED (D-019, D-020)
  the exact form of `renormalize`, and the shapes of the transition model,
  projection and prediction head. The gin only switches these on, and the
  paper defers them to SPR (Schwarzer et al. 2021) / SR-SPR, neither of which
  is in `reference/`. They follow the SPR design as published; PROTOCOL
  section 5 A (the ALE sanity arm) is what tests whether the reconstruction
  is good enough to be called BBF.

Module names matter: U07's shrink-and-perturb targets the gin's
`shrink_perturb_keys = "encoder,transition_model"` by attribute name, so
``encoder`` and ``transition_model`` must keep those names.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from playtrain_trainers.bbf.config import BBFConfig

# ProcGen/Impala stage depths at width_scale 1 (Espeholt et al. 2018).
BASE_DEPTHS = (16, 32, 32)


def renormalize(x: torch.Tensor) -> torch.Tensor:
    """Per-sample min-max rescale of a latent to [0, 1] (D-019).

    This is SPR's `renormalize`: flatten everything but the batch axis, then
    map each sample's own min and max onto 0 and 1. PROTOCOL section 1
    describes it as "layer-norm style", which is a paraphrase -- it shares
    layer norm's per-sample, all-features scope but not its mean/variance
    form. The gin only says `renormalize = True`.

    The 1e-5 floor keeps a constant latent (max == min) finite; without it the
    first forward of a freshly reset network can divide by zero.
    """
    flat = x.flatten(1)
    lo = flat.min(dim=-1, keepdim=True).values
    hi = flat.max(dim=-1, keepdim=True).values
    return ((flat - lo) / (hi - lo + 1e-5)).view_as(x)


class ResidualBlock(nn.Module):
    """Impala residual block: pre-activation, two 3x3 convs, identity skip."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv2(F.relu(self.conv1(F.relu(x))))
        return x + h


class ImpalaStage(nn.Module):
    """conv 3x3 -> maxpool 3x3 stride 2 -> `num_blocks` residual blocks."""

    def __init__(self, in_c: int, out_c: int, num_blocks: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, 3, 1, 1)
        self.pool = nn.MaxPool2d(3, 2, 1)
        self.blocks = nn.Sequential(*[ResidualBlock(out_c) for _ in range(num_blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.pool(self.conv(x)))


class BBFEncoder(nn.Module):
    """Impala-CNN at `width_scale`, ending in a renormalized spatial latent.

    Output is the (C, H, W) latent, NOT a flat feature vector: the transition
    model predicts in this spatial space, so flattening happens later, in the
    projection.

    At the protocol's 84x84 input the three stride-2 pools give 84 -> 42 -> 21
    -> 11, so the latent is (128, 11, 11) at width_scale 4.
    """

    def __init__(
        self,
        in_channels: int,
        width_scale: int = 4,
        num_blocks: int = 2,
        do_renormalize: bool = True,
    ) -> None:
        super().__init__()
        self.depths = tuple(d * width_scale for d in BASE_DEPTHS)
        self.do_renormalize = do_renormalize
        stages, c = [], in_channels
        for d in self.depths:
            stages.append(ImpalaStage(c, d, num_blocks))
            c = d
        self.stages = nn.Sequential(*stages)
        self.out_channels = c

    def forward(self, obs_uint8: torch.Tensor) -> torch.Tensor:
        x = obs_uint8.float() / 255.0 if obs_uint8.dtype == torch.uint8 else obs_uint8
        x = F.relu(self.stages(x))
        return renormalize(x) if self.do_renormalize else x

    def latent_shape(self, obs_shape: tuple[int, int, int]) -> tuple[int, int, int]:
        """Latent (C, H, W) for an observation shape, by a dry forward."""
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                dev = next(self.parameters()).device
                out = self(torch.zeros(1, *obs_shape, device=dev))
        finally:
            self.train(was_training)
        return tuple(out.shape[1:])  # type: ignore[return-value]


class TransitionModel(nn.Module):
    """One-step latent dynamics for the SPR loss (D-020, reconstructed).

    The action is one-hot encoded and broadcast over the latent's spatial
    extent, concatenated onto the channel axis, then two 3x3 convs bring it
    back to the latent's own channel count so the model can be applied
    repeatedly for the `jumps` rollout. The output is renormalized, matching
    what the encoder hands out, so predicted and encoded latents live on the
    same scale -- without that the cosine loss would be comparing differently
    scaled spaces.
    """

    def __init__(self, channels: int, num_actions: int, do_renormalize: bool = True) -> None:
        super().__init__()
        self.num_actions = num_actions
        self.do_renormalize = do_renormalize
        self.conv1 = nn.Conv2d(channels + num_actions, channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)

    def forward(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        b, _, h, w = latent.shape
        onehot = F.one_hot(action.long().view(b), self.num_actions).to(latent.dtype)
        plane = onehot.view(b, self.num_actions, 1, 1).expand(b, self.num_actions, h, w)
        x = F.relu(self.conv1(torch.cat([latent, plane], dim=1)))
        x = F.relu(self.conv2(x))
        return renormalize(x) if self.do_renormalize else x


class BBFNetwork(nn.Module):
    """The whole agent network.

    Layout, and why the projection is shared:

        obs -> encoder -> latent (C,H,W)
                            |-> flatten -> projection(2048) -> relu -> dueling C51
                            |-> flatten -> projection(2048) ------> predictor -> SPR

    SPR projects through the SAME dense layer the Q-head uses, so the
    self-predictive loss shapes the representation the Q-head reads rather than
    a private one beside it. The predictor is online-only: the EMA target
    encodes and projects but does not predict, which is what breaks the
    symmetry and stops the loss collapsing to a constant.
    """

    def __init__(self, cfg: BBFConfig, num_actions: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_actions = num_actions
        self.num_atoms = cfg.num_atoms
        self.obs_shape = cfg.obs_shape

        self.encoder = BBFEncoder(
            in_channels=cfg.obs_channels,
            width_scale=cfg.width_scale,
            num_blocks=cfg.num_blocks,
            do_renormalize=cfg.renormalize,
        )
        c, h, w = self.encoder.latent_shape(cfg.obs_shape)
        self.latent_shape = (c, h, w)
        self.latent_dim = c * h * w

        # The gin's hidden_dim = 2048 dense layer, and SPR's projection.
        self.projection = nn.Linear(self.latent_dim, cfg.hidden_dim)

        # Dueling C51: a value stream over atoms and an advantage stream over
        # (action, atom). Combined on the LOGITS, before the softmax.
        self.value_head = nn.Linear(cfg.hidden_dim, cfg.num_atoms)
        self.advantage_head = nn.Linear(cfg.hidden_dim, num_actions * cfg.num_atoms)

        self.transition_model = TransitionModel(
            channels=c, num_actions=num_actions, do_renormalize=cfg.renormalize
        )
        # SPR predictor: a SINGLE linear layer, online path only.
        # D-031: an earlier reconstruction used
        # Linear -> LayerNorm -> ReLU -> Linear, which is ~2x the parameters
        # and adds a nonlinearity BBF does not have. The official
        # `spr_networks.py` has `self.predictor = nn.Dense(hidden_dim)`.
        self.predictor = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)

        # The C51 support. A buffer, so it follows the module across devices
        # and lands in the state_dict beside the weights that assume it.
        self.register_buffer(
            "support", torch.linspace(cfg.v_min, cfg.v_max, cfg.num_atoms)
        )
        # D-037: the official network passes `nn.initializers.xavier_uniform()`
        # as `kernel_init` to every Conv and Dense, and Flax's default bias
        # init is zeros. Torch's defaults (kaiming_uniform with a = sqrt(5),
        # uniform biases) are ~1.7x narrower for these layers, which changes
        # the scale of every shrink-and-perturb reset.
        self.apply(init_flax_style_)

    # ------------------------------------------------------------------
    # Forward paths
    # ------------------------------------------------------------------
    def project(self, latent: torch.Tensor) -> torch.Tensor:
        """Flatten a spatial latent and run the shared projection."""
        return self.projection(latent.flatten(1))

    def heads(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Dueling C51 from projected features.

        Returns ``(logits, q)`` with logits ``[B, A, atoms]`` and q ``[B, A]``.
        """
        h = F.relu(features)
        value = self.value_head(h).view(-1, 1, self.num_atoms)
        adv = self.advantage_head(h).view(-1, self.num_actions, self.num_atoms)
        # Dueling: the advantage stream is identified only up to a per-atom
        # constant, so its mean over actions is removed.
        logits = value + adv - adv.mean(dim=1, keepdim=True)
        probs = F.softmax(logits, dim=-1)
        q = (probs * self.support).sum(dim=-1)
        return logits, q

    def forward(self, obs: torch.Tensor) -> dict[str, torch.Tensor]:
        """Encode and score one batch of observations.

        ``obs`` is ``[B, C, H, W]`` uint8 (or float already in [0, 1]).
        """
        latent = self.encoder(obs)
        features = self.project(latent)
        logits, q = self.heads(features)
        return {
            "latent": latent,
            "features": features,
            "logits": logits,          # [B, A, atoms], pre-softmax
            "log_probs": F.log_softmax(logits, dim=-1),
            "probs": F.softmax(logits, dim=-1),
            "q": q,                    # [B, A]
        }

    def q_values(self, obs: torch.Tensor) -> torch.Tensor:
        return self.forward(obs)["q"]

    def spr_predictions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """Roll the transition model forward and predict future projections.

        ``obs`` is ``[B, C, H, W]`` (the window's first observation) and
        ``actions`` is ``[B, K]``, the K actions taken from it. Returns
        ``[B, K, hidden_dim]``: the predictor's output after 1..K latent steps.
        The k-th entry is compared against the EMA target's projection of the
        real observation k steps later (``target_projections``).
        """
        if actions.dim() != 2:
            raise ValueError(f"actions must be [B, K], got {tuple(actions.shape)}")
        latent = self.encoder(obs)
        outs = []
        for k in range(actions.shape[1]):
            latent = self.transition_model(latent, actions[:, k])
            outs.append(self.predictor(self.project(latent)))
        return torch.stack(outs, dim=1)

    def forward_with_spr(
        self, obs: torch.Tensor, spr_actions: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Q values and SPR predictions from ONE encoder pass.

        `forward` and `spr_predictions` each encode the observation, and the
        training step needs both -- so calling them separately doubles the
        encoder cost of every update. Here the latent is computed once and
        both heads read it.
        """
        if spr_actions.dim() != 2:
            raise ValueError(f"spr_actions must be [B, K], got {tuple(spr_actions.shape)}")
        latent = self.encoder(obs)
        features = self.project(latent)
        logits, q = self.heads(features)
        preds = []
        rolled = latent
        for k in range(spr_actions.shape[1]):
            rolled = self.transition_model(rolled, spr_actions[:, k])
            preds.append(self.predictor(self.project(rolled)))
        return {
            "latent": latent,
            "features": features,
            "logits": logits,
            "log_probs": F.log_softmax(logits, dim=-1),
            "probs": F.softmax(logits, dim=-1),
            "q": q,
            "spr_predictions": torch.stack(preds, dim=1),
        }

    @torch.no_grad()
    def target_projections(self, obs: torch.Tensor) -> torch.Tensor:
        """The SPR target: encode and project, no predictor, no gradient.

        Called on the EMA copy of the network. Kept here rather than in the
        loss so the online and target paths provably share one projection
        implementation.
        """
        return self.project(self.encoder(obs))

    # ------------------------------------------------------------------
    # Bookkeeping
    # ------------------------------------------------------------------
    def parameter_counts(self) -> dict[str, int]:
        """Trainable parameters per component, for the record (U04)."""
        groups = {
            "encoder": self.encoder,
            "projection": self.projection,
            "value_head": self.value_head,
            "advantage_head": self.advantage_head,
            "transition_model": self.transition_model,
            "predictor": self.predictor,
        }
        counts = {
            name: sum(p.numel() for p in mod.parameters() if p.requires_grad)
            for name, mod in groups.items()
        }
        counts["total"] = sum(p.numel() for p in self.parameters() if p.requires_grad)
        # Every parameter must be attributed, or the reset unit (U07) could
        # silently leave a component untouched.
        assert counts["total"] == sum(v for k, v in counts.items() if k != "total"), (
            f"parameter groups do not partition the model: {counts}"
        )
        return counts


@torch.no_grad()
def init_flax_style_(module: nn.Module) -> None:
    """Xavier-uniform kernels and zero biases, as the official Flax network."""
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def resolve_device(spec: str = "auto") -> torch.device:
    """`auto` picks cuda, then mps, then cpu."""
    if spec != "auto":
        return torch.device(spec)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_network(cfg: BBFConfig, num_actions: int, device: str | torch.device | None = None):
    """Construct the network on the resolved device."""
    dev = resolve_device(cfg.device) if device is None else torch.device(device)
    return BBFNetwork(cfg, num_actions).to(dev), dev
