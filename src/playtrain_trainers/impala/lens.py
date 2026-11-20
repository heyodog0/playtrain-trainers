"""Fast-weight lens: an episodic key-value memory INSIDE the encoder, keyed on
local appearance.

Every earlier memory core in this trainer (the LSTM, DeltaNet, CompFWP) sits
after the CNN's flatten, so its keys and queries are projections of one
whole-frame vector. A sprite never has a key of its own there: the same item at
another tile, or next to another item, is a different key. The lens moves the
memory to where sprites still are separate things, the feature map.

    key map    k[x,y]  = local appearance relative to the rest of the frame: a
                         shallow stem whose receptive field is about one tile
                         (10 px, stride 4), standardised over locations
    memory     M       = one (role_dim x key_dim) matrix per env, zero at reset
    read       R[x,y]  = M k[x,y]            at EVERY location, every step
    write      a few heads each point at one location (spatial softmax over the
               current and the previous key map), take the appearance key found
               there and store a value computed from the wide-context features:
               M <- M + beta (v - M k_w) k_w^T          (delta rule, unit key)

The read is concatenated to the stage-1 feature map, so the rest of the CNN,
the LSTM and the heads see the frame with every item painted with whatever the
memory holds for things that look like it. The read depends on appearance only
through similarities to keys written earlier in this episode, never through
slow weights: ``R[x,y] = sum_w beta_w v_w (k_w . k[x,y])``. A sprite the slow
weights have never met gets a usable key as long as the stem keeps it distinct.

The previous key map is a write candidate because a pickup can remove the icon
it is about: the item was on the tile one step ago and is a body transform now.

Only the matrix updates are sequential. Keys, write proposals, reads and all
convolutions are computed for the whole unroll at once; the scan is T x heads
rank-1 updates of a small matrix, in fp32 whatever the autocast precision.

State contract (appended after the LSTM's (h, c)): two tensors, batch at dim 1,
``M`` as [1, B, role_dim * key_dim] and the previous key map as
[1, B, key_dim * G * G].
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from playtrain_trainers.policy import _ImpalaStage

#: Encoder names (the trainer's ``net`` option) that build a lens encoder.
LENS_VARIANTS: dict[str, dict] = {
    "impala_lens": {},
    # Control: identical parameters and compute, but nothing is ever written,
    # so the read is zero. Separates the memory from the extra capacity.
    "impala_lens_nowrite": {"write": False},
    # The key stem stays at its random initialisation: keys are generic local
    # descriptors that cannot specialise to the training sprites.
    "impala_lens_frozenkey": {"freeze_keys": True},
}


@torch.compiler.disable
def lens_scan(M, kw, v, beta, notdone):
    """Delta-rule writes over an unroll. Kept out of torch.compile like the
    segmented LSTM: T x heads tiny ops, nothing for inductor to fuse.

    M [B, R, K]; kw [T, B, H, K] unit keys; v [T, B, H, R]; beta [T, B, H];
    notdone [T, B]. Returns the memory after each step's writes, [T, B, R, K].
    """
    T, _, H, _ = kw.shape
    out = []
    for t in range(T):
        M = M * notdone[t].view(-1, 1, 1)
        for h in range(H):
            k = kw[t, :, h]
            err = v[t, :, h] - torch.einsum("brk,bk->br", M, k)
            M = M + (beta[t, :, h].unsqueeze(-1) * err).unsqueeze(-1) * k.unsqueeze(1)
        out.append(M)
    return torch.stack(out)


class LensEncoder(nn.Module):
    """ImpalaCNN with a fast-weight lens between stage 1 and stage 2."""

    def __init__(self, in_channels: int = 3, features_dim: int = 256,
                 input_hw: int = 64, depths: tuple[int, int, int] = (16, 32, 32),
                 key_dim: int = 32, role_dim: int = 16, heads: int = 4,
                 write: bool = True, freeze_keys: bool = False) -> None:
        super().__init__()
        if input_hw % 4:
            raise ValueError(f"lens encoder needs input_hw divisible by 4, got {input_hw}")
        self.key_dim, self.role_dim, self.heads = key_dim, role_dim, heads
        self.write = write
        self.s0 = _ImpalaStage(in_channels, depths[0])
        self.s1 = _ImpalaStage(depths[0], depths[1])
        self.s2 = _ImpalaStage(depths[1] + role_dim, depths[2])
        # Receptive field 10 px at stride 4: about one 8 px tile. The max-pools
        # give the key some tolerance to sub-tile position and to the
        # nearest-neighbour aliasing of sprites in the continuous-motion games.
        self.key_stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, key_dim, 1),
        )
        if freeze_keys:
            for p in self.key_stem.parameters():
                p.requires_grad_(False)
        self.write_trunk = nn.Sequential(
            nn.Conv2d(depths[1] + 3 * key_dim, 64, 1), nn.ReLU())
        self.write_attn = nn.Conv2d(64, 2 * heads, 1)   # per head: current map, previous map
        self.write_value = nn.Conv2d(64, role_dim, 1)
        self.write_beta = nn.Conv2d(64, heads, 1)
        nn.init.constant_(self.write_beta.bias, -2.0)   # beta 0.12: writes start small
        with torch.no_grad():
            g = self.s1(self.s0(torch.zeros(1, in_channels, input_hw, input_hw))).shape[-1]
            flat = depths[2] * (g // 2 + g % 2) ** 2
        self.grid = g
        self.fc = nn.Linear(flat, features_dim)
        self.features_dim = features_dim
        # Diagnostics of the last forward (detached, non-persistent so the
        # state_dict and the actor weight sync are unaffected).
        self.register_buffer("last_beta", torch.zeros(()), persistent=False)
        self.register_buffer("last_read", torch.zeros(()), persistent=False)

    @property
    def state_sizes(self) -> tuple[int, int]:
        return (self.role_dim * self.key_dim, self.key_dim * self.grid * self.grid)

    def initial_state(self, batch_size: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
        return tuple(torch.zeros(1, batch_size, n) for n in self.state_sizes)

    @staticmethod
    def normalise_keys(k: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        """Raw key map [N, K, G, G] -> keys that describe how a location
        differs from the rest of its frame.

        ReLU features of one mostly-background frame share a large common
        component: measured on cavequest_hard_pool36 at initialisation, plain
        unit-norm keys of DIFFERENT sprites have cosine 0.98, so every read
        would return the same thing everywhere. Each channel is therefore
        standardised over the frame's locations first (different sprites drop
        to ~0.5, the same sprite on another tile stays at ~0.86), then scaled
        by a soft norm, ``c / sqrt(|c|^2 + mean_xy |c|^2)``: salient locations
        approach unit length and typical ones shrink, with every key's norm
        below 1 so reads stay bounded.
        """
        c = k - k.mean(dim=(2, 3), keepdim=True)
        c = c / (c.std(dim=(2, 3), keepdim=True) + eps)
        n2 = c.pow(2).sum(dim=1, keepdim=True)
        return c / torch.sqrt(n2 + n2.mean(dim=(2, 3), keepdim=True) + eps)

    def proposals(self, d, kc, kp):
        """Write heads for every step at once.

        d [N, C, G, G] context; kc, kp [N, K, G, G] current / previous keys.
        Returns kw [N, H, K] unit write keys, v [N, H, R], beta [N, H].
        """
        N, K, G, _ = kc.shape
        H, n = self.heads, G * G
        w = self.write_trunk(torch.cat([d, kc.to(d.dtype), kp.to(d.dtype),
                                        (kc - kp).to(d.dtype)], dim=1))
        logits = self.write_attn(w).float().reshape(N, 2, H, n)
        a = torch.softmax(torch.cat([logits[:, 0], logits[:, 1]], dim=-1), dim=-1)  # [N, H, 2n]
        cand = torch.cat([kc.reshape(N, K, n), kp.reshape(N, K, n)], dim=-1)        # [N, K, 2n]
        kw = F.normalize(torch.einsum("nhs,nks->nhk", a, cand), dim=-1)
        a_sp = a[..., :n] + a[..., n:]                                               # [N, H, n]
        v = torch.tanh(torch.einsum(
            "nhs,nrs->nhr", a_sp, self.write_value(w).float().reshape(N, self.role_dim, n)))
        beta = torch.sigmoid(torch.einsum(
            "nhs,nhs->nh", a_sp, self.write_beta(w).float().reshape(N, H, n)))
        return kw, v, beta

    def forward(self, obs_uint8, T: int, B: int, notdone, state):
        """obs [T*B, C, H, W] uint8; notdone [T, B] float; state (M, kprev).
        Returns (features [T*B, F], new state)."""
        K, R, G = self.key_dim, self.role_dim, self.grid
        x = obs_uint8.float() / 255.0
        d = self.s1(self.s0(x))                                   # [TB, C1, G, G]
        k = self.key_stem(x)
        dev = x.device.type
        with torch.autocast(device_type=dev, enabled=False):
            nd = notdone.float()
            kc = self.normalise_keys(k.float()).reshape(T, B, K, G, G)
            kp = torch.cat([state[1].float().reshape(1, B, K, G, G), kc[:-1]], dim=0)
            kp = kp * nd.view(T, B, 1, 1, 1)
        kw, v, beta = self.proposals(d, kc.flatten(0, 1), kp.flatten(0, 1))
        with torch.autocast(device_type=dev, enabled=False):
            if not self.write:
                beta = beta * 0.0
            Ms = lens_scan(state[0].float().reshape(B, R, K),
                           kw.reshape(T, B, self.heads, K), v.reshape(T, B, self.heads, R),
                           beta.reshape(T, B, self.heads), nd)
            read = torch.einsum("tbrk,tbks->tbrs", Ms, kc.flatten(3)).reshape(T * B, R, G, G)
            self.last_beta.copy_(beta.detach().mean())
            self.last_read.copy_(read.detach().abs().mean())
            new_state = (Ms[-1].reshape(1, B, R * K), kc[-1].reshape(1, B, K * G * G))
        z = self.s2(torch.cat([d, read.to(d.dtype)], dim=1))
        z = F.relu(z).flatten(1)
        return F.relu(self.fc(z)), new_state


def build_lens_encoder(net: str, in_channels: int, features_dim: int,
                       input_hw: int) -> LensEncoder:
    return LensEncoder(in_channels, features_dim, input_hw=input_hw, **LENS_VARIANTS[net])
