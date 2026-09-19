"""Fast-weight cores for ImpalaNet: a matrix state written by the delta rule.

The state is a square matrix S per env, flattened to a single
``[1, B, fwp_dim*fwp_dim]`` tensor and returned as a 1-tuple, so the buffers,
actors and learner — which assume only "a tuple of tensors with batch at dim 1"
— need no changes. At fwp_dim=128 that is 64 KB per env in fp32, so 64 envs
carry about 4 MB.

Both cores read with M input-dependent queries produced from the timestep's
features. Nothing here segments entities or knows anything about the task: the
queries are projections of the CNN features, and M is a width, not a count of
things in the world.

The unroll is a per-step Python scan that multiplies the state by ``notdone``
at every step. That is the same reset the LSTM path gets from segmenting at
episode boundaries — between boundaries the mask is all ones and the multiply
is an exact no-op — but expressed without the data-dependent split points that
force a host sync, so the scan stays compilable.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class FastWeightCore(nn.Module):
    """Shared machinery: projections, the scan, and the state contract.

    Subclasses differ only in :meth:`write` (what error the delta rule stores)
    and :meth:`read` (whether the retrieved rows see each other).
    """

    kind = "fastweight"

    def __init__(self, features_dim: int, fwp_dim: int = 128, n_heads: int = 8):
        super().__init__()
        if fwp_dim < 1 or n_heads < 1:
            raise ValueError("fwp_dim and n_heads must be positive")
        self.features_dim = features_dim
        self.fwp_dim = fwp_dim
        self.n_heads = n_heads

        self.W_k = nn.Linear(features_dim, fwp_dim, bias=False)
        self.W_v = nn.Linear(features_dim, fwp_dim, bias=False)
        self.W_q = nn.Linear(features_dim, n_heads * fwp_dim, bias=False)
        self.w_b = nn.Linear(features_dim, 1)
        self.W_o = nn.Linear(n_heads * fwp_dim, features_dim)
        # Small output gain: the core starts close to a pass-through, so
        # swapping it in does not shove a freshly initialised net around, but
        # gradients still reach the query and write projections from step one.
        nn.init.orthogonal_(self.W_o.weight, gain=0.1)
        nn.init.zeros_(self.W_o.bias)

    @property
    def state_size(self) -> int:
        return self.fwp_dim * self.fwp_dim

    def initial_state(self, batch_size: int = 1) -> tuple[torch.Tensor]:
        """Zero matrix state, flattened, batch at dim 1 — a 1-tuple."""
        return (torch.zeros(1, batch_size, self.state_size),)

    # -- the two things a subclass changes ---------------------------------

    def write(self, state, k, v, beta, rows):
        raise NotImplementedError

    def read(self, state, q):
        """Retrieve the M query rows. Independent by default."""
        return torch.einsum("bij,bmj->bmi", state, q)

    # -- one timestep and the unroll ---------------------------------------

    def step(self, x: torch.Tensor, state: torch.Tensor):
        """One timestep: write, then read the updated state."""
        k = F.normalize(self.W_k(x), dim=-1)
        v = self.W_v(x)
        beta = torch.sigmoid(self.w_b(x))
        q = self.W_q(x).view(x.shape[0], self.n_heads, self.fwp_dim)
        state = self.write(state, k, v, beta, self.read(state, q))
        rows = self.read(state, q)
        return x + self.W_o(rows.flatten(1)), state

    def forward(self, core_input: torch.Tensor, notdone: torch.Tensor, core_state):
        """core_input [T, B, F], notdone [T, B] float -> ([T*B, F], state)."""
        T, B, _ = core_input.shape
        state = core_state[0].reshape(B, self.fwp_dim, self.fwp_dim)
        outs = []
        for t in range(T):
            state = state * notdone[t].view(B, 1, 1)
            out, state = self.step(core_input[t], state)
            outs.append(out)
        flat = torch.flatten(torch.stack(outs), 0, 1)
        return flat, (state.reshape(1, B, self.state_size),)


class DeltaNetCore(FastWeightCore):
    """The delta rule with an independent read: S <- S + beta (v - S k) k^T.

    The error is measured at the write key alone, so nothing the other query
    rows retrieved influences what gets stored. This is the baseline the
    competitive core has to beat.
    """

    kind = "deltanet"

    def write(self, state, k, v, beta, rows):
        del rows  # the independent error ignores what the read returned
        pred = torch.einsum("bij,bj->bi", state, k)
        return state + beta.unsqueeze(-1) * (v - pred).unsqueeze(-1) * k.unsqueeze(-2)


CORE_CLASSES: dict[str, type[FastWeightCore]] = {
    DeltaNetCore.kind: DeltaNetCore,
}


def build_fwp_core(kind: str, features_dim: int, fwp_dim: int, n_heads: int):
    if kind not in CORE_CLASSES:
        raise NotImplementedError(
            f"core={kind!r} is accepted by the config but not yet built"
        )
    return CORE_CLASSES[kind](features_dim, fwp_dim=fwp_dim, n_heads=n_heads)
