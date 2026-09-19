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


#: Cores whose state is a flattened matrix rather than an (h, c) pair.
FWP_CORES = ("deltanet", "compfwp")


def matrix_state_norms(core_state) -> tuple[torch.Tensor, torch.Tensor] | None:
    """(mean, max) per-env Frobenius norm of a flattened matrix state.

    Returns detached tensors and never touches the host, so the caller decides
    when to pay the sync — the learner logs on a cadence and a per-step .item()
    here would stall the CUDA stream and starve the inference thread.

    None when there is no matrix state to measure.
    """
    if not core_state:
        return None
    flat = core_state[0]  # [1, B, fwp_dim * fwp_dim]
    per_env = flat.reshape(-1, flat.shape[-1]).norm(dim=-1)
    return per_env.mean().detach(), per_env.max().detach()


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

    def write(self, state, k, v, beta, q):
        raise NotImplementedError

    def read(self, state, q, k=None):
        """Retrieve the M query rows. Independent by default, so `k` is unused."""
        del k
        return torch.einsum("bij,bmj->bmi", state, q)

    @staticmethod
    def retrieve(state, q):
        """Raw matrix read, before any competition between the rows."""
        return torch.einsum("bij,bmj->bmi", state, q)

    # -- one timestep and the unroll ---------------------------------------

    def step(self, x: torch.Tensor, state: torch.Tensor):
        """One timestep: write, then read the updated state."""
        k = F.normalize(self.W_k(x), dim=-1)
        v = self.W_v(x)
        beta = torch.sigmoid(self.w_b(x))
        q = self.W_q(x).view(x.shape[0], self.n_heads, self.fwp_dim)
        state = self.write(state, k, v, beta, q)
        rows = self.read(state, q, k)
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

    def write(self, state, k, v, beta, q):
        del q  # the independent error never looks at the other query rows
        pred = torch.einsum("bij,bj->bi", state, k)
        return state + beta.unsqueeze(-1) * (v - pred).unsqueeze(-1) * k.unsqueeze(-2)


class SetBlock(nn.Module):
    """One permutation-equivariant self-attention layer across the query rows.

    No positional information of any kind, so the block cannot tell row 0 from
    row 7 except by content. This is where the queries compete.
    """

    def __init__(self, dim: int, n_heads: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim)
        )

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        h = self.norm1(rows)
        rows = rows + self.attn(h, h, h, need_weights=False)[0]
        return rows + self.mlp(self.norm2(rows))


class CompFWPCore(FastWeightCore):
    """Competitive read, surprise-gated write (review §12), in the trainer.

    The M retrieved rows pass through a set block so they can suppress one
    another, and the delta-rule error is measured *after* that joint read:
    whatever the competitive read already predicted is not written again.

    **The write key rides along as an extra query row.** Stage 0 could name the
    row to predict, because each query was a known item and the token said
    which one it was about. Here the queries are projections of the features
    and nothing names anything, so to have "the joint read's prediction for the
    written key" actually be a joint read *at that key*, k is appended as an
    (M+1)-th row. The block therefore sees M+1 rows; the readout uses the first
    M and the write error uses the last. Pooling the M rows instead would give
    a prediction *from* the competitive read but not *at* the key, which is the
    coupling the contribution rests on.

    **The joint term is a correction, not a replacement.** The error is
    ``v - (S k + W_p(joint_row - S k))``, not ``v - W_p(joint_row)``. This is
    what keeps the state bounded, and it is structural rather than a tuning
    choice. The delta rule is contractive along the write key because it
    measures what the state already holds there: the component along k goes to
    ``(1-beta) S k + beta v``. Writing ``W_p(joint_row)`` instead puts ``W_p S
    k`` inside the error, making that component ``(I - beta W_p) S k``, which
    grows without bound whenever W_p has spectral radius above 1 — measured at
    ||S|| = 1.9e8 by step 3000, against DeltaNet's plateau near 105. Because
    the set block is a residual over LayerNorm'd inputs, ``joint_row - S k`` is
    O(1) no matter how large S is, so subtracting it perturbs the contraction
    without removing it.

    W_p is zero-initialised, so at initialisation the write path is exactly the
    delta rule and only the read differs. Competition then learns how far to
    move the error away from it.

    The flags are the review's ablation table. ``read="indep"`` drops the set
    block, ``error="indep"`` measures the error at the write key with a plain
    independent read, and ``write="additive"`` drops the error term. With
    indep/indep/delta this core is DeltaNet exactly, which the tests check.
    """

    kind = "compfwp"

    def __init__(
        self,
        features_dim: int,
        fwp_dim: int = 128,
        n_heads: int = 8,
        read: str = "joint",
        error: str = "joint",
        write: str = "delta",
        attn_heads: int = 4,
    ):
        super().__init__(features_dim, fwp_dim=fwp_dim, n_heads=n_heads)
        if read not in ("joint", "indep"):
            raise ValueError(f"read must be joint|indep, got {read!r}")
        if error not in ("joint", "indep"):
            raise ValueError(f"error must be joint|indep, got {error!r}")
        if write not in ("delta", "additive"):
            raise ValueError(f"write must be delta|additive, got {write!r}")
        if error == "joint" and read == "indep":
            raise ValueError("error='joint' needs read='joint': there is no joint read to use")
        self.read_mode, self.error_mode, self.write_mode = read, error, write
        self.set_block = SetBlock(fwp_dim, attn_heads) if read == "joint" else None
        self.W_p = nn.Linear(fwp_dim, fwp_dim, bias=False) if error == "joint" else None
        if self.W_p is not None:
            # Zero: the write path starts as the plain delta rule, so the
            # contraction is exact at initialisation and competition has to
            # earn any departure from it. Gradients still flow, because the
            # error depends on W_p through a generally non-zero difference.
            nn.init.zeros_(self.W_p.weight)

    @property
    def variant(self) -> str:
        return f"{self.read_mode}/{self.error_mode}/{self.write_mode}"

    def _compete(self, state, q, k):
        """Retrieve the M query rows plus the write key, then let them compete."""
        rows = self.retrieve(state, torch.cat([q, k.unsqueeze(1)], dim=1))
        return self.set_block(rows)

    def read(self, state, q, k=None):
        if self.set_block is None or k is None:
            return self.retrieve(state, q)
        return self._compete(state, q, k)[:, : self.n_heads]

    def write(self, state, k, v, beta, q):
        if self.write_mode == "additive":
            err = v
        else:
            indep = torch.einsum("bij,bj->bi", state, k)
            if self.error_mode == "joint":
                # Correction to the independent prediction. See the class
                # docstring: replacing it outright is what diverges.
                joint = self._compete(state, q, k)[:, -1]
                err = v - (indep + self.W_p(joint - indep))
            else:
                err = v - indep
        return state + beta.unsqueeze(-1) * err.unsqueeze(-1) * k.unsqueeze(-2)


CORE_CLASSES: dict[str, type[FastWeightCore]] = {
    DeltaNetCore.kind: DeltaNetCore,
    CompFWPCore.kind: CompFWPCore,
}


def build_fwp_core(kind: str, features_dim: int, fwp_dim: int, n_heads: int, **flags):
    """``flags`` carries the CompFWP ablation options; other cores take none."""
    if kind not in CORE_CLASSES:
        raise NotImplementedError(
            f"core={kind!r} is accepted by the config but not yet built"
        )
    cls = CORE_CLASSES[kind]
    if cls is not CompFWPCore:
        flags = {}
    return cls(features_dim, fwp_dim=fwp_dim, n_heads=n_heads, **flags)
