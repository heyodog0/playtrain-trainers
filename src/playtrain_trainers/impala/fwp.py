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
FWP_CORES = ("deltanet", "compfwp", "deltanet_ref")

#: Key/query feature maps for the trainer cores (T.2). ``l2k`` is the original.
FEATURE_MAPS = ("l2k", "elu_sumnorm")


def elu_p1_sum_norm(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """The reference feature map: ELU+1, then normalise to sum 1.

    Verbatim from IDSIA/recurrent-fwp `torchbeast/layer.py`:
    ``y = F.elu(x, 1., False) + 1.; y / (y.sum(-1, keepdim=True) + 1e-5)``.
    Applied to BOTH queries and keys there, which makes the read ``W q`` a
    convex combination of what was written — bounded whatever ``W`` does.
    """
    y = F.elu(x, 1.0) + 1.0
    return y / (y.sum(-1, keepdim=True) + eps)


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

    def __init__(
        self,
        features_dim: int,
        fwp_dim: int = 128,
        n_heads: int = 8,
        decay: float = 0.0,
        w_o_gain: float = 0.1,
        read_norm: bool = False,
        feature_map: str = "l2k",
        multihead: bool = False,
    ):
        super().__init__()
        if fwp_dim < 1 or n_heads < 1:
            raise ValueError("fwp_dim and n_heads must be positive")
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"decay must be in [0, 1), got {decay}")
        if feature_map not in FEATURE_MAPS:
            raise ValueError(f"feature_map must be one of {FEATURE_MAPS}, got {feature_map!r}")
        if multihead and fwp_dim % n_heads:
            raise ValueError(f"multihead needs fwp_dim divisible by n_heads; got {fwp_dim}/{n_heads}")
        self.features_dim = features_dim
        self.fwp_dim = fwp_dim
        self.n_heads = n_heads
        # T.2: the two ingredients of the reference's bounded read.
        # ``feature_map``: "l2k" is this repo's original (unit-norm key, raw
        # query); "elu_sumnorm" puts BOTH keys and query rows on the simplex,
        # so every read is a convex combination of stored values.
        # ``multihead``: the state is per-head ``(n_heads, d_h, d_h)`` with
        # ``d_h = fwp_dim // n_heads``; query row m reads only head m, and
        # the key/value/beta are split per head. Both default off, and off is
        # bit-identical to the pre-T.2 code.
        self.feature_map = feature_map
        self.multihead = multihead
        self.d_h = fwp_dim // n_heads if multihead else fwp_dim
        # Per-step multiplicative forgetting applied before each write. 0.0 is
        # off and is the default, so a run that does not ask for it is
        # unchanged. It bounds the memory horizon directly — roughly 1/decay
        # steps — which is the handle episode length turned out not to give.
        self.decay = float(decay)

        self.W_k = nn.Linear(features_dim, fwp_dim, bias=False)
        self.W_v = nn.Linear(features_dim, fwp_dim, bias=False)
        # Query rows: n_heads rows of width d_h. Single-head d_h == fwp_dim,
        # so the shapes below reduce to the originals.
        self.W_q = nn.Linear(features_dim, n_heads * self.d_h, bias=False)
        self.w_b = nn.Linear(features_dim, n_heads if multihead else 1)
        self.W_o = nn.Linear(n_heads * self.d_h, features_dim)
        # Output gain. 0.1 was chosen so the core starts near a pass-through;
        # the audit (F1) found that it also throttles the gradient reaching the
        # set block so hard that the block moved 5.8% in 30M steps. Kept as the
        # default so existing runs are unchanged; 1.0 is the candidate.
        nn.init.orthogonal_(self.W_o.weight, gain=w_o_gain)
        nn.init.zeros_(self.W_o.bias)
        # Optional LayerNorm on the retrieved rows before the output
        # projection, so the readout is bounded whatever ||S|| does. Applied
        # only on the read path: the write's post-minus-pre difference, and
        # with it the contraction argument, is untouched.
        self.read_norm = nn.LayerNorm(self.d_h) if read_norm else None

    @property
    def state_shape(self) -> tuple[int, ...]:
        """Per-env state: (d, d) single-head, (H, d_h, d_h) multihead."""
        if self.multihead:
            return (self.n_heads, self.d_h, self.d_h)
        return (self.fwp_dim, self.fwp_dim)

    @property
    def state_size(self) -> int:
        n = 1
        for d in self.state_shape:
            n *= d
        return n

    def initial_state(self, batch_size: int = 1) -> tuple[torch.Tensor]:
        """Zero matrix state, flattened, batch at dim 1 — a 1-tuple."""
        return (torch.zeros(1, batch_size, self.state_size),)

    # -- the two things a subclass changes ---------------------------------

    def write(self, state, k, v, beta, q):
        raise NotImplementedError

    def read(self, state, q, k=None):
        """Retrieve the M query rows. Independent by default, so `k` is unused."""
        del k
        return self.retrieve(state, q)

    def retrieve(self, state, q):
        """Raw matrix read, before any competition between the rows.

        Single-head: every row reads the one matrix. Multihead: row m reads
        head m, and ``q`` may carry several rows per head stacked along dim 1
        (``q.shape[1]`` a multiple of ``n_heads``, head-major).
        """
        if not self.multihead:
            return torch.einsum("bij,bmj->bmi", state, q)
        B, M, d_h = q.shape
        q = q.view(B, -1, self.n_heads, d_h)  # [B, rows_per_head, H, d_h]
        rows = torch.einsum("bhij,brhj->brhi", state, q)
        return rows.reshape(B, M, d_h)

    def key_pred(self, state, k):
        """The state's current prediction at the write key: S k, per head."""
        if not self.multihead:
            return torch.einsum("bij,bj->bi", state, k)
        return torch.einsum("bhij,bhj->bhi", state, k)

    def key_rows(self, k):
        """The write key as query row(s): one row single-head, one per head multihead."""
        return k if self.multihead else k.unsqueeze(1)

    def outer_update(self, state, beta, err, k):
        """``S + beta * err k^T`` in either layout."""
        return state + beta.unsqueeze(-1) * err.unsqueeze(-1) * k.unsqueeze(-2)

    def project(self, x: torch.Tensor):
        """k, v, beta, q for one timestep, in the configured layout.

        Single-head: k, v [B, d]; beta [B, 1]; q [B, M, d].
        Multihead:   k, v [B, H, d_h]; beta [B, H, 1]; q [B, H, d_h].
        """
        B = x.shape[0]
        k = self.W_k(x)
        v = self.W_v(x)
        beta = torch.sigmoid(self.w_b(x))
        q = self.W_q(x).view(B, self.n_heads, self.d_h)
        if self.multihead:
            k = k.view(B, self.n_heads, self.d_h)
            v = v.view(B, self.n_heads, self.d_h)
            beta = beta.view(B, self.n_heads, 1)
        if self.feature_map == "elu_sumnorm":
            k = elu_p1_sum_norm(k)
            q = elu_p1_sum_norm(q)
        else:
            k = F.normalize(k, dim=-1)
        return k, v, beta, q

    # -- one timestep and the unroll ---------------------------------------

    def step(self, x: torch.Tensor, state: torch.Tensor):
        """One timestep: forget a little, write, then read the updated state."""
        if self.decay:
            state = state * (1.0 - self.decay)
        k, v, beta, q = self.project(x)
        state = self.write(state, k, v, beta, q)
        rows = self.read(state, q, k)
        if self.read_norm is not None:
            rows = self.read_norm(rows)
        return x + self.W_o(rows.flatten(1)), state

    def forward(self, core_input: torch.Tensor, notdone: torch.Tensor, core_state):
        """core_input [T, B, F], notdone [T, B] float -> ([T*B, F], state)."""
        T, B, _ = core_input.shape
        state = core_state[0].reshape(B, *self.state_shape)
        outs = []
        mask_shape = (B,) + (1,) * len(self.state_shape)
        for t in range(T):
            state = state * notdone[t].view(*mask_shape)
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
        return self.outer_update(state, beta, v - self.key_pred(state, k), k)


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
        decay: float = 0.0,
        w_o_gain: float = 0.1,
        read_norm: bool = False,
        w_p_init: float = 0.0,
        feature_map: str = "l2k",
        multihead: bool = False,
    ):
        super().__init__(features_dim, fwp_dim=fwp_dim, n_heads=n_heads, decay=decay,
                         w_o_gain=w_o_gain, read_norm=read_norm,
                         feature_map=feature_map, multihead=multihead)
        if read not in ("joint", "indep"):
            raise ValueError(f"read must be joint|indep, got {read!r}")
        if error not in ("joint", "indep"):
            raise ValueError(f"error must be joint|indep, got {error!r}")
        if write not in ("delta", "additive"):
            raise ValueError(f"write must be delta|additive, got {write!r}")
        if error == "joint" and read == "indep":
            raise ValueError("error='joint' needs read='joint': there is no joint read to use")
        self.read_mode, self.error_mode, self.write_mode = read, error, write
        # Rows are d_h wide (== fwp_dim single-head). Multihead appends one
        # write-key row PER head, so the block sees 2H rows and W_p is shared
        # across heads.
        self.set_block = SetBlock(self.d_h, attn_heads) if read == "joint" else None
        self.W_p = nn.Linear(self.d_h, self.d_h, bias=False) if error == "joint" else None
        if self.W_p is not None:
            # Zero by default: the write path starts as the plain delta rule.
            # The audit (F1) found zero also starves the set block of write-
            # path gradient; w_p_init > 0 gives it a small random start.
            if w_p_init > 0:
                nn.init.normal_(self.W_p.weight, std=w_p_init)
            else:
                nn.init.zeros_(self.W_p.weight)

    @property
    def variant(self) -> str:
        return f"{self.read_mode}/{self.error_mode}/{self.write_mode}"

    def _compete(self, state, q, k):
        """Retrieve the M query rows plus the write key, then let them compete."""
        rows = self.retrieve(state, torch.cat([q, self.key_rows(k)], dim=1))
        return self.set_block(rows)

    def read(self, state, q, k=None):
        if self.set_block is None or k is None:
            return self.retrieve(state, q)
        return self._compete(state, q, k)[:, : self.n_heads]

    def write(self, state, k, v, beta, q):
        if self.write_mode == "additive":
            err = v
        else:
            indep = self.key_pred(state, k)
            if self.error_mode == "joint":
                # Correction to the independent prediction. See the class
                # docstring: replacing it outright is what diverges.
                joint = self._compete(state, q, k)[:, self.n_heads :]
                if not self.multihead:
                    joint = joint[:, -1]
                err = v - (indep + self.W_p(joint - indep))
            else:
                err = v - indep
        return self.outer_update(state, beta, err, k)


class RefDeltaNetCore(nn.Module):
    """Irie et al.'s RL DeltaNet, ported as a reference core.

    Faithful to IDSIA/recurrent-fwp ``reinforcement_learning/torchbeast/
    layer.py`` and ``fast_weight/__init__.py`` (read 2026-09-20):

    * one linear produces q, k, v per head and a beta per head;
    * q and k both pass through ELU+1 sum-normalisation;
    * per-head fast weights ``W_h`` of shape ``(dim_head, dim_head)``;
    * ``v_old = W k;  W += beta * (v - v_old) ⊗ k;  out = W q`` — read AFTER
      the write;
    * ``x + out_linear(out)``, no LayerNorm.

    Differences from the reference that are deliberate: one layer instead of
    two, no dropout (RL), and no clipped-reward input to the core — the
    reference concatenates it to the CNN features; that is a separate flag if
    wanted. The state keeps this repo's contract: a 1-tuple, batch at dim 1,
    flattened to ``(1, B, H * dh * dh)``.
    """

    kind = "deltanet_ref"

    def __init__(self, features_dim: int, n_heads: int = 4, dim_head: int = 64):
        super().__init__()
        if n_heads * dim_head != features_dim:
            raise ValueError(
                f"reference core needs n_heads*dim_head == features_dim; "
                f"got {n_heads}*{dim_head} != {features_dim}"
            )
        self.features_dim, self.n_heads, self.dim_head = features_dim, n_heads, dim_head
        self.qkvb = nn.Linear(features_dim, 3 * n_heads * dim_head + n_heads)
        self.out_linear = nn.Linear(n_heads * dim_head, features_dim)

    @property
    def state_size(self) -> int:
        return self.n_heads * self.dim_head * self.dim_head

    def initial_state(self, batch_size: int = 1) -> tuple[torch.Tensor]:
        return (torch.zeros(1, batch_size, self.state_size),)

    def project(self, x: torch.Tensor):
        B, H, dh = x.shape[0], self.n_heads, self.dim_head
        qkvb = self.qkvb(x)
        q, k, v, beta = torch.split(qkvb, [H * dh, H * dh, H * dh, H], dim=-1)
        q = elu_p1_sum_norm(q.view(B, H, dh))
        k = elu_p1_sum_norm(k.view(B, H, dh))
        v = v.view(B, H, dh)
        beta = torch.sigmoid(beta).view(B, H, 1)
        return q, k, v, beta

    def step(self, x: torch.Tensor, W: torch.Tensor):
        q, k, v, beta = self.project(x)
        v_old = torch.einsum("bhij,bhj->bhi", W, k)
        W = W + torch.einsum("bhi,bhj->bhij", beta * (v - v_old), k)
        out = torch.einsum("bhij,bhj->bhi", W, q).reshape(x.shape[0], -1)
        return x + self.out_linear(out), W

    def forward(self, core_input: torch.Tensor, notdone: torch.Tensor, core_state):
        T, B, _ = core_input.shape
        W = core_state[0].reshape(B, self.n_heads, self.dim_head, self.dim_head)
        outs = []
        for t in range(T):
            W = W * notdone[t].view(B, 1, 1, 1)
            out, W = self.step(core_input[t], W)
            outs.append(out)
        return torch.flatten(torch.stack(outs), 0, 1), (W.reshape(1, B, self.state_size),)


CORE_CLASSES: dict[str, type[nn.Module]] = {
    DeltaNetCore.kind: DeltaNetCore,
    CompFWPCore.kind: CompFWPCore,
    RefDeltaNetCore.kind: RefDeltaNetCore,
}


def build_fwp_core(
    kind: str, features_dim: int, fwp_dim: int, n_heads: int, decay: float = 0.0,
    w_o_gain: float = 0.1, read_norm: bool = False,
    ref_heads: int = 4, ref_dim_head: int = 64,
    feature_map: str = "l2k", multihead: bool = False, **flags
):
    """``flags`` carries the CompFWP-only options; other cores take none."""
    if kind not in CORE_CLASSES:
        raise NotImplementedError(
            f"core={kind!r} is accepted by the config but not yet built"
        )
    cls = CORE_CLASSES[kind]
    if cls is RefDeltaNetCore:
        return cls(features_dim, n_heads=ref_heads, dim_head=ref_dim_head)
    if cls is not CompFWPCore:
        flags = {}
    return cls(features_dim, fwp_dim=fwp_dim, n_heads=n_heads, decay=decay,
               w_o_gain=w_o_gain, read_norm=read_norm,
               feature_map=feature_map, multihead=multihead, **flags)
