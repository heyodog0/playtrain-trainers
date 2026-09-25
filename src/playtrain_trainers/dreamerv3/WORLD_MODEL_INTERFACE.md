# The world-model interface

MISSION deliverable 2. The point of this file is that the next project can replace what
the world model predicts, or how it factors its latent, without reading the
actor-critic, the replay or the training loop.

Everything DreamerV3-specific — the convolutional encoder, the block-diagonal GRU, the
32x64 categorical latent, the reconstruction decoder, the KL pair — lives behind
`WorldModel` in `world_model.py`. `RSSMWorldModel` is the faithful port; a replacement
is another class with the same seven members and one changed line in `Agent`.

The proof that the seam is real, not aspirational, is
`tests/test_dreamerv3_interface.py`: a stub with MLP dynamics, **no decoder**, no
categorical latent and no KL trains for 100 gradient steps through the real
`Agent.train_step`, and runs the whole of `train.py` end to end. Writing it found one
genuine leak (`assemble_imagined` had reached in for `repfeat["deter"]`), which is what
a stub is for.

## The tensors

`B` is the batch of sequences, `T` the steps in a replayed window, `H` the imagination
horizon, `A` the action dimension.

| name | shape | what it is |
|---|---|---|
| `obs["image"]` | `(B, T, 64, 64, 3)` uint8 | observations, HWC, as the env emits them |
| `obs["reward"]` | `(B, T)` float | raw, unclipped |
| `obs["is_first"]`, `is_last`, `is_terminal` | `(B, T)` bool | episode structure |
| `prevact` | `(B, T, A)` float | one-hot; the action that PRODUCED each step |
| `carry` | dict of tensors | the recurrent state, opaque outside the model |
| `feat` | dict of `(B, T, ...)` | features; opaque except through `feat2tensor` |
| `entries` | dict of `(B, T, ...)` | the subset of `feat` replay stores per step |
| `feat2tensor(feat)` | `(B, T, feat_dim)` | the only thing the heads consume |

## The seven members

### `feat_dim: int`
Width of `feat2tensor`. The actor and critic are sized from this alone.

### `entry_keys: tuple[str, ...]`
Which parts of `feat` survive a round trip through replay. `Agent.policy` stores exactly
these alongside each transition; `Agent.train_step` reads them back as the starting
carry for a replayed window (`replay_context`), and writes the freshly computed ones
back after each gradient step. `RSSMWorldModel` declares `("deter", "stoch")`; the stub
declares `("latent",)`. Nothing outside the model names these keys.

### `initial(batch_size, device) -> carry`
A zeroed carry for `batch_size` independent streams.

### `observe_step(carry, obs, prevact, is_first) -> (carry, feat)`
One **acting** step, batch of 1 in the driver. `is_first` must reset both the carry and
the incoming action — an episode boundary arrives as a flag, never as a separate call.

### `loss(carry, obs, prevact, scales) -> (carry, entries, feat, losses, metrics)`
One **training** pass over a replayed batch. Contract:

- every value in `losses` is `(B, T)` **before** any reduction, and its keys are exactly
  `loss_keys()`;
- `feat` is the posterior feature for all `T` steps, and is what imagination starts from;
- `entries` is `(B, T, ...)` for each of `entry_keys`.

`scales` is passed in rather than owned, so the model does not decide its own weighting.

### `imagine(carry, policy, length) -> (carry, feat, actions)`
Roll forward with **no observations**. `policy` is either a callable — which must be
called on a **detached** carry, so the actor's gradient comes from the return and not
from differentiating the rollout into its own start — or a `(B, length, A)` tensor of
recorded actions, in which case step `t` uses `policy[:, t]` (D-028).

### `predict_reward(feat_tensor) -> Output` and `predict_continue(feat_tensor) -> Output`
Predictions at given features, as `outs.Output` objects so the caller can take `.pred()`,
`.prob1()` or `.loss(target)` without knowing the distribution. The actor-critic uses
`predict_reward(...).pred()` and `predict_continue(...).prob1()` inside imagination.

### `feat2tensor(feat) -> Tensor`
The only bridge from the opaque feature dict to the heads.

### `loss_keys() -> tuple[str, ...]`
Which loss terms `loss` returns. `world_model.scales_for` builds the scale table from
this, so a model with different terms gets a table with different keys rather than
DreamerV3's five silently assumed.

## What the rest of the agent is allowed to assume

Only the above. Specifically it does **not** see tokens, the encoder, the decoder, the
KL structure, the latent's shape, or any key of `feat` or `carry` by name.

The actor-critic (`ac.py`) additionally assumes the standard RL semantics that are not
world-model-specific: rewards are scalars per step, `continue` is a probability in
`[0, 1]`, and the imagination horizon is fixed. Replay assumes only that `entry_keys`
are arrays it can store and hand back.

## Where a change would plug in

**A different prediction target** (say, predicting a learned embedding instead of
pixels): change `loss` to emit the new term and `loss_keys` to name it. Nothing else
moves — the stub does exactly this, replacing `image` with `pred`.

**A factored latent** (separate slots for, say, agent and world): put the factors in
`carry` and `feat` under whatever keys you like, list in `entry_keys` the ones replay
must carry, and have `feat2tensor` concatenate whichever the heads should see. The
actor-critic never learns the factorization exists. If some factor should be visible to
the critic but not the actor, that is the one change this interface does NOT yet
support — it would need `feat2tensor` split into two, and that is a deliberate
simplification rather than an oversight.

**A different dynamics model** (transformer, SSM): only `observe_step`, `imagine` and
`loss` change. `imagine` is the method to watch: the actor-critic calls it with
`B * T` parallel rollouts at `imag_last: 0`, so a model whose rollout cost is
superlinear in batch will dominate the step time.

## Where the seam is thin

Honest notes rather than claims.

- `Agent.loss` still computes the imagination and actor-critic terms itself, so a model
  that wanted a different **actor-critic** (not just a different world model) would have
  to change `Agent`. The seam is around the world model only, which is what MISSION
  asked for.
- `train.py` writes `entry_keys` into replay as plain arrays, so an entry that is not a
  fixed-shape float array would need the replay's chunk dtype logic revisited.
- `RSSMWorldModel.dec` is reached directly by the reconstruction PNG in `train.py`.
  That is a reporting path, not a training one, and it is guarded — but it is the one
  place a caller names something the interface does not promise.
