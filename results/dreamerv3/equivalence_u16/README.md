# U16: closing the equivalence gaps U15 left

## (a) Faithful bf16 (done)
The port previously used torch autocast. It now casts explicitly at the official sites (D-012, revised):
`nets.cast` on the RSSM inputs/carry/samples/feats, the encoder image, the decoder inputs, `feat2tensor` and
the imagined action embedding; kernels follow the activation dtype; Norm runs in f32 and casts back; every
output distribution takes f32 logits. Commit b9a3962.

- **Cast policy is identical:** every dot (112) and convolution (23) in the official unoptimized HLO is
  bf16 x bf16 -> bf16, and 100% of the port's traced matmul/conv FLOPs are bf16 x bf16. See `bf16/audit_and_sweep.txt`.
- **4-seed sweep on H100** (each seed has its own params and batch; official and port each in fp32 and bf16;
  step-1 relative gaps, mean over seeds):

| metric | port fp32 vs off fp32 | off bf16 vs off fp32 | port bf16 vs off fp32 |
|---|---|---|---|
| loss/image | 0 | 5.3e-3 | 3.9e-3 |
| loss/dyn | 9e-8 | 2.0e-3 | 2.1e-3 |
| loss/con | 5e-8 | 9.9e-3 | 2.4e-2 |
| loss/value | 8e-8 | 3.1e-2 | 3.7e-2 |
| grad_norm | 1e-7 | 3.2e-3 | 4.1e-3 |

In fp32 the port reproduces the official code on the H100 to ~1e-7. In bf16 both sides deviate from fp32 by
the same order on every metric, which is as close as bf16 rounding (XLA vs cuBLAS summation order) lets two
implementations be. `--xla_allow_excess_precision=false` does not change the official numbers.

## (b) Samplers (done)
Real samplers on fixed logits: policy `Categorical` over 18 actions, no unimix (D-027); latent `OneHot` over
32x64 with unimix 0.01. The port runs on the H100, the official code runs jax.random. Job 48159681, `samplers/`.

- Exact probabilities after unimix agree to 1.8e-7.
- Chi-square of each side's draws against those probabilities (200k policy draws, 20k latent draws per row):
  policy p-values port min 0.188 / official min 0.014; latent 128 rows, fraction below 0.01 is 0.016 port / 0.000
  official (1.3 rows expected by chance); KS test of p ~ Uniform: port 0.60 / 0.16, official 0.25 / 0.34.
  **Both samplers draw from the stated distribution.**
- Straight-through `OneHot` value: identical; gradient w.r.t. logits within 1.8e-7 (scale 0.62).

## (c) Acting path (done)
`agent.policy` on both sides, identical params, over one recorded 40-step Frostbite sequence with an env
reset at step 20, argmax sampling, H100. Jobs 48159870/48160036 (fp32), 48160123 (bf16, same sequence).

| comparison | actions equal | deter rel err step 1 | stoch mismatches steps 1-5 |
|---|---|---|---|
| port fp32 vs official fp32 | **40/40** | 8.9e-7 (<= 8.9e-7 every step) | 0 0 0 0 0 (0 on all 40 steps) |
| port bf16 vs official bf16 | 14/40 | 0.27 | 0 2 1 1 1 |
| official bf16 vs official fp32 (scale) | 9/40 | 0.89 | 4 3 3 5 4 |

In fp32 the acting path, including the reset, is exact. In bf16 the port is closer to the official bf16 run
than the official code's own fp32 run is: near-tied argmax latents flip and the trajectories fork, on both sides.

**Finding for (d):** the official agent ACTS WITH STALE POLICY PARAMS. `train()` stashes the policy keys
(`^(enc|dyn|dec|pol)/`) as they were BEFORE that update, and `policy()` computes its action with the current
`policy_params` and only then swaps the stash in (embodied/jax/agent.py:160-170, 240-250, 276-282). The port
acts with the live params.

## (d) Replay and driver loop (done)
**A read-through against the official driver, streams and agent found three timing semantics the port lacked. All three are now copied:**
- **D-030 lagged acting params.** The agent acts with a copy of `^(enc|dyn|dec|pol)/` that is refreshed from a
  pre-update stash only after the next action.
- **D-031 prefetched batches.** `Prefetch(amount=1)`: the first batch is drawn when the replay has one item;
  each hand-over draws the next batch before the current one trains or writes back.
- **D-032 lagged write-back.** `agent.train` returns the previous call's outs, so replay write-back lags one
  train step.

Everything else already matched: the transition layout and terminal masking, the callback order, the ratio
clock, and the online queue (train-only).

**Deterministic end to end** (`e2e/`). The official side uses its own Driver, Replay, make_stream and
agent.stream, with trainfn copied verbatim. The port runs `train.run` unchanged. Both use identical initial
params, argmax sampling, ALE seed 0, replay seed 0 and fp32 on the H100; the official Prefetch is made
synchronous with the port's idealization. 1330 agent steps, 61 train steps. Job 48161840:
- **The first 1222 stored transitions are identical** (the whole 1088-step warmup plus 33 train steps of acting).
- **Sampled batches 0-34 are identical** (images, actions, rewards, flags), and the written-back latents agree to ~1e-6.
- **Per-step losses** agree to 1e-7 at train 0, 7e-7 at 10, 1e-5 at 20 and 1e-4 at 30: smooth ~10x per 10 steps,
  i.e. float noise amplified chaotically. Around train 34 one near-tied argmax action flips (env step 1222)
  and the runs fork.
- **Control** (job 48162069): the port before D-030/031/032 diverges at env step 175, and its batch 0 already
  differs. That run also still has the old encoder, which cast images to bf16 on CUDA even in fp32 mode;
  the U16a rewrite removed that.

## (e) Long horizon (done)
The stateful parts in isolation: official code on one side, port on the other, fed byte-identical 10,000-step
streams regenerated from per-step seeds (`long_horizon/`, job 48163184, fp32 CPU). The streams are
nonstationary on purpose: log-scale-walking heavy-tailed gradients with AGC-triggering spikes and all-zero
steps, drifting heavy-tailed returns, and a random-walk source network.

| component | worst gap over 10k steps |
|---|---|
| optimizer: official `_make_opt` chain (AGC 0.3, RMS 0.999, momentum 0.9, lr 4e-5 with 1000-step warmup) vs port LaProp | params 2-3e-6; displacement from init 2e-4 early, 3.6e-5 at 10k (does not grow) |
| retnorm: official `Normalize('perc', rate 0.01, limit 1, 5/95, no debias)` vs port | offset 1.9e-6 abs (values up to 20.7), scale 1.9e-6 rel |
| slow critic: official `SlowModel` rate 0.02 every 1 | 5.3e-5 (float noise; a formula or count error would be ~1e-2) |
| slow critic: rate 0.1 every 3 | 6.8e-7 (the `count % every` phase matches) |
