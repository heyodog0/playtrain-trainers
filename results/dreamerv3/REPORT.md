# DreamerV3 in PyTorch — port, gate, and what the gate turned out to measure

Loop: `playtrain-internal/dreamerv3-loop`. Official source pinned at
`danijar/dreamerv3 @ e3f0224`, vendored read-only. Port on branch `dreamerv3` of
`playtrain-trainers`, never pushed. Every number below comes from a committed metrics
file; the tools that regenerate them are named beside each table.

## The headline

**Revised 2026-09-23 (U14).** An earlier version of this report said the official code
could not reproduce its published Frostbite score on this cluster, and that our port beat
it. **Both claims were wrong.** They rested on five official seeds that happened to be
the unlucky tail of a very high-variance game.

With 45 official seeds on this cluster, the official JAX code **does reach the published
range**. Our port breaks out somewhat less often, but not significantly less.

| arm | n | last-50k mean | broke out (>= 1000) |
|---|---|---|---|
| official JAX, published score file | 5 | 2938.1 | 4/5 |
| **official JAX, our cluster, all runs** | 45 | **1449.1** | **20/45** (Wilson [0.31, 0.59]) |
| of which: the first 5 seeds (the old "306") | 5 | 306.2 | 0/5 |
| of which: the next 15 seeds, same setup | 15 | 2025.7 | 10/15 |
| **our port** (v3, after A-1 and D-027) | 15 | **890.6** | **5/15** |

- **Frostbite at 100k is essentially a coin flip per seed.** A seed either stays at
  ~250-450 or breaks out to ~2000-5000. The official code breaks out about 44% of the
  time here, and the published 4/5 is consistent with that (Fisher p = 0.15).
- **The "306" was bad luck, not a reproduction failure.** Against the next 15 seeds of
  the identical setup, its 0/5 has p = 0.016.
- **Nothing varied here detectably changes the official code.** float32 instead of bf16
  gives 4/10 (mean Welch t = 0.05); the 2023 gym + atari-py env gives 5/10; A100
  instead of H100 gives 1/5, too few seeds to call. Full record:
  `official_u14/README.md`, `official_u14/arms.png`.
- **Our port against the official code:** 5/15 against 20/45 (p = 0.33), mean 890.6
  against 1449.1 (Welch t = 1.80). The port's point estimates are lower, but the gap is
  not established.
- **U15a: the ALE version is not the cause.** The official code on the port's ale-py 0.12.1
  breaks out 6/10, mean 1822.4, against 10/20 and 1595.9 on ale-py 0.9.0 (Welch t = 0.46).
  The Frostbite ROM is byte-identical in every setup. `official_u15_ale012/README.md`.
- **U15c': the port is the same computation.** Given identical params and batch, the port
  matches the official code on the H100 to within 3e-5 on every loss term over three full-lr
  training steps, with parameter updates within 1e-4 median. Its Atari env is byte-identical
  over 15,000 steps. No porting defect was found; the breakout gap is consistent with chance.
  An intermediate CPU-only artifact (XLA on CPU biases the official two-hot prediction by
  -0.166) was identified and ruled out. `equivalence_u15/README.md`.

### U13: the official code at the commit that produced the score file (superseded by U14)

*Kept for the record. Its comparisons use the unlucky 5-seed "306" as the official-here
reference, and U14 resolved its "two readings": the published draw is within the official
code's normal spread here. Its 2411f7d1 arms (2/10) are in line with that too.*

To separate repo drift from hardware, the official code was also run at `2411f7d1`, the
commit `scores/atari100k-dreamerv3.json.gz` was produced at. It ran with a 2024-era
environment (jax 0.4.26, ale-py 0.8.1) under both of that commit's Frostbite env
presets, 5 seeds each. Full record: `official_2411f7d1/README.md`, `official_2411f7d1/curves.png`.

| arm | n | last-50k mean | sd | escaped | per seed |
|---|---|---|---|---|---|
| published score file | 5 | 2938.1 | 1603.9 | 4/5 | 350 / 2847 / 2944 / 4190 / 4359 |
| e3f0224 on our cluster | 5 | 306.2 | 74.2 | 0/5 | 250 / 252 / 282 / 316 / 430 |
| 2411f7d1 `atari_frostbite` | 5 | 834.2 | 1180.4 | 1/5 | 248 / 275 / 302 / 403 / 2943 |
| 2411f7d1 `atari100k_frostbite` | 5 | 629.6 | 627.4 | 1/5 | 259 / 373 / 377 / 391 / 1748 |
| **2411f7d1 pooled** | 10 | **731.9** | 897.7 | **2/10** | |
| **our port** | 15 | **890.6** | 886.3 | **5/15** | |

The result was neither of the two outcomes the test was designed to tell apart:

- **The hardware can produce published-class runs from the official code.** The
  escaping 2411f7d1 seeds climb to ~3000 (A seed 3: 2943 in the window, ~3400 by 440k;
  B seed 0 reaches ~4000). Those are the published per-seed levels. So "this cluster
  can't do it" is out.
- **What differs is how OFTEN a seed escapes, not how well an escaped seed does.**
  Published: 4/5. Everything the official code does here: 2/15. The published rate is
  higher than 2411f7d1's 2/10 at Fisher one-sided p = 0.047. Against all 30 seeds run
  here (the official code at both commits plus our port), 7/30, it is higher at p = 0.026.
  This was an unplanned comparison, so read those as suggestive rather than decisive.
- **Repo drift is not shown.** 2411f7d1's 2/10 against e3f0224's 0/5 gives p = 0.43.
  The means move 306 → 732, but at this n that is well inside the noise of a bimodal
  score.
- **Our port behaves like the official code here.** 5/15 against 2411f7d1's 2/10 gives
  p = 0.40, and 890.6 against 731.9 in the mean. The two sets of per-seed curves have
  the same shape: most seeds sit at 250–400, and an occasional one breaks out to
  ~3000.

Two readings remain. The published score file may have drawn an unusually escape-heavy
set of seeds. A draw of 4/5 or better has probability 0.0014 at the official code's 2/15
rate here, 0.0067 at 0.2, and 0.045 at our port's 0.33. That makes pure luck unlikely,
though not impossible at the higher rate. It is also possible that some unrecorded detail of the
original setup raised the escape rate (driver/env versions, the batch_length-65 era's
0.246 ratio). Nothing run here distinguishes those two explanations.

## Fidelity

The architecture is **exact**. The official run printed its parameter count at startup
and it matches ours module for module — two independent derivations, ours by hand from
`configs.yaml` shapes in U03, theirs from the live JAX model:

| module | official | ours |
|---|---|---|
| dyn | 95,496,192 | 95,496,192 |
| dec | 20,282,115 | 20,282,115 |
| val | 12,850,431 | 12,850,431 |
| pol | 12,607,506 | 12,607,506 |
| rew | 10,749,183 | 10,749,183 |
| con | 10,488,833 | 10,488,833 |
| enc | 3,492,864 | 3,492,864 |
| **total** | **165,967,124** | **165,967,124** |

Three real fidelity defects were found and fixed during the loop, each visible in the
official source, which is why MISSION rule 1 requires diffing against it:

| id | defect | how it was found | effect |
|---|---|---|---|
| **A-1** | the terminal transition never reached the replay: the loop stored the CURRENT observation then stepped, so the `is_last` observation was replaced by the reset before it could be stored | arithmetic — `loss/con` sat at exactly **0.020440** = H(1 − 1/333), the cross-entropy floor of a CONSTANT continue target, for all 27,229 gradient steps | gate 290.8 → 943.5 |
| **D-027** | a 1 % unimix applied to the ACTOR. `Head.categorical` passes no unimix; `Head.unimix` reaches only `Head.onehot`, which a discrete policy never takes, so `agent.policy.unimix: 0.01` is dead in the pinned path despite the paper's table listing it | external audit | entropy +50-80 % at matched gradient steps, escapers 1 → 2, mean unchanged |
| **D-028** | a tensor policy in `imagine` used the whole array at every step instead of `policy[:, t]` | found while porting the open-loop report — the only caller that feeds recorded actions; training always supplies a callable, so the branch never ran | report path only |

The world model itself was independently cleared. 32-step open-loop prediction — twice
the training horizon — costs 1.7–15.3 % of a frozen-frame baseline across four
checkpoints, with no explosion in the per-step curve, and the saved strip tracks floe
positions, the player sprite and the igloo build-up deep into imagination.

Full side-by-side record: `playtrain-internal/dreamerv3-loop/DEVIATIONS.md` (29 entries)
and `AUDIT.md`.

## Where the paper and the code disagree

The code wins (PROTOCOL § 3), and all four are in the report because each changes a
number someone might try to reproduce:

| item | paper | pinned code |
|---|---|---|
| Atari-100k replay ratio | 128 (Table 2) | `train_ratio: 256` |
| optimizer β2 | 0.99 | 0.999 |
| actor unimix | "Actor unimix 1 %" | not applied — `Head.categorical` takes none (D-027) |
| Frostbite score | 3377 (Table 9) | 3377 exactly as the last-10k mean of the score file (the paper's window, danijar/dreamerv3#138); 2938 on this report's last-50k window |

A fifth, archaeological: `scores/atari100k-dreamerv3.json.gz` was committed when
`batch_length` was **65**, so the published curves ran at 256/(16×65) = 0.246 gradient
steps per agent step, against the pin's 0.25.

## The comparison table

PlayTrain frostbite, 100-episode eval, fixed seed pool. Regenerate with
`tools/dreamerv3/dv3_reference_table.py`; BBF arms are **v4** (bbf-loop closed DONE at
iteration 68 against v4 — MISSION's "v3" is stale).

| arm | n | mean | sd | win rate | normalized |
|---|---|---|---|---|---|
| random | 1 | 33.50 | — | 0.00 | 0.000 |
| PPO @100k greedy | 3 | 41.83 | 3.57 | 0.00 | 0.037 |
| IMPALA @100k greedy | 3 | 41.83 | 3.57 | 0.00 | 0.037 |
| BBF RR=2 v4 | 5 | 150.44 | 30.87 | 0.36 | 0.516 |
| BBF RR=8 v4 | 5 | 203.46 | 27.64 | 0.59 | 0.750 |
| **DreamerV3 (ours), eval sampled** | 5 | **96.28** | 24.35 | 0.05 | 0.277 |
| **DreamerV3 (ours), eval greedy** | 5 | **97.28** | 24.75 | 0.06 | 0.282 |

**PPO and IMPALA greedy are byte-identical because both policies collapsed to a single
constant action** (bbf-loop F-014). The honest statement is that neither reference
trainer learns this task at 100k agent steps — random is 33.50 — not that "PPO scores
41.83". Sampled evals: PPO 44.17, IMPALA 39.60.

**DreamerV3 (U09, faithful port) sits between the reference trainers and BBF.** It is above
PPO/IMPALA (p = 0.036) and below BBF RR=2 (p = 0.024) and RR=8 (p = 0.008). Training episodes over
the last 50k frames average 81.8. One seed stalled into the 108k-frame cap, which the protocol
assumed never happens. Full record: `playtrain_frostbite_u09/README.md`, `curves.png`.

One cross-check worth keeping: BBF v4 on **ALE** Frostbite scores 2844.9 (sd 690), close
to official DreamerV3's *published* 2938.1, at the same 100k budget on the same cluster.
So the hardware can produce official-DreamerV3-class Frostbite scores — just not from
DreamerV3. (BBF's ALE stack is a different protocol: 84 px grayscale, 4-stack, life
terminals, clipped rewards.)

## The substrate

MISSION deliverable 2 is complete. The world model sits behind a seven-member
`WorldModel` protocol (`world_model.py`); `RSSMWorldModel` holds the faithful port
unchanged; the actor-critic, replay and training loop name nothing DreamerV3-specific.

The proof is a stub with MLP dynamics, **no decoder**, no categorical latent and no KL
that trains 100 gradient steps through the real `Agent.train_step` and runs `train.py`
end to end. Writing it exposed a genuine leak (`assemble_imagined` reached in for
`repfeat["deter"]`) that no amount of re-reading had found.

Contract, and three places the seam is honestly thin:
`src/playtrain_trainers/dreamerv3/WORLD_MODEL_INTERFACE.md`.

## Measured cost

200M parameters at batch 16×64 with 1024 imagination rollouts × 15 steps fits one H100
at **25.07 GB** allocated / 30.49 GB reserved, so F-003 is retired with room to spare.
**0.5698 s per gradient step, 142.65 s per 1k agent steps steady-state, ~3.1 h per
110k-step seed** — reproduced on two nodes. The official JAX code runs the same
workload in **~1.05 h**, roughly 3× faster.

## The gate

**Closed (U16f).** With every equivalence gap closed and the three driver timing semantics copied
(D-030/031/032), the port on the frozen PROTOCOL 5A gate scores **2299.6 on the protocol's 5 seeds (PASS)**
and **1103.7 over 15 seeds (PLAUSIBLE)**, with 5/15 breakouts. The 15-seed number is the one to quote; the
5-seed PASS is a favourable draw. Against the official code on this cluster (1518.3 over 54 scored seeds,
26/55 breakouts) there is no detectable difference: Fisher p = 0.33, Welch t = -1.16. The threshold was
not changed; F-001 is resolved by the threshold as written being met. U09 (PlayTrain) is unblocked by
MISSION rule 2. `ale_frostbite_gate_u16/README.md`, `equivalence_u16/README.md`.
