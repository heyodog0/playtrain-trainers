# U15c': is the port's training step and env the same computation as the official code?

**Yes. With identical inputs, the port reproduces the official code's training step, optimizer and
Atari env; no porting defect was found.**

## Method
- Official pinned e3f0224 (`equiv_official.py`) and the port at 8be1a32 (`equiv_port.py`), both in float32.
  `Categorical.sample` is replaced by argmax on both sides, which covers all three sampling sites (the RSSM
  latents, the imagination policy and the acting policy), so a step is deterministic.
- Identical inputs: the official initial params, with all 104 port tensors mapped by name (the decoder's conv
  stack is indexed in reverse), and one batch of 2 x 9 real Frostbite steps.
- Three training steps. Every loss term, the gradient norm and every parameter update are compared.

## Results
| comparison | job(s) | outcome |
|---|---|---|
| official CPU vs port CPU, lr warmup 1000 (updates ~1e-8) | 48154720 / 48155290 | every world-model, value and repval loss and the grad norm within 4e-6; **policy differs**, see next row |
| probe: official `TwoHot.pred` at zero logits | twohot_probe.py (CPU), 48155903 (H100) | **CPU jit -0.166, H100 jit 0.0, eager 0.0.** XLA-on-CPU reorders the symmetric sum; the official reference on CPU is therefore not faithful to the official GPU runs |
| official H100 vs port, full lr (warmup 0), identical params and batch | 48157139 / 48157190* | **all 8 loss terms including policy and value within 3e-5 over 3 steps; parameter updates median rel 6.7e-6 (step 1) / 1.1e-4 (step 3), sign agreement >= 0.997**; the largest relative gaps are float noise on ~1e-6 slow-critic EMA steps |
| env: official atari.py vs port envs.py, ale-py 0.12.1, same seeds and actions | env_equiv.py | **byte-identical** over 15,000 steps, 37 episodes, seeds 0-4 |

\* `port_gpu_48157190` ran the port on **CPU** (`Agent` does not move itself to `cfg.device`; `train.py` does),
so that row is port CPU against official H100, a stricter cross-device check.

## What it leaves open
The acting-time sampler (torch multinomial against jax's Gumbel-max: the same distribution, different
draws), the port's bf16 autocast on GPU against the official full bf16 (U14 found that precision does not
change the official code's outcome), and replay/driver scheduling (audited in U05-U08; the ratio matches
exactly). The port's lower breakout rate (5/15 against the official 26/55, p = 0.25) is consistent with
chance.
