# U11 — the reference numbers for the DreamerV3 report

Every number below was re-derived from the committed `results/bbf/*/metrics_seed*.json`
files, not copied from any prose (MISSION rule 6). Regenerate with
`tools/dreamerv3/dv3_reference_table.py`.

## Which BBF arms are final

**MISSION's U11 text says "v3 in `results/bbf/`". That is stale.** This harness was
written 2026-09-20; bbf-loop then ran a v4 round and closed at `STATUS: DONE`,
iteration 68, with `LAST_COMMIT ... "bbf: report against v4"`. The v4 arms exist, its
own REPORT.md is written against them, and v3 is superseded. **The DreamerV3 report
must quote v4.** No BBF-side number is missing, so nothing needs flagging under U11's
"flag rather than run" clause.

## PlayTrain frostbite (100-episode eval, fixed seed pool)

| arm | n seeds | mean | sd | per seed | win rate | floes | normalized |
|---|---|---|---|---|---|---|---|
| random | 1 | **33.50** | — | — | 0.00 | 3.4 | 0.000 |
| PPO @100k, greedy | 3 | 41.83 | 3.57 | 37.8 / 43.1 / 44.6 | 0.00 | 4.2 | 0.037 |
| PPO @100k, sampled | 3 | 44.17 | 3.50 | 40.2 / 45.5 / 46.8 | 0.00 | 4.4 | 0.047 |
| IMPALA @100k, greedy | 3 | 41.83 | 3.57 | 37.8 / 43.1 / 44.6 | 0.00 | 4.2 | 0.037 |
| IMPALA @100k, sampled | 3 | 39.60 | 2.26 | 38.1 / 38.5 / 42.2 | 0.00 | 4.0 | 0.027 |
| BBF RR=2 v4 | 5 | **150.44** | 30.87 | 111.3 … 187.1 | 0.36 | 11.4 | 0.516 |
| BBF RR=8 v4 | 5 | **203.46** | 27.64 | 166.0 … 230.8 | 0.59 | 14.4 | 0.750 |

Two things the report must not get wrong:

- **PPO and IMPALA greedy are byte-identical (41.83, same per-seed values).** That is
  not a copy-paste error; bbf-loop's F-014 diagnosed it — both policies collapsed to a
  single constant action (UP), so with the shared per-episode eval seed pool they
  replay identical trajectories. The honest statement is **"neither reference trainer
  learns this task at 100k agent steps"** (random is 33.50), not "PPO scores 41.83".
  Both evals are reported for exactly this reason.
- Normalized progress is `(score - 33.50) / (260 - 33.50)`; the PlayTrain ceiling is 260
  because a WIN ends the episode.

## ALE Frostbite — and what it says about our DreamerV3 gap

| arm | n | mean | sd | per seed |
|---|---|---|---|---|
| **BBF RR=2 v4 (ALE)** | 5 | **2844.9** | 690.1 | 1841 / 2582 / 2853 / 3340 / 3609 |
| official DreamerV3 | 5 | 2938.1 | 1603.9 | 350 / 2847 / 2944 / 4190 / 4359 |
| **our DreamerV3, frozen 5 seeds** | 5 | **958.7** | 1284.9 | 266 / 278 / 308 / 1067 / 2875 |
| our DreamerV3, 15 seeds | 15 | 890.6 | 886.3 | — |

This is the most useful thing U11 produced, and it was not the point of the unit.

**The BBF port reaches 2844.9 on ALE Frostbite at the same 100k-agent-step budget —
statistically indistinguishable from official DreamerV3's 2938.1 — while our DreamerV3
port reaches 890.6.** Same repository, same cluster, same ALE ROM, same budget, same
`ale-py`.

What that rules out: the cluster, the ROM, the 100k budget, and the general lab
infrastructure. A different agent in the same repo demonstrably reaches
official-DreamerV3-class scores on this game, so "Frostbite is hard here" does not
explain our gap.

What it does **not** rule out, and the report must say so: BBF's ALE stack is a
*different protocol* — 84 px grayscale, 4-frame stack, life-loss terminals, clipped
rewards, `gymnasium.AtariPreprocessing` — against DreamerV3's 64 px RGB, no stack, no
life terminals, unclipped rewards, and a hand-ported `atari.py`. So this does not
isolate our DreamerV3 env wiring; it isolates everything underneath it.

Also worth noting for the fidelity section: BBF's 5 ALE seeds have **sd 690** and all
five land in [1841, 3609], whereas the official DreamerV3 five have **sd 1604** with one
seed at 350. DreamerV3 on Frostbite is genuinely the more bimodal algorithm, which is
the context our 5/15 escape rate belongs in.
