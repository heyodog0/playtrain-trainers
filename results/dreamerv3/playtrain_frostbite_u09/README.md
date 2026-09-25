# U09: DreamerV3 (faithful port) on PlayTrain frostbite

Port at 73c6bc6 (D-012 explicit bf16 casts, D-030/031/032 driver timing), `configs/dreamerv3/frostbite.json`
(atari100k preset, `env_backend: playtrain`), 5 seeds x 110k agent steps, job 48223890 (H100; seeds 0-2 on
slower nodes, ~4.3 h; seeds 3-4 ~3.0 h). Every seed ran 27,229 grad steps = expected. Evals: job 48280179 (seeds
3-4) and 48299965 (seeds 0-2), `tools/dreamerv3/dv3_evaluate.py`, 100 whole games per seed on the BBF arms'
fixed game-seed pool (8,000,000 + i), sampled and greedy, with statistics from `bbf.evaluate.aggregate`.
Table: `tools/dreamerv3/dv3_playtrain_table.py` -> `table.txt`. Curves: `plot.py` -> `curves.png`.

| arm | n | mean | 95% CI over seeds | win rate | floes | normalized |
|---|---|---|---|---|---|---|
| PlayTrain random | 1 | 33.50 | - | 0.00 | 3.4 | 0.000 |
| PPO @100k greedy / sampled | 3 | 41.83 / 44.17 | | 0.00 | 4.2 / 4.4 | 0.037 / 0.047 |
| IMPALA @100k greedy / sampled | 3 | 41.83 / 39.60 | | 0.00 | 4.2 / 4.0 | 0.037 / 0.027 |
| BBF RR=2 v4 | 5 | 150.44 | [127.3, 173.3] | 0.36 | 11.4 | 0.516 |
| BBF RR=8 v4 | 5 | 203.46 | [182.8, 224.1] | 0.59 | 14.4 | 0.750 |
| **DreamerV3, eval sampled** | 5 | **96.28** | [78.3, 115.7] | 0.05 | 9.1 | 0.277 |
| **DreamerV3, eval greedy** | 5 | **97.28** | [79.2, 117.2] | 0.06 | 9.2 | 0.282 |
| DreamerV3, training episodes (last 50k frames) | 5 | 81.8 | [66.5, 95.6] | | | |

Per seed (sampled eval): 126.3 / 85.8 / 64.2 / 113.8 / 91.3; seed 0 wins 22% of games.

Exact unpaired permutation tests (`bbf_compare.exact_permutation_p`): DreamerV3 is above every PPO/IMPALA arm
(p = 0.036 for 5 vs 3; that design's true minimum is 1/56 = 0.018, not the 0.036 "floor" bbf_compare prints,
which assumes equal arm sizes). DreamerV3 is below BBF RR=2 (p = 0.024) and BBF RR=8 (p = 0.008, the 5 vs 5 floor).

**Finding (contradicts PROTOCOL 4 D-006/D-008, "the cap is never hit on this game"):** seed 2 played one
26,983-agent-step episode (107,928 frames) that hit the 108k-frame cap, and another of 12,816 steps, ~36% of its
budget in two stalled games. It is the weakest seed. Following the copied official `is_terminal = is_last`,
that truncation entered replay as a terminal.
