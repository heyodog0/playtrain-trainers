# U16f / U08: the ALE Frostbite gate with the faithful port

Port at commit 73c6bc6: explicit bf16 casts (D-012 revised), lagged acting params (D-030), prefetched
batches (D-031) and lagged write-back (D-032). Frozen gate config `configs/dreamerv3/ale_frostbite.json`
(atari100k preset, ALE Frostbite, 110k agent steps, train ratio 0.25). Seeds 0-14, job 48163586, H100,
~3.0 h per seed; seed 7 ran on a slower node (4.1 h). Every seed has grad_steps = 27,229 = expected.
Aggregation: PROTOCOL 5A, i.e. training episodes ending in [350k, 400k) frames. Regenerate with
`tools/dreamerv3/dv3_gate_summary.py results/dreamerv3/ale_frostbite_gate_u16`.

| | n | last-50k mean | broke out (>=1000) | PROTOCOL 5A |
|---|---|---|---|---|
| **frozen protocol seeds 0-4** | 5 | **2299.6** (345 / 2452 / 2874 / 3115 / 2712) | 4/5 | **PASS** (>= 1500) |
| **all 15 seeds** (the honest estimate) | 15 | **1103.7** (sd 1180.9) | 5/15 | PLAUSIBLE (1000-1499) |
| pre-fix port (v3) | 15 | 890.6 | 5/15 | FAIL |
| official code, our cluster, all arms | 54 scored | 1518.3 | 26/55 | |

The seeds 0-4 PASS is a favourable draw: 4/5 breakouts, as the official code's first 5 seeds (0/5) were an
unfavourable one. Against the official code, 5/15 vs 26/55 gives Fisher p = 0.33, and the means give Welch
t = -1.16: no detectable difference. The faithfulness fixes did not visibly change the breakout rate
(5/15 before and after).
