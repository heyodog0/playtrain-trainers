# U13: official DreamerV3 at the score-file commit 2411f7d1, on our cluster

- Source: `danijar/dreamerv3 @ 2411f7d136832378c0291c587cdbf2fca6506873` (2024-04-15), the
  commit `scores/atari100k-dreamerv3.json.gz` was produced at. It is unmodified and vendored at
  `playtrain-internal/dreamerv3-loop/reference/dreamerv3_2411f7d1/`.
- Environment: job 48000713 (partition shared). jax[cuda12] 0.4.26, tensorflow-cpu 2.16.1,
  tfp 0.24.0, chex 0.1.86, optax 0.2.2, ale-py 0.8.1, numpy<2, py3.11. The ROM is the
  frostbite.bin from the pinned run's AutoROM dir (`ALE_ROM_PATH`).
- Runs: `--configs atari100k --task <task> --seed {0..4}` on kempner_h100, 1 H100 each,
  ~1.7 h per seed.
  - arm A, job 48000722: `atari_frostbite`, which is env.atari: gray, sticky, pooling 2, noops 0.
  - arm B, job 48000726: `atari100k_frostbite`, which is env.atari100k: RGB, not sticky,
    length 100000, noops 0.
  - Both arms use batch 16x65, train_ratio 256, replay_context 1, and 110k agent steps
    (440k frames). All 10 tasks ended with EXIT=0.
- Aggregation: mean training-episode score over episodes whose logged step (frames) is in
  [350k, 400k). Regenerate with `tools/dreamerv3/dv3_official_summary.py
  results/dreamerv3/official_2411f7d1`; `plot.py` redraws `curves.png`.
- Raw data: `raw/` holds metrics.jsonl, config.yaml, slurm logs and sbatch files.
  `<task>.json` uses the same shape as `../official_jax_frostbite.json`.

| arm | n | last-50k mean | sd | escaped (>=1000) | per seed |
|---|---|---|---|---|---|
| published score file | 5 | 2938.1 | 1603.9 | 4 | 350 / 2847 / 2944 / 4190 / 4359 |
| e3f0224 on our cluster | 5 | 306.2 | 74.2 | 0 | 250 / 252 / 282 / 316 / 430 |
| **2411f7d1 atari_frostbite** | 5 | **834.2** | 1180.4 | 1 | 248 / 275 / 302 / 403 / 2943 |
| **2411f7d1 atari100k_frostbite** | 5 | **629.6** | 627.4 | 1 | 259 / 373 / 377 / 391 / 1748 |
| 2411f7d1 pooled | 10 | 731.9 | 897.7 | 2 | |
| our port | 15 | 890.6 | 886.3 | 5 | |
