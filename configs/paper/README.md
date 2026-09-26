# The paper's training configurations

One JSON per run, as each run wrote it (`config.json` in its output directory),
except where noted. Pass any of them to the trainer that wrote it:

    python -m playtrain_trainers.train_impala    --config <file>   # has total_steps
    python -m playtrain_trainers.train_ppo_clean --config <file>   # has total_timesteps

| folder | paper | runs |
|---|---|---|
| `suite/` | fig:suite_trainers, fig:suite_enc_impala, fig:suite_enc_ppo, tab:eval, tab:hyperparams | 24 games x 3 seeds, for each of IMPALA and PPO with the IMPALA-CNN and Nature-CNN encoders (288) |
| `learning_curves/` | fig:learning | every run whose curves the composite draws that is not already in `suite/` |
| `human_wallclock/` | fig:human_wallclock | the agent curves for flappy_bird and VVVVVV (the win-bonus runs, `vvwin_*`); the other six games use `suite/` |
| `train_throughput/` | tab:train-throughput (a) | the four rows' templates, rebuilt from the batch script (`reproduction/runs/44748571/` in the playtrain repo); the benchmark driver sets `game` per task and stops each run itself |
| `dbuf_ablation/` | tab:dbuf-ablation | the two arms, rebuilt from the batch script (`reproduction/runs/44861569/` in the playtrain repo) |

Paths inside the files (`log_dir` and similar) point at the cluster tree the runs
used; set `log_dir` before running one. Some runs were
made under a game's original file name (e.g. `breakout_multi`, `qbert_v2`,
`flappy_bird_dunk2`); the playtrain repo's `reproduction/manifests/variant_names.tsv` maps them to the
paper's names.
