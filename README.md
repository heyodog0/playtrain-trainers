# playtrain-trainers

[**Paper**](https://arxiv.org/abs/2609.09059) | [**PlayTrain**](https://github.com/heyodog0/playtrain) | [**Project page**](https://playtrain.org)

RL trainers for [PlayTrain](https://github.com/heyodog0/playtrain) environments. They
produced the training results in the paper.

There are two. Both reach environments only through `playtrain.runtime`.

- IMPALA with V-trace. Async actor-learner, optional LSTM. The V-trace math and losses
  are equal against FAIR's torchbeast. See `tests/test_impala_vtrace.py`.
- PPO. Optional LSTM.

## Install

Needs Python 3.11 or newer, plus clang and cargo. On macOS the default `python3` is
often older, so check with `python3 --version` first.

`playtrain` is a dependency and is not on PyPI yet, so clone both repositories side by
side. With [uv](https://docs.astral.sh/uv/) that is the only requirement, since
`pyproject.toml` points `playtrain` at `../playtrain`:

```console
$ git clone https://github.com/heyodog0/playtrain
$ git clone https://github.com/heyodog0/playtrain-trainers
$ cd playtrain-trainers
$ uv venv && uv pip install -e .
```

If PlayTrain is somewhere else, install it by path and skip the pinned source:

```console
$ uv pip install -e <path-to-playtrain> && uv pip install -e . --no-sources
```

Without uv, install PlayTrain first. `pip` ignores `[tool.uv.sources]`, so it looks for
`playtrain` on PyPI and fails with `No matching distribution found` if this package goes
first.

```console
$ cd playtrain-trainers
$ python3 -m venv .venv
$ .venv/bin/pip install -e ../playtrain
$ .venv/bin/pip install -e .
```

Either way this builds PlayTrain's native backend, which takes about a minute the first
time. That is why clang and cargo are needed.

## Train with IMPALA

A run is defined by a JSON config. Fields not listed are left at their defaults, which
are in `ImpalaConfig` in `src/playtrain_trainers/impala/train.py`.

```console
$ python -m playtrain_trainers.train_impala --config configs/impala_quickstart.json
```

A minimal config:

```json
{
  "game": "breakout",
  "env_backend": "playtrain",
  "inference_mode": "vec",
  "total_steps": 5000000,
  "obs_shape": [3, 64, 64],
  "num_actions": 8,
  "net": "impala",
  "vec_workers": 12,
  "vec_env_threads": 5,
  "vec_double_buffer": true,
  "batch_size": 32,
  "unroll_length": 64,
  "learning_rate": 0.0005,
  "device": "auto",
  "log_dir": "outputs/run"
}
```

`env_backend` must be set to `"playtrain"`. It defaults to `"minigrid"`, and so does
`game`, so a config without it will not run a PlayTrain game.

`inference_mode` selects the rollout topology. `"vec"` runs environments on an
in-process C++ threadpool and is what you want on one machine. The default,
`"shared_cpu"`, uses separate actor processes.

The number of environment threads is `vec_workers` times `vec_env_threads`. Raise it
until the GPU stops being fed. `vec_double_buffer` overlaps stepping with inference and
is worth having on.

`obs_shape` is channels-first and `num_actions` is 8 for every game in the shipped
catalog, since they are all authored against `default8`.

## Train with PPO

```console
$ python -m playtrain_trainers.train_ppo_clean --config <config>.json
```

PPO takes a different config schema. It counts `total_timesteps` rather than
`total_steps`, and it sizes rollouts with `n_envs` and `n_steps` rather than the `vec_`
fields. The full set is `Config` in `src/playtrain_trainers/train_ppo_clean.py`.

```json
{
  "game": "breakout",
  "env_backend": "playtrain",
  "total_timesteps": 5000000,
  "net": "impala",
  "n_envs": 64,
  "n_steps": 128,
  "n_minibatches": 4,
  "n_epochs": 3,
  "learning_rate": 0.00025,
  "gamma": 0.999,
  "gae_lambda": 0.95,
  "clip_coef": 0.2,
  "ent_coef": 0.01,
  "vec_backend": "native",
  "native_env_threads": 8,
  "device": "auto",
  "log_dir": "outputs/ppo_run"
}
```

Set `vec_backend` to `"native"` and `native_env_threads` above zero to use the C++
threadpool. PPO reuses each frame across `n_epochs`, so it is more sample-efficient than
IMPALA and slower in wall-clock terms at the same step count.

## Output

Both trainers write TensorBoard scalars to `log_dir/tb` and checkpoints to `log_dir`.

```console
$ tensorboard --logdir outputs/run/tb
```

The scalar to watch is `charts/mean_episode_return`. `charts/sps` is throughput.

## Use with PlayTrain

The trainers do not contain games. They request one from `playtrain.runtime`.

```python
from playtrain.runtime import NativeVecEnv

venv = NativeVecEnv(game="breakout", num_envs=64, num_threads=8)
```

A game created with `playtrain-generate` is trained by putting its name in the config.
For a game outside the shipped catalog, set the directory as well.

```json
{ "game": "my_game", "vec_games_dir": "games/js" }
```

## Layout

| directory | contents |
|---|---|
| `src/playtrain_trainers/` | the trainers |
| `configs/` | three configs: a quickstart, a full-node run, and a throughput template |
| `benchmarks/` | training-throughput measurement |
| `tests/` | the test suite |
| `tools/` | `remote_env_actor.py`, the worker for running environments on separate CPU nodes |

## Extras

`wandb` adds Weights & Biases and TensorBoard tracking, with imageio for eval clips. `dev` adds pytest and ruff.

Two capabilities are opt-in through `playtrain_trainers.plugins`: generalization seed
pools, and non-PlayTrain backends such as MiniGrid. With no provider registered the
trainers are PlayTrain-only.

## License

MIT.
