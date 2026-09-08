"""playtrain-trainers — fast RL trainers for playtrain (playtrain.runtime) environments.

Two environment-agnostic trainers that reach environments through ``playtrain.runtime``:

  * ``playtrain_trainers.impala``    — IMPALA (V-trace) with optional LSTM, plus a
    centralized-GPU-inference vectorized actor path (the 7k -> 100k SPS work).
  * ``playtrain_trainers.train_ppo_clean`` — PPO with optional LSTM.

Generalization (seed pools) and backends other than PlayTrain (e.g. MiniGrid) are
opt-in and supplied by the consumer via ``playtrain_trainers.plugins``.
"""

__version__ = "0.1.0"
