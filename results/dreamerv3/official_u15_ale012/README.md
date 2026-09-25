# U15a: official DreamerV3 on the port's ALE (ale-py 0.12.1)

- Source: pinned e3f0224 with ONE edit, `embodied/envs/atari.py:54`, which changes `setInt(b'random_seed', ...)`
  to `setInt('random_seed', ...)` because 0.12 only accepts str keys (D-026). The diff is in `raw/venv_48089986.out`.
- Venv job 48089986: jax 0.4.33, ale_py 0.12.1, everything else as the pinned venv. The Frostbite ROM
  md5 is 4ca73eb9..., the same file as the pinned run, the 2023 atari-py setup and our port.
- Runs: job 48090345_[0-9], `--configs atari100k --task atari100k_frostbite`, H100. All 10 logs print
  `ale_py 0.12.1 [CudaDevice(id=0)]`.

| arm | n | last-50k mean | broke out (>=1000) | per seed |
|---|---|---|---|---|
| **official, ale-py 0.12.1** | 10 | **1822.4** (sd 1233.4) | **6/10** | 2721 / 403 / 2699 / 3328 / 2426 / 275 / 416 / 908 / 3323 / 1725 |
| official, ale-py 0.9.0, H100 bf16 | 20 | 1595.9 | 10/20 | U12 + U14 `seeds` |
| our port (ale-py 0.12.1) | 15 | 890.6 | 5/15 | |

Against ale-py 0.9.0: Welch t = 0.46, and Fisher p(fewer breakouts on 0.12.1) = 0.82. **The ALE version
does not explain the port's lower breakout rate.**

Regenerate: `summary.json` from `raw/`; the U14 plot script pattern applies.
