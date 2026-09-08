"""IMPALA (V-trace, async actor-learner) training entrypoint.

Wires playtrain_trainers.impala.train.train(cfg) to a JSON config file. Config schema
matches ImpalaConfig dataclass; see src/playtrain_trainers/impala/train.py for fields.

Usage:
    uv run python -m playtrain_trainers.train_impala --config configs/impala_quickstart.json
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from pathlib import Path

from playtrain_trainers.impala.train import ImpalaConfig, train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    logging.basicConfig(
        format="[%(levelname)s %(asctime)s] %(message)s", level=logging.INFO
    )

    raw = json.loads(args.config.read_text())
    valid = {f.name for f in dataclasses.fields(ImpalaConfig)}
    ignored = set(raw) - valid
    if ignored:
        logging.warning("Ignoring unknown config keys: %s", sorted(ignored))
    cfg = ImpalaConfig(**{k: v for k, v in raw.items() if k in valid})

    result = train(cfg)
    logging.info("Done: %s", result)


if __name__ == "__main__":
    main()
