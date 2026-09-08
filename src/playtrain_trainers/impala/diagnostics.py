"""Lightweight in-process memory diagnostics for the IMPALA loop.

Reads VmRSS from /proc/<pid>/status — Linux-only, zero deps. On non-Linux
(e.g. macOS dev), every function returns NaN so callers can log without
guarding. Designed to be cheap enough to call every actor rollout / learn
step without measurably affecting throughput.

Usage:
    from playtrain_trainers.impala.diagnostics import rss_mb, log_rss
    log_rss("actor", 0, rollouts=42)  # logs: "[mem] actor.0 rollouts=42 rss=812.4MB"
"""
from __future__ import annotations

import logging
import math


def rss_mb(pid: int | str = "self") -> float:
    """Resident set size of the given PID in MB. NaN if not readable."""
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0  # kB -> MB
    except (FileNotFoundError, OSError):
        pass
    return math.nan


def log_rss(role: str, idx: int, **extra) -> None:
    """One-line memory log. extra={'rollouts': 42, 'step': 12800} becomes
    ' rollouts=42 step=12800'."""
    suffix = " ".join(f"{k}={v}" for k, v in extra.items())
    rss = rss_mb()
    # NaN off Linux, where /proc does not exist. Say nothing rather than "nanMB".
    if rss != rss:
        return
    logging.info(f"[mem] {role}.{idx} {suffix} rss={rss:.1f}MB")


def rss_note() -> str:
    """" (rss=812.4MB)" on Linux, and "" anywhere /proc is missing.

    Callers embed this instead of formatting rss_mb() directly, so a machine without
    /proc logs a clean line rather than "rss=nanMB".
    """
    v = rss_mb()
    return f" (rss={v:.1f}MB)" if v == v else ""
