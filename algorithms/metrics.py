"""Append-only CSV metrics logger.

Columns mirror the marlbase reference `results.csv` subset we can compare
against (deep_marl_data/rware_tiny_4ag/mappo/2/results.csv): per-update
environment_steps / updates plus the scalars we track. Each row is
self-indexed by `updates`, so rows written from an *unordered* io_callback
remain sortable after the fact.
"""

from __future__ import annotations

import csv
import os

COLUMNS = [
    "environment_steps",
    "updates",
    "mean_episode_returns",
    "entropy",
    "loss",
    "actor_loss",
    "value_loss",
    "reward_std_mean",
]


class CSVLogger:
    def __init__(self, path: str, resume: bool = False):
        self.path = path
        d = os.path.dirname(os.path.abspath(path))
        os.makedirs(d, exist_ok=True)
        write_header = not (resume and os.path.isfile(path) and os.path.getsize(path) > 0)
        self._f = open(path, "a", newline="")
        self._w = csv.writer(self._f)
        if write_header:
            self._w.writerow(COLUMNS)
            self._f.flush()

    def log(self, row: dict) -> None:
        self._w.writerow([row.get(c, "") for c in COLUMNS])
        self._f.flush()

    def close(self) -> None:
        try:
            self._f.close()
        except Exception:
            pass
