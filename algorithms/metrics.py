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
    "mean_entropy",
    "mean_loss",
    "mean_actor_loss",
    "mean_value_loss",
    "mean_reward_std",
    "episode_time",
    #behavioral signals (commentary)
    "mean_success",
    "mean_success_rate",
    "mean_step_count",
    "mean_FPS",
    "mean_deliveries",
    "mean_block",
    "mean_block_rate",
    "mean_idle_rate",
    "mean_pickup_rate",
    "mean_deliveries_early",
    "mean_deliveries_mid",
    "mean_deliveries_late",
    "mean_block_early",
    "mean_block_mid",
    "mean_block_late",
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
