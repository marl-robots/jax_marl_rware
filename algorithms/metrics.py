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

import numpy as np

COLUMNS = [
    "environment_steps",
    "updates",
    "mean_episode_returns",
    "entropy",
    "loss",
    "actor_loss",
    "value_loss",
    "reward_std_mean",
    # behavioral signals (commentary)
    "deliveries",
    "block_rate",
    "idle_rate",
    "pickup_rate",
    "deliveries_early",
    "deliveries_mid",
    "deliveries_late",
    "block_early",
    "block_mid",
    "block_late",
]


class CSVLogger:
    def __init__(
        self,
        path: str,
        dict_keys: list[str],
        full_metrics: bool = False,
        resume: bool = False,
    ):
        self.path = path
        self.full_metrics = full_metrics
        d = os.path.dirname(os.path.abspath(path))
        os.makedirs(d, exist_ok=True)
        write_header = not (
            resume and os.path.isfile(path) and os.path.getsize(path) > 0
        )
        self._f = open(path, "a", newline="")
        self._w = csv.writer(self._f)
        if write_header:
            if not full_metrics:
                self._w.writerow(COLUMNS)
            else:
                self.dict_keys = dict_keys
                self._w.writerow(dict_keys)
            self._f.flush()

    def log(self, raw: dict) -> None:
        if self.full_metrics:
            self._w.writerow([raw.get(c, f"{np.finfo(np.float32).max}") for c in self.dict_keys])
            self._f.flush()
        else:
            self._w.writerow([raw.get(c, "") for c in COLUMNS])
            self._f.flush()

    def close(self) -> None:
        try:
            self._f.close()
        except Exception:
            pass
