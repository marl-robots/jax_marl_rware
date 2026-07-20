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
    "episode_returns_mean",
    "entropy_mean",
    "loss_mean",
    "actor_loss_mean",
    "value_loss_mean",
    "reward_std_mean",
# behavioral signals (commentary)
    "deliveries_mean",
    "block_rate_mean",
    "idle_rate_mean",
    "pickup_rate_mean",
    "deliveries_early_mean",
    "deliveries_mid_mean",
    "deliveries_late_mean",
    "block_early_mean",
    "block_mid_mean",
    "block_late_mean",
]
COLUMNS_DQN = [
    "environment_steps",
    "updates",
    "episode_returns_mean",
    "loss_mean",
    "epsilon",
    "reward_std_mean",
    "deliveries_mean",
    "block_rate_mean",
    "idle_rate_mean",
    "pickup_rate_mean",
    "deliveries_early_mean",
    "deliveries_mid_mean",
    "deliveries_late_mean",
    "block_early_mean",
    "block_mid_mean",
    "block_late_mean",
]

class CSVLogger:
    def __init__(self, path: str, resume: bool = False, isdqn: bool = False):
        self.path = path
        self.isdqn = isdqn
        d = os.path.dirname(os.path.abspath(path))
        os.makedirs(d, exist_ok=True)
        write_header = not (
            resume and os.path.isfile(path) and os.path.getsize(path) > 0
        )
        self._f = open(path, "a", newline="")
        self._w = csv.writer(self._f)
        if write_header:
            if isdqn:
                self._w.writerow(COLUMNS_DQN)
            else:
                self._w.writerow(COLUMNS)
            self._f.flush()

    def log(self, row: dict) -> None:
        if self.isdqn:
            self._w.writerow([row.get(c, "") for c in COLUMNS_DQN])
        else:
            self._w.writerow([row.get(c, "") for c in COLUMNS])
        self._f.flush()

    def close(self) -> None:
        try:
            self._f.close()
        except Exception:
            pass
