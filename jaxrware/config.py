"""Static environment configuration for the JAX RWARE port.

Mirrors semitable/robotic-warehouse registration (rware/__init__.py) exactly for
the configs we target. All values here are *static* (plain Python ints / enums)
so they can be baked into a jitted env as compile-time constants.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class Action(IntEnum):
    NOOP = 0
    FORWARD = 1
    LEFT = 2
    RIGHT = 3
    TOGGLE_LOAD = 4


class Direction(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class RewardType(IntEnum):
    GLOBAL = 0
    INDIVIDUAL = 1
    TWO_STAGE = 2


# (shelf_rows, shelf_columns) per named size, from rware/__init__.py:_sizes
_SIZES = {
    "tiny": (1, 3),
    "small": (2, 3),
    "medium": (2, 5),
    "large": (3, 5),
}

# difficulty -> request_queue multiplier (rware/__init__.py:_difficulty)
_DIFFICULTY = {"easy": 2.0, "normal": 1.0, "hard": 0.5}


@dataclass(frozen=True)
class Config:
    """Static config for a Warehouse instance."""

    shelf_columns: int
    shelf_rows: int
    column_height: int
    n_agents: int
    sensor_range: int = 1
    request_queue_size: int = 1
    max_steps: int = 500
    max_inactivity_steps: int | None = None
    reward_type: RewardType = RewardType.INDIVIDUAL
    msg_bits: int = 0  # we only support 0 (default for all registered envs)

    @property
    def grid_height(self) -> int:
        return (self.column_height + 1) * self.shelf_rows + 2

    @property
    def grid_width(self) -> int:
        return (2 + 1) * self.shelf_columns + 1


def make_config(size: str, n_agents: int, difficulty: str = "normal") -> Config:
    """Build a Config matching ``rware-{size}-{n}ag{-diff}-v2``."""
    assert size in _SIZES, f"unknown size {size!r}, choose from {list(_SIZES)}"
    assert difficulty in _DIFFICULTY, f"unknown difficulty {difficulty!r}"
    shelf_rows, shelf_columns = _SIZES[size]
    return Config(
        shelf_columns=shelf_columns,
        shelf_rows=shelf_rows,
        column_height=8,
        n_agents=n_agents,
        sensor_range=1,
        request_queue_size=int(n_agents * _DIFFICULTY[difficulty]),
        max_steps=500,
        max_inactivity_steps=None,
        reward_type=RewardType.INDIVIDUAL,
        msg_bits=0,
    )


# Convenience: the proven MAPPO target.
def tiny_4ag() -> Config:
    return make_config("tiny", 4, "normal")
