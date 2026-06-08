"""Static layout generation, ported from Warehouse._make_layout_from_params.

Produces numpy arrays computed once at construction time (not traced):
  - highways[H, W] uint8: 1 where a cell is a corridor/goal (no shelf storage)
  - goals: list of (x, y) goal cells
  - shelf_xy[S, 2] int32: home (x, y) of each shelf, in row-major (y, x) order,
    matching the shelf-id assignment order used by rware.reset().
"""

from __future__ import annotations

import numpy as np

from .config import Config


def build_highways(cfg: Config) -> np.ndarray:
    """Replicate Warehouse._make_layout_from_params.highway_func exactly."""
    H, W = cfg.grid_height, cfg.grid_width
    col_h = cfg.column_height
    highways = np.zeros((H, W), dtype=np.uint8)
    for x in range(W):
        for y in range(H):
            is_on_vertical_highway = x % 3 == 0
            is_on_horizontal_highway = y % (col_h + 1) == 0
            is_on_delivery_row = y == H - 1
            is_on_queue = (y > H - (col_h + 3)) and (x == W // 2 - 1 or x == W // 2)
            highways[y, x] = int(
                is_on_vertical_highway
                or is_on_horizontal_highway
                or is_on_delivery_row
                or is_on_queue
            )
    return highways


def goal_cells(cfg: Config) -> list[tuple[int, int]]:
    """Two goal cells (x, y), as in Warehouse._make_layout_from_params."""
    H, W = cfg.grid_height, cfg.grid_width
    return [(W // 2 - 1, H - 1), (W // 2, H - 1)]


def shelf_home_positions(highways: np.ndarray) -> np.ndarray:
    """Shelf home (x, y) for every non-highway cell, row-major (y outer, x inner).

    Matches rware.reset(): shelves are created via
    ``zip(np.indices(grid)[0].reshape(-1), np.indices(grid)[1].reshape(-1))``
    which iterates y (rows) then x (cols), skipping highway cells. Shelf ids are
    assigned 1..S in this order.
    """
    H, W = highways.shape
    xs = []
    ys = []
    for y in range(H):
        for x in range(W):
            if not highways[y, x]:
                xs.append(x)
                ys.append(y)
    return np.stack([np.array(xs, np.int32), np.array(ys, np.int32)], axis=1)
