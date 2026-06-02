"""Headless Pillow renderer for the JAX RWARE env.

Reimplements the look of semitable/robotic-warehouse's pyglet `Viewer`
(rware/rendering.py) without an OpenGL context, so a checkpoint rollout can be
turned into frames anywhere (WSL, CI, no display). Reads an `EnvState` directly.

Geometry/colours are copied from the original. The original renders in pyglet
(origin bottom-left) then flips the framebuffer vertically before returning the
RGB array, so the *returned image* has world y=0 at the top. We draw top-down
directly (row == world y), which reproduces that same image without a flip.
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# colours (from rware/rendering.py)
_WHITE = (255, 255, 255)
_BLACK = (0, 0, 0)
_RED = (255, 0, 0)
_DARKORANGE = (255, 140, 0)
_DARKSLATEBLUE = (72, 61, 139)
_TEAL = (0, 128, 128)
_GOAL_COLOR = (60, 60, 60)

_BACKGROUND_COLOR = _WHITE
_GRID_COLOR = _BLACK
_SHELF_COLOR = _DARKSLATEBLUE
_SHELF_REQ_COLOR = _TEAL
_AGENT_COLOR = _DARKORANGE
_AGENT_LOADED_COLOR = _RED
_AGENT_DIR_COLOR = _BLACK

_GRID_SIZE = 30
_PITCH = _GRID_SIZE + 1  # 31; one-pixel gridline between cells
_SHELF_PADDING = 2

# Direction enum values (UP, DOWN, LEFT, RIGHT) == (0, 1, 2, 3)
_UP, _DOWN, _LEFT, _RIGHT = 0, 1, 2, 3


def _load_font():
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, 18)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


_FONT = _load_font()


def image_size(env):
    """(width, height) in pixels for this env's grid."""
    rows, cols = env.H, env.W
    return 1 + cols * _PITCH, 2 + rows * _PITCH


def render_state(env, state) -> np.ndarray:
    """Render one `EnvState` to an (H, W, 3) uint8 RGB array (top-down)."""
    rows, cols = env.H, env.W
    width, height = image_size(env)
    img = Image.new("RGB", (width, height), _BACKGROUND_COLOR)
    d = ImageDraw.Draw(img)

    # ---- grid lines ----
    for r in range(rows + 1):
        y = _PITCH * r + 1
        d.line([(0, y), (_PITCH * cols, y)], fill=_GRID_COLOR, width=1)
    for c in range(cols + 1):
        x = _PITCH * c + 1
        d.line([(x, 0), (x, _PITCH * rows)], fill=_GRID_COLOR, width=1)

    # ---- goals (filled cell + white "G") ----
    goal_x = np.asarray(env.goal_x)
    goal_y = np.asarray(env.goal_y)
    for gx, gy in zip(goal_x.tolist(), goal_y.tolist()):
        d.rectangle(
            [_PITCH * gx + 1, _PITCH * gy + 1, _PITCH * (gx + 1), _PITCH * (gy + 1)],
            fill=_GOAL_COLOR,
        )
        cx = _PITCH * gx + _PITCH / 2
        cy = _PITCH * gy + _PITCH / 2
        d.text((cx, cy), "G", fill=_WHITE, font=_FONT, anchor="mm")

    # ---- shelves (padded quads; teal if requested) ----
    shelf_x = np.asarray(state.shelf_x).tolist()
    shelf_y = np.asarray(state.shelf_y).tolist()
    in_queue = np.asarray(state.in_queue).tolist()
    p = _SHELF_PADDING
    for sx, sy, req in zip(shelf_x, shelf_y, in_queue):
        color = _SHELF_REQ_COLOR if req else _SHELF_COLOR
        d.rectangle(
            [
                _PITCH * sx + p + 1,
                _PITCH * sy + p + 1,
                _PITCH * (sx + 1) - p,
                _PITCH * (sy + 1) - p,
            ],
            fill=color,
        )

    # ---- agents (hexagon + black direction line) ----
    radius = _GRID_SIZE / 3.0
    half = _GRID_SIZE // 2 + 1
    agent_x = np.asarray(state.agent_x).tolist()
    agent_y = np.asarray(state.agent_y).tolist()
    agent_dir = np.asarray(state.agent_dir).tolist()
    carrying = (np.asarray(state.agent_carrying) > 0).tolist()
    for ax, ay, adir, carry in zip(agent_x, agent_y, agent_dir, carrying):
        cx = _PITCH * ax + half
        cy = _PITCH * ay + half
        verts = [
            (
                cx + radius * math.cos(2 * math.pi * i / 6),
                cy + radius * math.sin(2 * math.pi * i / 6),
            )
            for i in range(6)
        ]
        d.polygon(verts, fill=_AGENT_LOADED_COLOR if carry else _AGENT_COLOR)

        # direction tick (top-down: UP = -y, DOWN = +y, LEFT = -x, RIGHT = +x)
        dx = radius if adir == _RIGHT else (-radius if adir == _LEFT else 0.0)
        dy = radius if adir == _DOWN else (-radius if adir == _UP else 0.0)
        d.line([(cx, cy), (cx + dx, cy + dy)], fill=_AGENT_DIR_COLOR, width=2)

    return np.asarray(img, dtype=np.uint8)
