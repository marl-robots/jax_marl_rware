"""Dynamic environment state (a flax struct dataclass of JAX arrays)."""

from __future__ import annotations

import chex
from flax import struct


@struct.dataclass
class EnvState:
    # agents (N = n_agents)
    agent_x: chex.Array  # int32 [N]
    agent_y: chex.Array  # int32 [N]
    agent_dir: chex.Array  # int32 [N]  (Direction value)
    agent_carrying: chex.Array  # int32 [N]  shelf id (1-based) or 0
    agent_distance_traveled: chex.Array  # int32 [N]
    agent_has_delivered: chex.Array  # bool [N]

    # shelves (S = number of non-highway cells)
    shelf_x: chex.Array  # int32 [S]  current x (moves when carried)
    shelf_y: chex.Array  # int32 [S]  current y

    # request queue, represented as a membership mask over shelves.
    # Order within the queue does not affect dynamics/observations (rware only
    # ever tests membership), so a boolean mask of fixed popcount == queue size
    # captures the semantics exactly.
    in_queue: chex.Array  # bool [S]  shelf i requested?

    step_count: chex.Array  # int32 scalar
    inactive_count: chex.Array  # int32 scalar
    key: chex.PRNGKey
