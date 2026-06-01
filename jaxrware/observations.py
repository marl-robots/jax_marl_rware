"""FLATTENED observation builder, matching Warehouse._get_default_obs (fast path).

Per-agent layout (msg_bits == 0, sensor_range == sr):
  self:  [x, y, carrying] + onehot(dir, 4) + [on_highway]          (8 values)
  then, for each of (2*sr+1)^2 sensor cells in row-major (dy, dx) order:
        [has_agent] + onehot(other_dir, 4) + [has_shelf, shelf_requested]
  where an empty cell writes onehot == [1,0,0,0] for direction (matching rware).

Total length = 8 + (2*sr+1)^2 * 7. For sr == 1 -> 71.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def obs_length(cfg) -> int:
    cells = (2 * cfg.sensor_range + 1) ** 2
    return 8 + cells * (1 + 4 + 2)


def _onehot4(v):
    return jax.nn.one_hot(v, 4, dtype=jnp.float32)


def build_obs(cfg, agent_x, agent_y, agent_dir, agent_carrying,
              grid_agents, grid_shelfs, in_queue, highways):
    """Return obs[N, obs_length] float32."""
    sr = cfg.sensor_range
    N = cfg.n_agents

    # padded helper grids (pad with sr zeros so windows never go out of bounds)
    pad = ((sr, sr), (sr, sr))
    # agent direction + 1 at each agent cell (0 == no agent)
    dirp1 = jnp.zeros_like(grid_agents)
    dirp1 = dirp1.at[agent_y, agent_x].set(agent_dir + 1)
    dirp1_p = jnp.pad(dirp1, pad)

    shelf_present = (grid_shelfs > 0).astype(jnp.int32)
    shelf_present_p = jnp.pad(shelf_present, pad)

    # requested mask laid onto the grid via shelf id at each cell
    sid = jnp.clip(grid_shelfs - 1, 0, in_queue.shape[0] - 1)
    requested = ((grid_shelfs > 0) & in_queue[sid]).astype(jnp.int32)
    requested_p = jnp.pad(requested, pad)

    win = 2 * sr + 1

    def per_agent(x, y, d, carry):
        # self block
        self_block = jnp.concatenate([
            jnp.array([x, y], dtype=jnp.float32),
            jnp.array([(carry > 0).astype(jnp.float32)]),
            _onehot4(d),
            jnp.array([highways[y, x].astype(jnp.float32)]),
        ])

        # sensor windows: padded index of agent cell is (y+sr, x+sr); window top-left
        # corresponds to original (y-sr, x-sr) == padded (y, x).
        dwin = jax.lax.dynamic_slice(dirp1_p, (y, x), (win, win)).reshape(-1)
        swin = jax.lax.dynamic_slice(shelf_present_p, (y, x), (win, win)).reshape(-1)
        rwin = jax.lax.dynamic_slice(requested_p, (y, x), (win, win)).reshape(-1)

        has_agent = (dwin > 0).astype(jnp.float32)              # [cells]
        other_dir = jnp.maximum(dwin - 1, 0)                    # [cells]
        dir_oh = _onehot4(other_dir)                            # [cells, 4]
        has_shelf = swin.astype(jnp.float32)                    # [cells]
        req = rwin.astype(jnp.float32)                          # [cells]

        per_cell = jnp.concatenate([
            has_agent[:, None], dir_oh, has_shelf[:, None], req[:, None]
        ], axis=1).reshape(-1)                                  # [cells*7]

        return jnp.concatenate([self_block, per_cell])

    return jax.vmap(per_agent)(agent_x, agent_y, agent_dir, agent_carrying)
