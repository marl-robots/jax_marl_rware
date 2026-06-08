"""Vectorized movement / collision resolution matching rware's NetworkX logic.

rware builds a directed graph of (start -> requested) edges and, per weakly
connected component:
  * if a cycle exists: length-2 cycles (direct swaps) commit nothing; longer
    cycles (rotations) commit every agent on the cycle.
  * otherwise (a DAG): commit every agent on the single longest path.

Key structural fact: each occupied cell emits exactly one out-edge (each agent
proposes exactly one target), so the graph is a *functional graph* — every
weakly connected component contains at most one cycle, with trees feeding into
it. That lets us resolve everything with bounded, fully-vectorized iteration:

  1. detect rotation cycles (length >= 3) -> those agents move; swaps don't;
  2. for the acyclic remainder, an agent advances iff its target cell is empty
     or vacated by another *acyclic* mover, winning per-cell contests by longest
     upstream chain (the dag-longest-path tie-break), then lowest agent index.

n_agents is static, so the Python loops below unroll under jit.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .config import Action, Config, Direction


def _req_location(cfg: Config, x, y, d, is_forward):
    """Clamped target cell for a FORWARD action (else stay), per Agent.req_location."""
    H, W = cfg.grid_height, cfg.grid_width
    up = (x, jnp.maximum(0, y - 1))
    down = (x, jnp.minimum(H - 1, y + 1))
    left = (jnp.maximum(0, x - 1), y)
    right = (jnp.minimum(W - 1, x + 1), y)
    tx = jnp.where(
        d == Direction.UP, up[0],
        jnp.where(d == Direction.DOWN, down[0],
        jnp.where(d == Direction.LEFT, left[0], right[0])),
    )
    ty = jnp.where(
        d == Direction.UP, up[1],
        jnp.where(d == Direction.DOWN, down[1],
        jnp.where(d == Direction.LEFT, left[1], right[1])),
    )
    tx = jnp.where(is_forward, tx, x)
    ty = jnp.where(is_forward, ty, y)
    return tx, ty


def resolve_moves(cfg, agent_x, agent_y, agent_dir, req_action, agent_carrying,
                  grid_agents, grid_shelfs):
    """Return (moves[N] bool, tx[N], ty[N], cancelled[N] bool).

    moves[i]   -> whether agent i actually advances to (tx[i], ty[i]).
    cancelled  -> a carrying FORWARD agent whose move was cancelled by a standing
                  shelf (its req_action becomes NOOP, matching rware).
    """
    N = cfg.n_agents
    W = cfg.grid_width

    is_forward = req_action == Action.FORWARD
    tx, ty = _req_location(cfg, agent_x, agent_y, agent_dir, is_forward)
    wants_diff = is_forward & ((tx != agent_x) | (ty != agent_y))

    # --- cancel rule: carrying agent moving into a standing (uncarried) shelf ---
    standing_shelf = grid_shelfs[ty, tx] != 0
    occ_id = grid_agents[ty, tx]                      # 1-based agent id at target, 0 empty
    occ_idx = jnp.clip(occ_id - 1, 0, N - 1)
    occ_carrying = (occ_id > 0) & (agent_carrying[occ_idx] != 0)
    carrying = agent_carrying != 0
    cancelled = wants_diff & carrying & standing_shelf & (~occ_carrying)

    is_forward = is_forward & (~cancelled)
    tx = jnp.where(is_forward, tx, agent_x)
    ty = jnp.where(is_forward, ty, agent_y)
    wants_diff = is_forward & ((tx != agent_x) | (ty != agent_y))

    # blocker[i] = index of agent currently occupying agent i's target cell, or -1
    occ_id = grid_agents[ty, tx]
    blocker = jnp.where(wants_diff & (occ_id > 0), occ_id - 1, -1)
    nxt = blocker  # functional-graph pointer; -1 == points at empty cell

    idx = jnp.arange(N)

    # --- cycle detection (follow nxt; detect return to self) ---
    in_cycle = jnp.zeros(N, dtype=bool)
    cyclen = jnp.zeros(N, dtype=jnp.int32)
    cur = nxt
    for t in range(1, N + 1):
        hit = (cur == idx) & (cur >= 0)
        newly = hit & (~in_cycle)
        cyclen = jnp.where(newly, t, cyclen)
        in_cycle = in_cycle | hit
        safe = jnp.where(cur >= 0, cur, 0)
        cur = jnp.where(cur >= 0, nxt[safe], -1)

    rotates = in_cycle & (cyclen >= 3)   # rotations commit; 2-cycles (swaps) do not

    # acyclic forward movers (candidates for chain-into-empty resolution)
    acyclic = wants_diff & (~in_cycle)

    # --- longest upstream chain length T[i] over acyclic feeders (tie-break key) ---
    # predmat[i, j] == True iff acyclic agent j targets agent i's current cell.
    predmat = (nxt[None, :] == idx[:, None]) & acyclic[None, :]
    T = jnp.ones(N, dtype=jnp.int32)
    for _ in range(N):
        upstream = jnp.where(predmat, T[None, :], 0)
        tnew = 1 + jnp.max(upstream, axis=1)
        T = jnp.where(acyclic, jnp.maximum(T, tnew), T)

    cell_id = ty * W + tx  # per-agent target cell id

    # --- fixed-point: resolve acyclic moves into empty / vacated cells ---
    moves = rotates
    BIG = N + 1
    for _ in range(N + 1):
        # which target cells are already claimed by a committed mover?
        claimed = jnp.zeros(cfg.grid_height * W, dtype=bool)
        claimed = claimed.at[jnp.where(moves, cell_id, 0)].max(moves)

        occ = blocker
        occ_safe = jnp.where(occ >= 0, occ, 0)
        occ_moves_acyclic = (occ >= 0) & moves[occ_safe] & acyclic[occ_safe]
        free = acyclic & ((occ < 0) | occ_moves_acyclic)
        candidate = free & (~moves) & (~claimed[cell_id])

        # per-cell contest: highest (T, -index) wins
        key = jnp.where(candidate, T * BIG - idx, -1)
        best = jnp.full(cfg.grid_height * W, -1, dtype=key.dtype)
        best = best.at[cell_id].max(key)
        win = candidate & (key == best[cell_id]) & (key >= 0)
        moves = moves | win

    return moves, tx, ty, cancelled
