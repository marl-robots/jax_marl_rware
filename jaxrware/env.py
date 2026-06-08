"""JAX port of semitable/robotic-warehouse Warehouse (FLATTENED obs).

Pure, jittable, vmap-friendly. Static config baked in at construction; dynamic
state is an EnvState of JAX arrays. Semantics follow rware/warehouse.py exactly
(see collision.py and observations.py for the tricky parts).
"""

from __future__ import annotations

import functools
import time

import jax
import jax.numpy as jnp
import numpy as np

from .collision import resolve_moves
from .config import Action, Config, Direction, RewardType
from .layout import build_highways, goal_cells, shelf_home_positions
from .observations import build_obs, obs_length
from .state import EnvState

# rotation lookup tables (indexed by Direction enum value), from Agent.req_direction
_RIGHT_MAP = np.array([3, 2, 0, 1], dtype=np.int32)  # turn clockwise
_LEFT_MAP = np.array([2, 3, 1, 0], dtype=np.int32)   # turn counter-clockwise


class Warehouse:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.H = cfg.grid_height
        self.W = cfg.grid_width
        highways = build_highways(cfg)
        self.highways = jnp.asarray(highways)
        shelf_xy = shelf_home_positions(highways)  # [S, 2] (x, y)
        self.shelf_home_x = jnp.asarray(shelf_xy[:, 0])
        self.shelf_home_y = jnp.asarray(shelf_xy[:, 1])
        self.n_shelves = int(shelf_xy.shape[0])
        goals = goal_cells(cfg)  # [(x, y), ...]
        self.goal_x = jnp.asarray(np.array([g[0] for g in goals], np.int32))
        self.goal_y = jnp.asarray(np.array([g[1] for g in goals], np.int32))
        self.n_goals = len(goals)
        self.right_map = jnp.asarray(_RIGHT_MAP)
        self.left_map = jnp.asarray(_LEFT_MAP)
        self.obs_dim = obs_length(cfg)
        self.num_actions = len(Action)

    # ---- grids -------------------------------------------------------------
    def _grids(self, state: EnvState):
        N = self.cfg.n_agents
        ga = jnp.zeros((self.H, self.W), jnp.int32)
        ga = ga.at[state.agent_y, state.agent_x].set(jnp.arange(N, dtype=jnp.int32) + 1)
        gs = jnp.zeros((self.H, self.W), jnp.int32)
        gs = gs.at[state.shelf_y, state.shelf_x].set(
            jnp.arange(self.n_shelves, dtype=jnp.int32) + 1
        )
        return ga, gs

    def _obs(self, state: EnvState):
        ga, gs = self._grids(state)
        return build_obs(
            self.cfg, state.agent_x, state.agent_y, state.agent_dir,
            state.agent_carrying, ga, gs, state.in_queue, self.highways,
        )

    # ---- reset -------------------------------------------------------------
    @functools.partial(jax.jit, static_argnums=0)
    def reset(self, key):
        N = self.cfg.n_agents
        k_loc, k_dir, k_q = jax.random.split(key, 3)

        # agent locations: choice over H*W without replacement -> (y, x)
        locs = jax.random.permutation(k_loc, self.H * self.W)[:N]
        agent_y = (locs // self.W).astype(jnp.int32)
        agent_x = (locs % self.W).astype(jnp.int32)
        agent_dir = jax.random.randint(k_dir, (N,), 0, 4).astype(jnp.int32)

        # request queue: choice over shelves without replacement
        qperm = jax.random.permutation(k_q, self.n_shelves)
        in_queue = jnp.zeros((self.n_shelves,), bool)
        in_queue = in_queue.at[qperm[: self.cfg.request_queue_size]].set(True)

        state = EnvState(
            agent_x=agent_x,
            agent_y=agent_y,
            agent_dir=agent_dir,
            agent_carrying=jnp.zeros((N,), jnp.int32),
            agent_has_delivered=jnp.zeros((N,), bool),
            shelf_x=self.shelf_home_x,
            shelf_y=self.shelf_home_y,
            in_queue=in_queue,
            step_count=jnp.array(0, jnp.int32),
            inactive_count=jnp.array(0, jnp.int32),
            distance_traveled=jnp.zeros((N,),jnp.int32),
            key=key,
        )
        return state, self._obs(state)

    # ---- step --------------------------------------------------------------
    @functools.partial(jax.jit, static_argnums=0)
    def step(self, state: EnvState, actions):
        start_time=time.time()
        cfg = self.cfg
        N = cfg.n_agents
        actions = actions.astype(jnp.int32)
        ga, gs = self._grids(state)

        moves, tx, ty, _cancelled = resolve_moves(
            cfg, state.agent_x, state.agent_y, state.agent_dir, actions,
            state.agent_carrying, ga, gs,
        )

        carrying = state.agent_carrying  # 1-based shelf id or 0
        is_carry = carrying > 0
        rewards = jnp.zeros((N,), jnp.float32)

        # --- FORWARD (committed) ---
        old_x, old_y = state.agent_x, state.agent_y
        new_x = jnp.where(moves, tx, state.agent_x)
        new_y = jnp.where(moves, ty, state.agent_y)

        # --- LEFT / RIGHT rotation ---
        new_dir = jnp.where(
            actions == Action.RIGHT, self.right_map[state.agent_dir],
            jnp.where(actions == Action.LEFT, self.left_map[state.agent_dir],
                      state.agent_dir),
        )

        # --- TOGGLE_LOAD ---
        on_highway = self.highways[state.agent_y, state.agent_x] > 0
        shelf_here = gs[state.agent_y, state.agent_x]  # id at agent's (unchanged) cell
        pickup = (actions == Action.TOGGLE_LOAD) & (~is_carry) & (shelf_here > 0)
        drop = (actions == Action.TOGGLE_LOAD) & is_carry & (~on_highway)

        new_carrying = jnp.where(pickup, shelf_here, carrying)
        new_carrying = jnp.where(drop, 0, new_carrying)

        # two-stage: +0.5 when returning a delivered shelf
        two_stage = cfg.reward_type == RewardType.TWO_STAGE
        rewards = rewards + jnp.where(
            drop & state.agent_has_delivered & two_stage, 0.5, 0.0
        ).astype(jnp.float32)
        has_delivered = jnp.where(drop, False, state.agent_has_delivered)

        # move carried shelves with their agent. Non-carrying agents scatter to
        # an out-of-bounds slot (dropped) so they cannot clobber a real shelf;
        # carriers hold distinct shelf ids, so there are no index collisions.
        carry_after = new_carrying > 0
        sidx = jnp.where(carry_after, new_carrying - 1, self.n_shelves)
        shelf_x = state.shelf_x.at[sidx].set(new_x, mode="drop")
        shelf_y = state.shelf_y.at[sidx].set(new_y, mode="drop")

        state = state.replace(
            agent_x=new_x, agent_y=new_y, agent_dir=new_dir,
            agent_carrying=new_carrying, agent_has_delivered=has_delivered,
            shelf_x=shelf_x, shelf_y=shelf_y,
        )

        # --- delivery detection (sequential over the static goal cells) ---
        ga2, gs2 = self._grids(state)
        in_queue = state.in_queue
        delivered_any = jnp.array(False)
        deliveries = jnp.zeros((N,), jnp.int32)  # per-agent delivery count (commentary)
        key = state.key

        def deliver_at(carry, g):
            in_queue, rewards, has_delivered, delivered_any, deliveries, key, gs2, ga2 = carry
            gx, gy = g
            shelf_id = gs2[gy, gx]
            agent_id = ga2[gy, gx]
            sidx = jnp.clip(shelf_id - 1, 0, self.n_shelves - 1)
            aidx = jnp.clip(agent_id - 1, 0, N - 1)
            is_delivery = (shelf_id > 0) & in_queue[sidx]

            # remove delivered shelf from queue and add a random non-queued shelf
            cleared = in_queue.at[sidx].set(jnp.where(is_delivery, False, in_queue[sidx]))
            key, sub = jax.random.split(key)
            cand = (~cleared).astype(jnp.float32)
            logits = jnp.log(cand + 1e-12)
            new_req = jax.random.categorical(sub, logits)
            cleared = cleared.at[new_req].set(
                jnp.where(is_delivery, True, cleared[new_req])
            )
            in_queue = jnp.where(is_delivery, cleared, in_queue)

            # rewards by type
            r_global = jnp.where(is_delivery, 1.0, 0.0)
            add = jnp.zeros((N,), jnp.float32)
            if cfg.reward_type == RewardType.GLOBAL:
                add = add + r_global
            elif cfg.reward_type == RewardType.INDIVIDUAL:
                add = add.at[aidx].add(r_global)
            else:  # TWO_STAGE: +0.5 here, mark delivered
                add = add.at[aidx].add(jnp.where(is_delivery, 0.5, 0.0))
                has_delivered = has_delivered.at[aidx].set(
                    jnp.where(is_delivery, True, has_delivered[aidx])
                )
            rewards = rewards + add
            deliveries = deliveries.at[aidx].add(
                jnp.where(is_delivery, 1, 0).astype(jnp.int32)
            )
            delivered_any = delivered_any | is_delivery
            return (in_queue, rewards, has_delivered, delivered_any, deliveries, key, gs2, ga2), None

        carry0 = (in_queue, rewards, has_delivered, delivered_any, deliveries, key, gs2, ga2)
        for gi in range(self.n_goals):
            carry0, _ = deliver_at(carry0, (self.goal_x[gi], self.goal_y[gi]))
        in_queue, rewards, has_delivered, delivered_any, deliveries, key, _, _ = carry0

        # --- counters / termination ---
        inactive = jnp.where(delivered_any, 0, state.inactive_count + 1)
        step_count = state.step_count + 1
        done = step_count >= cfg.max_steps
        if cfg.max_inactivity_steps is not None:
            done = done | (inactive >= cfg.max_inactivity_steps)

        new_travel=state.distance_traveled + abs(new_x- old_x) + abs(new_y - old_y)

        state = state.replace(
            agent_has_delivered=has_delivered, in_queue=in_queue,
            step_count=step_count, inactive_count=inactive, distance_traveled=new_travel,key=key,
        )
        obs = self._obs(state)
        # ---- behavioral signals (commentary; no effect on dynamics) ----
        # forward-blocked = tried to step forward, lost movement contention, and
        # was not a voluntary carry-into-shelf cancel -> a collision/contention proxy.
        forward_blocked = (actions == Action.FORWARD) & (~moves) & (~_cancelled)
        noop = actions == Action.NOOP
        end_time=time.time()
        step_time=end_time-start_time

        info = {
            "delivered": delivered_any,         # scalar bool (kept for compatibility)
            "deliveries": deliveries,           # [N] int32, per-agent deliveries this step
            "forward_blocked": forward_blocked, # [N] bool
            "noop": noop,                       # [N] bool
            "pickup": pickup,                   # [N] bool
            "drop": drop,                       # [N] bool
            "distance_traveled":new_travel,     # [N] int32
            "step_time":step_time               # [N] float32
        }
        return state, obs, rewards, done, info
