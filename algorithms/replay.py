"""Record full episode trajectories ("replays") from a policy snapshot.

A replay is everything the dashboard needs to *re-draw* an episode after the
fact: per-step agent poses, shelf positions, the request queue, plus the
per-step action/reward/event streams. It is captured with a single-env jitted
rollout (one `lax.scan`, one device->host sync), so recording one snapshot
costs milliseconds and can run between training chunks without touching the
fused training scan.

Storage: one compressed ``.npz`` per snapshot under ``<run_dir>/replays/``,
with a ``meta_json`` string member carrying scalars + static env geometry, so
the dashboard side (arena/replay_data.py) can read it with numpy + stdlib only.

Trajectory arrays cover T steps; the *state* arrays have T+1 entries (index 0
is the post-reset state, index t+1 the state after step t), so a player can
draw the initial frame too.
"""

from __future__ import annotations

import functools
import json
import os

import jax
import jax.numpy as jnp
import numpy as np

from algorithms.networks import ScannedGRU

REPLAY_DIR = "replays"
REPLAY_PREFIX = "replay_upd"


def make_recorder(env, actor, cfg):
    """Build a jitted single-env episode recorder bound to (env, actor, cfg).

    Returns ``record(actor_params, key, greedy=False) -> (traj, scalars)``
    where ``traj`` is a dict of stacked device arrays (see module docstring)
    and ``scalars`` is ``(ep_return, total_deliveries)``. Compiles once per
    (greedy,) variant and is reused across calls — cheap enough to run every
    training chunk.
    """
    N = cfg.n_agents
    obs_dim = env.obs_dim
    H = cfg.hidden_dim
    T = cfg.time_limit
    zeros_1N = jnp.zeros((1, N))

    @functools.partial(jax.jit, static_argnums=(2,))
    def record(actor_params, key, greedy: bool = False):
        kreset, krun = jax.random.split(key)
        state0, obs0 = env.reset(kreset)
        h0 = ScannedGRU.initialize_carry(N, H)

        def step(carry, _):
            state, obs, h_actor, rng = carry
            rng, ksamp = jax.random.split(rng)
            obs_flat = obs.reshape(N, obs_dim)
            h_actor, dist = actor.apply(
                actor_params, h_actor, (obs_flat[None], zeros_1N))
            logits = dist.logits[0]  # [N, num_actions]
            if greedy:
                actions = jnp.argmax(logits, axis=-1)
            else:
                actions = jax.random.categorical(ksamp, logits)
            nstate, nobs, reward, done, info = env.step(state, actions)
            out = {
                "state": nstate,
                "actions": actions,
                "rewards": reward,
                "deliveries": info["deliveries"],
                "blocked": info["forward_blocked"],
                "noop": info["noop"],
                "pickup": info["pickup"],
                "drop": info["drop"],
            }
            return (nstate, nobs, h_actor, rng), out

        _, out = jax.lax.scan(step, (state0, obs0, h0, krun), None, length=T)

        # prepend the post-reset state so players can draw frame 0
        states_T1 = jax.tree_util.tree_map(
            lambda s0, sT: jnp.concatenate([s0[None], sT], axis=0),
            state0, out["state"])
        traj = {
            "agent_x": states_T1.agent_x,          # [T+1, N]
            "agent_y": states_T1.agent_y,
            "agent_dir": states_T1.agent_dir,
            "agent_carrying": states_T1.agent_carrying,
            "shelf_x": states_T1.shelf_x,          # [T+1, S]
            "shelf_y": states_T1.shelf_y,
            "in_queue": states_T1.in_queue,
            "actions": out["actions"],             # [T, N]
            "rewards": out["rewards"],             # [T, N]
            "deliveries": out["deliveries"],       # [T, N]
            "blocked": out["blocked"],
            "noop": out["noop"],
            "pickup": out["pickup"],
            "drop": out["drop"],
        }
        ep_return = out["rewards"].sum()
        total_deliveries = out["deliveries"].sum()
        return traj, (ep_return, total_deliveries)

    return record


def make_seac_recorder(env, actor, cfg):
    """SEAC variant of :func:`make_recorder`: per-agent independent actor
    params (leading agent axis), each agent acting with its own policy."""
    N = cfg.n_agents
    obs_dim = env.obs_dim
    H = cfg.hidden_dim
    T = cfg.time_limit
    resets_11 = jnp.zeros((1, 1))

    def act_i(p_i, h_i, o_i):                  # o_i [obs], h_i [1, H]
        h_i, dist = actor.apply(p_i, h_i, (o_i[None, None], resets_11))
        return h_i, dist.logits[0, 0]          # [A]

    @functools.partial(jax.jit, static_argnums=(2,))
    def record(actor_params, key, greedy: bool = False):
        kreset, krun = jax.random.split(key)
        state0, obs0 = env.reset(kreset)
        h0 = jnp.zeros((N, 1, H))              # per-agent hidden, single env

        def step(carry, _):
            state, obs, h_actor, rng = carry
            rng, ksamp = jax.random.split(rng)
            obs_N = obs.reshape(N, obs_dim)
            h_actor, logits = jax.vmap(act_i)(actor_params, h_actor, obs_N)
            if greedy:
                actions = jnp.argmax(logits, axis=-1)
            else:
                actions = jax.random.categorical(ksamp, logits)
            nstate, nobs, reward, done, info = env.step(state, actions)
            out = {
                "state": nstate,
                "actions": actions,
                "rewards": reward,
                "deliveries": info["deliveries"],
                "blocked": info["forward_blocked"],
                "noop": info["noop"],
                "pickup": info["pickup"],
                "drop": info["drop"],
            }
            return (nstate, nobs, h_actor, rng), out

        _, out = jax.lax.scan(step, (state0, obs0, h0, krun), None, length=T)
        states_T1 = jax.tree_util.tree_map(
            lambda s0, sT: jnp.concatenate([s0[None], sT], axis=0),
            state0, out["state"])
        traj = {
            "agent_x": states_T1.agent_x,
            "agent_y": states_T1.agent_y,
            "agent_dir": states_T1.agent_dir,
            "agent_carrying": states_T1.agent_carrying,
            "shelf_x": states_T1.shelf_x,
            "shelf_y": states_T1.shelf_y,
            "in_queue": states_T1.in_queue,
            "actions": out["actions"],
            "rewards": out["rewards"],
            "deliveries": out["deliveries"],
            "blocked": out["blocked"],
            "noop": out["noop"],
            "pickup": out["pickup"],
            "drop": out["drop"],
        }
        return traj, (out["rewards"].sum(), out["deliveries"].sum())

    return record


def save_replay(run_dir: str, traj, scalars, *, env, cfg, update: int,
                env_steps: int, seed: int, greedy: bool = False,
                algo: str | None = None) -> str:
    """Write one replay snapshot to ``<run_dir>/replays/replay_updNNNNNN.npz``.

    Compacts dtypes (positions fit int16, event streams int8) so a tiny-grid
    500-step episode lands well under 100 KB compressed.
    """
    ep_return, total_deliveries = (float(scalars[0]), int(scalars[1]))
    t = jax.tree_util.tree_map(np.asarray, traj)

    meta = {
        "update": int(update),
        "env_steps": int(env_steps),
        "seed": int(seed),
        "greedy": bool(greedy),
        "ep_return": round(ep_return, 4),
        "total_deliveries": total_deliveries,
        "H": int(env.H),
        "W": int(env.W),
        "goal_x": np.asarray(env.goal_x).tolist(),
        "goal_y": np.asarray(env.goal_y).tolist(),
        "n_agents": int(cfg.n_agents),
        "size": cfg.size,
        "time_limit": int(cfg.time_limit),
        "algo": algo or getattr(cfg, "algo", "?"),
    }

    out_dir = os.path.join(run_dir, REPLAY_DIR)
    os.makedirs(out_dir, exist_ok=True)
    tag = f"_s{seed}" + ("_g" if greedy else "")
    path = os.path.join(out_dir, f"{REPLAY_PREFIX}{update:06d}{tag}.npz")
    np.savez_compressed(
        path,
        agent_x=t["agent_x"].astype(np.int16),
        agent_y=t["agent_y"].astype(np.int16),
        agent_dir=t["agent_dir"].astype(np.int8),
        agent_carrying=t["agent_carrying"].astype(np.int16),
        shelf_x=t["shelf_x"].astype(np.int16),
        shelf_y=t["shelf_y"].astype(np.int16),
        in_queue=t["in_queue"].astype(bool),
        actions=t["actions"].astype(np.int8),
        rewards=t["rewards"].astype(np.float32),
        deliveries=t["deliveries"].astype(np.int8),
        blocked=t["blocked"].astype(np.int8),
        noop=t["noop"].astype(np.int8),
        pickup=t["pickup"].astype(np.int8),
        drop=t["drop"].astype(np.int8),
        meta_json=np.array(json.dumps(meta)),
    )
    return path
