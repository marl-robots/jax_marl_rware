"""Differential parity test: JAX Warehouse vs original semitable/rware.

Strategy (per the plan): we cannot match NumPy's PRNG stream, so instead we
*inject* the original env's post-reset state into a JAX EnvState, then drive both
with identical action sequences and assert exact equality of the deterministic
dynamics (positions, directions, carrying, per-agent rewards, done). The request
queue is replaced via RNG inside step(), so we re-sync our queue mask from the
original after every step to keep the two aligned.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

gym = pytest.importorskip("gymnasium")
pytest.importorskip("rware")

from jaxrware import Warehouse, make_config  # noqa: E402


def _orig_state_to_jax(env, wh):
    u = env.unwrapped
    N = u.n_agents
    agent_x = jnp.array([a.x for a in u.agents], jnp.int32)
    agent_y = jnp.array([a.y for a in u.agents], jnp.int32)
    agent_dir = jnp.array([a.dir.value for a in u.agents], jnp.int32)
    carrying = jnp.array(
        [a.carrying_shelf.id if a.carrying_shelf is not None else 0 for a in u.agents],
        jnp.int32,
    )
    shelf_x = jnp.array([s.x for s in u.shelfs], jnp.int32)
    shelf_y = jnp.array([s.y for s in u.shelfs], jnp.int32)
    in_queue = np.zeros(wh.n_shelves, bool)
    for s in u.request_queue:
        in_queue[s.id - 1] = True
    from jaxrware import EnvState

    return EnvState(
        agent_x=agent_x, agent_y=agent_y, agent_dir=agent_dir,
        agent_carrying=carrying,agent_distance_traveled=jnp.zeros((N,), jnp.int32), agent_has_delivered=jnp.zeros((N,), bool),
        shelf_x=shelf_x, shelf_y=shelf_y, in_queue=jnp.array(in_queue),
        step_count=jnp.array(0, jnp.int32), inactive_count=jnp.array(0, jnp.int32),
        key=jax.random.PRNGKey(0),
    )


def _sync_queue(env, state):
    u = env.unwrapped
    in_queue = np.zeros(state.in_queue.shape[0], bool)
    for s in u.request_queue:
        in_queue[s.id - 1] = True
    return state.replace(in_queue=jnp.array(in_queue))


@pytest.mark.parametrize("seed", [0, 1, 2, 7, 13])
def test_dynamics_parity(seed):
    cfg = make_config("tiny", 4, "normal")
    wh = Warehouse(cfg)
    env = gym.make("rware:rware-tiny-4ag-v2", max_steps=500)
    env.reset(seed=seed)

    state = _orig_state_to_jax(env, wh)
    rng = np.random.default_rng(seed)

    for t in range(300):
        actions = rng.integers(0, 5, size=cfg.n_agents)
        o_obs, o_rew, o_done, o_trunc, o_info = env.step(actions.tolist())
        state, j_obs, j_rew, j_done, j_info = wh.step(state, jnp.array(actions))

        u = env.unwrapped
        ox = np.array([a.x for a in u.agents])
        oy = np.array([a.y for a in u.agents])
        od = np.array([a.dir.value for a in u.agents])
        oc = np.array([a.carrying_shelf.id if a.carrying_shelf else 0 for a in u.agents])

        assert np.array_equal(np.array(state.agent_x), ox), f"x mismatch t={t}"
        assert np.array_equal(np.array(state.agent_y), oy), f"y mismatch t={t}"
        assert np.array_equal(np.array(state.agent_dir), od), f"dir mismatch t={t}"
        assert np.array_equal(np.array(state.agent_carrying), oc), f"carry mismatch t={t}"
        assert np.allclose(np.array(j_rew), np.array(o_rew)), f"reward mismatch t={t}"

        # shelf positions are deterministic (not RNG-driven) and must match exactly
        osx = np.zeros(wh.n_shelves, np.int32)
        osy = np.zeros(wh.n_shelves, np.int32)
        for sh in u.shelfs:
            osx[sh.id - 1] = sh.x
            osy[sh.id - 1] = sh.y
        assert np.array_equal(np.array(state.shelf_x), osx), f"shelf x mismatch t={t}"
        assert np.array_equal(np.array(state.shelf_y), osy), f"shelf y mismatch t={t}"

        state = _sync_queue(env, state)
        if o_done or o_trunc:
            break


@pytest.mark.parametrize("seed", [0, 1, 5, 42])
def test_obs_encoding_parity(seed):
    """At reset (identical state + queue) the 71-dim FLATTENED obs must match."""
    cfg = make_config("tiny", 4, "normal")
    wh = Warehouse(cfg)
    env = gym.make("rware:rware-tiny-4ag-v2", max_steps=500)
    o_obs, _ = env.reset(seed=seed)

    state = _orig_state_to_jax(env, wh)
    j_obs = np.array(wh._obs(state))
    o_obs = np.stack([np.asarray(o, np.float32) for o in o_obs])

    assert j_obs.shape == o_obs.shape, (j_obs.shape, o_obs.shape)
    assert np.allclose(j_obs, o_obs), np.argwhere(~np.isclose(j_obs, o_obs))
