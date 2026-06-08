"""Replay seed 2 to the failing step and dump the resolver's view."""
import numpy as np
import jax, jax.numpy as jnp
import gymnasium as gym
import rware  # noqa

from jaxrware import Warehouse, make_config, EnvState, Action, Direction
from jaxrware.collision import resolve_moves, _req_location
from tests.test_parity_env import _orig_state_to_jax, _sync_queue

FAIL_T = 132
SEED = 2

cfg = make_config("tiny", 4, "normal")
wh = Warehouse(cfg)
env = gym.make("rware:rware-tiny-4ag-v2", max_steps=500)
env.reset(seed=SEED)
state = _orig_state_to_jax(env, wh)
rng = np.random.default_rng(SEED)

DIRNAME = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
ACTNAME = {0: "NOOP", 1: "FWD", 2: "LEFT", 3: "RIGHT", 4: "TOGGLE"}

for t in range(FAIL_T + 1):
    actions = rng.integers(0, 5, size=cfg.n_agents)
    if t == FAIL_T:
        print(f"\n=== STEP t={t}  seed={SEED} ===")
        print("actions:", [ACTNAME[a] for a in actions])
        u = env.unwrapped
        print("PRE-STEP (from original, injected into jax state):")
        for i, a in enumerate(u.agents):
            cs = a.carrying_shelf.id if a.carrying_shelf else 0
            print(f"  agent{i}: x={a.x} y={a.y} dir={DIRNAME[a.dir.value]} carry={cs}")
        ax, ay, ad = state.agent_x, state.agent_y, state.agent_dir
        ac = state.agent_carrying
        ga, gs = wh._grids(state)
        is_fwd = (jnp.array(actions) == Action.FORWARD)
        tx, ty = _req_location(cfg, ax, ay, ad, is_fwd)
        print("targets (tx,ty):", list(zip(np.array(tx), np.array(ty))))
        moves, mtx, mty, canc = resolve_moves(
            cfg, ax, ay, ad, jnp.array(actions), ac, ga, gs)
        print("moves:", np.array(moves), "cancelled:", np.array(canc))
        print("grid_agents nonzero:", np.argwhere(np.array(ga) > 0))
        print("grid_shelfs at agent cells:",
              [int(gs[int(ay[i]), int(ax[i])]) for i in range(cfg.n_agents)])

    o_obs, o_rew, o_done, o_trunc, o_info = env.step(actions.tolist())
    state, *_ = wh.step(state, jnp.array(actions))
    if t == FAIL_T:
        u = env.unwrapped
        print("POST-STEP original:")
        for i, a in enumerate(u.agents):
            print(f"  agent{i}: x={a.x} y={a.y} dir={DIRNAME[a.dir.value]}")
        print("POST-STEP jax:")
        print("  x:", np.array(state.agent_x), "y:", np.array(state.agent_y),
              "dir:", np.array(state.agent_dir))
    state = _sync_queue(env, state)
