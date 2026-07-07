"""Minimal reproductions of Jumanji RobotWarehouse divergences from RWARE.

Two parts, run in different conda envs (they have conflicting deps):

  # part 1: Jumanji behavior (env with jumanji installed, e.g. jumanji_135_01)
  python scripts/repro_jumanji_divergence.py --part jumanji

  # part 2: original rware behavior on the SAME scenarios (env jax_env_1)
  python scripts/repro_jumanji_divergence.py --part rware

Each part appends its findings to docs/data/divergence_repro.json. The
scenarios:

  A. CONVOY: two agents in single file on the goal-row highway, both facing
     RIGHT, both step FORWARD — a perfectly legal, conflict-free joint move
     (the original moves both agents).
  B. CONVOY-REVERSED: identical, but the agent *ids* of leader/follower are
     swapped. Dynamics should obviously not depend on id order.
  C. CONTENTION: two agents facing each other across a gap, both step FORWARD
     onto the same empty cell — the original resolves it (neither moves);
     no episode end.
  D. RANDOM-POLICY HORIZON: how long does an episode actually last under a
     uniform-random policy? (Original rware: always time_limit. Jumanji:
     terminates on its 'collision' predicate.) This measures the *effective
     exploration horizon* an algorithm gets early in training.

Geometry is the canonical tiny-4ag (Mava's scenario yaml): column_height=8,
shelf_rows=1, shelf_columns=3, sensor_range=1, request_queue_size=4 — an
11x10 grid whose bottom row (row 10) is goal/highway corridor.
"""

from __future__ import annotations

import argparse
import json
import os

OUT_PATH = os.path.join("docs", "data", "divergence_repro.json")
N_EPISODES = 512
TIME_LIMIT = 500


def _save(part: str, data: dict) -> None:
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    blob = {}
    if os.path.isfile(OUT_PATH):
        with open(OUT_PATH) as f:
            blob = json.load(f)
    blob[part] = data
    with open(OUT_PATH, "w") as f:
        json.dump(blob, f, indent=1)
    print(f"\nwrote results -> {OUT_PATH} [{part}]")


# ===========================================================================
# part 1: jumanji
# ===========================================================================
def run_jumanji() -> None:
    import jax
    import jax.numpy as jnp
    from jumanji.environments.routing.robot_warehouse import RobotWarehouse
    from jumanji.environments.routing.robot_warehouse.generator import RandomGenerator
    from jumanji.environments.routing.robot_warehouse.types import Agent, Position
    from jumanji.environments.routing.robot_warehouse.utils import compute_action_mask

    import jumanji
    print(f"jumanji {jumanji.__version__}  backend={jax.default_backend()}")

    def make_env(num_agents: int) -> RobotWarehouse:
        return RobotWarehouse(
            generator=RandomGenerator(
                column_height=8, shelf_rows=1, shelf_columns=3,
                num_agents=num_agents, sensor_range=1, request_queue_size=4),
            time_limit=TIME_LIMIT)

    env2 = make_env(2)
    H_, W_ = env2.grid_size
    goal_row = H_ - 1  # bottom corridor (goals live here): all highway

    def place(state, xs, ys, dirs):
        """Surgically place the two agents (positions/directions) in a state."""
        grid = state.grid.at[0].set(0)  # clear agents channel
        for i, (x, y) in enumerate(zip(xs, ys)):
            grid = grid.at[0, x, y].set(i + 1)
        agents = Agent(
            position=Position(x=jnp.array(xs, jnp.int32),
                              y=jnp.array(ys, jnp.int32)),
            direction=jnp.array(dirs, jnp.int32),
            is_carrying=jnp.zeros(len(xs), jnp.int32))
        return state.replace(
            grid=grid, agents=agents,
            action_mask=compute_action_mask(grid, agents))

    step = jax.jit(env2.step)
    state0, _ = env2.reset(jax.random.PRNGKey(0))
    FORWARD = jnp.array([1, 1], jnp.int32)
    results: dict = {}

    # --- A. convoy: agent 0 behind agent 1, both move right ---
    s = place(state0, xs=[goal_row, goal_row], ys=[0, 1], dirs=[1, 1])
    _, ts = step(s, FORWARD)
    results["convoy_follower_id0_terminates"] = bool(ts.last())

    # --- B. same convoy, ids swapped (agent 1 behind agent 0) ---
    s = place(state0, xs=[goal_row, goal_row], ys=[1, 0], dirs=[1, 1])
    _, ts = step(s, FORWARD)
    results["convoy_follower_id1_terminates"] = bool(ts.last())

    # --- C. contention: facing each other across one empty cell ---
    s = place(state0, xs=[goal_row, goal_row], ys=[0, 2], dirs=[1, 3])
    ns, ts = step(s, FORWARD)
    results["contention_terminates"] = bool(ts.last())
    results["contention_same_cell"] = bool(
        (ns.agents.position.x[0] == ns.agents.position.x[1])
        & (ns.agents.position.y[0] == ns.agents.position.y[1]))

    print("A. legal convoy (follower has LOWER id): terminates =",
          results["convoy_follower_id0_terminates"])
    print("B. legal convoy (follower has HIGHER id): terminates =",
          results["convoy_follower_id1_terminates"])
    print("C. head-on contention for one cell:       terminates =",
          results["contention_terminates"],
          " both agents on same cell =", results["contention_same_cell"])

    # --- D. random-policy horizon, canonical tiny-4ag ---
    env4 = make_env(4)
    reset4 = jax.jit(jax.vmap(env4.reset))
    step4 = jax.jit(jax.vmap(env4.step))

    keys = jax.random.split(jax.random.PRNGKey(42), N_EPISODES)
    states, _ = reset4(keys)

    def body(carry, t):
        states, ended_at, done, key = carry
        key, k = jax.random.split(key)
        actions = jax.random.randint(k, (N_EPISODES, 4), 0, 5)
        nstates, ts = step4(states, actions)
        now_done = ts.last()
        ended_at = jnp.where(done | ~now_done, ended_at, t + 1)
        done = done | now_done
        return (nstates, ended_at, done, key), None

    init = (states, jnp.full((N_EPISODES,), TIME_LIMIT, jnp.int32),
            jnp.zeros((N_EPISODES,), bool), jax.random.PRNGKey(7))
    (_, ended_at, done, _), _ = jax.lax.scan(
        body, init, jnp.arange(TIME_LIMIT))
    ended_at = jax.device_get(ended_at)

    import numpy as np
    results["random_policy"] = {
        "episodes": int(N_EPISODES),
        "time_limit": TIME_LIMIT,
        "terminated_early_frac": float((ended_at < TIME_LIMIT).mean()),
        "median_length": float(np.median(ended_at)),
        "mean_length": float(ended_at.mean()),
        "p90_length": float(np.percentile(ended_at, 90)),
    }
    r = results["random_policy"]
    print(f"D. random policy on tiny-4ag over {N_EPISODES} episodes: "
          f"{r['terminated_early_frac'] * 100:.1f}% end early, "
          f"median length {r['median_length']:.0f} / {TIME_LIMIT}, "
          f"mean {r['mean_length']:.1f}, p90 {r['p90_length']:.0f}")
    _save("jumanji", results)


# ===========================================================================
# part 2: original rware
# ===========================================================================
def run_rware() -> None:
    import gymnasium as gym
    import numpy as np
    import rware  # noqa: F401

    results: dict = {}

    def fresh(n_agents: int):
        env = gym.make(f"rware:rware-tiny-{n_agents}ag-v2", max_steps=TIME_LIMIT)
        env.reset(seed=0)
        return env

    def place(env, xs, ys, dirs):
        from rware.warehouse import Direction
        u = env.unwrapped
        dmap = {0: Direction.UP, 1: Direction.RIGHT, 2: Direction.DOWN,
                3: Direction.LEFT}
        # rware coordinates: agent.x = column, agent.y = row (note: transposed
        # vs jumanji, where x is the row). Inputs here use (row, col) like the
        # jumanji part, so swap.
        for a, row, col, d in zip(u.agents, xs, ys, dirs):
            a.x, a.y, a.dir = col, row, dmap[d]
        u._recalc_grid()

    goal_row = 10  # tiny grid is 11 rows; bottom row is the goal corridor
    FORWARD = [1, 1]

    # --- A/B. convoy (both id orders) ---
    for tag, ys in (("convoy_follower_id0", [0, 1]),
                    ("convoy_follower_id1", [1, 0])):
        env = fresh(2)
        place(env, [goal_row, goal_row], ys, [1, 1])
        before = [(a.y, a.x) for a in env.unwrapped.agents]
        _, _, done, trunc, _ = env.step(FORWARD)
        after = [(a.y, a.x) for a in env.unwrapped.agents]
        results[f"{tag}_terminates"] = bool(done or trunc)
        results[f"{tag}_both_moved"] = bool(
            all(b != a for b, a in zip(before, after)))

    # --- C. contention ---
    env = fresh(2)
    place(env, [goal_row, goal_row], [0, 2], [1, 3])
    before = [(a.y, a.x) for a in env.unwrapped.agents]
    _, _, done, trunc, _ = env.step(FORWARD)
    after = [(a.y, a.x) for a in env.unwrapped.agents]
    results["contention_terminates"] = bool(done or trunc)
    results["contention_nobody_moved"] = bool(before == after)

    print("A. legal convoy (follower id 0): terminates =",
          results["convoy_follower_id0_terminates"],
          " both moved =", results["convoy_follower_id0_both_moved"])
    print("B. legal convoy (follower id 1): terminates =",
          results["convoy_follower_id1_terminates"],
          " both moved =", results["convoy_follower_id1_both_moved"])
    print("C. head-on contention:           terminates =",
          results["contention_terminates"],
          " nobody moved =", results["contention_nobody_moved"])

    # --- D. random-policy horizon (rware never ends early; verify) ---
    eps = 50  # the gym env is slow; 50 episodes suffice to verify 'never'
    lengths = []
    env = fresh(4)
    rng = np.random.default_rng(7)
    for ep in range(eps):
        env.reset(seed=ep)
        for t in range(TIME_LIMIT):
            _, _, done, trunc, _ = env.step(
                rng.integers(0, 5, size=4).tolist())
            if done or trunc:
                break
        lengths.append(t + 1)
    lengths = np.array(lengths)
    results["random_policy"] = {
        "episodes": eps,
        "time_limit": TIME_LIMIT,
        "terminated_early_frac": float((lengths < TIME_LIMIT).mean()),
        "median_length": float(np.median(lengths)),
    }
    r = results["random_policy"]
    print(f"D. random policy on tiny-4ag over {eps} episodes: "
          f"{r['terminated_early_frac'] * 100:.1f}% end early, "
          f"median length {r['median_length']:.0f} / {TIME_LIMIT}")
    _save("rware", results)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["jumanji", "rware"], required=True)
    args = ap.parse_args()
    (run_jumanji if args.part == "jumanji" else run_rware)()
