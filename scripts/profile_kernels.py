"""Count GPU kernels per operation by inspecting compiled (optimized) HLO.

Each `fusion` / `custom-call` in the optimized module is ~one GPU kernel
launch. The env step runs once per rollout timestep (T per update), so a high
kernel count there means many tiny launches per step -> the launch-bound stall.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr

from jaxrware import Warehouse, make_config
from algorithms.config import MAPPOConfig
from algorithms.mappo import make_train
from algorithms.networks import ActorRNN, ScannedGRU


def count(text):
    return {
        "fusion": text.count("fusion("),
        "custom_call": text.count("custom-call("),
        "while": text.count("while("),
        "reduce": text.count(" reduce("),
        "sort": text.count("sort("),
        "scatter": text.count("scatter("),
        "gather": text.count("gather("),
        "hlo_lines": text.count("\n"),
    }


def main():
    cfg = MAPPOConfig(size="tiny", n_agents=4, difficulty="normal", seed=2,
                      parallel_envs=10)
    env = Warehouse(make_config(cfg.size, cfg.n_agents, cfg.difficulty))
    E, N = cfg.parallel_envs, cfg.n_agents

    keys = jr.split(jr.PRNGKey(0), E)
    states, obs = jax.vmap(env.reset)(keys)
    actions = jnp.zeros((E, N), dtype=jnp.int32)

    step = jax.jit(jax.vmap(env.step))
    c = step.lower(states, actions).compile()
    print("ENV STEP (vmap over", E, "envs), per call =", count(c.as_text()))

    # one actor forward step (rollout style: 1 timestep)
    actor = ActorRNN(env.num_actions, cfg.hidden_dim, cfg.orthogonal_gain)
    h = ScannedGRU.initialize_carry(E * N, cfg.hidden_dim)
    obs1 = jnp.zeros((1, E * N, env.obs_dim))
    d1 = jnp.zeros((1, E * N))
    p = actor.init(jr.PRNGKey(1), h, (obs1, d1))
    af = jax.jit(lambda p, h, x: actor.apply(p, h, x))
    ca = af.lower(p, h, (obs1, d1)).compile()
    print("ACTOR 1-step forward, per call =", count(ca.as_text()))

    # full training graph (2 updates) -- distinct kernels in the whole program
    train, _ = make_train(cfg, num_updates=2)
    ct = jax.jit(train).lower(jr.PRNGKey(0)).compile()
    print("FULL TRAIN (2 updates), distinct =", count(ct.as_text()))


if __name__ == "__main__":
    main()
