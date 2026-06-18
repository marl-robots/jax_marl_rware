"""Train the value-based family on the JAX RWARE env.

  * iql       -- Layer 1 baseline (epsilon-greedy, double-Q TD(0), target net).
  * iql-emax  -- IDQN-EMAX (arXiv 2302.03439): K-ensemble, UCB exploration,
                 ensemble-mean targets, no target net. The algorithm that is
                 reported to actually SOLVE RWARE (beats IPPO/MAPPO).

Off-policy episodic loop mirroring epymarl: collect full episodes into a host
replay buffer, then sample minibatches of episodes and take gradient steps.

Examples (WSL, conda env jax_env_1):
    JAX_PLATFORMS=cpu python -m scripts.train_dqn --algo iql-emax --iters 600
    JAX_PLATFORMS=cpu python -m scripts.train_dqn --algo iql --smoke
"""

from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np

from algorithms.dqn_config import DQNConfig
from algorithms.dqn import (
    make_iql_trainer, make_emax_trainer, feed_batch, epsilon_at,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", default="iql", choices=["iql", "iql-emax"])
    ap.add_argument("--size", default="tiny")
    ap.add_argument("--n-agents", type=int, default=4)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--parallel-envs", type=int, default=8)
    ap.add_argument("--time-limit", type=int, default=None)
    ap.add_argument("--buffer-size", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--ensemble-size", type=int, default=None)
    ap.add_argument("--ucb-beta", type=float, default=None)
    ap.add_argument("--log-every", type=int, default=1)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    is_emax = args.algo == "iql-emax"
    overrides = dict(size=args.size, n_agents=args.n_agents, seed=args.seed,
                     parallel_envs=args.parallel_envs)
    if is_emax:
        # feedforward ensemble for now (recurrent rollout needs per-member hidden
        # carry -- a TODO); UCB replaces epsilon-greedy.
        overrides["use_rnn"] = False
    for name, attr in (("time_limit", "time_limit"), ("buffer_size", "buffer_size"),
                       ("batch_size", "batch_size"), ("ensemble_size", "ensemble_size"),
                       ("ucb_beta", "ucb_beta")):
        v = getattr(args, attr)
        if v is not None:
            overrides[name] = v
    if args.smoke:
        overrides.update(time_limit=64, parallel_envs=4, buffer_size=64,
                         batch_size=8, target_update_interval=20,
                         epsilon_anneal_time=4000)
    cfg = DQNConfig.from_algo(args.algo, **overrides)
    iters = 20 if args.smoke else args.iters

    tr = (make_emax_trainer if is_emax else make_iql_trainer)(cfg)
    N, E, T = tr["N"], tr["E"], tr["T"]
    update = jax.jit(tr["update"])

    key = jax.random.PRNGKey(cfg.seed)
    if is_emax:
        params, opt_state, rew_ms, key = tr["init_state"](key)
    else:
        params, target_params, opt_state, rew_ms, key = tr["init_state"](key)

    buf_obs = np.zeros((cfg.buffer_size, T + 1, N, tr["in_dim"]), np.float32)
    buf_act = np.zeros((cfg.buffer_size, T, N), np.int32)
    buf_rew = np.zeros((cfg.buffer_size, T), np.float32)
    write, filled = 0, 0
    rng = np.random.default_rng(cfg.seed)

    tag = f"K={cfg.ensemble_size} beta={cfg.ucb_beta} (UCB)" if is_emax \
        else f"eps-greedy target/{cfg.target_update_interval}ep"
    print(f"algo={cfg.algo}  env=rware-{cfg.size}-{cfg.n_agents}ag  "
          f"parallel_envs={E}  time_limit={T}  use_rnn={cfg.use_rnn}  "
          f"buffer={cfg.buffer_size}ep batch={cfg.batch_size}ep  {tag}  iters={iters}")

    t_env = episodes = last_target = 0
    t0 = time.perf_counter()
    for it in range(iters):
        eps = float(epsilon_at(cfg, t_env))
        if is_emax:
            obs_b, act_b, rew_b, metrics, key = tr["collect"](params, key, eps)
        else:
            obs_b, act_b, rew_b, metrics, key = tr["collect"](params, key, eps)
        obs_b = np.asarray(obs_b); act_b = np.asarray(act_b); rew_b = np.asarray(rew_b)

        for e in range(E):
            buf_obs[write] = obs_b[e]; buf_act[write] = act_b[e]; buf_rew[write] = rew_b[e]
            write = (write + 1) % cfg.buffer_size
            filled = min(filled + 1, cfg.buffer_size)
        t_env += E * T
        episodes += E

        loss = np.nan
        if filled >= cfg.batch_size:
            for _ in range(E):
                idx = rng.integers(0, filled, size=cfg.batch_size)
                batch = feed_batch(jnp.asarray(buf_obs[idx]), jnp.asarray(buf_act[idx]),
                                   jnp.asarray(buf_rew[idx]), T, N)
                if is_emax:
                    params, opt_state, rew_ms, diag = update(
                        params, opt_state, rew_ms, batch)
                else:
                    params, opt_state, rew_ms, diag = update(
                        params, target_params, opt_state, rew_ms, batch)
                loss = float(diag["loss"])
            if not is_emax and episodes - last_target >= cfg.target_update_interval:
                target_params = params
                last_target = episodes

        if it % args.log_every == 0 or it == iters - 1:
            print(f"  it {it:4d} | ep {episodes:5d} | eps {eps:4.2f} | "
                  f"return {float(metrics['ep_return']):7.3f} | "
                  f"deliv {float(metrics['deliveries']):6.2f} | loss {loss:9.4f}",
                  flush=True)

    dt = time.perf_counter() - t0
    print(f"\nran {iters} iters ({episodes} episodes, {t_env:,} env steps) in {dt:.1f}s")


if __name__ == "__main__":
    main()
