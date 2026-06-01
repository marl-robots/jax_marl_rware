"""Train MAPPO on the JAX RWARE env.

Defaults to a short smoke run; pass --total-steps 20000000 for the full proven
run (rware-tiny-4ag, parallel_envs=10, time_limit=500, seed=2).

Examples (WSL, conda env jax_env_1):
    python -m scripts.train_mappo --updates 20          # quick smoke
    python -m scripts.train_mappo --total-steps 20000000
"""

from __future__ import annotations

import argparse
import time

import jax
import numpy as np

from algorithms.config import MAPPOConfig
from algorithms.mappo import make_train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="tiny")
    ap.add_argument("--n-agents", type=int, default=4)
    ap.add_argument("--difficulty", default="normal")
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--total-steps", type=int, default=None,
                    help="overrides config total_steps; ignored if --updates given")
    ap.add_argument("--updates", type=int, default=None,
                    help="number of MAPPO updates to run (smoke runs)")
    ap.add_argument("--parallel-envs", type=int, default=None,
                    help="override parallel_envs (e.g. 128 for fast wall-clock)")
    ap.add_argument("--entropy-coef", type=float, default=None,
                    help="override entropy_coef (raise for large-batch exploration)")
    ap.add_argument("--lr", type=float, default=None,
                    help="override Adam learning rate (raise for large batch)")
    ap.add_argument("--num-epochs", type=int, default=None,
                    help="override PPO epochs per update (more grad steps/update)")
    args = ap.parse_args()

    overrides = {}
    if args.parallel_envs is not None:
        overrides["parallel_envs"] = args.parallel_envs
    if args.entropy_coef is not None:
        overrides["entropy_coef"] = args.entropy_coef
    if args.lr is not None:
        overrides["lr"] = args.lr
    if args.num_epochs is not None:
        overrides["num_epochs"] = args.num_epochs

    cfg = MAPPOConfig(
        size=args.size, n_agents=args.n_agents, difficulty=args.difficulty,
        seed=args.seed,
        total_steps=args.total_steps if args.total_steps is not None
        else MAPPOConfig.total_steps,
        **overrides,
    )
    n_updates = args.updates if args.updates is not None else cfg.num_updates

    print(f"env=rware-{cfg.size}-{cfg.n_agents}ag  parallel_envs={cfg.parallel_envs} "
          f"time_limit={cfg.time_limit}  updates={n_updates}  "
          f"(steps/update={cfg.batch_steps})  seed={cfg.seed}")

    train, _env = make_train(cfg, num_updates=n_updates, live_log=True)
    train = jax.jit(train)
    print("live per-update log (return / entropy / loss) follows; "
          "watch entropy — a fast drop toward 0 means exploration collapse:\n")

    t0 = time.perf_counter()
    out = jax.block_until_ready(train(jax.random.PRNGKey(cfg.seed)))
    dt = time.perf_counter() - t0

    m = out["metrics"]
    rets = np.array(m["episode_return"])
    total_env_steps = n_updates * cfg.batch_steps
    print(f"\ncompiled+ran {n_updates} updates ({total_env_steps:,} env steps) "
          f"in {dt:.1f}s  ({total_env_steps / dt:,.0f} steps/s)")
    print("episode_return: first={:.3f}  last={:.3f}  max={:.3f}".format(
        float(rets[0]), float(rets[-1]), float(rets.max())))

    # coarse curve
    k = max(1, len(rets) // 10)
    idxs = list(range(0, len(rets), k))
    print("\nupdate :  return   loss     value_loss  entropy")
    for i in idxs:
        print(f"{i:6d} : {float(rets[i]):7.3f}  {float(m['loss'][i]):7.3f}  "
              f"{float(m['value_loss'][i]):9.3f}  {float(m['entropy'][i]):7.3f}")


if __name__ == "__main__":
    main()
