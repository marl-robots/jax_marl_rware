"""Train potential-MAPPO (MAPPO + self-supervised Phi shaping) on JAX RWARE.

Same chunked/orbax/CSV machinery as scripts/train_mappo.py, but the trainer adds
the Phi potential network (algorithms/mappo_potential.py). Defaults = MAPPOConfig
defaults, which is what the baseline run (runs/mappo_tiny-4ag_seed2) actually
used (parallel_envs=10, num_epochs=4, entropy_coef=1e-3, lr=3e-4) so the ONLY
difference from the baseline is the Phi shaping -- and it writes to a NEW run-dir
so the results.csv compares directly against runs/mappo_tiny-4ag_seed2.

  python -m scripts.train_mappo_potential --updates 20            # smoke
  python -m scripts.train_mappo_potential --total-steps 30000000  # comparison run
"""

from __future__ import annotations

import argparse
import os
import time

import jax
import numpy as np

from algorithms.checkpoint import CheckpointManager, has_checkpoint
from algorithms.config import MAPPOConfig
from algorithms.mappo_potential import make_resumable_train_potential
from algorithms.metrics import CSVLogger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="tiny")
    ap.add_argument("--n-agents", type=int, default=4)
    ap.add_argument("--difficulty", default="normal")
    ap.add_argument("--seed", type=int, default=2)
    # MAPPO knobs -- defaults = MAPPOConfig defaults, which is what the baseline
    # run (runs/mappo_tiny-4ag_seed2) actually used: parallel_envs=10,
    # num_epochs=4, entropy_coef=1e-3, lr=3e-4. (Its config.json is stale.)
    ap.add_argument("--parallel-envs", type=int, default=10)
    ap.add_argument("--num-epochs", type=int, default=4)
    ap.add_argument("--entropy-coef", type=float, default=1e-3)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--total-steps", type=int, default=None)
    ap.add_argument("--updates", type=int, default=None)
    # Phi shaping knobs
    ap.add_argument("--phi-beta", type=float, default=1.0,
                    help="shaping weight; 0 == plain MAPPO")
    ap.add_argument("--phi-epochs", type=int, default=4,
                    help="Phi gradient steps per update")
    ap.add_argument("--phi-lr", type=float, default=3e-4)
    ap.add_argument("--phi-hidden", type=int, default=64)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--checkpoint-every", type=int, default=50)
    ap.add_argument("--max-to-keep", type=int, default=5)
    ap.add_argument("--ema-decay", type=float, default=0.99)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no-live-log", action="store_true")
    args = ap.parse_args()

    cfg = MAPPOConfig.from_algo(
        "mappo", size=args.size, n_agents=args.n_agents,
        difficulty=args.difficulty, seed=args.seed,
        parallel_envs=args.parallel_envs, num_epochs=args.num_epochs,
        entropy_coef=args.entropy_coef, lr=args.lr,
    )
    if args.total_steps is not None:
        n_updates = args.total_steps // (cfg.time_limit * cfg.parallel_envs)
    elif args.updates is not None:
        n_updates = args.updates
    else:
        n_updates = cfg.num_updates

    run_dir = args.run_dir or os.path.join(
        "runs", f"mappo_pot_{cfg.size}-{cfg.n_agents}ag_seed{cfg.seed}")
    csv_path = os.path.join(run_dir, "results.csv")
    batch_steps = cfg.batch_steps
    chunk = max(1, args.checkpoint_every)

    print(f"potential-MAPPO  env=rware-{cfg.size}-{cfg.n_agents}ag  "
          f"parallel_envs={cfg.parallel_envs}  num_epochs={cfg.num_epochs}  "
          f"entropy={cfg.entropy_coef}  seed={cfg.seed}")
    print(f"phi: beta={args.phi_beta} epochs={args.phi_epochs} lr={args.phi_lr} "
          f"hidden={args.phi_hidden}")
    print(f"updates={n_updates} (steps/update={batch_steps})  run-dir={run_dir}  "
          f"chunk={chunk}")
    print(f"backend={jax.default_backend()}")

    mgr = CheckpointManager(run_dir, max_to_keep=args.max_to_keep)
    mgr.save_config(cfg)
    trainer = make_resumable_train_potential(
        cfg, phi_beta=args.phi_beta, phi_epochs=args.phi_epochs,
        phi_lr=args.phi_lr, phi_hidden=args.phi_hidden,
        live_log=not args.no_live_log)

    key = jax.random.PRNGKey(cfg.seed)
    carry = trainer["init_carry"](key)
    start = 0
    resume = args.resume and has_checkpoint(run_dir)
    if resume:
        step = mgr.latest_step()
        carry = mgr.restore(step, carry)
        start = int(step)
        print(f"resumed from update {start}")

    logger = CSVLogger(csv_path, resume=resume)
    if start >= n_updates:
        print(f"nothing to do: start={start} >= n_updates={n_updates}")
        logger.close(); mgr.wait(); return

    ema = None
    t0 = time.perf_counter()
    upd = start
    while upd < n_updates:
        k = min(chunk, n_updates - upd)
        carry, metrics = trainer["train_from"](carry, upd, k)
        carry = jax.block_until_ready(carry)
        m = {key_: np.asarray(val) for key_, val in metrics.items()}
        for i in range(k):
            done = upd + i + 1
            ret_i = float(m["episode_return"][i])
            ema = ret_i if ema is None else (
                args.ema_decay * ema + (1.0 - args.ema_decay) * ret_i)
            logger.log({
                "environment_steps": done * batch_steps,
                "updates": done,
                "mean_episode_returns": ret_i,
                "entropy": float(m["entropy"][i]),
                "loss": float(m["loss"][i]),
                "actor_loss": float(m["actor_loss"][i]),
                "value_loss": float(m["value_loss"][i]),
                "reward_std_mean": float(m["reward_std_mean"][i]),
                "deliveries": float(m["deliveries"][i]),
                "block_rate": float(m["block_rate"][i]),
                "idle_rate": float(m["idle_rate"][i]),
                "pickup_rate": float(m["pickup_rate"][i]),
                "deliveries_early": float(m["deliveries_early"][i]),
                "deliveries_mid": float(m["deliveries_mid"][i]),
                "deliveries_late": float(m["deliveries_late"][i]),
                "block_early": float(m["block_early"][i]),
                "block_mid": float(m["block_mid"][i]),
                "block_late": float(m["block_late"][i]),
                # potential-shaping diagnostics
                "phi_loss": float(m["phi_loss"][i]),
                "phi_target_mean": float(m["phi_target_mean"][i]),
                "mean_abs_bonus": float(m["mean_abs_bonus"][i]),
            })
        upd += k
        mgr.save(upd, carry, smoothed_return=ema)
        deliv = float(m["deliveries"][-1])
        print(f"  upd {upd:5d}/{n_updates} | deliv {deliv:6.2f} | "
              f"return {ret_i:8.3f} | phi_loss {float(m['phi_loss'][-1]):.4f} | "
              f"|bonus| {float(m['mean_abs_bonus'][-1]):.4f} | "
              f"{(upd - start) * batch_steps / max(time.perf_counter() - t0, 1e-9):,.0f} steps/s",
              flush=True)

    mgr.wait(); logger.close()
    dt = time.perf_counter() - t0
    total = (n_updates - start) * batch_steps
    print(f"\nran {n_updates - start} updates ({total:,} env steps) in {dt:.1f}s "
          f"({total / dt:,.0f} steps/s)  best={mgr.best_step()}  -> {run_dir}")


if __name__ == "__main__":
    main()
