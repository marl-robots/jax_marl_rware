"""Train SEAC (Shared Experience Actor-Critic) on the JAX RWARE env.

Per-agent independent actor-critics with the canonical shared-experience term
(importance-weighted cross-agent training). Same chunked checkpoint/resume loop
as train_mappo. Defaults to a short smoke; pass --total-steps for a real run.

Examples (WSL, conda env jax_env_1):
    python -m scripts.train_seac --updates 20                 # smoke
    python -m scripts.train_seac --total-steps 20000000       # full run
    python -m scripts.train_seac --no-rnn --total-steps 5000000
"""

from __future__ import annotations

import argparse
import os
import time

import jax
import numpy as np

from algorithms.checkpoint import CheckpointManager, has_checkpoint
from algorithms.config import MAPPOConfig
from algorithms.seac import make_resumable_train
from algorithms.metrics import CSVLogger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="tiny")
    ap.add_argument("--n-agents", type=int, default=4)
    ap.add_argument("--difficulty", default="normal")
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--total-steps", type=int, default=None)
    ap.add_argument("--updates", type=int, default=None,
                    help="number of SEAC updates to run (smoke runs)")
    ap.add_argument("--seac-coef", type=float, default=None,
                    help="shared-experience weight lambda (default 1.0; 0 = IA2C)")
    ap.add_argument("--is-clip", type=float, default=None,
                    help="clip SEAC importance weight to <= this (0 = no clip = canonical)")
    ap.add_argument("--entropy-coef", type=float, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--no-rnn", action="store_true",
                    help="feedforward network instead of GRU (faster)")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--checkpoint-every", type=int, default=100)
    ap.add_argument("--max-to-keep", type=int, default=5)
    ap.add_argument("--ema-decay", type=float, default=0.99)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--resume-from", default="latest")
    ap.add_argument("--no-live-log", action="store_true")
    args = ap.parse_args()

    overrides = {}
    if args.seac_coef is not None:
        overrides["seac_coef"] = args.seac_coef
    if args.is_clip is not None:
        overrides["is_clip"] = args.is_clip
    if args.entropy_coef is not None:
        overrides["entropy_coef"] = args.entropy_coef
    if args.lr is not None:
        overrides["lr"] = args.lr
    if args.no_rnn:
        overrides["use_rnn"] = False

    cfg = MAPPOConfig(
        size=args.size, n_agents=args.n_agents, difficulty=args.difficulty,
        seed=args.seed,
        total_steps=args.total_steps if args.total_steps is not None
        else MAPPOConfig.total_steps,
        **overrides,
    )
    n_updates = args.updates if args.updates is not None else cfg.num_updates

    net_tag = "" if cfg.use_rnn else "_fc"
    run_dir = args.run_dir or os.path.join(
        "runs", f"seac{net_tag}_{cfg.size}-{cfg.n_agents}ag_seed{cfg.seed}")
    csv_path = os.path.join(run_dir, "results.csv")
    batch_steps = cfg.batch_steps
    chunk = max(1, args.checkpoint_every)

    print(f"algo=seac (per-agent A2C, seac_coef={cfg.seac_coef}, "
          f"is_clip={cfg.is_clip}, use_rnn={cfg.use_rnn})")
    print(f"env=rware-{cfg.size}-{cfg.n_agents}ag  parallel_envs={cfg.parallel_envs} "
          f"time_limit={cfg.time_limit}  updates={n_updates}  "
          f"(steps/update={batch_steps})  seed={cfg.seed}")
    print(f"run-dir={run_dir}  chunk={chunk} updates/save")

    mgr = CheckpointManager(run_dir, max_to_keep=args.max_to_keep)
    mgr.save_config(cfg)
    trainer = make_resumable_train(cfg, live_log=not args.no_live_log)

    key = jax.random.PRNGKey(cfg.seed)
    carry = trainer["init_carry"](key)
    resume = args.resume and has_checkpoint(run_dir)
    start = 0
    if resume:
        sel = args.resume_from.lower()
        step = (mgr.latest_step() if sel == "latest"
                else mgr.best_step() if sel == "best" else int(args.resume_from))
        carry = mgr.restore(step, carry)
        start = int(step)
        print(f"resumed from checkpoint step {start} ({sel})")

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
            done_count = upd + i + 1
            ret_i = float(m["episode_return"][i])
            ema = ret_i if ema is None else (
                args.ema_decay * ema + (1.0 - args.ema_decay) * ret_i)
            logger.log({
                "environment_steps": done_count * batch_steps,
                "updates": done_count,
                "mean_episode_returns": ret_i,
                "entropy": float(m["entropy"][i]),
                "loss": float(m["loss"][i]),
                "actor_loss": float(m["actor_loss"][i]),
                "value_loss": float(m["value_loss"][i]),
                "reward_std_mean": float(m["reward_std_mean"][i]),
                "deliveries": float(m["deliveries"][i]),
                "block_rate": float(m["block_rate"][i]),
                "idle_rate": float(m["idle_rate"][i]),
            })
        upd += k
        mgr.save(upd, carry, smoothed_return=ema)

    mgr.wait()
    logger.close()
    dt = time.perf_counter() - t0
    ran = n_updates - start
    total_env_steps = ran * batch_steps
    print(f"\nran {ran} updates ({total_env_steps:,} env steps) "
          f"in {dt:.1f}s  ({total_env_steps / dt:,.0f} steps/s)")
    print(f"checkpoints -> {os.path.join(run_dir, 'checkpoints')}  "
          f"(latest={mgr.latest_step()}, best={mgr.best_step()})   metrics -> {csv_path}")


if __name__ == "__main__":
    main()
