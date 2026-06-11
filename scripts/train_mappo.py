"""Train MAPPO on the JAX RWARE env, with orbax checkpointing + resume.

Training runs as fully-fused `lax.scan` chunks (full speed within a chunk). A
host loop drives the chunks and, between them (main thread), writes per-update
CSV metrics and an orbax checkpoint with keep-best (by EMA-smoothed return, so a
late collapse never loses the peak) + keep-last-N. Kill any time and rerun with
--resume to continue from the latest checkpoint (losing at most one chunk).

Defaults to a short smoke run; pass --total-steps 20000000 for the full proven
run (rware-tiny-4ag, parallel_envs=10, time_limit=500, seed=2).

Examples (WSL, conda env jax_env_1):
    python -m scripts.train_mappo --updates 20             # quick smoke
    python -m scripts.train_mappo --total-steps 20000000   # full proven run
    python -m scripts.train_mappo --total-steps 20000000 --resume   # continue
"""

from __future__ import annotations

import argparse
from datetime import datetime
import os
import time

import jax
import numpy as np

from algorithms.checkpoint import CheckpointManager, has_checkpoint
from algorithms.commentary import Narrator
from algorithms.config import MAPPOConfig
from algorithms.mappo import make_resumable_train
from algorithms.metrics import CSVLogger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", default="mappo",
                    choices=["ia2c", "ippo", "maa2c", "mappo"],
                    help="AC-family algorithm: {ind,cent} critic x {a2c,ppo} update")
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
    ap.add_argument("--no-rnn", action="store_true",
                    help="feedforward network instead of the GRU (faster, no BPTT; "
                         "reactive policy). Tags the run-dir with _fc.")
    ap.add_argument("--run-dir", default=None,
                    help="dir for checkpoints/ + results.csv "
                         "(default runs/<env>_seed<seed>)")
    ap.add_argument("--checkpoint-every", type=int, default=100,
                    help="updates per chunk = save cadence (pick a divisor of the "
                         "run to compile exactly once)")
    ap.add_argument("--max-to-keep", type=int, default=5,
                    help="orbax keep-last-N (best is always retained too)")
    ap.add_argument("--ema-decay", type=float, default=0.99,
                    help="EMA decay for the smoothed return used by keep-best")
    ap.add_argument("--resume", action="store_true",
                    help="resume from a checkpoint in --run-dir if present")
    ap.add_argument("--resume-from", default="latest",
                    help="which checkpoint to resume: 'latest', 'best', or a step "
                         "number (requires --resume)")
    ap.add_argument("--no-live-log", action="store_true",
                    help="disable the per-update stdout live log")
    ap.add_argument("--no-commentary", action="store_true",
                    help="disable the per-chunk behavioral commentary (Layer 2)")
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
    if args.no_rnn:
        overrides["use_rnn"] = False

    cfg = MAPPOConfig.from_algo(
        args.algo,
        size=args.size, n_agents=args.n_agents, difficulty=args.difficulty,
        seed=args.seed,
        total_steps=args.total_steps if args.total_steps is not None
        else MAPPOConfig.total_steps,
        **overrides,
    )
    n_updates = args.updates if args.updates is not None else cfg.num_updates

    now = datetime.now()
    date_time = now.strftime("%Y_%m_%d_%H_%M_%S")
    
    net_tag = "" if cfg.use_rnn else "_fc"
    if args.resume:
        run_dir = args.run_dir or os.path.join(
        "runs","resume",f"{date_time}" ,f"{cfg.algo}{net_tag}_Warehouse_{cfg.size}-{cfg.n_agents}ag")
    else:
        run_dir = args.run_dir or os.path.join(
        "runs",f"{date_time}" , f"{cfg.algo}{net_tag}_Warehouse_{cfg.size}-{cfg.n_agents}ag")
    csv_path = os.path.join(run_dir, "results.csv")
    batch_steps = cfg.batch_steps
    chunk = max(1, args.checkpoint_every)

    print(f"algo={cfg.algo} (centralised_critic={cfg.centralised_critic}, "
          f"use_ppo={cfg.use_ppo}, use_rnn={cfg.use_rnn})")
    print(f"env=rware-{cfg.size}-{cfg.n_agents}ag  parallel_envs={cfg.parallel_envs} "
          f"time_limit={cfg.time_limit}  updates={n_updates}  "
          f"(steps/update={batch_steps})  seed={cfg.seed}")
    print(f"run-dir={run_dir}  chunk={chunk} updates/save  max_to_keep={args.max_to_keep}")
    if n_updates % chunk != 0:
        print(f"  note: {n_updates} not divisible by chunk {chunk}; the final "
              f"short chunk triggers one extra XLA compile.")
    mgr = CheckpointManager(run_dir, max_to_keep=args.max_to_keep)
    mgr.save_config(cfg)
    trainer = make_resumable_train(cfg, live_log=not args.no_live_log)

    # ---- init or resume ----
    key = jax.random.PRNGKey(cfg.seed)
    carry = trainer["init_carry"](key)  # also the target structure for restore
    resume = args.resume and has_checkpoint(run_dir)
    start = 0
    if resume:
        sel = args.resume_from.lower()
        if sel == "latest":
            step = mgr.latest_step()
        elif sel == "best":
            step = mgr.best_step()
        else:
            step = int(args.resume_from)
            if step not in mgr.all_steps():
                raise SystemExit(
                    f"step {step} not in saved steps {mgr.all_steps()}")
        carry = mgr.restore(step, carry)
        start = int(step)
        print(f"resumed from checkpoint step {start} ({sel}) "
              f"({start * batch_steps:,} env steps)")

    logger = CSVLogger(csv_path, resume=resume)

    if start >= n_updates:
        print(f"nothing to do: start={start} >= n_updates={n_updates}")
        logger.close()
        mgr.wait()
        return

    if not args.no_live_log:
        print("live per-update log (return / deliveries / blocked% / idle% / "
              "entropy) follows; watch entropy — a fast drop toward 0 means "
              "exploration collapse:\n")

    # ---- chunked training loop (orbax saves between chunks, main thread) ----
    narrator = None if args.no_commentary else Narrator()
    ema = None
    t0 = time.perf_counter()
    upd = start
    while upd < n_updates:
        k = min(chunk, n_updates - upd)
        
        carry, metrics = trainer["train_from"](carry=carry, base_upd=upd, n=k,n_updates=n_updates)
        carry = jax.block_until_ready(carry)

        # all per-update metrics for this chunk, as np arrays of shape [k]
        m = {key: np.asarray(val) for key, val in metrics.items()}
        for i in range(k):
            done_count = upd + i + 1
            ret_i = float(m["episode_return"][i])
            ema = ret_i if ema is None else (
                args.ema_decay * ema + (1.0 - args.ema_decay) * ret_i)
            logger.log({
                "environment_steps": done_count * batch_steps,
                "updates": done_count,
                "episode_time":metrics["episode_time"][i],
                "mean_episode_returns": float(m["episode_return"][i]),
                "mean_entropy": float(m["entropy"][i]),
                "mean_loss": float(m["loss"][i]),
                "mean_actor_loss": float(m["actor_loss"][i]),
                "mean_value_loss": float(m["value_loss"][i]),
                "mean_reward_std": float(m["reward_std_mean"][i]),
                "mean_deliveries": float(m["deliveries"][i]),
                "mean_block_rate": float(m["block_rate"][i]),
                "mean_idle_rate": float(m["idle_rate"][i]),
                "mean_pickup_rate": float(m["pickup_rate"][i]),
                "mean_deliveries_early": float(m["deliveries_early"][i]),
                "mean_deliveries_mid": float(m["deliveries_mid"][i]),
                "mean_deliveries_late": float(m["deliveries_late"][i]),
                "mean_block_early": float(m["block_early"][i]),
                "mean_block_mid": float(m["block_mid"][i]),
                "mean_block_late": float(m["block_late"][i]),
                "mean_distance_traveled": int(m["distance_traveled"][i]),
                "mean_step_time": float(m["step_time"][i]),
                "mean_episode_time":float(m["episode_time"][i]),
                "mean_step_count":int(m["step_count"][i]),
                "mean_success":float(m["success"][i]),
                "mean_success_rate":float(m["success_rate"][i]),
                "mean_FPS":float(m["FPS"][i]),

                "std_episode_return": float(m["episode_return_std"][i]),
                "std_entropy": float(m["entropy_std"][i]),
                "std_loss": float(m["loss_std"][i]),
                "std_actor_loss": float(m["actor_loss_std"][i]),
                "std_value_loss": float(m["value_loss_std"][i]),
                "std_reward_std": float(m["reward_std_std"][i]),
                "std_deliveries": float(m["deliveries_std"][i]),          
                "std_block_rate": float(m["block_rate_std"][i]),              
                "std_idle_rate": float(m["idle_rate_std"][i]),
                "std_pickup_rate": float(m["pickup_rate_std"][i]),
                "std_deliveries_early": float(m["deliveries_early_std"][i]),
                "std_deliveries_mid": float(m["deliveries_mid_std"][i]),
                "std_deliveries_late": float(m["deliveries_late_std"][i]),
                "std_block_early": float(m["block_early_std"][i]),
                "std_block_mid": float(m["block_mid_std"][i]),
                "std_block_late": float(m["block_late_std"][i]),
                "std_distance_traveled": float(m["distance_traveled_std"][i]),
                "std_step_time": float(m["step_time_std"][i]),   
                "std_step_count": float(m["step_count_std"][i]),
                "std_success":float(m["success_std"][i]),
                "std_success_rate":float(m["success_rate_std"][i]),
                "std_FPS": float(m["FPS_std"][i]),
            })
        upd += k
        mgr.save(upd, carry, smoothed_return=ema)

        # Layer 2: one behavioral commentary block per chunk (host-side, between
        # chunks -> no effect on the fused-scan rollout speed).
        if narrator is not None:
            stats = {key: float(m[key].mean()) for key in (#[k,]
                "episode_return", "deliveries", "block_rate", "idle_rate",
                "deliveries_early", "deliveries_mid", "deliveries_late")}
            print(narrator.chunk(upd, stats), flush=True)

    mgr.wait()
    logger.close()
    dt = time.perf_counter() - t0

    ran = n_updates - start
    total_env_steps = ran * batch_steps
    best = mgr.best_step()
    print(f"\nran {ran} updates ({total_env_steps:,} env steps) "
          f"in {dt:.1f}s  ({total_env_steps / dt:,.0f} steps/s)")
    print(f"checkpoints -> {os.path.join(run_dir, 'checkpoints')}  "
          f"(latest={mgr.latest_step()}, best={best})   metrics -> {csv_path}")


if __name__ == "__main__":
    main()
