"""Fast on-device GPU trainer for IDQN-EMAX, mirroring scripts/train_mappo.py.

The whole loop is a fused jitted lax.scan (algorithms/dqn_train.py); this host
script only drives chunks and, between them, checkpoints (orbax) + logs CSV --
the same main-thread division of labour as the MAPPO trainer. Runs on the
DEFAULT backend (GPU if present); do NOT set JAX_PLATFORMS=cpu.

Supersedes the host-loop scripts/train_dqn.py (kept for reference). Example:
    python -m scripts.train_dqn_fast --iters 800 --checkpoint-every 50
    python -m scripts.train_dqn_fast --iters 800 --resume
"""

from __future__ import annotations

import argparse
import os
import time

import jax
import numpy as np

from algorithms.checkpoint import CheckpointManager, has_checkpoint
from algorithms.dqn_config import DQNConfig
from algorithms.dqn_train import make_emax_resumable_train
from algorithms.metrics import CSVLogger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="tiny")
    ap.add_argument("--n-agents", type=int, default=4)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--iters", type=int, default=800,
                    help="collection iterations (each = parallel_envs episodes)")
    ap.add_argument("--parallel-envs", type=int, default=8)
    ap.add_argument("--time-limit", type=int, default=500)
    ap.add_argument("--buffer-size", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--ensemble-size", type=int, default=5)
    ap.add_argument("--ucb-beta", type=float, default=1.0)
    ap.add_argument("--use-rnn", action="store_true",
                    help="recurrent (GRU) ensemble matching epymarl IDQN; rollout "
                         "carries per-member hidden state.")
    ap.add_argument("--action-mask", action="store_true",
                    help="mask provably-no-op actions (FORWARD into a wall, inert "
                         "TOGGLE_LOAD) at selection + target; optimum-preserving, "
                         "removes wasted exploration. Off == unchanged behaviour.")
    ap.add_argument("--bptt-window", type=int, default=0,
                    help="truncated-BPTT window for --use-rnn (0 = full episode). "
                         "Cuts gradients every N steps to bound BPTT memory while "
                         "keeping full batch; hidden still flows forward.")
    ap.add_argument("--updates-per-iter", type=int, default=None,
                    help="EMAX grad steps per iter (default = parallel_envs)")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--checkpoint-every", type=int, default=50,
                    help="iters per chunk = save cadence (pick a divisor of iters)")
    ap.add_argument("--max-to-keep", type=int, default=5)
    ap.add_argument("--ema-decay", type=float, default=0.95)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    cfg = DQNConfig.from_algo(
        "iql-emax", size=args.size, n_agents=args.n_agents, seed=args.seed,
        parallel_envs=args.parallel_envs, time_limit=args.time_limit,
        buffer_size=args.buffer_size, batch_size=args.batch_size,
        ensemble_size=args.ensemble_size, ucb_beta=args.ucb_beta,
        use_rnn=args.use_rnn, bptt_window=args.bptt_window,
        use_action_mask=args.action_mask,
    )
    iters = args.iters
    chunk = max(1, args.checkpoint_every)
    net_tag = ("_rnn" if cfg.use_rnn else "_fc") + ("_mask" if cfg.use_action_mask else "")
    run_dir = args.run_dir or os.path.join(
        "runs", f"emax{net_tag}_{cfg.size}-{cfg.n_agents}ag_K{cfg.ensemble_size}_seed{cfg.seed}")
    csv_path = os.path.join(run_dir, "results.csv")

    print(f"backend={jax.default_backend()}  devices={jax.devices()}")
    print(f"algo={cfg.algo}  env=rware-{cfg.size}-{cfg.n_agents}ag  "
          f"parallel_envs={cfg.parallel_envs}  time_limit={cfg.time_limit}  "
          f"K={cfg.ensemble_size} beta={cfg.ucb_beta}  buffer={cfg.buffer_size}ep "
          f"batch={cfg.batch_size}ep  mask={cfg.use_action_mask}  "
          f"iters={iters}  chunk={chunk}")

    trainer = make_emax_resumable_train(cfg, updates_per_iter=args.updates_per_iter)
    mgr = CheckpointManager(run_dir, max_to_keep=args.max_to_keep)
    mgr.save_config(cfg)

    key = jax.random.PRNGKey(cfg.seed)
    carry = trainer["init_carry"](key)
    start = 0
    resume = args.resume and has_checkpoint(run_dir)
    if resume:
        step = mgr.latest_step()
        target = trainer["save_subtree"](carry)
        saved = mgr.restore(step, target)
        carry = trainer["merge_subtree"](carry, saved)
        start = int(step)
        print(f"resumed from checkpoint iter {start}")

    logger = CSVLogger(csv_path, resume=resume)
    E, T = cfg.parallel_envs, cfg.time_limit
    ema = None
    t0 = time.perf_counter()
    it = start
    while it < iters:
        k = min(chunk, iters - it)
        carry, metrics = trainer["train_from"](carry, it, k)
        carry = jax.block_until_ready(carry)
        m = {key_: np.asarray(val) for key_, val in metrics.items()}
        for i in range(k):
            done = it + i + 1
            deliv = float(m["deliveries"][i])
            ret = float(m["episode_return"][i])
            ema = deliv if ema is None else args.ema_decay * ema + (1 - args.ema_decay) * deliv
            logger.log({
                "environment_steps": done * E * T, "updates": done,
                "mean_episode_returns": ret, "deliveries": deliv,
                "loss": float(m["loss"][i]),
            })
        it += k
        mgr.save(it, trainer["save_subtree"](carry), smoothed_return=ema)
        elapsed = time.perf_counter() - t0
        steps = (it - start) * E * T
        print(f"  iter {it:4d}/{iters} | deliv {deliv:6.2f} (ema {ema:5.2f}) | "
              f"return {ret:7.3f} | loss {float(m['loss'][-1]):8.4f} | "
              f"{steps / max(elapsed, 1e-9):,.0f} steps/s", flush=True)

    mgr.wait()
    logger.close()
    dt = time.perf_counter() - t0
    total = (iters - start) * E * T
    print(f"\nran {iters - start} iters ({total:,} env steps) in {dt:.1f}s "
          f"({total / dt:,.0f} steps/s)  best_iter={mgr.best_step()}  -> {run_dir}")


if __name__ == "__main__":
    main()
