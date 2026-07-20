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
from datetime import datetime

import jax
import numpy as np

from algorithms.checkpoint import CheckpointManager, has_checkpoint
from algorithms.commentary import Narrator
from algorithms.dqn_config import DQNConfig
from algorithms.dqn_train import make_emax_resumable_train
from algorithms.metrics import CSVLogger

from algorithms.metrics_parallel import BlockProcessor, init_Asylogger
from algorithms.replay import make_dqn_recorder, save_replay
from jaxrware.config import Action


def main():
    init_start=time.perf_counter()

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--algo",
        default="iql-emax",
        choices=["iql", "iql-emax", "qmix", "vdn"],
        help="pick one Off-Policy-family algorithms",
    )
    ap.add_argument("--size", default="tiny")
    ap.add_argument("--n-agents", type=int, default=4)
    ap.add_argument("--difficulty", default="normal")
    ap.add_argument("--time-limit", type=int, default=500)
    ap.add_argument("--parallel-envs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument(
        "--iters",
        type=int,
        default=800,
        help="collection iterations (each = parallel_envs episodes)",
    )
    ap.add_argument("--buffer-size", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--ensemble-size", type=int, default=5)
    ap.add_argument("--ucb-beta", type=float, default=1.0)
    ap.add_argument(
        "--use-rnn",
        action="store_true",
        help="recurrent (GRU) ensemble matching epymarl IDQN; rollout "
        "carries per-member hidden state. default use_rnn=True  -> recurrent (IQL) if use_rnn=False -> feedforward (VDN/QMIX)",
    )
    ap.add_argument(
        "--action-mask",
        action="store_true",
        # type=bool,
        # default=False,
        help="mask provably-no-op actions (FORWARD into a wall, inert "
        "TOGGLE_LOAD) at selection + target; optimum-preserving, "
        "removes wasted exploration. Off == unchanged behaviour.",
    )
    ap.add_argument(
        "--bptt-window",
        type=int,
        default=0,
        help="truncated-BPTT window for --use-rnn (0 = full episode). "
        "Cuts gradients every N steps to bound BPTT memory while "
        "keeping full batch; hidden still flows forward.",
    )
    ap.add_argument(
        "--updates-per-iter",
        type=int,
        default=None,
        help="EMAX grad steps per iter (default = parallel_envs)",
    )
    ap.add_argument(
        "--use_emax",
        action="store_true",
        help="Use EMAX version ",
    )
    ap.add_argument(
        "--checkpoint-every",
        type=int,
        default=50,
        help="iters per chunk = save cadence (pick a divisor of iters)",
    )
    ap.add_argument(
        "--max-to-keep",
        type=int,
        default=5,
        help="orbax keep-last-N (best is always retained too)",
    )
    ap.add_argument(
        "--ema-decay",
        type=float,
        default=0.95,
        help="EMA decay for the smoothed return used by keep-best",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="resume from a checkpoint in --run-dir if present",
    )
    ###################ADDED##########################
    ap.add_argument(
        "--run-dir",
        default=None,
        help="dir for checkpoints/ + results.csv " "(default runs/<env>_seed<seed>)",
    )
    ap.add_argument(
        "--full-metrics",
        action="store_true",
        help="add full metrics to logger",
    )
    ap.add_argument(
        "--resume-from",
        default="latest",
        help="which checkpoint to resume: 'latest', 'best', or a step "
        "number (requires --resume)",
    )
    ap.add_argument(
        "--no-live-log",
        action="store_true",
        help="disable the per-update stdout live log",
    )
    ap.add_argument(
        "--replay-seed",
        type=int,
        default=0,
        help="fixed RNG seed for replay episodes, so snapshots across "
        "training differ only by the policy",
    )
    ap.add_argument(
        "--replay-every",
        type=int,
        default=None,
        help="record an episode replay (.npz under <run-dir>/replays/) "
        "every K updates, at chunk boundaries (default: every "
        "chunk; 0 disables)",
    )
    ap.add_argument(
        "--no-commentary",
        action="store_true",
        help="disable the per-chunk behavioral commentary (Layer 2)",
    )
    args = ap.parse_args()

    cfg = DQNConfig.from_algo(
        args.algo,
        size=args.size,
        n_agents=args.n_agents,
        seed=args.seed,
        parallel_envs=args.parallel_envs,
        time_limit=args.time_limit,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        ensemble_size=args.ensemble_size,
        ucb_beta=args.ucb_beta,
        use_rnn=args.use_rnn,
        bptt_window=args.bptt_window,
        use_action_mask=args.action_mask,
        lr=args.lr,
        use_emax=args.use_emax,
    )

    iters = args.iters
    chunk = max(1, args.checkpoint_every)
    now = datetime.now()
    date_time = now.strftime("%Y_%m_%d_%H_%M_%S")
    net_tag = ("_rnn" if cfg.use_rnn else "_fc") + (
        "_mask" if cfg.use_action_mask else ""
    )
    if args.resume:
        run_dir = args.run_dir or os.path.join(
            "runs",
            "resume",
            f"{date_time}",
            f"{cfg.algo}{net_tag}_Warehouse_{cfg.size}-{cfg.n_agents}ag_seed{cfg.seed}",
        )
    else:
        run_dir = args.run_dir or os.path.join(
            "runs",
            f"{date_time}",
            f"{cfg.algo}{net_tag}_Warehouse_{cfg.size}-{cfg.n_agents}ag_seed{cfg.seed}",
        )
    csv_path = os.path.join(run_dir, "results.csv")

    print(f"backend={jax.default_backend()}  devices={jax.devices()}")
    print(
        f"algo={cfg.algo}  env=rware-{cfg.size}-{cfg.n_agents}ag  "
        f"parallel_envs={cfg.parallel_envs}  time_limit={cfg.time_limit}  "
        f"K={cfg.ensemble_size} beta={cfg.ucb_beta}  buffer={cfg.buffer_size}ep "
        f"batch={cfg.batch_size}ep  mask={cfg.use_action_mask}  "
        f"iters={iters}  chunk={chunk}"
    )

    mgr = CheckpointManager(run_dir, max_to_keep=args.max_to_keep)
    mgr.save_config(cfg)
    trainer = make_emax_resumable_train(
        cfg, updates_per_iter=args.updates_per_iter, live_log=(not args.no_live_log)
    )
    key = jax.random.PRNGKey(cfg.seed)
    carry = trainer["init_carry"](key)
    start = 0
    resume = args.resume and has_checkpoint(run_dir)
    if resume:
        step = mgr.latest_step()
        target = trainer["save_subtree"](carry)
        if step is not None:
            saved = mgr.restore(step, target)
            carry = trainer["merge_subtree"](carry, saved)
            start = int(step)
            print(f"resumed from checkpoint iter {start}")
    E, T = cfg.parallel_envs, cfg.time_limit
    A = len(Action)
    N = cfg.n_agents
    use_full_metrics = args.full_metrics
    block_processor=None
    if use_full_metrics:
        logger = init_Asylogger(csv_path, E, N, A, use_full_metrics, resume,True)
    else:
        logger = CSVLogger(csv_path, resume,True)
    if start >= iters:
        print(f"nothing to do: start={start} >= iters={iters}")
        log=logger.result() if not isinstance(logger,CSVLogger) else logger
        log.close()
        mgr.wait()
        return
    block_processor = BlockProcessor(logger)

    if not args.no_live_log:
        print(
            "live per-update log (return / deliveries / blocked% / idle% / "
            "epsilon) follows; watch epsilon\n"
        )
    # ---- chunked training loop (orbax saves between chunks, main thread) ----

    narrator = None if args.no_commentary else Narrator()
    replay_every = chunk if args.replay_every is None else args.replay_every
    recorder = (
        make_dqn_recorder(trainer["env"], trainer["qnet"], cfg)
        if replay_every > 0
        else None
    )
    replay_key = jax.random.PRNGKey(args.replay_seed)
    next_replay = (
        (start // replay_every + 1) * replay_every if replay_every > 0 else None
    )

    def record_snapshot(upd_now: int) -> None:
        """One fixed-seed eval episode from the current qnet -> replays/."""
        if recorder is not None:
            return
            # for k ,val in carry[0]["params"].items():
            #        for key ,value in val.items():
            #            print(k,key,value.shape)
            # exit(0)
            traj, scalars = recorder(
                carry[0]["params"], replay_key, cfg.use_action_mask
            )
            path = save_replay(
                run_dir,
                traj,
                scalars,
                env=trainer["env"],
                cfg=cfg,
                update=upd_now,
                env_steps=upd_now * T * E,
                seed=args.replay_seed,
                algo=cfg.algo,
            )
            print(
                f"  [replay @ update {upd_now}] return {float(scalars[0]):.2f}  "
                f"deliveries {int(scalars[1])}  -> {path}",
                flush=True,
            )

    if recorder is not None:
        record_snapshot(start)
    last_replay = start

    ema = None
    t0 = time.perf_counter()
    it = start
    init_end=time.perf_counter()
    print("init_time total:",init_end-init_start)
    while it < iters:
        k = min(chunk, iters - it)
        carry, (metrics_tensors, episode_metrics) = trainer["train_from"](
            carry=carry, iters=iters, base_iter=it, n=k
        )

        carry = jax.block_until_ready(carry)
        if use_full_metrics and block_processor is not None:
            args_taks = (metrics_tensors, it, T * E, A)
            block_processor.add_task(args_taks)
        m = {key: np.asarray(val) for key, val in episode_metrics.items()}

        deliv = None
        ret = None
        for i in range(k):
            done = it + i + 1
            deliv = float(m["deliveries"][i])
            ret = float(m["episode_return"][i])
            ema = (
               ret
               if ema is None
               else (args.ema_decay * ema + (1 - args.ema_decay) * ret)
            )

            if not use_full_metrics and isinstance(logger,CSVLogger) :
                logger.log(
                    {
                        "environment_steps": done * T * E,
                        "updates": done,
                        "episode_returns_mean": ret,
                        "loss_mean": float(m["loss"][i]),
                        "epsilon": float(m["epsilon"][i]),
                        "reward_std_mean": float(
                            metrics_tensors["rewards_std_mean"][i].sum()
                        ),
                        "deliveries_mean": deliv,
                        "block_rate_mean": float(m["block_rate"][i]),
                        "idle_rate_mean": float(m["idle_rate"][i]),
                        "pickup_rate_mean": float(m["pickup_rate"][i]),
                        "deliveries_early_mean": float(m["deliveries_early"][i]),
                        "deliveries_mid_mean": float(m["deliveries_mid"][i]),
                        "deliveries_late_mean": float(m["deliveries_late"][i]),
                        "block_early_mean": float(m["block_early"][i]),
                        "block_mid_mean": float(m["block_mid"][i]),
                        "block_late_mean": float(m["block_late"][i]),
                    }
                )
        it += k
        if recorder is not None and next_replay is not None and it >= next_replay:
            record_snapshot(it)
            last_replay = it
            while next_replay <= it:
                next_replay += replay_every
        elapsed = time.perf_counter() - t0
        steps = (it - start) * E * T
        if not args.no_live_log:
            print(
                f"  iter {it:4d}/{iters} | deliv {deliv:6.2f} (ema {ema:5.2f}) | "
                f"return {ret:7.3f} | loss {float(m['loss'][-1]):8.4f} | "
                f"{steps / max(elapsed, 1e-9):,.0f} steps/s",
                flush=True,
            )
        if narrator is not None:
            stats = {
                key: float(episode_metrics[key].mean())
                for key in (  # [k,]
                    "episode_return",
                    "deliveries",
                    "block_rate",
                    "idle_rate",
                    "deliveries_early",
                    "deliveries_mid",
                    "deliveries_late",
                )
            }
            print(narrator.chunk(it, stats), flush=True)
        #if ema is not None:
        #    mgr.save(it, trainer["save_subtree"](carry), smoothed_return=ema)

    if recorder is not None and last_replay != iters:
        record_snapshot(iters)
    mgr.wait()
    if block_processor is not None:
        block_processor.close()
    log=logger.result() if not isinstance(logger,CSVLogger) else logger
    log.close()
    dt = time.perf_counter() - t0
    ran = iters - start
    total = ran * E * T
    best = mgr.best_step()
    print(
        f"\nran {ran} iters ({total:,} env steps) in {dt:.1f}s "
        f"({total / dt:,.0f} steps/s)  best_iter={best}  -> {run_dir}"
    )
    print(
        f"checkpoints -> {os.path.join(run_dir, 'checkpoints')}  "
        f"latest={mgr.latest_step()}   metrics -> {csv_path}"
    )
    path = os.path.join(f"{run_dir}", "DONE")
    open(path, "w").close()


if __name__ == "__main__":
    main()
