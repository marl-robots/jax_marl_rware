"""Record episode replays (.npz) from saved checkpoints of a finished run.

Where ``scripts/render_rollout.py`` bakes pixels, this captures the raw
trajectory (agent poses, shelves, queue, events) that the arena dashboard's
animated player re-draws client-side — so one snapshot is ~tens of KB instead
of a video, and the player can interpolate/scrub/restyle it freely.

By default records ONE replay from the 'best' checkpoint; pass --all-steps to
record every saved checkpoint (gives a coarse policy-evolution scrubber for
runs that trained before --replay-every existed).

Examples (WSL, conda env jax_env_1):
    python -m scripts.record_replay --run-dir runs/mappo_tiny-4ag_seed2
    python -m scripts.record_replay --run-dir runs/mappo_tiny-4ag_seed2 --all-steps
    python -m scripts.record_replay --run-dir runs/ippo_tiny-4ag_seed2 \
        --step latest --episodes 3 --greedy
"""

from __future__ import annotations

import argparse

import jax

from algorithms.checkpoint import CheckpointManager, has_checkpoint
from algorithms.config import MAPPOConfig
from algorithms.mappo import make_resumable_train
from algorithms.replay import make_recorder, save_replay


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--step", default="best",
                    help="checkpoint: 'best', 'latest', or a step number")
    ap.add_argument("--all-steps", action="store_true",
                    help="record every saved checkpoint (ignores --step)")
    ap.add_argument("--episodes", type=int, default=1,
                    help="episodes per checkpoint (seeds seed, seed+1, ...)")
    ap.add_argument("--seed", type=int, default=0, help="replay RNG seed")
    ap.add_argument("--greedy", action="store_true",
                    help="argmax actions instead of sampling")
    args = ap.parse_args()

    if not has_checkpoint(args.run_dir):
        raise SystemExit(f"no checkpoints found in {args.run_dir}")

    cfg = MAPPOConfig(**CheckpointManager.load_config(args.run_dir))
    mgr = CheckpointManager(args.run_dir)

    if args.all_steps:
        steps = mgr.all_steps()
    else:
        sel = args.step.lower()
        if sel == "latest":
            steps = [mgr.latest_step()]
        elif sel == "best":
            steps = [mgr.best_step()]
        else:
            steps = [int(sel)]
            if steps[0] not in mgr.all_steps():
                raise SystemExit(
                    f"step {steps[0]} not in saved steps {mgr.all_steps()}")

    trainer = make_resumable_train(cfg)
    recorder = make_recorder(trainer["env"], trainer["actor"], cfg)
    carry = trainer["init_carry"](jax.random.PRNGKey(0))  # restore target

    for step in steps:
        restored = mgr.restore(step, carry)
        actor_params = restored[0]["actor"]
        for ep in range(args.episodes):
            key = jax.random.PRNGKey(args.seed + ep)
            traj, scalars = recorder(actor_params, key, args.greedy)
            path = save_replay(
                args.run_dir, traj, scalars, env=trainer["env"], cfg=cfg,
                update=step, env_steps=step * cfg.batch_steps,
                seed=args.seed + ep, greedy=args.greedy, algo=cfg.algo)
            print(f"update {step}  seed {args.seed + ep}  "
                  f"return {float(scalars[0]):.2f}  "
                  f"deliveries {int(scalars[1])}  -> {path}")


if __name__ == "__main__":
    main()
