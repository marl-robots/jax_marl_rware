"""Render a MAPPO checkpoint: roll out one episode and write a video/frames.

Loads the run's saved config (``<run-dir>/config.json``), restores the actor
params from a checkpoint (latest / best / a specific step), rolls out a single
environment for one episode carrying the shared actor's GRU hidden state, and
draws each step with the headless Pillow renderer (jaxrware/render.py).

Output format is chosen with --format:
  * mp4 (default)  -- single H.264 video (imageio + imageio-ffmpeg)
  * gif            -- single animated GIF
  * png            -- a directory of numbered frame PNGs

Action selection with --action: 'sampled' (default, stochastic policy) or
'greedy' (argmax logits).

Examples (WSL, conda env jax_env_1):
    python -m scripts.render_rollout --run-dir runs/tiny-4ag_seed2
    python -m scripts.render_rollout --run-dir runs/tiny-4ag_seed2 --step best \
        --format gif --action greedy
"""

from __future__ import annotations

import argparse
import os

import imageio.v2 as imageio
import jax
import jax.numpy as jnp
import numpy as np

from algorithms.checkpoint import CheckpointManager, has_checkpoint
from algorithms.config import MAPPOConfig
from algorithms.mappo import make_resumable_train
from algorithms.networks import ScannedGRU
from jaxrware.render import render_state


def _select_step(mgr: CheckpointManager, sel: str) -> int:
    sel = sel.lower()
    if sel == "latest":
        return mgr.latest_step()
    if sel == "best":
        return mgr.best_step()
    step = int(sel)
    if step not in mgr.all_steps():
        raise SystemExit(f"step {step} not in saved steps {mgr.all_steps()}")
    return step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True,
                    help="run directory containing config.json + checkpoints/")
    ap.add_argument("--step", default="best",
                    help="checkpoint to render: 'best', 'latest', or a step number")
    ap.add_argument("--format", choices=["mp4", "gif", "png"], default="mp4")
    ap.add_argument("--action", choices=["sampled", "greedy"], default="sampled")
    ap.add_argument("--out", default=None, help="output file (or dir for png)")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="frames to render (default = env time_limit)")
    ap.add_argument("--seed", type=int, default=0, help="rollout RNG seed")
    ap.add_argument("--fps", type=int, default=15, help="video/gif frame rate")
    args = ap.parse_args()

    if not has_checkpoint(args.run_dir):
        raise SystemExit(f"no checkpoints found in {args.run_dir}")

    cfg = MAPPOConfig(**CheckpointManager.load_config(args.run_dir))
    mgr = CheckpointManager(args.run_dir)
    step = _select_step(mgr, args.step)

    trainer = make_resumable_train(cfg)
    env, actor = trainer["env"], trainer["actor"]
    N = cfg.n_agents
    obs_dim = env.obs_dim
    H = cfg.hidden_dim
    T = args.max_steps or cfg.time_limit

    # restore params (the carry is typed by a fresh init_carry)
    key = jax.random.PRNGKey(args.seed)
    carry = trainer["init_carry"](key)
    carry = mgr.restore(step, carry)
    actor_params = carry[0]["actor"]

    print(f"rendering step {step} from {args.run_dir}  "
          f"(env=rware-{cfg.size}-{cfg.n_agents}ag, action={args.action}, T={T})")

    # ---- single-env rollout, shared actor folded into the agent batch (B=N) ----
    # Run the whole episode as one on-device `lax.scan` (no per-step host sync),
    # collecting the per-step EnvState + reward/done/deliveries. We block ONCE,
    # then do all the Pillow drawing host-side -- otherwise 500 device->host
    # syncs dominate and the render crawls.
    state0, obs0 = env.reset(jax.random.split(key)[0])
    h_actor0 = ScannedGRU.initialize_carry(N, H)
    zeros_1N = jnp.zeros((1, N))
    greedy = args.action == "greedy"

    def rollout_step(carry, _):
        state, obs, h_actor, rng = carry
        rng, ksamp = jax.random.split(rng)
        obs_flat = obs.reshape(N, obs_dim)
        h_actor, dist = actor.apply(actor_params, h_actor, (obs_flat[None], zeros_1N))
        logits = dist.logits[0]  # [N, num_actions]
        if greedy:
            actions = jnp.argmax(logits, axis=-1)
        else:
            actions = jax.random.categorical(ksamp, logits)
        nstate, nobs, reward, done, info = env.step(state, actions)
        out = (nstate, reward.sum(), done, info["deliveries"].sum())
        return (nstate, nobs, h_actor, rng), out

    _, (traj_states, rewards, dones, delivs) = jax.lax.scan(
        rollout_step, (state0, obs0, h_actor0, key), None, length=T
    )
    traj_states, rewards, dones, delivs = jax.block_until_ready(
        (traj_states, rewards, dones, delivs)
    )

    # truncate at first done (rware episodes are fixed-length, so usually all T)
    dones = np.asarray(dones)
    n = int(np.argmax(dones)) + 1 if dones.any() else T
    rewards = np.asarray(rewards)
    delivs = np.asarray(delivs)
    traj_states = jax.tree_util.tree_map(np.asarray, traj_states)

    frames = [render_state(env, state0)]  # initial state
    for t in range(n):
        st = jax.tree_util.tree_map(lambda x: x[t], traj_states)
        frames.append(render_state(env, st))
    total_reward = float(rewards[:n].sum())
    deliveries = int(delivs[:n].sum())

    print(f"episode: {n} steps  team_return={total_reward:.2f}  "
          f"deliveries={deliveries}")

    # ---- encode ----
    ext = args.format
    out = args.out or os.path.join(
        args.run_dir, f"rollout_step{step}_{args.action}.{ext if ext != 'png' else ''}".rstrip("."))
    if ext == "png":
        os.makedirs(out, exist_ok=True)
        for i, fr in enumerate(frames):
            imageio.imwrite(os.path.join(out, f"frame_{i:04d}.png"), fr)
        print(f"wrote {len(frames)} PNG frames -> {out}/")
    elif ext == "gif":
        imageio.mimsave(out, frames, fps=args.fps, loop=0)
        print(f"wrote GIF ({len(frames)} frames @ {args.fps} fps) -> {out}")
    else:  # mp4
        imageio.mimsave(out, frames, fps=args.fps, codec="libx264",
                        macro_block_size=1)
        print(f"wrote MP4 ({len(frames)} frames @ {args.fps} fps) -> {out}")


if __name__ == "__main__":
    main()
