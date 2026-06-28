"""Environment throughput benchmark: jaxrware vs Jumanji vs original rware.

Measures raw env steps/second under random actions on the canonical tiny-4ag
task. Each implementation runs in its own conda env / process and APPENDS its
rows to docs/data/speed_bench.json, so results accumulate across runs:

  # ours + original (conda env jax_env_1)
  python scripts/bench_speed.py --impl jaxrware --device gpu
  python scripts/bench_speed.py --impl jaxrware --device cpu
  python scripts/bench_speed.py --impl rware
  # jumanji (conda env jumanji_135_01)
  python scripts/bench_speed.py --impl jumanji --device gpu

Protocol: for JAX impls, a jitted lax.scan of `--steps` env steps per batch
size in --batches, vmapped over the batch; compile excluded (one warmup call);
best of --repeats timings. steps/sec counts ENV steps (x batch), not agent
steps. Original rware is a single-process python/gym loop (its real usage;
vector envs scale at best linearly with CPU processes).

The 'steps' here are environment transitions for the whole 4-agent team.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

# allow `python scripts/bench_speed.py` as well as `python -m scripts.bench_speed`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT_PATH = os.path.join("docs", "data", "speed_bench.json")


def _append(rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    blob = []
    if os.path.isfile(OUT_PATH):
        with open(OUT_PATH) as f:
            blob = json.load(f)
    # replace rows with the same identity (impl, device, batch)
    keys = {(r["impl"], r["device"], r["batch"]) for r in rows}
    blob = [r for r in blob if (r["impl"], r["device"], r["batch"]) not in keys]
    blob.extend(rows)
    with open(OUT_PATH, "w") as f:
        json.dump(blob, f, indent=1)
    print(f"\nwrote {len(rows)} rows -> {OUT_PATH}")


def _row(impl: str, device: str, batch: int, sps: float, note: str = "") -> dict:
    return {
        "impl": impl, "device": device, "batch": batch,
        "steps_per_sec": round(sps, 1), "task": "tiny-4ag",
        "host": platform.node(), "note": note,
    }


# ---------------------------------------------------------------------------
def bench_jax_env(step_batch, reset_batch, sample_actions, batch: int,
                  steps: int, repeats: int) -> float:
    """Generic JAX benchmark: jitted scan of `steps` steps over a batch."""
    import jax
    import jax.numpy as jnp

    @jax.jit
    def run(states, key):
        def body(carry, _):
            states, key = carry
            key, k = jax.random.split(key)
            actions = sample_actions(k)
            states = step_batch(states, actions)
            return (states, key), None
        (states, key), _ = jax.lax.scan(body, (states, key), None, length=steps)
        return states, key

    key = jax.random.PRNGKey(0)
    states = reset_batch(jax.random.split(key, batch))
    states, key = jax.block_until_ready(run(states, key))  # compile + warmup

    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        states, key = jax.block_until_ready(run(states, key))
        best = min(best, time.perf_counter() - t0)
    return steps * batch / best


def bench_jaxrware(batches, steps, repeats) -> list[dict]:
    import jax
    import jax.numpy as jnp

    from jaxrware import Warehouse, make_config

    device = jax.default_backend()
    env = Warehouse(make_config("tiny", 4, "normal"))
    rows = []
    for batch in batches:
        def reset_batch(keys):
            states, _ = jax.vmap(env.reset)(keys)
            return states

        def step_batch(states, actions):
            nstates, *_ = jax.vmap(env.step)(states, actions)
            return nstates

        def sample_actions(k):
            return jax.random.randint(k, (batch, 4), 0, env.num_actions)

        sps = bench_jax_env(step_batch, reset_batch, sample_actions,
                            batch, steps, repeats)
        rows.append(_row("jaxrware (ours)", device, batch, sps))
        print(f"jaxrware  {device}  batch {batch:5d}: {sps:12,.0f} steps/s")
    return rows


def bench_jumanji(batches, steps, repeats) -> list[dict]:
    import jax
    import jax.numpy as jnp
    import jumanji
    from jumanji.environments.routing.robot_warehouse import RobotWarehouse
    from jumanji.environments.routing.robot_warehouse.generator import RandomGenerator

    device = jax.default_backend()
    env = RobotWarehouse(
        generator=RandomGenerator(column_height=8, shelf_rows=1,
                                  shelf_columns=3, num_agents=4,
                                  sensor_range=1, request_queue_size=4),
        time_limit=500)
    note = f"jumanji {jumanji.__version__}; terminates on collision (not reset)"
    rows = []
    for batch in batches:
        def reset_batch(keys):
            states, _ = jax.vmap(env.reset)(keys)
            return states

        def step_batch(states, actions):
            nstates, _ = jax.vmap(env.step)(states, actions)
            return nstates

        def sample_actions(k):
            return jax.random.randint(k, (batch, 4), 0, 5)

        sps = bench_jax_env(step_batch, reset_batch, sample_actions,
                            batch, steps, repeats)
        rows.append(_row("jumanji", device, batch, sps, note))
        print(f"jumanji   {device}  batch {batch:5d}: {sps:12,.0f} steps/s")
    return rows


def bench_rware(steps, repeats) -> list[dict]:
    import gymnasium as gym
    import numpy as np
    import rware  # noqa: F401

    env = gym.make("rware:rware-tiny-4ag-v2", max_steps=500)
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    acts = [rng.integers(0, 5, size=4).tolist() for _ in range(steps)]

    best = float("inf")
    for _ in range(repeats):
        env.reset(seed=0)
        t0 = time.perf_counter()
        for t, a in enumerate(acts):
            _, _, done, trunc, _ = env.step(a)
            if done or trunc:
                env.reset(seed=t)
        best = min(best, time.perf_counter() - t0)
    sps = steps / best
    print(f"rware     cpu  batch     1: {sps:12,.0f} steps/s")
    return [_row("rware (original)", "cpu", 1, sps,
                 "single python process; vector envs scale ~linearly "
                 "with CPU workers")]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", choices=["jaxrware", "jumanji", "rware"],
                    required=True)
    ap.add_argument("--device", choices=["cpu", "gpu"], default="gpu",
                    help="for JAX impls; set JAX_PLATFORMS accordingly "
                         "(this flag only labels + asserts)")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--batches", type=int, nargs="+",
                    default=[1, 32, 256, 1024, 4096])
    args = ap.parse_args()

    if args.impl == "rware":
        rows = bench_rware(args.steps, args.repeats)
    else:
        import jax
        backend = jax.default_backend()
        if backend != args.device:
            raise SystemExit(
                f"requested --device {args.device} but JAX backend is "
                f"{backend}; set JAX_PLATFORMS={args.device} (and for gpu, "
                "make sure the GPU is free)")
        fn = bench_jaxrware if args.impl == "jaxrware" else bench_jumanji
        rows = fn(args.batches, args.steps, args.repeats)
    _append(rows)
