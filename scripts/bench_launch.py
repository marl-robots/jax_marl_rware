"""Micro-benchmark to isolate launch-bound behaviour and test command buffers.

Compiles once (warmup, discarded), then times K execution-only calls so the
reported steps/s reflects *execution*, not XLA compilation. Run with and
without XLA command-buffer flags to see if host kernel-launch overhead is the
bottleneck for the proven 10-env config.
"""
from __future__ import annotations

import os
import time

import jax
import numpy as np

from algorithms.config import MAPPOConfig
from algorithms.mappo import make_train


def main():
    updates = int(os.environ.get("BENCH_UPDATES", "100"))
    envs = int(os.environ.get("BENCH_ENVS", "10"))
    reps = int(os.environ.get("BENCH_REPS", "3"))

    print("devices:", jax.devices())
    print("XLA_FLAGS:", os.environ.get("XLA_FLAGS", "(unset)"))

    cfg = MAPPOConfig(size="tiny", n_agents=4, difficulty="normal", seed=2,
                      parallel_envs=envs)
    train, _ = make_train(cfg, num_updates=updates)
    train = jax.jit(train)

    steps_per_call = updates * cfg.batch_steps

    # warmup = compile (timed separately, then discarded)
    t0 = time.perf_counter()
    out = jax.block_until_ready(train(jax.random.PRNGKey(0)))
    compile_s = time.perf_counter() - t0
    print(f"compile+first run: {compile_s:.1f}s")

    # execution-only timing
    times = []
    for i in range(reps):
        t0 = time.perf_counter()
        out = jax.block_until_ready(train(jax.random.PRNGKey(i + 1)))
        times.append(time.perf_counter() - t0)
    times = np.array(times)
    sps = steps_per_call / times
    print(f"envs={envs}  updates={updates}  steps/call={steps_per_call:,}")
    print(f"exec time per call: {times.mean():.2f}s +/- {times.std():.2f}  "
          f"(min {times.min():.2f})")
    print(f"steps/s (exec only): {sps.mean():,.0f}  (max {sps.max():,.0f})")


if __name__ == "__main__":
    main()
