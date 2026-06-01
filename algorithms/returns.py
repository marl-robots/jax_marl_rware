"""Truncated n-step returns, matching marlbase utils/utils.py:compute_nstep_returns.

Reference (PyTorch):

    for t_start in range(ep_length):
        nstep_return_t = 0
        for step in range(nsteps + 1):
            t = t_start + step
            if t >= ep_length:           # episode ended -> stop (no bootstrap)
                break
            elif step == nsteps:         # bootstrap from target-critic value
                nstep_return_t += gamma**step * next_values[t] * (1 - done[t])
            else:
                nstep_return_t += gamma**step * rewards[t] * (1 - done[t])
        nstep_values[t_start] = nstep_return_t

Vectorised here with clamp+mask so it is fully jittable (no data-dependent break).
All arrays are shaped (T, B, N): time, parallel envs, agents.
"""

from __future__ import annotations

import jax.numpy as jnp


def compute_nstep_returns(rewards, done, next_values, nsteps: int, gamma: float):
    T = rewards.shape[0]
    steps = jnp.arange(nsteps + 1)               # [n+1]
    t = jnp.arange(T)[:, None] + steps[None, :]  # [T, n+1]
    valid = t < T
    t_clamped = jnp.minimum(t, T - 1)

    r_g = rewards[t_clamped]                      # [T, n+1, B, N]
    d_g = done[t_clamped]
    v_g = next_values[t_clamped]

    discount = (gamma ** steps)[None, :, None, None]
    is_boot = (steps == nsteps)[None, :, None, None]
    base = jnp.where(is_boot, v_g, r_g)
    term = discount * base * (1.0 - d_g) * valid[:, :, None, None]
    return term.sum(axis=1)                       # [T, B, N]
