"""On-device, fully-jitted, GPU trainer for IDQN-EMAX -- mirrors mappo.py's
make_resumable_train pattern instead of a host Python loop.

The whole training loop is a `jax.lax.scan` over collection iterations; the
rollout is an inner scan; the replay buffer lives ON-DEVICE inside the scan
carry as fixed-size ring arrays, sampled in-graph with `jax.random`. Nothing
touches host Python mid-run, and there is no host<->device copy per update --
that is the difference from scripts/train_dqn.py (kept for reference) and the
reason this is fast on GPU.

Each collection iteration:
  1. roll out E full episodes with UCB(+epsilon warmup) actions (inner scan);
  2. scatter them into the device ring buffer (dynamic indices);
  3. once the buffer is warm (>= batch_size episodes), take `updates_per_iter`
     EMAX gradient steps, each sampling a fresh minibatch in-graph
     (lax.cond gates this until warm);
EMAX has no target network, so the carry is just trainable state + buffer.

`train_from` is jit'd with a static chunk length and scans `n` iterations, so
the host loop only runs between chunks to checkpoint (orbax) + log CSV -- the
exact division of labour mappo.py uses. Checkpoints save only the small
trainable subtree (not the big replay buffer); on resume the buffer re-warms.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp

from jaxrware import Warehouse, make_config
from .dqn_networks import QNetwork
from .dqn import (
    _augment_obs, _ensemble_q, epsilon_at, emax_update, feed_batch,
    init_ensemble, make_optimizer, rms_init,
)


def make_emax_resumable_train(cfg, updates_per_iter: int | None = None):
    env = Warehouse(make_config(cfg.size, cfg.n_agents, cfg.difficulty))
    N, E, T = cfg.n_agents, cfg.parallel_envs, cfg.time_limit
    A, H = env.num_actions, cfg.hidden_dim
    B = E * N
    K = cfg.ensemble_size
    C = cfg.buffer_size
    bs = cfg.batch_size
    in_dim = env.obs_dim + (N if cfg.obs_agent_id else 0)
    n_grad = E if updates_per_iter is None else updates_per_iter  # grad steps/iter
    qnet = QNetwork(num_actions=A, hidden_dim=H, use_rnn=cfg.use_rnn)
    tx = make_optimizer(cfg)

    # -- trainable subtree <-> full carry (buffer excluded from checkpoints) ----
    def init_carry(key):
        key, ki = jax.random.split(key)
        params_ens = init_ensemble(qnet, ki, K, B, H, in_dim)
        opt_state = tx.init(params_ens)
        rew_ms = rms_init((1,))
        buf_obs = jnp.zeros((C, T + 1, N, in_dim), jnp.float32)
        buf_act = jnp.zeros((C, T, N), jnp.int32)
        buf_rew = jnp.zeros((C, T), jnp.float32)
        ptr = jnp.array(0, jnp.int32)
        count = jnp.array(0, jnp.int32)
        return (params_ens, opt_state, rew_ms, buf_obs, buf_act, buf_rew,
                ptr, count, key)

    def save_subtree(carry):
        params_ens, opt_state, rew_ms, _, _, _, ptr, count, key = carry
        return (params_ens, opt_state, rew_ms, ptr, count, key)

    def merge_subtree(full_carry, saved):
        params_ens, opt_state, rew_ms, ptr, count, key = saved
        _, _, _, buf_obs, buf_act, buf_rew, _, _, _ = full_carry
        # buffer re-warms from empty on resume
        return (params_ens, opt_state, rew_ms,
                jnp.zeros_like(buf_obs), jnp.zeros_like(buf_act),
                jnp.zeros_like(buf_rew), jnp.array(0, jnp.int32),
                jnp.array(0, jnp.int32), key)

    # -- in-graph rollout: E episodes of UCB(+eps) actions --------------------
    def rollout(params_ens, key, epsilon):
        key, kreset = jax.random.split(key)
        states, obs = jax.vmap(env.reset)(jax.random.split(kreset, E))
        obs = _augment_obs(obs, N, cfg.obs_agent_id)

        def step(carry, _):
            states, obs, key = carry
            key, ka, kr = jax.random.split(key, 3)
            obs_flat = obs.reshape(B, in_dim)
            q = _ensemble_q(qnet, params_ens, obs_flat[None], B, H)[:, 0]  # [K,B,A]
            ucb = q.mean(axis=0) + cfg.ucb_beta * q.std(axis=0)
            greedy = jnp.argmax(ucb, axis=-1)
            rand_a = jax.random.randint(ka, (B,), 0, A)
            actions = jnp.where(jax.random.uniform(kr, (B,)) < epsilon,
                                rand_a, greedy).reshape(E, N)
            nstates, nobs, rewards, done, info = jax.vmap(env.step)(states, actions)
            nobs = _augment_obs(nobs, N, cfg.obs_agent_id)
            return (nstates, nobs, key), (obs_flat, actions, rewards.sum(-1),
                                          info["deliveries"], rewards)

        (_, final_obs, _), (obs_t, act_t, rew_t, deliv_t, raw_r_t) = jax.lax.scan(
            step, (states, obs, key), None, length=T)
        obs_seq = jnp.concatenate([obs_t, final_obs.reshape(1, B, in_dim)], axis=0)
        obs_b = obs_seq.reshape(T + 1, E, N, in_dim).transpose(1, 0, 2, 3)
        metrics = {"episode_return": raw_r_t.sum(0).sum(-1).mean(),
                   "deliveries": deliv_t.sum(0).sum(-1).mean()}
        return obs_b, act_t.transpose(1, 0, 2), rew_t.transpose(1, 0), metrics

    # -- one collection iteration ---------------------------------------------
    def update_step(carry, idx):
        (params_ens, opt_state, rew_ms, buf_obs, buf_act, buf_rew,
         ptr, count, key) = carry
        key, kroll = jax.random.split(key)
        eps = epsilon_at(cfg, idx * E * T)
        obs_b, act_b, rew_b, metrics = rollout(params_ens, kroll, eps)

        # scatter E episodes into the ring buffer
        pos = (ptr + jnp.arange(E)) % C
        buf_obs = buf_obs.at[pos].set(obs_b)
        buf_act = buf_act.at[pos].set(act_b)
        buf_rew = buf_rew.at[pos].set(rew_b)
        ptr = (ptr + E) % C
        count = jnp.minimum(count + E, C)

        def train_branch(op):
            params_ens, opt_state, rew_ms, key = op

            def grad_step(c, _):
                params_ens, opt_state, rew_ms, key = c
                key, ks = jax.random.split(key)
                idxs = jax.random.randint(ks, (bs,), 0, count)  # sample warm portion
                batch = feed_batch(buf_obs[idxs], buf_act[idxs], buf_rew[idxs], T, N)
                params_ens, opt_state, rew_ms, diag = emax_update(
                    qnet, tx, cfg, params_ens, opt_state, rew_ms, batch)
                return (params_ens, opt_state, rew_ms, key), diag["loss"]

            (params_ens, opt_state, rew_ms, key), losses = jax.lax.scan(
                grad_step, op, None, length=n_grad)
            return params_ens, opt_state, rew_ms, key, losses.mean()

        def skip_branch(op):
            params_ens, opt_state, rew_ms, key = op
            return params_ens, opt_state, rew_ms, key, jnp.array(jnp.nan, jnp.float32)

        params_ens, opt_state, rew_ms, key, loss = jax.lax.cond(
            count >= bs, train_branch, skip_branch,
            (params_ens, opt_state, rew_ms, key))

        carry = (params_ens, opt_state, rew_ms, buf_obs, buf_act, buf_rew,
                 ptr, count, key)
        metrics = {**metrics, "loss": loss, "epsilon": eps}
        return carry, metrics

    @functools.partial(jax.jit, static_argnums=(2,))
    def train_from(carry, base_iter, n):
        idxs = base_iter + jnp.arange(n)
        return jax.lax.scan(update_step, carry, idxs, length=n)

    return {"env": env, "qnet": qnet, "init_carry": init_carry,
            "train_from": train_from, "save_subtree": save_subtree,
            "merge_subtree": merge_subtree}
