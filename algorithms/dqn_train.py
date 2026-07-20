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

Optional action masking (cfg.use_action_mask): when on, the env emits a
[N, A] mask of provably-no-op actions (Warehouse.action_masks) that is applied
identically to the UCB greedy action, the epsilon-random action, and the
bootstrap-max target (emax_update). The mask is stored alongside obs in the
ring buffer so the target sees the same mask the behaviour did. When off, the
mask buffer is a zero-size placeholder and every numerical path is unchanged.
"""

from __future__ import annotations

import functools
import time

import jax
from jax.experimental import io_callback
import jax.numpy as jnp
from jaxrware import make_config, Warehouse

from algorithms.metrics_funcs import extract_flat_grads, gradient_norms

from .dqn import (
    _augment_obs,
    emax_update,
    epsilon_at,
    feed_batch,
    init_ensemble,
    make_optimizer,
    rms_init,
)
from .dqn_networks import QNetwork

_NEG = -1e8  # masked-action fill for argmax / categorical sampling
def _host_log(n_updates, upd, ret, eps, deliveries, block_rate, idle_rate):
    """Host-side live print, fired once per update via io_callback (cheap).

    Shows the behavioral view used for watching progress: team return,
    deliveries/episode, forward-block (contention) %, idle %, and entropy
    (watch entropy — a fast drop toward 0 means exploration collapse)."""
    print(
        f"  upd {int(upd):4d}/{int(n_updates):4d} | return {float(ret):8.3f} | "
        f"deliv {float(deliveries):6.2f} | blocked {float(block_rate) * 100:4.1f}% | "
        f"idle {float(idle_rate) * 100:4.1f}% | eps {float(eps):5.3f}",
        flush=True,
    )

def make_emax_resumable_train(cfg, updates_per_iter: int | None = None,live_log=False):
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
    mask_on = bool(cfg.use_action_mask)

    # -- trainable subtree <-> full carry (buffer excluded from checkpoints) ----
    def init_carry(key):
        key, ki = jax.random.split(key)
        params_ens = init_ensemble(qnet, ki, K, B, H, in_dim)
        opt_state = tx.init(params_ens)
        rew_ms = rms_init((1,))
        buf_obs = jnp.zeros((C, T + 1, N, in_dim), jnp.float32)
        buf_act = jnp.zeros((C, T, N), jnp.int32)
        buf_rew = jnp.zeros((C, T), jnp.float32)
        # zero-size placeholder when masking is off -> ~no memory, constant carry
        buf_mask = jnp.zeros((C, T + 1, N, A) if mask_on else (0,), jnp.float32)
        ptr = jnp.array(0, jnp.int32)
        count = jnp.array(0, jnp.int32)
        return (
            params_ens,
            opt_state,
            rew_ms,
            buf_obs,
            buf_act,
            buf_rew,
            buf_mask,
            ptr,
            count,
            key,
        )

    def save_subtree(carry):
        params_ens, opt_state, rew_ms, _, _, _, _, ptr, count, key = carry
        return (params_ens, opt_state, rew_ms, ptr, count, key)
        

    def merge_subtree(full_carry, saved):
        params_ens, opt_state, rew_ms, ptr, count, key = saved
        _, _, _, buf_obs, buf_act, buf_rew, buf_mask, _, _, _ = full_carry
        # buffer re-warms from empty on resume
        return (
            params_ens,
            opt_state,
            rew_ms,
            jnp.zeros_like(buf_obs),
            jnp.zeros_like(buf_act),
            jnp.zeros_like(buf_rew),
            jnp.zeros_like(buf_mask),
            jnp.array(0, jnp.int32),
            jnp.array(0, jnp.int32),
            key,
        )

    # -- in-graph rollout: E episodes of UCB(+eps) actions --------------------
    def rollout(params_ens, key, epsilon):
        key, kreset = jax.random.split(key)
        states, obs = jax.vmap(env.reset)(jax.random.split(kreset, E))
        obs = _augment_obs(obs, N, cfg.obs_agent_id)
        episode_start_time = time.perf_counter()

        def step(carry, _):
            states, obs, h_ens, key = carry
            key, ka, kr = jax.random.split(key, 3)
            obs_flat = obs.reshape(B, in_dim)

            # per-member single step carrying each member's GRU hidden state
            # (mirrors mappo.py rollout_step; for use_rnn=False the QNetwork
            # returns `hidden` unchanged, so this is correct for both nets).
            def one(p, h):
                h2, q = qnet.apply(p, h, (obs_flat[None], jnp.zeros((1, B))))
                return h2, q[0]  # q[0]: [B, A]

            h_ens, q = jax.vmap(one)(params_ens, h_ens)  # [K,B,H], [K,B,A]

            ucb = q.mean(axis=0) + cfg.ucb_beta * q.std(axis=0)  # [B, A]
            if mask_on:
                amask = jax.vmap(env.action_masks)(states).reshape(B, A)  # [B,A]
                greedy = jnp.argmax(jnp.where(amask > 0, ucb, _NEG), axis=-1)
                # epsilon-random restricted to valid actions (uniform over them)
                rand_a = jax.random.categorical(ka, jnp.where(amask > 0, 0.0, _NEG))
            else:
                greedy = jnp.argmax(ucb, axis=-1)
                rand_a = jax.random.randint(ka, (B,), 0, A)
            actions = jnp.where(
                jax.random.uniform(kr, (B,)) < epsilon, rand_a, greedy
            ).reshape(E, N)
            nstates, nobs, rewards, done, info = jax.vmap(env.step)(states, actions)
            nobs = _augment_obs(nobs, N, cfg.obs_agent_id)
            sig = (
                info["deliveries"],
                info["forward_blocked"],
                info["noop"],
                info["pickup"],
                info["drop"],
                info["distance_traveled"],
                info["step_time"],
            )  # each [E,N]
            out = (obs_flat, actions, rewards.sum(-1), *sig, rewards)
            if mask_on:
                out = out + (amask.reshape(E, N, A),)
            return (nstates, nobs, h_ens, key), out

        h0 = jnp.zeros((K, B, H))  # hidden starts at zeros each episode
        (fstates, final_obs, _, _), scanned = jax.lax.scan(
            step, (states, obs, h0, key), None, length=T
        )
        episode_end_time = time.perf_counter()
        delta_time = episode_end_time - episode_start_time
        mask_t = None
        if mask_on:
            (
                obs_t,
                act_t,
                rew_t,
                deliv_t,
                blocked_t,
                noop_t,
                pickup_t,
                drop_t,
                distance_traveled_t,
                step_time_t,
                raw_r_t,
                mask_t,
            ) = scanned
        else:
            (
                obs_t,
                act_t,
                rew_t,
                deliv_t,
                blocked_t,
                noop_t,
                pickup_t,
                drop_t,
                distance_traveled_t,
                step_time_t,
                raw_r_t,
            ) = scanned

        obs_seq = jnp.concatenate([obs_t, final_obs.reshape(1, B, in_dim)], axis=0)
        obs_b = obs_seq.reshape(T + 1, E, N, in_dim).transpose(1, 0, 2, 3)

        t1, t2 = T // 3, 2 * (T // 3)
        team_per_ep = lambda x: x.sum(axis=0).sum(axis=-1).mean()
        frac = lambda x: x.mean()
        metrics = {
            "episode_return": raw_r_t.sum(0).sum(-1).mean(),
            "deliveries": deliv_t.sum(0).sum(-1).mean(),
            "block_rate": frac(blocked_t),
            "idle_rate": frac(noop_t),
            "pickup_rate": frac(pickup_t),
            "deliveries_early": team_per_ep(deliv_t[:t1]),
            "deliveries_mid": team_per_ep(deliv_t[t1:t2]),
            "deliveries_late": team_per_ep(deliv_t[t2:]),
            "block_early": frac(blocked_t[:t1]),
            "block_mid": frac(blocked_t[t1:t2]),
            "block_late": frac(blocked_t[t2:]),
        }
                    

        metrics_extra = {
            "obs_t": obs_t,
            "act_t": act_t,
            "deliv_t": deliv_t,
            "blocked_t": blocked_t,
            "noop_t": noop_t,
            "pickup_t": pickup_t,
            "drop_t": drop_t,
            "distance_traveled_t": distance_traveled_t,
            "step_time_t": step_time_t,
            "raw_r_t": raw_r_t,
            "mask_t": jnp.nan if mask_t is None else mask_t,
            "fstates": fstates,
            "episode_time": delta_time,
        }
        if mask_on and mask_t is not None:
            final_mask = jax.vmap(env.action_masks)(fstates)  # [E, N, A]
            mask_seq = jnp.concatenate([mask_t, final_mask[None]], axis=0)
            mask_b = mask_seq.transpose(1, 0, 2, 3)  # [E, T+1, N, A]
        else:
            mask_b = None
        return (
            obs_b,
            act_t.transpose(1, 0, 2),
            rew_t.transpose(1, 0),
            mask_b,
            metrics,
            metrics_extra,
        )

    # -- one collection iteration ---------------------------------------------
    def update_step(carry, data):
        (
            params_ens,
            opt_state,
            rew_ms,
            buf_obs,
            buf_act,
            buf_rew,
            buf_mask,
            ptr,
            count,
            key,
        ) = carry
        key, kroll = jax.random.split(key)
        idx, iters = data

        eps = epsilon_at(cfg, idx * E * T)
        obs_b, act_b, rew_b, mask_b, metrics, metrics_extra = rollout(
            params_ens, kroll, eps
        )

        # scatter E episodes into the ring buffer
        pos = (ptr + jnp.arange(E)) % C
        buf_obs = buf_obs.at[pos].set(obs_b)
        buf_act = buf_act.at[pos].set(act_b)
        buf_rew = buf_rew.at[pos].set(rew_b)
        if mask_on:
            buf_mask = buf_mask.at[pos].set(mask_b)
        ptr = (ptr + E) % C
        count = jnp.minimum(count + E, C)

        def train_branch(op):
            params_ens, opt_state, rew_ms, key = op

            def grad_step(c, _):
                params_ens, opt_state, rew_ms, key = c
                key, ks, kb = jax.random.split(key, 3)
                idxs = jax.random.randint(ks, (bs,), 0, count)  # sample warm portion
                mb = buf_mask[idxs] if mask_on else None
                batch = feed_batch(
                    buf_obs[idxs], buf_act[idxs], buf_rew[idxs], T, N, mask_b=mb
                )
                # per-member bootstrap mask over the bs sampled episodes
                batch["bootstrap_mask"] = (
                    jax.random.uniform(kb, (K, bs)) < cfg.bootstrap_mask_prob
                ).astype(jnp.float32)
                params_ens, opt_state, rew_ms, diag = emax_update(
                    qnet, tx, cfg, params_ens, opt_state, rew_ms, batch
                )
                return (params_ens, opt_state, rew_ms, key), (
                    diag["loss"],
                    diag["q_mean"],
                    diag["td"],
                    diag["w"],
                    diag["grads"],
                    diag["mac_ens"],
                )

            (params_ens, opt_state, rew_ms, key), (
                losses,
                q_means,
                td,
                w,
                grads,
                mac_ens,
            ) = jax.lax.scan(grad_step, op, None, length=n_grad)
            flat_grads = extract_flat_grads(grads)
            gradient_norms_metric = gradient_norms(flat_grads)

            return (
                params_ens,
                opt_state,
                rew_ms,
                key,
                losses.mean(),
                losses,
                q_means,
                td,
                w,
                mac_ens,
                gradient_norms_metric,
            )

        def skip_branch(op):
            params_ens, opt_state, rew_ms, key = op
            return (
                params_ens,
                opt_state,
                rew_ms,
                key,
                jnp.array(0.0, jnp.float32),
                jnp.zeros((E,), jnp.float32),
                jnp.zeros((E, T + 1, bs, N, A), jnp.float32),
                jnp.zeros((E, K, T, bs, N), jnp.float32),
                jnp.zeros((E, K, T, bs, N), jnp.float32),
                jnp.zeros((E, K, T + 1, bs, N, A), jnp.float32),
                {
                    "grad_norm_mean": jnp.array(0.0, jnp.float32),
                    "grad_norm_std": jnp.array(0.0, jnp.float32),
                    "grad_norm_p90": jnp.array(0.0, jnp.float32),
                },
            )

        (
            params_ens,
            opt_state,
            rew_ms,
            key,
            loss,
            losses,
            q_means,
            td,
            w,
            mac_ens,
            grads_metric,
        ) = jax.lax.cond(
            count >= bs,
            train_branch,
            skip_branch,
            (params_ens, opt_state, rew_ms, key),
        )  # type: ignore
        (new_mean, new_var, tot_count) = rew_ms
        metrics_tensors = {
            "episode_time": metrics_extra["episode_time"],
            "rewards_tensor_raw": metrics_extra["raw_r_t"],
            "actions_tensor_raw": metrics_extra["act_t"],
            "observation_tensor_raw": metrics_extra["obs_t"],
            "epoch_loss_tensor_raw": losses,
            "epoch_q_value_tensor_raw": mac_ens,
            "epoch_q_value_mean_tensor_raw": q_means,
            "epoch_TD_target_tensor_raw": td,
            "epoch_step_valid_mask_tensor_raw": w,
            "epsilon_schedule": eps,
            "epoch_grad_norm_mean": grads_metric["grad_norm_mean"],
            "epoch_grad_norm_std": grads_metric["grad_norm_std"],
            "epoch_grad_norm_p90": grads_metric["grad_norm_p90"],
            "rewards_std_mean": new_mean,
            "rewards_std_var": new_var,
            "rewards_std_tot_count": tot_count,
            "deliveries_tensor_raw": metrics_extra["deliv_t"],
            "block_tensor_raw": metrics_extra["blocked_t"],
            "idle_tensor_raw": metrics_extra["noop_t"],
            "pickup_tensor_raw": metrics_extra["pickup_t"],
            "distance_traveled_tensor_raw": metrics_extra["distance_traveled_t"],
            "step_time_tensor_raw": metrics_extra["step_time_t"],  # [T,E]
            "step_count_tensor_raw": metrics_extra["fstates"].step_count,  # [E,]
            "mask_tensor_raw": (
                jnp.array(jnp.nan, jnp.float32)
                if metrics_extra["mask_t"] is jnp.nan
                else metrics_extra["mask_t"]
            ),
        }

        carry = (
            params_ens,
            opt_state,
            rew_ms,
            buf_obs,
            buf_act,
            buf_rew,
            buf_mask,
            ptr,
            count,
            key,
        )
        if live_log:
            io_callback(
                _host_log,
                None,
                iters,
                idx,
                metrics["episode_return"],
                eps.mean(),
                metrics["deliveries"],
                metrics["block_rate"],
                metrics["idle_rate"],
                ordered=False,
            )
        metrics = {**metrics, "loss": loss, "epsilon": eps}
        return carry, (metrics_tensors, metrics)

    @functools.partial(jax.jit, static_argnums=(3,))
    def train_from(carry, iters, base_iter, n):
        idxs = base_iter + jnp.arange(n)
        total_updates = jnp.full((n,), iters)
        xs = (idxs, total_updates)
        carry, (metrics_tensors, episode_metrics) = jax.lax.scan(
            update_step, carry, xs, length=n
        )

        return carry, (metrics_tensors, episode_metrics)

    return {
        "env": env,
        "qnet": qnet,
        "init_carry": init_carry,
        "train_from": train_from,
        "save_subtree": save_subtree,
        "merge_subtree": merge_subtree,
    }
