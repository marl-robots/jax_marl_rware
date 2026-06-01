"""End-to-end jitted MAPPO on the JAX RWARE env (PureJaxRL/JaxMARL style).

Replicates marlbase ac/model.py:PPONetwork + ac/train.py semantics exactly:
 - per update: fresh env reset + one full `time_limit`-step episode across
   `parallel_envs`; GRU hidden starts at zeros, no mid-episode reset.
 - shared-parameter actor; centralised shared critic (input = concat of all
   agents' obs) + soft-updated target critic.
 - old log-probs and values are recomputed (not stored) at update start.
 - truncated n-step returns bootstrapped from the *target* critic.
 - 4 PPO epochs over the whole sequence batch (no minibatching), clipped
   surrogate, value_loss = advantage^2, loss = actor + 0.5*value - 1e-3*ent.
 - streaming Welford reward standardisation, stats persisting across updates.
 - soft target-critic update (tau) once per update after the epochs.

Agents are folded into the batch axis (B = parallel_envs * n_agents) so the
shared networks need no per-agent stacking.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import optax

from jaxrware import Warehouse, make_config
from .config import MAPPOConfig
from .networks import ActorRNN, CriticRNN, ScannedGRU
from .returns import compute_nstep_returns


def _welford_standardise(state, reward):
    """Streaming weighted Welford (weight=1) matching utils/wrappers.StandardiseReward.

    `state` = (sumw, wmean, m2, n) with sumw/wmean/m2 shaped [E, N] and n scalar.
    First step (n==1) returns the raw reward; thereafter (r-wmean)/(sqrt(var)+1e-6).
    """
    sumw, wmean, m2, n = state
    q = reward - wmean
    temp_sumw = sumw + 1.0
    r = q / temp_sumw
    wmean = wmean + r
    m2 = m2 + q * r * sumw
    sumw = temp_sumw
    n = n + 1.0
    var = (m2 * n) / (sumw * (n - 1.0) + 1e-12)
    std = (reward - wmean) / (jnp.sqrt(var) + 1e-6)
    out = jnp.where(n <= 1.0, reward, std)
    return (sumw, wmean, m2, n), out


def make_train(cfg: MAPPOConfig, num_updates: int | None = None):
    env_cfg = make_config(cfg.size, cfg.n_agents, cfg.difficulty)
    env = Warehouse(env_cfg)
    N = cfg.n_agents
    E = cfg.parallel_envs
    B = E * N
    obs_dim = env.obs_dim
    central_dim = obs_dim * N
    T = cfg.time_limit
    H = cfg.hidden_dim
    n_updates = cfg.num_updates if num_updates is None else num_updates

    actor = ActorRNN(env.num_actions, H, cfg.orthogonal_gain)
    critic = CriticRNN(H, cfg.orthogonal_gain)
    tx = optax.adam(cfg.lr)  # grad_clip=false in the proven config

    h0 = lambda: ScannedGRU.initialize_carry(B, H)
    zeros_BT = lambda t: jnp.zeros((t, B))

    def _central(obs_TENobs):
        """[T,E,N,obs] -> [T, E*N, N*obs] (each agent gets the env's concat obs)."""
        cat = obs_TENobs.reshape(T, E, N * obs_dim)
        rep = jnp.broadcast_to(cat[:, :, None, :], (T, E, N, N * obs_dim))
        return rep.reshape(T, B, N * obs_dim)

    def train(key):
        key, ka, kc = jax.random.split(key, 3)
        actor_params = actor.init(ka, h0(), (jnp.zeros((1, B, obs_dim)), zeros_BT(1)))
        critic_params = critic.init(kc, h0(), (jnp.zeros((1, B, central_dim)), zeros_BT(1)))
        params = {"actor": actor_params, "critic": critic_params}
        target_critic = critic_params
        opt_state = tx.init(params)
        welford = (jnp.zeros((E, N)), jnp.zeros((E, N)), jnp.zeros((E, N)), jnp.array(0.0))

        def update_step(carry, _):
            params, target_critic, opt_state, welford, key = carry
            key, kreset, krun = jax.random.split(key, 3)

            states, obs = jax.vmap(env.reset)(jax.random.split(kreset, E))  # obs [E,N,obs]

            # ---- rollout: one full episode, hidden starts at zeros ----
            def rollout_step(rc, _):
                states, obs, h_actor, welford, key = rc
                key, ksamp = jax.random.split(key)
                obs_flat = obs.reshape(B, obs_dim)
                h_actor, dist = actor.apply(
                    params["actor"], h_actor, (obs_flat[None], zeros_BT(1))
                )
                actions_flat = jax.random.categorical(ksamp, dist.logits[0])  # [B]
                actions = actions_flat.reshape(E, N)
                nstates, nobs, rewards, done, _ = jax.vmap(env.step)(states, actions)
                welford, rstd = _welford_standardise(welford, rewards)
                return (nstates, nobs, h_actor, welford, key), (obs, actions, rstd, rewards)

            init = (states, obs, h0(), welford, krun)
            (states, *_unused, welford, _), traj = jax.lax.scan(
                rollout_step, init, None, length=T
            )
            obs_t, act_t, rstd_t, rraw_t = traj  # [T,E,N,*]

            obs_flat_T = obs_t.reshape(T, B, obs_dim)
            act_flat_T = act_t.reshape(T, B)
            central_T = _central(obs_t)
            dones0 = jnp.zeros((T, B))  # fresh full episode -> no in-rollout terminations

            # ---- target-critic returns (detached) ----
            _, v_target = critic.apply(target_critic, h0(), (central_T, dones0))
            v_target = v_target.reshape(T, E, N)
            returns = compute_nstep_returns(
                rstd_t, jnp.zeros((T, E, N)), v_target, cfg.n_steps, cfg.gamma
            )
            returns = jax.lax.stop_gradient(returns)

            # ---- snapshot old log-probs from current actor ----
            _, old_dist = actor.apply(params["actor"], h0(), (obs_flat_T, dones0))
            old_logp = old_dist.log_prob(act_flat_T).reshape(T, E, N)
            old_logp = jax.lax.stop_gradient(old_logp)

            def loss_fn(p):
                _, dist = actor.apply(p["actor"], h0(), (obs_flat_T, dones0))
                logp = dist.log_prob(act_flat_T).reshape(T, E, N)
                entropy = dist.entropy().reshape(T, E, N)
                _, values = critic.apply(p["critic"], h0(), (central_T, dones0))
                values = values.reshape(T, E, N)

                advantage = returns - values
                value_loss = (advantage ** 2).sum(-1).mean()

                adv = jax.lax.stop_gradient(advantage)
                ratio = jnp.exp(logp - old_logp)
                surr1 = ratio * adv
                surr2 = jnp.clip(ratio, 1.0 - cfg.ppo_clip, 1.0 + cfg.ppo_clip) * adv
                actor_loss = (
                    -jnp.minimum(surr1, surr2).sum(-1) - cfg.entropy_coef * entropy.sum(-1)
                ).mean()

                loss = actor_loss + cfg.value_loss_coef * value_loss
                return loss, (actor_loss, value_loss, entropy.mean())

            def epoch(carry, _):
                params, opt_state = carry
                (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
                updates, opt_state = tx.update(grads, opt_state)
                params = optax.apply_updates(params, updates)
                return (params, opt_state), (loss, *aux)

            (params, opt_state), epoch_metrics = jax.lax.scan(
                epoch, (params, opt_state), None, length=cfg.num_epochs
            )

            # ---- soft target-critic update (once, after epochs) ----
            tau = cfg.target_update_tau
            target_critic = jax.tree_util.tree_map(
                lambda tp, sp: (1.0 - tau) * tp + tau * sp, target_critic, params["critic"]
            )

            ep_return = rraw_t.sum(axis=0).mean()  # mean over envs & agents of episode return
            metrics = {
                "episode_return": ep_return,
                "loss": epoch_metrics[0][-1],
                "actor_loss": epoch_metrics[1][-1],
                "value_loss": epoch_metrics[2][-1],
                "entropy": epoch_metrics[3][-1],
                "reward_std_mean": rstd_t.mean(),
            }
            carry = (params, target_critic, opt_state, welford, key)
            return carry, metrics

        init_carry = (params, target_critic, opt_state, welford, key)
        carry, metrics = jax.lax.scan(update_step, init_carry, None, length=n_updates)
        return {"params": carry[0], "metrics": metrics}

    return train, env
