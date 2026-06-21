"""MAPPO + a self-supervised potential network (Phi) for dense, FAITHFUL shaping.

This is the on-policy realisation of the "understanding-first" idea we converged
on: a small network Phi learns, by supervised regression on the agent's OWN
rollouts, how *close to a delivery* each observation is, and that closeness is
used as a potential to densify the reward -- without ever touching the env's
reward function.

Mechanism (all in-graph, reusing the proven mappo.py trainer):
  * Phi(obs) -> scalar, a small MLP with a ZERO-initialised output head, so at
    the start Phi == 0 -> bonus == 0 -> this is byte-for-byte plain MAPPO. As Phi
    learns, the shaping ramps in on its own.
  * Per step the rollout adds  bonus = beta * (gamma * Phi(s') - Phi(s))  to the
    (standardised) reward used for the PPO update. This is POTENTIAL-BASED
    shaping (Ng et al.): for ANY Phi it provably preserves the optimal policy, so
    a bad Phi can only fail to help, never change what's optimal -- the run can't
    do meaningfully worse than MAPPO.
  * Phi's regression target is computed in-graph from the rollout's own delivery
    flags: target_t = gamma^(steps until this agent's next delivery), in [0,1]
    (1 at a delivery, decaying back in time, 0 if no future delivery). A few
    gradient steps per update fit Phi to it.
  * The LOGGED deliveries / return use the RAW signal (shaping touches only the
    PPO reward), so the CSV is directly comparable to a plain-MAPPO run.

Everything else (actor/critic, PPO update, Welford reward std, target critic,
metrics) is reused verbatim from mappo.py.
"""

from __future__ import annotations

import functools

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from jax.experimental import io_callback

from jaxrware import Warehouse, make_config
from .config import MAPPOConfig
from .mappo import mappo_update, _host_log
from .networks import ActorRNN, CriticRNN, ScannedGRU


class PhiNet(nn.Module):
    """Closeness-to-delivery potential. Zero-init head => Phi==0 at start."""
    hidden: int = 64

    @nn.compact
    def __call__(self, x):                       # x: [..., obs_dim] -> [...]
        h = nn.tanh(nn.Dense(self.hidden)(x))
        h = nn.tanh(nn.Dense(self.hidden)(h))
        v = nn.Dense(1, kernel_init=nn.initializers.zeros,
                     bias_init=nn.initializers.zeros)(h)
        return v[..., 0]


def _setup_potential(cfg: MAPPOConfig, *, phi_beta: float, phi_epochs: int,
                     phi_lr: float, phi_hidden: int, live_log: bool = False):
    env = Warehouse(make_config(cfg.size, cfg.n_agents, cfg.difficulty))
    N, E = cfg.n_agents, cfg.parallel_envs
    B = E * N
    obs_dim = env.obs_dim
    critic_dim = obs_dim * N if cfg.centralised_critic else obs_dim
    T, H = cfg.time_limit, cfg.hidden_dim

    actor = ActorRNN(env.num_actions, H, cfg.orthogonal_gain, cfg.use_rnn)
    critic = CriticRNN(H, cfg.orthogonal_gain, cfg.use_rnn)
    phi_net = PhiNet(hidden=phi_hidden)
    tx = optax.adam(cfg.lr)
    phi_tx = optax.adam(phi_lr)

    h0 = lambda: ScannedGRU.initialize_carry(B, H)
    zeros_BT = lambda t: jnp.zeros((t, B))

    def _critic_input(obs_TENobs):
        if cfg.centralised_critic:
            cat = obs_TENobs.reshape(T, E, N * obs_dim)
            rep = jnp.broadcast_to(cat[:, :, None, :], (T, E, N, N * obs_dim))
            return rep.reshape(T, B, N * obs_dim)
        return obs_TENobs.reshape(T, B, obs_dim)

    def init_carry(key):
        key, ka, kc, kp = jax.random.split(key, 4)
        params = {
            "actor": actor.init(ka, h0(), (jnp.zeros((1, B, obs_dim)), zeros_BT(1))),
            "critic": critic.init(kc, h0(), (jnp.zeros((1, B, critic_dim)), zeros_BT(1))),
        }
        target_critic = params["critic"]
        opt_state = tx.init(params)
        welford = (jnp.zeros((E, N)), jnp.zeros((E, N)), jnp.zeros((E, N)), jnp.array(0.0))
        phi_params = phi_net.init(kp, jnp.zeros((1, obs_dim)))
        phi_opt_state = phi_tx.init(phi_params)
        return (params, target_critic, opt_state, welford,
                phi_params, phi_opt_state, key)

    from .mappo import _welford_standardise

    def update_step(carry, upd_idx):
        (params, target_critic, opt_state, welford,
         phi_params, phi_opt_state, key) = carry
        key, kreset, krun = jax.random.split(key, 3)
        states, obs = jax.vmap(env.reset)(jax.random.split(kreset, E))

        # ---- rollout: one full episode; add the Phi potential bonus per step ----
        def rollout_step(rc, _):
            states, obs, h_actor, welford, key = rc
            key, ksamp = jax.random.split(key)
            obs_flat = obs.reshape(B, obs_dim)
            h_actor, dist = actor.apply(
                params["actor"], h_actor, (obs_flat[None], zeros_BT(1)))
            actions = jax.random.categorical(ksamp, dist.logits[0]).reshape(E, N)
            nstates, nobs, rewards, done, info = jax.vmap(env.step)(states, actions)
            welford, rstd = _welford_standardise(welford, rewards)

            # potential-based shaping bonus (beta * (gamma*Phi(s') - Phi(s)))
            phi_cur = phi_net.apply(phi_params, obs_flat).reshape(E, N)
            phi_nxt = phi_net.apply(phi_params, nobs.reshape(B, obs_dim)).reshape(E, N)
            bonus = phi_beta * (cfg.gamma * phi_nxt - phi_cur)
            rstd_shaped = rstd + bonus

            sig = (info["deliveries"], info["forward_blocked"],
                   info["noop"], info["pickup"], info["drop"])
            return (nstates, nobs, h_actor, welford, key), (
                obs, actions, rstd_shaped, rewards, bonus, *sig)

        init = (states, obs, h0(), welford, krun)
        (states, *_u, welford, _), traj = jax.lax.scan(
            rollout_step, init, None, length=T)
        (obs_t, act_t, rstd_t, rraw_t, bonus_t,
         deliv_t, blocked_t, noop_t, pickup_t, drop_t) = traj      # [T,E,N,*]

        # ---- PPO update on the SHAPED reward (everything else verbatim) ----
        batch = {
            "obs_flat_T": obs_t.reshape(T, B, obs_dim),
            "act_flat_T": act_t.reshape(T, B),
            "central_T": _critic_input(obs_t),
            "rstd_TEN": rstd_t,
            "dones_TEN": jnp.zeros((T, E, N)),
        }
        params, target_critic, opt_state, diag = mappo_update(
            actor, critic, tx, cfg, params, target_critic, opt_state, batch)

        # ---- Phi target: gamma^(steps to this agent's next delivery), in [0,1] ----
        d = (deliv_t > 0).astype(jnp.float32)                      # [T,E,N]
        def close_step(c_next, d_t):
            c = jnp.where(d_t > 0, 1.0, cfg.gamma * c_next)
            return c, c
        _, closeness = jax.lax.scan(
            close_step, jnp.zeros((E, N)), d, reverse=True)         # [T,E,N]
        phi_target = jax.lax.stop_gradient(closeness.reshape(T, B))
        phi_obs = obs_t.reshape(T, B, obs_dim)

        def phi_loss_fn(pp):
            pred = phi_net.apply(pp, phi_obs)                       # [T,B]
            return ((pred - phi_target) ** 2).mean()

        def phi_epoch(c, _):
            pp, ps = c
            loss, g = jax.value_and_grad(phi_loss_fn)(pp)
            upd, ps = phi_tx.update(g, ps)
            return (optax.apply_updates(pp, upd), ps), loss
        (phi_params, phi_opt_state), phi_losses = jax.lax.scan(
            phi_epoch, (phi_params, phi_opt_state), None, length=phi_epochs)

        # ---- metrics (deliveries/return from RAW signal -> comparable) ----
        ep_return = rraw_t.sum(axis=0).sum(-1).mean()
        t1, t2 = T // 3, 2 * (T // 3)
        team_per_ep = lambda x: x.sum(axis=0).sum(axis=-1).mean()
        frac = lambda x: x.mean()
        metrics = {
            "episode_return": ep_return,
            "loss": diag["epoch_loss"][-1],
            "actor_loss": diag["epoch_actor_loss"][-1],
            "value_loss": diag["epoch_value_loss"][-1],
            "entropy": diag["epoch_entropy"][-1],
            "reward_std_mean": rstd_t.mean(),
            "deliveries": team_per_ep(deliv_t),
            "block_rate": frac(blocked_t),
            "idle_rate": frac(noop_t),
            "pickup_rate": frac(pickup_t),
            "deliveries_early": team_per_ep(deliv_t[:t1]),
            "deliveries_mid": team_per_ep(deliv_t[t1:t2]),
            "deliveries_late": team_per_ep(deliv_t[t2:]),
            "block_early": frac(blocked_t[:t1]),
            "block_mid": frac(blocked_t[t1:t2]),
            "block_late": frac(blocked_t[t2:]),
            # potential-shaping diagnostics
            "phi_loss": phi_losses[-1],
            "phi_target_mean": phi_target.mean(),
            "mean_abs_bonus": jnp.abs(bonus_t).mean(),
        }
        carry = (params, target_critic, opt_state, welford,
                 phi_params, phi_opt_state, key)
        if live_log:
            io_callback(
                _host_log, None, upd_idx, metrics["episode_return"],
                metrics["entropy"], metrics["deliveries"], metrics["block_rate"],
                metrics["idle_rate"], ordered=False)
        return carry, metrics

    return env, actor, critic, phi_net, init_carry, update_step


def make_resumable_train_potential(cfg: MAPPOConfig, *, phi_beta: float = 1.0,
                                   phi_epochs: int = 4, phi_lr: float = 3e-4,
                                   phi_hidden: int = 64, live_log: bool = False):
    """Resumable potential-MAPPO trainer, same chunked API as mappo.make_resumable_train."""
    env, actor, critic, phi_net, init_carry, update_step = _setup_potential(
        cfg, phi_beta=phi_beta, phi_epochs=phi_epochs, phi_lr=phi_lr,
        phi_hidden=phi_hidden, live_log=live_log)

    @functools.partial(jax.jit, static_argnums=(2,))
    def train_from(carry, base_upd, n):
        idxs = base_upd + jnp.arange(n)
        return jax.lax.scan(update_step, carry, idxs, length=n)

    return {
        "env": env, "actor": actor, "critic": critic, "phi_net": phi_net,
        "init_carry": init_carry, "train_from": train_from,
    }
