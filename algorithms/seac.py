"""End-to-end jitted SEAC (Shared Experience Actor-Critic) on JAX RWARE.

Replicates the *canonical* implementation (semitable/seac, Christianos, Schäfer,
Albrecht, NeurIPS 2020) rather than the simplified marlbase stub:

 - PER-AGENT independent actor + critic (NO parameter sharing): each agent owns
   its own param tree (we stack the N trees along a leading axis and vmap).
 - on-policy A2C: one full-episode rollout across `parallel_envs`, then a single
   gradient step (no PPO epochs, no target critic -- the online critic both acts
   and bootstraps).
 - n-step returns, bootstrapped from each agent's OWN critic on its OWN obs.
 - SHARED-EXPERIENCE term: every agent i also trains on every other agent k's
   trajectory, importance-weighted by  IS = pi_i(a_k|o_k) / pi_k(a_k|o_k)
   (detached), with advantage computed under agent i's own critic on k's data.
   On the diagonal (i==k) IS == 1, so the own A2C term and the shared term
   UNIFY into one IS-weighted N x N computation; seac_coef (lambda, =1.0)
   multiplies the off-diagonal (shared) contributions only.

Canonical SEAC loss (per evaluating agent i, summed over data-owner k):
    adv[i,k]   = returns[k] - V_i(o_k)
    IS[i,k]    = exp(logp_i(a_k|o_k) - logp_k(a_k|o_k))          (stop-grad)
    policy[i,k]= -(IS[i,k] * logp_i(a_k|o_k) * stop_grad(adv[i,k]))
    value[i,k] =  IS[i,k] * adv[i,k]**2
    loss = sum_{i,k} coef[i,k] * (mean_te(policy) + value_coef * mean_te(value))
           - entropy_coef * sum_i mean_te(entropy_i(o_i))
    coef[i,k] = 1 if i==k else seac_coef
returns[k] uses stop_grad(V_k(o_k)) for the n-step bootstrap.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import optax
from jax.experimental import io_callback

from jaxrware import Warehouse, make_config
from .config import MAPPOConfig
from .networks import ActorRNN, CriticRNN, ScannedGRU
from .returns import compute_nstep_returns
from .mappo import _welford_standardise, _host_log


def _setup(cfg: MAPPOConfig, live_log: bool = False):
    """Build env + per-agent networks + the per-update step closure.

    Returns (env, actor, critic, tx, init_carry, update_step). The carry is a
    flat pytree (actor_params, critic_params, opt_state, welford, key) where
    actor_params/critic_params have a leading agent axis of size N (independent
    per-agent params; agents are NOT folded into a shared net).
    """
    env_cfg = make_config(cfg.size, cfg.n_agents, cfg.difficulty)
    env = Warehouse(env_cfg)
    N = cfg.n_agents
    E = cfg.parallel_envs
    obs_dim = env.obs_dim          # SEAC critic is independent -> own-obs input
    T = cfg.time_limit
    H = cfg.hidden_dim
    A = env.num_actions

    actor = ActorRNN(A, H, cfg.orthogonal_gain, cfg.use_rnn)
    critic = CriticRNN(H, cfg.orthogonal_gain, cfg.use_rnn)   # output dim 1
    tx = optax.adam(cfg.lr)

    eye = jnp.eye(N)                                   # [Ni, Nk]
    coefmat = eye + (1.0 - eye) * cfg.seac_coef        # 1 on diag, lambda off-diag

    def init_carry(key):
        key, ka, kc = jax.random.split(key, 3)
        dummy_o = jnp.zeros((1, 1, obs_dim))
        dummy_r = jnp.zeros((1, 1))
        h1 = ScannedGRU.initialize_carry(1, H)
        # N independent param trees, stacked along a leading axis.
        actor_params = jax.vmap(
            lambda k: actor.init(k, h1, (dummy_o, dummy_r)))(jax.random.split(ka, N))
        critic_params = jax.vmap(
            lambda k: critic.init(k, h1, (dummy_o, dummy_r)))(jax.random.split(kc, N))
        opt_state = tx.init((actor_params, critic_params))
        welford = (jnp.zeros((E, N)), jnp.zeros((E, N)),
                   jnp.zeros((E, N)), jnp.array(0.0))
        return (actor_params, critic_params, opt_state, welford, key)

    def update_step(carry, upd_idx):
        actor_params, critic_params, opt_state, welford, key = carry
        key, kreset, krun = jax.random.split(key, 3)
        states, obs = jax.vmap(env.reset)(jax.random.split(kreset, E))  # obs [E,N,obs]

        # ---- rollout: full episode, each agent acts with its OWN policy ----
        h_actor0 = jnp.zeros((N, E, H))           # per-agent hidden, per env
        resets_1E = jnp.zeros((1, E))

        def act_i(p_i, h_i, o_i):                 # o_i [E,obs], h_i [E,H]
            h_i, dist = actor.apply(p_i, h_i, (o_i[None], resets_1E))
            return h_i, dist.logits[0]            # logits [E, A]

        def rollout_step(rc, _):
            states, obs, h_actor, welford, key = rc
            key, ksamp = jax.random.split(key)
            obs_NEo = jnp.swapaxes(obs, 0, 1)     # [N, E, obs]
            h_actor, logits = jax.vmap(act_i)(actor_params, h_actor, obs_NEo)  # [N,E,A]
            actions_NE = jax.random.categorical(ksamp, logits)                 # [N, E]
            actions_EN = actions_NE.T                                          # [E, N]
            nstates, nobs, rewards, done, info = jax.vmap(env.step)(states, actions_EN)
            welford, rstd = _welford_standardise(welford, rewards)             # [E,N]
            rstd = jnp.where(cfg.standardise_rewards, rstd, rewards)
            sig = (info["deliveries"], info["forward_blocked"], info["noop"])
            out = (obs, actions_EN, rstd, rewards, *sig)                       # each [E,N,*]
            return (nstates, nobs, h_actor, welford, key), out

        init = (states, obs, h_actor0, welford, krun)
        (states, *_u, welford, _), traj = jax.lax.scan(
            rollout_step, init, None, length=T)
        obs_t, act_t, rstd_t, rraw_t, deliv_t, blocked_t, noop_t = traj  # [T,E,N,*]

        # ---- SEAC loss: N x N cross-evaluation ----
        obs_eval = obs_t.reshape(T, E * N, obs_dim)        # all agents' obs, folded
        act_eval = act_t.reshape(T, E * N)                 # all agents' actions
        resets_T = jnp.zeros((T, E * N))
        h0_eval = ScannedGRU.initialize_carry(E * N, H)
        dones_TEN = jnp.zeros((T, E, N))                   # fresh full episode

        def eval_i(ap_i, cp_i):
            """Agent i's policy/critic on EVERY agent's trajectory."""
            _, dist = actor.apply(ap_i, h0_eval, (obs_eval, resets_T))
            logp = dist.log_prob(act_eval).reshape(T, E, N)     # [T,E,Nk]
            ent = dist.entropy().reshape(T, E, N)
            _, v = critic.apply(cp_i, h0_eval, (obs_eval, resets_T))
            return logp, ent, v.reshape(T, E, N)

        def loss_fn(params):
            ap, cp = params
            logp, ent, v = jax.vmap(eval_i)(ap, cp)        # each [Ni,T,E,Nk]

            # diagonals: behaving agent's own logp/value on its own data
            logp_diag = jnp.diagonal(logp, axis1=0, axis2=3)   # [T,E,N]
            v_diag = jnp.diagonal(v, axis1=0, axis2=3)         # [T,E,N]
            returns = compute_nstep_returns(
                rstd_t, dones_TEN, jax.lax.stop_gradient(v_diag),
                cfg.n_steps, cfg.gamma)                        # [T,E,N]  (per owner k)
            returns = jax.lax.stop_gradient(returns)

            adv = returns[None] - v                            # [Ni,T,E,Nk]
            IS = jax.lax.stop_gradient(jnp.exp(logp - logp_diag[None]))
            if cfg.is_clip > 0.0:   # V-trace-style truncation of the IS weight
                IS = jnp.minimum(IS, cfg.is_clip)
            policy_full = -(IS * logp * jax.lax.stop_gradient(adv))
            value_full = IS * adv ** 2

            mt = lambda x: x.mean(axis=(1, 2))                 # mean over T,E -> [Ni,Nk]
            policy_loss = (coefmat * mt(policy_full)).sum()
            value_loss = (coefmat * mt(value_full)).sum()
            ent_diag = jnp.diagonal(ent, axis1=0, axis2=3)     # [T,E,N]
            ent_term = ent_diag.mean(axis=(0, 1)).sum()        # sum over agents
            loss = (policy_loss
                    - cfg.entropy_coef * ent_term
                    + cfg.value_loss_coef * value_loss)
            # off-diagonal IS diagnostics (i != k): is the shared weight exploding?
            offb = jnp.broadcast_to((1.0 - eye)[:, None, None, :] > 0, IS.shape)
            is_off = jnp.where(offb, IS, 0.0)
            is_mean = is_off.sum() / offb.sum()
            is_max = jnp.max(is_off)
            return loss, (policy_loss, value_loss, ent_term / N, is_mean, is_max)

        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            (actor_params, critic_params))
        updates, opt_state = tx.update(grads, opt_state)
        actor_params, critic_params = optax.apply_updates(
            (actor_params, critic_params), updates)

        # ---- metrics (team return: sum over agents + time, mean over envs) ----
        ep_return = rraw_t.sum(axis=0).sum(-1).mean()
        metrics = {
            "episode_return": ep_return,
            "loss": loss,
            "actor_loss": aux[0],
            "value_loss": aux[1],
            "entropy": aux[2],
            "reward_std_mean": rstd_t.mean(),
            "deliveries": deliv_t.sum(0).sum(-1).mean(),
            "block_rate": blocked_t.mean(),
            "idle_rate": noop_t.mean(),
            "is_mean": aux[3],
            "is_max": aux[4],
        }
        carry = (actor_params, critic_params, opt_state, welford, key)
        if live_log:
            io_callback(
                _host_log, None, upd_idx, metrics["episode_return"],
                metrics["entropy"], metrics["deliveries"], metrics["block_rate"],
                metrics["idle_rate"], ordered=False)
            io_callback(
                lambda u, m, x: print(
                    f"           IS off-diag: mean={float(m):7.2f}  max={float(x):10.1f}",
                    flush=True),
                None, upd_idx, metrics["is_mean"], metrics["is_max"], ordered=False)
        return carry, metrics

    return env, actor, critic, tx, init_carry, update_step


def make_train(cfg: MAPPOConfig, num_updates: int | None = None, live_log: bool = False):
    env, actor, critic, tx, init_carry, update_step = _setup(cfg, live_log)
    n_updates = cfg.num_updates if num_updates is None else num_updates

    def train(key):
        carry = init_carry(key)
        carry, metrics = jax.lax.scan(
            update_step, carry, jnp.arange(n_updates), length=n_updates)
        return {"params": carry[0], "metrics": metrics, "carry": carry}

    return train, env


def make_resumable_train(cfg: MAPPOConfig, live_log: bool = False):
    """Resumable SEAC trainer (same chunked interface as mappo.make_resumable_train)."""
    env, actor, critic, tx, init_carry, update_step = _setup(cfg, live_log)

    @functools.partial(jax.jit, static_argnums=(2,))
    def train_from(carry, base_upd, n):
        idxs = base_upd + jnp.arange(n)
        carry, metrics = jax.lax.scan(update_step, carry, idxs, length=n)
        return carry, metrics

    return {
        "env": env,
        "actor": actor,
        "critic": critic,
        "init_carry": init_carry,
        "train_from": train_from,
    }
