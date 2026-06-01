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


def ppo_losses(returns, values, logp, old_logp, entropy, *,
               ppo_clip, entropy_coef, value_loss_coef, filled=None):
    """Pure PPO loss math, shaped [..., N] over the agent axis.

    Transcribes marlbase ac/model.py:PPONetwork.update exactly:
      value_loss = (advantage**2).sum(-1)              # sum over agents
      actor_loss = -min(surr1, surr2).sum(-1) - entropy_coef * entropy.sum(-1)
      loss       = actor_loss + value_loss_coef * value_loss
    each reduced as (x * filled).sum() / filled.sum() when `filled` given,
    else a plain mean over the leading (time/env) axes. `returns` and
    `old_logp` are treated as constants (caller detaches them).
    """
    advantage = returns - values
    value_loss_t = (advantage ** 2).sum(-1)

    adv = jax.lax.stop_gradient(advantage)
    ratio = jnp.exp(logp - old_logp)
    surr1 = ratio * adv
    surr2 = jnp.clip(ratio, 1.0 - ppo_clip, 1.0 + ppo_clip) * adv
    actor_loss_t = -jnp.minimum(surr1, surr2).sum(-1) - entropy_coef * entropy.sum(-1)

    if filled is None:
        value_loss = value_loss_t.mean()
        actor_loss = actor_loss_t.mean()
    else:
        denom = filled.sum()
        value_loss = (value_loss_t * filled).sum() / denom
        actor_loss = (actor_loss_t * filled).sum() / denom

    loss = actor_loss + value_loss_coef * value_loss
    return loss, (actor_loss, value_loss, entropy.mean())


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


def mappo_update(actor, critic, tx, cfg, params, target_critic, opt_state, batch):
    """One MAPPO update over a fixed batch — the exact path the trainer uses.

    `batch` is a dict of jnp arrays:
      obs_flat_T  [T, B, obs_dim]       (B = E*N, agents folded into batch)
      act_flat_T  [T, B]
      central_T   [T, B, central_dim]
      rstd_TEN    [T, E, N]             standardised rewards
      dones_TEN   [T, E, N]             terminations (returns masking only)
    GRU hidden resets are always zeros within an update — marlbase's
    RNNNetwork does NOT reset hidden state mid-sequence; `dones` only mask
    the n-step returns. Returns updated (params, target_critic, opt_state)
    and a diagnostics dict (returns, old_logp, per-epoch losses) for parity.
    """
    obs_flat_T = batch["obs_flat_T"]
    act_flat_T = batch["act_flat_T"]
    central_T = batch["central_T"]
    rstd_TEN = batch["rstd_TEN"]
    dones_TEN = batch["dones_TEN"]
    T, E, N = rstd_TEN.shape
    B = E * N
    H = cfg.hidden_dim
    h0 = lambda: ScannedGRU.initialize_carry(B, H)
    gru_resets = jnp.zeros((T, B))  # no mid-sequence hidden reset (matches marlbase)

    # ---- target-critic returns (detached) ----
    _, v_target = critic.apply(target_critic, h0(), (central_T, gru_resets))
    v_target = v_target.reshape(T, E, N)
    returns = compute_nstep_returns(rstd_TEN, dones_TEN, v_target, cfg.n_steps, cfg.gamma)
    returns = jax.lax.stop_gradient(returns)

    # ---- snapshot old log-probs from current actor ----
    _, old_dist = actor.apply(params["actor"], h0(), (obs_flat_T, gru_resets))
    old_logp = old_dist.log_prob(act_flat_T).reshape(T, E, N)
    old_logp = jax.lax.stop_gradient(old_logp)

    def loss_fn(p):
        _, dist = actor.apply(p["actor"], h0(), (obs_flat_T, gru_resets))
        logp = dist.log_prob(act_flat_T).reshape(T, E, N)
        entropy = dist.entropy().reshape(T, E, N)
        _, values = critic.apply(p["critic"], h0(), (central_T, gru_resets))
        values = values.reshape(T, E, N)
        return ppo_losses(
            returns, values, logp, old_logp, entropy,
            ppo_clip=cfg.ppo_clip, entropy_coef=cfg.entropy_coef,
            value_loss_coef=cfg.value_loss_coef,
        )

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

    diagnostics = {
        "returns": returns,
        "old_logp": old_logp,
        "epoch_loss": epoch_metrics[0],
        "epoch_actor_loss": epoch_metrics[1],
        "epoch_value_loss": epoch_metrics[2],
        "epoch_entropy": epoch_metrics[3],
    }
    return params, target_critic, opt_state, diagnostics


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

            batch = {
                "obs_flat_T": obs_t.reshape(T, B, obs_dim),
                "act_flat_T": act_t.reshape(T, B),
                "central_T": _central(obs_t),
                "rstd_TEN": rstd_t,
                "dones_TEN": jnp.zeros((T, E, N)),  # fresh full episode -> no terminations
            }
            params, target_critic, opt_state, diag = mappo_update(
                actor, critic, tx, cfg, params, target_critic, opt_state, batch
            )

            # team episode return (sum over agents, mean over envs) to match
            # marlbase's logged `mean_episode_returns`.
            ep_return = rraw_t.sum(axis=0).sum(-1).mean()
            metrics = {
                "episode_return": ep_return,
                "loss": diag["epoch_loss"][-1],
                "actor_loss": diag["epoch_actor_loss"][-1],
                "value_loss": diag["epoch_value_loss"][-1],
                "entropy": diag["epoch_entropy"][-1],
                "reward_std_mean": rstd_t.mean(),
            }
            carry = (params, target_critic, opt_state, welford, key)
            return carry, metrics

        init_carry = (params, target_critic, opt_state, welford, key)
        carry, metrics = jax.lax.scan(update_step, init_carry, None, length=n_updates)
        return {"params": carry[0], "metrics": metrics}

    return train, env
