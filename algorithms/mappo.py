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
import time

import jax
import jax.numpy as jnp
import optax
from jax.experimental import io_callback

from jaxrware import make_config, Warehouse

from .config import MAPPOConfig
from .networks import ActorRNN, CriticRNN, ScannedGRU
from .returns import compute_nstep_returns


def ppo_losses(
    returns,
    values,
    logp,
    old_logp,
    entropy,
    *,
    ppo_clip,
    entropy_coef,
    value_loss_coef,
    filled=None,
):
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
    value_loss_t = (advantage**2).sum(-1)

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
    return loss, (
        actor_loss,
        value_loss,
        entropy.mean(),
        actor_loss_t,
        value_loss_t,
        entropy,
        advantage,
        values,
        logp,
        returns,
    )


def a2c_losses(
    returns, values, logp, entropy, *, entropy_coef, value_loss_coef, filled=None
):
    """Pure A2C loss math, shaped [..., N] over the agent axis.

    Transcribes marlbase ac/model.py:A2CNetwork.update exactly:
      advantage  = returns - values
      value_loss = (advantage**2).sum(-1)                       # sum over agents
      actor_loss = -(logp * advantage.detach()).sum(-1) - entropy_coef*entropy.sum(-1)
      loss       = actor_loss + value_loss_coef * value_loss
    Reduced as (x * filled).sum() / filled.sum() when `filled` given, else a
    plain mean. `returns` is treated as a constant (caller detaches it). Unlike
    PPO there is no ratio/clip and no old_logp snapshot — a single grad step.
    """
    advantage = returns - values
    value_loss_t = (advantage**2).sum(-1)
    adv = jax.lax.stop_gradient(advantage)
    actor_loss_t = -(logp * adv).sum(-1) - entropy_coef * entropy.sum(-1)

    if filled is None:
        value_loss = value_loss_t.mean()
        actor_loss = actor_loss_t.mean()
    else:
        denom = filled.sum()
        value_loss = (value_loss_t * filled).sum() / denom
        actor_loss = (actor_loss_t * filled).sum() / denom
    loss = actor_loss + value_loss_coef * value_loss

    return loss, (
        actor_loss,
        value_loss,
        entropy.mean(),
        actor_loss_t,
        value_loss_t,
        entropy,
        advantage,
        values,
        logp,
        returns,
    )


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
    """One AC-family update over a fixed batch — the exact path the trainer uses.

    Handles all four algorithms via `cfg.use_ppo`:
      * PPO (use_ppo=True):  snapshot old log-probs, `cfg.num_epochs` epochs of
        the clipped surrogate (`ppo_losses`).
      * A2C (use_ppo=False): a single grad step of the policy-gradient loss
        (`a2c_losses`); no old-logp snapshot, no clip.
    The critic axis (independent vs centralised) is handled upstream by what the
    caller packs into `batch["central_T"]` (own obs vs concat of all agents').

    `batch` is a dict of jnp arrays:
      obs_flat_T  [T, B, obs_dim]       (B = E*N, agents folded into batch)
      act_flat_T  [T, B]
      central_T   [T, B, critic_dim]    critic_dim = obs_dim*N (cent) or obs_dim (ind)
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
    returns = compute_nstep_returns(
        rstd_TEN, dones_TEN, v_target, cfg.n_steps, cfg.gamma
    )
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
        if cfg.use_ppo:
            return ppo_losses(
                returns,
                values,
                logp,
                old_logp,
                entropy,
                ppo_clip=cfg.ppo_clip,
                entropy_coef=cfg.entropy_coef,
                value_loss_coef=cfg.value_loss_coef,
            )
        return a2c_losses(
            returns,
            values,
            logp,
            entropy,
            entropy_coef=cfg.entropy_coef,
            value_loss_coef=cfg.value_loss_coef,
        )

    def epoch(carry, _):
        params, opt_state = carry
        (loss, aux_tensors), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = tx.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return (params, opt_state), (loss, *aux_tensors, grads)

    # PPO: num_epochs passes over the batch; A2C: a single grad step.

    n_epochs = cfg.num_epochs if cfg.use_ppo else 1
    (params, opt_state), epoch_tensors = jax.lax.scan(
        epoch, (params, opt_state), None, length=n_epochs
    )

    # ---- soft target-critic update (once, after epochs) ----

    tau = cfg.target_update_tau
    target_critic = jax.tree_util.tree_map(
        lambda tp, sp: (1.0 - tau) * tp + tau * sp, target_critic, params["critic"]
    )

    diagnostics = {
        ##"returns": returns,
        "old_logp": old_logp,
        "epoch_loss": epoch_tensors[0],
        "epoch_actor_loss": epoch_tensors[1],
        "epoch_value_loss": epoch_tensors[2],
        "epoch_entropy": epoch_tensors[3],
        "epoch_loss_tensor": epoch_tensors[0],
        "epoch_actor_loss_tensor": epoch_tensors[4],
        "epoch_value_loss_tensor": epoch_tensors[5],
        "epoch_entropy_tensor": epoch_tensors[6],
        "epoch_advantage_tensor": epoch_tensors[7],
        "epoch_values_tensor": epoch_tensors[8],
        "epoch_logp_tensor": epoch_tensors[9],
        "epoch_returns_tensor": epoch_tensors[10],
        "epoch_grads_tensor": epoch_tensors[11],
    }
    return params, target_critic, opt_state, diagnostics


def _host_log(n_updates, upd, ret, ent, deliveries, block_rate, idle_rate):
    """Host-side live print, fired once per update via io_callback (cheap).

    Shows the behavioral view used for watching progress: team return,
    deliveries/episode, forward-block (contention) %, idle %, and entropy
    (watch entropy — a fast drop toward 0 means exploration collapse)."""
    print(
        f"  upd {int(upd):4d}/{int(n_updates):4d} | return {float(ret):8.3f} | "
        f"deliv {float(deliveries):6.2f} | blocked {float(block_rate) * 100:4.1f}% | "
        f"idle {float(idle_rate) * 100:4.1f}% | ent {float(ent):5.3f}",
        flush=True,
    )


def _setup(cfg: MAPPOConfig, live_log: bool = False):
    """Build env + networks + the per-update step closure, shared by the
    one-shot `make_train` and the resumable `make_resumable_train`.

    Returns (env, actor, critic, tx, init_carry, update_step) where:
      * init_carry(key) -> the training carry, a flat pytree
            (params, target_critic, opt_state, welford, key)
        so it serialises cleanly for checkpointing.
      * update_step(carry, upd_idx) -> (carry, metrics)  [scan-compatible]

    `live_log` fires a cheap per-update stdout print via an unordered
    io_callback. Checkpointing + CSV logging are NOT done here: the resumable
    driver runs `train_from` in chunks and writes them from the main thread
    between chunks (orbax wants to be driven from the main thread).
    """
    env_cfg = make_config(cfg.size, cfg.n_agents, cfg.difficulty)
    env = Warehouse(env_cfg)
    N = cfg.n_agents
    E = cfg.parallel_envs
    B = E * N
    obs_dim = env.obs_dim
    # centralised critic sees the concat of all agents' obs; independent sees own.

    critic_dim = obs_dim * N if cfg.centralised_critic else obs_dim
    T = cfg.time_limit
    H = cfg.hidden_dim
    A = env.num_actions

    actor = ActorRNN(A, H, cfg.orthogonal_gain, cfg.use_rnn)
    critic = CriticRNN(H, cfg.orthogonal_gain, cfg.use_rnn)
    tx = optax.adam(cfg.lr)  # grad_clip=false in the proven config

    h0 = lambda: ScannedGRU.initialize_carry(B, H)
    zeros_BT = lambda t: jnp.zeros((t, B))

    def _critic_input(obs_TENobs):
        """Pack the critic input from per-agent obs [T,E,N,obs] -> [T,B,critic_dim].

        Centralised: each agent gets the env's concat of all agents' obs
        ([T,B,N*obs]). Independent: each agent gets its own obs ([T,B,obs])."""
        if cfg.centralised_critic:
            cat = obs_TENobs.reshape(T, E, N * obs_dim)
            rep = jnp.broadcast_to(cat[:, :, None, :], (T, E, N, N * obs_dim))
            return rep.reshape(T, B, N * obs_dim)
        return obs_TENobs.reshape(T, B, obs_dim)

    def init_carry(key):
        key, ka, kc = jax.random.split(key, 3)
        actor_params = actor.init(ka, h0(), (jnp.zeros((1, B, obs_dim)), zeros_BT(1)))
        critic_params = critic.init(
            kc, h0(), (jnp.zeros((1, B, critic_dim)), zeros_BT(1))
        )
        params = {"actor": actor_params, "critic": critic_params}
        target_critic = critic_params
        opt_state = tx.init(params)
        welford = (
            jnp.zeros((E, N)),
            jnp.zeros((E, N)),
            jnp.zeros((E, N)),
            jnp.array(0.0),
        )
        return (params, target_critic, opt_state, welford, key)

    def update_step(carry, data):
        params, target_critic, opt_state, welford, key = carry
        key, kreset, krun = jax.random.split(key, 3)
        upd_idx, n_updates = data
        states, obs = jax.vmap(env.reset)(jax.random.split(kreset, E))  # obs [E,N,obs]

        # ---- rollout: one full episode, hidden starts at zeros ----

        episode_start_time = time.perf_counter()

        def rollout_step(rc, _):
            states, obs, h_actor, welford, key = rc
            key, ksamp = jax.random.split(key)
            obs_flat = obs.reshape(B, obs_dim)
            h_actor, dist = actor.apply(
                params["actor"], h_actor, (obs_flat[None], zeros_BT(1))
            )
            actions_flat = jax.random.categorical(
                ksamp, dist.logits[0]  # pyright: ignore[reportAttributeAccessIssue]
            )  # [B]
            actions = actions_flat.reshape(E, N)
            nstates, nobs, rewards, done, info = jax.vmap(env.step)(states, actions)
            welford, rstd = _welford_standardise(welford, rewards)  # [E,N]
            ##rstd = jnp.where(cfg.standardise_rewards, rstd, rewards)

            sig = (
                info["deliveries"],
                info["forward_blocked"],
                info["noop"],
                info["pickup"],
                info["drop"],
                info["distance_traveled"],
                info["step_time"],
            )  # each [E,N]
            return (nstates, nobs, h_actor, welford, key), (
                obs,
                actions,
                rstd,
                rewards,
                *sig,
            )

        init = (states, obs, h0(), welford, krun)
        (states, *_unused, welford, _), traj = jax.lax.scan(
            rollout_step, init, None, length=T
        )

        (
            obs_t,
            act_t,
            rstd_t,
            rraw_t,
            deliv_t,
            blocked_t,
            noop_t,
            pickup_t,
            drop_t,
            distance_traveled_t,
            step_time_t,
        ) = traj  # [T,E,N,*]

        batch = {
            "obs_flat_T": obs_t.reshape(T, B, obs_dim),
            "act_flat_T": act_t.reshape(T, B),
            "central_T": _critic_input(obs_t),
            "rstd_TEN": rstd_t,
            "dones_TEN": jnp.zeros((T, E, N)),  # fresh full episode -> no terminations
        }
        params, target_critic, opt_state, diag = mappo_update(
            actor, critic, tx, cfg, params, target_critic, opt_state, batch
        )

        episode_end_time = time.perf_counter()
        delta_time = episode_end_time - episode_start_time

        metrics_tensors = {
            "episode_time": delta_time,
            "rewards_tensor_raw": rraw_t,
            "actions_tensor_raw": act_t,
            "observation_tensor_raw": obs_t,
            "epoch_loss_tensor_raw": diag["epoch_loss_tensor"],
            "epoch_actor_loss_tensor_raw": diag["epoch_actor_loss_tensor"],
            "epoch_value_loss_tensor_raw": diag["epoch_value_loss_tensor"],
            "epoch_advantage_tensor_raw": diag["epoch_advantage_tensor"],
            "epoch_values_tensor_raw": diag["epoch_values_tensor"],
            "epoch_logp_tensor_raw": diag["epoch_logp_tensor"],
            "epoch_old_logp_tensor_raw": diag["old_logp"],
            "epoch_returns_tensor_raw": diag["epoch_returns_tensor"],
            "epoch_entropy_tensor_raw": diag["epoch_entropy_tensor"],
            "epoch_grads_tensor_raw": diag["epoch_grads_tensor"],
            "rewards_std_tensor_raw": rstd_t,
            "deliveries_tensor_raw": deliv_t,
            "block_tensor_raw": blocked_t,
            "idle_tensor_raw": noop_t,
            "pickup_tensor_raw": pickup_t,
            "distance_traveled_tensor_raw": distance_traveled_t,
            "step_time_tensor_raw": step_time_t,  # [T,E]
            "step_count_tensor_raw": states.step_count,  # [E,]
        }

        # team episode return (sum over agents, mean over envs) to match
        # marlbase's logged `mean_episode_returns`.
        ep_return = rraw_t.sum(axis=0).sum(-1).mean()

        # ---- behavioral aggregates (all in-graph; only per-update scalars leave
        # the scan). team_per_ep sums over time + agents, means over envs; `frac`
        # is the fraction of agent-steps. Episode is split into thirds to show how
        # behavior shifts within the 500-step episode.
        t1, t2 = T // 3, 2 * (T // 3)
        team_per_ep = lambda x: x.sum(axis=0).sum(axis=-1).mean()
        frac = lambda x: x.mean()
        episode_metrics = {
            "episode_return": ep_return,
            "loss": diag["epoch_loss"][-1],
            "actor_loss": diag["epoch_actor_loss"][-1],
            "value_loss": diag["epoch_value_loss"][-1],
            "entropy": diag["epoch_entropy"][-1],
            "reward_std_mean": rstd_t.mean(),
            # behavioral signals
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
        }
        carry = (params, target_critic, opt_state, welford, key)
        if live_log:
            io_callback(
                _host_log,
                None,
                n_updates,
                upd_idx,
                episode_metrics["episode_return"],
                episode_metrics["entropy"],
                episode_metrics["deliveries"],
                episode_metrics["block_rate"],
                episode_metrics["idle_rate"],
                ordered=False,
            )
        return carry, (metrics_tensors, episode_metrics)

    return env, actor, critic, tx, init_carry, update_step


def make_train(
    cfg: MAPPOConfig, num_updates: int | None = None, live_log: bool = False
):
    """One-shot trainer: the whole run compiled as a single scan (no resume)."""
    env, actor, critic, tx, init_carry, update_step = _setup(cfg, live_log)
    n_updates = cfg.num_updates if num_updates is None else num_updates

    def train(key):
        carry = init_carry(key)
        carry, metrics = jax.lax.scan(
            update_step, carry, jnp.arange(n_updates), length=n_updates
        )
        return {"params": carry[0], "metrics": metrics, "carry": carry}

    return train, env


def make_resumable_train(cfg: MAPPOConfig, live_log: bool = False):
    """Resumable trainer driven in chunks by a host loop.

    Each `train_from` call is a fully-fused `lax.scan` over `n` updates (full
    speed within the chunk). The host loop runs it chunk-by-chunk and, between
    chunks (main thread), does the orbax checkpoint save (keep-best +
    keep-last-N) and CSV logging. Killing the process and restarting with the
    last checkpoint loses at most one chunk of updates. Returns a dict:
      env, actor, critic        -- reused for evaluation / rendering
      init_carry(key) -> carry
      train_from(carry, base_upd, n) -> (carry, metrics_stacked)
          base_upd = absolute index of the first update (so the live log
          reports true global update numbers across a resume); n is static
          (the scan length -> recompiles only when the chunk size changes, so
          pick a chunk size that divides the run to compile exactly once).
    """
    env, actor, critic, tx, init_carry, update_step = _setup(cfg, live_log)

    @functools.partial(jax.jit, static_argnums=(3,))
    def train_from(carry, n_updates, base_upd, n):
        idxs = base_upd + jnp.arange(n)
        total_updates = jnp.full((n,), n_updates)
        xs = (idxs, total_updates)
        carry, (metrics_tensors, episode_metrics) = jax.lax.scan(
            update_step, carry, xs, length=n
        )

        return carry, (metrics_tensors, episode_metrics)

    return {
        "env": env,
        "actor": actor,
        "critic": critic,
        "init_carry": init_carry,
        "train_from": train_from,
    }
