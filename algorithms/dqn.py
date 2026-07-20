"""Off-policy value-based learner, transcribing epymarl QLearner.train exactly.

This module implements the BASE algorithms EMAX extends. Layer 1 (this commit)
is IQL: independent per-agent Q-learning, no mixer. VDN/QMIX mixers and the EMAX
ensemble layer build on top of the same `q_update` skeleton.

Faithful to uoe-agents/epymarl src/learners/q_learner.py:QLearner.train:
  * mac_out over the full episode; chosen-action Q via gather on mac_out[:-1].
  * target_mac_out over the full episode, sliced [1:] for the next-state value.
  * double-Q (default): next action = argmax of the *online* Q at t+1; its value
    is read from the *target* net.
  * 1-step TD(0) target  r + gamma*(1 - terminated)*target_max  (detached).
  * mask = filled[:-1], with mask[1:] *= (1 - terminated[:-2]); loss is the
    mask-normalised mean of squared TD error: (masked_td**2).sum()/mask.sum().
  * reward standardisation via RunningMeanStd (epymarl standarize_stream.py).
  * Adam(lr) with grads clipped to global L2 norm `grad_norm_clip` BEFORE the
    step. Hard target copy every `target_update_interval` episodes is done by
    the trainer, not here.

Arrays are time-major (axis 0 = time T, the episode length+1 of stored steps).
Agents are folded into the batch axis B = E*N (E = episodes in the minibatch,
N = agents) so the shared QNetwork needs no per-agent stacking. avail_actions
default to all-ones (true for unmasked RWARE), so the -inf availability masking
in epymarl is a no-op. When the env supplies an action mask (cfg.use_action_mask,
Warehouse.action_masks), `emax_update` applies it to the bootstrap-max target
via batch["action_mask"]; the parity path passes no mask and is unchanged.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import optax

from jaxrware import make_config, Warehouse

from .dqn_networks import QNetwork
from .networks import ScannedGRU

_NEG_INF = -1e8  # fill for masked-out action Q-values (matches marlbase -1e8)


# ----------------------------------------------------------------------------
# RunningMeanStd  (epymarl components/standarize_stream.py, transcribed)
# ----------------------------------------------------------------------------
def rms_init(shape, epsilon: float = 1e-4):
    """State = (mean, var, count); mean=0, var=1, count=epsilon (=1e-4)."""
    return (jnp.zeros(shape), jnp.ones(shape), jnp.asarray(epsilon, jnp.float32))


def rms_update(state, arr):
    """Parallel-variance merge. `arr` is reshaped to (-1, shape[-1]); batch var
    uses the UNBIASED estimator (torch.var default, ddof=1)."""
    mean, var, count = state
    arr = arr.reshape(-1, arr.shape[-1])
    batch_mean = arr.mean(axis=0)
    batch_var = arr.var(axis=0, ddof=1)
    batch_count = arr.shape[0]

    delta = batch_mean - mean
    tot_count = count + batch_count
    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    m_2 = m_a + m_b + jnp.square(delta) * count * batch_count / tot_count
    new_var = m_2 / tot_count
    return (new_mean, new_var, tot_count)


def make_optimizer(cfg):
    """Adam(lr) with global-norm grad clipping applied first (epymarl order)."""
    return optax.chain(
        optax.clip_by_global_norm(cfg.grad_norm_clip),
        optax.adam(cfg.lr, eps=cfg.adam_eps),
    )


def iql_update_test(
    qnet: QNetwork, tx, cfg, params, target_params, opt_state, rew_ms, batch
):
    """One IQL gradient step over a minibatch of episodes (epymarl mixer=None).

    `batch` (time-major; T = stored steps, E = episodes, N = agents, A = actions):
      obs_T        [T, B, in_dim]   B = E*N, agent-id-augmented obs
      actions_T    [T, E, N]        int32 actions taken
      reward_T     [T, E]           common (shared) reward at each step
      terminated_T [T, E]           termination flag (excludes time-limit)
      filled_T     [T, E]           1 for real stored steps, 0 for padding
    Returns (params, opt_state, rew_ms, diagnostics). The target net is held
    fixed; the trainer performs the periodic hard copy.
    """
    obs_T = batch["obs_T"]
    actions_T = batch["actions_T"]
    reward_T = batch["reward_T"]
    terminated_T = batch["terminated_T"]
    filled_T = batch["filled_T"]
    T, E, N = actions_T.shape
    B = E * N
    H = cfg.hidden_dim
    h0 = lambda: ScannedGRU.initialize_carry(B, H)
    resets = jnp.zeros((T, B))  # one episode per batch element -> no mid-seq reset

    def q_all(p):
        _, q = qnet.apply(p, h0(), (obs_T, resets))  # [T, B, A]
        return q.reshape(T, E, N, qnet.num_actions)

    # ---- reward standardisation (update stats, then normalise) ----
    r = reward_T[: T - 1]  # [T-1, E]
    if cfg.standardise_rewards:
        rew_ms = rms_update(rew_ms, r[..., None])  # shape (1,)
        mean, var, _ = rew_ms
        r = (r - mean[0]) / jnp.sqrt(var[0])
    r_EN = jnp.broadcast_to(r[..., None], (T - 1, E, N))  # expand to agents

    # ---- target_max via double-Q (target-net values at online-net argmax) ----
    mac_out = q_all(params)  # [T, E, N, A]
    target_mac_out = q_all(target_params)
    target_next = target_mac_out[1:]  # [T-1, E, N, A]
    if cfg.double_q:
        cur_max_actions = jnp.argmax(
            jax.lax.stop_gradient(mac_out)[1:], axis=-1
        )  # [T-1, E, N]
        target_max = jnp.take_along_axis(
            target_next, cur_max_actions[..., None], axis=-1
        )[..., 0]
    else:
        target_max = target_next.max(axis=-1)
    target_max = jax.lax.stop_gradient(target_max)

    terminated_used = terminated_T[: T - 1]  # [T-1, E]
    targets = r_EN + cfg.gamma * (1.0 - terminated_used)[..., None] * target_max

    # ---- mask = filled[:-1]; mask[1:] *= (1 - terminated[:-2]) ----
    mask = filled_T[: T - 1]
    factor = jnp.ones_like(mask).at[1:].set(1.0 - terminated_T[: T - 2])
    mask = mask * factor
    mask_EN = jnp.broadcast_to(mask[..., None], (T - 1, E, N))

    def loss_fn(p):
        q = q_all(p)
        chosen = jnp.take_along_axis(
            q[: T - 1], actions_T[: T - 1][..., None], axis=-1
        )[
            ..., 0
        ]  # [T-1,E,N]
        td_error = chosen - jax.lax.stop_gradient(targets)
        masked = td_error * mask_EN
        loss = (masked**2).sum() / mask_EN.sum()
        return loss, chosen

    (loss, chosen), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    updates, opt_state = tx.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    diagnostics = {
        "loss": loss,
        "chosen_action_qvals": chosen,
        "targets": targets,
        "target_max": target_max,
        "mask": mask_EN,
    }
    return params, opt_state, rew_ms, diagnostics


# ----------------------------------------------------------------------------
# Off-policy training loop: epsilon-greedy episodic collection + replay + IQL.
# This is what makes IQL actually *train* (not just match one update). Fixed
# 500-step RWARE episodes (max_inactivity_steps=None) -> terminated=0, filled=1
# throughout, so episodes need no padding. Per-agent (individual) rewards are
# summed into the common team reward, matching epymarl's RWARE setup.
# ----------------------------------------------------------------------------
def _augment_obs(obs, n_agents, obs_agent_id):
    """Append a one-hot agent id to each obs (epymarl obs_agent_id=True)."""
    if not obs_agent_id:
        return obs
    E = obs.shape[0]
    ids = jnp.broadcast_to(jnp.eye(n_agents), (E, n_agents, n_agents))
    return jnp.concatenate([obs, ids], axis=-1)


def epsilon_at(cfg, t):
    """Linear DecayThenFlat schedule (epymarl epsilon_schedules.py)."""
    delta = (cfg.epsilon_start - cfg.epsilon_finish) / cfg.epsilon_anneal_time
    return jnp.maximum(cfg.epsilon_finish, cfg.epsilon_start - delta * t)


def make_iql_trainer(cfg):
    """Build (env, qnet, tx, init_state, collect, update) for IQL training.

    `collect(params, key, epsilon)` runs one batch of `parallel_envs` full
    episodes with epsilon-greedy actions and returns per-episode tensors in
    batch-major layout [E, T+1/T, N, ...] plus behavioural metrics. `update`
    is `iql_update` partially applied with the net/optimiser/cfg.
    """
    env = Warehouse(make_config(cfg.size, cfg.n_agents, cfg.difficulty))
    N = cfg.n_agents
    E = cfg.parallel_envs
    T = cfg.time_limit
    A = env.num_actions
    H = cfg.hidden_dim
    B = E * N
    in_dim = env.obs_dim + (N if cfg.obs_agent_id else 0)
    qnet = QNetwork(num_actions=A, hidden_dim=H, use_rnn=cfg.use_rnn)
    tx = make_optimizer(cfg)

    def init_state(key):
        key, ki = jax.random.split(key)
        params = qnet.init(
            ki,
            ScannedGRU.initialize_carry(B, H),
            (jnp.zeros((1, B, in_dim)), jnp.zeros((1, B))),
        )
        return params, params, tx.init(params), rms_init((1,)), key

    @jax.jit
    def collect(params, key, epsilon):
        key, kreset = jax.random.split(key)
        states, obs = jax.vmap(env.reset)(jax.random.split(kreset, E))  # obs [E,N,obs]
        obs = _augment_obs(obs, N, cfg.obs_agent_id)

        def step(carry, _):
            states, obs, h, key = carry
            key, ka, kr = jax.random.split(key, 3)
            obs_flat = obs.reshape(B, in_dim)
            h, q = qnet.apply(params, h, (obs_flat[None], jnp.zeros((1, B))))
            q = q[0]  # [B, A]
            greedy = jnp.argmax(q, axis=-1)
            rand_a = jax.random.randint(ka, (B,), 0, A)
            pick_random = jax.random.uniform(kr, (B,)) < epsilon
            actions = jnp.where(pick_random, rand_a, greedy).reshape(E, N)
            nstates, nobs, rewards, done, info = jax.vmap(env.step)(states, actions)
            nobs = _augment_obs(nobs, N, cfg.obs_agent_id)
            r_common = rewards.sum(axis=-1)  # [E] team reward
            out = (obs_flat, actions, r_common, info["deliveries"], rewards)
            return (nstates, nobs, h, key), out

        h0 = ScannedGRU.initialize_carry(B, H)
        (_, final_obs, _, _), (obs_t, act_t, rew_t, deliv_t, raw_r_t) = jax.lax.scan(
            step, (states, obs, h0, key), None, length=T
        )
        # obs_t: [T, B, in] (pre-step obs_0..obs_{T-1}); append final obs_T -> [T+1]
        obs_seq = jnp.concatenate(
            [obs_t, final_obs.reshape(1, B, in_dim)], axis=0
        )  # [T+1, B, in]
        metrics = {
            "deliveries": deliv_t.sum(axis=0).sum(axis=-1).mean(),  # team deliveries/ep
            "ep_return": raw_r_t.sum(axis=0).sum(axis=-1).mean(),
        }
        # to batch-major per-episode: obs [E, T+1, N, in]; rest [E, T, ...]
        obs_b = obs_seq.reshape(T + 1, E, N, in_dim).transpose(1, 0, 2, 3)
        act_b = act_t.transpose(1, 0, 2)  # [E, T, N]
        rew_b = rew_t.transpose(1, 0)  # [E, T]
        return obs_b, act_b, rew_b, metrics, key

    update = functools.partial(iql_update, qnet, tx, cfg)
    return dict(
        env=env,
        qnet=qnet,
        tx=tx,
        N=N,
        E=E,
        T=T,
        in_dim=in_dim,
        init_state=init_state,
        collect=collect,
        update=update,
    )


# ----------------------------------------------------------------------------
# EMAX (arXiv 2302.03439) on top of IDQN: ensemble value functions.
#
# Differences from IQL, straight from the paper:
#   * K independent Q-nets per agent, NO parameter sharing across members
#     (we carry a leading K axis on the params and vmap the net over it).
#   * exploration is a UCB policy  argmax_a(Q_mean(a) + beta*Q_std(a))  over the
#     ensemble, REPLACING epsilon-greedy.
#   * targets use the ENSEMBLE-MEAN estimate, r + gamma*(1-term)*max_a Qmean(s'),
#     shared across members; there is NO separate target network.
#   * evaluation uses majority vote over members' greedy actions.
# Diversity here comes from independent init + the UCB-driven data; the paper's
# extra bootstrapped per-member batch sampling is a refinement left as a TODO
# (we use one shared minibatch for all members).
# ----------------------------------------------------------------------------
def init_ensemble(qnet, key, K, B, H, in_dim):
    """K independently-initialised param trees, stacked on a leading K axis."""

    def init_one(k):
        return qnet.init(
            k,
            ScannedGRU.initialize_carry(B, H),
            (jnp.zeros((1, B, in_dim)), jnp.zeros((1, B))),
        )

    return jax.vmap(init_one)(jax.random.split(key, K))


def _ensemble_q(qnet, params_ens, obs_T, B, H):
    """Apply every ensemble member over a [T, B, in] sequence -> [K, T, B, A]."""
    resets = jnp.zeros((obs_T.shape[0], B))

    def one(p):
        _, q = qnet.apply(p, ScannedGRU.initialize_carry(B, H), (obs_T, resets))
        return q

    return jax.vmap(one)(params_ens)


def _ensemble_q_windowed(qnet, params_ens, obs_T, B, H, window):
    """Recurrent ensemble forward with TRUNCATED BPTT.

    The GRU hidden state still flows forward across the WHOLE episode, but the
    gradient is cut every `window` steps (stop_gradient on the carried hidden at
    each window boundary). Forward outputs are byte-identical to the full
    forward -- only the backward pass is truncated.

    Memory: stop_gradient alone does NOT lower peak memory, because every
    window's Q feeds one loss so XLA must keep all windows' activations for the
    backward pass. So each window's forward is wrapped in jax.checkpoint
    (rematerialisation): its activations are recomputed during backprop instead
    of stored, bounding peak BPTT memory to a single window (~window/T of the
    full unroll) at the cost of ~one extra forward. That is what actually lets
    the recurrent net fit a small GPU at full batch. The window loop is unrolled
    at trace time (T, window are static)."""
    T = obs_T.shape[0]

    @jax.checkpoint  # recompute this window's activations in backward, don't store
    def win_apply(p, h, obs_win):
        return qnet.apply(p, h, (obs_win, jnp.zeros((obs_win.shape[0], B))))

    def one(p):
        h = ScannedGRU.initialize_carry(B, H)
        qs = []
        start = 0
        while start < T:
            L = min(window, T - start)
            h, q_win = win_apply(p, h, obs_T[start : start + L])
            h = jax.lax.stop_gradient(h)  # cut BPTT at the window boundary
            qs.append(q_win)
            start += L
        return jnp.concatenate(qs, axis=0)  # [T, B, A]

    return jax.vmap(one)(params_ens)


def emax_update(qnet, tx, cfg, params_ens, opt_state, rew_ms, batch):
    """One EMAX gradient step: each member regresses to the shared, detached
    ensemble-mean TD(0) target. Returns (params_ens, opt_state, rew_ms, diag).

    `batch["bootstrap_mask"]` (optional, shape [K, E]) is a per-member 0/1 weight
    over the minibatch episodes: member k's loss only counts episodes with
    mask=1, so each member trains on its own bootstrap resample (paper diversity
    technique). Absent -> all-ones, i.e. the shared-minibatch behaviour.
    """
    obs_T = batch["obs_T"]
    actions_T = batch["actions_T"]
    reward_T = batch["reward_T"]
    terminated_T = batch["terminated_T"]
    filled_T = batch["filled_T"]
    T, E, N = actions_T.shape
    B = E * N
    H = cfg.hidden_dim
    K = cfg.ensemble_size
    A = qnet.num_actions
    bmask = batch.get("bootstrap_mask")
    if bmask is None:
        bmask = jnp.ones((K, E))

    # truncated BPTT only matters for the recurrent net on the differentiated
    # (loss) path; the target forward is detached, so it never stores backward
    # activations and can stay full-length.
    window = getattr(cfg, "bptt_window", 0)
    use_window = bool(cfg.use_rnn and window and 0 < window < T)

    def q_all(p_ens, windowed=False):
        if windowed and use_window:
            q = _ensemble_q_windowed(qnet, p_ens, obs_T, B, H, window)
        else:
            q = _ensemble_q(qnet, p_ens, obs_T, B, H)  # [K, T, B, A]
        return q.reshape(K, T, E, N, A)

    # ---- ensemble-mean target (detached; no target net, no double-Q) ----
    mac_ens = q_all(params_ens)
    q_mean = mac_ens.mean(axis=0)  # [T, E, N, A]
    q_next = q_mean[1:]  # [T-1, E, N, A]
    amask = batch.get("action_mask")  # [T, E, N, A] or None
    if amask is not None:
        # mask provably-no-op actions out of the bootstrap max, identically to
        # the rollout's greedy/eps selection (Warehouse.action_masks), so the
        # target and the behaviour policy stay consistent.
        q_next = jnp.where(amask[1:] > 0, q_next, _NEG_INF)
    target_max = q_next.max(axis=-1)  # [T-1, E, N]

    r = reward_T[: T - 1]
    if cfg.standardise_rewards:
        rew_ms = rms_update(rew_ms, r[..., None])
        mean, var, _ = rew_ms
        r = (r - mean[0]) / jnp.sqrt(var[0])
    r_EN = jnp.broadcast_to(r[..., None], (T - 1, E, N))

    term_used = terminated_T[: T - 1]
    targets = r_EN + cfg.gamma * (1.0 - term_used)[..., None] * target_max
    targets = jax.lax.stop_gradient(targets)

    mask = filled_T[: T - 1]
    factor = jnp.ones_like(mask).at[1:].set(1.0 - terminated_T[: T - 2])
    mask = mask * factor
    mask_EN = jnp.broadcast_to(mask[..., None], (T - 1, E, N))

    def loss_fn(p_ens):
        mac = q_all(p_ens, windowed=True)  # [K, T, E, N, A]
        idx = jnp.broadcast_to(actions_T[: T - 1], (K, T - 1, E, N))
        chosen = jnp.take_along_axis(mac[:, : T - 1], idx[..., None], axis=-1)[
            ..., 0
        ]  # [K, T-1, E, N]
        td = chosen - targets[None]  # broadcast over K
        # weight = step-validity mask * per-member bootstrap mask [K, T-1, E, N]
        w = mask_EN[None] * bmask[:, None, :, None]
        loss = ((td**2) * w).sum() / w.sum()
        return loss, (td, w)  # masked mean over K,T,E,N

    (loss, aux), grads = jax.value_and_grad(loss_fn,has_aux=True)(params_ens)
    updates, opt_state = tx.update(grads, opt_state, params_ens)
    params_ens = optax.apply_updates(params_ens, updates)
    td=aux[0]
    w=aux[1]
    loss_metric = {"loss": loss, "q_mean": q_mean, "td": td, "w": w,"grads":grads,"mac_ens":mac_ens}

    return params_ens, opt_state, rew_ms, loss_metric


def make_emax_trainer(cfg):
    """IDQN-EMAX trainer: ensemble params, UCB rollout, ensemble-mean updates."""
    env = Warehouse(make_config(cfg.size, cfg.n_agents, cfg.difficulty))
    N, E, T = cfg.n_agents, cfg.parallel_envs, cfg.time_limit
    A, H, B = env.num_actions, cfg.hidden_dim, cfg.parallel_envs * cfg.n_agents
    K = cfg.ensemble_size
    in_dim = env.obs_dim + (N if cfg.obs_agent_id else 0)
    qnet = QNetwork(num_actions=A, hidden_dim=H, use_rnn=cfg.use_rnn)
    tx = make_optimizer(cfg)

    def init_state(key):
        key, ki = jax.random.split(key)
        params_ens = init_ensemble(qnet, ki, K, B, H, in_dim)
        return params_ens, tx.init(params_ens), rms_init((1,)), key

    @jax.jit
    def collect(params_ens, key, epsilon):
        key, kreset = jax.random.split(key)
        states, obs = jax.vmap(env.reset)(jax.random.split(kreset, E))
        obs = _augment_obs(obs, N, cfg.obs_agent_id)

        def step(carry, _):
            states, obs, key = carry
            key, ka, kr = jax.random.split(key, 3)
            obs_flat = obs.reshape(B, in_dim)
            # greedy action = UCB over the ensemble: argmax_a (Qmean + beta*Qstd);
            # wrapped in epsilon-greedy for cold-start warmup (epymarl keeps the
            # epsilon schedule for all value methods; UCB is the *greedy* rule).
            q = _ensemble_q(qnet, params_ens, obs_flat[None], B, H)  # [K,1,B,A]
            q = q[:, 0]  # [K, B, A]
            ucb = q.mean(axis=0) + cfg.ucb_beta * q.std(axis=0)  # [B, A]
            greedy = jnp.argmax(ucb, axis=-1)
            rand_a = jax.random.randint(ka, (B,), 0, A)
            pick_random = jax.random.uniform(kr, (B,)) < epsilon
            actions = jnp.where(pick_random, rand_a, greedy).reshape(E, N)
            nstates, nobs, rewards, done, info = jax.vmap(env.step)(states, actions)
            nobs = _augment_obs(nobs, N, cfg.obs_agent_id)
            out = (obs_flat, actions, rewards.sum(-1), info["deliveries"], rewards)
            return (nstates, nobs, key), out

        (_, final_obs, _), (obs_t, act_t, rew_t, deliv_t, raw_r_t) = jax.lax.scan(
            step, (states, obs, key), None, length=T
        )
        obs_seq = jnp.concatenate([obs_t, final_obs.reshape(1, B, in_dim)], axis=0)
        metrics = {
            "deliveries": deliv_t.sum(0).sum(-1).mean(),
            "ep_return": raw_r_t.sum(0).sum(-1).mean(),
        }
        obs_b = obs_seq.reshape(T + 1, E, N, in_dim).transpose(1, 0, 2, 3)
        return obs_b, act_t.transpose(1, 0, 2), rew_t.transpose(1, 0), metrics, key

    update = functools.partial(emax_update, qnet, tx, cfg)
    return dict(
        env=env,
        qnet=qnet,
        N=N,
        E=E,
        T=T,
        in_dim=in_dim,
        K=K,
        init_state=init_state,
        collect=collect,
        update=update,
    )


def feed_batch(obs_b, act_b, rew_b, T, N, mask_b=None):
    """Convert sampled episodes (batch-major) to the time-major arrays
    iql_update/emax_update expect, with all tensors length T+1 (last row
    dummy/0). `mask_b` (optional, [bs, T+1, N, A]) adds the time-major action
    mask consumed by emax_update; absent -> no masking (unchanged behaviour)."""
    bs = obs_b.shape[0]
    obs_T = obs_b.transpose(1, 0, 2, 3).reshape(T + 1, bs * N, -1)  # [T+1, bs*N, in]
    pad = lambda x, last: jnp.concatenate(
        [x, jnp.full((1,) + x.shape[1:], last, x.dtype)], axis=0
    )
    act_T = pad(act_b.transpose(1, 0, 2), 0)  # [T+1, bs, N]
    rew_T = pad(rew_b.transpose(1, 0), 0.0)  # [T+1, bs]
    terminated_T = jnp.zeros((T + 1, bs))
    filled_T = pad(jnp.ones((T, bs)), 0.0)
    out = {
        "obs_T": obs_T,
        "actions_T": act_T,
        "reward_T": rew_T,
        "terminated_T": terminated_T,
        "filled_T": filled_T,
    }
    if mask_b is not None:
        out["action_mask"] = mask_b.transpose(1, 0, 2, 3)  # [T+1, bs, N, A]
    return out
