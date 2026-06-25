from typing import Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np
import re

# Fairness / Imbalance (Gini, Lorenz, per-agent stats)


def gini_coefficient(x):
    """measures how much inequality between agents there is in a distribution.
    Args:
        x: shape[T, E, N]
    Returns:
        Gini per episode shape[E]
    What does he actually say?
        If Gini is high -> there is a dominant agent or a small group of agents who do most of the work.
        If Gini is low -> the Contribution is divided equally among the agents.
    Notes:
        Any contribution (eg. Number of pick-up/drop-off operations,Number of effective steps,Number of object interactions)
    Use on:
        Deliveries/rewards/any_contribution
    """
    per_agent = x.sum(axis=(0, 1))  # Optional collapse T,E -> get [N]
    per_ep = x.sum(axis=0)  # collapse T -> [E, N]
    x = jnp.asarray(per_ep)
    sorted_vals = jnp.sort(x, axis=-1)
    n = x.shape[-1]
    idx = jnp.arange(1, n + 1)
    a = 2 * jnp.sum(idx * sorted_vals, axis=-1)
    b = n * jnp.sum(sorted_vals, axis=-1)
    c = (n + 1) / (n)

    gini = a / b - c
    gini = jnp.where(jnp.isnan(gini), 0, gini)

    return gini


def lorenz_curve(x):
    """axis graph that shows how the contribution is divided between the agents.
    Args:
        x: shape[T, E, N]
    Returns:
        Tuple:
        @X-axis: Percentage of population (agents) shape[N] per episode.

        @Y-axis: Percentage of cumulative contribution shape[E,N] per episode.
    What does he actually say?
        cumulative share (Lorenz curve) It is a graph that shows how the contribution is divided between the agents.
        Lorenz is the full form, from which Gini is calculated.
        If all agents contribute equally → the line is a straight diagonal (equality line).
        If there is inequality → the curve bends downward.
    Use on:
        Deliveries/rewards/any_contribution
    """
    N = int(x.shape[-1])
    per_agent = x.sum(axis=(0, 1))  # Optional collapse T,E -> get [N]
    per_ep = x.sum(axis=0)  # collapse T -> [E, N]
    x = jnp.asarray(per_ep)
    sorted_vals = jnp.sort(x, axis=-1)  # shape (E, N)
    cum = jnp.cumsum(sorted_vals, axis=-1)  # shape (E, N)
    x_axis = jnp.arange(1, N + 1) / N  # shape (N,)
    y_axis = cum / cum[:, -1][:, None]  # shape (E, N)
    x_axis = jnp.where(jnp.isnan(x_axis), 0, x_axis)
    y_axis = jnp.where(jnp.isnan(y_axis), 0, y_axis)
    return x_axis, y_axis


def fairness_metrics(x):
    """Use Gini and lorenz_curve
    Args:
        x: shape[T, E, N]
    Returns:
        Dict:
            fairness_gini_mean: scalar

            fairness_gini_std: scalar

            fairness_lorenz_curve_x: shape (N,)

            fairness_lorenz_curve_y: shape (E,N)

            fairness_per_ep_totals: shape (E, N,)
    Use on:
        Deliveries/rewards/Any Contribution
    """
    gini = gini_coefficient(x)  # [T, E, N] -> [E]

    gini_mean = gini.mean()
    gini_std = gini.std()

    # Lorenz curve for a representative episode
    lorenz = lorenz_curve(x)  # [T, E, N] -> [E]

    per_ep = x.sum(axis=0)  # collapse T -> [E, N]
    return {
        "fairness_gini_mean": gini_mean,
        "fairness_gini_std": gini_std,
        "fairness_lorenz_curve_x": lorenz[0],  # [N]
        "fairness_lorenz_curve_y": lorenz[1],  # [E,N]
        # "fairness_per_ep_totals": per_ep,  # [E, N]
    }


# Actor / Critic losses (mean/std/p10/p90/skew)


def loss_stats(loss):
    """Calculate mean, std,skew and percentile 10,50,90
    Args:
        loss: shape[P]
    Returns:
        loss_mean: scalar

        loss_std: scalar

        loss_p10: scalar

        loss_p50: scalar

        loss_p90: scalar

        loss_skew: scalar
    """
    # flat = loss.reshape(-1)
    p10, p50, p90 = jnp.percentile(loss, jnp.array([10, 50, 90]))
    mean = loss.mean()
    std = loss.std()
    skew = jnp.mean(((loss - mean) / (std + 1e-8)) ** 3)
    return {
        "loss_mean": mean,
        "loss_std": std,
        "loss_p10": p10,
        "loss_p50": p50,
        "loss_p90": p90,
        "loss_skew": skew,
    }


# KL divergence old/new policy


def kl_divergence(old_logp, new_logp):
    """
    Calculate KL mean,std,percentile 90 KL = E[ old_logp - new_logp ]
    Compute both:
      1. KL_mean_over_P = E[ old_logp - mean_P(new_logp) ]
      2. KL_per_P       = E[ old_logp - new_logp[p] ] for each p

    Args:
        old_logp: [T, E, N]
        new_logp: [P, T, E, N]

    Returns:
        Dict:
            kl_mean_over_P: scalar

            kl_std_over_P: scalar

            kl_p90_over_P: scalar

            kl_per_P_mean: (P,)

            kl_per_P_std: (P,)

            kl_per_P_p90: (P,)

    """

    # ---------- 1. KL mean over P ----------
    new_logp_mean = new_logp.mean(axis=0)  # [T, E, N]
    kl_over_P = old_logp - new_logp_mean  # [T, E, N]
    flat_over_P = kl_over_P.reshape(-1)  # [S]

    # ---------- 2. KL per-P ----------
    kl_per_P = old_logp[None, ...] - new_logp  # [P, T, E, N]
    flat_per_P = kl_per_P.reshape(kl_per_P.shape[0], -1)  # [P, S]

    return {
        # Mean-over-P KL
        "kl_mean_over_P": flat_over_P.mean(),
        "kl_std_over_P": flat_over_P.std(),
        "kl_p90_over_P": jnp.percentile(flat_over_P, 90),
        # "kl_raw_over_P": flat_over_P,
        # Per-P KL
        "kl_per_P_mean": flat_per_P.mean(axis=1),  # [P]
        "kl_per_P_std": flat_per_P.std(axis=1),  # [P]
        "kl_per_P_p90": jnp.percentile(flat_per_P, 90, axis=1),  # [P]
        # "kl_per_P_raw": flat_per_P,  # [P, S]
    }


def extract_flat_grads(grads_raw, prefix="", out=None):
    """

    """
    if out is None:
        out = {}
    for key, value in grads_raw.items():
        full_key = f"{prefix}/{key}" if prefix else str(key)

        if isinstance(value, dict):
            # recursive descent
            extract_flat_grads(value, prefix=full_key, out=out)

        elif isinstance(value, jnp.ndarray):
            # stop condition → save tensor
            out[full_key] = value

        else:
            # ignore anything else (lists, None, scalars, etc.)
            pass
    
    return out


# Per-agent gradient norms
def gradient_norms_per_agent(grads, N, O):
    """Calculate L2 norm per agent on centralized critic ONLY
    Args:
        grads: flat dict[param_name]
        N: number of agents
        O: observation size per agent


    Returns:
        per_agent_norms: shape(P,N,)

        mean_per_agent: shape (N,),

        std_per_agent: shape (N,),

        grad_norms_per_agent_p90: shape (N,),
    """

    def filter_centralized_params(grads_raw, N, O):
        filtered = {}
        for name, arr in grads_raw.items():
            # for k,v in centralized.items():
            if arr.ndim == 3 and arr.shape[1] == N * O:
                filtered[name] = arr
        return filtered

    centralized = filter_centralized_params(grads, N, O)
    per_param_agent_norms = []

    for _, g in centralized.items():
        # g: (P, N*O, H0) → reshape to (P, N, O, H0)
        g = g.reshape(g.shape[0], N, O, g.shape[-1])
        # L2 norm over O and H0 → (P, N)
        param_norm = jnp.sqrt(jnp.sum(g**2, axis=(2, 3)))
        per_param_agent_norms.append(param_norm)
    
    if not per_param_agent_norms:
        # no centralized params found → zeros
        P = next(iter(grads.values())).shape[0]
        total_norm = jnp.zeros((P, N))
    else:
        # sum across parameters → shape (P, N)
        total_norm = jnp.sum(jnp.stack(per_param_agent_norms, axis=0), axis=0)

    return {
        "grad_norms_per_agent_per_epoch": total_norm,  # (P, N)
        "grad_norms_per_agent_mean": total_norm.mean(axis=0),  # (N,)
        "grad_norms_per_agent_std": total_norm.std(axis=0),  # (N,)
        # (N,)
        "grad_norms_per_agent_p90": jnp.percentile(total_norm, 90, axis=0),
    }


def gradient_norms(grads):
    """Calculate L2 norm per update
    Args:
        grads: dict

    Returns:
        grad_norm_per_epoch: shape (P,)

        grad_norm_mean: scalar

        grad_norm_std: scalar

        grad_norm_p90: scalar
    """
    per_param_norms = []

    for _, g in grads.items():

        # axes 1..end = param dims
        param_axes = tuple(range(1, g.ndim))

        # L2 norm over param dims → shape (P,) g shape (P, ...)
        param_norm = jnp.sqrt(jnp.sum(g**2, axis=param_axes))

        per_param_norms.append(param_norm)

    # sum norms across all parameters → shape (P,)
    total_norm = jnp.sum(jnp.stack(per_param_norms, axis=0), axis=0)

    return {
        # "grad_norm_per_epoch": total_norm,  # (P,)
        "grad_norm_mean": total_norm.mean(),  # scalar
        "grad_norm_std": total_norm.std(),  # scalar
        "grad_norm_p90": jnp.percentile(total_norm, 90),  # scalar
    }


# Reward-weighted metrics (RWARE)
def rware_metric(rewards, deliveries):
    """Calculate on episode succeeded Reward-weighted metrics (RWARE)
    Args:
        rewards: [T, E, N]
        deliveries: [T, E, N]
    Returns:
        Dict:
            rware_mean: scalar

            rware_std: scalar

            rware_p10: scalar

            rware_p50: scalar

            rware_p90: scalar
    Info:
        Calculate succeeded by mask reward only where was deliverie

    """

    def compute_success_mask(deliveries):
        """
        Args:
            deliveries: [T, E, N] binary
        Returns:
            success_mask: [E] 1 if episode succeeded
        """
        # total deliveries per environment
        per_env = deliveries.sum(axis=0).sum(axis=-1)  # [E]

        # success if at least one delivery happened
        success_mask = (per_env > 0).astype(bool)

        return success_mask

    success_mask = compute_success_mask(deliveries)  # [E]

    ep_return = rewards.sum(axis=0).sum(axis=-1)  # [E]
    weighted: jnp.ndarray = ep_return[success_mask]  # shape [(0,)-(E,)]
    if weighted.size > 0:
        weighted_metric = {
            "rware_mean": weighted.mean(),
            "rware_std": weighted.std(),
            "rware_p10": jnp.percentile(weighted, 10),
            "rware_p50": jnp.percentile(weighted, 50),
            "rware_p90": jnp.percentile(weighted, 90),
        }
    else:
        weighted_metric = {
            "rware_mean": jnp.float32(0.0),
            "rware_std": jnp.float32(0.0),
            "rware_p10": jnp.float32(0.0),
            "rware_p50": jnp.float32(0.0),
            "rware_p90": jnp.float32(0.0),
        }

    return weighted_metric


# Credit‑assignment proxies (correlation, Shapley‑approx)
def credit_assignment_proxies(actions: jnp.ndarray, rewards: jnp.ndarray):
    """Calculate leave-one-out marginal contributions (approximate Shapley by sampling agents permutations)
    per episode and aggregate
    Args:
      actions: jnp.ndarray with shape (T, E, N)(required shapes)
      rewards: jnp.ndarray with shape (T, E, N)(required shapes)

    Returns:
        Dict:
            credit_corr: jnp.ndarray shape (N,),        # Pearson r per agent across E parallel envs

            credit_shapley: jnp.ndarray shape (N,),     # exact Shapley under additive team-value (mean agent return)

            credit_shapley_loo: jnp.ndarray shape (N,)  # leave-one-out marginal proxy per agent

    Notes:
      - Aggregation over time: actions -> mean over time per env; returns -> sum over time per env.
      - Correlations computed across the E parallel environments for each agent.
      - If you prefer different time aggregation (e.g., returns.mean(axis=0)), change returns.sum(axis=0) accordingly.

    """
    # --- shape checks ---
    if actions.ndim != 3 or rewards.ndim != 3:
        raise ValueError("actions and returns must have shape (T, E, N)")
    if actions.shape != rewards.shape:
        raise ValueError("actions and returns must have identical shapes (T, E, N)")

    T, E, N = actions.shape

    # --- Episode / rollout level aggregation per parallel environment ---
    # Per-env per-agent action magnitude: average absolute action over time
    act_mag_env = jnp.abs(actions).mean(axis=0)  # shape (E, N)

    # Per-env per-agent return: sum rewards over time (episode return within that env)
    returns_env = rewards.sum(axis=0)  # shape (E, N)

    # --- Per-agent Pearson correlation across parallel environments ---
    def pearson_r(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        # x, y shape (E,)
        xm = x - x.mean()
        ym = y - y.mean()
        cov = (xm * ym).mean()
        x_std = jnp.sqrt((xm * xm).mean())
        y_std = jnp.sqrt((ym * ym).mean())
        denom = x_std * y_std
        return jnp.where(denom > 0.0, cov / denom, 0.0)

    idx = jnp.arange(N)
    credit_corr = jax.vmap(lambda i: pearson_r(act_mag_env[:, i], returns_env[:, i]))(
        idx
    )
    # credit_corr shape (N,)

    # --- Leave-one-out marginal contribution (explicit LOO proxy) ---
    # Team return per env (sum over agents)
    team_return_env = returns_env.sum(axis=1)  # shape (E,)
    mean_team = team_return_env.mean()  # scalar

    # mean team return without each agent i
    mean_without_i = jax.vmap(lambda i: (team_return_env - returns_env[:, i]).mean())(
        idx
    )
    credit_shapley_loo = mean_team - mean_without_i  # shape (N,)

    # --- Exact Shapley for additive team-value v(S) = sum_i returns_i ---
    # For additive value, Shapley_i = E_env[ returns_env[:, i] ] (mean agent return across envs)
    credit_shapley = returns_env.mean(axis=0)  # shape (N,)

    return {
        "credit_per_agent_correlations": credit_corr,  # shape (N,)
        "credit_per_agent_shapley": credit_shapley,  # shape (N,)
        "credit_per_agent_shapley_loo": credit_shapley_loo,  # shape (N,)
    }


# Per-agent action distribution / diversity index


def action_distribution_metrics(
    actions: jnp.ndarray,
    num_bins,
):
    """
    Compute per-agent action histograms, entropies, and pairwise JSD for actions shaped (T, E, N).

    Args:
      actions: jnp.ndarray with shape (T, E, N). Each entry is a scalar action value for agent n.
      num_bins: number of histogram bins per agent.

    Returns:
        Dict:
            "action_histogram": jnp.ndarray shape (N, A)  -- counts per bin (float)

            "action_entropy_per_agent": jnp.ndarray shape (N,)  -- Shannon entropy in nats

            "action_jsd_matrix": jnp.ndarray shape (N, N)       -- pairwise Jensen-Shannon divergences
    """
    clip_eps = 1e-12  # Avoid zero range by expanding tiny ranges
    # --- shape checks ---
    if actions.ndim != 3:
        raise ValueError("actions must have shape (T, E, N)")
    T, E, N = actions.shape

    # Flatten time and parallel-env axes to get samples per agent: shape (T*E, N)
    samples = actions.reshape((T * E, N))  # shape (S, N) where S = T*E

    # Compute per-agent min and max to build per-agent binning range
    # Use global min/max across samples for stable shared bins, or per-agent bins if preferred.
    # Here we use per-agent min/max to preserve agent-specific ranges.
    mins = jnp.min(samples, axis=0)  # shape (N,)
    maxs = jnp.max(samples, axis=0)  # shape (N,)

    # Avoid zero range by expanding tiny ranges
    ranges = jnp.maximum(maxs - mins, clip_eps)

    # Normalize samples to [0, 1) per agent, then map to bin indices 0..num_bins-1
    # samples_norm shape (S, N)
    samples_norm = (samples - mins[None, :]) / ranges[None, :]

    # compute bin indices
    bin_idx = jnp.floor(samples_norm * num_bins).astype(jnp.int32)
    bin_idx = jnp.clip(bin_idx, 0, num_bins - 1)  # shape (S, N)

    # One-hot encode and sum over samples to get counts per bin per agent
    # one_hot shape (S, N, num_bins)
    one_hot = jax.nn.one_hot(bin_idx, num_bins, dtype=jnp.float32)
    # Sum over samples axis -> hist shape (N, num_bins)
    hist = one_hot.sum(axis=0)  # shape (N, num_bins)

    # Convert counts to probabilities per agent with numerical safety
    hist_sum = hist.sum(axis=1, keepdims=True)  # shape (N, 1)
    hist_sum_safe = jnp.maximum(hist_sum, clip_eps)
    p = hist / hist_sum_safe  # shape (N, num_bins)

    # Entropy per agent (Shannon entropy in nats)
    entropy = -jnp.sum(p * jnp.log(jnp.maximum(p, clip_eps)), axis=1)  # shape (N,)

    # Pairwise Jensen-Shannon divergence matrix
    # p shape (N, B). Build pairwise mixtures m_ij = 0.5*(p_i + p_j)
    p_i = p[:, None, :]  # (N, 1, B)
    p_j = p[None, :, :]  # (1, N, B)
    m = 0.5 * (p_i + p_j)  # (N, N, B)

    # KL divergence safe helper: KL(p || q) = sum p * log(p / q)
    def kl_divergence(p_arr, q_arr):
        # both shape (..., B)
        ratio = jnp.where(p_arr > 0.0, p_arr / jnp.maximum(q_arr, clip_eps), 1.0)
        return jnp.sum(jnp.where(p_arr > 0.0, p_arr * jnp.log(ratio), 0.0), axis=-1)

    kl1 = kl_divergence(p_i, m)  # shape (N, N) diversity index
    kl2 = kl_divergence(p_j, m)  # shape (N, N) diversity index

    jsd_matrix = 0.5 * (kl1 + kl2)  # shape (N, N)

    return {
        "action_histogram": hist,  # counts per bin
        "action_entropy_per_agent": entropy,  # entropy per agent
        "action_jsd_matrix": jsd_matrix,  # pairwise JSD
    }


# Advantage / TD error stats


def advantage_stats(advantage):
    """Aggregates all entries and returns summary statistics.
    Args:
    advantage: shape [P, T, E, N]
    Returns:
        Dict:
            advantage_mean:scalar

            advantage_std:scalar

            advantage_p10:scalar

            advantage_p50:scalar

            advantage_p90:scalar

            advantage_skew:scalar
    Notes:
        - Uses sample (Fisher-Pearson) skewness when n >= 3, else returns 0.
        - Percentiles computed with jnp.percentile.
        - Optionally returns flattened raw array if return_raw=True.
    """
    return_raw = False
    eps = 1e-12  # Use eps to avoid division by zero
    if advantage.ndim != 4:
        raise ValueError("advantage must have shape [P, T, E, N]")

    flat = advantage.ravel().astype(jnp.float32)
    n = flat.size

    mean = flat.mean()
    # population std (ddof=0). Use eps to avoid division by zero.
    std = jnp.sqrt(jnp.mean((flat - mean) ** 2))
    std_safe = jnp.maximum(std, eps)

    # percentiles
    p10 = jnp.percentile(flat, 10.0)
    p50 = jnp.percentile(flat, 50.0)
    p90 = jnp.percentile(flat, 90.0)

    # sample skewness (Fisher-Pearson g1) when n >= 3
    if n >= 3:
        m3 = (
            jnp.sum(((flat - mean) / std_safe) ** 3) / n
        )  # mean third standardized moment
        # Fisher-Pearson adjusted g1:
        g1 = (n / ((n - 1) * (n - 2))) * m3
        skew = g1
    else:
        skew = jnp.array(0.0, dtype=jnp.float32)

    out = {
        "advantage_mean": mean,
        "advantage_std": std,
        "advantage_p10": p10,
        "advantage_p50": p50,
        "advantage_p90": p90,
        "advantage_skew": skew,
    }
    if return_raw:
        out["adv_raw"] = flat
    return out


# Number of critic updates vs actor updates
def analyze_update(
    epoch_actor_loss,
    epoch_value_loss,
    epoch_entropy,
    epoch_values,
):
    """Compute PPO/Actor-Critic diagnostics.
    Args:
        epoch_actor_loss: shape [P,T,E]
        epoch_value_loss: shape [P,T,E]
        epoch_entropy: shape [P,T,E,N]
        epoch_values: shape [P,T,E,N]
    Returns:
        Dict:
        per_epoch_ratio: (P,)

        cumulative_ratio: (P,)

        critic_loss_per_epoch: (P,)

        actor_loss_per_epoch: (P,)

        entropy_per_epoch: (P,)

        q_value_magnitude: (P,)

        UTD: scalar

        critic_loss_trend: scalar,

        actor_loss_trend: scalar,

        q_value_trend: scalar,

        entropy_trend: scalar,

        ratio_trend: scalar,

    Notes:
        - Per-epoch ratio - Critic loss / Actor loss per PPO epoch.
        - Cumulative ratio - Running ratio across epochs.
        - Critic loss trend - Detects critic overfitting or collapse.
        - Actor entropy trend - Detects policy collapse or over-exploration.
        - Q-value magnitude - Detects critic overestimation (SAC/TD3 style).
    """

    # -----------------------------
    # Extract tensors for update i
    # -----------------------------
    actor_loss = epoch_actor_loss  # (P, T, E)
    critic_loss = epoch_value_loss  # (P, T, E)
    entropy = epoch_entropy  # (P, T, E, N)
    values = epoch_values  # (P, T, E, N)
    env_steps_per_update = int(epoch_values.shape[1])
    P = actor_loss.shape[0]

    # -----------------------------
    # 1. Per-epoch ratio (critic/actor)
    # -----------------------------
    # mean loss per epoch
    actor_loss_per_epoch = actor_loss.mean(axis=(1, 2))  # (P,)
    critic_loss_per_epoch = critic_loss.mean(axis=(1, 2))  # (P,)

    # avoid division by zero
    per_epoch_ratio = critic_loss_per_epoch / (actor_loss_per_epoch + 1e-8)

    # -----------------------------
    # 2. Cumulative ratio
    # -----------------------------
    cumulative_ratio = critic_loss_per_epoch.cumsum() / (
        actor_loss_per_epoch.cumsum() + 1e-8
    )

    # -----------------------------
    # 4. Actor entropy trend
    # -----------------------------
    entropy_per_epoch = entropy.mean(axis=(1, 2, 3))  # (P,)

    # -----------------------------
    # 5. Q-value magnitude (only if critic exists)
    # -----------------------------
    q_magnitude = values.mean(axis=(1, 2, 3))  # (P,)

    # -----------------------------
    # 6. UTD (only for actor-critic)
    # -----------------------------
    if env_steps_per_update is not None:
        utd = P / env_steps_per_update
    else:
        utd = None

    # -----------------------------
    # 7. Trend classifier
    # -----------------------------
    def classify_trend(arr):
        delta = arr[-1] - arr[0]
        # if jnp.abs(delta) < 1e-6:
        #    return jnp.float32(0.0)
        # return jnp.float32(1.0) if delta > 0 else jnp.float32(-1.0)
        return jnp.float32(delta)

    # -----------------------------
    # Return everything
    # -----------------------------
    return {
        "per_epoch_ratio": per_epoch_ratio,  # (P,)
        "cumulative_ratio": cumulative_ratio,  # (P,)
        "critic_loss_per_epoch": critic_loss_per_epoch,  # (P,)
        "actor_loss_per_epoch": actor_loss_per_epoch,  # (P,)
        "entropy_per_epoch": entropy_per_epoch,  # (P,)
        "q_value_magnitude": q_magnitude,  # (P,)
        "UTD": jnp.float32(utd) if not None else jnp.inf,  # scalar
        "critic_loss_trend": classify_trend(critic_loss_per_epoch),  # scalar
        "actor_loss_trend": classify_trend(actor_loss_per_epoch),  # scalar
        "entropy_trend": classify_trend(entropy_per_epoch),  # scalar
        "q_value_trend": classify_trend(q_magnitude),  # scalar
        "ratio_trend": classify_trend(per_epoch_ratio),  # scalar
    }


# Per-agent entropy
def per_agent_entropy(epoch_entropy):
    """
    Compute per-agent entropy statistics.

    Args:
        epoch_entropy_tensor_raw: jnp array (P, T, E, N)

    Returns:
        Dict:
            entropy_per_agent_mean: (N,)
            entropy_per_agent_std: (N,)
            entropy_per_agent_p10: (N,)
            entropy_per_agent_p50: (N,)
            entropy_per_agent_p90: (N,)
            entropy_per_agent_trend: (N,)
    """

    # Flatten P, T, E → leave N separate
    # shape becomes (P*T*E, N)
    flat = epoch_entropy.reshape(-1, epoch_entropy.shape[-1])

    # Per-agent stats
    entropy_per_agent_mean = flat.mean(axis=0)  # (N,)
    entropy_per_agent_std = flat.std(axis=0)  # (N,)
    entropy_per_agent_p10 = jnp.percentile(flat, 10, axis=0)
    entropy_per_agent_p50 = jnp.percentile(flat, 50, axis=0)
    entropy_per_agent_p90 = jnp.percentile(flat, 90, axis=0)

    # Trend per agent: compare first epoch vs last epoch
    # shape (P, T, E, N) → mean over T,E → (P, N)
    per_epoch_per_agent = epoch_entropy.mean(axis=(1, 2))  # (P, N)
    entropy_per_agent_trend = per_epoch_per_agent[-1] - per_epoch_per_agent[0]

    return {
        # "entropy_raw": ent,                         # (P, T, E, N)
        "entropy_per_agent_mean": entropy_per_agent_mean,  # (N,)
        "entropy_per_agent_std": entropy_per_agent_std,  # (N,)
        "entropy_per_agent_p10": entropy_per_agent_p10,  # (N,)
        "entropy_per_agent_p50": entropy_per_agent_p50,  # (N,)
        "entropy_per_agent_p90": entropy_per_agent_p90,  # (N,)
        "entropy_per_agent_trend": entropy_per_agent_trend,  # (N,)
    }


# Per-epoch entropy
def entropy_Per_epoch(epoch_entropy):
    """
    epoch_entropy: (P, T, E, N)
    """

    # Per-epoch entropy
    entropy_per_epoch = epoch_entropy.mean(axis=(1, 2, 3))  # (P,)

    # Flatten for stats
    flat = epoch_entropy.reshape(-1)

    stats = {
        # "entropy_per_epoch": entropy_per_epoch,  # (P,)
        "entropy_mean": flat.mean(),  # scalar
        "entropy_std": flat.std(),  # scalar
        "entropy_p10": jnp.percentile(flat, 10),  # scalar
        "entropy_p50": jnp.percentile(flat, 50),  # scalar
        "entropy_p90": jnp.percentile(flat, 90),  # scalar
        # "entropy_raw": epoch_entropy_tensor_raw,
        "entropy_trend": entropy_per_epoch[-1] - entropy_per_epoch[0],  # scalar
    }

    return stats


def reward_per_agent(rewards):
    """
    Compute per-agent reward statistics.

    Args:
        rewards: jnp array (T, E, N)

    Returns:
        dict with:
            - reward_per_agent_mean: (N,)
            - reward_per_agent_std: (N,)
            - reward_per_agent_p10: (N,)
            - reward_per_agent_p50: (N,)
            - reward_per_agent_p90: (N,)
            - reward_per_agent_trend: (N,)
    """

    # Flatten T, E → leave N separate
    # shape becomes (T*E, N)
    flat = rewards.reshape(-1, rewards.shape[-1])

    # Per-agent stats
    reward_per_agent_mean = flat.mean(axis=0)  # (N,)
    reward_per_agent_std = flat.std(axis=0)  # (N,)
    reward_per_agent_p10 = jnp.percentile(flat, 10, axis=0)
    reward_per_agent_p50 = jnp.percentile(flat, 50, axis=0)
    reward_per_agent_p90 = jnp.percentile(flat, 90, axis=0)

    # Trend per agent: compare first timestep vs last timestep
    # mean over E → (T, N)
    per_timestep_per_agent = rewards.mean(axis=1)  # (T, N)
    reward_per_agent_trend = per_timestep_per_agent[-1] - per_timestep_per_agent[0]

    return {
        # "rewards_raw": rewards,                             # (T, E, N)
        "reward_per_agent_mean": reward_per_agent_mean,
        "reward_per_agent_std": reward_per_agent_std,
        "reward_per_agent_p10": reward_per_agent_p10,
        "reward_per_agent_p50": reward_per_agent_p50,
        "reward_per_agent_p90": reward_per_agent_p90,
        "reward_per_agent_trend": reward_per_agent_trend,
    }


# ---- behavioral aggregates (all in-graph; only per-update scalars leave
# the scan). team_per_ep sums over time + agents, means over envs; `frac`
# is the fraction (rate) of agent-steps.
# for x.shape and deliv_t.shape [T,E,...,N] only
def team_per_ep(str, x):
    return (
        x.sum(axis=0).sum(axis=-1).mean()
        if str == "mean"
        else x.sum(axis=0).sum(axis=-1).std()
    )


def num_successful(str, deliv_t):
    return (
        (deliv_t.sum(axis=0) > 0).sum(axis=-1).mean()
        if str == "mean"
        else (deliv_t.sum(axis=0) > 0).sum(axis=-1).std()
    )


def team_success_rate(str, deliv_t):
    return (
        jnp.all((deliv_t.sum(axis=0) > 0), axis=-1).mean()
        if str == "mean"
        else jnp.all((deliv_t.sum(axis=0) > 0), axis=-1).std()
    )


# for returns,
def per_agent(str, x):
    return (
        x.sum(axis=0).mean(axis=0) if str == "mean" else x.sum(axis=0).std(axis=0)
    )  # [N]


# for x.shape [T,...] only
def frac(str, x):
    return x.mean() if str == "mean" else x.std()


# for x.shape [T,E,...] only


def E_per_T(str, x):
    return x.sum(axis=0).mean() if str == "mean" else x.sum(axis=0).std()


def percentile_stats(x):
    """
    # for returns, deliveries, step_time
    The p-percentile is the value below which p% of the data are found.
    That is:
    10th percentile → 10% of the samples are smaller than it
    50th percentile → 50% of the samples are smaller than it (this is the median)
    90th percentile → 90% of the samples are smaller than it
    This is a way of describing the distribution of the data, not just the mean.
    """
    return jnp.percentile(x, jnp.array([10, 50, 90]))


def bootstrap_ci(x, n_boot=2000, alpha=0.05, key=None):
    flat = x.reshape(-1)
    n = flat.size
    if key is None:
        key = jax.random.PRNGKey(0)

    def one_bootstrap(k):
        idx = jax.random.randint(k, (n,), 0, n)
        sample = flat[idx]
        return jnp.percentile(sample, 50)  # median CI

    keys = jax.random.split(key, n_boot)
    samples = jax.vmap(one_bootstrap)(keys)
    quintent = jnp.asarray([100 * alpha / 2, 100 * (1 - alpha / 2)])
    return jnp.percentile(samples, quintent)


def empirical_cdf(x, grid=None):
    flat = jnp.sort(x.reshape(-1))
    if grid is None:
        grid = flat
    cdf = jnp.searchsorted(flat, grid, side="right") / flat.size
    return grid, cdf


def time_to_completion(deliv):
    """
    deliv: [T, E, N] binary
    target_deliveries:
        None → completion = at least one delivery
        int  → completion = that many deliveries
    """
    target_deliveries = None
    # target_deliveries=each episode requires 3 pickups + 3 dropoffs
    # collapse agents
    team_deliv = deliv.sum(axis=-1)  # [T, E]

    # cumulative deliveries
    cum = jnp.cumsum(team_deliv, axis=0)  # [T, E]

    # default: completion = at least one delivery
    if target_deliveries is None:
        target_deliveries = 1

    reached = cum >= target_deliveries  # [T, E]

    first_idx = jnp.argmax(reached, axis=0)  # [E]

    no_completion = ~reached.any(axis=0)

    return jnp.where(no_completion, jnp.inf, first_idx)


def additional_metrics(delivet_t):
    # collapse agents [T, E, N]->[T, E]->[E] ,find first True along T ,if no delivery happened → mark as inf
    ttfd = jnp.where(
        ~delivet_t.any(axis=-1).any(axis=0),
        jnp.inf,
        jnp.argmax(delivet_t.any(axis=-1), axis=0),
    )  # [E]
    ttc = time_to_completion(delivet_t)  # [E]

    # mask out inf for stats that shouldn't be dominated by "never"
    finite_ttfd = ttfd[jnp.isfinite(ttfd)]
    finite_ttc = ttc[jnp.isfinite(ttc)]

    # mean
    ttfd_mean = finite_ttfd.mean() if finite_ttfd.size > 0 else jnp.inf
    ttc_mean = finite_ttc.mean() if finite_ttc.size > 0 else jnp.inf
    # std
    ttfd_std = finite_ttfd.std() if finite_ttfd.size > 0 else jnp.inf
    ttc_std = finite_ttc.std() if finite_ttc.size > 0 else jnp.inf

    # percentiles (10/50/90)
    ttfd_p10, ttfd_p50, ttfd_p90 = (
        percentile_stats(finite_ttfd)
        if finite_ttfd.size > 0
        else (jnp.inf, jnp.inf, jnp.inf)
    )
    ttc_p10, ttc_p50, ttc_p90 = (
        percentile_stats(finite_ttc)
        if finite_ttc.size > 0
        else (jnp.inf, jnp.inf, jnp.inf)
    )

    # CDF (for plotting later)
    ttfd_grid, ttfd_cdf = empirical_cdf(finite_ttfd)
    ttc_grid, ttc_cdf = empirical_cdf(finite_ttc)

    # bootstrap CI on median
    ttfd_ci = (
        bootstrap_ci(finite_ttfd)
        if finite_ttfd.size > 0
        else jnp.array([jnp.inf, jnp.inf])
    )
    ttc_ci = (
        bootstrap_ci(finite_ttc)
        if finite_ttc.size > 0
        else jnp.array([jnp.inf, jnp.inf])
    )

    def flat_arr_to_metric(x: jnp.ndarray, dict_key):
        metric = {}
        if x.shape[0] > 0:
            for i, val in enumerate(x):
                metric[dict_key + f"_{i}"] = val
        else:
            metric[dict_key] = x
        return metric

    episode_metrics_mean = {
        "time_to_first_delivery_mean": ttfd_mean,
        "time_to_first_delivery_p10": ttfd_p10,
        "time_to_first_delivery_p50": ttfd_p50,
        "time_to_first_delivery_p90": ttfd_p90,
        "time_to_first_delivery_bootstrap_CI_on_lower_bound_of_median": ttfd_ci[0],
        "time_to_first_delivery_bootstrap_CI_on_upper_bound_bound_of_median": ttfd_ci[
            1
        ],
        **flat_arr_to_metric(ttfd_grid, "time_to_first_delivery_cdf_grid"),
        **flat_arr_to_metric(ttfd_cdf, "time_to_first_delivery_cdf"),
        "time_to_completion_mean": ttc_mean,
        "time_to_completion_p10": ttc_p10,
        "time_to_completion_p50": ttc_p50,
        "time_to_completion_p90": ttc_p90,
        "time_to_completion_bootstrap_CI_on_lower_bound_of_median": ttc_ci[0],
        "time_to_completion_bootstrap_CI_on_upper_bound_bound_of_median": ttc_ci[1],
        **flat_arr_to_metric(ttc_grid, "time_to_completion_cdf_grid"),
        **flat_arr_to_metric(ttc_cdf, "time_to_completion_cdf"),
    }

    episode_metrics_std = {
        "time_to_first_delivery_std": ttfd_std,
        "time_to_completion_std": ttc_std,
    }
    return episode_metrics_mean, episode_metrics_std


def per_agent_dict(dict: Dict | jnp.ndarray, dictKey: Optional[str] = None):
    """
    Args:
        dict: Dict[jnp.ndarray] | jnp.ndarray shape(N,)
        dictKey:Optional[str] if dict is jnp.ndarray
    Returns:
        Per agent dict
    """
    if isinstance(dict, Dict):
        per_agent_metrics = {}
        for k, v in dict.items():
            for i, agent_v in enumerate(v):
                if "per_agent" in k:
                    new_key = k.replace("per_agent", f"agent_{i}")
                else:
                    new_key = k + f"_agent_{i}"
                per_agent_metrics[new_key] = agent_v
        return per_agent_metrics
    else:
        per_agent_metrics = {}
        for i, agent_v in enumerate(dict):
            if dictKey is None:
                ValueError("Need to include dictKey in per_agent_dict() args")
                exit(0)
            elif "per_agent" in dictKey:
                new_key = dictKey.replace("per_agent", f"agent_{i}")
            else:
                new_key = dictKey + f"_agent_{i}"
            per_agent_metrics[new_key] = agent_v
        return per_agent_metrics


def flat_action_histogram(action_histogram: jnp.ndarray):
    N = action_histogram.shape[0]
    num_bins = action_histogram.shape[1]
    row = {}
    for n in range(N):  # (N, num_bins)
        for b in range(num_bins):
            row[f"action_histogram_flat_a{n}_b{b}"] = action_histogram[n, b]
    return row


def flat_action_jsd_metric(jsd_metric: jnp.ndarray):
    N = jsd_metric.shape[0]
    row = {}
    for i in range(N):  # (N, N)
        for j in range(N):
            row[f"jsd_a{i}_a{j}"] = jsd_metric[i, j]
    return row


def flat_fairness_metrics_lorenz_y(lorenz: jnp.ndarray, key: str):
    E = lorenz.shape[0]
    N = lorenz.shape[0]
    row = {}
    for e in range(E):
        for n in range(N):
            row[f"{key}_e{e}_a{n}"] = lorenz[e, n]
    return row

def create_dummy_metrics(
    E,
    N ,
    A,
    U = 1,
    P = 1,
    T = 1,
    O = 1,
    H0 = 1,
):

    episode_time = jnp.ones(
        (U,),
    )
    step_count_tensor_raw = jnp.ones(
        (U, E),
    )
    epoch_loss_tensor_raw = jnp.ones(
        (U, P),
    )
    step_time_tensor_raw = jnp.ones(
        (U, T, E),
    )
    block_tensor_raw = jnp.ones(
        (U, T, E, N),
    )
    rewards_tensor_raw = jnp.ones(
        (U, T, E, N),
    )
    actions_tensor_raw = jnp.ones(
        (U, T, E, N),
    )
    observation_tensor_raw = jnp.ones(
        (U, T, E, N, O),
    )
    deliveries_tensor_raw = jnp.ones(
        (U, T, E, N),
    )
    distance_traveled_tensor_raw = jnp.ones(
        (U, T, E, N),
    )
    epoch_old_logp_tensor_raw = jnp.ones(
        (U, T, E, N),
    )
    epoch_returns_tensor_raw = jnp.ones(
        (U, T, E, N),
    )
    idle_tensor_raw = jnp.ones(
        (U, T, E, N),
    )
    pickup_tensor_raw = jnp.ones(
        (U, T, E, N),
    )
    logits_tensor_raw = jnp.ones(
        (U, T, 1, E * N, A),
    )
    rewards_std_tensor_raw = jnp.ones(
        (U, T, E, N),
    )
    epoch_actor_loss_tensor_raw = jnp.ones(
        (U, P, T, E),
    )
    epoch_value_loss_tensor_raw = jnp.ones(
        (U, P, T, E),
    )
    epoch_advantage_tensor_raw = jnp.ones(
        (U, P, T, E, N),
    )
    epoch_entropy_tensor_raw = jnp.ones(
        (U, P, T, E, N),
    )
    epoch_logp_tensor_raw = jnp.ones(
        (U, P, T, E, N),
    )
    epoch_values_tensor_raw = jnp.ones(
        (U, P, T, E, N),
    )
    epoch_grads_tensor_raw = {
        "critic": {"params": {"Dense_0": {"kernel": jnp.ones((U, P, N * O, H0))}}},
        "actor": {"params": {"Dense_0": {"kernel": jnp.ones((U, P, O, H0))}}},
    }

    dummy_metric = {
        "episode_time": episode_time,
        "step_count_tensor_raw": step_count_tensor_raw,
        "epoch_loss_tensor_raw": epoch_loss_tensor_raw,
        "step_time_tensor_raw": step_time_tensor_raw,
        "block_tensor_raw": block_tensor_raw,
        "rewards_tensor_raw": rewards_tensor_raw,
        "actions_tensor_raw": actions_tensor_raw,
        "observation_tensor_raw": observation_tensor_raw,
        "deliveries_tensor_raw": deliveries_tensor_raw,
        "distance_traveled_tensor_raw": distance_traveled_tensor_raw,
        "epoch_old_logp_tensor_raw": epoch_old_logp_tensor_raw,
        "epoch_returns_tensor_raw": epoch_returns_tensor_raw,
        "idle_tensor_raw": idle_tensor_raw,
        "pickup_tensor_raw": pickup_tensor_raw,
        "logits_tensor_raw": logits_tensor_raw,
        "rewards_std_tensor_raw": rewards_std_tensor_raw,
        "epoch_actor_loss_tensor_raw": epoch_actor_loss_tensor_raw,
        "epoch_value_loss_tensor_raw": epoch_value_loss_tensor_raw,
        "epoch_advantage_tensor_raw": epoch_advantage_tensor_raw,
        "epoch_entropy_tensor_raw": epoch_entropy_tensor_raw,
        "epoch_logp_tensor_raw": epoch_logp_tensor_raw,
        "epoch_values_tensor_raw": epoch_values_tensor_raw,
        "epoch_grads_tensor_raw": epoch_grads_tensor_raw,
    }
    return dummy_metric


# manual priority from "most meaningful" to less (based on MARL practice / your header)
def sort_metrics(keys):
    PRIORITY_ORDER = [
        "environment_steps",
        "updates",
        "episode_time",
        "episode_return_mean",
        "episode_return_std",
        "success_mean",
        "success_std",
        "success_rate_mean",
        "success_rate_std",
        "deliveries_mean",
        "deliveries_std",
        "deliveries_early_mean",
        "deliveries_early_std",
        "deliveries_mid_mean",
        "deliveries_mid_std",
        "deliveries_late_mean",
        "deliveries_late_std",
        "block_rate_mean",
        "block_rate_std",
        "idle_rate_mean",
        "idle_rate_std",
        "pickup_rate_mean",
        "pickup_rate_std",
        "distance_traveled_mean",
        "distance_traveled_std",
        "reward_std_mean",
        "reward_std_std",
        "loss_mean",
        "loss_std",
        "FPS_mean",
        "FPS_std",
        "step_count_mean",
        "step_count_std",
        "step_time",
        "step_time_std",
        "time_to_first_delivery_mean",
        "time_to_first_delivery_std",
        "time_to_completion_mean",
        "time_to_completion_std",
        "entropy_mean",
        "entropy_std",
        "advantage_mean",
        "advantage_std",
        "value_loss_per_epoch_mean",
        "value_loss_per_epoch_std",
        "actor_loss_per_epoch_mean",
        "actor_loss_per_epoch_std",
        "entropy_per_epoch_mean",
        "entropy_per_epoch_std",
        "q_value_magnitude_mean",
        "q_value_magnitude_std",
        "UTD",
    ]

    matrix_pattern = re.compile(r".*(_lorenz_x_agent_\d+|_e\d+_a\d+|_a\d+_b\d+|_a\d+_a\d+|_grid_\d+|_cdf_\d+)$")

    def sort_keys(keys):
        ordered = []
        used = set()

        # 1. put all priority keys in order if they exist
        for k in PRIORITY_ORDER:
            if k in keys and k not in used:
                ordered.append(k)
                used.add(k)

        # 2. identify mean/std relations for remaining keys
        mean_keys = {}
        std_to_base = {}

        for k in keys:
            if k in used:
                continue
            if k.endswith("_mean"):
                base = k[:-5]
                mean_keys[base] = k
            elif k.endswith("_std"):
                base = k[:-4]
                std_to_base[k] = base

        # 3. add all means (even if no std), then matching std
        for base, mean_key in sorted(mean_keys.items()):
            if mean_key not in used:
                ordered.append(mean_key)
                used.add(mean_key)

            std_key = f"{base}_std"
            if std_key in std_to_base and std_key not in used:
                ordered.append(std_key)
                used.add(std_key)

        # 4. add std keys that have no matching mean (should be rare)
        for std_key, base in sorted(std_to_base.items()):
            if std_key not in used:
                ordered.append(std_key)
                used.add(std_key)

        # 5. split remaining into normal vs matrix/lorenz/axes
        matrix_like = []
        normal = []

        for k in keys:
            if k in used:
                continue
            if matrix_pattern.match(k):
                matrix_like.append(k)
            else:
                normal.append(k)

        # 6. add remaining normal keys (alphabetical)
        ordered.extend(sorted(normal))

        # 7. add matrix/lorenz keys last
        ordered.extend(sorted(matrix_like))

        return ordered
    return sort_keys(keys)