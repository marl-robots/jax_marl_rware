"""Unit tests for MAPPO building blocks (no training).

1. compute_nstep_returns vs an independent NumPy port of the marlbase loop.
2. Network shapes + orthogonal-init sanity for the shared GRU actor/critic.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)  # float64 so parity checks reflect math, not rounding
import jax.numpy as jnp  # noqa: E402

from algorithms.returns import compute_nstep_returns  # noqa: E402
from algorithms.networks import ActorRNN, CriticRNN, ScannedGRU  # noqa: E402
from algorithms.mappo import _welford_standardise, ppo_losses  # noqa: E402


def _np_nstep_returns(rewards, done, next_values, nsteps, gamma):
    """Direct transcription of marlbase compute_nstep_returns (the reference)."""
    ep_length = rewards.shape[0]
    out = np.zeros_like(rewards)
    for t_start in range(ep_length):
        acc = np.zeros_like(rewards[0])
        for step in range(nsteps + 1):
            t = t_start + step
            if t >= ep_length:
                break
            elif step == nsteps:
                acc = acc + gamma**step * next_values[t] * (1 - done[t])
            else:
                acc = acc + gamma**step * rewards[t] * (1 - done[t])
        out[t_start] = acc
    return out


@pytest.mark.parametrize("nsteps", [1, 3, 5])
@pytest.mark.parametrize("with_dones", [False, True])
def test_nstep_returns_match_reference(nsteps, with_dones):
    rng = np.random.default_rng(0)
    T, B, N = 23, 4, 3
    rewards = rng.standard_normal((T, B, N)).astype(np.float32)
    next_values = rng.standard_normal((T, B, N)).astype(np.float32)
    if with_dones:
        done = (rng.random((T, B, N)) < 0.1).astype(np.float32)
    else:
        done = np.zeros((T, B, N), np.float32)
    gamma = 0.99

    ref = _np_nstep_returns(rewards, done, next_values, nsteps, gamma)
    got = np.array(
        compute_nstep_returns(
            jnp.asarray(rewards), jnp.asarray(done), jnp.asarray(next_values),
            nsteps, gamma,
        )
    )
    assert np.allclose(got, ref, atol=1e-5), np.abs(got - ref).max()


def test_actor_critic_shapes_and_init():
    T, B, obs_dim, central_dim, num_actions, hidden = 7, 5, 71, 71 * 4, 5, 128
    actor = ActorRNN(num_actions=num_actions, hidden_dim=hidden)
    critic = CriticRNN(hidden_dim=hidden)

    key = jax.random.PRNGKey(0)
    obs = jnp.zeros((T, B, obs_dim))
    cobs = jnp.zeros((T, B, central_dim))
    dones = jnp.zeros((T, B))
    h = ScannedGRU.initialize_carry(B, hidden)

    a_params = actor.init(key, h, (obs, dones))
    c_params = critic.init(key, h, (cobs, dones))

    _, dist = actor.apply(a_params, h, (obs, dones))
    _, value = critic.apply(c_params, h, (cobs, dones))

    assert dist.logits.shape == (T, B, num_actions)
    assert value.shape == (T, B)

    # final-layer kernel should be orthogonal (gain sqrt(2)): columns ~ orthonormal*gain
    fk = a_params["params"]["Dense_1"]["kernel"]  # [hidden, num_actions]
    gram = np.array(fk.T @ fk)
    expected = 2.0 * np.eye(num_actions)  # gain^2 = 2
    assert np.allclose(gram, expected, atol=1e-3), gram


def test_rollout_stepwise_equals_batched_apply():
    """The rollout applies the actor one step at a time carrying the GRU hidden;
    the update applies it over the whole sequence. With hidden init = zeros and
    no mid-sequence reset, both must yield identical logits — otherwise the
    behaviour policy that collected the data differs from the one being updated.
    """
    T, B, obs_dim, A, H = 9, 6, 10, 5, 8
    actor = ActorRNN(A, H)
    key = jax.random.PRNGKey(3)
    obs_seq = jax.random.normal(jax.random.PRNGKey(4), (T, B, obs_dim), dtype=jnp.float64)
    zeros1 = jnp.zeros((1, B))
    params = actor.init(key, ScannedGRU.initialize_carry(B, H),
                        (jnp.zeros((1, B, obs_dim)), zeros1))

    # stepwise (rollout style): carry hidden from zeros
    h = ScannedGRU.initialize_carry(B, H)
    step_logits = []
    for t in range(T):
        h, dist = actor.apply(params, h, (obs_seq[t][None], zeros1))
        step_logits.append(dist.logits[0])
    step_logits = jnp.stack(step_logits)  # [T, B, A]

    # batched (update style): whole sequence at once
    _, dist_full = actor.apply(params, ScannedGRU.initialize_carry(B, H),
                               (obs_seq, jnp.zeros((T, B))))
    batched_logits = dist_full.logits  # [T, B, A]

    assert np.allclose(np.array(step_logits), np.array(batched_logits), atol=1e-9), \
        np.abs(np.array(step_logits) - np.array(batched_logits)).max()


def _np_standardise(rewards):
    """Plain-Python transcription of marlbase StandardiseReward (weight=1)."""
    T = rewards.shape[0]
    sumw = np.zeros(rewards.shape[1:], np.float64)
    wmean = np.zeros_like(sumw)
    m2 = np.zeros_like(sumw)
    n = 0.0
    out = np.zeros_like(rewards)
    for t in range(T):
        r = rewards[t].astype(np.float64)
        q = r - wmean
        temp_sumw = sumw + 1.0
        rr = q / temp_sumw
        wmean = wmean + rr
        m2 = m2 + q * rr * sumw
        sumw = temp_sumw
        n = n + 1.0
        var = (m2 * n) / (sumw * (n - 1.0) + 1e-12)
        std = (r - wmean) / (np.sqrt(var) + 1e-6)
        out[t] = r if n <= 1.0 else std
    return out


def test_welford_standardise_matches_reference():
    rng = np.random.default_rng(1)
    T, E, N = 40, 3, 4
    rewards = rng.standard_normal((T, E, N)).astype(np.float32)

    state = (jnp.zeros((E, N)), jnp.zeros((E, N)), jnp.zeros((E, N)), jnp.array(0.0))

    def step(carry, r):
        carry, out = _welford_standardise(carry, r)
        return carry, out

    _, got = jax.lax.scan(step, state, jnp.asarray(rewards))
    ref = _np_standardise(rewards)
    assert np.allclose(np.array(got), ref, atol=1e-4), np.abs(np.array(got) - ref).max()


def test_ppo_losses_match_reference():
    rng = np.random.default_rng(2)
    T, E, N = 16, 3, 4
    returns = rng.standard_normal((T, E, N)).astype(np.float32)
    values = rng.standard_normal((T, E, N)).astype(np.float32)
    logp = rng.standard_normal((T, E, N)).astype(np.float32) * 0.1
    old_logp = rng.standard_normal((T, E, N)).astype(np.float32) * 0.1
    entropy = rng.random((T, E, N)).astype(np.float32)
    ppo_clip, ec, vlc = 0.2, 1e-3, 0.5

    adv = returns - values
    value_loss = (adv ** 2).sum(-1).mean()
    ratio = np.exp(logp - old_logp)
    surr1 = ratio * adv
    surr2 = np.clip(ratio, 1.0 - ppo_clip, 1.0 + ppo_clip) * adv
    actor_loss = (-np.minimum(surr1, surr2).sum(-1) - ec * entropy.sum(-1)).mean()
    ref_loss = actor_loss + vlc * value_loss

    loss, (al, vl, ent) = ppo_losses(
        jnp.asarray(returns), jnp.asarray(values), jnp.asarray(logp),
        jnp.asarray(old_logp), jnp.asarray(entropy),
        ppo_clip=ppo_clip, entropy_coef=ec, value_loss_coef=vlc,
    )
    assert np.allclose(float(loss), ref_loss, atol=1e-4), (float(loss), ref_loss)
    assert np.allclose(float(al), actor_loss, atol=1e-4)
    assert np.allclose(float(vl), value_loss, atol=1e-4)
    assert np.allclose(float(ent), entropy.mean(), atol=1e-4)
