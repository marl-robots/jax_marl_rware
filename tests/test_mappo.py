"""Unit tests for MAPPO building blocks (no training).

1. compute_nstep_returns vs an independent NumPy port of the marlbase loop.
2. Network shapes + orthogonal-init sanity for the shared GRU actor/critic.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from algorithms.returns import compute_nstep_returns  # noqa: E402
from algorithms.networks import ActorRNN, CriticRNN, ScannedGRU  # noqa: E402


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
