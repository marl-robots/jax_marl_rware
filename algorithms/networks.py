"""Flax networks for MAPPO: shared-parameter GRU actor + centralised critic.

Replicates marlbase utils/models.py:RNNNetwork exactly:

    Linear(in, hidden) -> ReLU -> GRU(hidden, hidden) -> Linear(hidden, out)

Orthogonal init (gain sqrt(2)) is applied ONLY to the final Linear; the input
Dense and the GRU keep Flax defaults. Hidden state is carried across the
episode and reset (to zeros) at episode boundaries via the `dones` signal.

Parameter sharing = a single param tree; agents are folded into the batch axis.
"""

from __future__ import annotations

import functools

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp


class ScannedGRU(nn.Module):
    """GRUCell scanned over the leading time axis, with per-step hidden reset."""

    hidden_dim: int = 128

    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        split_rngs={"params": False},
        in_axes=0,
        out_axes=0,
    )
    @nn.compact
    def __call__(self, carry, x):
        ins, resets = x  # ins: [B, feat], resets: [B] bool (episode start)
        carry = jnp.where(
            resets[:, None],
            self.initialize_carry(ins.shape[0], self.hidden_dim),
            carry,
        )
        new_carry, out = nn.GRUCell(features=self.hidden_dim)(carry, ins)
        return new_carry, out

    @staticmethod
    def initialize_carry(batch_size: int, hidden_dim: int):
        return jnp.zeros((batch_size, hidden_dim))


class ActorRNN(nn.Module):
    """Shared actor: obs -> Dense -> ReLU -> GRU -> Dense(logits) -> Categorical."""

    num_actions: int
    hidden_dim: int = 128
    orthogonal_gain: float = 2.0 ** 0.5

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x  # obs: [T, B, obs_dim], dones: [T, B]
        embed = nn.Dense(self.hidden_dim)(obs)  # default init (lecun_normal)
        embed = nn.relu(embed)
        hidden, gru_out = ScannedGRU(self.hidden_dim)(hidden, (embed, dones))
        logits = nn.Dense(
            self.num_actions,
            kernel_init=nn.initializers.orthogonal(self.orthogonal_gain),
            bias_init=nn.initializers.zeros,
        )(gru_out)
        return hidden, distrax.Categorical(logits=logits)


class CriticRNN(nn.Module):
    """Shared centralised critic: global obs -> Dense -> ReLU -> GRU -> Dense(1)."""

    hidden_dim: int = 128
    orthogonal_gain: float = 2.0 ** 0.5

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x  # obs: [T, B, central_dim], dones: [T, B]
        embed = nn.Dense(self.hidden_dim)(obs)
        embed = nn.relu(embed)
        hidden, gru_out = ScannedGRU(self.hidden_dim)(hidden, (embed, dones))
        value = nn.Dense(
            1,
            kernel_init=nn.initializers.orthogonal(self.orthogonal_gain),
            bias_init=nn.initializers.zeros,
        )(gru_out)
        return hidden, jnp.squeeze(value, axis=-1)
