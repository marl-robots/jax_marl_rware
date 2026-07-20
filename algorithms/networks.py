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
    """Shared actor: obs -> Dense -> ReLU -> [GRU | Dense->ReLU] -> Dense(logits).

    use_rnn=True  (default): Dense -> ReLU -> GRU -> Dense  (recurrent, marlbase RNN).
    use_rnn=False: Dense -> ReLU -> Dense -> ReLU -> Dense  (feedforward; the GRU
    is replaced by a second hidden layer, no recurrence). `hidden` is passed
    through untouched in the FC case so the carry plumbing is unchanged.
    """

    num_actions: int
    hidden_dim: int = 128
    orthogonal_gain: float = 2.0 ** 0.5
    use_rnn: bool = True
    use_cnn: bool = False        # spatial conv over the 3x3 sensor window (sr=1)
    cnn_filters: int = 32

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x  # obs: [T, B, obs_dim], dones: [T, B]
        ortho = nn.initializers.orthogonal(self.orthogonal_gain)
        if self.use_cnn:
            # obs layout (sr=1): [8 self] + [9 cells x 7]; the 9 cells are a 3x3
            # grid in row-major (dy,dx) order -> conv over it, keep self separate.
            self_bits = obs[..., :8]
            cells = obs[..., 8:].reshape(*obs.shape[:-1], 3, 3, 7)
            h = nn.relu(nn.Conv(self.cnn_filters, (2, 2), padding="SAME",
                                kernel_init=ortho)(cells))
            h = nn.relu(nn.Conv(self.cnn_filters, (2, 2), padding="SAME",
                                kernel_init=ortho)(h))
            h = h.reshape(*obs.shape[:-1], -1)            # flatten 3x3xF
            cat = jnp.concatenate([self_bits, h], axis=-1)
            feat = nn.relu(nn.Dense(self.hidden_dim, kernel_init=ortho)(cat))
        elif self.use_rnn:
            # marlbase RNNNetwork: default init on first layer + GRU, orthogonal
            # only on the final layer.
            embed = nn.relu(nn.Dense(self.hidden_dim)(obs))  # default (lecun_normal)
            hidden, feat = ScannedGRU(self.hidden_dim)(hidden, (embed, dones))
        else:
            # marlbase FCNetwork: orthogonal (gain sqrt(2)) on EVERY layer.
            embed = nn.relu(nn.Dense(self.hidden_dim, kernel_init=ortho)(obs))
            feat = nn.relu(nn.Dense(self.hidden_dim, kernel_init=ortho)(embed))
        logits = nn.Dense(
            self.num_actions, kernel_init=ortho, bias_init=nn.initializers.zeros,
        )(feat)
        return hidden, distrax.Categorical(logits=logits)


class CriticRNN(nn.Module):
    """Shared critic: obs -> Dense -> ReLU -> [GRU | Dense->ReLU] -> Dense(1).

    use_rnn toggles recurrent vs feedforward exactly as in ActorRNN.
    """

    hidden_dim: int = 128
    orthogonal_gain: float = 2.0 ** 0.5
    use_rnn: bool = True

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x  # obs: [T, B, central_dim], dones: [T, B]
        ortho = nn.initializers.orthogonal(self.orthogonal_gain)
        if self.use_rnn:
            embed = nn.relu(nn.Dense(self.hidden_dim)(obs))  # default (lecun_normal)
            hidden, feat = ScannedGRU(self.hidden_dim)(hidden, (embed, dones))
        else:
            embed = nn.relu(nn.Dense(self.hidden_dim, kernel_init=ortho)(obs))
            feat = nn.relu(nn.Dense(self.hidden_dim, kernel_init=ortho)(embed))
        value = nn.Dense(
            1, kernel_init=ortho, bias_init=nn.initializers.zeros,
        )(feat)
        return hidden, jnp.squeeze(value, axis=-1)
