"""Flax Q-network for the value-based family, mirroring epymarl RNNAgent exactly.

epymarl src/modules/agents/rnn_agent.py:RNNAgent.forward is:

    x = relu(fc1(inputs))
    h = GRUCell(x, h_in)            if use_rnn   (recurrent: IQL default)
    h = relu(fc2_linear(x))         if not use_rnn  (feedforward: VDN/QMIX)
    q = fc_out(h)

i.e.  Linear(in,H) -> ReLU -> [GRUCell(H,H) | Linear(H,H)->ReLU] -> Linear(H,A),
with DEFAULT init on every layer (no orthogonal). We reuse ScannedGRU from
networks.py for the recurrent path; the hidden state inits to zeros at the
episode start and is NOT reset mid-sequence (each batch element is one full
episode), matching how the q_learner unrolls mac_out over max_seq_length.

The output is RAW Q-values (no distribution), so the learner can gather the
chosen-action Q and take the max for double-Q targets.
"""

from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp

from .networks import ScannedGRU


class QNetwork(nn.Module):
    """Per-agent Q-network; agents are folded into the batch axis (shared params).

    use_rnn=True  -> recurrent (IQL):   Dense -> ReLU -> GRU -> Dense
    use_rnn=False -> feedforward (VDN/QMIX): Dense -> ReLU -> Dense -> ReLU -> Dense
    """

    num_actions: int
    hidden_dim: int = 64
    use_rnn: bool = False

    @nn.compact
    def __call__(self, hidden, x):
        obs, resets = x  # obs: [T, B, in_dim], resets: [T, B] (episode-start mask)
        embed = nn.relu(nn.Dense(self.hidden_dim)(obs))
        if self.use_rnn:
            hidden, feat = ScannedGRU(self.hidden_dim)(hidden, (embed, resets))
        else:
            feat = nn.relu(nn.Dense(self.hidden_dim)(embed))
        q = nn.Dense(self.num_actions)(feat)  # [T, B, A]
        return hidden, q
