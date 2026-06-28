"""Gradient parity for the SEAC loss vs an independent PyTorch transcription.

The decisive correctness test for the SEAC port's loss math (the training
*regime* — short rollouts, RMSProp+clip in the canonical repo — is a separate,
documented question; see algorithms/seac.py docstring and README status).

The torch side transcribes the canonical update structure of uoe-agents/seac
(seac/a2c.py) as N explicit per-agent loops:

    own:     policy = -(adv.detach() * logp).mean();  value = adv.pow(2).mean()
    shared:  IS     = (logp_i - logp_k).exp().detach()           (per element)
             policy += seac_coef * (-(IS * logp_i * adv_i.detach())).mean()
             value  += seac_coef * (IS * adv_i.pow(2)).mean()
    loss    = sum_i [ policy_i + vcoef * value_i - ecoef * entropy_i(own) ]

(The canonical code adds 1e-7 to the denominator prob as a numerical guard;
mathematically IS is the plain ratio, which is what both sides compute here.)

Our JAX side computes the same thing as one N x N IS-weighted tensor with a
coefficient matrix (1 on the diagonal, seac_coef off it). We transplant
identical per-agent FC weights into both stacks, feed one identical synthetic
batch, and assert agreement of the loss and of every parameter gradient in
float64. The FC network path is used so this test isolates the SEAC-specific
math (the GRU/BPTT path is already covered by tests/test_mappo_parity.py).
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
torch = pytest.importorskip("torch")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from algorithms.config import MAPPOConfig  # noqa: E402
from algorithms.networks import ActorRNN, CriticRNN  # noqa: E402
from algorithms.seac import seac_loss_from_batch  # noqa: E402

# small but non-degenerate dims
N, T, E, OBS, A, H = 3, 6, 2, 5, 4, 8
torch.set_default_dtype(torch.float64)


def _torch_nstep_returns(rewards, nsteps, gamma, next_values):
    """marlbase compute_nstep_returns transcription (dones all zero here)."""
    Tn = rewards.shape[0]
    out = torch.zeros_like(rewards)
    for t_start in range(Tn):
        acc = torch.zeros_like(rewards[0])
        for step in range(nsteps + 1):
            t = t_start + step
            if t >= Tn:
                break
            elif step == nsteps:
                acc = acc + gamma ** step * next_values[t]
            else:
                acc = acc + gamma ** step * rewards[t]
        out[t_start] = acc
    return out


class _TorchFC(torch.nn.Module):
    """ActorRNN/CriticRNN with use_rnn=False: Dense-ReLU-Dense-ReLU-Dense."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.l0 = torch.nn.Linear(in_dim, H)
        self.l1 = torch.nn.Linear(H, H)
        self.l2 = torch.nn.Linear(H, out_dim)

    def forward(self, x):
        h = torch.relu(self.l0(x))
        h = torch.relu(self.l1(h))
        return self.l2(h)


def _transplant(torch_net: _TorchFC, jax_params, agent: int) -> None:
    """Copy agent `agent`'s flax FC params (leading agent axis) into torch."""
    p = jax_params["params"]
    layers = sorted(p.keys())  # Dense_0, Dense_1, Dense_2
    assert len(layers) == 3, layers
    with torch.no_grad():
        for name, lin in zip(layers, (torch_net.l0, torch_net.l1, torch_net.l2)):
            k = np.asarray(p[name]["kernel"][agent])  # [in, out]
            b = np.asarray(p[name]["bias"][agent])
            lin.weight.copy_(torch.from_numpy(k.T))
            lin.bias.copy_(torch.from_numpy(b))


@pytest.mark.parametrize("seac_coef", [1.0, 0.5])
def test_seac_loss_and_grad_parity(seac_coef):
    cfg = MAPPOConfig(n_agents=N, hidden_dim=H, use_rnn=False,
                      seac_coef=seac_coef, is_clip=0.0)
    actor = ActorRNN(A, H, cfg.orthogonal_gain, use_rnn=False)
    critic = CriticRNN(H, cfg.orthogonal_gain, use_rnn=False)

    # ---- init N independent param trees (leading agent axis), like seac._setup
    key = jax.random.PRNGKey(0)
    ka, kc, kobs, kact, krew = jax.random.split(key, 5)
    dummy_o = jnp.zeros((1, 1, OBS))
    dummy_r = jnp.zeros((1, 1))
    h1 = jnp.zeros((1, H))
    actor_params = jax.vmap(
        lambda k: actor.init(k, h1, (dummy_o, dummy_r)))(jax.random.split(ka, N))
    critic_params = jax.vmap(
        lambda k: critic.init(k, h1, (dummy_o, dummy_r)))(jax.random.split(kc, N))
    # flax initializes float32 params even under x64; promote for tight parity
    to64 = lambda t: jax.tree_util.tree_map(
        lambda x: x.astype(jnp.float64), t)
    actor_params, critic_params = to64(actor_params), to64(critic_params)

    # ---- one synthetic batch
    obs_t = jax.random.normal(kobs, (T, E, N, OBS))
    act_t = jax.random.randint(kact, (T, E, N), 0, A)
    rstd_t = 0.3 * jax.random.normal(krew, (T, E, N))

    eye = jnp.eye(N)
    coefmat = eye + (1.0 - eye) * cfg.seac_coef

    (j_loss, j_aux), j_grads = jax.value_and_grad(
        lambda p: seac_loss_from_batch(actor, critic, cfg, coefmat, p,
                                       obs_t, act_t, rstd_t),
        has_aux=True)((actor_params, critic_params))

    # ---- torch reference: explicit canonical per-agent loops
    obs_T = torch.from_numpy(np.asarray(obs_t))      # [T,E,N,OBS]
    act_T = torch.from_numpy(np.asarray(act_t))      # [T,E,N]
    rew_T = torch.from_numpy(np.asarray(rstd_t))     # [T,E,N]

    actors = [_TorchFC(OBS, A) for _ in range(N)]
    critics = [_TorchFC(OBS, 1) for _ in range(N)]
    for i in range(N):
        _transplant(actors[i], actor_params, i)
        _transplant(critics[i], critic_params, i)

    # returns per data-owner k, bootstrapped from k's OWN critic (detached)
    returns = []
    for k in range(N):
        v_kk = critics[k](obs_T[:, :, k]).squeeze(-1).detach()   # [T,E]
        returns.append(_torch_nstep_returns(
            rew_T[:, :, k], cfg.n_steps, cfg.gamma, v_kk))
    total = 0.0
    for i in range(N):
        # entropy on own data only
        own_logits = actors[i](obs_T[:, :, i])
        own_dist = torch.distributions.Categorical(logits=own_logits)
        entropy_i = own_dist.entropy().mean()
        policy_i, value_i = 0.0, 0.0
        for k in range(N):
            logits_ik = actors[i](obs_T[:, :, k])
            dist_ik = torch.distributions.Categorical(logits=logits_ik)
            logp_ik = dist_ik.log_prob(act_T[:, :, k])           # [T,E]
            v_ik = critics[i](obs_T[:, :, k]).squeeze(-1)
            adv_ik = returns[k] - v_ik
            if i == k:
                policy_i = policy_i + (-(adv_ik.detach() * logp_ik).mean())
                value_i = value_i + adv_ik.pow(2).mean()
            else:
                logits_kk = actors[k](obs_T[:, :, k])
                logp_kk = torch.distributions.Categorical(
                    logits=logits_kk).log_prob(act_T[:, :, k])
                IS = (logp_ik - logp_kk).exp().detach()
                policy_i = policy_i + cfg.seac_coef * (
                    -(IS * logp_ik * adv_ik.detach())).mean()
                value_i = value_i + cfg.seac_coef * (IS * adv_ik.pow(2)).mean()
        total = total + (policy_i
                         + cfg.value_loss_coef * value_i
                         - cfg.entropy_coef * entropy_i)
    total.backward()

    # ---- loss parity
    np.testing.assert_allclose(float(j_loss), float(total), rtol=1e-10)

    # ---- gradient parity, every parameter of every agent
    jga, jgc = j_grads
    for i in range(N):
        for params_tree, nets in ((jga, actors), (jgc, critics)):
            p = params_tree["params"]
            layers = sorted(p.keys())
            for name, lin in zip(layers,
                                 (nets[i].l0, nets[i].l1, nets[i].l2)):
                jk = np.asarray(p[name]["kernel"][i])
                jb = np.asarray(p[name]["bias"][i])
                tk = lin.weight.grad.numpy().T
                tb = lin.bias.grad.numpy()
                np.testing.assert_allclose(jk, tk, rtol=1e-8, atol=1e-12,
                                           err_msg=f"agent{i} {name} kernel")
                np.testing.assert_allclose(jb, tb, rtol=1e-8, atol=1e-12,
                                           err_msg=f"agent{i} {name} bias")
