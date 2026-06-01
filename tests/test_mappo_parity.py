"""Gold-standard differential parity: JAX MAPPO update vs an independent PyTorch.

This is the decisive correctness test for the MAPPO port (not learning curves).
We build a PyTorch reference of the marlbase MAPPO update -- same network
(Linear -> ReLU -> GRU -> Linear), same truncated n-step target-critic returns,
same clipped-surrogate PPO loss, same 4-epoch Adam, same soft target-critic
update -- transplant *identical* weights into the flax nets, feed *one identical
synthetic batch* to both, run one update on each, and assert exact agreement of:
returns, old log-probs, per-epoch losses, and post-update parameters.

GRU bias convention: torch nn.GRU carries hidden-side biases (b_hr, b_hz, b_hn);
flax GRUCell folds them so only ir/iz/in and hn carry bias. We make the two
networks *identical functions* by pinning torch's b_hr, b_hz to zero (and keeping
them zero across the optimisation), which is exactly the flax parametrisation.
This isolates the part under test -- the returns/loss math, BPTT gradients, Adam
steps, and soft target update -- against a fully independent autograd stack.
"""

import dataclasses

import numpy as np
import pytest

jax = pytest.importorskip("jax")
torch = pytest.importorskip("torch")
jax.config.update("jax_enable_x64", True)  # match torch float64 for tight parity
import jax.numpy as jnp  # noqa: E402
import optax  # noqa: E402
from flax.core import freeze  # noqa: E402

from algorithms.config import MAPPOConfig  # noqa: E402
from algorithms.networks import ActorRNN, CriticRNN, ScannedGRU  # noqa: E402
from algorithms.mappo import mappo_update  # noqa: E402


# ----------------------------------------------------------------------------
# PyTorch reference network + update (independent transcription of marlbase)
# ----------------------------------------------------------------------------
class TorchRNNNet(torch.nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.first = torch.nn.Linear(in_dim, hidden)
        self.gru = torch.nn.GRU(hidden, hidden, num_layers=1, batch_first=False)
        self.final = torch.nn.Linear(hidden, out_dim)
        self.hidden = hidden
        self._pin_gru_rz_biases_zero()

    def _pin_gru_rz_biases_zero(self):
        # flax GRUCell has no hidden-side bias on the r,z gates; pin them to 0.
        with torch.no_grad():
            self.gru.bias_hh_l0[: 2 * self.hidden].zero_()

    def forward(self, x):  # x: [T, B, in]
        T, B, _ = x.shape
        h = torch.relu(self.first(x))
        h0 = torch.zeros(1, B, self.hidden, dtype=x.dtype)
        out, _ = self.gru(h, h0)  # no mid-sequence reset (dones only mask returns)
        return self.final(out)  # [T, B, out]


def _torch_nstep_returns(rewards, done, next_values, nsteps, gamma):
    """marlbase compute_nstep_returns, transcribed in torch. Shapes [T, E, N]."""
    T = rewards.shape[0]
    out = torch.zeros_like(rewards)
    for t_start in range(T):
        acc = torch.zeros_like(rewards[0])
        for step in range(nsteps + 1):
            t = t_start + step
            if t >= T:
                break
            elif step == nsteps:
                acc = acc + gamma ** step * next_values[t] * (1 - done[t])
            else:
                acc = acc + gamma ** step * rewards[t] * (1 - done[t])
        out[t_start] = acc
    return out


def _torch_update(actor, critic, target_critic, obs_TB, central_TB, act_TB,
                  rstd_TEN, dones_TEN, cfg, E, N):
    params = list(actor.parameters()) + list(critic.parameters())
    opt = torch.optim.Adam(params, lr=cfg.lr)
    T = obs_TB.shape[0]

    with torch.no_grad():
        vt = target_critic(central_TB).squeeze(-1).reshape(T, E, N)
        returns = _torch_nstep_returns(rstd_TEN, dones_TEN, vt, cfg.n_steps, cfg.gamma)
        old_logits = actor(obs_TB)  # [T, B, A]
        old_logp = torch.distributions.Categorical(logits=old_logits).log_prob(
            act_TB
        ).reshape(T, E, N)

    epoch_losses = []
    for _ in range(cfg.num_epochs):
        logits = actor(obs_TB)
        dist = torch.distributions.Categorical(logits=logits)
        logp = dist.log_prob(act_TB).reshape(T, E, N)
        entropy = dist.entropy().reshape(T, E, N)
        values = critic(central_TB).squeeze(-1).reshape(T, E, N)

        advantage = returns - values
        value_loss = (advantage ** 2).sum(-1).mean()
        adv = advantage.detach()
        ratio = torch.exp(logp - old_logp)
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - cfg.ppo_clip, 1.0 + cfg.ppo_clip) * adv
        actor_loss = (
            -torch.min(surr1, surr2).sum(-1) - cfg.entropy_coef * entropy.sum(-1)
        ).mean()
        loss = actor_loss + cfg.value_loss_coef * value_loss

        opt.zero_grad()
        loss.backward()
        opt.step()
        actor._pin_gru_rz_biases_zero()
        critic._pin_gru_rz_biases_zero()
        epoch_losses.append(float(loss.detach()))

    # soft target-critic update once, after the epochs
    tau = cfg.target_update_tau
    with torch.no_grad():
        for tp, sp in zip(target_critic.parameters(), critic.parameters()):
            tp.mul_(1.0 - tau).add_(tau * sp)

    return {
        "returns": returns.detach().numpy(),
        "old_logp": old_logp.detach().numpy(),
        "epoch_losses": np.array(epoch_losses),
    }


# ----------------------------------------------------------------------------
# torch -> flax weight transplant
# ----------------------------------------------------------------------------
def _gru_slices(t, H):
    return t[:H], t[H:2 * H], t[2 * H:3 * H]  # r, z, n


def _torch_to_flax_params(net, H):
    """Build a flax param dict matching ActorRNN/CriticRNN from a TorchRNNNet."""
    g = net.gru
    Wir, Wiz, Win = _gru_slices(g.weight_ih_l0.detach().numpy(), H)
    Whr, Whz, Whn = _gru_slices(g.weight_hh_l0.detach().numpy(), H)
    bir, biz, bin_ = _gru_slices(g.bias_ih_l0.detach().numpy(), H)
    _, _, bhn = _gru_slices(g.bias_hh_l0.detach().numpy(), H)  # bhr,bhz pinned 0
    return {
        "Dense_0": {
            "kernel": jnp.asarray(net.first.weight.detach().numpy().T),
            "bias": jnp.asarray(net.first.bias.detach().numpy()),
        },
        "ScannedGRU_0": {
            "GRUCell_0": {
                "ir": {"kernel": jnp.asarray(Wir.T), "bias": jnp.asarray(bir)},
                "iz": {"kernel": jnp.asarray(Wiz.T), "bias": jnp.asarray(biz)},
                "in": {"kernel": jnp.asarray(Win.T), "bias": jnp.asarray(bin_)},
                "hr": {"kernel": jnp.asarray(Whr.T)},
                "hz": {"kernel": jnp.asarray(Whz.T)},
                "hn": {"kernel": jnp.asarray(Whn.T), "bias": jnp.asarray(bhn)},
            }
        },
        "Dense_1": {
            "kernel": jnp.asarray(net.final.weight.detach().numpy().T),
            "bias": jnp.asarray(net.final.bias.detach().numpy()),
        },
    }


def _flax_vs_torch_net(flax_params, net, H, atol):
    """Assert a flax param tree equals a TorchRNNNet (post-update parity)."""
    p = flax_params["params"]
    g = net.gru
    Wir, Wiz, Win = _gru_slices(g.weight_ih_l0.detach().numpy(), H)
    Whr, Whz, Whn = _gru_slices(g.weight_hh_l0.detach().numpy(), H)
    bir, biz, bin_ = _gru_slices(g.bias_ih_l0.detach().numpy(), H)
    bhr, bhz, bhn = _gru_slices(g.bias_hh_l0.detach().numpy(), H)

    def close(a, b, name):
        a, b = np.asarray(a), np.asarray(b)
        assert np.allclose(a, b, atol=atol), f"{name}: max|d|={np.abs(a - b).max():.2e}"

    close(p["Dense_0"]["kernel"], net.first.weight.detach().numpy().T, "first.kernel")
    close(p["Dense_0"]["bias"], net.first.bias.detach().numpy(), "first.bias")
    gc = p["ScannedGRU_0"]["GRUCell_0"]
    close(gc["ir"]["kernel"], Wir.T, "ir.kernel"); close(gc["ir"]["bias"], bir, "ir.bias")
    close(gc["iz"]["kernel"], Wiz.T, "iz.kernel"); close(gc["iz"]["bias"], biz, "iz.bias")
    close(gc["in"]["kernel"], Win.T, "in.kernel"); close(gc["in"]["bias"], bin_, "in.bias")
    close(gc["hr"]["kernel"], Whr.T, "hr.kernel")
    close(gc["hz"]["kernel"], Whz.T, "hz.kernel")
    close(gc["hn"]["kernel"], Whn.T, "hn.kernel"); close(gc["hn"]["bias"], bhn, "hn.bias")
    close(p["Dense_1"]["kernel"], net.final.weight.detach().numpy().T, "final.kernel")
    close(p["Dense_1"]["bias"], net.final.bias.detach().numpy(), "final.bias")
    # the pinned r,z hidden biases must remain zero (flax has no counterpart)
    assert np.allclose(bhr, 0.0) and np.allclose(bhz, 0.0), "torch b_hr/b_hz drifted"


def test_mappo_update_matches_pytorch():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    T, E, N, obs_dim, A, H = 12, 3, 2, 10, 5, 8
    B = E * N
    central_dim = obs_dim * N
    cfg = dataclasses.replace(MAPPOConfig(), hidden_dim=H)

    # reference torch nets (actor over obs, critic over central obs)
    t_actor = TorchRNNNet(obs_dim, H, A).double()
    t_critic = TorchRNNNet(central_dim, H, 1).double()
    t_target = TorchRNNNet(central_dim, H, 1).double()
    t_target.load_state_dict(t_critic.state_dict())

    # one identical synthetic batch
    obs = rng.standard_normal((T, B, obs_dim))
    central = rng.standard_normal((T, B, central_dim))
    act = rng.integers(0, A, size=(T, B))
    rstd = rng.standard_normal((T, E, N))
    dones = np.zeros((T, E, N))

    # ---- flax side: transplant identical weights, run mappo_update ----
    actor = ActorRNN(A, H)
    critic = CriticRNN(H)
    f_params = {
        "actor": freeze({"params": _torch_to_flax_params(t_actor, H)}),
        "critic": freeze({"params": _torch_to_flax_params(t_critic, H)}),
    }
    f_target = freeze({"params": _torch_to_flax_params(t_target, H)})
    tx = optax.adam(cfg.lr)
    opt_state = tx.init(f_params)
    batch = {
        "obs_flat_T": jnp.asarray(obs),
        "act_flat_T": jnp.asarray(act),
        "central_T": jnp.asarray(central),
        "rstd_TEN": jnp.asarray(rstd),
        "dones_TEN": jnp.asarray(dones),
    }
    f_params2, f_target2, _, diag = mappo_update(
        actor, critic, tx, cfg, f_params, f_target, opt_state, batch
    )

    # ---- torch side: same batch, same update ----
    ref = _torch_update(
        t_actor, t_critic, t_target,
        torch.tensor(obs), torch.tensor(central), torch.tensor(act),
        torch.tensor(rstd), torch.tensor(dones), cfg, E, N,
    )

    # ---- pre-update agreement (returns, old log-probs) ----
    assert np.allclose(np.array(diag["returns"]), ref["returns"], atol=1e-6), \
        np.abs(np.array(diag["returns"]) - ref["returns"]).max()
    assert np.allclose(np.array(diag["old_logp"]), ref["old_logp"], atol=1e-6), \
        np.abs(np.array(diag["old_logp"]) - ref["old_logp"]).max()

    # ---- per-epoch loss agreement ----
    assert np.allclose(np.array(diag["epoch_loss"]), ref["epoch_losses"], atol=1e-5), \
        (np.array(diag["epoch_loss"]), ref["epoch_losses"])

    # ---- post-update parameter agreement (the decisive check) ----
    _flax_vs_torch_net(f_params2["actor"], t_actor, H, atol=1e-5)
    _flax_vs_torch_net(f_params2["critic"], t_critic, H, atol=1e-5)
    _flax_vs_torch_net(f_target2, t_target, H, atol=1e-5)
