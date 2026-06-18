"""Gold-standard differential parity: JAX IQL update vs an independent PyTorch.

Decisive correctness test for the value-based base (Layer 1: IQL), the same
methodology as test_mappo_parity.py. We transcribe epymarl's RNNAgent +
QLearner.train (mixer=None) independently in PyTorch -- same recurrent net
(Linear -> ReLU -> GRUCell -> Linear), same RunningMeanStd reward
standardisation, same double-Q 1-step TD target, same mask, same
Adam-with-global-norm-clip step -- transplant *identical* weights into the flax
QNetwork, feed *one identical synthetic batch* to both, run one update each, and
assert exact agreement of: chosen-action Q, TD targets, loss, and post-update
parameters.

GRU bias convention (identical to the MAPPO test): torch GRUCell carries
hidden-side biases b_hr,b_hz,b_hn; flax GRUCell folds r,z so only ir/iz/in and
hn carry bias. We pin torch b_hr,b_hz to zero AND zero their gradient (flax has
no such parameter, so its grad is identically zero) -- otherwise they would
inflate torch's global grad-norm and break the clip-path parity. This isolates
the part under test: the TD math, BPTT gradients, the clip+Adam step.
"""

import dataclasses

import numpy as np
import pytest

jax = pytest.importorskip("jax")
torch = pytest.importorskip("torch")
jax.config.update("jax_enable_x64", True)  # match torch float64 for tight parity
import jax.numpy as jnp  # noqa: E402
from flax.core import freeze  # noqa: E402

from algorithms.dqn_config import DQNConfig  # noqa: E402
from algorithms.dqn_networks import QNetwork  # noqa: E402
from algorithms.dqn import (  # noqa: E402
    iql_update, emax_update, make_optimizer, rms_init,
)


# ----------------------------------------------------------------------------
# PyTorch reference: RNNAgent (use_rnn=True) + QLearner.train (mixer=None)
# ----------------------------------------------------------------------------
class TorchRNNAgent(torch.nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_dim, hidden)
        self.gru = torch.nn.GRUCell(hidden, hidden)
        self.fc2 = torch.nn.Linear(hidden, out_dim)
        self.hidden = hidden
        self._pin()

    def _pin(self):
        # flax GRUCell has no hidden-side bias on the r,z gates; pin to 0.
        with torch.no_grad():
            self.gru.bias_hh[: 2 * self.hidden].zero_()

    def zero_pinned_grads(self):
        # flax has no b_hr,b_hz, so their grad is identically zero; match it.
        if self.gru.bias_hh.grad is not None:
            self.gru.bias_hh.grad[: 2 * self.hidden].zero_()

    def forward_seq(self, obs_TB):  # [T, B, in] -> [T, B, A]
        T, B, _ = obs_TB.shape
        h = torch.zeros(B, self.hidden, dtype=obs_TB.dtype)
        outs = []
        for t in range(T):
            x = torch.relu(self.fc1(obs_TB[t]))
            h = self.gru(x, h)
            outs.append(self.fc2(h))
        return torch.stack(outs, dim=0)


class TorchRunningMeanStd:
    """epymarl standarize_stream.py:RunningMeanStd, transcribed."""

    def __init__(self, shape, epsilon=1e-4):
        self.mean = torch.zeros(shape, dtype=torch.float64)
        self.var = torch.ones(shape, dtype=torch.float64)
        self.count = epsilon

    def update(self, arr):
        arr = arr.reshape(-1, arr.size(-1))
        batch_mean = arr.mean(dim=0)
        batch_var = arr.var(dim=0)  # unbiased (ddof=1), torch default
        batch_count = arr.shape[0]
        delta = batch_mean - self.mean
        tot = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + torch.square(delta) * self.count * batch_count / tot
        self.mean, self.var, self.count = new_mean, m_2 / tot, tot


def _torch_iql_update(agent, target_agent, obs_TB, actions_TEN, reward_TE,
                      terminated_TE, filled_TE, cfg, E, N):
    A = agent.fc2.out_features
    opt = torch.optim.Adam(agent.parameters(), lr=cfg.lr, eps=cfg.adam_eps)
    T = obs_TB.shape[0]

    mac_out = agent.forward_seq(obs_TB).reshape(T, E, N, A)
    with torch.no_grad():
        target_mac_out = target_agent.forward_seq(obs_TB).reshape(T, E, N, A)
    target_next = target_mac_out[1:]

    # double-Q: argmax of online Q at t+1, valued by the target net
    cur_max = mac_out.detach()[1:].argmax(dim=-1, keepdim=True)
    target_max = torch.gather(target_next, -1, cur_max).squeeze(-1)  # [T-1,E,N]

    r = reward_TE[: T - 1].clone()  # [T-1, E]
    if cfg.standardise_rewards:
        rms = TorchRunningMeanStd((1,))
        rms.update(r.reshape(-1, 1))
        r = (r - rms.mean[0]) / torch.sqrt(rms.var[0])
    r_EN = r.unsqueeze(-1).expand(-1, -1, N)

    term_used = terminated_TE[: T - 1]
    targets = r_EN + cfg.gamma * (1 - term_used).unsqueeze(-1) * target_max.detach()

    mask = filled_TE[: T - 1].clone()
    mask[1:] = mask[1:] * (1 - terminated_TE[: T - 2])
    mask_EN = mask.unsqueeze(-1).expand(-1, -1, N)

    chosen = torch.gather(
        mac_out[: T - 1], -1, actions_TEN[: T - 1].unsqueeze(-1)
    ).squeeze(-1)
    td = chosen - targets.detach()
    loss = ((td * mask_EN) ** 2).sum() / mask_EN.sum()

    opt.zero_grad()
    loss.backward()
    agent.zero_pinned_grads()
    torch.nn.utils.clip_grad_norm_(agent.parameters(), cfg.grad_norm_clip)
    opt.step()
    agent._pin()
    return {
        "chosen": chosen.detach().numpy(),
        "targets": targets.detach().numpy(),
        "loss": float(loss.detach()),
    }


# ----------------------------------------------------------------------------
# torch -> flax weight transplant (GRUCell variant; mirrors the MAPPO test)
# ----------------------------------------------------------------------------
def _slices(t, H):
    return t[:H], t[H:2 * H], t[2 * H:3 * H]  # r, z, n


def _to_flax(agent, H):
    g = agent.gru
    Wir, Wiz, Win = _slices(g.weight_ih.detach().numpy(), H)
    Whr, Whz, Whn = _slices(g.weight_hh.detach().numpy(), H)
    bir, biz, bin_ = _slices(g.bias_ih.detach().numpy(), H)
    _, _, bhn = _slices(g.bias_hh.detach().numpy(), H)  # b_hr,b_hz pinned 0
    return {
        "Dense_0": {
            "kernel": jnp.asarray(agent.fc1.weight.detach().numpy().T),
            "bias": jnp.asarray(agent.fc1.bias.detach().numpy()),
        },
        "ScannedGRU_0": {"GRUCell_0": {
            "ir": {"kernel": jnp.asarray(Wir.T), "bias": jnp.asarray(bir)},
            "iz": {"kernel": jnp.asarray(Wiz.T), "bias": jnp.asarray(biz)},
            "in": {"kernel": jnp.asarray(Win.T), "bias": jnp.asarray(bin_)},
            "hr": {"kernel": jnp.asarray(Whr.T)},
            "hz": {"kernel": jnp.asarray(Whz.T)},
            "hn": {"kernel": jnp.asarray(Whn.T), "bias": jnp.asarray(bhn)},
        }},
        "Dense_1": {
            "kernel": jnp.asarray(agent.fc2.weight.detach().numpy().T),
            "bias": jnp.asarray(agent.fc2.bias.detach().numpy()),
        },
    }


def _assert_params_match(flax_params, agent, H, atol):
    p = flax_params["params"]
    g = agent.gru
    Wir, Wiz, Win = _slices(g.weight_ih.detach().numpy(), H)
    Whr, Whz, Whn = _slices(g.weight_hh.detach().numpy(), H)
    bir, biz, bin_ = _slices(g.bias_ih.detach().numpy(), H)
    bhr, bhz, bhn = _slices(g.bias_hh.detach().numpy(), H)

    def close(a, b, name):
        a, b = np.asarray(a), np.asarray(b)
        assert np.allclose(a, b, atol=atol), f"{name}: max|d|={np.abs(a - b).max():.2e}"

    close(p["Dense_0"]["kernel"], agent.fc1.weight.detach().numpy().T, "fc1.kernel")
    close(p["Dense_0"]["bias"], agent.fc1.bias.detach().numpy(), "fc1.bias")
    gc = p["ScannedGRU_0"]["GRUCell_0"]
    close(gc["ir"]["kernel"], Wir.T, "ir.k"); close(gc["ir"]["bias"], bir, "ir.b")
    close(gc["iz"]["kernel"], Wiz.T, "iz.k"); close(gc["iz"]["bias"], biz, "iz.b")
    close(gc["in"]["kernel"], Win.T, "in.k"); close(gc["in"]["bias"], bin_, "in.b")
    close(gc["hr"]["kernel"], Whr.T, "hr.k")
    close(gc["hz"]["kernel"], Whz.T, "hz.k")
    close(gc["hn"]["kernel"], Whn.T, "hn.k"); close(gc["hn"]["bias"], bhn, "hn.b")
    close(p["Dense_1"]["kernel"], agent.fc2.weight.detach().numpy().T, "fc2.kernel")
    close(p["Dense_1"]["bias"], agent.fc2.bias.detach().numpy(), "fc2.bias")
    assert np.allclose(bhr, 0.0) and np.allclose(bhz, 0.0), "torch b_hr/b_hz drifted"


def test_iql_update_matches_pytorch():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    T, E, N, in_dim, A, H = 8, 3, 2, 10, 5, 4
    B = E * N
    cfg = dataclasses.replace(
        DQNConfig.from_algo("iql"),
        hidden_dim=H, use_rnn=True, double_q=True, standardise_rewards=True,
        gamma=0.99, lr=3e-4, grad_norm_clip=10.0,
    )

    # reference torch agent + target (identical weights)
    agent = TorchRNNAgent(in_dim, H, A).double()
    target = TorchRNNAgent(in_dim, H, A).double()
    target.load_state_dict(agent.state_dict())

    # one identical synthetic batch (time-major)
    obs = rng.standard_normal((T, B, in_dim))
    actions = rng.integers(0, A, size=(T, E, N))
    reward = rng.standard_normal((T, E))
    terminated = np.zeros((T, E))
    filled = np.ones((T, E))
    # exercise the mask: episode 0 terminates at t=T-3, padded thereafter
    terminated[T - 3, 0] = 1.0
    filled[T - 2:, 0] = 0.0

    # ---- flax side: transplant identical weights, run iql_update ----
    qnet = QNetwork(num_actions=A, hidden_dim=H, use_rnn=True)
    f_params = freeze({"params": _to_flax(agent, H)})
    f_target = freeze({"params": _to_flax(target, H)})
    tx = make_optimizer(cfg)
    opt_state = tx.init(f_params)
    batch = {
        "obs_T": jnp.asarray(obs),
        "actions_T": jnp.asarray(actions),
        "reward_T": jnp.asarray(reward),
        "terminated_T": jnp.asarray(terminated),
        "filled_T": jnp.asarray(filled),
    }
    f_params2, _, _, diag = iql_update(
        qnet, tx, cfg, f_params, f_target, opt_state, rms_init((1,)), batch
    )

    # ---- torch side: same batch, same update ----
    ref = _torch_iql_update(
        agent, target,
        torch.tensor(obs), torch.tensor(actions), torch.tensor(reward),
        torch.tensor(terminated), torch.tensor(filled), cfg, E, N,
    )

    # ---- pre-step agreement (chosen-action Q, TD targets, loss) ----
    assert np.allclose(np.array(diag["chosen_action_qvals"]), ref["chosen"], atol=1e-6), \
        np.abs(np.array(diag["chosen_action_qvals"]) - ref["chosen"]).max()
    assert np.allclose(np.array(diag["targets"]), ref["targets"], atol=1e-6), \
        np.abs(np.array(diag["targets"]) - ref["targets"]).max()
    assert np.allclose(float(diag["loss"]), ref["loss"], atol=1e-6), \
        (float(diag["loss"]), ref["loss"])

    # ---- post-update parameter agreement (the decisive check) ----
    _assert_params_match(f_params2, agent, H, atol=1e-6)


# ============================================================================
# EMAX (Layer 4): ensemble-mean TD update, feedforward. No public reference
# code exists, so this is cross-stack parity -- JAX vs an independent PyTorch
# transcription of the paper's equations (ensemble-mean target shared across
# members, per-member TD loss, independent grads, clip+Adam).
# ============================================================================
class TorchMLP(torch.nn.Module):
    """Feedforward QNetwork: Linear->ReLU->Linear->ReLU->Linear (use_rnn=False)."""

    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.fc0 = torch.nn.Linear(in_dim, hidden)
        self.fc1 = torch.nn.Linear(hidden, hidden)
        self.fc2 = torch.nn.Linear(hidden, out_dim)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(torch.relu(self.fc0(x)))))


def _mlp_to_flax(mlp):
    return {
        "Dense_0": {"kernel": jnp.asarray(mlp.fc0.weight.detach().numpy().T),
                    "bias": jnp.asarray(mlp.fc0.bias.detach().numpy())},
        "Dense_1": {"kernel": jnp.asarray(mlp.fc1.weight.detach().numpy().T),
                    "bias": jnp.asarray(mlp.fc1.bias.detach().numpy())},
        "Dense_2": {"kernel": jnp.asarray(mlp.fc2.weight.detach().numpy().T),
                    "bias": jnp.asarray(mlp.fc2.bias.detach().numpy())},
    }


def _torch_emax_update(members, obs_TB, actions_TEN, reward_TE, terminated_TE,
                       filled_TE, cfg, E, N, bmask=None):
    A = members[0].fc2.out_features
    K = len(members)
    if bmask is None:
        bmask = torch.ones(K, E, dtype=torch.float64)
    T = obs_TB.shape[0]
    params = [p for m in members for p in m.parameters()]
    opt = torch.optim.Adam(params, lr=cfg.lr, eps=cfg.adam_eps)

    def all_q():
        return torch.stack([m(obs_TB).reshape(T, E, N, A) for m in members], 0)

    with torch.no_grad():
        q_mean = all_q().mean(0)                       # [T,E,N,A]
        target_max = q_mean[1:].max(dim=-1)[0]         # [T-1,E,N]
        r = reward_TE[: T - 1].clone()
        if cfg.standardise_rewards:
            rms = TorchRunningMeanStd((1,))
            rms.update(r.reshape(-1, 1))
            r = (r - rms.mean[0]) / torch.sqrt(rms.var[0])
        r_EN = r.unsqueeze(-1).expand(-1, -1, N)
        term_used = terminated_TE[: T - 1]
        targets = r_EN + cfg.gamma * (1 - term_used).unsqueeze(-1) * target_max

    mask = filled_TE[: T - 1].clone()
    mask[1:] = mask[1:] * (1 - terminated_TE[: T - 2])
    mask_EN = mask.unsqueeze(-1).expand(-1, -1, N)

    mac = all_q()
    idx = actions_TEN[: T - 1].unsqueeze(0).expand(K, -1, -1, -1).unsqueeze(-1)
    chosen = torch.gather(mac[:, : T - 1], -1, idx).squeeze(-1)   # [K,T-1,E,N]
    td = chosen - targets.unsqueeze(0)
    w = mask_EN.unsqueeze(0) * bmask[:, None, :, None]            # [K,T-1,E,N]
    loss = ((td ** 2) * w).sum() / w.sum()

    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, cfg.grad_norm_clip)
    opt.step()
    return {"loss": float(loss.detach()), "q_mean": q_mean.numpy()}


def test_emax_update_matches_pytorch():
    torch.manual_seed(0)
    rng = np.random.default_rng(1)
    T, E, N, in_dim, A, H, K = 8, 3, 2, 10, 5, 4, 5
    B = E * N
    cfg = dataclasses.replace(
        DQNConfig.from_algo("iql-emax"),
        hidden_dim=H, use_rnn=False, standardise_rewards=True,
        gamma=0.99, lr=3e-4, grad_norm_clip=10.0, ensemble_size=K,
    )

    members = [TorchMLP(in_dim, H, A).double() for _ in range(K)]

    obs = rng.standard_normal((T, B, in_dim))
    actions = rng.integers(0, A, size=(T, E, N))
    reward = rng.standard_normal((T, E))
    terminated = np.zeros((T, E))
    filled = np.ones((T, E))
    terminated[T - 3, 0] = 1.0
    filled[T - 2:, 0] = 0.0

    # flax ensemble params: stack the K transplanted member trees on axis 0
    qnet = QNetwork(num_actions=A, hidden_dim=H, use_rnn=False)
    member_trees = [_mlp_to_flax(m) for m in members]
    stacked = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *member_trees)
    params_ens = freeze({"params": stacked})
    tx = make_optimizer(cfg)
    opt_state = tx.init(params_ens)
    batch = {
        "obs_T": jnp.asarray(obs), "actions_T": jnp.asarray(actions),
        "reward_T": jnp.asarray(reward), "terminated_T": jnp.asarray(terminated),
        "filled_T": jnp.asarray(filled),
    }
    p2, _, _, diag = emax_update(qnet, tx, cfg, params_ens, opt_state,
                                 rms_init((1,)), batch)
    ref = _torch_emax_update(
        members, torch.tensor(obs), torch.tensor(actions), torch.tensor(reward),
        torch.tensor(terminated), torch.tensor(filled), cfg, E, N)

    # ensemble-mean Q and loss agree
    assert np.allclose(np.array(diag["q_mean"]), ref["q_mean"], atol=1e-6), \
        np.abs(np.array(diag["q_mean"]) - ref["q_mean"]).max()
    assert np.allclose(float(diag["loss"]), ref["loss"], atol=1e-6), \
        (float(diag["loss"]), ref["loss"])

    # post-update params agree for every ensemble member
    p2 = p2["params"]
    for k, m in enumerate(members):
        for dense, fc in (("Dense_0", m.fc0), ("Dense_1", m.fc1), ("Dense_2", m.fc2)):
            assert np.allclose(np.array(p2[dense]["kernel"][k]),
                               fc.weight.detach().numpy().T, atol=1e-6), (dense, k)
            assert np.allclose(np.array(p2[dense]["bias"][k]),
                               fc.bias.detach().numpy(), atol=1e-6), (dense, k)


def test_emax_bootstrap_mask_matches_pytorch():
    """EMAX update with a per-member bootstrap mask: JAX vs torch transcription."""
    torch.manual_seed(0)
    rng = np.random.default_rng(2)
    T, E, N, in_dim, A, H, K = 8, 4, 2, 10, 5, 4, 5
    B = E * N
    cfg = dataclasses.replace(
        DQNConfig.from_algo("iql-emax"),
        hidden_dim=H, use_rnn=False, standardise_rewards=True,
        gamma=0.99, lr=3e-4, grad_norm_clip=10.0, ensemble_size=K,
    )
    members = [TorchMLP(in_dim, H, A).double() for _ in range(K)]

    obs = rng.standard_normal((T, B, in_dim))
    actions = rng.integers(0, A, size=(T, E, N))
    reward = rng.standard_normal((T, E))
    terminated = np.zeros((T, E))
    filled = np.ones((T, E))
    terminated[T - 3, 0] = 1.0
    filled[T - 2:, 0] = 0.0
    # non-trivial bootstrap mask (and not all-ones for any member)
    bmask = rng.integers(0, 2, size=(K, E)).astype(np.float64)
    bmask[:, 0] = 1.0  # guarantee a nonzero denominator

    qnet = QNetwork(num_actions=A, hidden_dim=H, use_rnn=False)
    stacked = jax.tree_util.tree_map(
        lambda *xs: jnp.stack(xs), *[_mlp_to_flax(m) for m in members])
    params_ens = freeze({"params": stacked})
    tx = make_optimizer(cfg)
    opt_state = tx.init(params_ens)
    batch = {
        "obs_T": jnp.asarray(obs), "actions_T": jnp.asarray(actions),
        "reward_T": jnp.asarray(reward), "terminated_T": jnp.asarray(terminated),
        "filled_T": jnp.asarray(filled),
        "bootstrap_mask": jnp.asarray(bmask),
    }
    p2, _, _, diag = emax_update(qnet, tx, cfg, params_ens, opt_state,
                                 rms_init((1,)), batch)
    ref = _torch_emax_update(
        members, torch.tensor(obs), torch.tensor(actions), torch.tensor(reward),
        torch.tensor(terminated), torch.tensor(filled), cfg, E, N,
        bmask=torch.tensor(bmask))

    assert np.allclose(float(diag["loss"]), ref["loss"], atol=1e-6), \
        (float(diag["loss"]), ref["loss"])
    p2 = p2["params"]
    for k, m in enumerate(members):
        for dense, fc in (("Dense_0", m.fc0), ("Dense_1", m.fc1), ("Dense_2", m.fc2)):
            assert np.allclose(np.array(p2[dense]["kernel"][k]),
                               fc.weight.detach().numpy().T, atol=1e-6), (dense, k)
            assert np.allclose(np.array(p2[dense]["bias"][k]),
                               fc.bias.detach().numpy(), atol=1e-6), (dense, k)
