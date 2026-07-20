"""Value-based (off-policy) hyperparameters mirroring the EPyMARL configs.

Source: uoe-agents/epymarl src/config/{default,algs/{iql,vdn,qmix}}.yaml (the
public benchmark configs). These are the BASE algorithms that EMAX
(arXiv 2302.03439) extends with an ensemble; this module covers the base, and
the EMAX-specific knobs (ensemble size, UCB beta) hang off the same config so
the trainer can scale iql -> vdn -> qmix -> +EMAX from one parametrisation,
the same way `config.py` does for the on-policy AC 2x2.

The three base algorithms differ along two axes EPyMARL implements as one
codebase (the `q_learner` + a swappable `mixer`):

      mixer        recurrence      algorithm
      None         GRU (use_rnn)   IQL    (independent Q, per-agent TD)
      VDN sum      feedforward     VDN    (additive value decomposition)
      QMIX hypernet feedforward    QMIX   (monotonic mixing via hypernetwork)

NOTE — differences from the AC family (config.py) that matter for parity:
  * optimiser is Adam (torch defaults: betas 0.9/0.999, eps 1e-8) with grads
    clipped to global L2 norm 10 BEFORE the Adam step. The yaml's optim_alpha/
    optim_eps are vestigial — epymarl's QLearner hardcodes `Adam(params, lr)`.
  * targets are 1-step TD(0) with double-Q action selection (not n-step).
  * reward standardisation uses epymarl's RunningMeanStd (parallel-variance,
    unbiased var, count init 1e-4), NOT the AC family's Welford in mappo.py.
  * off-policy: an episode replay buffer (5000 episodes) sampled in batches of
    32 episodes; a HARD target-network copy every 200 episodes; double-Q.
  * obs is augmented with a one-hot agent id (obs_agent_id=True in EPyMARL);
    obs_last_action is False in all three RWARE configs, so it is omitted here.
  * IQL is recurrent (use_rnn=True, hidden 128, lr 3e-4); VDN/QMIX are
    feedforward (use_rnn=False, hidden 64, lr 5e-4). These per-algo splits are
    baked into `from_algo` below so each algorithm gets its reference defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

# Per-algorithm reference overrides, transcribed from the EPyMARL yamls.
# Anything not listed here falls back to the dataclass defaults (= default.yaml).
_ALGO_DEFAULTS = {
    # iql.yaml: recurrent, lr 3e-4, hidden 128, eps anneal 200k, no mixer.
    "iql":  dict(mixer="none", use_rnn=True,  hidden_dim=128, lr=3e-4,
                 epsilon_anneal_time=200_000),
    # vdn.yaml: feedforward, default lr/hidden, eps anneal 50k, additive mixer.
    "vdn":  dict(mixer="vdn",  use_rnn=False, hidden_dim=64,  lr=5e-4,
                 epsilon_anneal_time=50_000),
    # qmix.yaml: feedforward, default lr/hidden, eps anneal 50k, hypernet mixer.
    "qmix": dict(mixer="qmix", use_rnn=False, hidden_dim=64,  lr=5e-4,
                 epsilon_anneal_time=50_000),
}


@dataclass(frozen=True)
class DQNConfig:
    # --- environment (shared with MAPPOConfig) ---
    size: str = "tiny"
    n_agents: int = 4
    difficulty: str = "normal"
    time_limit: int = 500            # rware episode length (== Config.max_steps)
    parallel_envs: int = 1           # EPyMARL "episode" runner collects 1 env/episode

    # --- optimisation (Adam; epymarl QLearner hardcodes Adam(params, lr)) ---
    lr: float = 5e-4                 # default.yaml; iql.yaml overrides to 3e-4
    adam_eps: float = 1e-8           # torch Adam default (betas 0.9/0.999)
    grad_norm_clip: float = 10.0     # global-L2-norm clip applied before Adam
    gamma: float = 0.99
    double_q: bool = True
    standardise_rewards: bool = True
    standardise_returns: bool = False

    # --- replay / target ---
    buffer_size: int = 1_000         # episodes (algs override default.yaml's 32)
    batch_size: int = 16             # episodes sampled per gradient update
    target_update_interval: int = 200  # HARD copy every N episodes (tau path unused)

    # --- exploration (epsilon-greedy; anneal time is per-algo) ---
    epsilon_start: float = 1.0
    epsilon_finish: float = 0.05
    epsilon_anneal_time: int = 50_000   # env steps; iql uses 200_000
    evaluation_epsilon: float = 0.0

    # --- network ---
    hidden_dim: int = 64             # default rnn agent; iql overrides to 128
    use_rnn: bool = False            # iql=True; vdn/qmix=False
    mixer: str = "none"              # {none (IQL), vdn, qmix}
    mixing_embed_dim: int = 32       # qmix mixer embed width
    hypernet_layers: int = 2         # qmix hypernetwork depth
    hypernet_embed: int = 64         # qmix hypernetwork hidden width
    parameter_sharing: bool = True
    obs_agent_id: bool = True        # append one-hot agent id to obs (EPyMARL)

    # --- action masking (opt-in; off == byte-identical to the unmasked path) ---
    # When True the env emits a [N, A] mask of PROVABLY-no-op actions
    # (Warehouse.action_masks) that is applied identically at rollout action
    # selection (UCB greedy + eps-random) and in the bootstrap-max target, so
    # the optimal policy is provably unchanged and only wasted exploration is
    # removed. Default off -> no mask is computed, stored, or applied.
    use_action_mask: bool = False

    # --- EMAX extension (arXiv 2302.03439); off when ensemble_size == 1 ---
    use_emax: bool = False
    ensemble_size: int = 6           # K value functions per agent; paper uses 5
    ucb_beta: float = 1.0            # exploration weight in argmax[Q_mean + beta*Q_std]
    bootstrap_mask_prob: float = 0.5  # per-member Bernoulli mask over the minibatch
                                     # (bootstrapped sampling -> ensemble diversity;
                                     # 1.0 == shared batch / no bootstrapping)

    # --- recurrent training ---
    bptt_window: int = 0             # truncated BPTT window for the recurrent net:
                                     # gradients are cut (stop_gradient on hidden)
                                     # every N steps; hidden still flows forward the
                                     # whole episode. 0 == full-episode BPTT.

    # --- run ---
    t_max: int = 10_000_000          # total env steps (RWARE benchmark budget)
    seed: int = 2
    algo_name: str="iql-emax"

    @property
    def algo(self) -> str:
        """Base algorithm name implied by the mixer/recurrence combination."""
        base = {"none": "iql", "vdn": "vdn", "qmix": "qmix"}[self.mixer]
        return f"{base}-emax" if self.use_emax else base

    @classmethod
    def from_algo(cls, algo: str, **kwargs) -> "DQNConfig":
        """Build a config for a named value-based algorithm.

        `algo` in {iql, vdn, qmix}, optionally suffixed `-emax` to enable the
        ensemble (sets use_emax=True and ensemble_size=5 unless overridden).
        Per-algo reference defaults (mixer, recurrence, lr, hidden, eps anneal)
        are applied first; explicit kwargs win over them.
        """
        key = algo.lower()
        use_emax = key.endswith("-emax")
        base = key[: -len("-emax")] if use_emax else key
        if base not in _ALGO_DEFAULTS:
            raise ValueError(
                f"unknown algo {algo!r}; choose from "
                f"{sorted(_ALGO_DEFAULTS)} (optionally with a '-emax' suffix)")
        defaults = dict(_ALGO_DEFAULTS[base])
        if use_emax:
            defaults.setdefault("use_emax", True)
            defaults.setdefault("ensemble_size", 5)  # paper default K=5
        defaults.update(kwargs)  # caller overrides reference defaults
        return cls(**defaults) # type: ignore

    def with_overrides(self, **kwargs) -> "DQNConfig":
        return replace(self, **kwargs)
