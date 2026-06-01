"""MAPPO hyperparameters mirroring the *proven* marlbase config.

Source: marl-book-exercises/deep_marl_data/rware_tiny_4ag/mappo/2/config.yaml
(the tested run that produced the published curve), NOT mappo.yaml defaults.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MAPPOConfig:
    # --- environment ---
    size: str = "tiny"
    n_agents: int = 4
    difficulty: str = "normal"
    time_limit: int = 500            # rware episode length (== Config.max_steps)
    parallel_envs: int = 10

    # --- optimisation ---
    lr: float = 3e-4
    grad_clip: bool = False          # config: grad_clip=false
    max_grad_norm: float = 0.5       # only used if grad_clip
    n_steps: int = 5                 # n-step return horizon
    gamma: float = 0.99
    entropy_coef: float = 1e-3
    value_loss_coef: float = 0.5
    num_epochs: int = 4              # PPO epochs over the whole batch (no minibatching)
    ppo_clip: float = 0.2
    standardise_returns: bool = False
    standardise_rewards: bool = True
    target_update_tau: float = 0.01  # soft Polyak update every update (< 1 -> soft)

    # --- network ---
    hidden_dim: int = 128            # actor/critic [128,128]: pre-GRU Dense + GRU width
    use_rnn: bool = True
    parameter_sharing: bool = True
    centralised_critic: bool = True
    orthogonal_gain: float = 2.0 ** 0.5  # gain on the FINAL Dense only

    # --- run ---
    total_steps: int = 20_000_000
    seed: int = 2

    @property
    def batch_steps(self) -> int:
        """Env steps collected per update = one full episode across all envs."""
        return self.time_limit * self.parallel_envs

    @property
    def num_updates(self) -> int:
        return self.total_steps // self.batch_steps
