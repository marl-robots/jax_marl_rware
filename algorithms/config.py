"""AC-family hyperparameters mirroring the *proven* marlbase configs.

Source: marl-book-exercises/deep_marl_data/rware_tiny_4ag/{ia2c,ippo,maa2c,
mappo}/config.yaml (the tested runs that produced the published curves), NOT
the repo's *.yaml defaults.

The four on-policy actor-critic algorithms form a 2x2 matrix over two axes that
marlbase implements as a single parametrised codebase:

      critic \\ update   A2C (1 grad step)   PPO (clipped, num_epochs)
      independent        IA2C                IPPO
      centralised        MAA2C               MAPPO   (default here)

`centralised_critic` toggles the critic input (own obs vs concat of all agents'
obs); `use_ppo` toggles the actor loss (policy-gradient vs clipped surrogate)
and the epoch count (1 vs num_epochs). Everything else is shared and identical
across the four reference configs.
"""

from __future__ import annotations

from dataclasses import dataclass

# (centralised_critic, use_ppo) for each algorithm name.
_ALGO_FLAGS = {
    "ia2c": (False, False),
    "ippo": (False, True),
    "maa2c": (True, False),
    "mappo": (True, True),
}


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
    use_ppo: bool = True             # True -> PPO (clipped, num_epochs); False -> A2C (1 step)
    num_epochs: int = 4              # PPO epochs over the whole batch (no minibatching); A2C uses 1
    ppo_clip: float = 0.2            # PPO clip range; ignored for A2C
    seac_coef: float = 1.0           # SEAC shared-experience weight (lambda); only used by seac.py
    is_clip: float = 0.0             # SEAC: clip importance weight to <= is_clip (0 = no clip = canonical)
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
    def algo(self) -> str:
        """Algorithm name implied by (centralised_critic, use_ppo)."""
        inv = {flags: name for name, flags in _ALGO_FLAGS.items()}
        return inv[(self.centralised_critic, self.use_ppo)]

    @classmethod
    def from_algo(cls, algo: str, **kwargs) -> "MAPPOConfig":
        """Build a config for a named AC-family algorithm.

        `algo` in {ia2c, ippo, maa2c, mappo}; sets centralised_critic + use_ppo
        accordingly. Extra kwargs (size, n_agents, seed, ...) override defaults.
        """
        key = algo.lower()
        if key not in _ALGO_FLAGS:
            raise ValueError(
                f"unknown algo {algo!r}; choose from {sorted(_ALGO_FLAGS)}")
        centralised, use_ppo = _ALGO_FLAGS[key]
        return cls(centralised_critic=centralised, use_ppo=use_ppo, **kwargs)

    @property
    def batch_steps(self) -> int:
        """Env steps collected per update = one full episode across all envs."""
        return self.time_limit * self.parallel_envs

    @property
    def num_updates(self) -> int:
        return self.total_steps // self.batch_steps
