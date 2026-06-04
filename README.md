# jax_marl3 — Fast, faithful JAX port of RWARE + cooperative MARL baselines

A fully-jitted [JAX](https://github.com/google/jax) re-implementation of the
**Multi-Robot Warehouse (RWARE)** environment and a suite of cooperative
multi-agent RL algorithms, built for GPU/TPU speed while staying **behaviorally
faithful** to the original.

The environment is validated **step-for-step against the original**
[semitable/robotic-warehouse](https://github.com/semitable/robotic-warehouse),
and the algorithms are validated against the published **marl-book / `marlbase`**
reference curves ([`deep_marl_data`](https://github.com/uoe-agents/marl-book-codebase)).

> **Why another JAX warehouse env?** The only prior JAX implementation (Jumanji's
> `RobotWarehouse`) diverges from canonical RWARE on its difficulty-defining
> mechanics — it *terminates the episode on collision* (vs. the original's
> collision **resolution**), drops the two-stage collect→deliver→**return**
> reward, and (per its own paper, App. A.5 + Future Work) trains it as a
> **single-agent** task (CTCE), not decentralized MARL. This project provides a
> parity-validated faithful environment **and** proper decentralized
> (CTDE / independent) multi-agent training.

---

## Layout

```
jaxrware/        JAX RWARE env (pure, jit/vmap-friendly, fixed shapes)
  config.py        static Config + named sizes (tiny/small/medium/large)
  layout.py        highways / goals / shelf positions (exact rware layout)
  collision.py     vectorized fixed-point collision resolver (NetworkX-free)
  observations.py  FLATTENED per-agent sensor-window obs
  env.py           reset/step, delivery, two-stage reward, auto-reset
  render.py        headless Pillow renderer
algorithms/      PureJaxRL/JaxMARL-style trainers
  config.py        MAPPOConfig (hyperparams; algo 2x2 + use_rnn + seac flags)
  networks.py      shared GRU / FC actor + critic (orthogonal init)
  returns.py       n-step returns (target-critic bootstrap)
  mappo.py         AC-family 2x2: IA2C / IPPO / MAA2C / MAPPO
  seac.py          SEAC (per-agent + shared-experience)   [WIP — see status]
  checkpoint.py    orbax keep-best + keep-last-N + resume
  metrics.py / commentary.py   CSV logging + behavioral commentary
scripts/         train_mappo / train_seac / render_rollout / bench_launch
tests/           env parity + collision + MAPPO unit/gradient-parity tests
```

## Environment

Runs in WSL, conda env `jax_env_1` (JAX+CUDA, flax, optax, chex, distrax, orbax):

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate jax_env_1
cd /mnt/c/Users/user1/projects/jax_marl3
```

## Quickstart

```bash
# tests: env parity vs original rware + MAPPO gradient parity vs PyTorch
python -m pytest tests/ -q

# train an AC-family algorithm (2x2: {ia2c, ippo, maa2c, mappo})
python -m scripts.train_mappo --algo mappo --total-steps 20000000     # full proven run
python -m scripts.train_mappo --algo ippo  --updates 20               # quick smoke
python -m scripts.train_mappo --algo mappo --no-rnn                   # FC (~11x faster, lower ceiling)
python -m scripts.train_mappo --algo mappo --resume                   # continue from last checkpoint

# render a trained checkpoint to mp4/gif/png
python -m scripts.render_rollout --run-dir runs/mappo_tiny-4ag_seed2 --step best
```

The **2×2** is two flags: `centralised_critic` × `use_ppo` →
IA2C (ind, A2C) · IPPO (ind, PPO) · MAA2C (cent, A2C) · MAPPO (cent, PPO).
`--no-rnn` swaps the GRU for a feedforward net.

## Status

| Component | State |
|---|---|
| JAX RWARE env | ✅ step-for-step parity vs original rware (randomized seeds) |
| Collision resolver | ✅ vectorized; matches rware's longest-path resolution |
| MAPPO (cent + PPO) | ✅ reproduces EPyMARL reference (tiny-4ag, seed 2) |
| IA2C / IPPO / MAA2C | ✅ implemented; MAA2C tracks reference through liftoff (≈4.2 @ 15M vs ref 4.1) |
| FC vs GRU | ✅ characterized: FC ~11× faster, GRU higher ceiling on RWARE |
| Checkpointing / resume / render | ✅ |
| SEAC | ⚠️ **WIP** — loss faithfully transcribed, **not yet validated** (regime mismatch vs canonical short-rollout setup; see commit notes) |
| Value-based (IDQN/VDN/QMIX) | ❌ not yet implemented |

## Validation philosophy

Claims are backed by **numerical parity**, not narrative:
- **Env:** inject identical post-reset state + identical action sequences into
  both this env and the original rware; assert exact equality of positions,
  grid, queue, rewards, done across many seeds (`tests/test_parity_env.py`).
- **Algorithms:** gradient-parity of the PPO update against a PyTorch transcription
  (`tests/test_mappo_parity.py`), plus learning-curve comparison to the
  `deep_marl_data` reference runs.

## Attribution

This is a re-implementation/port; algorithm *semantics* and the environment are
not original to this project:

- **RWARE** — Christianos, Papoudakis, Schäfer, Albrecht (Univ. of Edinburgh);
  [semitable/robotic-warehouse](https://github.com/semitable/robotic-warehouse).
- **Algorithm semantics + reference configs/curves** — the marl-book companion
  code (`marlbase` / `deep_marl_data`); benchmark methodology from Papoudakis et al., 2021.
- **MAPPO** — Yu et al., 2021. **SEAC** — Christianos et al., NeurIPS 2020.

Please retain the upstream licenses and cite the above when using this code.
