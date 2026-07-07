<<<<<<< HEAD
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
> collision **resolution**; under a random policy **98.4%** of its episodes end
> early, median length **58/500**), pays a **shared team reward** instead of the
> benchmark's individual credit, and (per its own paper, App. A.5) trains it as
> a **single-agent** task (CTCE), not decentralized MARL. Its sequential agent
> update even makes legal convoy moves terminate the episode *depending on
> agent id order*. Full investigation with code citations and deterministic
> reproductions: [docs/jumanji_mava_divergence.md](docs/jumanji_mava_divergence.md).
> This project provides a parity-validated faithful environment **and** proper
> decentralized (CTDE / independent) multi-agent training.

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
  replay.py        episode trajectory snapshots (.npz) for the dashboard
arena/           the dashboard layer (pandas/numpy + stdlib; no jax import)
  app.py           Streamlit app: overview/leaderboard, compare, replay
                   theater (animated canvas player), LLM story, research
  player.py        self-contained HTML/JS canvas replay player + evolution tour
  replay_data.py / run_data.py   data contracts for runs + replays
  narrator_llm.py  LLM narration (claude CLI / ollama / template fallback)
docs/            jumanji_mava_divergence.md (investigation) + data/
scripts/         train_mappo / train_seac / record_replay / render_rollout
                 / bench_speed / repro_jumanji_divergence / train_queue.sh
tests/           env parity + collision + MAPPO unit/gradient-parity tests
```

## Environment

**Python 3.11** (developed/tested on 3.11.15). Pinned dependencies are in
[`requirements.txt`](requirements.txt); the original results were produced in a
WSL conda env with a CUDA-12 build of JAX.

```bash
# fresh install (conda recommended)
conda create -n jax_marl3 python=3.11 && conda activate jax_marl3
pip install -r requirements.txt          # GPU (CUDA 12) by default; see file for CPU
cd /path/to/jax_marl3

# OR, just to VIEW the dashboard on a shared snapshot (no JAX/GPU needed):
pip install -r requirements-dashboard.txt && streamlit run arena/app.py
```

Key versions: `jax==0.10.0`, `flax==0.12.7`, `optax==0.2.8`, `chex==0.1.91`,
`distrax==0.1.8`, `orbax-checkpoint==0.11.36`, `numpy==2.4.4`,
`streamlit==1.58.0`. (The reproduction in
[`docs/jumanji_mava_divergence.md`](docs/jumanji_mava_divergence.md) needs
`jumanji==1.1.1` in a **separate** env — it pulls an older JAX.)

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

# record replay snapshots from a finished run (for the dashboard theater)
python -m scripts.record_replay --run-dir runs/mappo_tiny-4ag_seed2 --episodes 2

# the dashboard (overview · compare · replay theater · LLM story · research)
streamlit run arena/app.py
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
| SEAC | ⚠️ **partial** — loss/gradients **parity-validated vs a PyTorch transcription of uoe-agents/seac** (`tests/test_seac_parity.py`); training *regime* still differs from canonical (full-episode rollouts + Adam here vs 5-step + RMSprop/clip there) and no machine-readable reference curve exists to close that gap |
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