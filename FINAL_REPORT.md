# Autonomous week — final report

## What this week added

The engine + algorithm ports were already in place. This week built the
**presentation/research layer** on top: a polished live-telemetry dashboard, an
animated training-replay theater, an LLM narrator, a rigorous investigation of
the Jumanji/Mava environment divergences, speed benchmarks, and a decisive
validation of the SEAC loss. 11 commits, full test suite green (22 passing).

### 1. Dashboard (`arena/`, run with `streamlit run arena/app.py`)
Five tabs, dark "mission-control" theme:
- **🏆 Overview** — headline KPIs, category medals, final-return bars, ranked
  leaderboard, the "race" curve.
- **📊 Compare** — EMA-smoothed learning curves (raw signal ghosted behind),
  grouped by performance / behavior / optimization / episode-phase.
- **🎬 Replay theater** — the showpiece (see §2).
- **🧠 Story** — LLM narration of the selected runs (see §3).
- **🔬 Research** — the speed benchmark chart + the Jumanji divergence findings
  (see §4, §5).

### 2. Replay capture + animated theater
- `algorithms/replay.py` records full episode trajectories during training (one
  jitted rollout, ~9 KB/episode) at fixed seeds, so snapshots taken across
  training differ **only by the policy**. `scripts/record_replay.py` does the
  same from saved checkpoints of finished runs (MAPPO-family and SEAC).
- `arena/player.py` is a self-contained HTML/JS canvas player (no external
  assets → works offline): interpolated agent motion with trails, glowing
  requested shelves, delivery flash rings, live KPIs / event feed / sparkline,
  play-pause-speed-scrub. With multiple snapshots it shows a **TRAINING
  PROGRESS slider** and an **EVOLUTION TOUR** that auto-plays from untrained
  chaos to the final policy. Plus a **duel mode** (two algorithms, same episode
  seed, side by side).
- `mappo_evo_tiny-4ag_seed2` has a dense 26-snapshot evolution trail recorded
  during training — use it for the tour.

### 3. LLM narration (`arena/narrator_llm.py`)
Narrates a run from **measured numbers only** (the prompt forbids invented
figures). Backend chain, all optional: **Claude CLI** → **Ollama** →
deterministic **template**. Narrations cache to `runs/_narrations/`.
- ⚠️ **Claude CLI needs a one-time login**: run `claude /login` once in a
  terminal and the "claude" backend lights up. Until then it falls back to
  Ollama (works, ~slow) or the template (instant, always works).

### 4. Speed benchmarks (`scripts/bench_speed.py`, data in `docs/data/`)
tiny-4ag, random actions, GPU (RTX 3050) / CPU:

| impl | CPU peak | GPU peak (batch 4096) |
|---|---|---|
| **jaxrware (ours)** | 202k steps/s | **9.1M steps/s** |
| jumanji | 107k | 2.4M |
| rware (original gym) | 2.7k | — (single process) |

Ours is **~3.8× Jumanji** and **~3400× the original** on GPU — while staying
step-for-step parity-validated against the original.

### 5. Jumanji/Mava divergence investigation
`docs/jumanji_mava_divergence.md` (+ `scripts/repro_jumanji_divergence.py`,
deterministic, runs in both environments). Findings, each code-cited:
- Jumanji **terminates the episode on "collision"** instead of resolving it;
  under a random policy **98.4% of episodes end early, median 58/500 steps**
  (vs 0% / always-500 in the original). Early training optimizes *survival*,
  not logistics.
- Its sequential agent update makes a **legal convoy move terminate the episode
  depending on agent id order** (undocumented), and head-on contention leaves
  **two agents on the same cell**.
- It uses a **shared team reward**; Mava broadcasts it to all agents — erasing
  the individual credit-assignment problem the benchmark exists to test.
- This propagates into Mava's 15 RWARE scenarios and the **Sable paper**, which
  calls the collision change "minor" (App. B.1). The report scopes honestly
  what stays valid (within-paper comparisons) vs what breaks (cross-paper).

### 6. SEAC validation
`tests/test_seac_parity.py`: the SEAC loss/gradients now match an independent
PyTorch transcription of the canonical `uoe-agents/seac` update to **1e-8 in
float64** (seac_coef ∈ {1.0, 0.5}). Status upgraded WIP → **partial**: the loss
is validated; the *training regime* still differs from canonical (full-episode
rollouts + Adam here vs 5-step + RMSprop there) and no machine-readable
reference curve exists to close that last gap.

## Results (leaderboard, final tail-mean team return)
| algo | env steps | final return | note |
|---|---|---|---|
| MAPPO | 60M | 36.2 | the proven reference run |
| MAPPO (evo) | 50M | 35.2 | recorded with replay trail |
| **IA2C** | 20M | **20.0** | independent beats centralised here |
| IPPO | 20M | 5.4 | |
| MAA2C | 20M | 4.0 | |
| SEAC | 5M | ≤2.1 | short runs; see §6 |

A genuine talking point: on tiny-4ag at 20M, **independent IA2C outperforms the
centralised-critic MAA2C/IPPO** — centralisation is not a free win on
sparse-reward RWARE.

## How to run (WSL, conda env jax_env_1)
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate jax_env_1
cd /mnt/c/Users/user1/projects/jax_marl3
streamlit run arena/app.py                       # the dashboard
python -m pytest tests/ -q                        # 22 passing
```
See `docs/DEMO_GUIDE.md` for the 4-act presentation script.

## State & caveats
- All work is committed on **`arena-v2`**, **not pushed** (your call to keep it
  local). `master` is untouched.
- The **failsafe shutdown task was deleted** at your request (2026-06-13); the
  PC will not auto-shut-down.
- Old smoke/stub runs were moved to `runs/_archive/`.
- `_tmp_*` files at the repo root (jumanji source copy, paper text, scratch
  scripts) are gitignored scratch — safe to delete.
- Claude CLI narration backend is the only thing needing a manual step
  (`claude /login`).

## Suggested next steps (if you want to continue)
- Run more seeds per algorithm for confidence bands on the curves.
- A longer IA2C/IPPO/MAA2C run (60M, matching MAPPO) for a fair final ranking.
- Close the SEAC regime gap (short-rollout + RMSprop variant) if a reference
  curve becomes available.
- Value-based baselines (IDQN/VDN/QMIX) are still unimplemented.
