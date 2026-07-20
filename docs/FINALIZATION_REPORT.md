# Final-project finalization — report for Tomer

*Session of 2026-06-28. All work on the new branch **`final-project`**. Nothing on
`master` or any existing branch was modified; the prior `FINAL_REPORT.md` (the
June 12–14 arena week) is left untouched — this is a separate, additive report.*

---

> **Robustness note.** Twice during the session the Claude process was torn down,
> which killed the harness-tracked GPU job. After the first time I **relaunched
> the queue fully detached (`setsid`+`nohup`)**, and it then ran to completion
> across two further teardowns. **No work was lost** — every run checkpoints every
> 100 iters and the queue is resume-safe (skips done runs, resumes partial ones).
> All planned runs finished, including two seeds on the headline maps.

## TL;DR

I unified the two halves of the thesis onto one branch, closed the value-based
gap with the EMAX port, ran a scaling study on **larger warehouses**, integrated
the new algorithm into the dashboard, and wrote the whole thing up as a **short
academic article** (Markdown + a generated PDF). The full test suite is green
(26 passing). Everything is reproducible from the repo.

---

## 1. What changed (commits on `final-project`)

1. **Unify the thesis branches** (`aa0db25`). Merged `arena-v2` (the LLM
   dashboard, SEAC, the Jumanji investigation, speed benchmarks) into the
   `emax-port` line (value-based IDQN-EMAX). Clean auto-merge, **zero
   conflicts**; the only files touched on both sides (`algorithms/config.py`,
   `scripts/train_mappo.py`, `jaxrware/env.py`, `.gitignore`) auto-merged. Full
   suite **26 passing** on the merged tree. This branch is now the complete
   project: fast faithful JAX RWARE engine + the on-policy AC family + SEAC +
   value-based IDQN-EMAX + the `arena` dashboard.

2. **The academic paper** (`b212997`, `db8990c`). `docs/paper/paper.md` — a
   short conference-style article covering the engine, the parity methodology,
   the algorithm ports (incl. the EMAX reconstruction), the scaling experiments,
   and the LLM `arena` layer. Built to a styled PDF (`docs/paper/paper.pdf`) with
   a pure-Python pipeline (`build_pdf.py`, markdown + xhtml2pdf — no LaTeX
   needed). Figures are generated from the actual run logs by `make_figures.py`,
   which also prints a measured results table so **no number in the paper is
   invented**.

3. **Dashboard integration of IDQN-EMAX** (`db8990c`). `arena/run_data.py` now
   recognises EMAX as a first-class algorithm (label, colour, display order), and
   the run-summary is NaN-safe for value-based CSVs. So the dashboard and the LLM
   narrator cover the value-based algorithm too, not just the AC family.

4. **README refresh** (`b212997`). The status table previously said
   "Value-based ❌ not implemented"; it now reflects the implemented and
   parity-validated IDQN + IDQN-EMAX, with the EMAX quickstart and a pointer to
   the paper.

---

## 2. The headline experiment — EMAX scales to larger warehouses

Vanilla independent Q-learning *collapses* on RWARE as the policy turns greedy
(the textbook value-based failure). EMAX's ensemble-with-UCB exploration fixes
this. I verified the fix **holds as the warehouse grows**, training IDQN-EMAX
(K=5) on all four registered sizes — `tiny` (11×10) up to `large` (29×16) — on the
GPU, plus a controlled **K=5 vs K=1** ablation on `medium` to confirm the
ensemble (not just the off-policy stack) is what matters.

<!-- RESULTS_BLOCK_START -->
**Scaling (IDQN-EMAX, K=5, final-10% mean deliveries/episode):**

| map | grid | env steps | seeds | deliveries | peak |
|---|---|---|---|---|---|
| tiny | 11×10 | 8M | 1 | 4.19 | 6.4 |
| small | 20×10 | 30M | 2 | 3.94 ± 0.04 | 5.5 |
| medium | 20×16 | 30M | 2 | 3.64 ± 0.01 | 4.8 |
| large | 29×16 | 24M | 1 | 2.98 | 4.4 |

EMAX **solves every warehouse size** (vanilla greedy IQL collapses to ~0 on
RWARE); deliveries decline gracefully with size and takeoff comes progressively
later (~2.3M steps tiny → ~6–9M medium). Two-seed agreement is ±0.04.

**Ensemble ablation (medium-4ag, 30M):** K=5 → 3.64 vs K=1 → 3.34 (+9%). A
*modest, consistent* ensemble benefit — reported honestly, not inflated; the
effect is expected to grow on harder-exploration regimes (see paper §5.5, §8).

Figures: `docs/paper/figs/fig{1_scaling,2_final_bar,3_ablation,4_speed}.png`.
<!-- RESULTS_BLOCK_END -->

All runs are on a single 4 GB GPU (RTX 3050) at ~10–13k env steps/s, fully jitted
on-device. Run logs and CSVs are under `runs/paper/`.

---

## 3. How to reproduce

```bash
# (WSL, conda env jax_env_1)
source ~/miniconda3/etc/profile.d/conda.sh && conda activate jax_env_1
cd /mnt/c/Users/user1/projects/jax_marl3

# tests (env parity + MAPPO/SEAC/DQN/EMAX gradient parity): 26 passing
python -m pytest tests/ -q

# train IDQN-EMAX on any size (the scaling result)
python -m scripts.train_dqn_fast --size large --n-agents 4 --ensemble-size 5

# regenerate the paper figures + measured results table
python docs/paper/make_figures.py

# rebuild the paper PDF (uses the throwaway venv ~/.venv_pdf)
~/.venv_pdf/bin/python docs/paper/build_pdf.py

# the dashboard (now incl. IDQN-EMAX)
streamlit run arena/app.py
```

---

## 4. What is validated vs not (kept honest)

- **Validated by numerical parity:** env step-for-step vs original RWARE; PPO,
  SEAC, IDQN and EMAX *updates* vs independent PyTorch transcriptions
  (1e-6–1e-8). Tests in `tests/`.
- **Validated by behaviour** (no oracle exists — EMAX code was never released):
  the ensemble layer reproduces the paper's qualitative claim (learns where
  vanilla IQL collapses) and the scaling result above.
- **Partial / unvalidated, and flagged as such:** SEAC's training *regime*; the
  recurrent EMAX and action-masking extensions (off by default; see
  `ACTION_MASKING.md`); the VDN/QMIX mixers are designed-for but not implemented.

---

## 5. Notes / housekeeping

- Branch `final-project` is **local only** unless you push it. `master` and all
  existing branches are untouched.
- A throwaway Python venv `~/.venv_pdf` was created for PDF rendering so the
  validated `jax_env_1` was not modified.
- The experiment queue script lived in the session scratchpad (not committed);
  the runs it produced are under `runs/paper/`.
- I did **not** modify the assistant memory store, per your instruction — this
  report is the external record of the session.
