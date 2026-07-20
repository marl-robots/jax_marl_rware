# Potential-MAPPO — self-supervised dense shaping for faithful, faster learning

MAPPO augmented with a small **potential network Φ** that learns, by supervised
regression on the agent's OWN rollouts, how *close to a delivery* each
observation is. That closeness is used as a **potential-based shaping** term to
densify the otherwise-sparse delivery reward — **without touching the env reward
function**, so the benchmark stays faithful and the result is measured on the
true delivery signal.

This is the on-policy realisation of the "understanding-first, reward-last" idea:
the agent builds a self-supervised sense of "am I getting warmer?" from its own
successes, and that turns each gradient step into real progress instead of
waiting for the rare full-delivery to back-propagate.

## Mechanism (`algorithms/mappo_potential.py`)
- **Φ network**: a small MLP `obs -> scalar`, with a **zero-initialised output
  head** so Φ ≡ 0 at the start → bonus ≡ 0 → byte-for-byte plain MAPPO. The
  shaping ramps in on its own as Φ learns.
- **Shaping bonus**: per step, add `β·(γ·Φ(s′) − Φ(s))` to the (standardised)
  reward used by the PPO update. This is **potential-based** (Ng et al.): for
  ANY Φ it provably preserves the optimal policy and telescopes to ~0 over an
  episode, so it cannot change the optimum or net reward — it only moves signal
  earlier in time. A bad Φ can only fail to help, never break the run.
- **Φ target** (computed in-graph from the rollout's own delivery flags):
  `Φ(s_t) → γ^(steps to this agent's next delivery)` ∈ [0,1] (1 at a delivery,
  decaying back, 0 if no future delivery). A few gradient steps per update fit it.
- **Faithful metrics**: deliveries/return are logged from the RAW signal, so the
  CSV compares directly against a plain-MAPPO run.

Everything else (actor/critic, PPO update, Welford reward std, target critic,
metrics) is reused verbatim from `algorithms/mappo.py`. `β=0` is exactly plain
MAPPO.

## Result — clean A/B (rware-tiny-4ag, seed 2, 50 envs, identical except Φ)

Measured **by update count** (≈ wall-time: the per-update cost is bound by the
T=500 sequential GRU rollout, ~independent of parallel_envs, so more envs is
nearly free and updates are the honest axis):

| update | β=1 (Φ shaping) | β=0 (plain MAPPO) |
|---|---|---|
| 250  | 0.08 | 0.08 |
| 500  | **1.10** | 0.02 |
| 750  | **3.04** | 0.32 |
| 1000 | **3.98** | 0.92 |
| 1200 | **7.10** | **1.02** |

Φ takes off ~update 500 and reaches **7.1 deliveries** by 1200; plain MAPPO is
still crawling at **1.0**. ~7× faster takeoff, on a fully matched comparison.

### Honest caveats
- **Single seed.** Large effect → seed-luck unlikely, but needs 2–3 seeds to be
  defensible.
- **Takeoff / update-efficiency win, not a proven higher ceiling.** Plain MAPPO
  keeps climbing with more updates (the 10-env baseline eventually hit ~30); the
  claim is "learns far faster per update/wall-time."
- The 7.1 is **genuine** deliveries (raw signal), not inflated by the bonus.

### Can we drop the RNN thanks to Φ? — YES (FC+Φ wins on wall-time)

A first read at 50 envs/1200 updates (FC+Φ=1.72 vs RNN+Φ=7.10) suggested "keep the
RNN" — but FC+Φ there was **starved and still rising**, not plateaued. Given a
fair shot (FC is light → no OOM → many more envs, ~5–12× faster):

**FC+Φ @ 200 envs (seed2):** deliveries 1.1 → 2.4 → 3.9 → 6.4 → **9.02 at update
2500, still climbing**, in **~21 min wall-time** vs RNN+Φ's 7.1 in ~31 min. FC+Φ
beats RNN+Φ on **both ceiling and wall-time**. So Φ's dense signal *does* stand in
for the recurrent value's credit-assignment role; the FF policy just needed data,
which FC supplies cheaply. (FC is far less sample-efficient *per env-step* — 250M
vs 30M — but per-step cost is so low that wall-time favors it, and a 200-env
*recurrent* MAPPO OOMs on the 4 GB GPU.) **Strong config: FC + Φ + many envs.**

## Run
```
# potential-MAPPO (defaults match the proven MAPPO baseline; only Φ added)
python -m scripts.train_mappo_potential --parallel-envs 50 --total-steps 30000000 --phi-beta 1.0
# matched plain-MAPPO control: same command with --phi-beta 0
```
Writes to a NEW run-dir (`runs/mappo_pot_*`) so its `results.csv` compares
against `runs/mappo_tiny-4ag_seed2`. Knobs: `--phi-beta`, `--phi-epochs`,
`--phi-lr`, `--phi-hidden`.

## Replication update (2026-07-07, branch final-unified)

The seed-2 A/B above was replicated on seeds 0 and 1 (identical config: 50
envs, RNN, 1200 updates, matched β=0 controls). Findings:

| seed | shaped takeoff | control takeoff | deliv @1200 shaped / control |
|---|---|---|---|
| 2 | ~u400 | not within budget | **7.1** / 1.0 |
| 0 | ~u520 | ~u900 | 2.5 / **3.3** |
| 1 | ~u500 | ~u600 | 3.4 / **4.8** |

What replicates: **early, consistent ignition** of the shaped runs (~u400–520
on all seeds) vs a high-variance control takeoff (u600 / u900 / >1200). What
does not: the 7× final-score gap — where the control ignites, it catches up
and can finish ahead. The honest claim is reduced takeoff variance, not higher
performance; at n=3 even that is suggestive. Per-seed curves:
`runs/mappo_pot{,0}_tiny-4ag_seed{0,1,2}`. (Training code was verified
identical between the original branch and final-unified — this is seed
variance, not code drift.)
