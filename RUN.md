# How to run — quick guides per algorithm family

## EMAX port

Value-based ensemble method (arXiv 2302.03439) — the first value method to beat
IPPO/MAPPO on RWARE. This branch builds the off-policy infra (replay, IDQN) the
repo lacked, then layers EMAX on top.

### Best result

IDQN-EMAX (bootstrapped per-member ensemble + UCB exploration), fully jitted on
GPU:

```bash
python scripts/train_dqn.py --algo iql-emax --iters 2000 --parallel-envs 8
```

Start from the plain IQL base to sanity-check the off-policy stack:

```bash
python scripts/train_dqn.py --algo iql --iters 2000
```

Ensemble/UCB knobs (leave `None` to use validated defaults):
`--ensemble-size`, `--ucb-beta`, `--buffer-size`, `--batch-size`.

### Validate first
Parity tests guard the port — run before trusting any result:

```bash
python -m pytest tests/test_dqn_parity.py
```

Quick wiring check: `--smoke`.

### What is NOT validated yet
- **Action masking** (`opt-in`, gated) — committed but unvalidated; off by
  default. Don't report numbers with it on until parity is re-checked.
- **Recurrent EMAX** (truncated BPTT) — present but heavier; FC is the
  reference config above.

## Potential-MAPPO

Self-supervised **Phi** ("closeness to delivery") used as a faithful
potential-based dense reward shaping on top of MAPPO. The shaping is
potential-based, so the optimal policy is unchanged — it only speeds learning.

### Best result

FeedForward actor + Phi shaping, given enough data, beats the GRU baseline and
takes off ~7x faster than plain MAPPO:

```bash
python scripts/train_mappo_potential.py --no-rnn --total-steps 20_000_000
```

Defaults that matter: `--phi-beta 1.0`, `--phi-epochs 4`. Leave them as-is.

### Knobs worth trying
- `--phi-beta-anneal N --phi-beta-end 0.0` — decay shaping to 0 over N updates.
- `--ppo-envs K` — decouple Phi-training data from the PPO/BPTT batch.
- `--cnn` — spatial CNN actor (alternative to `--no-rnn` FC).
- `--phi-square` — signed-square Phi-diff shaping (experimental).

### What did NOT work
- **Scout / staleness obs-channel** — fully reverted. It broke the fragile
  FC+Phi initialization and never helped. Measure the scout bottleneck before
  retrying.
- **RNN (GRU) actor** — loses to FC+Phi once you give it enough data. The early
  "RNN wins" reading was a low-data artifact (see commit 757d2b8).
