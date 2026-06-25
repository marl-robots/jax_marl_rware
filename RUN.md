# EMAX port — how to run

Value-based ensemble method (arXiv 2302.03439) — the first value method to beat
IPPO/MAPPO on RWARE. This branch builds the off-policy infra (replay, IDQN) the
repo lacked, then layers EMAX on top.

## Best result

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

## Validate first
Parity tests guard the port — run before trusting any result:

```bash
python -m pytest tests/test_dqn_parity.py
```

Quick wiring check: `--smoke`.

## What is NOT validated yet
- **Action masking** (`opt-in`, gated) — committed but unvalidated; off by
  default. Don't report numbers with it on until parity is re-checked.
- **Recurrent EMAX** (truncated BPTT) — present but heavier; FC is the
  reference config above.
