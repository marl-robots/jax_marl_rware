# Potential-MAPPO — how to run

Self-supervised **Phi** ("closeness to delivery") used as a faithful
potential-based dense reward shaping on top of MAPPO. The shaping is
potential-based, so the optimal policy is unchanged — it only speeds learning.

## Best result

FeedForward actor + Phi shaping, given enough data, beats the GRU baseline and
takes off ~7x faster than plain MAPPO:

```bash
python scripts/train_mappo_potential.py --no-rnn --total-steps 20_000_000
```

Defaults that matter: `--phi-beta 1.0`, `--phi-epochs 4`. Leave them as-is.

## Knobs worth trying
- `--phi-beta-anneal N --phi-beta-end 0.0` — decay shaping to 0 over N updates.
- `--ppo-envs K` — decouple Phi-training data from the PPO/BPTT batch.
- `--cnn` — spatial CNN actor (alternative to `--no-rnn` FC).
- `--phi-square` — signed-square Phi-diff shaping (experimental).

## What did NOT work
- **Scout / staleness obs-channel** — fully reverted. It broke the fragile
  FC+Phi initialization and never helped. Measure the scout bottleneck before
  retrying.
- **RNN (GRU) actor** — loses to FC+Phi once you give it enough data. The early
  "RNN wins" reading was a low-data artifact (see commit 757d2b8).
