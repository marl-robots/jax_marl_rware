#!/usr/bin/env bash
# Sequential training queue for the demo-data week (2026-06-12).
# Launch detached from Windows:
#   wsl -e bash -lic 'cd /mnt/c/Users/user1/projects/jax_marl3; \
#       nohup bash scripts/train_queue.sh >> runs/queue.log 2>&1 &'
# Each job logs to runs/<name>_train.log; every run records replay snapshots
# (--replay-every defaults to the checkpoint cadence) for the dashboard's
# evolution theater. MEM_FRACTION=.6 leaves VRAM headroom so short interactive
# jobs (replay recording, benchmarks) can run alongside.
set -u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate jax_env_1
cd /mnt/c/Users/user1/projects/jax_marl3
export XLA_PYTHON_CLIENT_MEM_FRACTION=.6

run() {
  name=$1; shift
  echo "=== [$(date '+%F %T')] START $name: $*"
  python -m scripts.train_mappo "$@" > "runs/${name}_train.log" 2>&1
  echo "=== [$(date '+%F %T')] DONE  $name (exit $?)"
}

# 1) flagship: MAPPO evolution run with a dense replay trail (~50M steps).
#    Fresh dir (not the existing 60M run) because early/mid checkpoints of
#    that run are gone -- the policy's childhood can't be re-recorded.
run mappo_evo --algo mappo --total-steps 50000000 --checkpoint-every 100 \
    --run-dir runs/mappo_evo_tiny-4ag_seed2 --no-commentary

# 2) complete the AC-family comparison set (the old ippo/ia2c dirs were
#    20-update smoke stubs; archived under runs/_archive)
run ippo --algo ippo --total-steps 20000000 --checkpoint-every 100 \
    --run-dir runs/ippo_tiny-4ag_seed2 --no-commentary
run ia2c --algo ia2c --total-steps 20000000 --checkpoint-every 100 \
    --run-dir runs/ia2c_tiny-4ag_seed2 --no-commentary

# 3) extend MAA2C from 15M to the 20M reference horizon
run maa2c_ext --algo maa2c --total-steps 20000000 --checkpoint-every 100 \
    --run-dir runs/maa2c_tiny-4ag_seed2 --resume --no-commentary

echo "=== [$(date '+%F %T')] QUEUE COMPLETE"
