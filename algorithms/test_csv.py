import json

import jax.numpy as jnp
import numpy as np

from flax import serialization
from algorithms.metrics import CSVLogger

from algorithms.metrics_handler import process_raw


def load_metrics():
    with open("metrics_tensors.msgpack", "rb") as f:
        raw = f.read()
    with open("metrics_template.json", "r") as f:
        metrics_template = json.load(f)
    metrics_loaded = serialization.from_bytes(metrics_template, raw)
    return metrics_loaded

import re

# manual priority from "most meaningful" to less (based on MARL practice / your header)
PRIORITY_ORDER = [
    "environment_steps",
    "updates",
    "episode_time",
    "episode_return_mean",
    "episode_return_std",
    "success_mean",
    "success_std",
    "success_rate_mean",
    "success_rate_std",
    "deliveries_mean",
    "deliveries_std",
    "deliveries_early_mean",
    "deliveries_early_std",
    "deliveries_mid_mean",
    "deliveries_mid_std",
    "deliveries_late_mean",
    "deliveries_late_std",
    "block_rate_mean",
    "block_rate_std",
    "idle_rate_mean",
    "idle_rate_std",
    "pickup_rate_mean",
    "pickup_rate_std",
    "distance_traveled_mean",
    "distance_traveled_std",
    "reward_std_mean",
    "reward_std_std",
    "loss_mean",
    "loss_std",
    "FPS_mean",
    "FPS_std",
    "step_count_mean",
    "step_count_std",
    "step_time",
    "step_time_std",
    "time_to_first_delivery_mean",
    "time_to_first_delivery_std",
    "time_to_completion_mean",
    "time_to_completion_std",
    "entropy_mean",
    "entropy_std",
    "advantage_mean",
    "advantage_std",
    "value_loss_per_epoch_mean",
    "value_loss_per_epoch_std",
    "actor_loss_per_epoch_mean",
    "actor_loss_per_epoch_std",
    "entropy_per_epoch_mean",
    "entropy_per_epoch_std",
    "q_value_magnitude_mean",
    "q_value_magnitude_std",
    "UTD",
]

matrix_pattern = re.compile(r".*(_e\d+_a\d+|_grid_\d+|_cdf_\d+)$")

def sort_keys(keys):
    ordered = []
    used = set()

    # 1. put all priority keys in order if they exist
    for k in PRIORITY_ORDER:
        if k in keys and k not in used:
            ordered.append(k)
            used.add(k)

    # 2. identify mean/std relations for remaining keys
    mean_keys = {}
    std_to_base = {}

    for k in keys:
        if k in used:
            continue
        if k.endswith("_mean"):
            base = k[:-5]
            mean_keys[base] = k
        elif k.endswith("_std"):
            base = k[:-4]
            std_to_base[k] = base

    # 3. add all means (even if no std), then matching std
    for base, mean_key in sorted(mean_keys.items()):
        if mean_key not in used:
            ordered.append(mean_key)
            used.add(mean_key)

        std_key = f"{base}_std"
        if std_key in std_to_base and std_key not in used:
            ordered.append(std_key)
            used.add(std_key)

    # 4. add std keys that have no matching mean (should be rare)
    for std_key, base in sorted(std_to_base.items()):
        if std_key not in used:
            ordered.append(std_key)
            used.add(std_key)

    # 5. split remaining into normal vs matrix/lorenz/axes
    matrix_like = []
    normal = []

    for k in keys:
        if k in used:
            continue
        if matrix_pattern.match(k):
            matrix_like.append(k)
        else:
            normal.append(k)

    # 6. add remaining normal keys (alphabetical)
    ordered.extend(sorted(normal))

    # 7. add matrix/lorenz keys last
    ordered.extend(sorted(matrix_like))

    return ordered


def test(
    E,
    N ,
    A,
    U = 1,
    P = 1,
    T = 1,
    O = 1,
    H0 = 1,
):

    episode_time = jnp.ones(
        (U,),
    )
    step_count_tensor_raw = jnp.ones(
        (U, E),
    )
    epoch_loss_tensor_raw = jnp.ones(
        (U, P),
    )
    step_time_tensor_raw = jnp.ones(
        (U, T, E),
    )
    block_tensor_raw = jnp.ones(
        (U, T, E, N),
    )
    rewards_tensor_raw = jnp.ones(
        (U, T, E, N),
    )
    actions_tensor_raw = jnp.ones(
        (U, T, E, N),
    )
    observation_tensor_raw = jnp.ones(
        (U, T, E, N, O),
    )
    deliveries_tensor_raw = jnp.ones(
        (U, T, E, N),
    )
    distance_traveled_tensor_raw = jnp.ones(
        (U, T, E, N),
    )
    epoch_old_logp_tensor_raw = jnp.ones(
        (U, T, E, N),
    )
    epoch_returns_tensor_raw = jnp.ones(
        (U, T, E, N),
    )
    idle_tensor_raw = jnp.ones(
        (U, T, E, N),
    )
    pickup_tensor_raw = jnp.ones(
        (U, T, E, N),
    )
    logits_tensor_raw = jnp.ones(
        (U, T, 1, E * N, A),
    )
    rewards_std_tensor_raw = jnp.ones(
        (U, T, E, N),
    )
    epoch_actor_loss_tensor_raw = jnp.ones(
        (U, P, T, E),
    )
    epoch_value_loss_tensor_raw = jnp.ones(
        (U, P, T, E),
    )
    epoch_advantage_tensor_raw = jnp.ones(
        (U, P, T, E, N),
    )
    epoch_entropy_tensor_raw = jnp.ones(
        (U, P, T, E, N),
    )
    epoch_logp_tensor_raw = jnp.ones(
        (U, P, T, E, N),
    )
    epoch_values_tensor_raw = jnp.ones(
        (U, P, T, E, N),
    )
    epoch_grads_tensor_raw = {
        "critic": {"params": {"Dense_0": {"kernel": np.ones((U, P, N * O, H0))}}},
        "actor": {"params": {"Dense_0": {"kernel": np.ones((U, P, O, H0))}}},
    }

    dummy_metric = {
        "episode_time": episode_time,
        "step_count_tensor_raw": step_count_tensor_raw,
        "epoch_loss_tensor_raw": epoch_loss_tensor_raw,
        "step_time_tensor_raw": step_time_tensor_raw,
        "block_tensor_raw": block_tensor_raw,
        "rewards_tensor_raw": rewards_tensor_raw,
        "actions_tensor_raw": actions_tensor_raw,
        "observation_tensor_raw": observation_tensor_raw,
        "deliveries_tensor_raw": deliveries_tensor_raw,
        "distance_traveled_tensor_raw": distance_traveled_tensor_raw,
        "epoch_old_logp_tensor_raw": epoch_old_logp_tensor_raw,
        "epoch_returns_tensor_raw": epoch_returns_tensor_raw,
        "idle_tensor_raw": idle_tensor_raw,
        "pickup_tensor_raw": pickup_tensor_raw,
        "logits_tensor_raw": logits_tensor_raw,
        "rewards_std_tensor_raw": rewards_std_tensor_raw,
        "epoch_actor_loss_tensor_raw": epoch_actor_loss_tensor_raw,
        "epoch_value_loss_tensor_raw": epoch_value_loss_tensor_raw,
        "epoch_advantage_tensor_raw": epoch_advantage_tensor_raw,
        "epoch_entropy_tensor_raw": epoch_entropy_tensor_raw,
        "epoch_logp_tensor_raw": epoch_logp_tensor_raw,
        "epoch_values_tensor_raw": epoch_values_tensor_raw,
        "epoch_grads_tensor_raw": epoch_grads_tensor_raw,
    }
    ########init after
    from metrics_funcs import create_dummy_metrics
    dummy_metric=create_dummy_metrics(10,4,5)
    init_columns = process_raw(dummy_metric, 1, 1, actionDim=A)
    key_list = [str(k) for k, _ in init_columns[0].items()]
    arrange_key_list=sort_keys(key_list)
    import os

    file_path = "test_results.csv"

    if os.path.exists(file_path):
        os.remove(file_path)

    logger = CSVLogger("test_results.csv", arrange_key_list, True)
    exit(0)
    ######

    metrics_loaded = load_metrics()

    list_dict = process_raw(metrics_loaded, 100, 0, actionDim=A)
    for dict in list_dict:
        logger.log(dict)
    


# for i, log in enumerate(logs):
#    for k, v in log.items():
#        print(f"({i}){k}:{v}")
import subprocess

def multi_run():
    subprocess.run([
        "powershell",
        "Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned"
    ])

    subprocess.run([
        "powershell",
        "& C:\\test\\.venv\\Scripts\\Activate.ps1"
    ])

    subprocess.run(["python", "-m", "scripts.train_mappo"])
    subprocess.run(["python", "-m", "scripts.train_mappo", "--algo", "maa2c"])
    subprocess.run(["python", "-m", "scripts.train_mappo", "--algo", "ippo"])
    subprocess.run(["python", "-m", "scripts.train_mappo", "--algo", "ia2c"])

multi_run()

#test(10,4,5)

