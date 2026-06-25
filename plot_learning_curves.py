# ——————————————————————————————————————————————
#  PLOT SYSTEM WITH META-PLOTS, SAVING, MULTI-ALG, MULTI-ENV
# ——————————————————————————————————————————————


import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from algorithms.metrics import COLUMNS
from pandas import Series


# ============================================================
# Graph categories
# ============================================================

GRAPH_CATEGORIES = {
    "performance": [
        "mean_episode_returns",
        "episode_time",
    ],
    "coordination": [
        "mean_deliveries",
        "mean_pickup_rate",
        "mean_block_rate",
        "mean_idle_rate",
        "mean_distance_traveled",
    ],
    "coordination_scatter": [("mean_block_rate", "mean_deliveries")],
    "success": [
        "mean_success",
        "mean_success_rate",
    ],
    "success_time": [("environment_steps", "mean_success_rate")],
    "system": [
        "mean_FPS",
        "mean_step_time",
        "mean_step_count",
        "updates",
    ],
    "system_scatter": [("updates", "mean_FPS")],
}
METRICS_LIST = [
    "Performance",
    "Stability",
    "Variance",
    "SensitivityToRandomness",
    "SampleEfficiency",
    "ConvergenceSpeed",
    "Coordination",
    "NormWeightedScore",
]

IGNORE_KEYS_LIST = [
    "algo_name",
    "size",
    "n_agents",
    "difficulty",
    "use_ppo",
]

# ============================================================
# Utility: smoothing
# ============================================================


def smooth(series: Series, window=50) -> Series:
    return series.rolling(window, min_periods=1).mean()


# ============================================================
# Config loading
# ============================================================


def extract_config_path(csv_path: str):
    p = Path(csv_path)
    config_path = p.parent.rglob("config.json")
    return str(config_path.__next__())


def load_cfg_json(path):
    if path is None:
        return {}
    with open(path, "r") as f:
        return json.load(f)


def flatten_dict(d, parent_key=""):
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}.{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key))
        else:
            items[new_key] = v
    return items


def checkMissingfiles(base: Path):

    missing = []
    for run_dir in base.iterdir():
        if run_dir.is_dir():
            # Search anywhere inside this run folder

            results = list(run_dir.rglob("results.csv"))
            config = list(run_dir.rglob("config.json"))

            missing_files = []
            if not results:
                missing_files.append("results.csv")
            if not config:
                missing_files.append("config.json")
            if missing_files:
                missing.append((f"{run_dir}", missing_files))
    # Print summary

    if missing:
        missings = ""
        for run, files in missing:
            missings = f"{run} is missing: {', '.join(files)}\n"
        assert False, f"{missings}"


# ============================================================
# Label builder (algorithm + env + differing params)
# ============================================================


def build_labels(csv_paths):
    configs = [flatten_dict(load_cfg_json(extract_config_path(p))) for p in csv_paths]

    all_keys = set().union(*configs)
    differing = []
    for key in all_keys:
        vals = [cfg.get(key, None) for cfg in configs]
        if None in vals:
            vals = [v for v in vals if v is not None]
        if len(set(map(str, vals))) > 1 and key not in IGNORE_KEYS_LIST:
            differing.append(key)
    labels = []

    for cfg in configs:
        parts = []
        algo = cfg.get("algo_name", "algo?")
        size = cfg.get("size", "env?")
        n_agents = cfg.get("n_agents", "env?")
        difficulty = cfg.get("difficulty", "env?")

        parts.append(f"{algo}@rware-{size}-{n_agents}ag-{difficulty}")
        for key in differing:
            if key in cfg:
                if "layout_path" in key:  # TODO :for now not exist
                    layout_name = Path(cfg[key]).stem
                    parts.append(f"{layout_name}")
                else:
                    if "lr" in key:
                        s = f"{cfg[key]:.0e}"
                        s = s.replace("e-0", "e-")  # .replace("e+0", "e+")
                        parts.append(f"{key}={s}")
                    else:
                        parts.append(f"{key}={cfg[key]}")
        labels.append(" | ".join(parts))
    return labels


# ============================================================
# Plot single metric
# ============================================================


def plot_learning_curves(
    csv_paths, metric_x, metric_y, window, save, save_dir, dateTime
):
    labels = build_labels(csv_paths)

    plt.figure(figsize=(10, 6))
    markers = ["o", "s", "D", "^", "v", "x", "+", "*"]

    for idx, (path, label) in enumerate(zip(csv_paths, labels)):
        df = pd.read_csv(path)
        if metric_x not in df or metric_y not in df:
            continue
        x = df[metric_x]
        # y = smooth(df[metric_y], window)

        if "mean" in metric_y:
            std = df[metric_y].replace("mean", "std")
            cv = std / (df[metric_y] + 1e-8)
            mean_s = smooth(df[metric_y], window)
            std_s = smooth(cv, window)
        else:
            mean = df[metric_y].replace("std", "mean")
            cv = mean / (df[metric_y] + 1e-8)
            mean_s = smooth(mean, window)
            std_s = smooth(df[metric_y], window)
        marker = markers[idx % len(markers)]

        plt.plot(
            x,
            mean_s,
            linewidth=2,
            label=label,
            marker=marker,
            markevery=max(len(x) // 30, 1),
            markersize=6,
        )

        plt.fill_between(x, mean_s - std_s, mean_s + std_s, alpha=0.2)
    plt.xlabel(metric_x)
    plt.ylabel(metric_y)
    plt.title(f"{metric_y}")
    plt.grid(True, alpha=1)
    plt.legend()
    plt.tight_layout()

    if save:
        out = Path(save_dir) / f"{metric_y}_{dateTime}.png"
        plt.savefig(out, dpi=200)
        print(f"Saved: {out}")
    plt.show()


# ============================================================
# Scatter plot
# ============================================================


def plot_scatter(csv_paths, x_metric, y_metric, save, save_dir, dateTime):
    labels = build_labels(csv_paths)
    plt.figure(figsize=(8, 6))

    for path, label in zip(csv_paths, labels):
        df = pd.read_csv(path)
        if x_metric not in df or y_metric not in df:
            continue
        plt.scatter(df[x_metric], df[y_metric], alpha=0.6, label=label)
    plt.xlabel(x_metric)
    plt.ylabel(y_metric)
    plt.title(f"{y_metric} vs {x_metric}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if save:
        out = Path(save_dir) / f"{y_metric}_vs_{x_metric}_{dateTime}.png"
        plt.savefig(out, dpi=200)
        print(f"Saved: {out}")
    plt.show()


# ============================================================
# Plot category
# ============================================================


def plot_category(csv_paths, category, window, save, save_dir, dateTime):
    metrics = GRAPH_CATEGORIES.get(category)
    if not metrics:
        print("Unknown category")
        return
    for metric in metrics:
        if isinstance(metric, tuple):
            x, y = metric
            plot_scatter(csv_paths, x, y, save, save_dir, dateTime)
        else:
            plot_learning_curves(
                csv_paths, "environment_steps", metric, window, save, save_dir, dateTime
            )


# ============================================================
# Meta-metrics
# ============================================================


def group_runs_by_label(csv_paths):
    labels = build_labels(csv_paths)
    groups = {}
    for path, label in zip(csv_paths, labels):
        groups.setdefault(label, []).append(pd.read_csv(path))
    return groups


def compute_meta_for_group(dfs):
    steps = dfs[0]["environment_steps"].values
    mean_episode_returns = []
    std_returns = []
    mean_success_rate = []
    for df in dfs:
        df = df.sort_values("environment_steps")
        if "mean_episode_returns" in df:
            mean_episode_returns.append(
                np.interp(steps, df["environment_steps"], df["mean_episode_returns"])
            )
        if "std_episode_returns" in df:
            std_returns.append(
                np.interp(steps, df["environment_steps"], df["std_episode_returns"])
            )
        if "mean_success_rate" in df:
            mean_success_rate.append(
                np.interp(steps, df["environment_steps"], df["mean_success_rate"])
            )
    returns = np.array(mean_episode_returns)
    success = np.array(mean_success_rate)
    std_returns = np.array(std_returns)
    metrics = {}

    # Performance

    metrics["Performance"] = float(np.mean(returns[:, -1]))

    # Stability

    metrics["Stability"] = float(np.mean(np.std(returns, axis=0)))
    # metrics["Stability"] = float(np.mean(np.std(returns, axis=1)))

    # Variance

    metrics["Variance"] = float(np.std(returns[:, -1]))

    # Sensitivity

    metrics["SensitivityToRandomness"] = float(np.std(np.diff(returns, axis=1)))

    # SampleEfficiency

    sr = np.mean(success, axis=0)
    # print("df="+f"{df['success_rate']}")

    if np.any(sr >= 0.8):
        idx = np.argmax(sr >= 0.8)
        metrics["SampleEfficiency"] = int(steps[idx])
    else:
        metrics["SampleEfficiency"] = np.nan
    # ConvergenceSpeed

    max_sr = np.max(sr)
    if max_sr > 0:
        target = 0.9 * max_sr
        if np.any(sr >= target):
            idx = np.argmax(sr >= target)
            metrics["ConvergenceSpeed"] = int(steps[idx])
        else:
            metrics["ConvergenceSpeed"] = np.nan
    else:
        metrics["ConvergenceSpeed"] = np.nan
    # Coordination

    last_coll = []
    last_deliv = []
    for df in dfs:
        if "mean_block_rate" in df:
            last_coll.append(df["mean_block_rate"].values[-1])
        if "mean_deliveries" in df:
            last_deliv.append(df["mean_deliveries"].values[-1])
    if last_coll and last_deliv:
        metrics["Coordination"] = float(np.mean(last_deliv) / (1 + np.mean(last_coll)))
    else:
        metrics["Coordination"] = np.nan
    # NormWeightedScore Normalize metrics to [0,1]

    perf = metrics["Performance"]
    stab = metrics["Stability"]
    samp = metrics["SampleEfficiency"]
    conv = metrics["ConvergenceSpeed"]

    # Higher performance is better

    perf_norm = 1 / (1 + np.exp(-perf / 50))  # logistic normalization

    # Lower stability (std) is better

    stab_norm = 1 / (1 + stab)

    # Lower sample efficiency is better (fewer steps to reach 0.8)

    samp_norm = 1 / (1 + samp / 1e6) if not np.isnan(samp) else 0

    # Lower convergence speed is better

    conv_norm = 1 / (1 + conv / 1e6) if not np.isnan(conv) else 0

    # Combine

    metrics["NormWeightedScore"] = np.mean([perf_norm, stab_norm, samp_norm, conv_norm])

    return metrics


def plot_meta(csv_paths, save, save_dir, meta_plot, dateTime):
    groups = group_runs_by_label(csv_paths)

    results = {label: compute_meta_for_group(dfs) for label, dfs in groups.items()}

    for metric in METRICS_LIST:
        if isinstance(meta_plot, str) and meta_plot != metric:
            continue
        plt.figure(figsize=(10, 5))
        labels = list(results.keys())
        values = [results[l][metric] for l in labels]

        plt.bar(labels, values)
        plt.title(metric)
        plt.xticks(rotation=45, ha="right")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        if save:
            out = Path(save_dir) / f"meta_{metric}_{dateTime}.png"
            plt.savefig(out, dpi=200)
            print(f"Saved: {out}")
        plt.show()


# ============================================================
# Main
# ============================================================


def main():
    parser = argparse.ArgumentParser(
        description="Advanced MARL Plotting Tool — supports learning curves, categories, meta-metrics, meta-plots, saving, and multi-run comparisons.",
        # add_help=True,
    )

    parser.add_argument(
        "--path",
        required=True,
        type=str,
        help="Root directory (Relative path) that containing experiment runs. The script will recursively search for results.csv files.",
    )

    parser.add_argument(
        "--smooth",
        type=int,
        default=50,
        help="Smoothing window size for learning curves (moving average). Default: 50.",
    )

    parser.add_argument(
        "--metric",
        type=str,
        default="mean_episode_returns",
        help="Plot a single metric over environment_steps. Default: mean_episode_returns.",
    )

    parser.add_argument(
        "--category",
        type=str,
        default=False,
        help=(
            "Plot a predefined category of metrics. Options:\n"
            + ",\n ".join(GRAPH_CATEGORIES.keys())
            + "."
            + " Default: performance"
        ),
    )

    parser.add_argument(
        "--meta",
        nargs="?",  # one argument is optional
        const=True,  # value when user writes only --meta
        default=False,  # value when --meta is not used at all
        help="Compute and print meta-metrics across grouped runs. Options:\n"
        + ",\n ".join(METRICS_LIST)
        + "."
        + "Note: Stability and Variance are only meaningful in multi runs.",
    )

    parser.add_argument(
        "--meta-plot",
        nargs="?",  # one argument is optional
        const=True,  # value when user writes only --meta-plot
        default=False,  # value when --meta-plot is not used at all
        help="Plot meta-metrics as bar charts across algorithms/environments. Options:\n"
        + "same as --meta.",
    )

    parser.add_argument(
        "--save",
        action="store_true",
        help="Save all generated plots as PNG files instead of only displaying them.",
    )

    parser.add_argument(
        "--save-dir",
        type=str,
        default="plots",
        help="Directory where plots will be saved when using --save. Default: 'plots'.",
    )

    args = parser.parse_args()

    now = datetime.now()
    time = now.strftime("%H-%M")
    date = now.strftime("%Y-%m-%d")
    dateTime = date + time

    base = Path(args.path)
    checkMissingfiles(base)

    results = [str(p) for p in base.rglob("results.csv")]

    if args.save:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    if args.meta:
        groups = group_runs_by_label(results)
        for label, dfs in groups.items():
            print(f"\n=== META: {label} ===")
            metrics = compute_meta_for_group(dfs)
            for k, v in metrics.items():
                if isinstance(args.meta, str) and args.meta != k:
                    continue
                print(f"{k:25s}: {v}")
        return
    if args.meta_plot:
        plot_meta(results, args.save, args.save_dir, args.meta_plot, dateTime)
        return
    if args.category:
        plot_category(
            results, args.category, args.smooth, args.save, args.save_dir, dateTime
        )
        return
    if args.metric and args.metric not in COLUMNS:
        columans = COLUMNS
        columans.remove("environment_steps")
        columans.remove("updates")
        columans.sort()
        print(f"unknown metric {args.metric!r}; choose from")
        for name in columans:
            print(name)
        return
    if args.metric:
        plot_learning_curves(
            results,
            "environment_steps",
            args.metric,
            args.smooth,
            args.save,
            args.save_dir,
            dateTime,
        )
        return
    print("Use --metric, --category, --meta, or --meta-plot")


if __name__ == "__main__":
    main()
