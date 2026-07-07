"""Read training-run artifacts for the dashboard and the LLM agents.

A *run* is a directory under ``runs/`` holding the contract the trainers write
(see scripts/train_mappo.py, algorithms/checkpoint.py, algorithms/metrics.py):

    config.json    -- the frozen MAPPOConfig (algorithm + env + hyperparameters)
    results.csv    -- per-update metrics (algorithms/metrics.py:COLUMNS)
    checkpoints/   -- orbax step subdirectories (numeric names)
    rollout_*.{gif,mp4} | <dir>/frame_*.png  -- optional rendered episode(s)

This module is deliberately **dependency-light** (pandas + stdlib only) so the
Streamlit app and the agents share ONE data contract without importing
jax/flax/orbax. The compact :class:`RunSummary` produced here -- not the raw
CSV -- is what the LLM agents consume.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


# ---------------------------------------------------------------------------
# algorithm identity + presentation
# ---------------------------------------------------------------------------
# The five contestants. Order is the canonical display/medal order; the colour
# is reused everywhere (charts, leaderboard) so each algorithm reads the same.
ALGO_ORDER = ["ia2c", "ippo", "maa2c", "mappo", "seac"]
ALGO_LABELS = {
    "ia2c": "IA2C",
    "ippo": "IPPO",
    "maa2c": "MAA2C",
    "mappo": "MAPPO",
    "seac": "SEAC",
}
ALGO_COLORS = {
    "ia2c": "#4C78A8",  # blue
    "ippo": "#54A24B",  # green
    "maa2c": "#E45756",  # red
    "mappo": "#F58518",  # orange
    "seac": "#B279A2",  # purple
}
_UNKNOWN_COLOR = "#8C8C8C"

# Metric columns grouped for tidy dashboard panels. Keys are CSV column names
# (algorithms/metrics.py:COLUMNS); values are human labels.
FULL_METRIC_GROUPS: dict[str, dict[str, str]] = {
    "Performance": {
        "episode_returns_mean": "Episode Return Mean",
        "episode_returns_std": "Episode Return Std",
        "returns_mean": "Return Mean",
        "returns_std": "Return Std",
        "returns_percentile_stats_p10": "Return P10",
        "returns_percentile_stats_p50": "Return Median (P50)",
        "returns_percentile_stats_p90": "Return P90",
        "returns_skew": "Return Skewness",
    },
    "Task success": {
        "success_mean": "Success Mean",
        "success_std": "Success Std",
        "success_rate_mean": "Success Rate Mean",
        "success_rate_std": "Success Rate Std",
        "deliveries_mean": "Deliveries Mean",
        "deliveries_std": "Deliveries Std",
    },
    "Coordination": {
        "deliveries_early_mean": "Early Deliveries Mean",
        "deliveries_early_std": "Early Deliveries Std",
        "deliveries_mid_mean": "Mid Deliveries Mean",
        "deliveries_mid_std": "Mid Deliveries Std",
        "deliveries_late_mean": "Late Deliveries Mean",
        "deliveries_late_std": "Late Deliveries Std",
    },
    "Behaviour": {
        "block_rate_mean": "Block Rate Mean",
        "block_rate_std": "Block Rate Std",
        "block_early_mean": "Early Block Rate Mean",
        "block_early_std": "Early Block Rate Std",
        "block_mid_mean": "Mid Block Rate Mean",
        "block_mid_std": "Mid Block Rate Std",
        "block_late_mean": "Late Block Rate Mean",
        "block_late_std": "Late Block Rate Std",
        "idle_rate_mean": "Idle Rate Mean",
        "idle_rate_std": "Idle Rate Std",
        "pickup_rate_mean": "Pickup Rate Mean",
        "pickup_rate_std": "Pickup Rate Std",
        "distance_traveled_mean": "Distance Traveled Mean",
        "distance_traveled_std": "Distance Traveled Std",
        "distance_traveled_agent": "Agent Distance Traveled",  # shape (N,)
    },
    "Temporal efficiency": {
        # scalars or jnp.finfo(jnp.float32).max
        "time_to_completion_mean": "Time to Completion Mean",
        "time_to_completion_std": "Time to Completion Std",
        "time_to_completion_p10": "Time to Completion P10",
        "time_to_completion_p50": "Time to Completion Median (P50)",
        "time_to_completion_p90": "Time to Completion P90",
        "time_to_first_delivery_mean": "Time to First Delivery Mean",
        "time_to_first_delivery_std": "Time to First Delivery Std",
        "time_to_first_delivery_p10": "Time to First Delivery P10",
        "time_to_first_delivery_p50": "Time to First Delivery Median (P50)",
        "time_to_first_delivery_p90": "Time to First Delivery P90",
    },
    "Completion curves": {
        "episode_time": "Episode Time",
        "step_count_mean": "Step Count Mean",
        #"step_count_std": "Step Count Std",
        "time_to_completion_cdf_grid": "Completion CDF Grid",  # shape (E,)
        #"time_to_completion_cdf": "Completion CDF",  # shape (E,)
        #"time_to_completion_bootstrap_CI_on_lower_bound_of_median": "Completion Median CI Lower",
        #"time_to_completion_bootstrap_CI_on_upper_bound_of_median": "Completion Median CI Upper",
        #"time_to_first_delivery_cdf": "First Delivery CDF",  # shape (E,)
        "time_to_first_delivery_cdf_grid": "First Delivery CDF Grid",  # shape (E,)
        #"time_to_first_delivery_bootstrap_CI_on_lower_bound_of_median": "First Delivery Median CI Lower",
        #"time_to_first_delivery_bootstrap_CI_on_upper_bound_of_median": "First Delivery Median CI Upper",
    },
    "Fairness outcomes": {
        # only mean scalar or jnp.finfo(jnp.float32).max
        "fairness_deliveries_gini_mean": "Delivery Gini Mean",
        "fairness_deliveries_gini_std": "Delivery Gini Std",
        "fairness_rewards_gini_mean": "Reward Gini Mean",
        "fairness_rewards_gini_std": "Reward Gini Std",
    },
    "Fairness distributions per agent": {
        # scalars or jnp.finfo(jnp.float32).max
        "fairness_deliveries_lorenz_x_agent": "Agent Delivery Lorenz X",  # shape (N,)
        "fairness_rewards_lorenz_x_agent": "Agent Reward Lorenz X",  # shape (N,)
        #"fairness_rewards_lorenz_y_agent": "Agent Reward Lorenz Y",  # shape (E,N)
        #"fairness_deliveries_lorenz_y_agent": "Agent Delivery Lorenz Y",  # shape (E,N)
    },
    "Credit correlations": {
        "credit_correlations_mean": "Credit Correlation Mean",
        "credit_correlations_std": "Credit Correlation Std",
        "credit_agent_correlations": "Agent Credit Correlation",  # shape (N,)
    },
    "Credit shapley": {
        "credit_shapley_mean": "Shapley Value Mean",
        "credit_shapley_std": "Shapley Value Std",
        "credit_shapley_loo_mean": "LOO Shapley Mean",
        "credit_shapley_loo_std": "LOO Shapley Std",
        "credit_agent_shapley": "Agent Shapley Value",  # shape (N,)
        "credit_agent_shapley_loo": "Agent LOO Shapley",  # shape (N,)
    },
    "Advantage statistics": {
        "advantage_mean": "Advantage Mean",
        "advantage_std": "Advantage Std",
        "advantage_p10": "Advantage P10",
        "advantage_p50": "Advantage Median (P50)",
        "advantage_p90": "Advantage P90",
        "advantage_skew": "Advantage Skewness",
    },
    "Value statistics": {
        "q_value_magnitude_mean": "Q-Value Magnitude Mean",
        "q_value_magnitude_std": "Q-Value Magnitude Std",
        "q_value_trend": "Q-Value Trend",
        "returns_agent_mean": "Agent Return Mean",  # shape (N,)
        "returns_agent_std": "Agent Return Std",  # shape (N,)
    },
    "Loss statistics": {
        "loss_mean": "Loss Mean",
        "loss_std": "Loss Std",
        "loss_percentile_stats_p10": "Loss P10",
        "loss_percentile_stats_p50": "Loss Median (P50)",
        "loss_percentile_stats_p90": "Loss P90",
        "loss_skew": "Loss Skewness",
    },
    "Loss dynamics per epoch": {
        "actor_loss_mean": "Epochs Actor Loss Mean",
        "actor_loss_std": "Epochs Actor Loss Std",
        "actor_loss_trend": "Epochs Actor Loss Trend",
        "value_loss_mean": "Epochs Value Loss Mean",
        "value_loss_std": "Epochs Value Loss Std",
        "value_loss_trend": "Epochs Value Loss Trend",
        "ratio_mean": "Epochs Ratio Mean",
        "ratio_std": "Epochs Ratio Std",
        "ratio_trend": "Epochs Ratio Trend",
        "cumulative_ratio_mean": "Epochs Cumulative Ratio Mean",
        "cumulative_ratio_std": "Epochs Cumulative Ratio Std",
        "UTD": "Update-to-Data Ratio",  # scalar or jnp.finfo(jnp.float32).max
    },
    "Entropy global per epoch": {
        "entropy_mean": "Entropy Mean",
        "entropy_std": "Entropy Std",
        "entropy_p10": "Entropy P10",
        "entropy_p50": "Entropy Median (P50)",
        "entropy_p90": "Entropy P90",
        "entropy_trend": "Entropy Trend",
    },
    "Entropy per agent": {
        "entropy_agent_mean": "Agent Entropy Mean",  # shape (N,)
        "entropy_agent_std": "Agent Entropy Std",  # shape (N,)
        "entropy_agent_p10": "Agent Entropy P10",  # shape (N,)
        "entropy_agent_p50": "Agent Entropy Median (P50)",  # shape (N,)
        "entropy_agent_p90": "Agent Entropy P90",  # shape (N,)
        "entropy_agent_trend": "Agent Entropy Trend",  # shape (N,)
    },
    "Action distribution per agent": {
        "action_histogram_flat_agent": "Agent Action Histogram",  # shape (N, A)
        "action_entropy_agent": "Agent Action Entropy",  # shape (N,)
    },
    "Action similarity agents": { 
        "action_jsd": "Action Jensen-Shannon Divergence",  # shape (N, N)
    },
    "KL global": {
        "kl_over_P_mean": "KL Divergence Mean",
        "kl_over_P_std": "KL Divergence Std",
        "kl_p90_over_P": "KL Divergence P90",
    },
    "KL per parameter": {
        "kl_per_P_mean_mean": "KL per Parameter Mean",
        "kl_per_P_std_std": "KL per Parameter Std",
        "kl_per_P_p90_mean": "KL per Parameter P90 Mean",
        "kl_per_P_p90_max": "KL per Parameter P90 Max",
        "kl_per_P_p90_median": "KL per Parameter P90 Median",
    },
    "Gradient global": {
        "grad_norm_mean": "Gradient Norm Mean",
        "grad_norm_std": "Gradient Norm Std",
        "grad_norm_p90": "Gradient Norm P90",
    },
    "Gradient per agent (Only if centralised critic)": {
        "grad_norms_per_agent_mean": "Gradient Norm Agents Mean",
        "grad_norms_per_agent_std": "Gradient Norm Agents Std",
        "grad_norms_per_agent_p90_mean": "Gradient Norm Agents P90 Mean",
        "grad_norms_per_agent_p90_max": "Gradient Norm Agents P90 Max",
        "grad_norms_per_agent_p90_median": "Gradient Norm Agents P90 Median",
        "grad_norms_agent_p90": "Agent Gradient Norm P90",  # shape (N,)
        "grad_norms_agent_mean": "Agent Gradient Norm Mean",  # shape (N,)
        "grad_norms_agent_std": "Agent Gradient Norm Std",  # shape (N,)
    },
    "Rewards per agent": {
        "reward_agent_mean": "Agent Reward Mean",  # shape (N,)
        "reward_agent_std": "Agent Reward Std",  # shape (N,)
        "reward_agent_p10": "Agent Reward P10",  # shape (N,)
        "reward_agent_p50": "Agent Reward Median (P50)",  # shape (N,)
        "reward_agent_p90": "Agent Reward P90",  # shape (N,)
        "reward_agent_trend": "Agent Reward Trend",  # shape (N,)
    },
    "Reward standardise": {
        "reward_std_mean": "Reward Std Mean",
        "reward_std_std": "Reward Std Std",
    },
    "Environment metrics": {
        # scalars or jnp.finfo(jnp.float32).max
        "rware_mean": "Environment Reward Mean",
        "rware_std": "Environment Reward Std",
        "rware_p10": "Environment Reward P10",
        "rware_p50": "Environment Reward Median (P50)",
        "rware_p90": "Environment Reward P90",
    },
    "System performance": {
        "FPS_mean": "Frames per Second Mean",
        #"FPS_std": "Frames per Second Std",
    },
    "System timing": {
        "step_time_mean": "Step Time Mean",
        #"step_time_std": "Step Time Std",
        #"step_time_percentile_stats_p10": "Step Time P10",
        #"step_time_percentile_stats_p50": "Step Time Median (P50)",
        #"step_time_percentile_stats_p90": "Step Time P90",
    },
}

METRIC_GROUPS: dict[str, dict[str, str]] = {
    "Performance": {
        "episode_returns_mean": "Team return",
        "deliveries_mean": "Deliveries / episode",
    },
    "Behaviour": {
        "block_rate_mean": "Contention (forward-blocked)",
        "idle_rate_mean": "Idle rate",
        "pickup_rate_mean": "Pickup rate",
    },
    "Optimisation": {
        "entropy_mean": "Policy entropy",
        "loss_mean": "Total loss",
        "actor_loss_mean": "Actor loss",
        "value_loss_mean": "Value loss",
    },
    "Episode phase (deliveries)": {
        "deliveries_early_mean": "Early third",
        "deliveries_mid_mean": "Mid third",
        "deliveries_late_mean": "Late third",
    },
}

_MEDIA_EXTS = (".gif", ".mp4")


# ---------------------------------------------------------------------------
# identity helpers
# ---------------------------------------------------------------------------
def algo_family(run_name: str, config: Optional[dict] = None) -> str:
    """Map a run to one of ALGO_ORDER (or 'other').

    The directory name is authoritative because SEAC runs are stored as a
    MAPPOConfig (its `algo` property would mislabel them as 'mappo'); we parse
    the leading token of the run name first, then fall back to the config.
    """
    head = run_name.lower().replace("-", "_").split("_", 1)[0]
    if head in ALGO_LABELS:
        return head
    if config is not None:
        cc = bool(config.get("centralised_critic"))
        pp = bool(config.get("use_ppo"))
        flags = {
            (False, False): "ia2c",
            (False, True): "ippo",
            (True, False): "maa2c",
            (True, True): "mappo",
        }
        return flags.get((cc, pp), "other")
    return "other"


def algo_label(family: str) -> str:
    return ALGO_LABELS.get(family, family.upper() if family != "other" else "Other")


def algo_color(family: str) -> str:
    return ALGO_COLORS.get(family, _UNKNOWN_COLOR)


# ---------------------------------------------------------------------------
# the run record
# ---------------------------------------------------------------------------
@dataclass(eq=False)
class RunData:
    name: str  # directory basename
    path: str  # absolute path
    config: dict
    df: pd.DataFrame  # results.csv (may be empty)
    checkpoint_steps: list[int] = field(default_factory=list)
    media: list[str] = field(default_factory=list)  # rendered rollout files

    @property
    def family(self) -> str:
        return algo_family(self.name, self.config)

    @property
    def label(self) -> str:
        return algo_label(self.family)

    @property
    def color(self) -> str:
        return algo_color(self.family)

    @property
    def env_tag(self) -> str:
        c = self.config
        size = c.get("size", "?")
        n = c.get("n_agents", "?")
        diff = c.get("difficulty", "")
        diff = "" if diff in ("normal", "") else f"-{diff}"
        return f"rware-{size}-{n}ag{diff}"

    @property
    def updates(self) -> int:
        return int(self.df["updates"].max()) if len(self.df) else 0
    
    def __eq__(self, other):
        return isinstance(other, RunData) and (self.name, self.path) == (other.name, other.path)

    def __hash__(self):
        return hash((self.name, self.path))

# ---------------------------------------------------------------------------
# discovery + loading
# ---------------------------------------------------------------------------
def _numeric_subdirs(path: str) -> list[int]:
    if not os.path.isdir(path):
        return []
    return sorted(int(n) for n in os.listdir(path) if n.isdigit())


def _find_media(run_dir: str) -> list[str]:
    out = []
    for name in sorted(os.listdir(run_dir)):
        full = os.path.join(run_dir, name)
        if os.path.isfile(full) and name.lower().endswith(_MEDIA_EXTS):
            out.append(full)
    return out


def discover_runs(root: str = "runs") -> list[str]:
    """Absolute paths of every run dir under `root` that has a results.csv."""
    root = os.path.abspath(root)
    if not os.path.isdir(root):
        return []
    runs = []
    for name in sorted(os.listdir(root)):
        date_dir = os.path.join(root, name)
        if not os.path.isdir(date_dir):
            continue

        # NEW: look inside the date_dir for subfolders containing results.csv
        for sub in os.listdir(date_dir):
            d = os.path.join(date_dir, sub)
            if os.path.isdir(d) and os.path.isfile(os.path.join(d, "results.csv")):
                runs.append(d)

    return runs


def load_run(run_dir: str) -> RunData:
    """Load one run directory into a RunData (tolerant of missing pieces)."""
    run_dir = os.path.abspath(run_dir)
    name = os.path.basename(run_dir)

    config: dict = {}
    cfg_path = os.path.join(run_dir, "config.json")
    if os.path.isfile(cfg_path):
        with open(cfg_path) as f:
            config = json.load(f)

    csv_path = os.path.join(run_dir, "results.csv")
    try:
        df = pd.read_csv(csv_path)
        # interrupted-and-restarted runs can leave stray header rows mid-file;
        # coerce everything numeric and drop the casualties
        df = df.apply(pd.to_numeric, errors="coerce")
        df = df.dropna(subset=["updates"]) if "updates" in df.columns else df
        if "updates" in df.columns:
            df = df.sort_values("updates").reset_index(drop=True)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        df = pd.DataFrame()

    return RunData(
        name=name,
        path=run_dir,
        config=config,
        df=df,
        checkpoint_steps=_numeric_subdirs(os.path.join(run_dir, "checkpoints")),
        media=_find_media(run_dir),
    )


def load_runs(root: str = "runs") -> list[RunData]:
    return [load_run(d) for d in discover_runs(root)]


# ---------------------------------------------------------------------------
# compact summary for the LLM agents (NEVER pass raw CSV to a model)
# ---------------------------------------------------------------------------
def _tail_mean(series: pd.Series, frac: float = 0.1) -> float:
    """Mean over the final `frac` of a series (robust 'final' value)."""
    if len(series) == 0:
        return 0.0
    k = max(1, int(len(series) * frac))
    return float(series.tail(k).mean())


def _steps_to_threshold(df: pd.DataFrame, col: str, thresh: float) -> Optional[int]:
    """First environment_steps at which `col` reaches `thresh` (sample efficiency)."""
    if col not in df.columns or "environment_steps" not in df.columns:
        return None
    hit = df[df[col] >= thresh]
    return int(hit["environment_steps"].iloc[0]) if len(hit) else None


def leaderboard(runs: list["RunData"]) -> pd.DataFrame:
    """One row per run, the columns the Overview tab ranks/medals on."""
    rows = []
    for r in runs:
        s = summarize(r)
        rows.append(
            {
                "run": r.name,
                "algo": r.label,
                "family": r.family,
                "env": s.get("env", ""),
                "env_steps": s.get("env_steps", 0),
                "final_return": s.get("final_return", 0.0),
                "best_return": s.get("best_return", 0.0),
                "deliveries": s.get("final_deliveries", 0.0),
                "steps_to_half_best": s.get("steps_to_half_best"),
                "contention": s.get("final_contention", 0.0),
                "idle": s.get("final_idle", 0.0),
                "entropy": s.get("final_entropy", 0.0),
            }
        )
    return pd.DataFrame(rows)


def summarize(run: RunData) -> dict:
    """Compact, grounded summary of a run for LLM consumption.

    Every field is a measured number; the agents must justify claims against
    these (no invented intent). Categories map directly onto competition awards.
    """
    df = run.df
    out: dict = {
        "run": run.name,
        "algorithm": run.label,
        "env": run.env_tag,
        "seed": run.config.get("seed"),
        "use_rnn": run.config.get("use_rnn"),
        "updates": run.updates,
        "env_steps": int(df["environment_steps"].max()) if len(df) else 0,
    }
    if not len(df):
        out["note"] = "no metrics logged yet"
        return out

    ret = df["episode_returns_mean"]
    final_ret = _tail_mean(ret)
    out["final_return"] = round(final_ret, 3)
    out["best_return"] = round(float(ret.max()), 3)
    out["final_deliveries"] = round(_tail_mean(df["deliveries_mean"]), 3)
    out["final_contention"] = round(_tail_mean(df["block_rate_mean"]), 4)
    out["final_idle"] = round(_tail_mean(df["idle_rate_mean"]), 4)
    out["return_volatility"] = round(float(ret.tail(max(1, len(ret) // 5)).std()), 3)
    # late collapse: did the tail drop well below the peak?
    out["peak_to_final_drop"] = round(float(ret.max()) - final_ret, 3)
    # milestone thresholds are taken on a smoothed signal so one lucky episode
    # doesn't count as a milestone
    sm = df[["environment_steps"]].copy()
    win = max(1, min(50, len(df) // 4))
    sm["episode_returns_mean"] = ret.rolling(win, min_periods=win).mean()
    sm["deliveries_mean"] = df["deliveries_mean"].rolling(win, min_periods=win).mean()
    # sample efficiency: steps to reach 50% of this run's own best (smoothed)
    out["steps_to_half_best"] = _steps_to_threshold(
        sm, "mean_episode_returns", 0.5 * float(sm["episode_returns_mean"].max())
    )
    # first sustained deliveries
    out["steps_to_first_delivery"] = _steps_to_threshold(sm, "deliveries_mean", 0.5)
    # phase tilt of throughput
    e = _tail_mean(df["deliveries_early_mean"])
    m = _tail_mean(df["deliveries_mid_mean"])
    l = _tail_mean(df["deliveries_late_mean"])
    out["delivery_phase_split"] = {
        "early": round(e, 2),
        "mid": round(m, 2),
        "late": round(l, 2),
    }
    out["final_entropy"] = round(_tail_mean(df["entropy_mean"]), 3)
    return out
