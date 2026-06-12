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
    "ia2c": "#4C78A8",   # blue
    "ippo": "#54A24B",   # green
    "maa2c": "#E45756",  # red
    "mappo": "#F58518",  # orange
    "seac": "#B279A2",   # purple
}
_UNKNOWN_COLOR = "#8C8C8C"

# Metric columns grouped for tidy dashboard panels. Keys are CSV column names
# (algorithms/metrics.py:COLUMNS); values are human labels.
METRIC_GROUPS: dict[str, dict[str, str]] = {
    "Performance": {
        "mean_episode_returns": "Team return",
        "deliveries": "Deliveries / episode",
    },
    "Behaviour": {
        "block_rate": "Contention (forward-blocked)",
        "idle_rate": "Idle rate",
        "pickup_rate": "Pickup rate",
    },
    "Optimisation": {
        "entropy": "Policy entropy",
        "loss": "Total loss",
        "actor_loss": "Actor loss",
        "value_loss": "Value loss",
    },
    "Episode phase (deliveries)": {
        "deliveries_early": "Early third",
        "deliveries_mid": "Mid third",
        "deliveries_late": "Late third",
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
        cc = config.get("centralised_critic")
        pp = config.get("use_ppo")
        flags = {(False, False): "ia2c", (False, True): "ippo",
                 (True, False): "maa2c", (True, True): "mappo"}
        return flags.get((cc, pp), "other")
    return "other"


def algo_label(family: str) -> str:
    return ALGO_LABELS.get(family, family.upper() if family != "other" else "Other")


def algo_color(family: str) -> str:
    return ALGO_COLORS.get(family, _UNKNOWN_COLOR)


# ---------------------------------------------------------------------------
# the run record
# ---------------------------------------------------------------------------
@dataclass
class RunData:
    name: str                       # directory basename
    path: str                       # absolute path
    config: dict
    df: pd.DataFrame                # results.csv (may be empty)
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
        d = os.path.join(root, name)
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
        rows.append({
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
        })
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

    ret = df["mean_episode_returns"]
    final_ret = _tail_mean(ret)
    out["final_return"] = round(final_ret, 3)
    out["best_return"] = round(float(ret.max()), 3)
    out["final_deliveries"] = round(_tail_mean(df["deliveries"]), 3)
    out["final_contention"] = round(_tail_mean(df["block_rate"]), 4)
    out["final_idle"] = round(_tail_mean(df["idle_rate"]), 4)
    out["return_volatility"] = round(float(ret.tail(max(1, len(ret) // 5)).std()), 3)
    # late collapse: did the tail drop well below the peak?
    out["peak_to_final_drop"] = round(float(ret.max()) - final_ret, 3)
    # sample efficiency: steps to reach 50% of this run's own best return
    out["steps_to_half_best"] = _steps_to_threshold(
        df, "mean_episode_returns", 0.5 * float(ret.max()))
    # first delivery
    out["steps_to_first_delivery"] = _steps_to_threshold(df, "deliveries", 0.5)
    # phase tilt of throughput
    e = _tail_mean(df["deliveries_early"])
    m = _tail_mean(df["deliveries_mid"])
    l = _tail_mean(df["deliveries_late"])
    out["delivery_phase_split"] = {
        "early": round(e, 2), "mid": round(m, 2), "late": round(l, 2)}
    out["final_entropy"] = round(_tail_mean(df["entropy"]), 3)
    return out
