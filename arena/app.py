"""Arena dashboard (Streamlit) -- M2: overview/leaderboard, smoothed compare,
animated replay theater, run detail.

Run from the repo root (WSL, conda env jax_env_1):

    streamlit run arena/app.py

Reads run artifacts via arena.run_data / arena.replay_data (pandas/numpy +
stdlib only -- no jax import), so the dashboard starts instantly and can run on
a machine without the training stack. Dark mission-control theme lives in
.streamlit/config.toml; the plotly side of it is `_styled` below.
"""

from __future__ import annotations

import math
import os
import random
import sys

import time

# `streamlit run arena/app.py` puts arena/ (not the repo root) on sys.path, so
# `import arena.*` would fail. Add the repo root explicitly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from arena.player import player_height, player_html
from arena.replay_data import list_replays, load_payload
from arena.run_data import (
    discover_runs,
    FULL_METRIC_GROUPS,
    leaderboard,
    load_run,
    METRIC_GROUPS,
    RunData,
    summarize,
)
from arena.run_data_handler import process_metrics, sanitize_masked, sanitize_sentinel

st.set_page_config(page_title="MARL Arena", page_icon="🤖", layout="wide")

ACCENT = "#9db6e6"
GRID = "#5a6c87"
MAX_POINTS = 1500  # per trace; long runs are stride-downsampled for the browser


# ---------------------------------------------------------------------------
# data loading (cached on the runs root + dir mtimes so new runs show up)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _load(run_dir: str, _mtime: float) -> RunData:
    return load_run(run_dir)


def _load_all(root: str) -> list[RunData]:
    runs = []
    for d in discover_runs(root):
        # include results.csv mtime in the cache key -> live runs refresh
        csv = os.path.join(d, "results.csv")
        mtime = os.path.getmtime(csv) if os.path.isfile(csv) else 0.0
        runs.append(_load(d, mtime))
    return runs


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------
def _styled(fig: go.Figure, title: str, height: int = 320) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=13)),
        height=height,
        margin=dict(l=10, r=10, t=42, b=10),
        font=dict(family="Consolas, monospace", size=11),
        xaxis=dict(gridcolor=GRID,zeroline=False),
        yaxis=dict(gridcolor=GRID, zeroline=False),
        legend=dict(orientation="h"),
        hoverlabel=dict(font_family="Consolas, monospace"),
        xaxis_title_standoff=30,
    )
    return fig

def _legend_names(runs: list[RunData] | RunData) -> dict[str, str]:
    """Short algo label when unique among the selection, else the run name."""
    if isinstance(runs, list):
        fams = [r.family for r in runs]
        return {
            r.name: (r.label if fams.count(r.family) == 1 else r.name) for r in runs
        }
    else:
        return {r.name: r.label}


def _smooth_series(series: pd.Series, smooth: int):
    if smooth > 1:
        return series.ewm(span=smooth, min_periods=1).mean()
    return series


def jsd_heatmap_fig(
    r: RunData, col: str, title: str, smooth: int, show_raw: bool
) -> go.Figure:
    fig = go.Figure()
    names = _legend_names(r)

    def build_scale(base_hex):
        import colorsys

        def hex_to_rgb(hex_color):
            hex_color = hex_color.lstrip("#")
            return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))

        def rgb_to_hex(rgb):
            return "#{:02x}{:02x}{:02x}".format(*rgb)

        def adjust_lightness(rgb, factor):
            r, g, b = [x / 255 for x in rgb]
            h, l, s = colorsys.rgb_to_hls(r, g, b)
            l = max(0, min(1, l * factor))
            r2, g2, b2 = colorsys.hls_to_rgb(h, l, s)
            return rgb_to_hex((int(r2 * 255), int(g2 * 255), int(b2 * 255)))

        rgb = hex_to_rgb(base_hex)
        factors = [0.25, 0.45, 0.65, 0.8, 1.0]  # dark -> brighter
        return [
            [i / (len(factors) - 1), adjust_lightness(rgb, f)]
            for i, f in enumerate(factors)
        ]

    orange_scale = build_scale(r.color)

    cols = [c for c in r.df.columns if c.startswith(col)]
    if not cols:
        assert "cols not found on CSV header"

    agents = sorted({int(c.split("_")[2][1:]) for c in cols})

    mat = []
    for i in agents:
        row = []
        for j in agents:
            cname = f"{col}_a{i}_a{j}"
            if cname in r.df:
                value = r.df[cname].sum()
            else:
                value = None
            row.append(value)
        mat.append(row)

    fig.add_trace(
        go.Heatmap(
            z=mat,
            x=[f"Agent {j}" for j in agents],
            y=[f"Agent {i}" for i in agents],
            colorscale=orange_scale,
            name=names[r.name],
            showscale=True,
            text=mat,
            texttemplate="%{text:.3f}",
        )
    )

    return _styled(fig, title)


def lorenz_curve_fig(
    r: RunData, col: str, title: str, smooth: int, show_raw: bool
) -> go.Figure:
    fig = go.Figure()
    names = _legend_names(r)
    agent = col[-1]

    bootstrap_col = col.split("_")[0] + "_" + col.split("_")[1]  # fairness_something
    cols = [c for c in r.df.columns if c.startswith(bootstrap_col) and "lorenz" in c]
    if not cols:
        assert "cols not found on CSV header"

    x = sanitize_sentinel(r.df[col], True)

    episodes = sorted({int(c.split("_")[-1][1]) for c in cols if "_lorenz_y_" in c})

    for e in episodes:
        y_raw = sanitize_sentinel(
            r.df[f"{bootstrap_col}_lorenz_y_agent_{agent}_e{e}"], True
        )
        y = _smooth_series(y_raw, smooth)

        if show_raw and smooth > 1:
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y_raw,
                    mode="lines",
                    showlegend=False,
                    line=dict(color=r.color, width=1),
                    opacity=0.18,
                    hoverinfo="skip",
                )
            )

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name=f"{names[r.name]} ep{e}",
                line=dict(color=r.color, width=2.2),
            )
        )

    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(color=r.color, dash="dash"),
            showlegend=False,
        )
    )
    fig.update_layout(
        title=dict(
            text=title,
            x=0.05,
        ),
        xaxis_title="fraction of agents",
        yaxis_title="fraction of cumulative contribution",
        xaxis_title_standoff=50,
        yaxis_title_standoff=40,
    )

    return _styled(fig, title + f" Agent {col[-1]}")


def cdf_fig(r: RunData, col: str, title: str, smooth: int, show_raw: bool) -> go.Figure:
    fig = go.Figure()
    names = _legend_names(r)
    col = col[:-9]  # remove _cdf_grid

    grid_cols = sorted(
        [c for c in r.df.columns if f"{col}_cdf_grid_" in c],
        key=lambda x: int(x.split("_")[-1]),
    )
    cdf_cols = sorted(
        [c for c in r.df.columns if f"{col}_cdf_" in c and "grid" not in c],
        key=lambda x: int(x.split("_")[-1]),
    )
    if not grid_cols or not cdf_cols:
        assert "grid_cols or cdf_cols not found on CSV header"

    # sort grid
    grid = sanitize_sentinel(r.df[grid_cols], False)

    # build CDF from all values
    y_raw = sanitize_sentinel(r.df[cdf_cols], False)

    y = _smooth_series(y_raw, smooth)
    if show_raw and smooth > 1:
        fig.add_trace(
            go.Scatter(
                x=grid,
                y=y_raw,
                mode="lines",
                showlegend=False,
                line=dict(color=r.color, width=1),
                opacity=0.18,
                hoverinfo="skip",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=grid,
            y=y_raw,
            mode="lines",
            name=names[r.name],
            line=dict(color=r.color, width=2.2),
        )
    )

    # median line
    median_x = next((x for x, yv in zip(grid, y) if yv >= 0.5), None)
    if median_x is not None:
        fig.add_trace(
            go.Scatter(
                x=[median_x, median_x],
                y=[0, 1],
                mode="lines+text",
                line=dict(color=r.color, width=1, dash="dot"),
                showlegend=False,
                text=[f"Median: {median_x:.3f}", ""],
                textposition="bottom center",
                textfont=dict(size=14),
                hoverinfo="skip",
            )
        )

    # CI shading
    lo = f"{col}_bootstrap_CI_on_lower_bound_of_median"
    hi = f"{col}_bootstrap_CI_on_upper_bound_of_median"

    if lo in r.df and hi in r.df:
        lo_vals = sanitize_masked(r.df[lo])
        hi_vals = sanitize_masked(r.df[hi])
        lo_v = lo_vals.mean()
        hi_v = hi_vals.mean()
        # lower CI line
        fig.add_trace(
            go.Scatter(
                x=[lo_v, lo_v],
                y=[0, 1],
                mode="lines+text",
                line=dict(color=r.color, width=2, dash="dot"),
                text=[f"{lo_v:.3f}", ""],
                textposition="bottom center",
                textfont=dict(size=14),
                showlegend=False,
                hoverinfo="skip",
            )
        )

        # upper CI line
        fig.add_trace(
            go.Scatter(
                x=[hi_v, hi_v],
                y=[0, 1],
                mode="lines+text",
                line=dict(color=r.color, width=2, dash="dot"),
                text=[f"{hi_v:.3f}", ""],
                textposition="bottom center",
                textfont=dict(size=14),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    fig.update_layout(xaxis_title="CDF grid", yaxis_title="CDF")
    return _styled(fig, title)


def action_histogram_fig(
    r: RunData, col: str, title: str, smooth: int, show_raw: bool
) -> go.Figure:
    fig = go.Figure()
    names = _legend_names(r)
    agent = col[-1]

    # find all bins for this agent
    cols = sorted(
        [c for c in r.df.columns if c.startswith(col)],
        key=lambda x: int(x.split("_")[-1][1]),
    )

    if not cols:
        assert "cols not found on CSV header"

    # x-axis = bin indices
    x_vals = [int(c.split("_")[-1][1:]) for c in cols]

    y_raw = r.df[cols].mean(axis=0)
    # raw + smooth values
    y = _smooth_series(y_raw, smooth)

    # raw line (optional)
    if show_raw and smooth > 1:
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_raw,
                mode="lines",
                showlegend=False,
                line=dict(color=r.color, width=1),
                opacity=0.18,
                hoverinfo="skip",
            )
        )

    # main bar chart
    fig.add_trace(
        go.Bar(
            x=x_vals,
            y=y_raw,
            name=names[r.name],
            marker=dict(color=r.color),
            text=[f"{v:.2f}" for v in y],  # show values above bars
            textposition="outside",
            textfont=dict(size=12),
        )
    )

    fig.update_layout(
        xaxis_title=f"Agent {agent} — Actions",
        yaxis_title="Count",
        bargap=0.05,
    )

    return _styled(fig, f"{title} — Agent {agent}")


def action_entropy_fig(
    r: RunData, col: str, title: str, smooth: int, show_raw: bool
) -> go.Figure:
    fig = go.Figure()
    names = _legend_names(r)
    agent_num = []

    if "environment_steps" not in r.df:
        assert "environment_steps not found on CSV header"

    agent_num = sorted(
        {int(c.split("_")[-1]) for c in r.df.columns if c.startswith(col)}
    )

    cname = f"{col}"
    y_raw = r.df[cname]
    y = _smooth_series(y_raw, smooth)

    if show_raw and smooth > 1:
        fig.add_trace(
            go.Scatter(
                x=r.df["environment_steps"],
                y=y_raw,
                mode="lines",
                showlegend=False,
                line=dict(color=r.color, width=1),
                opacity=0.18,
                hoverinfo="skip",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=r.df["environment_steps"],
            y=y,
            mode="lines",
            name=f"{names[r.name]}",
            line=dict(color=r.color, width=2.2),
        )
    )

    fig.update_layout(xaxis_title="environment steps")
    return _styled(fig, title + " " + str(agent_num[0]))


def _scalar_fig(r: RunData, col: str, title: str) -> go.Figure:
    fig = go.Figure()
    names = _legend_names(r)
    vals = sanitize_masked(r.df[col])
    value = pd.Series(vals).mean()
    if "step_time_mean" in col:
        fig.add_annotation(
            x=0.5,
            y=0.5,
            text=f"{value:.3f} \u00b5Sec",
            showarrow=False,
            font=dict(color=r.color, size=32),
            xanchor="center",
            yanchor="middle",
            name=f"{names[r.name]}",
        )
    elif "episode_time" in col:
        fig.add_annotation(
            x=0.5,
            y=0.5,
            text=f"{value:.3f} Sec",
            showarrow=False,
            font=dict(color=r.color, size=32),
            xanchor="center",
            yanchor="middle",
            name=f"{names[r.name]}",
        )
    else:
        fig.add_annotation(
            x=0.5,
            y=0.5,
            text=f"{value:.0f}",
            showarrow=False,
            font=dict(color=r.color, size=32),
            xanchor="center",
            yanchor="middle",
            name=f"{names[r.name]}",
        )

    # remove axes
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)

    return _styled(fig, title)


def auto_plot(run: RunData, title: str, col: str, smooth: int, show_raw: bool):

    if col.startswith("action_jsd"):
        fig = jsd_heatmap_fig(run, col, "Action JSD", smooth, show_raw)
    elif col.startswith("fairness_rewards_lorenz_x_agent"):
        fig = lorenz_curve_fig(run, col, "Fairness rewards", smooth, show_raw)
    elif col.startswith("fairness_deliveries_lorenz_x_agent"):
        fig = lorenz_curve_fig(run, col, "Fairness deliveries", smooth, show_raw)
    elif col.startswith("time_to_completion_cdf_grid"):
        fig = cdf_fig(run, col, "Time to completion CDF", smooth, show_raw)
    elif col.startswith("time_to_first_delivery_cdf_grid"):
        fig = cdf_fig(run, col, "Time to first delivery CDF", smooth, show_raw)
    elif col.startswith("action_histogram_flat"):
        fig = action_histogram_fig(run, col, "Action Histogram", smooth, show_raw)
    elif col.startswith("action_entropy_agent"):
        fig = action_entropy_fig(run, col, "Action Entropy Agent", smooth, show_raw)
    else:
        ignore_list = [
            "UTD",
            "step_time_mean",
            "FPS_mean",
            "step_count_mean",
            "episode_time",
        ]
        if col in ignore_list:
            fig = _scalar_fig(run, col, title)
        else:
            fig = _curve_fig(run, col, title, smooth, show_raw)
    return fig


def _curve_fig(
    runs: list[RunData] | RunData,
    col: str,
    title: str,
    smooth: int,
    show_raw: bool,
) -> go.Figure:
    fig = go.Figure()
    names = _legend_names(runs)
    if isinstance(runs, list):
        for r in runs:
            if (
                col not in r.df.columns
                or "environment_steps" not in r.df.columns
                or not len(r.df)
            ):
                continue
            stride = max(1, math.ceil(len(r.df) / MAX_POINTS))
            x = r.df["environment_steps"][::stride]
            y_raw = r.df[col][::stride]
            list_clean = [
                "time_to_completion",
                "time_to_first_delivery",
                "gini",
                "rware",
            ]
            if any(key in col for key in list_clean):
                y_raw = sanitize_masked(r.df[col][::stride])
                y_raw = pd.Series(y_raw)

            y = _smooth_series(y_raw, smooth)
            if show_raw and smooth > 1:
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=y_raw,
                        mode="lines",
                        showlegend=False,
                        line=dict(color=r.color, width=1),
                        opacity=0.18,
                        hoverinfo="skip",
                    )
                )
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    name=names[r.name],
                    line=dict(color=r.color, width=2.2),
                    hovertemplate=f"{r.name}<br>%{{x:,}} steps<br>%{{y:.3f}}<extra></extra>",
                )
            )
    else:
        r = runs
        if (
            col not in r.df.columns
            or "environment_steps" not in r.df.columns
            or not len(r.df)
        ):
            assert f"col:{col} or environment_steps found on CSV header CSV file empty run:{r.name}"
        stride = max(1, math.ceil(len(r.df) / MAX_POINTS))
        x = r.df["environment_steps"][::stride]
        y_raw = r.df[col][::stride]
        list_clean = ["time_to_completion", "time_to_first_delivery", "gini", "rware"]
        if any(key in col for key in list_clean):
            y_raw = sanitize_masked(r.df[col][::stride])
            y_raw = pd.Series(y_raw)

        y = _smooth_series(y_raw, smooth)
        if show_raw and smooth > 1:
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y_raw,
                    mode="lines",
                    showlegend=False,
                    line=dict(color=r.color, width=1),
                    opacity=0.18,
                    hoverinfo="skip",
                )
            )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name=names[r.name],
                line=dict(color=r.color, width=2.2),
                hovertemplate=f"{r.name}<br>%{{x:,}} steps<br>%{{y:.3f}}<extra></extra>",
            )
        )
    fig.update_layout(xaxis_title="environment steps")
    return _styled(fig, title)


def _media(run: RunData) -> None:
    if not run.media:
        st.caption(
            "no rendered rollout yet — generate one with " "`scripts/render_rollout.py`"
        )
        return
    path = run.media[-1]
    if path.lower().endswith(".mp4"):
        st.video(path)
    else:
        st.image(path, caption=os.path.basename(path))


# ---------------------------------------------------------------------------
# sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("🤖 MARL Arena")
root = st.sidebar.text_input("runs directory", value="runs")
all_runs = _load_all(root)

if not all_runs:
    st.warning(
        f"No runs found under `{os.path.abspath(root)}` "
        "(a run dir needs a results.csv)."
    )
    st.stop()

by_name = {r.name: r for r in all_runs}
# default selection: one of each algorithm family, longest run first per family
default, seen_fams = [], set()
for r in sorted(all_runs, key=lambda r: -r.updates):
    if r.family not in seen_fams and r.family != "other":
        default.append(r.name)
        seen_fams.add(r.family)
selected = st.sidebar.multiselect(
    "runs to show", options=list(by_name), default=default or list(by_name)[:5]
)
runs = [by_name[n] for n in selected]

smooth = st.sidebar.slider(
    "smoothing (EMA span, updates)",
    min_value=1,
    max_value=500,
    value=100,
    help="curves are EMA-smoothed; the raw signal stays as a faint trace",
)
show_raw = st.sidebar.checkbox("show raw traces", value=True)
import csv
def check_full_metrics(runs: list[RunData]):
    runs_full_met:list[RunData]=[]
    for r in runs:
        results_path = os.path.join(r.path, "results.csv")

        # 1. Check if results.csv exists
        if not os.path.isfile(results_path):
            st.warning(f"results.csv not found in: {r.path}")
            st.stop()   # Immediately stop Streamlit execution

        # 2. Load CSV header and check if it has more than 300 columns
        try:
            with open(results_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader)
                if len(header) > 300:
                    runs_full_met.append(r)
        except Exception as e:
            st.warning(f"Failed to read results.csv in: {r.path}\nError: {e}")
            st.stop()
    return runs_full_met

runs_full_met=check_full_metrics(all_runs)

figs: dict[RunData, dict[str, list[go.Figure]]] = {} 
new_full_metrics: dict[str, dict[str, str]]={}
unique_numbers = iter(random.sample(range(9999), 1000))
if len(runs_full_met):
    new_full_metrics = process_metrics(
    FULL_METRIC_GROUPS, os.path.join(runs_full_met[0].path, "results.csv")
)
    metrics_count = 0
    for _, metrics in new_full_metrics.items():
        metrics_count += len(metrics.items())
    figs_count = len(runs_full_met) * len(new_full_metrics.items()) * metrics_count * 2
    unique_numbers = iter(random.sample(range(figs_count), figs_count))

     
    for r in runs_full_met:
        figs[r] = {}
        for group, metrics in new_full_metrics.items():
            figs_list: list[go.Figure] = []
            # figs[r][group]=[]
            for i, (col, label) in enumerate(metrics.items()):
                figs_list.append(auto_plot(r, label, col, smooth=smooth, show_raw=show_raw))
            figs[r][group] = figs_list# (~30MB for one run)

st.sidebar.caption(f"{len(all_runs)} runs discovered")

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
st.title("Multi-Agent RL Arena")
st.caption(
    "JAX RWARE · IA2C · IPPO · MAA2C · MAPPO · SEAC — "
    "faithful env, fully-jitted training, live telemetry"
)

if not runs:
    st.info("Select one or more runs in the sidebar.")
    st.stop()

(
    tab_overview,
    tab_compare,
    tab_full_compare,
    tab_theater,
    tab_story,
    tab_research,
    tab_detail,
) = st.tabs(
    [
        "🏆 Overview",
        "📊 Compare",
        "📊 Full Compare",
        "🎬 Replay theater",
        "🧠 Story",
        "🔬 Research",
        "🔎 Run detail",
    ]
)

# ---------------------------------------------------------------- overview --
with tab_overview:
    lb = leaderboard(runs)

    # headline KPIs across the selection
    k1, k2, k3, k4 = st.columns(4)
    total_steps = int(lb["env_steps"].sum())
    k1.metric("env steps trained", f"{total_steps / 1e6:,.0f} M")
    best_i = lb["final_return"].idxmax()
    k2.metric(
        "best final return",
        f"{lb.loc[best_i, 'final_return']:.1f}",
        lb.loc[best_i, "algo"],  # pyright: ignore[reportArgumentType]
    )
    k3.metric("best deliveries/ep", f"{lb['deliveries'].max():.1f}")
    eff = lb.dropna(subset=["steps_to_half_best"])
    if len(eff):
        fast_i = eff["steps_to_half_best"].idxmin()
        k4.metric(
            "fastest to half-best",
            f"{eff.loc[fast_i, 'steps_to_half_best'] / 1e6:.1f} M steps",  # pyright: ignore[reportOperatorIssue]
            eff.loc[fast_i, "algo"],  # pyright: ignore[reportArgumentType]
        )

    st.divider()

    # medals
    medals = []
    medals.append(
        (
            "🏆 Top performer",
            lb.loc[best_i, "algo"],
            f"final return {lb.loc[best_i, 'final_return']:.1f}",
        )
    )
    if len(eff):
        medals.append(
            (
                "🚀 Fastest learner",
                eff.loc[
                    fast_i, "algo"  # pyright: ignore[reportPossiblyUnboundVariable]
                ],
                f"half-best at {eff.loc[fast_i, 'steps_to_half_best'] / 1e6:.1f}M steps",  # pyright: ignore[reportOperatorIssue,reportPossiblyUnboundVariable]
            )
        )
    calm_i = lb["contention"].idxmin()
    medals.append(
        (
            "🕊️ Calmest traffic",
            lb.loc[calm_i, "algo"],
            f"{lb.loc[calm_i, 'contention'] * 100:.1f}% forward-blocked",  # pyright: ignore[reportOperatorIssue]
        )
    )
    busy_i = lb["idle"].idxmin()
    medals.append(
        (
            "⚙️ Least idle",
            lb.loc[busy_i, "algo"],
            f"{lb.loc[busy_i, 'idle'] * 100:.1f}% idle steps",  # pyright: ignore[reportOperatorIssue]
        )
    )
    mcols = st.columns(len(medals))
    for c, (title, algo, why) in zip(mcols, medals):
        c.markdown(f"**{title}**")
        c.markdown(
            f"<span style='color:{ACCENT};font-size:1.25em'>{algo}</span>",
            unsafe_allow_html=True,
        )
        c.caption(why)

    st.divider()

    # final-return bars + the ranked table
    cbar, ctab = st.columns([1, 1])
    with cbar:
        fig = go.Figure(
            go.Bar(
                x=lb["final_return"],
                y=lb["run"],
                orientation="h",
                marker_color=[by_name[n].color for n in lb["run"]],
                text=[f"{v:.1f}" for v in lb["final_return"]],
                textposition="outside",
            )
        )
        fig.update_layout(xaxis_title="final return (tail mean)")
        st.plotly_chart(
            _styled(fig, "Final team return", height=300),
            width="stretch",
        )
    with ctab:
        show = lb[
            [
                "algo",
                "env",
                "env_steps",
                "final_return",
                "best_return",
                "deliveries",
                "contention",
            ]
        ].copy()
        show["env_steps"] = (show["env_steps"] / 1e6).round(1)
        show["contention"] = (show["contention"] * 100).round(1)
        show = show.rename(
            columns={
                "env_steps": "M steps",
                "final_return": "final ret",
                "best_return": "best ret",
                "deliveries": "deliv/ep",
                "contention": "blocked %",
            }
        )
        st.dataframe(
            show.sort_values("final ret", ascending=False),
            width="stretch",
            hide_index=True,
        )

    st.plotly_chart(
        _curve_fig(
            runs, "episode_returns_mean", "The race — team return", smooth, show_raw
        ),
        width="stretch",
    )

# ----------------------------------------------------------------- compare --
with tab_compare:
    cols_per_row = 2

    for group, metrics in METRIC_GROUPS.items():
        st.subheader(group)
        gcols = st.columns(min(cols_per_row, len(metrics)))
        for i, (col, label) in enumerate(metrics.items()):
            with gcols[i % len(gcols)]:
                st.plotly_chart(
                    _curve_fig(runs, col, label, smooth, show_raw),
                    width="stretch",
                )


# ----------------------------------------------------------------- full compare --
def gen_plot_chart(group_figs_list, cols_per_row):
    gcols = st.columns(min(cols_per_row, len(group_figs_list)))
    for i, fig in enumerate(group_figs_list):
        with gcols[i % len(gcols)]:
            key = next(unique_numbers)
            st.plotly_chart(fig, width="stretch", key=key)


with tab_full_compare:
    if len(runs_full_met):
        pick = st.radio(
            "Runs with full metrics",
            options=[r.name for r in runs_full_met],
        )
        r = by_name[pick]

        st.markdown(
            f"""
            <style>
                div[data-testid="stRadio"] > details > summary p {{
                    color: {r.color};
                    font-weight: bold;
                    font-size: 18px;
                }}
            </style>
            """,
            unsafe_allow_html=True,
        )
        cols_per_row = 2

        for group, metrics in new_full_metrics.items():
            expander = st.expander(group, expanded=False, on_change="rerun")
            # style the expander header
            expander.markdown(
                f"""
                    <style>
                        div[data-testid="stExpander"] > details > summary p {{
                            font-size: 22px;
                            font-weight: bold;
                            color: {r.color};
                        }}
                    </style>
                    """,
                unsafe_allow_html=True,
            )
            if expander.open:
                gen_plot_chart(figs[r][group], cols_per_row)
    else:
        st.info("No full metrics results found")
# ----------------------------------------------------------------- theater --
with tab_theater:
    # runs (selected or not) that actually have recorded replays
    runs_with_replays = {r.name: list_replays(r.path) for r in all_runs}
    runs_with_replays = {k: v for k, v in runs_with_replays.items() if v}
    if not runs_with_replays:
        st.info(
            "No replays recorded yet. Train with `--replay-every` or run "
            "`python -m scripts.record_replay --run-dir runs/<run>` "
            "on a finished run."
        )
    elif st.toggle(
        "🆚 duel mode — two algorithms, same episode seed", value=False, key="duel"
    ):

        @st.cache_data(show_spinner="loading replays…")
        def _payload1(path: str) -> dict:
            return load_payload(path)

        names = list(runs_with_replays)
        ca, cb = st.columns(2)
        pick_a = ca.selectbox("contender A", options=names, index=0, key="duel_a")
        pick_b = cb.selectbox(
            "contender B", options=names, index=min(1, len(names) - 1), key="duel_b"
        )
        st.caption(
            "each side shows that run's latest replay snapshot for the "
            "same episode seed — identical warehouse, identical "
            "requests, different brains"
        )
        for col, pick in ((ca, pick_a), (cb, pick_b)):
            infos = [
                i for i in runs_with_replays[pick] if i.seed == 0 and not i.greedy
            ] or runs_with_replays[pick]
            snap = _payload1(infos[-1].path)
            with col:
                st.iframe(player_html([snap]), height=player_height([snap]) + 20)
    else:
        csel, cvar, cinfo = st.columns([2, 1, 2])
        pick = csel.selectbox("run", options=list(runs_with_replays), key="theater_run")
        infos = runs_with_replays[pick]
        # one player feed = one (seed, greedy) variant across training updates
        variants = sorted({(i.seed, i.greedy) for i in infos})
        vlabel = {v: f"seed {v[0]}" + (" · greedy" if v[1] else "") for v in variants}
        var = cvar.selectbox(
            "episode variant",
            options=variants,
            format_func=lambda v: vlabel[v],
            key="theater_var",
        )
        chosen = [i for i in infos if (i.seed, i.greedy) == var]

        # keep the embedded payload sane: at most ~24 snapshots, evenly spaced,
        # always including the first and the last
        MAX_SNAPS = 24
        if len(chosen) > MAX_SNAPS:
            idx = {
                round(k * (len(chosen) - 1) / (MAX_SNAPS - 1)) for k in range(MAX_SNAPS)
            }
            chosen = [chosen[k] for k in sorted(idx)]

        @st.cache_data(show_spinner="loading replays…")
        def _payloads(paths: tuple[str, ...]) -> list[dict]:
            return [load_payload(p) for p in paths]

        snaps = _payloads(tuple(i.path for i in chosen))
        cinfo.caption(
            f"{len(infos)} replays on disk · showing {len(snaps)} snapshots "
            f"from update {chosen[0].update:,} to {chosen[-1].update:,} · "
            "fixed seed ⇒ every visible difference is learning"
        )
        st.iframe(player_html(snaps), height=player_height(snaps) + 20)

# ------------------------------------------------------------------- story --
with tab_story:
    from arena.narrator_llm import AUDIENCES, backend_status, cached_narration, narrate

    st.caption(
        "An LLM (or a deterministic fallback) narrates the selected "
        "runs — every number it may use is measured from results.csv; "
        "nothing else is shown to the model."
    )
    c1, c2, c3 = st.columns([1.2, 1, 1.4])
    audience = c1.radio("audience", options=list(AUDIENCES), horizontal=True, index=0)
    backend = c2.selectbox("backend", options=["auto", "claude", "ollama", "template"])
    with c3:
        status = backend_status()
        st.caption("backends: " + " · ".join(f"**{b}** {s}" for b, s in status.items()))

    cached = cached_narration(runs, audience)
    go_btn = st.button(
        "📜 Re-narrate (try LLM)",
        type="primary",
        help="generate fresh with the selected backend; LLM "
        "backends are used when reachable, else template",
    )
    force = st.checkbox("regenerate (ignore cache)", value=False)

    if go_btn:
        with st.spinner("the commentator is watching the replays…"):
            text, used = narrate(runs, audience, backend, use_cache=not force)
        st.markdown(text)
        st.caption(f"backend: {used}")
    elif cached is not None:
        text, used = cached
        st.markdown(text)
        st.caption(f"backend: {used}")
    else:
        # never leave the panel empty (and never hang on a slow LLM at load):
        # show the instant deterministic narration; the button upgrades it.
        text, used = narrate(runs, audience, backend="template")
        st.markdown(text)
        st.caption(f"backend: {used} — press **Re-narrate** for an LLM version")

# ---------------------------------------------------------------- research --
with tab_research:
    import json as _json

    st.subheader("Environment throughput")
    bench_path = os.path.join("docs", "data", "speed_bench.json")
    if os.path.isfile(bench_path):
        with open(bench_path) as f:
            bench = pd.DataFrame(_json.load(f))
        bench["label"] = bench["impl"] + " · " + bench["device"]
        fig = go.Figure()
        palette = {
            "jaxrware (ours)": ACCENT,
            "jumanji": "#f58518",
            "rware (original)": "#8c8c8c",
        }
        for label, gdf in bench.groupby("label"):
            gdf = gdf.sort_values("batch")
            impl = gdf["impl"].iloc[0]
            fig.add_trace(
                go.Scatter(
                    x=gdf["batch"],
                    y=gdf["steps_per_sec"],
                    mode="lines+markers",
                    name=label,
                    line=dict(
                        color=palette.get(impl, "#c9d6e8"),
                        dash="dot" if gdf["device"].iloc[0] == "cpu" else "solid",
                    ),
                    hovertemplate="batch %{x}<br>%{y:,.0f} steps/s<extra></extra>",
                )
            )
        fig.update_layout(
            xaxis_type="log",
            yaxis_type="log",
            xaxis_title="parallel environments (batch)",
            yaxis_title="env steps / second",
        )
        st.plotly_chart(
            _styled(
                fig, "Random-action stepping speed, tiny-4ag " "(log-log)", height=420
            ),
            width="stretch",
        )
        best = bench.loc[bench["steps_per_sec"].idxmax()]
        base = bench[bench["impl"] == "rware (original)"]
        if len(base):
            ratio = best["steps_per_sec"] / base["steps_per_sec"].iloc[0]
            st.caption(
                f"peak: **{best['steps_per_sec']:,.0f} steps/s** "
                f"({best['label']}, batch {best['batch']}) — "
                f"**{ratio:,.0f}×** the original single-process env."
            )
    else:
        st.info("run `scripts/bench_speed.py` to populate the speed benchmark")

    st.divider()
    st.subheader("Why the only other JAX 'RWARE' is a different benchmark")
    repro_path = os.path.join("docs", "data", "divergence_repro.json")
    if os.path.isfile(repro_path):
        with open(repro_path) as f:
            rep = _json.load(f)
        jj, rw = rep.get("jumanji", {}), rep.get("rware", {})
        jr, rr = jj.get("random_policy", {}), rw.get("random_policy", {})
        c1, c2, c3 = st.columns(3)
        c1.metric(
            "episodes ending early (random policy)",
            f"{jr.get('terminated_early_frac', 0) * 100:.1f}%",
            f"rware: {rr.get('terminated_early_frac', 0) * 100:.0f}%",
            delta_color="inverse",
        )
        c2.metric(
            "median episode length",
            f"{jr.get('median_length', 0):.0f} / 500",
            f"rware: {rr.get('median_length', 0):.0f} / 500",
            delta_color="inverse",
        )
        c3.metric(
            "legal convoy move terminates?",
            "id-dependent",
            "rware: never",
            delta_color="inverse",
        )
        rows = [
            (
                "Legal convoy move (follower id 0) ends episode",
                jj.get("convoy_follower_id0_terminates"),
                rw.get("convoy_follower_id0_terminates"),
            ),
            (
                "Same move, agent ids swapped, ends episode",
                jj.get("convoy_follower_id1_terminates"),
                rw.get("convoy_follower_id1_terminates"),
            ),
            (
                "Head-on contention ends episode",
                jj.get("contention_terminates"),
                rw.get("contention_terminates"),
            ),
            (
                "…and leaves two agents on the same cell",
                jj.get("contention_same_cell"),
                False,
            ),
        ]
        st.dataframe(
            pd.DataFrame(
                [
                    (q, "💥 yes" if a else "✓ no", "💥 yes" if b else "✓ no")
                    for q, a, b in rows
                ],
                columns=["scenario", "Jumanji 1.1.1", "original rware"],
            ),
            width="stretch",
            hide_index=True,
        )
        st.caption(
            "Full investigation with code citations: "
            "`docs/jumanji_mava_divergence.md` · reproduce with "
            "`scripts/repro_jumanji_divergence.py`"
        )
    else:
        st.info(
            "run `scripts/repro_jumanji_divergence.py` to populate "
            "the divergence findings"
        )

# ------------------------------------------------------------------ detail --
with tab_detail:
    pick = st.selectbox("run", options=[r.name for r in runs])
    r = by_name[pick]
    st.subheader("Summary")
    sc1, sc2 = st.columns([1, 1])
    with sc1:
        st.json(summarize(r))
    with sc2:
        with st.expander("config.json", expanded=False):
            st.json(r.config)

    st.subheader("Rollout")
    detail_replays = list_replays(r.path)
    if detail_replays:
        # prefer the latest seed-0 episode; animate it with the canvas player
        latest = [
            i for i in detail_replays if i.seed == 0 and not i.greedy
        ] or detail_replays
        snap = load_payload(latest[-1].path)
        st.iframe(player_html([snap]), height=player_height([snap]) + 20)
        st.caption(
            f"{len(detail_replays)} replay snapshots · "
            f"{len(r.checkpoint_steps)} checkpoints on disk"
        )
    else:
        _media(r)
        st.caption(
            f"no replays yet · {len(r.checkpoint_steps)} checkpoints "
            "— record with `python -m scripts.record_replay "
            f"--run-dir runs/{r.name}`"
        )
    st.subheader("All metrics")
    gcols = st.columns(2)
    i = 0
    for group, metrics in METRIC_GROUPS.items():
        for col, label in metrics.items():
            with gcols[i % 2]:
                key = next(unique_numbers)
                st.plotly_chart(
                    _curve_fig([r], col, label, smooth, show_raw),
                    width="stretch",
                    key=key,
                )
            i += 1
