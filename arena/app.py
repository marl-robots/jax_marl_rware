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
import sys

# `streamlit run arena/app.py` puts arena/ (not the repo root) on sys.path, so
# `import arena.*` would fail. Add the repo root explicitly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from arena.player import player_html, player_height
from arena.replay_data import list_replays, load_payload
from arena.run_data import (
    METRIC_GROUPS,
    RunData,
    discover_runs,
    leaderboard,
    load_run,
    summarize,
)

st.set_page_config(page_title="MARL Arena", page_icon="🤖", layout="wide")

ACCENT = "#00e5cc"
GRID = "#1d2737"
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
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Consolas, monospace", color="#c9d6e8", size=11),
        xaxis=dict(gridcolor=GRID, zeroline=False),
        yaxis=dict(gridcolor=GRID, zeroline=False),
        legend=dict(orientation="h", bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(font_family="Consolas, monospace"),
    )
    return fig


def _legend_names(runs: list[RunData]) -> dict[str, str]:
    """Short algo label when unique among the selection, else the run name."""
    fams = [r.family for r in runs]
    return {r.name: (r.label if fams.count(r.family) == 1 else r.name)
            for r in runs}


def _curve_fig(runs: list[RunData], col: str, title: str,
               smooth: int, show_raw: bool) -> go.Figure:
    fig = go.Figure()
    names = _legend_names(runs)
    for r in runs:
        if col not in r.df.columns or "environment_steps" not in r.df.columns \
                or not len(r.df):
            continue
        stride = max(1, math.ceil(len(r.df) / MAX_POINTS))
        x = r.df["environment_steps"][::stride]
        y_raw = r.df[col][::stride]
        if show_raw and smooth > 1:
            fig.add_trace(go.Scatter(
                x=x, y=y_raw, mode="lines", showlegend=False,
                line=dict(color=r.color, width=1), opacity=0.18,
                hoverinfo="skip"))
        y = r.df[col].ewm(span=smooth, min_periods=1).mean()[::stride] \
            if smooth > 1 else y_raw
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="lines", name=names[r.name],
            line=dict(color=r.color, width=2.2),
            hovertemplate=f"{r.name}<br>%{{x:,}} steps<br>%{{y:.3f}}<extra></extra>",
        ))
    fig.update_layout(xaxis_title="environment steps")
    return _styled(fig, title)


def _media(run: RunData) -> None:
    if not run.media:
        st.caption("no rendered rollout yet — generate one with "
                   "`scripts/render_rollout.py`")
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
    st.warning(f"No runs found under `{os.path.abspath(root)}` "
               "(a run dir needs a results.csv).")
    st.stop()

by_name = {r.name: r for r in all_runs}
# default selection: one of each algorithm family, longest run first per family
default, seen_fams = [], set()
for r in sorted(all_runs, key=lambda r: -r.updates):
    if r.family not in seen_fams and r.family != "other":
        default.append(r.name)
        seen_fams.add(r.family)
selected = st.sidebar.multiselect(
    "runs to show", options=list(by_name), default=default or list(by_name)[:5])
runs = [by_name[n] for n in selected]

smooth = st.sidebar.slider(
    "smoothing (EMA span, updates)", min_value=1, max_value=500, value=100,
    help="curves are EMA-smoothed; the raw signal stays as a faint trace")
show_raw = st.sidebar.checkbox("show raw traces", value=True)
st.sidebar.caption(f"{len(all_runs)} runs discovered")

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
st.title("Multi-Agent RL Arena")
st.caption("JAX RWARE · IA2C · IPPO · MAA2C · MAPPO · SEAC — "
           "faithful env, fully-jitted training, live telemetry")

if not runs:
    st.info("Select one or more runs in the sidebar.")
    st.stop()

tab_overview, tab_compare, tab_theater, tab_detail = st.tabs(
    ["🏆 Overview", "📊 Compare", "🎬 Replay theater", "🔎 Run detail"])

# ---------------------------------------------------------------- overview --
with tab_overview:
    lb = leaderboard(runs)

    # headline KPIs across the selection
    k1, k2, k3, k4 = st.columns(4)
    total_steps = int(lb["env_steps"].sum())
    k1.metric("env steps trained", f"{total_steps / 1e6:,.0f} M")
    best_i = lb["final_return"].idxmax()
    k2.metric("best final return", f"{lb.loc[best_i, 'final_return']:.1f}",
              lb.loc[best_i, "algo"])
    k3.metric("best deliveries/ep", f"{lb['deliveries'].max():.1f}")
    eff = lb.dropna(subset=["steps_to_half_best"])
    if len(eff):
        fast_i = eff["steps_to_half_best"].idxmin()
        k4.metric("fastest to half-best",
                  f"{eff.loc[fast_i, 'steps_to_half_best'] / 1e6:.1f} M steps",
                  eff.loc[fast_i, "algo"])

    st.divider()

    # medals
    medals = []
    medals.append(("🏆 Top performer", lb.loc[best_i, "algo"],
                   f"final return {lb.loc[best_i, 'final_return']:.1f}"))
    if len(eff):
        medals.append(("🚀 Fastest learner", eff.loc[fast_i, "algo"],
                       f"half-best at {eff.loc[fast_i, 'steps_to_half_best'] / 1e6:.1f}M steps"))
    calm_i = lb["contention"].idxmin()
    medals.append(("🕊️ Calmest traffic", lb.loc[calm_i, "algo"],
                   f"{lb.loc[calm_i, 'contention'] * 100:.1f}% forward-blocked"))
    busy_i = lb["idle"].idxmin()
    medals.append(("⚙️ Least idle", lb.loc[busy_i, "algo"],
                   f"{lb.loc[busy_i, 'idle'] * 100:.1f}% idle steps"))
    mcols = st.columns(len(medals))
    for c, (title, algo, why) in zip(mcols, medals):
        c.markdown(f"**{title}**")
        c.markdown(f"<span style='color:{ACCENT};font-size:1.25em'>{algo}</span>",
                   unsafe_allow_html=True)
        c.caption(why)

    st.divider()

    # final-return bars + the ranked table
    cbar, ctab = st.columns([1, 1])
    with cbar:
        fig = go.Figure(go.Bar(
            x=lb["final_return"], y=lb["run"], orientation="h",
            marker_color=[by_name[n].color for n in lb["run"]],
            text=[f"{v:.1f}" for v in lb["final_return"]],
            textposition="outside",
        ))
        fig.update_layout(xaxis_title="final return (tail mean)")
        st.plotly_chart(_styled(fig, "Final team return", height=300),
                        use_container_width=True)
    with ctab:
        show = lb[["algo", "env", "env_steps", "final_return", "best_return",
                   "deliveries", "contention"]].copy()
        show["env_steps"] = (show["env_steps"] / 1e6).round(1)
        show["contention"] = (show["contention"] * 100).round(1)
        show = show.rename(columns={
            "env_steps": "M steps", "final_return": "final ret",
            "best_return": "best ret", "deliveries": "deliv/ep",
            "contention": "blocked %"})
        st.dataframe(show.sort_values("final ret", ascending=False),
                     use_container_width=True, hide_index=True)

    st.plotly_chart(
        _curve_fig(runs, "mean_episode_returns", "The race — team return",
                   smooth, show_raw), use_container_width=True)

# ----------------------------------------------------------------- compare --
with tab_compare:
    cols_per_row = 2
    for group, metrics in METRIC_GROUPS.items():
        st.subheader(group)
        gcols = st.columns(min(cols_per_row, len(metrics)))
        for i, (col, label) in enumerate(metrics.items()):
            with gcols[i % len(gcols)]:
                st.plotly_chart(_curve_fig(runs, col, label, smooth, show_raw),
                                use_container_width=True)

# ----------------------------------------------------------------- theater --
with tab_theater:
    # runs (selected or not) that actually have recorded replays
    runs_with_replays = {r.name: list_replays(r.path) for r in all_runs}
    runs_with_replays = {k: v for k, v in runs_with_replays.items() if v}
    if not runs_with_replays:
        st.info("No replays recorded yet. Train with `--replay-every` or run "
                "`python -m scripts.record_replay --run-dir runs/<run>` "
                "on a finished run.")
    else:
        csel, cvar, cinfo = st.columns([2, 1, 2])
        pick = csel.selectbox("run", options=list(runs_with_replays),
                              key="theater_run")
        infos = runs_with_replays[pick]
        # one player feed = one (seed, greedy) variant across training updates
        variants = sorted({(i.seed, i.greedy) for i in infos})
        vlabel = {v: f"seed {v[0]}" + (" · greedy" if v[1] else "")
                  for v in variants}
        var = cvar.selectbox("episode variant", options=variants,
                             format_func=lambda v: vlabel[v], key="theater_var")
        chosen = [i for i in infos if (i.seed, i.greedy) == var]

        # keep the embedded payload sane: at most ~24 snapshots, evenly spaced,
        # always including the first and the last
        MAX_SNAPS = 24
        if len(chosen) > MAX_SNAPS:
            idx = {round(k * (len(chosen) - 1) / (MAX_SNAPS - 1))
                   for k in range(MAX_SNAPS)}
            chosen = [chosen[k] for k in sorted(idx)]

        @st.cache_data(show_spinner="loading replays…")
        def _payloads(paths: tuple[str, ...]) -> list[dict]:
            return [load_payload(p) for p in paths]

        snaps = _payloads(tuple(i.path for i in chosen))
        cinfo.caption(
            f"{len(infos)} replays on disk · showing {len(snaps)} snapshots "
            f"from update {chosen[0].update:,} to {chosen[-1].update:,} · "
            "fixed seed ⇒ every visible difference is learning")
        components.html(player_html(snaps), height=player_height(snaps) + 20)

# ------------------------------------------------------------------ detail --
with tab_detail:
    pick = st.selectbox("run", options=[r.name for r in runs])
    r = by_name[pick]
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Rollout")
        _media(r)
        n_replays = len(list_replays(r.path))
        st.caption(f"{n_replays} replay snapshots · "
                   f"{len(r.checkpoint_steps)} checkpoints on disk")
    with right:
        st.subheader("Summary")
        st.json(summarize(r))
        with st.expander("config.json"):
            st.json(r.config)
    st.subheader("All metrics")
    gcols = st.columns(2)
    i = 0
    for group, metrics in METRIC_GROUPS.items():
        for col, label in metrics.items():
            with gcols[i % 2]:
                st.plotly_chart(_curve_fig([r], col, label, smooth, show_raw),
                                use_container_width=True)
            i += 1
