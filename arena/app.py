"""Arena dashboard (Streamlit) -- M1: run browser, metric graphs, renders, compare.

Run from the repo root:

    streamlit run arena/app.py

Reads run artifacts via arena.run_data (pandas + stdlib only -- no jax import),
so the dashboard starts instantly. Later milestones add the live training feed,
the LLM commentator panel, and the competition / judge-crew tab.
"""

from __future__ import annotations

import os
import sys

# `streamlit run arena/app.py` puts arena/ (not the repo root) on sys.path, so
# `import arena.*` would fail. Add the repo root explicitly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from arena.player import player_html, player_height
from arena.replay_data import list_replays, load_payload
from arena.run_data import (
    METRIC_GROUPS,
    RunData,
    load_run,
    discover_runs,
    summarize,
)

st.set_page_config(page_title="MARL Arena", page_icon="🤖", layout="wide")


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


def _line_chart(runs: list[RunData], col: str, title: str) -> go.Figure:
    fig = go.Figure()
    for r in runs:
        if col in r.df.columns and "environment_steps" in r.df.columns and len(r.df):
            fig.add_trace(go.Scatter(
                x=r.df["environment_steps"], y=r.df[col],
                mode="lines", name=r.label,
                line=dict(color=r.color, width=2),
                hovertemplate=f"{r.name}<br>%{{x:,}} steps<br>%{{y:.3f}}<extra></extra>",
            ))
    fig.update_layout(
        title=title, height=300,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_title="environment steps", legend=dict(orientation="h"),
        template="plotly_white",
    )
    return fig


def _media(run: RunData) -> None:
    if not run.media:
        st.caption("no rendered rollout yet — generate one with `scripts/render_rollout.py`")
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
# default selection: one of each algorithm family on the most common env
default = []
seen_fams = set()
for r in all_runs:
    if r.family not in seen_fams and r.family != "other":
        default.append(r.name)
        seen_fams.add(r.family)
selected = st.sidebar.multiselect(
    "runs to show", options=list(by_name), default=default or list(by_name)[:5])
runs = [by_name[n] for n in selected]

st.sidebar.caption(f"{len(all_runs)} runs discovered")

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
st.title("Multi-Agent RL Arena")
st.caption("JAX RWARE · IA2C · IPPO · MAA2C · MAPPO · SEAC")

if not runs:
    st.info("Select one or more runs in the sidebar.")
    st.stop()

tab_compare, tab_theater, tab_detail = st.tabs(
    ["📊 Compare", "🎬 Replay theater", "🔎 Run detail"])

with tab_compare:
    # headline scorecards
    cols = st.columns(len(runs))
    for c, r in zip(cols, runs):
        s = summarize(r)
        c.markdown(f"<span style='color:{r.color};font-size:1.3em'>●</span> "
                   f"**{r.label}**", unsafe_allow_html=True)
        c.caption(f"`{r.name}`")
        c.metric("final return", s.get("final_return", "—"))
        c.metric("deliveries/ep", s.get("final_deliveries", "—"))
        c.caption(f"{s.get('env','')} · {s.get('updates',0)} updates")

    st.divider()
    for group, metrics in METRIC_GROUPS.items():
        st.subheader(group)
        gcols = st.columns(min(2, len(metrics)))
        for i, (col, label) in enumerate(metrics.items()):
            with gcols[i % len(gcols)]:
                st.plotly_chart(_line_chart(runs, col, label),
                                use_container_width=True)

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

with tab_detail:
    pick = st.selectbox("run", options=[r.name for r in runs])
    r = by_name[pick]
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Rollout")
        _media(r)
    with right:
        st.subheader("Summary")
        st.json(summarize(r))
        with st.expander("config.json"):
            st.json(r.config)
    st.subheader("All metrics")
    for group, metrics in METRIC_GROUPS.items():
        for col, label in metrics.items():
            st.plotly_chart(_line_chart([r], col, label), use_container_width=True)
