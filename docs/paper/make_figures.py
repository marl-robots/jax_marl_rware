"""Generate publication figures + a results table for the final-project paper.

Reads the EMAX scaling-sweep CSVs under runs/paper/<tag>/results.csv (written by
scripts.train_dqn_fast) and the existing speed benchmark in
docs/data/speed_bench.json. Tolerant of in-progress / missing runs: it plots
whatever exists and prints a markdown results table to stdout so the numbers in
the paper are copied straight from measured data, never invented.

Run (in jax_env_1 or any env with pandas+matplotlib):
    python docs/paper/make_figures.py
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Prefer the live run dir; fall back to the CSVs archived in-repo so the figures
# regenerate from a fresh clone (runs/ is gitignored).
_RUNS = os.path.join(ROOT, "runs", "paper")
_ARCHIVE = os.path.join(ROOT, "docs", "paper", "data")
PAPER = _RUNS if os.path.exists(
    os.path.join(_RUNS, "emax_tiny_4ag_s1", "results.csv")) else _ARCHIVE
FIGS = os.path.join(ROOT, "docs", "paper", "figs")
os.makedirs(FIGS, exist_ok=True)

# Grid geometry per size (from jaxrware/config.py: (col+1)*rows+2  x  3*cols+1).
GRID = {"tiny": (11, 10), "small": (20, 10), "medium": (20, 16), "large": (29, 16)}
SIZES = ["tiny", "small", "medium", "large"]
COLORS = {"tiny": "#4cc9f0", "small": "#4895ef", "medium": "#f72585", "large": "#b5179e"}

plt.rcParams.update({
    "figure.dpi": 140, "font.size": 10, "axes.grid": True,
    "grid.alpha": 0.25, "axes.spines.top": False,
    "axes.spines.right": False, "savefig.bbox": "tight",
})


def load(tag):
    p = os.path.join(PAPER, tag, "results.csv")
    if not os.path.exists(p):
        return None
    try:
        df = pd.read_csv(p)
    except Exception:
        return None
    if "deliveries" not in df or len(df) < 2:
        return None
    return df


def ema(x, alpha=0.02):
    out = np.empty(len(x), float)
    acc = x[0]
    for i, v in enumerate(x):
        acc = (1 - alpha) * acc + alpha * v
        out[i] = acc
    return out


def tail_mean(df, frac=0.1):
    n = max(1, int(len(df) * frac))
    return float(df["deliveries"].tail(n).mean())


# ---- Figure 1: EMAX learning curves across map sizes (scaling) -------------
def seed_finals(sz):
    """Per-seed final (tail-10%) deliveries for a size; seeds s1, s2 if present."""
    vals = []
    for tag in (f"emax_{sz}_4ag_s1", f"emax_{sz}_4ag_s2"):
        df = load(tag)
        if df is not None:
            vals.append(tail_mean(df))
    return vals


def fig_scaling():
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    rows = []
    for sz in SIZES:
        df = load(f"emax_{sz}_4ag_s1")
        if df is None:
            continue
        x = df["environment_steps"].to_numpy() / 1e6
        y = df["deliveries"].to_numpy()
        ax.plot(x, y, color=COLORS[sz], alpha=0.18, lw=0.8)
        ax.plot(x, ema(y), color=COLORS[sz], lw=2.0,
                label=f"{sz} ({GRID[sz][0]}x{GRID[sz][1]})")
        finals = seed_finals(sz)
        mean_final = sum(finals) / len(finals)
        spread = (max(finals) - min(finals)) / 2 if len(finals) > 1 else 0.0
        rows.append((sz, x[-1], mean_final, spread, float(y.max()), len(finals)))
    ax.set_xlabel("environment steps (millions)")
    ax.set_ylabel("deliveries / episode")
    ax.set_title("IDQN-EMAX (K=5): learning across warehouse sizes")
    ax.legend(title="map (grid h x w)", frameon=False, fontsize=8)
    fig.savefig(os.path.join(FIGS, "fig1_scaling.png"))
    plt.close(fig)
    return rows


# ---- Figure 2: final deliveries by map size (bar) --------------------------
def fig_final_bar(rows):
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    labels = [f"{sz}\n{GRID[sz][0]}x{GRID[sz][1]}" for sz, *_ in rows]
    vals = [tm for _, _, tm, _, _, _ in rows]
    errs = [sp for _, _, _, sp, _, _ in rows]
    ax.bar(labels, vals, yerr=errs, capsize=4,
           color=[COLORS[sz] for sz, *_ in rows])
    for i, v in enumerate(vals):
        ax.text(i, v + errs[i], f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("deliveries / episode (final-10% mean)")
    ax.set_title("IDQN-EMAX scales to larger warehouses")
    ax.margins(y=0.15)
    fig.savefig(os.path.join(FIGS, "fig2_final_bar.png"))
    plt.close(fig)


# ---- Figure 3: ensemble ablation K=1 vs K=5 on medium ----------------------
def fig_ablation():
    d5 = load("emax_medium_4ag_s1")
    d1 = load("emax_medium_4ag_K1_s1")
    if d5 is None and d1 is None:
        return None
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    out = {}
    for df, lab, c in [(d5, "K=5 (EMAX ensemble)", "#f72585"),
                       (d1, "K=1 (no ensemble)", "#888888")]:
        if df is None:
            continue
        x = df["environment_steps"].to_numpy() / 1e6
        y = df["deliveries"].to_numpy()
        ax.plot(x, y, color=c, alpha=0.18, lw=0.8)
        ax.plot(x, ema(y), color=c, lw=2.0, label=lab)
        out[lab] = tail_mean(df)
    ax.set_xlabel("environment steps (millions)")
    ax.set_ylabel("deliveries / episode")
    ax.set_title("Ensemble ablation on medium-4ag")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(os.path.join(FIGS, "fig3_ablation.png"))
    plt.close(fig)
    return out


# ---- Figure 4: speed benchmark (existing data) -----------------------------
def fig_speed():
    p = os.path.join(ROOT, "docs", "data", "speed_bench.json")
    if not os.path.exists(p):
        return
    data = json.load(open(p))
    # peak steps/s per (impl, device)
    peak = {}
    for r in data:
        k = (r["impl"], r["device"])
        peak[k] = max(peak.get(k, 0), r["steps_per_sec"])
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    order = [("rware (original)", "cpu"), ("jumanji", "cpu"),
             ("jaxrware (ours)", "cpu"), ("jumanji", "gpu"),
             ("jaxrware (ours)", "gpu")]
    labels, vals = [], []
    for k in order:
        if k in peak:
            labels.append(f"{k[0].split()[0]}\n{k[1]}")
            vals.append(peak[k])
    bars = ax.bar(labels, vals, color="#4895ef")
    if labels:
        bars[-1].set_color("#f72585")
    ax.set_yscale("log")
    ax.set_ylabel("steps / s (log)")
    ax.set_title("Environment throughput (tiny-4ag, random policy)")
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v/1e6:.2f}M" if v >= 1e6 else f"{v/1e3:.0f}k",
                ha="center", va="bottom", fontsize=8)
    fig.savefig(os.path.join(FIGS, "fig4_speed.png"))
    plt.close(fig)


def main():
    rows = fig_scaling()
    fig_final_bar(rows)
    abl = fig_ablation()
    fig_speed()

    print("\n## EMAX scaling results (measured)\n")
    print("| map | grid | env steps (M) | seeds | final deliv (tail-10%) | peak deliv |")
    print("|---|---|---|---|---|---|")
    for sz, xm, tm, sp, pk, ns in rows:
        pm = f" +/- {sp:.2f}" if ns > 1 else ""
        print(f"| {sz} | {GRID[sz][0]}x{GRID[sz][1]} | {xm:.0f} | {ns} | "
              f"{tm:.2f}{pm} | {pk:.1f} |")
    if abl:
        print("\n## Ensemble ablation on medium-4ag (measured)\n")
        for k, v in abl.items():
            print(f"- {k}: final deliveries {v:.2f}")
    print("\nfigures ->", FIGS)


if __name__ == "__main__":
    main()
