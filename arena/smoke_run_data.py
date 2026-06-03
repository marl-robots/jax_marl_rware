"""Smoke-check arena.run_data against the runs/ directory (no streamlit needed).

    python -m arena.smoke_run_data
"""

from __future__ import annotations

import json

from arena.run_data import load_runs, summarize


def main() -> None:
    runs = load_runs("runs")
    print(f"discovered {len(runs)} runs under runs/\n")
    for r in runs:
        print(f"  {r.name:34s} {r.label:6s} {r.env_tag:20s} "
              f"updates={r.updates:<6d} ckpts={len(r.checkpoint_steps):<3d} "
              f"media={len(r.media)}")

    print("\n--- sample RunSummary (first contestant with metrics) ---")
    for r in runs:
        if r.family in {"ia2c", "ippo", "maa2c", "mappo", "seac"} and r.updates > 0:
            print(json.dumps(summarize(r), indent=2))
            break


if __name__ == "__main__":
    main()
