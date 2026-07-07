"""Smoke-check arena.run_data against the runs/ directory (no streamlit needed).

    python -m arena.smoke_run_data
"""

from __future__ import annotations

from arena.run_data import load_runs,FULL_METRIC_GROUPS
from arena.run_data_handler import process_metrics

def test_run_data_handler() -> None:
    runs = load_runs("runs")
    print(f"discovered {len(runs)} runs under runs/\n")
    for r in runs:
        print(f"  {r.name:34s} {r.label:6s} {r.env_tag:20s} "
              f"updates={r.updates:<6d} ckpts={len(r.checkpoint_steps):<3d} "
              f"media={len(r.media)}")
        grouped_metrics = process_metrics(FULL_METRIC_GROUPS, r.path)
        for group, metrics in grouped_metrics.items():
            print(group)
            for csv_m_name,display_m_name in metrics.items():
                print(csv_m_name,display_m_name)
if __name__ == "__main__":
    test_run_data_handler()
