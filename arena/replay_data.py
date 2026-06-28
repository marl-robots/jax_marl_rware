"""Read replay snapshots (.npz written by algorithms/replay.py) for the dashboard.

Dependency-light on purpose (numpy + stdlib — no jax), mirroring
arena/run_data.py: the Streamlit app must start instantly. The main product is
:func:`replay_payload`, a JSON-able dict the in-browser canvas player consumes
(frames of agent/shelf state + an event stream + cumulative series).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

import numpy as np

REPLAY_DIR = "replays"
_FNAME_RE = re.compile(
    r"replay_upd(?P<update>\d+)(?:_s(?P<seed>\d+))?(?P<greedy>_g)?\.npz$")


@dataclass
class ReplayInfo:
    """One discovered replay file (cheap: parsed from the filename only)."""
    path: str
    update: int
    seed: int
    greedy: bool


def list_replays(run_dir: str) -> list[ReplayInfo]:
    """Discover replays under <run_dir>/replays/, sorted by (update, seed)."""
    d = os.path.join(run_dir, REPLAY_DIR)
    if not os.path.isdir(d):
        return []
    out = []
    for name in os.listdir(d):
        m = _FNAME_RE.match(name)
        if m:
            out.append(ReplayInfo(
                path=os.path.join(d, name),
                update=int(m.group("update")),
                seed=int(m.group("seed") or 0),
                greedy=bool(m.group("greedy")),
            ))
    return sorted(out, key=lambda r: (r.update, r.seed, r.greedy))


def load_replay(path: str) -> dict:
    """Load one .npz replay into {meta: dict, <arrays>} (numpy arrays)."""
    with np.load(path) as z:
        data = {k: z[k] for k in z.files if k != "meta_json"}
        data["meta"] = json.loads(str(z["meta_json"]))
    return data


def replay_payload(replay: dict) -> dict:
    """JSON-able payload for the canvas player.

    Frames are time-major lists; state arrays have T+1 frames (0 = post-reset),
    event/series arrays have T entries aligned so events[t] happened during the
    transition from frame t to frame t+1.
    """
    meta = replay["meta"]
    T = int(replay["actions"].shape[0])

    # events: (t, agent) pairs for the moments the player highlights
    def _events(key: str) -> list[list[int]]:
        arr = replay[key]
        ts, agents = np.nonzero(arr)
        return [[int(t), int(a)] for t, a in zip(ts, agents)]

    rewards_team = replay["rewards"].sum(axis=1)            # [T]
    deliv_team = replay["deliveries"].sum(axis=1)           # [T]

    return {
        "meta": meta,
        "T": T,
        "agents": {
            "x": replay["agent_x"].tolist(),                # [T+1][N]
            "y": replay["agent_y"].tolist(),
            "dir": replay["agent_dir"].tolist(),
            "carrying": (replay["agent_carrying"] > 0).tolist(),
        },
        "shelves": {
            "x": replay["shelf_x"].tolist(),                # [T+1][S]
            "y": replay["shelf_y"].tolist(),
            "requested": replay["in_queue"].tolist(),
        },
        # rack zone = cells where shelves start (post-reset = home positions)
        "rack_cells": sorted({(int(x), int(y)) for x, y in
                              zip(replay["shelf_x"][0], replay["shelf_y"][0])}),
        "events": {
            "deliveries": _events("deliveries"),
            "pickups": _events("pickup"),
            "drops": _events("drop"),
            "blocked": _events("blocked"),
        },
        "series": {
            "reward_cum": np.cumsum(rewards_team).round(3).tolist(),
            "deliveries_cum": np.cumsum(deliv_team).astype(int).tolist(),
        },
    }


def load_payload(path: str) -> dict:
    return replay_payload(load_replay(path))
