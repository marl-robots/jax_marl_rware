"""Deterministic, host-side training narration (Layer 2).

Turns the per-update behavioral metrics (deliveries, forward-block/contention
rate, idle rate, and the early/mid/late episode-phase split) into short English
sentences, emitted once per chunk by the training loop. Runs entirely on the
main thread between fused-scan chunks, so it never touches the rollout and has
zero effect on training speed.

Every statement is tied to a number we actually measured; we report deltas
versus a slow EMA baseline rather than inferring intent ("learned to cooperate"
is an interpretation, not a measurement, so we don't say it). The baseline
persists across chunks so trend arrows mean something over the whole run.
"""

from __future__ import annotations


def _arrow(cur: float, base: float, eps: float = 1e-6) -> str:
    """Trend glyph for cur vs baseline (▲ up / ▼ down / ~ flat)."""
    if base is None or abs(cur - base) <= eps:
        return "~"
    return "▲" if cur > base else "▼"


def _phase_pattern(early: float, mid: float, late: float) -> str:
    """Describe how deliveries distribute across the episode's thirds."""
    total = early + mid + late
    if total < 0.5:
        return "barely any deliveries yet across the episode"
    vals = {"early": early, "mid": mid, "late": late}
    top = max(vals, key=vals.get)
    spread = (max(vals.values()) - min(vals.values())) / (total / 3.0 + 1e-6)
    if spread < 0.5:
        return "throughput is steady across the episode"
    return f"throughput is weighted toward the {top} of the episode"


class Narrator:
    """Rolling-baseline narrator. Call `chunk(update, stats)` once per chunk;
    `stats` holds the chunk-mean of each behavioral metric. Returns a multi-line
    string (or "" when there is nothing notable to say yet)."""

    def __init__(self, decay: float = 0.9):
        self.decay = decay
        self.base: dict | None = None

    def _ema(self, key: str, x: float) -> float:
        prev = None if self.base is None else self.base.get(key)
        return x if prev is None else self.decay * prev + (1.0 - self.decay) * x

    def chunk(self, update: int, stats: dict) -> str:
        d = stats["deliveries"]
        b = stats["block_rate"]
        idle = stats["idle_rate"]
        ret = stats["episode_return"]
        e, m, l = (stats["deliveries_early"], stats["deliveries_mid"],
                   stats["deliveries_late"])

        base = self.base or {}
        d_arrow = _arrow(d, base.get("deliveries"))
        b_arrow = _arrow(b, base.get("block_rate"))
        ret_arrow = _arrow(ret, base.get("episode_return"))

        lines = [
            f"[commentary @ update {update}]",
            f"  deliveries/episode ~{d:.1f} {d_arrow}   "
            f"team return ~{ret:.2f} {ret_arrow}",
            f"  contention (forward-blocked) {b * 100:.1f}% {b_arrow}   "
            f"idle {idle * 100:.1f}%",
            f"  {_phase_pattern(e, m, l)} "
            f"(early {e:.1f} / mid {m:.1f} / late {l:.1f})",
        ]

        # update the slow baseline for next time
        new_base = dict(base)
        for k in ("deliveries", "block_rate", "idle_rate", "episode_return"):
            new_base[k] = self._ema(k, stats[k])
        self.base = new_base
        return "\n".join(lines)
