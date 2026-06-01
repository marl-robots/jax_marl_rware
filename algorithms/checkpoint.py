"""Orbax-backed checkpointing for the MAPPO training carry.

Wraps ``orbax.checkpoint.CheckpointManager`` with:
  * keep-best  -- retains the step with the highest tracked metric (we feed an
                  EMA-smoothed episode return, since the per-update return is
                  very noisy), so a late-training collapse never loses the peak.
  * keep-last-N -- retains the N most recent steps for a rollback history.

The carry is a flat pytree (params, target_critic, opt_state, welford, key);
orbax's StandardSave/StandardRestore round-trip arbitrary array pytrees and, on
restore, rebuild the structure from a freshly-initialised `target` carry.

Saves are driven from the MAIN thread (the chunked training loop), which is how
orbax expects to be used -- never from inside an io_callback. The run config is
written once to ``<dir>/config.json`` so tools (e.g. rendering) can rebuild the
env/network without re-specifying flags.
"""

from __future__ import annotations

import dataclasses
import json
import os

import jax
import orbax.checkpoint as ocp

CONFIG_NAME = "config.json"


class CheckpointManager:
    def __init__(self, run_dir: str, max_to_keep: int = 5, best_mode: str = "max"):
        # orbax requires an absolute directory.
        self.run_dir = os.path.abspath(run_dir)
        os.makedirs(self.run_dir, exist_ok=True)
        self.ckpt_dir = os.path.join(self.run_dir, "checkpoints")
        options = ocp.CheckpointManagerOptions(
            max_to_keep=max_to_keep,
            best_fn=lambda metrics: float(metrics["smoothed_return"]),
            best_mode=best_mode,
            create=True,
        )
        self._mgr = ocp.CheckpointManager(self.ckpt_dir, options=options)

    # ---- config sidecar ----------------------------------------------------
    def save_config(self, cfg) -> None:
        cfg_dict = (dataclasses.asdict(cfg) if dataclasses.is_dataclass(cfg)
                    else dict(cfg))
        with open(os.path.join(self.run_dir, CONFIG_NAME), "w") as f:
            json.dump(cfg_dict, f, indent=2)

    @staticmethod
    def load_config(run_dir: str) -> dict:
        with open(os.path.join(os.path.abspath(run_dir), CONFIG_NAME)) as f:
            return json.load(f)

    # ---- save / restore ----------------------------------------------------
    def save(self, step: int, carry, smoothed_return: float) -> None:
        """Save `carry` at `step` (= updates completed). orbax handles retention
        (keep-best by `smoothed_return`, keep-last-N by recency)."""
        self._mgr.save(
            step,
            args=ocp.args.StandardSave(carry),
            metrics={"smoothed_return": float(smoothed_return)},
        )

    def wait(self) -> None:
        """Block until any in-flight async saves finish (call before exit)."""
        self._mgr.wait_until_finished()

    def latest_step(self):
        return self._mgr.latest_step()

    def best_step(self):
        return self._mgr.best_step()

    def all_steps(self):
        return sorted(self._mgr.all_steps())

    def restore(self, step: int, target_carry):
        """Restore the carry saved at `step`, typed by `target_carry`."""
        carry = self._mgr.restore(step, args=ocp.args.StandardRestore(target_carry))
        return jax.tree_util.tree_map(jax.numpy.asarray, carry)


def has_checkpoint(run_dir: str) -> bool:
    ckpt_dir = os.path.join(os.path.abspath(run_dir), "checkpoints")
    if not os.path.isdir(ckpt_dir):
        return False
    # orbax stores each step as a numbered subdirectory.
    return any(name.isdigit() for name in os.listdir(ckpt_dir))
