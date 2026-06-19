# Action Masking for value-based RWARE (EMAX) — status & future work

**Status: implemented, gated, opt-in, and NOT yet validated. Parked as future
work.** The feature is correct-by-construction in design and verified to run, but
we do *not* yet have an empirical result showing it helps the model learn. Pick
this up when returning to EMAX performance work.

## What it does

RWARE actions are never *illegal* — only sometimes *ineffective* (a no-op). This
feature lets the env emit a per-agent mask of **provably-no-op** actions and
applies it so the agent stops wasting exploration on them. Because a no-op action
produces the exact same transition as `NOOP` (which is always available), masking
it **provably preserves the optimal policy** — it only removes wasted exploration.

Opt-in via `DQNConfig.use_action_mask` (default `False`) / `--action-mask` on
`scripts/train_dqn_fast.py`. **Off ⇒ byte-identical to the unmasked path** (the
DQN parity tests pass 4/4 unchanged).

## The mask predicate (`Warehouse.action_masks(state) -> [N, A]`)

Masks ONLY actions whose effect is provably identical to `NOOP`:

| Action | Masked when | Source of truth |
|---|---|---|
| `FORWARD` | facing a **static grid boundary** (clamped → no move) | `collision.py` `_req_location` clamps target |
| `TOGGLE_LOAD` | `(not carrying & no shelf on cell)` **or** `(carrying & on highway)` | `env.py` step: pickup/drop predicates |
| `NOOP`, `LEFT`, `RIGHT` | **never** | turning is always effective; NOOP is the wait/safety valve |

**Deliberately NOT masked (the trap that breaks the guarantee):** `FORWARD` into a
cell occupied by an agent or a standing shelf. Whether that move is blocked depends
on the *other* agents' simultaneous moves (convoy resolution in `collision.py`), so
it is **not** a guaranteed no-op. Masking it would delete legal coordinated moves
and collapse the policy ("avoids the action forever"). The mask must stay a strict
subset of true no-ops, computed from static structure + the agent's own state.

The mask is a pure function of the agent's **own local observation** (carrying,
on_highway, dir, x/y, shelf-here), so it is decentralized-faithful — not privileged
info.

## How it's wired (the consistency that prevents value bias)

The mask is applied **identically** at three points — this consistency is what
avoids the `-1e8`-leaks-into-learned-values bias:

1. **UCB greedy** — `argmax` over masked `Q_mean + β·Q_std`.
2. **ε-random** — sampled uniformly over *valid* actions (`jax.random.categorical`
   over masked logits), so exploration never wastes a step either.
3. **Bootstrap target** — masked `max` over next-state ensemble-mean Q in
   `emax_update` (via `batch["action_mask"]`).

Design choice: it is an **env method**, not stuffed into `info` — the mask is
needed at *selection* time (current state, before stepping), which fits the
fully-jitted on-device `lax.scan` trainer. The mask is stored in the on-device
ring buffer aligned with obs; when off it is a zero-size placeholder.

### Files
- `jaxrware/env.py` — `Warehouse.action_masks`.
- `algorithms/dqn_config.py` — `use_action_mask` flag.
- `algorithms/dqn.py` — `emax_update` masks the bootstrap-max target; `feed_batch`
  optional `mask_b`.
- `algorithms/dqn_train.py` — masked selection + buffer + threading (gated).
- `scripts/train_dqn_fast.py` — `--action-mask`.

## Verification done
- `tests/test_dqn_parity.py` — **4/4 pass** (unmasked path unchanged).
- Masked smoke — buffer warms, a masked gradient step runs with finite loss on GPU.

## What is NOT done (the open work)
- **No performance result.** The one A/B run (FF EMAX, seed 2, 600 iters, masked vs
  unmasked) was **inconclusive — both runs were pre-takeoff**. The unmasked baseline
  only reached ~0.6 deliveries/episode by iter 600 (never ≥2); the masked run was
  stopped early at iter 350. EMAX takeoff on RWARE is late and high-variance, so
  600 iters / 1 seed is too short to judge.

## Next steps when resuming
1. **Rule out a regression first.** Run the pre-masking commit (`2229cca`) on seed 2,
   unmasked, to confirm the slow baseline is seed variance, not the trainer rewrite.
   (The unmasked code path is RNG-identical and parity passes, so regression is
   unlikely — but confirm with numbers.)
2. **Real A/B:** 2–3 seeds × ≥1000 iters, masked vs unmasked, FF EMAX on tiny-4ag.
   Compare takeoff iteration and plateau deliveries.
3. **Expect a bigger effect on harder maps** than tiny-4ag, where exploration is
   easy and there's little wasted exploration to reclaim. A harder map is also the
   honest setting for the broader "does EMAX beat MAPPO" question.
4. Leftover run dirs from the inconclusive A/B: `runs/{ab_unmask, ab_mask, _smoke_mask}`.
