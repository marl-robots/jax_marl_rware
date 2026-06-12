# When "RWARE" isn't RWARE: how Jumanji's RobotWarehouse diverges from the canonical benchmark, and why it matters

*jax_marl3 investigation — June 2026. Every claim below is backed by a source-code
citation or a reproduction script in this repo
([`scripts/repro_jumanji_divergence.py`](../scripts/repro_jumanji_divergence.py),
results in [`docs/data/divergence_repro.json`](data/divergence_repro.json)).*

## TL;DR

The only widely-used JAX implementation of the RWARE multi-robot-warehouse
benchmark — Jumanji's `RobotWarehouse`, consumed downstream by Mava and by the
Sable paper — differs from the canonical environment
([semitable/robotic-warehouse](https://github.com/semitable/robotic-warehouse),
benchmarked by Papoudakis et al., 2021) on the exact mechanics that define the
benchmark's difficulty:

| # | Divergence | Canonical rware | Jumanji `RobotWarehouse` | Documented upstream? |
|---|---|---|---|---|
| D1 | Collision handling | Resolved (blocked agents wait) | **Episode terminates** | yes (docstring + Sable App. B.1) |
| D2 | Sequential-update artifacts | Simultaneous resolution, id-order-independent | **Legal convoy moves falsely terminate the episode, depending on agent id order; contending agents can occupy the same cell** | **no** |
| D3 | Reward | **Individual** +1 to the delivering agent | **Scalar team reward**, broadcast to all agents by Mava's wrapper | partially (docstring states global reward; the difference vs the benchmark is not noted) |
| D4 | Invalid actions | Must be learned (blocked by physics) | Masked: coerced to NOOP, mask exposed in obs | yes (docstring) |
| D5 | Observation | 71-feature FLATTENED encoding | 66 features, different content + action mask + step count | no |
| D6 | Default task | — | Default generator labeled "tiny-4ag-easy" but `shelf_rows=2` makes a 20-row grid = canonical **small**, with the doubled (easy) request queue | no |

Measured consequence (D1+D2): under a uniform-random policy on tiny-4ag,
**98.4% of Jumanji episodes terminate early, median length 58 of 500 steps**
(512 episodes). In canonical rware the same experiment gives 0% early
terminations — episodes always run the full 500 steps. Early-stage exploration
therefore happens on an effective horizon ~9× shorter, and the learning
problem morphs from *sparse-reward cooperative logistics* into *collision
avoidance first, logistics second*.

All code citations below are for Jumanji **1.1.1** (repo `HEAD` =
`9b0997e`, 2026-03-09), files under
`jumanji/environments/routing/robot_warehouse/`.

---

## 1. Background: what canonical RWARE is supposed to test

RWARE (Christianos, Papoudakis, Schäfer, Albrecht) is a deliberately *hard
cooperative* benchmark:

- **Sparse, individual reward**: exactly +1 to *the agent* that delivers a
  requested shelf (`rware/warehouse.py`, `RewardType.INDIVIDUAL` default).
  A full fetch→deliver→return cycle takes dozens of coordinated steps; until
  one completes, there is no signal at all. The benchmark paper (Papoudakis
  et al., 2021) and SEAC (Christianos et al., 2020) study *credit assignment*
  on exactly this structure.
- **Long horizon**: episodes are fixed at 500 steps; there is **no failure
  state**. Collisions are not possible — conflicting moves are *resolved*
  (the original computes a movement dependency graph; agents whose target is
  taken simply stay put; convoys/chains move together).
- **Decentralized execution**: each agent acts on a small local sensor window.

Our `jaxrware` env replicates these semantics **step-for-step** against the
original (`tests/test_parity_env.py`: identical state + identical action
sequences → exact equality of positions, rewards, dones over hundreds of
steps; observation encodings equal to the float).

## 2. The divergences, with code

### D1 — Collisions terminate the episode

`env.py:296-298`:

```python
horizon_reached = steps >= self.time_limit
done = collision | horizon_reached
```

The docstring is upfront (`env.py:107-109`: "episode termination: ... Any
agent selects an action which causes two agents to collide"), and
`utils.py:85-89` notes the behavior "is specific to the JAX version". So this
is a *documented design choice* — but it replaces the benchmark's defining
no-failure-state, fixed-horizon structure with a survival problem.

### D2 — The sequential update makes "collision" id-order-dependent (undocumented)

Jumanji applies agent moves **sequentially in agent-id order**
(`env.py:253-262`, a `lax.scan` over agents) with **unconditional** grid
writes (`utils_agent.py:281-283`):

```python
grid = grid.at[_AGENTS, current_position.x, current_position.y].set(0)
grid = grid.at[_AGENTS, new_position.x, new_position.y].set(agent_id + 1)
```

"Collision" is then defined as *my own grid cell no longer holds my id*
(`utils.py:85-105`). Consequences (reproduced deterministically in
`scripts/repro_jumanji_divergence.py`, scenarios A-C):

| Scenario | Canonical rware | Jumanji 1.1.1 |
|---|---|---|
| A. Two-agent **convoy** (follower has *lower* id), both step FORWARD | both move, episode continues | **episode terminates** — the leader's departure wipes the follower's grid mark |
| B. Same convoy, follower has *higher* id | both move, episode continues | episode continues |
| C. Head-on **contention** for one empty cell | resolved; no overlap; episode continues | both agents end on the **same cell**, then the episode terminates |

A follow-the-leader move — *the* basic traffic pattern of a warehouse — is
"a collision" or not depending purely on which agent got which index at
reset. This is not a design choice anyone documents; the environment's
dynamics depend on an arbitrary labeling, and the terminal "collision" state
of scenario C (two robots on one cell) cannot occur in the system being
modeled.

### D3 — Shared team reward instead of individual credit

`env.py:104-105` ("global reward shared by all agents"), `env.py:270`,
`env.py:478` (`reward += 1.0` — a scalar). Mava's wrapper then broadcasts it
(`mava/wrappers/jumanji.py`):

```python
reward = jnp.repeat(timestep.reward, self.num_agents)
```

Canonical RWARE's default is `RewardType.INDIVIDUAL`: only the delivering
agent is paid. The difference is not cosmetic — *who gets paid* is the credit
assignment problem that motivated SEAC and the benchmark itself. With a
shared reward, an idle agent gets the same +1 as the one that did the work;
the multi-agent credit problem the benchmark was built to expose is gone.

### D4 — Action masking

`utils.py:68-82`: invalid actions (e.g. forward while carrying into a shelf)
are coerced to NOOP, and the mask is part of the observation
(`types.py:121-133`). In the original there is no mask; agents must *learn*
that loaded robots cannot pass under racks. Smaller effect, but it changes
both the exploration problem and the information available to the policy.

### D5 — Observation encoding

Jumanji's per-agent view is 66 features (`utils.py:108-150`: 8 self + 8×5
neighbor-agent + 9×2 shelf), plus the action mask and episode step count.
The canonical FLATTENED encoding is 71 features with different content and
ordering. Policies and results are therefore not transferable in either
direction.

### D6 — The default task is mislabeled

`env.py:159-167`: the default generator is commented as
"robot_warehouse-tiny-4ag-easy" with `shelf_rows=2, column_height=8` →
grid height = (8+1)·2+2 = **20 rows: canonical *small*, not tiny** — combined
with the doubled request queue (8 for 4 agents = the `-easy` suffix). This is
the configuration behind the Jumanji paper's own RobotWarehouse benchmark
(App. F: "RobotWarehouse-v0", 6.5h training). Mava's scenario configs are
*not* affected: its `tiny-4ag.yaml` uses the correct
`shelf_rows=1, request_queue_size=4`.

## 3. Measured impact: the exploration horizon collapses

`scripts/repro_jumanji_divergence.py --part jumanji` (512 episodes,
canonical tiny-4ag geometry, uniform-random policy):

| | canonical rware | Jumanji 1.1.1 |
|---|---|---|
| episodes ended before the 500-step limit | **0%** | **98.4%** |
| median episode length | 500 | **58** |
| mean / p90 episode length | 500 / 500 | 99.3 / 257 |

A random policy is what every RL algorithm *is* at initialization. On the
canonical benchmark, early training explores 500-step trajectories of a
no-failure world, and the challenge is finding the sparse delivery signal.
On Jumanji's variant, the expected lifetime of an exploring team is under a
minute of warehouse time; the gradient signal that actually shapes early
training is *episode termination*, a signal that does not exist in the
benchmark this environment is named after.

## 4. How this propagates into published research

- **Jumanji (ICLR 2024)** is explicit that it doesn't do MARL with this env:
  "Jumanji does lack a true multi-agent training algorithm ... Jumanji trains
  in the style of centralized training with centralized execution and treats
  the environment as a single-agent one" (App. A.5); its RobotWarehouse
  network is a transformer over all agents' features (App. B). Fine on its
  own terms — but the env is the one downstream MARL libraries adopted.
- **Mava** benchmarks its MARL systems on 15 "Multi-Robot Warehouse"
  scenarios named identically to the canonical tasks (tiny-2ag, tiny-4ag,
  small-4ag, ...), all through the Jumanji env + reward-broadcast wrapper.
- **Sable (Mahjoub et al., 2024)**, whose evaluation is built on Mava,
  acknowledges D1 in its Appendix B.1 — "there is a *minor* difference in how
  collisions are handled. The original implementation has some logic to
  resolve collisions, whereas the Jumanji implementation simply ends an
  episode if two agents collide" — and at the same time interprets its RWARE
  results against findings from the canonical benchmark (e.g. citing MAT's
  weakness "in sparse reward settings such as RWARE (Papoudakis et al.,
  2020)"). Our measurements above show "minor" is not a defensible adjective:
  the change flips the dominant early-training signal, and D2/D3 (which no
  paper mentions) alter the dynamics and the credit structure on top.

**What stays valid:** comparisons *within* one paper/codebase (every method
sees the same environment) — Sable vs MAT vs MAPPO on Jumanji-RWARE is a fair
race. **What breaks:** any cross-paper comparison ("our X beats the MAPPO
numbers reported on rware-tiny-4ag"), any transfer of hyperparameters or
qualitative conclusions ("algorithm class Y struggles on sparse-reward
RWARE") between the two environments, and any claim that results on the
Jumanji variant characterize the benchmark of Papoudakis et al.

## 5. Reproducing

```bash
# Jumanji behavior (conda env with jumanji 1.1.1)
python scripts/repro_jumanji_divergence.py --part jumanji
# canonical behavior (conda env with rware)
python scripts/repro_jumanji_divergence.py --part rware
# canonical-vs-this-repo parity (the faithful alternative)
python -m pytest tests/test_parity_env.py -q
```

## References

- Papoudakis, Christianos, Schäfer, Albrecht. *Benchmarking Multi-Agent Deep
  RL Algorithms in Cooperative Tasks.* NeurIPS D&B 2021.
- Christianos, Schäfer, Albrecht. *Shared Experience Actor-Critic for
  Multi-Agent Reinforcement Learning.* NeurIPS 2020.
- Bonnet et al. *Jumanji: a Diverse Suite of Scalable RL Environments in
  JAX.* ICLR 2024. (App. A.5, App. F.)
- Mahjoub et al. *Sable: a Performant, Efficient and Scalable Sequence Model
  for MARL.* arXiv:2410.01706. (App. B.1.)
- InstaDeep Mava: `mava/wrappers/jumanji.py`,
  `mava/configs/env/scenario/tiny-4ag.yaml` (develop branch, June 2026).
- semitable/robotic-warehouse (canonical RWARE).
