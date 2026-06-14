## MAPPO tops the arena with a final team return of 36.2

**The task.** RWARE multi-robot warehouse: robots must fetch requested shelves, deliver them to goal stations, and RETURN them to empty rack slots before getting a new request counts. Reward is sparse: exactly +1 per completed delivery, shared nothing — each robot is rewarded only for its own deliveries. Episodes last 500 steps. With reward this sparse, teams must stumble onto a full fetch→deliver→return cycle before any learning signal exists at all.

### MAPPO — rware-tiny-4ag

MAPPO: PPO with a centralised critic — the strongest standard baseline; central 'coach' critic during training, decentralised execution.

Team return went 0.1 → 35.4 across 60M environment steps (best window 42.8).

First reliable deliveries appeared after 0.0M steps.

Policy entropy fell 0.94 → 0.18 — the team moved from exploring to committed routines.

Aisle contention (forward-blocked moves) fell from 31% to 2% of agent-steps.

Late-training throughput splits early/mid/late = 12.6/11.7/11.9 deliveries per episode third.

### MAA2C — rware-tiny-4ag

MAA2C: A2C with a centralised critic — during training a 'coach' sees all robots at once to judge actions, but each robot still acts only on its own local view.

Team return went 0.2 → 2.8 across 15M environment steps (best window 5.3).

First reliable deliveries appeared after 0.1M steps.

Policy entropy fell 0.81 → 0.43 — the team moved from exploring to committed routines.

Aisle contention (forward-blocked moves) fell from 35% to 10% of agent-steps.

Late-training throughput splits early/mid/late = 2.3/0.4/0.2 deliveries per episode third.

**Head to head:** MAPPO (36.2) > MAA2C (2.9).

*Generated without an LLM (deterministic fallback) — every number above is measured from results.csv.*