## MAPPO tops the arena with a final team return of 36.2

**The task.** RWARE multi-robot warehouse: robots must fetch requested shelves, deliver them to goal stations, and RETURN them to empty rack slots before getting a new request counts. Reward is sparse: exactly +1 per completed delivery, shared nothing — each robot is rewarded only for its own deliveries. Episodes last 500 steps. With reward this sparse, teams must stumble onto a full fetch→deliver→return cycle before any learning signal exists at all.

### MAPPO — rware-tiny-4ag

MAPPO: PPO with a centralised critic — the strongest standard baseline; central 'coach' critic during training, decentralised execution.

Team return went 0.1 → 35.4 across 60M environment steps (best window 42.8).

First reliable deliveries appeared after 5.1M steps.

Policy entropy fell 0.94 → 0.18 — the team moved from exploring to committed routines.

Aisle contention (forward-blocked moves) fell from 31% to 2% of agent-steps.

Late-training throughput splits early/mid/late = 12.6/11.7/11.9 deliveries per episode third.

### IA2C — rware-tiny-4ag

IA2C: independent advantage actor-critic — every robot learns alone from its own observations; nobody shares anything.

Team return went 0.1 → 19.5 across 20M environment steps (best window 28.9).

First reliable deliveries appeared after 4.5M steps.

Policy entropy fell 0.77 → 0.21 — the team moved from exploring to committed routines.

Aisle contention (forward-blocked moves) fell from 55% to 6% of agent-steps.

Late-training throughput splits early/mid/late = 9.4/5.9/4.7 deliveries per episode third.

### IPPO — rware-tiny-4ag

IPPO: independent PPO — like IA2C but with PPO's clipped, more stable policy updates; still no information sharing.

Team return went 0.1 → 5.4 across 20M environment steps (best window 8.4).

First reliable deliveries appeared after 2.7M steps.

Policy entropy fell 0.64 → 0.14 — the team moved from exploring to committed routines.

Aisle contention (forward-blocked moves) fell from 26% to 2% of agent-steps.

Late-training throughput splits early/mid/late = 5.2/0.2/0.0 deliveries per episode third.

### MAA2C — rware-tiny-4ag

MAA2C: A2C with a centralised critic — during training a 'coach' sees all robots at once to judge actions, but each robot still acts only on its own local view.

Team return went 0.1 → 4.1 across 20M environment steps (best window 6.7).

First reliable deliveries appeared after 5.2M steps.

Policy entropy fell 0.85 → 0.44 — the team moved from exploring to committed routines.

Aisle contention (forward-blocked moves) fell from 35% to 9% of agent-steps.

Late-training throughput splits early/mid/late = 3.5/0.4/0.1 deliveries per episode third.

### SEAC — rware-tiny-4ag

SEAC: shared-experience actor-critic — robots learn from each other's experiences via importance-weighted off-policy corrections.

Team return went 0.1 → 0.4 across 5M environment steps (best window 1.1).

Policy entropy fell 0.95 → 0.30 — the team moved from exploring to committed routines.

Aisle contention (forward-blocked moves) rose from 19% to 23% of agent-steps.

**Head to head:** MAPPO (36.2) > IA2C (20.0) > IPPO (5.4) > MAA2C (4.0) > SEAC (0.3).

*Generated without an LLM (deterministic fallback) — every number above is measured from results.csv.*