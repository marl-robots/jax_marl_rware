## IA2C tops the arena with a final team return of 20.0

**The task.** RWARE multi-robot warehouse: robots must fetch requested shelves, deliver them to goal stations, and RETURN them to empty rack slots before getting a new request counts. Reward is sparse: exactly +1 per completed delivery, shared nothing — each robot is rewarded only for its own deliveries. Episodes last 500 steps. With reward this sparse, teams must stumble onto a full fetch→deliver→return cycle before any learning signal exists at all.

### IA2C — rware-tiny-4ag

IA2C: independent advantage actor-critic — every robot learns alone from its own observations; nobody shares anything.

Team return went 0.1 → 19.5 across 20M environment steps (best window 28.9).

First reliable deliveries appeared after 4.5M steps.

Policy entropy fell 0.77 → 0.21 — the team moved from exploring to committed routines.

Aisle contention (forward-blocked moves) fell from 55% to 6% of agent-steps.

Late-training throughput splits early/mid/late = 9.4/5.9/4.7 deliveries per episode third.

*Generated without an LLM (deterministic fallback) — every number above is measured from results.csv.*