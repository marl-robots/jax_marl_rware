## From chaos to clockwork: watching MAPPO learn a warehouse from scratch

**The headline.** In 50 million steps of practice, a team of four robots went
from bumping aimlessly around an empty-handed warehouse to running a smooth,
near-continuous delivery line — **35 deliveries per episode**, with almost no
collisions and almost no idle time.

**What they had to figure out.** Each robot sees only a small patch around
itself and earns a single point only when it personally completes a full job:
fetch a requested shelf, take it to a station, and return it. Nothing rewards
"getting close." So at the start there is no useful feedback at all — the team
has to *accidentally* complete deliveries before learning can even begin. This is
why the opening is the hard part.

**The story in the replay.** If you scrub the EVOLUTION TOUR, you are watching
this exact progression (same warehouse, same requests every time — so every
change you see is genuinely learning, not luck):

- **Early (updates 0–1000):** near-random motion. Deliveries hover under 1 per
  episode, and the aisles are jammed — almost half of all forward moves are
  blocked by another robot at the worst point (contention peaks at 47%).
  Meanwhile the robots quickly learn the one easy lesson — *stop standing still* —
  and idle time crashes from 17% to nearly zero within the first stretch.
- **Lift-off (around 3 million steps):** the first reliable deliveries appear.
  Once the team tastes reward, throughput climbs fast — 7, then 15, then 20+
  deliveries per episode.
- **Mastery (later updates):** deliveries reach the mid-30s, contention falls to
  under 2%, and the robots' behavior becomes decisive — their "randomness"
  (entropy) drops from 0.77 to 0.18 as exploration gives way to committed
  routines. Best of all, deliveries become evenly spread across the whole
  episode (about 12 in each third): a team running a sustained operation, not one
  that empties the easy shelves and stops.

**Bottom line.** No one programmed traffic rules, routes, or roles. From a single
sparse reward signal, the team discovered how to keep an aisle clear, when to
yield, and how to keep the delivery pipeline full for 500 steps straight. The
tour lets your audience watch that whole journey in a few seconds.

*Every figure is measured from this run's results.csv.*
