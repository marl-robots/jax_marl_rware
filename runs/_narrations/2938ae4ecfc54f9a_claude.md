## The lone wolves: how four un-coordinated robots still learned to deliver

**Headline.** With every robot learning entirely on its own — no shared brain, no
central coach, each one rewarded only for its own work — the team still reached
**20 deliveries per episode** (peaking near 29). For an algorithm this simple,
that is a genuinely strong result.

**What "independent" means here.** This is IA2C. Imagine four employees who never
talk, never see a shared dashboard, and are each paid only for jobs they
personally finish. Each one has to figure out the warehouse — fetch a requested
shelf, deliver it, return it — from its own narrow view alone. There is no
mechanism rewarding teamwork; any cooperation that emerges is purely each robot
staying out of the others' way to maximize its own score.

**The training story.**
- **Slow start, then a jump.** For the first several million steps the team is
  near-flat (under 1 delivery per episode) while it hunts for the sparse reward.
  First reliable deliveries land around 4.5 million steps; from there return
  climbs 4 → 7 → 15 → 20.
- **Aisles clear out.** Early on, contention is severe — more than half of all
  forward moves are blocked by another robot. By the end that falls to about 6%,
  and idle time drops to essentially zero. The robots learned to share space
  without ever being told to.
- **A telling weakness.** Look at *when* the deliveries happen across an episode:
  9.4 in the first third, 5.9 in the middle, 4.7 at the end. The team clears the
  convenient nearby requests quickly, then can't keep the pipeline as full
  late-episode. Compare this to MAPPO, whose deliveries stay flat across the
  episode — that even profile is exactly what central coordination buys you.

**Bottom line.** IA2C shows how far simple independent learning can go on a task
that was designed to need cooperation — strong enough, in fact, to beat the more
sophisticated centralized methods (IPPO, MAA2C) at the same 20-million-step
budget. Its limitation is endurance, not lift-off: it learns to deliver, but not
yet to deliver *steadily* for the whole episode.

*All numbers are measured from this run's results.csv.*
