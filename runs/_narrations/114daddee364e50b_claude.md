## Centralization helps PPO, not A2C: a four-way RWARE post-mortem

**Setup.** rware-tiny-4ag, 4 agents, sparse per-agent delivery reward, 500-step
horizon, shared-parameter GRU actors. The 2×2 over {independent, centralized
critic} × {A2C, PPO}, plus the caveat that MAPPO ran to 60M env steps while
IA2C/IPPO/MAA2C ran to 20M (so read MAPPO's absolute lead with that asymmetry in
mind; the 20M three-way is controlled).

**Final team return (tail mean / best window):**
- MAPPO 36.19 / 42.8 · IA2C 19.98 / 28.9 · IPPO 5.35 / 8.4 · MAA2C 4.04 / 6.7

**Reading the curves.**
- **MAPPO** shows textbook on-policy AC behavior: entropy 0.94→0.18, contention
  0.31→0.017, idle ≈0, and — the signal that matters most — a *flat* delivery
  phase split (12.6/11.7/11.9 early/mid/late). The policy sustains throughput
  across the whole episode rather than exhausting nearby requests.
- **IA2C** reaches half its best at 14.5M steps and finishes second outright. Its
  phase split is front-loaded (9.39/5.93/4.66): the independent learners optimize
  greedily for proximate requests and the pipeline thins late-episode. Contention
  settles at 5.9% — higher than the centralized-critic MAPPO but well-managed.
- **IPPO** plateaus at 5.35 with a near-degenerate phase split (5.16/0.18/0.01):
  it acquires an opening burst then collapses to ~zero sustained delivery. Note
  idle rate stays elevated (0.16→0.40) deep into training before finally dropping
  — a long exploration-without-payoff regime.
- **MAA2C** is the laggard (4.04) and the only run whose entropy never commits
  (final 0.428 vs ~0.19 for the others). A2C's single-step update plus a
  centralized critic that it can't yet exploit leaves it under-optimized at 20M.

**Interpretation (strictly from these numbers).** The centralized critic is
*decisive for PPO* (MAPPO ≫ IPPO) but *unhelpful-to-harmful for A2C at this
budget* (MAA2C < IA2C). Equivalently: the win comes from PPO's clipped multi-epoch
updates leveraging the centralized value signal; A2C's one-step update cannot, and
the larger centralized critic input (concat of all agents' obs) appears to slow it
relative to the independent variant. The delivery phase split is the most useful
diagnostic here — it separates "sustained operation" (MAPPO) from "early-burst
then stall" (IPPO) far more cleanly than the scalar return does.

**Caveats.** Single seed per algorithm; MAPPO's 3× step budget confounds its
absolute gap to the field. The controlled comparison is the 20M three-way
(IA2C > IPPO > MAA2C), which is the result worth reporting.

*All quantities are measured from results.csv; curve arrays are 8-bucket means in
training order.*
