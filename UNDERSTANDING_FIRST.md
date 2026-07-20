# Understanding-first, reward-last learning — a research direction

**Status: a vision / research direction, not yet a method or any code.** Captured
here so it persists exactly as framed, to be developed before anything is built.

## The problem with how the agents learn now

Every algorithm in this repo (IQL, EMAX, MAPPO, …) is **reward-first**: the agent
sees an observation, picks an action, and the only thing teaching it is whether the
final reward went up. It never forms any notion of what a shelf *is*, what picking
one up *means*, or *why* it is doing anything. Reward is doing 100% of the teaching.
That is why learning is so slow and feels stupid — a human grasps the point after a
couple of successful deliveries; these agents need millions of steps of reward
signal to back-propagate the same idea.

## The proposed inversion: understand first, use reward last

Learning should **not depend only on reward**. Instead the agent should first
understand its world and itself, and only then bring reward in — as confirmation,
not as the sole driver.

**Stage 1 — Learn the world, with no reward involved.**
By moving around and acting, the agent builds two internal models:
- **The map**: where it can go, walls, highways, where shelves sit, where the goal
  cells are.
- **Its own capabilities (action → effect)**: "forward moves me unless something
  blocks me", "toggle while standing on a shelf picks it up", "toggle while carrying
  puts it down". It learns what it *can do* — stroll, pick up, drop — as cause and
  effect, not as a table of Q-values.

**Stage 2 — Infer what's expected of it.**
Once it knows the pieces exist — there are shelves, they are movable, there are goal
cells — it can *reason out the likely point on its own*: "the environment probably
wants me to bring shelves to a goal." It forms a **hypothesis about the task from the
structure it has learned**, before anyone rewards it. (This is the most distinctive
and the hardest part of the idea.)

**Stage 3 — Test the theory against reward.**
*Now* reward enters — not as something to climb blindly, but as the way the agent
**confirms or rejects** its hypotheses and sharpens its behavior.

So reward stops being the teacher and becomes the **final exam**. The actual teaching
is done by the agent's own exploration and the understanding it builds on top of it.

## Why this stays faithful to the benchmark

This is faithful in the strict sense that matters here: **the environment and its
reward function are left untouched.** Reward is never modified, shaped, or augmented
— it is only consumed, and only at Stage 3, to confirm and refine. The map- and
capability-learning of Stages 1–2 are the agent's own internal understanding, not a
change to the task. So an agent built this way can still be evaluated on canonical
RWARE, and its numbers remain comparable. (Contrast: reward shaping would alter the
reward the agent optimizes — that is the one thing that breaks faithfulness, and it
is explicitly out.)

## Honest open questions (to work through before building)
- How does the agent **represent** "I have learned the map" and "I know what toggle
  does"? What is that understanding, concretely, as data structures / signals?
- **Stage 2 is the crux**: how does an agent turn "here is what exists and what I can
  do" into "here is what is probably wanted of me"? This is the novel, unproven step.
- How do Stages 1→2→3 connect — strictly sequential, or interleaved?
- How is any of this measured/validated without leaning back on reward as the metric?

## Relation to what's already here
- The committed **action-masking** work (`ACTION_MASKING.md`) is a tiny, hand-coded
  instance of injecting *structural understanding* ("this action does nothing here").
  This vision is the far larger ambition: the agent **learning** that understanding
  for itself rather than being handed it.
- The project's LLM/arena layer is a natural place for the "understanding" and the
  "what's expected of me" reasoning to live, if that route is taken — but that is a
  later design choice, not part of the core idea above.

## Next
Develop the concept — especially what "the agent has learned the map / knows what its
actions do" looks like concretely, and how Stage 2 could actually happen — *before*
committing to any implementation or method.
