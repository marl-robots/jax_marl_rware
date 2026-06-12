# Demo guide — running and presenting the MARL Arena

*For the final-project presentation. Everything here works offline (no
internet needed at the venue) except the optional live-LLM narration.*

## 1. Launching (2 minutes before the demo)

```bash
# WSL terminal
source ~/miniconda3/etc/profile.d/conda.sh && conda activate jax_env_1
cd /mnt/c/Users/user1/projects/jax_marl3
streamlit run arena/app.py
# opens http://localhost:8501 — use a Chromium browser, F11 fullscreen
```

The app reads `runs/` only — no JAX, no GPU needed, starts in seconds. A
laptop with the repo + conda env is the whole dependency list.

## 2. Suggested flow (10–12 minutes)

**Act 1 — the hook (Replay theater, 3 min).**
Open 🎬 *Replay theater*, pick `mappo_evo_tiny-4ag_seed2`. Hit **EVOLUTION
TOUR**: the audience watches the same fixed episode seed go from drunk
wandering (update 0) to an organized delivery operation — *every visible
difference is learning, not luck*. Talking points while it plays:
- each circle is a robot acting on a tiny local sensor window (no global view)
- teal glowing shelves are "requested"; flash rings are deliveries
- reward is +1 per delivery *to the robot that delivered* — nothing else;
  everything they do was discovered from that single sparse signal
- this is a *replay log* re-rendered in the browser, captured during training
  at ~zero cost because the env itself runs in JAX on GPU

**Act 2 — the science (Overview + Compare, 3 min).**
🏆 *Overview*: medals + the race curve. Story: MAPPO (centralised critic +
PPO) wins; independent learners struggle — this reproduces the published
benchmark ordering. 📊 *Compare*: show entropy (exploration → commitment)
and contention dropping while deliveries rise — the team *learned traffic
rules* nobody programmed.

**Act 3 — the research finding (Research tab, 3 min).**
The "industry standard" JAX version of this benchmark (Jumanji, used by Mava
and the Sable paper) is **a different game**: it kills the episode on
"collision", and its collision is id-order-dependent — show the scenario
table (a legal convoy move ends the episode for one agent ordering, not the
other). Headline: under a random policy its median episode is **58 of 500
steps** — early training optimizes survival, not logistics. Our env is
validated **step-for-step** against the original (`tests/test_parity_env.py`)
and still hits ~200k env steps/s on CPU (vs 2.7k for the original python env)
— faithful AND fast. Full writeup: `docs/jumanji_mava_divergence.md`.

**Act 4 — the AI commentator (Story tab, 2 min).**
🧠 *Story*: pick audience "general", hit **Narrate**. The LLM gets *only
measured numbers* (the JSON facts are on screen-adjacent cache) and produces
the plain-language story of the training. Offline fallbacks: a local Ollama
model, then a deterministic template — the panel can never be empty.
Cached narrations live in `runs/_narrations/` so the demo does not depend on
a live model.

## 3. Failure modes & recoveries

| Symptom | Fix |
|---|---|
| Theater empty | `python -m scripts.record_replay --run-dir runs/<run>` (needs jax env) |
| Story tab slow | use the cached narration (auto-shown) or backend=template |
| Streamlit port busy | `streamlit run arena/app.py --server.port 8502` |
| Animation frozen | the tab was backgrounded — browsers pause rendering; refocus the window |
| Claude backend "not logged in" | one-time `claude /login` in any terminal, or just use ollama/template |

## 4. Regenerating demo data from scratch

```bash
bash scripts/train_queue.sh              # ~24h on the RTX 3050 (see file)
python -m scripts.record_replay --run-dir runs/<run> --episodes 2
python scripts/bench_speed.py --impl jaxrware --device gpu   # GPU idle!
python scripts/repro_jumanji_divergence.py --part jumanji    # jumanji env
python scripts/repro_jumanji_divergence.py --part rware      # jax_env_1
```
