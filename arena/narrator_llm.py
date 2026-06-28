"""LLM-powered narration of training runs for the dashboard's Story tab.

Three backends, tried in order, all optional at runtime:

1. **Claude CLI** — shells out to the locally installed Claude Code binary
   (``claude -p``), using the user's existing subscription; no API key.
2. **Ollama** — local model over the Ollama HTTP API (offline-friendly).
3. **Template** — deterministic story assembled from the same facts; always
   available, so a live demo can never end up with an empty panel.

Grounding rule (the project's validation philosophy applied to prose): the
model only ever sees `build_facts()` — measured numbers from results.csv — and
the prompt forbids inventing anything beyond them. Narrations are cached to
``runs/_narrations/`` so a demo doesn't depend on a live model.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import urllib.request

from arena.run_data import RunData, summarize

CACHE_DIR = os.path.join("runs", "_narrations")

AUDIENCES = {
    "general": "a general audience at a university demo day (no RL background; "
               "explain ideas with everyday analogies, zero jargon)",
    "students": "computer-science students who know basic ML but not "
                "multi-agent RL (define terms briefly when first used)",
    "experts": "RL researchers (be precise and technical, no hand-holding)",
}


# ---------------------------------------------------------------------------
# facts: the ONLY thing the model is allowed to know
# ---------------------------------------------------------------------------
def _curve_digest(run: RunData, col: str, buckets: int = 8) -> list[float]:
    """The curve compressed to `buckets` bucket-means (training-order)."""
    if col not in run.df.columns or not len(run.df):
        return []
    s = run.df[col]
    n = max(1, len(s) // buckets)
    return [round(float(s[i:i + n].mean()), 3) for i in range(0, len(s), n)][:buckets]

ALGO_BLURBS = {
    "ia2c": "IA2C: independent advantage actor-critic — every robot learns "
            "alone from its own observations; nobody shares anything.",
    "ippo": "IPPO: independent PPO — like IA2C but with PPO's clipped, more "
            "stable policy updates; still no information sharing.",
    "maa2c": "MAA2C: A2C with a centralised critic — during training a "
             "'coach' sees all robots at once to judge actions, but each "
             "robot still acts only on its own local view.",
    "mappo": "MAPPO: PPO with a centralised critic — the strongest standard "
             "baseline; central 'coach' critic during training, decentralised "
             "execution.",
    "seac": "SEAC: shared-experience actor-critic — robots learn from each "
            "other's experiences via importance-weighted off-policy "
            "corrections.",
}


def build_facts(runs: list[RunData]) -> dict:
    """Measured facts for the prompt: summaries + compressed curves."""
    out = {"environment": (
        "RWARE multi-robot warehouse: robots must fetch requested shelves, "
        "deliver them to goal stations, and RETURN them to empty rack slots "
        "before getting a new request counts. Reward is sparse: exactly +1 "
        "per completed delivery, shared nothing — each robot is rewarded "
        "only for its own deliveries. Episodes last 500 steps."),
        "runs": []}
    for r in runs:
        s = summarize(r)
        s["algorithm_explained"] = ALGO_BLURBS.get(r.family, "")
        s["curves"] = {
            "team_return_over_training": _curve_digest(r, "mean_episode_returns"),
            "deliveries_over_training": _curve_digest(r, "deliveries"),
            "policy_entropy_over_training": _curve_digest(r, "entropy"),
            "contention_rate_over_training": _curve_digest(r, "block_rate"),
            "idle_rate_over_training": _curve_digest(r, "idle_rate"),
        }
        out["runs"].append(s)
    return out


def build_prompt(facts: dict, audience: str = "general") -> str:
    aud = AUDIENCES.get(audience, AUDIENCES["general"])
    multi = len(facts["runs"]) > 1
    compare = (
        "4. **Head to head** — compare the algorithms' outcomes and what the "
        "differences suggest (e.g. centralised critic vs independent, PPO vs "
        "A2C), strictly from the numbers.\n" if multi else "")
    return f"""You are the resident commentator of a Multi-Agent RL Arena — a live \
dashboard where reinforcement-learning algorithms train teams of warehouse \
robots. Write an engaging narration of the training below for {aud}.

HARD RULES:
- Every quantitative claim must come from the FACTS block. Do not invent \
numbers, events, or intentions ("the robots decided to cooperate" is out; \
"contention dropped from 30% to 9% while deliveries tripled" is in).
- The curve arrays are bucket-means in training order (first = start of \
training, last = end).
- Use markdown. Be vivid but honest; uncertainty is stated, not papered over.

Structure:
1. **Headline** — one punchy sentence summarizing the outcome.
2. **What the robots had to learn** — the task, in terms this audience gets.
3. **The training story** — how performance, entropy (exploration), \
contention and idleness evolved; pick out the 2–3 most interesting turns in \
the curves and what changed.
{compare}5. **Bottom line** — what a viewer should remember.

Keep it under 450 words.

FACTS:
```json
{json.dumps(facts, indent=1)}
```"""


# ---------------------------------------------------------------------------
# backends
# ---------------------------------------------------------------------------
def _find_claude_exe() -> str | None:
    exe = shutil.which("claude")
    if exe:
        return exe
    # Claude Code's per-version binary on Windows; from WSL via /mnt/c interop
    pats = [
        os.path.expandvars(r"%APPDATA%\Claude\claude-code\*\claude.exe"),
        "/mnt/c/Users/*/AppData/Roaming/Claude/claude-code/*/claude.exe",
        # Claude desktop is MSIX-packaged: its Roaming appdata is virtualized
        # and only visible at the package's LocalCache path from WSL
        "/mnt/c/Users/*/AppData/Local/Packages/Claude_*/LocalCache/Roaming/"
        "Claude/claude-code/*/claude.exe",
    ]
    for pat in pats:
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[-1]  # newest version
    return None


def narrate_claude(prompt: str, timeout: int = 180) -> str | None:
    exe = _find_claude_exe()
    if not exe:
        return None
    try:
        # prompt goes via stdin: argv has a ~32k limit on Windows and the
        # facts JSON can approach it
        res = subprocess.run(
            [exe, "-p", "--output-format", "text"],
            input=prompt, capture_output=True, text=True, timeout=timeout,
            encoding="utf-8", errors="replace")
        text = (res.stdout or "").strip()
        return text if res.returncode == 0 and len(text) > 80 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def _ollama_hosts() -> list[str]:
    hosts = ["http://localhost:11434"]
    # under WSL2 (non-mirrored networking) Windows services live at the
    # default-gateway address from resolv.conf
    try:
        with open("/etc/resolv.conf") as f:
            for line in f:
                if line.startswith("nameserver"):
                    hosts.append(f"http://{line.split()[1]}:11434")
    except OSError:
        pass
    return hosts


def _find_ollama_exe() -> str | None:
    exe = shutil.which("ollama")
    if exe:
        return exe
    for pat in (os.path.expandvars(r"%LOCALAPPDATA%\Programs\Ollama\ollama.exe"),
                "/mnt/c/Users/*/AppData/Local/Programs/Ollama/ollama.exe"):
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[-1]
    return None


def _clean_llm_text(text: str) -> str:
    # strip <think> blocks some local models emit
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def narrate_ollama(prompt: str, model: str = "qwen3:4b",
                   timeout: int = 300) -> str | None:
    # 1) HTTP API (works when the dashboard runs on the same OS as Ollama)
    body = json.dumps({"model": model, "prompt": prompt, "stream": False,
                       "options": {"temperature": 0.4}}).encode()
    for host in _ollama_hosts():
        try:
            req = urllib.request.Request(
                f"{host}/api/generate", data=body,
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                text = _clean_llm_text(json.loads(resp.read()).get("response", ""))
            if len(text) > 80:
                return text
        except OSError:
            continue
    # 2) CLI via Windows interop: from WSL, ollama.exe reaches the Windows
    #    localhost service without rebinding it to other interfaces
    exe = _find_ollama_exe()
    if exe:
        try:
            res = subprocess.run(
                [exe, "run", model], input=prompt, capture_output=True,
                text=True, timeout=timeout, encoding="utf-8", errors="replace")
            text = _clean_llm_text(res.stdout or "")
            if res.returncode == 0 and len(text) > 80:
                return text
        except (OSError, subprocess.TimeoutExpired):
            pass
    return None


def narrate_template(facts: dict) -> str:
    """Deterministic fallback story — same facts, no model required."""
    parts = []
    runs = facts["runs"]
    best = max(runs, key=lambda s: s.get("final_return", 0.0))
    parts.append(
        f"## {best['algorithm']} tops the arena with a final team return of "
        f"{best.get('final_return', 0):.1f}\n")
    parts.append(
        "**The task.** " + facts["environment"] + " With reward this sparse, "
        "teams must stumble onto a full fetch→deliver→return cycle before any "
        "learning signal exists at all.\n")
    for s in runs:
        c = s["curves"]
        ret, ent = c["team_return_over_training"], c["policy_entropy_over_training"]
        blk = c["contention_rate_over_training"]
        line = [f"### {s['algorithm']} — {s.get('env', '')}"]
        if s.get("algorithm_explained"):
            line.append(s["algorithm_explained"])
        if ret:
            line.append(
                f"Team return went {ret[0]:.1f} → {ret[-1]:.1f} across "
                f"{s.get('env_steps', 0) / 1e6:.0f}M environment steps "
                f"(best window {s.get('best_return', 0):.1f}).")
        if s.get("steps_to_first_delivery") is not None:
            line.append(
                f"First reliable deliveries appeared after "
                f"{s['steps_to_first_delivery'] / 1e6:.1f}M steps.")
        if ent:
            line.append(
                f"Policy entropy fell {ent[0]:.2f} → {ent[-1]:.2f} — the team "
                "moved from exploring to committed routines.")
        if blk:
            trend = "rose" if blk[-1] > blk[0] else "fell"
            line.append(
                f"Aisle contention (forward-blocked moves) {trend} from "
                f"{blk[0] * 100:.0f}% to {blk[-1] * 100:.0f}% of agent-steps.")
        sp = s.get("delivery_phase_split", {})
        if sp and max(sp.values() or [0]) > 0:
            line.append(
                f"Late-training throughput splits early/mid/late = "
                f"{sp.get('early', 0):.1f}/{sp.get('mid', 0):.1f}/"
                f"{sp.get('late', 0):.1f} deliveries per episode third.")
        parts.append("\n\n".join(line) + "\n")
    if len(runs) > 1:
        ranked = sorted(runs, key=lambda s: -s.get("final_return", 0.0))
        order = " > ".join(f"{s['algorithm']} ({s.get('final_return', 0):.1f})"
                           for s in ranked)
        parts.append(f"**Head to head:** {order}.\n")
    parts.append("*Generated without an LLM (deterministic fallback) — every "
                 "number above is measured from results.csv.*")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# orchestration + cache
# ---------------------------------------------------------------------------
BACKENDS = ("claude", "ollama", "template")


def _cache_key(facts: dict, audience: str, backend: str) -> str:
    blob = json.dumps([facts, audience, backend], sort_keys=True).encode()
    return hashlib.sha1(blob).hexdigest()[:16]


def cached_narration(runs: list[RunData],
                     audience: str = "general") -> tuple[str, str] | None:
    """Return (markdown, backend) if any backend's narration is already cached
    for exactly this data — without generating anything (cheap, rerun-safe)."""
    facts = build_facts(runs)
    for b in BACKENDS:
        cpath = os.path.join(CACHE_DIR, f"{_cache_key(facts, audience, b)}_{b}.md")
        if os.path.isfile(cpath):
            with open(cpath, encoding="utf-8") as f:
                return f.read(), f"{b} (cached)"
    return None


def backend_status() -> dict[str, str]:
    """Quick availability probe for the UI badges (no generation)."""
    status = {}
    status["claude"] = "installed" if _find_claude_exe() else "not found"
    ok = False
    for host in _ollama_hosts():
        try:
            with urllib.request.urlopen(f"{host}/api/version", timeout=1):
                ok = True
                break
        except OSError:
            continue
    if not ok and _find_ollama_exe():
        status["ollama"] = "CLI found (service unprobed)"
    else:
        status["ollama"] = "reachable" if ok else "not found"
    status["template"] = "always available"
    return status


def narrate(runs: list[RunData], audience: str = "general",
            backend: str = "auto", use_cache: bool = True) -> tuple[str, str]:
    """Produce (markdown, backend_used). `backend`: auto|claude|ollama|template."""
    facts = build_facts(runs)
    order = BACKENDS if backend == "auto" else (backend,)

    for b in order:
        key = _cache_key(facts, audience, b)
        cpath = os.path.join(CACHE_DIR, f"{key}_{b}.md")
        if use_cache and os.path.isfile(cpath):
            with open(cpath, encoding="utf-8") as f:
                return f.read(), f"{b} (cached)"
        if b == "template":
            text = narrate_template(facts)
        else:
            prompt = build_prompt(facts, audience)
            text = narrate_claude(prompt) if b == "claude" else narrate_ollama(prompt)
        if text:
            os.makedirs(CACHE_DIR, exist_ok=True)
            with open(cpath, "w", encoding="utf-8") as f:
                f.write(text)
            return text, b
    return narrate_template(facts), "template"  # unreachable, but total
