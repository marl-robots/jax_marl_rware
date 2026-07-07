"""In-browser animated replay player (self-contained HTML/JS canvas).

Renders the JSON payloads produced by arena/replay_data.py as a smooth,
real-time-looking warehouse animation: agents glide between cells, requested
shelves glow, deliveries flash, and a HUD tracks return/deliveries while a
scrub bar + speed control let the presenter drive it. When several snapshots
of the same run are passed (recorded across training), a TRAINING PROGRESS
slider scrubs the policy itself — same fixed seed, so every visible difference
is learning — and an EVOLUTION TOUR auto-plays from untrained chaos to the
final policy.

Everything (fonts, styling, logic) is inlined: no network access needed, so it
works in a live demo without internet. Embed with:

    import streamlit.components.v1 as components
    components.html(player_html(snapshots), height=player_height(snapshots))
"""

from __future__ import annotations

import json

# Distinct per-agent colours (dark-theme friendly, colourblind-aware-ish)
AGENT_COLORS = ["#ffb703", "#ff5d8f", "#4cc9f0", "#9ef01a",
                "#e0aaff", "#f77f00", "#80ffdb", "#ff9e00"]

_CSS = """
:root {
  --bg: #0a0e14; --panel: #10151f; --edge: #1d2737;
  --txt: #c9d6e8; --dim: #5d7290; --accent: #00e5cc; --warn: #ffb703;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: transparent; font-family: Consolas, 'Cascadia Mono', 'DejaVu Sans Mono', monospace; }
.wrap { background: var(--bg); border: 1px solid var(--edge); border-radius: 12px;
        padding: 12px 14px; color: var(--txt); }
.hud { display: flex; justify-content: space-between; align-items: baseline;
       flex-wrap: wrap; gap: 6px 16px; margin-bottom: 8px; }
.hud .title { color: var(--accent); font-size: 13px; letter-spacing: 2px;
              text-transform: uppercase; }
.hud .title .algo { color: var(--txt); }
.kpis { display: flex; gap: 18px; }
.kpi { text-align: right; }
.kpi .v { color: var(--accent); font-size: 20px; font-weight: bold;
          font-variant-numeric: tabular-nums; }
.kpi .v.warn { color: var(--warn); }
.kpi .l { color: var(--dim); font-size: 9px; letter-spacing: 1px;
          text-transform: uppercase; }
.stage { display: flex; gap: 12px; align-items: stretch; flex-wrap: wrap; }
canvas.world { background: #0c111b; border: 1px solid var(--edge);
               border-radius: 8px; flex: 0 0 auto; }
.side { flex: 1 1 auto; display: flex; flex-direction: column; gap: 8px; min-width: 170px; }
.side .panel { background: var(--panel); border: 1px solid var(--edge);
               border-radius: 8px; padding: 8px 10px; }
.side .panel h4 { color: var(--dim); font-size: 9px; letter-spacing: 1.5px;
                  text-transform: uppercase; margin-bottom: 6px; }
.agent-row { display: flex; align-items: center; gap: 7px; font-size: 11px;
             padding: 2px 0; font-variant-numeric: tabular-nums; }
.agent-dot { width: 9px; height: 9px; border-radius: 50%; flex: 0 0 auto; }
.agent-row .st { margin-left: auto; color: var(--dim); }
canvas.spark { width: 100%; height: 56px; display: block; }
.feed { font-size: 10px; line-height: 1.65; height: 110px; overflow: hidden;
        color: var(--dim); }
.feed .t { color: var(--accent); }
.controls { display: flex; align-items: center; gap: 10px; margin-top: 10px; }
button.ctl { background: var(--panel); color: var(--accent); border: 1px solid var(--edge);
             border-radius: 6px; padding: 5px 13px; font: inherit; font-size: 12px;
             cursor: pointer; letter-spacing: 1px; }
button.ctl:hover { border-color: var(--accent); }
button.ctl.tour { color: var(--warn); }
select.ctl { background: var(--panel); color: var(--txt); border: 1px solid var(--edge);
             border-radius: 6px; padding: 4px 6px; font: inherit; font-size: 12px; }
input[type=range] { -webkit-appearance: none; appearance: none; flex: 1;
                    height: 4px; background: var(--edge); border-radius: 2px; outline: none; }
input[type=range]::-webkit-slider-thumb { -webkit-appearance: none; appearance: none;
  width: 13px; height: 13px; border-radius: 50%; background: var(--accent);
  cursor: pointer; box-shadow: 0 0 8px var(--accent); }
.stepro { color: var(--dim); font-size: 11px; min-width: 86px; text-align: right;
          font-variant-numeric: tabular-nums; }
.evo { display: flex; align-items: center; gap: 10px; margin-top: 8px;
       padding-top: 8px; border-top: 1px dashed var(--edge); }
.evo .lbl { color: var(--dim); font-size: 9px; letter-spacing: 1.5px;
            text-transform: uppercase; white-space: nowrap; }
.evo .val { color: var(--warn); font-size: 11px; min-width: 120px; text-align: right;
            font-variant-numeric: tabular-nums; }
.live { display: inline-block; width: 7px; height: 7px; border-radius: 50%;
        background: var(--accent); margin-right: 6px;
        animation: blink 1.2s ease-in-out infinite; }
@keyframes blink { 50% { opacity: 0.15; } }
"""

_JS = r"""
const S = window.__SNAPSHOTS__;          // list of payloads, sorted by update
const COLORS = window.__AGENT_COLORS__;
const AUTOPLAY = window.__AUTOPLAY__;

let si = S.length - 1;                    // current snapshot (default: final policy)
let P = S[si];
let t = 0;                                // playback position, float steps
let playing = AUTOPLAY;
let speed = 4;
let touring = false;
let lastTs = null;
const BASE_SPS = 6;                       // steps/second at 1x

// ---------- layout ----------
const W = P.meta.W, H = P.meta.H;
const CELL = Math.max(22, Math.min(46, Math.floor(520 / Math.max(W, H))));
const PAD = 14;
const cw = W * CELL + PAD * 2, ch = H * CELL + PAD * 2;
const cv = document.getElementById('world');
const dpr = window.devicePixelRatio || 1;
cv.width = cw * dpr; cv.height = ch * dpr;
cv.style.width = cw + 'px'; cv.style.height = ch + 'px';
const cx = cv.getContext('2d');
cx.scale(dpr, dpr);

const sparkCv = document.getElementById('spark');
const sctx = sparkCv.getContext('2d');

const el = id => document.getElementById(id);
const fmt = n => n.toLocaleString('en-US');

// ---------- helpers ----------
const px = x => PAD + x * CELL + CELL / 2;     // cell centre
const py = y => PAD + y * CELL + CELL / 2;
const lerp = (a, b, u) => a + (b - a) * u;

function interpAgent(i, tf) {
  const t0 = Math.min(Math.floor(tf), P.T), t1 = Math.min(t0 + 1, P.T);
  const u = tf - Math.floor(tf);
  return {
    x: lerp(P.agents.x[t0][i], P.agents.x[t1][i], u),
    y: lerp(P.agents.y[t0][i], P.agents.y[t1][i], u),
    dir: P.agents.dir[t1][i],
    carrying: P.agents.carrying[t1][i],
  };
}

function setSnapshot(k, keepT) {
  si = Math.max(0, Math.min(S.length - 1, k));
  P = S[si];
  if (!keepT) t = 0;
  el('evoRange').value = si;
  el('evoVal').textContent = 'update ' + fmt(P.meta.update) +
      ' · ' + (P.meta.env_steps / 1e6).toFixed(1) + 'M steps';
  feedReset();
}

// ---------- event feed ----------
let feedLines = [];
let feedCursor = 0;   // index into delivery events already announced
function feedReset() { feedLines = []; feedCursor = 0; el('feed').innerHTML = ''; }
function feedPush(html) {
  feedLines.push(html);
  if (feedLines.length > 6) feedLines.shift();
  el('feed').innerHTML = feedLines.join('<br>');
}

// ---------- drawing ----------
function draw() {
  cx.clearRect(0, 0, cw, ch);
  const ti = Math.min(Math.floor(t), P.T);

  // rack zone shading
  cx.fillStyle = 'rgba(56,70,89,0.16)';
  for (const rc of P.rack_cells)
    cx.fillRect(PAD + rc[0] * CELL, PAD + rc[1] * CELL, CELL, CELL);

  // grid
  cx.strokeStyle = 'rgba(56,70,89,0.32)';
  cx.lineWidth = 1;
  cx.beginPath();
  for (let i = 0; i <= W; i++) { cx.moveTo(PAD + i * CELL, PAD); cx.lineTo(PAD + i * CELL, PAD + H * CELL); }
  for (let j = 0; j <= H; j++) { cx.moveTo(PAD, PAD + j * CELL); cx.lineTo(PAD + W * CELL, PAD + j * CELL); }
  cx.stroke();

  // goals: glowing pads
  for (let g = 0; g < P.meta.goal_x.length; g++) {
    const gx = P.meta.goal_x[g], gy = P.meta.goal_y[g];
    cx.fillStyle = 'rgba(0,229,204,0.10)';
    cx.fillRect(PAD + gx * CELL, PAD + gy * CELL, CELL, CELL);
    cx.strokeStyle = 'rgba(0,229,204,0.65)';
    cx.lineWidth = 1.5;
    cx.strokeRect(PAD + gx * CELL + 2, PAD + gy * CELL + 2, CELL - 4, CELL - 4);
    cx.fillStyle = 'rgba(0,229,204,0.85)';
    cx.font = `bold ${Math.floor(CELL * 0.42)}px Consolas, monospace`;
    cx.textAlign = 'center'; cx.textBaseline = 'middle';
    cx.fillText('G', px(gx), py(gy) + 1);
  }

  // shelves (interpolated only when carried; idle shelves sit on the grid)
  const t0 = Math.min(Math.floor(t), P.T), t1 = Math.min(t0 + 1, P.T);
  const u = t - Math.floor(t);
  const sp = Math.max(3, Math.floor(CELL * 0.12));
  for (let s = 0; s < P.shelves.x[0].length; s++) {
    const sx = lerp(P.shelves.x[t0][s], P.shelves.x[t1][s], u);
    const sy = lerp(P.shelves.y[t0][s], P.shelves.y[t1][s], u);
    const req = P.shelves.requested[ti][s];
    const X = PAD + sx * CELL, Y = PAD + sy * CELL;
    if (req) {
      cx.shadowColor = '#00e5cc'; cx.shadowBlur = 10;
      cx.fillStyle = 'rgba(0,229,204,0.78)';
    } else {
      cx.shadowBlur = 0;
      cx.fillStyle = 'rgba(94,118,153,0.55)';
    }
    cx.beginPath();
    cx.roundRect(X + sp, Y + sp, CELL - 2 * sp, CELL - 2 * sp, 3);
    cx.fill();
    cx.shadowBlur = 0;
  }

  // motion trails (last 14 frames per agent)
  const N = P.agents.x[0].length;
  for (let i = 0; i < N; i++) {
    cx.strokeStyle = COLORS[i % COLORS.length];
    for (let k = 13; k >= 1; k--) {
      const a = Math.floor(t) - k, b = a + 1;
      if (a < 0) continue;
      cx.globalAlpha = 0.22 * (1 - k / 14);
      cx.lineWidth = 2;
      cx.beginPath();
      cx.moveTo(px(P.agents.x[a][i]), py(P.agents.y[a][i]));
      cx.lineTo(px(P.agents.x[b][i]), py(P.agents.y[b][i]));
      cx.stroke();
    }
  }
  cx.globalAlpha = 1;

  // agents
  const R = CELL * 0.32;
  for (let i = 0; i < N; i++) {
    const a = interpAgent(i, t);
    const X = px(a.x), Y = py(a.y);
    const col = COLORS[i % COLORS.length];
    if (a.carrying) {           // glow ring when hauling a shelf
      cx.shadowColor = col; cx.shadowBlur = 14;
      cx.strokeStyle = col; cx.lineWidth = 2;
      cx.beginPath(); cx.arc(X, Y, R + 3.5, 0, Math.PI * 2); cx.stroke();
    }
    cx.shadowColor = col; cx.shadowBlur = a.carrying ? 12 : 6;
    cx.fillStyle = col;
    cx.beginPath(); cx.arc(X, Y, R, 0, Math.PI * 2); cx.fill();
    cx.shadowBlur = 0;
    // direction tick (UP,DOWN,LEFT,RIGHT = 0,1,2,3; screen y grows downward)
    const dx = a.dir === 3 ? 1 : a.dir === 2 ? -1 : 0;
    const dy = a.dir === 1 ? 1 : a.dir === 0 ? -1 : 0;
    cx.strokeStyle = '#0a0e14'; cx.lineWidth = 2.5;
    cx.beginPath(); cx.moveTo(X, Y); cx.lineTo(X + dx * R, Y + dy * R); cx.stroke();
    cx.fillStyle = '#0a0e14';
    cx.font = `bold ${Math.floor(CELL * 0.30)}px Consolas, monospace`;
    cx.textAlign = 'center'; cx.textBaseline = 'middle';
    cx.fillText(String(i), X, Y + 0.5);
  }

  // delivery flashes: expanding ring at the agent's cell, 2 steps long
  for (const [et, ea] of P.events.deliveries) {
    const age = t - (et + 1);
    if (age >= 0 && age < 2) {
      const a = interpAgent(ea, et + 1);
      const prog = age / 2;
      cx.strokeStyle = `rgba(0,229,204,${0.9 * (1 - prog)})`;
      cx.lineWidth = 3 * (1 - prog) + 1;
      cx.beginPath();
      cx.arc(px(a.x), py(a.y), R + prog * CELL * 1.2, 0, Math.PI * 2);
      cx.stroke();
    }
  }
}

// ---------- sparkline (cumulative deliveries) ----------
function drawSpark() {
  const w = sparkCv.clientWidth, h = sparkCv.clientHeight;
  if (sparkCv.width !== w * dpr) {
    sparkCv.width = w * dpr; sparkCv.height = h * dpr;
  }
  sctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  sctx.clearRect(0, 0, w, h);
  const ser = P.series.deliveries_cum;
  const maxv = Math.max(1, ser[ser.length - 1]);
  const ti = Math.min(Math.floor(t), P.T - 1);
  sctx.strokeStyle = 'rgba(0,229,204,0.25)';
  sctx.lineWidth = 1;
  sctx.beginPath();
  for (let k = 0; k < ser.length; k++) {
    const X = (k / (ser.length - 1)) * (w - 4) + 2;
    const Y = h - 4 - (ser[k] / maxv) * (h - 10);
    k ? sctx.lineTo(X, Y) : sctx.moveTo(X, Y);
  }
  sctx.stroke();
  sctx.strokeStyle = '#00e5cc';
  sctx.lineWidth = 1.8;
  sctx.beginPath();
  for (let k = 0; k <= ti; k++) {
    const X = (k / (ser.length - 1)) * (w - 4) + 2;
    const Y = h - 4 - (ser[k] / maxv) * (h - 10);
    k ? sctx.lineTo(X, Y) : sctx.moveTo(X, Y);
  }
  sctx.stroke();
  const X = (ti / (ser.length - 1)) * (w - 4) + 2;
  const Y = h - 4 - (ser[ti] / maxv) * (h - 10);
  sctx.fillStyle = '#00e5cc';
  sctx.shadowColor = '#00e5cc'; sctx.shadowBlur = 6;
  sctx.beginPath(); sctx.arc(X, Y, 2.6, 0, Math.PI * 2); sctx.fill();
  sctx.shadowBlur = 0;
}

// ---------- HUD ----------
function updateHud() {
  const ti = Math.min(Math.floor(t), P.T - 1);
  el('kpiStep').textContent = Math.floor(Math.min(t, P.T)) + '/' + P.T;
  el('kpiRet').textContent = P.series.reward_cum[ti].toFixed(1);
  el('kpiDel').textContent = P.series.deliveries_cum[ti];
  el('stepro').textContent = speed + 'x · step ' + Math.floor(Math.min(t, P.T));
  el('scrub').value = t;

  const N = P.agents.x[0].length;
  for (let i = 0; i < N; i++) {
    const a = interpAgent(i, t);
    el('ag' + i).textContent = a.carrying ? 'HAULING' : 'ROAMING';
    el('ag' + i).style.color = a.carrying ? '#ffb703' : '';
  }

  // announce deliveries in the feed
  const evs = P.events.deliveries;
  while (feedCursor < evs.length && evs[feedCursor][0] + 1 <= t) {
    const [et, ea] = evs[feedCursor++];
    feedPush(`<span class="t">t=${String(et + 1).padStart(3)}</span> ` +
             `unit-${ea} delivered <span style="color:${COLORS[ea % COLORS.length]}">▮</span>` +
             ` · total ${P.series.deliveries_cum[Math.min(et, P.T - 1)]}`);
  }
}

// ---------- main loop ----------
function frame(ts) {
  if (lastTs === null) lastTs = ts;
  const dt = Math.min(0.1, (ts - lastTs) / 1000);
  lastTs = ts;
  if (playing) {
    t += dt * BASE_SPS * speed;
    if (t >= P.T) {
      if (touring && si < S.length - 1) { setSnapshot(si + 1); }
      else if (touring) { touring = false; playing = false; t = P.T; el('tourBtn').textContent = 'EVOLUTION TOUR'; }
      else { t = 0; feedReset(); }   // loop single replay
    }
  }
  draw(); drawSpark(); updateHud();
  requestAnimationFrame(frame);
}

// ---------- controls ----------
el('playBtn').onclick = () => {
  playing = !playing;
  el('playBtn').textContent = playing ? '❚❚' : '▶';
};
el('speedSel').onchange = e => { speed = Number(e.target.value); };
el('scrub').oninput = e => {
  t = Number(e.target.value);
  feedReset();
  // fast-forward the feed cursor silently
  while (feedCursor < P.events.deliveries.length &&
         P.events.deliveries[feedCursor][0] + 1 <= t) feedCursor++;
};
el('restartBtn').onclick = () => { t = 0; feedReset(); };
if (S.length > 1) {
  el('evoRange').oninput = e => { setSnapshot(Number(e.target.value)); };
  el('tourBtn').onclick = () => {
    touring = !touring;
    if (touring) { setSnapshot(0); playing = true; speed = Math.max(speed, 8);
                   el('speedSel').value = String(speed);
                   el('playBtn').textContent = '❚❚';
                   el('tourBtn').textContent = '■ STOP TOUR'; }
    else el('tourBtn').textContent = 'EVOLUTION TOUR';
  };
}

setSnapshot(si, true);
el('scrub').max = P.T;
el('playBtn').textContent = playing ? '❚❚' : '▶';
requestAnimationFrame(frame);
"""


def player_html(snapshots: list[dict], *, autoplay: bool = True) -> str:
    """Build the full self-contained player HTML for one run's snapshots.

    `snapshots`: payloads from arena.replay_data.replay_payload, sorted by
    update (all from the same run/env). The last one is shown first (final
    policy); the evolution slider/tour appears when there is more than one.
    """
    snapshots = sorted(snapshots, key=lambda p: p["meta"]["update"])
    last = snapshots[-1]["meta"]
    n_agents = last["n_agents"]

    agent_rows = "\n".join(
        f'<div class="agent-row">'
        f'<span class="agent-dot" style="background:{AGENT_COLORS[i % len(AGENT_COLORS)]}"></span>'
        f'unit-{i} <span class="st" id="ag{i}">—</span></div>'
        for i in range(n_agents))

    evo = ""
    if len(snapshots) > 1:
        evo = f"""
  <div class="evo">
    <span class="lbl">Training progress</span>
    <input type="range" id="evoRange" min="0" max="{len(snapshots) - 1}"
           value="{len(snapshots) - 1}" step="1">
    <span class="val" id="evoVal"></span>
    <button class="ctl tour" id="tourBtn">EVOLUTION TOUR</button>
  </div>"""
    else:
        # keep the ids the JS references (display: none)
        evo = """
  <div style="display:none"><input id="evoRange"><span id="evoVal"></span>
  <button id="tourBtn"></button></div>"""

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>{_CSS}</style></head>
<body>
<div class="wrap">
  <div class="hud">
    <div>
      <div class="title"><span class="live"></span>
        <span class="algo">{last['algo'].upper()}</span> · rware-{last['size']}-{n_agents}ag
        · TELEMETRY REPLAY</div>
    </div>
    <div class="kpis">
      <div class="kpi"><div class="v" id="kpiStep">0/0</div><div class="l">step</div></div>
      <div class="kpi"><div class="v" id="kpiRet">0.0</div><div class="l">team return</div></div>
      <div class="kpi"><div class="v warn" id="kpiDel">0</div><div class="l">deliveries</div></div>
    </div>
  </div>
  <div class="stage">
    <canvas class="world" id="world"></canvas>
    <div class="side">
      <div class="panel"><h4>Agents</h4>{agent_rows}</div>
      <div class="panel"><h4>Deliveries (cumulative)</h4>
        <canvas class="spark" id="spark"></canvas></div>
      <div class="panel"><h4>Event feed</h4><div class="feed" id="feed"></div></div>
    </div>
  </div>
  <div class="controls">
    <button class="ctl" id="playBtn">▶</button>
    <button class="ctl" id="restartBtn">⟲</button>
    <select class="ctl" id="speedSel">
      <option>1</option><option>2</option><option selected>4</option>
      <option>8</option><option>16</option><option>32</option>
    </select>
    <input type="range" id="scrub" min="0" max="500" value="0" step="0.5">
    <span class="stepro" id="stepro"></span>
  </div>
  {evo}
</div>
<script>
window.__SNAPSHOTS__ = {json.dumps(snapshots)};
window.__AGENT_COLORS__ = {json.dumps(AGENT_COLORS)};
window.__AUTOPLAY__ = {json.dumps(autoplay)};
{_JS}
</script>
</body></html>"""


def player_height(snapshots: list[dict]) -> int:
    """Pixel height to give the Streamlit component (canvas + chrome)."""
    meta = snapshots[-1]["meta"]
    cell = max(22, min(46, 520 // max(meta["W"], meta["H"])))
    canvas_h = meta["H"] * cell + 28
    side_h = 110 + 56 + 24 * meta["n_agents"] + 120
    chrome = 60 + 46 + (52 if len(snapshots) > 1 else 0) + 28
    return max(canvas_h, side_h) + chrome
