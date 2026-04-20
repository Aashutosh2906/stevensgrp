"""
DVNC.AI Gradio Application

Tabs:
  1. Discovery Engine     — full 6-agent pipeline with visible routing
  2. Head-to-Head         — auto-populated when Discovery Engine runs
  3. Connectome Explorer  — table + premium D3 force-directed graph
  4. Database Inspector   — raw stats, node/synapse browser
  5. Add Papers           — ingest new papers into the connectome
"""

from __future__ import annotations
import json
import os
import re
import sqlite3

import gradio as gr

from ..db.neurographdb import NeuroGraphDB, LMM_LABELS
from ..routing.davinci_router import DaVinciRouter
from ..agents.orchestrator import DVNCOrchestrator


# ── Helpers ────────────────────────────────────────────────────────────────


def _format_route_panel(route_result) -> str:
    if route_result is None:
        return "No route computed."
    lines = []
    lines.append("╔══════════════════════════════════════════════════════╗")
    lines.append("║          DA VINCI ROUTING PANEL                     ║")
    lines.append("╚══════════════════════════════════════════════════════╝")
    lines.append("")
    lines.append("── PRIMARY ROUTE ──────────────────────────────────────")
    max_route_score = max((s.score for s in route_result.primary_route), default=1) or 1
    for i, step in enumerate(route_result.primary_route):
        cross = " ⟳ [CROSS-DOMAIN]" if step.is_cross_domain else ""
        rel = f" ─[{step.rel_from_prev}]→ " if step.rel_from_prev else ""
        norm = step.score / max_route_score
        score_bar = "█" * max(1, int(norm * 10))
        lines.append(f"  {i:2d}. {rel}{step.label:<25} score={step.score:.3f} [{score_bar}]{cross}")
        if step.lmm_tags:
            lmm_names = [LMM_LABELS.get(t, t) for t in step.lmm_tags[:2]]
            lines.append(f"       LMM: {', '.join(lmm_names)}")
    lines.append("")
    if route_result.alternative_routes:
        lines.append("── ALTERNATIVE ROUTES ─────────────────────────────────")
        for i, alt in enumerate(route_result.alternative_routes, 1):
            lines.append(f"  Alt {i}: {' → '.join(s.label for s in alt)}")
        lines.append("")
    if route_result.suppressed_hubs:
        lines.append("── SUPPRESSED HUBS (hub suppression active) ──────────")
        for hub in route_result.suppressed_hubs[:5]:
            lines.append(f"  ✕ {hub.split('::')[-1]}")
        lines.append("")
    lmm = route_result.lmm_activations
    if lmm:
        lines.append("── DA VINCI MENTAL MODEL ACTIVATIONS ─────────────────")
        for code, score in sorted(lmm.items(), key=lambda x: x[1], reverse=True)[:5]:
            name = LMM_LABELS.get(code, code)
            lines.append(f"  {code} {name:<28} {score:.2f} [{'█' * max(1, int(score * 5))}]")
        lines.append("")
    lines.append("── SPREADING ACTIVATION MAP (top 15 nodes) ───────────")
    max_score = max((s for _, s in route_result.activation_map[:15]), default=1)
    for node_id, score in route_result.activation_map[:15]:
        label = node_id.split("::")[-1][:25]
        norm = score / max_score if max_score else 0
        lines.append(f"  {label:<26} {score:.4f} [{'█' * max(1, int(norm * 20))}]")
    lines.append("")
    lines.append(f"Novelty score: {route_result.novelty_score:.3f}")
    lines.append(f"Cross-domain leaps: {route_result.cross_domain_count}")
    return "\n".join(lines)


def _format_agent_log(agent_log: list[dict]) -> str:
    lines = []
    for entry in agent_log:
        lines.append(f"╔══ {entry['agent']} ══════════════════════")
        lines.append(f"║ Role: {entry['role']}")
        lines.append("╚" + "═" * 50)
        lines.append(entry["output"])
        lines.append("")
    return "\n".join(lines)


def _set_api_key(key: str):
    key = key.strip()
    if not key:
        return
    if key.startswith("gsk_"):
        os.environ["GROQ_API_KEY"] = key
    elif key.startswith("AIza"):
        os.environ["GEMINI_API_KEY"] = key
    elif key.startswith("sk-ant"):
        os.environ["ANTHROPIC_API_KEY"] = key
    else:
        os.environ["DEEPSEEK_API_KEY"] = key


def _call_plain_llm(brief: str) -> str:
    system = (
        "You are a research scientist and innovation consultant. "
        "Given a design brief, produce a detailed innovation proposal. "
        "Include: a cross-domain insight, a specific testable hypothesis, "
        "a step-by-step experimental programme, and a commercial/IP lens. "
        "Be as specific as possible with numbers, materials, and measurements."
    )
    try:
        from ..agents.base import _call_claude
        return _call_claude(system, brief, max_tokens=1200)
    except Exception as e:
        return f"[Plain LLM error: {e}]"


def _score_output(text: str) -> dict:
    citations  = len(re.findall(r'\[S\d+\]', text))
    numbers    = len(re.findall(r'\d+\.?\d*\s*(?:kPa|MPa|GPa|S/cm|nm|um|mm|cm|mg|mL|wt%|%|Hz|C)', text))
    steps      = len(re.findall(r'(?:Step\s*\d|IF\s|THEN\s|->)', text, re.IGNORECASE))
    words      = len(text.split())
    cit_score    = min(100, citations * 14)
    spec_score   = min(100, numbers * 10 + steps * 6)
    struct_score = min(100, steps * 12 + (20 if len(text) > 400 else 0))
    concision    = 100 if 120 <= words <= 600 else max(0, 100 - abs(words - 360) // 4)
    overall = int(0.35 * cit_score + 0.30 * spec_score + 0.20 * struct_score + 0.15 * concision)
    return {"citations": cit_score, "specificity": spec_score, "structure": struct_score, "concision": concision, "overall": overall}


def _build_score_table(dvnc_scores: dict, plain_scores: dict) -> str:
    dims = [("Citations","citations"),("Specificity","specificity"),("Structure","structure"),("Concision","concision"),("OVERALL","overall")]
    lines = [f"{'DIMENSION':<14}  {'DVNC.AI':^16}  {'Plain LLM':^16}  WINNER", "─" * 62]
    for label, key in dims:
        d, p = dvnc_scores[key], plain_scores[key]
        d_bar = "█" * (d // 10) + "░" * (10 - d // 10)
        p_bar = "█" * (p // 10) + "░" * (10 - p // 10)
        winner = "<-- DVNC" if d > p else ("<-- LLM " if p > d else "  TIE  ")
        lines.append(f"{label:<14}  {d:3d} [{d_bar}]  {p:3d} [{p_bar}]  {winner}")
    return "\n".join(lines)


def _db_stats(db_path: str) -> str:
    try:
        conn = sqlite3.connect(db_path)
        cur  = conn.cursor()
        lines = ["╔══════════════════════════════════════════════╗", "║         DVNC CONNECTOME — DB STATS          ║", "╚══════════════════════════════════════════════╝", ""]
        for sql, label in [("SELECT COUNT(*) FROM nodes","Total Nodes"), ("SELECT COUNT(*) FROM synapses","Total Synapses")]:
            try:
                cur.execute(sql); lines.append(f"  {label:<30} {cur.fetchone()[0]}")
            except Exception: pass
        for sql, header in [
            ("SELECT kind, COUNT(*) FROM nodes GROUP BY kind ORDER BY COUNT(*) DESC", "Node breakdown:"),
            ("SELECT rel, COUNT(*) FROM synapses GROUP BY rel ORDER BY COUNT(*) DESC", "Synapse breakdown:"),
        ]:
            try:
                cur.execute(sql); lines.append(""); lines.append(f"  {header}")
                for row in cur.fetchall(): lines.append(f"    {row[0]:<26} {row[1]}")
            except Exception: pass
        conn.close()
        return "\n".join(lines)
    except Exception as e:
        return f"Stats error: {e}"


def _db_top_synapses(db_path: str, limit: int = 50) -> list:
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT pre, post, rel, weight, evidence_count, lmm_tags FROM synapses ORDER BY weight DESC LIMIT ?", (limit,)).fetchall()
        conn.close()
        result = []
        for r in rows:
            try: tags_str = ", ".join(json.loads(r["lmm_tags"])[:2])
            except Exception: tags_str = ""
            result.append([r["pre"].split("::")[-1], r["post"].split("::")[-1], r["rel"], round(r["weight"], 3), r["evidence_count"], tags_str])
        return result
    except Exception as e:
        return [[f"Error: {e}", "", "", "", "", ""]]


# ── D3 Graph HTML Generator ────────────────────────────────────────────────

def _build_graph_html(concept: str, neighbors: list[dict]) -> str:
    """Build a premium D3 force-directed graph for the given concept + its neighbors."""

    # Build nodes and links from neighbor data
    nodes_set = {concept}
    links_data = []
    max_weight = max((n["weight"] for n in neighbors), default=1) or 1

    for n in neighbors:
        src = n["pre"].split("::")[-1]
        tgt = n["post"].split("::")[-1]
        nodes_set.add(src)
        nodes_set.add(tgt)
        links_data.append({
            "source": src,
            "target": tgt,
            "rel":    n["rel"],
            "weight": round(n["weight"] / max_weight, 3),
            "raw":    round(n["weight"], 2),
        })

    nodes_data = []
    for nd in nodes_set:
        nodes_data.append({
            "id":      nd,
            "central": nd == concept,
        })

    nodes_json = json.dumps(nodes_data)
    links_json = json.dumps(links_data)
    concept_json = json.dumps(concept)

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
  @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;500;600&family=Crimson+Pro:ital,wght@0,300;0,400;1,300&family=JetBrains+Mono:wght@300;400&display=swap');

  * {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    background: #F7F3EC;
    font-family: 'Crimson Pro', Georgia, serif;
    overflow: hidden;
    width: 100%;
    height: 520px;
  }}

  #graph-wrap {{
    position: relative;
    width: 100%;
    height: 520px;
    background: radial-gradient(ellipse at 50% 40%, #FFFDF7 0%, #F0EAE0 55%, #E6DDD0 100%);
    border: 1px solid rgba(184,146,42,0.25);
    border-radius: 12px;
    overflow: hidden;
  }}

  /* Subtle grid lines like da Vinci notebook paper */
  #graph-wrap::before {{
    content: '';
    position: absolute;
    inset: 0;
    background-image:
      linear-gradient(rgba(184,146,42,0.04) 1px, transparent 1px),
      linear-gradient(90deg, rgba(184,146,42,0.04) 1px, transparent 1px);
    background-size: 36px 36px;
    pointer-events: none;
    border-radius: 12px;
  }}

  #graph-svg {{ width: 100%; height: 100%; }}

  /* Tooltip */
  #tooltip {{
    position: absolute;
    pointer-events: none;
    opacity: 0;
    transition: opacity 0.15s ease;
    background: rgba(28,25,20,0.92);
    border: 1px solid rgba(184,146,42,0.5);
    border-radius: 8px;
    padding: 8px 13px;
    font-family: 'Cinzel', serif;
    font-size: 11px;
    color: #E8D5A0;
    letter-spacing: 0.05em;
    max-width: 180px;
    line-height: 1.6;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    z-index: 99;
  }}
  #tooltip .tt-rel {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px;
    color: #B8922A;
    margin-top: 2px;
  }}
  #tooltip .tt-weight {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px;
    color: rgba(232,213,160,0.6);
  }}

  /* Legend */
  #legend {{
    position: absolute;
    bottom: 14px;
    right: 16px;
    display: flex;
    flex-direction: column;
    gap: 5px;
  }}
  .leg-row {{
    display: flex;
    align-items: center;
    gap: 7px;
    font-family: 'Cinzel', serif;
    font-size: 9px;
    letter-spacing: 0.08em;
    color: #4A4030;
  }}
  .leg-line {{
    width: 22px;
    height: 2px;
    border-radius: 1px;
  }}

  /* Title */
  #graph-title {{
    position: absolute;
    top: 14px;
    left: 18px;
    font-family: 'Cinzel', serif;
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.12em;
    color: #4A4030;
    text-transform: uppercase;
  }}
  #graph-title span {{
    color: #B8922A;
    font-weight: 600;
  }}

  /* Node count badge */
  #graph-meta {{
    position: absolute;
    top: 14px;
    right: 16px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 9.5px;
    color: #8B7B62;
  }}
</style>
</head>
<body>
<div id="graph-wrap">
  <div id="graph-title">Connectome · <span id="concept-label"></span></div>
  <div id="graph-meta" id="meta"></div>
  <svg id="graph-svg"></svg>
  <div id="tooltip"></div>
  <div id="legend">
    <div class="leg-row"><div class="leg-line" style="background:#B8922A"></div>EVOKES</div>
    <div class="leg-row"><div class="leg-line" style="background:#1B3A5C"></div>PRIMES</div>
    <div class="leg-row"><div class="leg-line" style="background:#6B8F71"></div>MENTIONS</div>
    <div class="leg-row"><div class="leg-line" style="background:#B5A898;height:1px"></div>CO_OCCURS</div>
  </div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.9.0/d3.min.js"></script>
<script>
const NODES_DATA = {nodes_json};
const LINKS_DATA = {links_json};
const CENTRAL    = {concept_json};

document.getElementById('concept-label').textContent = CENTRAL.toUpperCase();
document.getElementById('graph-meta').textContent =
  NODES_DATA.length + ' nodes · ' + LINKS_DATA.length + ' edges';

const relColor = r => ({{
  'EVOKES':   '#B8922A',
  'PRIMES':   '#1B3A5C',
  'MENTIONS': '#6B8F71',
  'CO_OCCURS':'#9C8E7E',
}})[ r ] || '#9C8E7E';

const relWidth = r => ({{ 'EVOKES': 2.4, 'PRIMES': 2.0, 'MENTIONS': 1.4, 'CO_OCCURS': 0.9 }})[r] || 1;
const relOpacity = r => ({{ 'EVOKES': 0.85, 'PRIMES': 0.8, 'MENTIONS': 0.65, 'CO_OCCURS': 0.45 }})[r] || 0.5;

const wrap = document.getElementById('graph-wrap');
const W = wrap.clientWidth  || 900;
const H = wrap.clientHeight || 520;

const svg = d3.select('#graph-svg')
  .attr('viewBox', `0 0 ${{W}} ${{H}}`);

// Arrow markers for directed edges
const defs = svg.append('defs');
['EVOKES','PRIMES','MENTIONS','CO_OCCURS'].forEach(rel => {{
  defs.append('marker')
    .attr('id', 'arr-' + rel)
    .attr('viewBox','0 -4 10 8')
    .attr('refX', 22).attr('refY', 0)
    .attr('markerWidth', 6).attr('markerHeight', 6)
    .attr('orient','auto')
    .append('path')
    .attr('d','M0,-4L10,0L0,4')
    .attr('fill', relColor(rel))
    .attr('opacity', 0.7);
}});

const zoom = d3.zoom().scaleExtent([0.3, 3])
  .on('zoom', e => g.attr('transform', e.transform));
svg.call(zoom);

const g = svg.append('g');

// Deep-copy nodes/links for D3 mutation
const nodes = NODES_DATA.map(d => ({{...d}}));
const links = LINKS_DATA.map(d => ({{...d}}));

const sim = d3.forceSimulation(nodes)
  .force('link', d3.forceLink(links).id(d => d.id)
    .distance(d => d.rel === 'EVOKES' ? 160 : d.rel === 'PRIMES' ? 130 : 110)
    .strength(0.4))
  .force('charge', d3.forceManyBody().strength(d => d.central ? -420 : -180))
  .force('center', d3.forceCenter(W / 2, H / 2))
  .force('collision', d3.forceCollide().radius(d => d.central ? 44 : 30));

// Glow filter
const filt = defs.append('filter').attr('id','glow').attr('x','-40%').attr('y','-40%').attr('width','180%').attr('height','180%');
filt.append('feGaussianBlur').attr('stdDeviation','4').attr('result','blur');
const feMerge = filt.append('feMerge');
feMerge.append('feMergeNode').attr('in','blur');
feMerge.append('feMergeNode').attr('in','SourceGraphic');

// Gold glow for central
const filtGold = defs.append('filter').attr('id','glow-gold').attr('x','-50%').attr('y','-50%').attr('width','200%').attr('height','200%');
filtGold.append('feGaussianBlur').attr('stdDeviation','6').attr('result','blur');
const feMerge2 = filtGold.append('feMerge');
feMerge2.append('feMergeNode').attr('in','blur');
feMerge2.append('feMergeNode').attr('in','SourceGraphic');

// Links
const link = g.append('g').selectAll('line').data(links).join('line')
  .attr('stroke', d => relColor(d.rel))
  .attr('stroke-width', d => relWidth(d.rel) * (0.4 + d.weight * 0.6))
  .attr('stroke-opacity', d => relOpacity(d.rel))
  .attr('marker-end', d => `url(#arr-${{d.rel}})`);

// Node groups
const nodeG = g.append('g').selectAll('g').data(nodes).join('g')
  .style('cursor', 'pointer')
  .call(d3.drag()
    .on('start', (e, d) => {{ if (!e.active) sim.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; }})
    .on('drag',  (e, d) => {{ d.fx = e.x; d.fy = e.y; }})
    .on('end',   (e, d) => {{ if (!e.active) sim.alphaTarget(0); d.fx = null; d.fy = null; }})
  );

// Outer ring for central node
nodeG.filter(d => d.central)
  .append('circle')
  .attr('r', 40)
  .attr('fill', 'none')
  .attr('stroke', '#B8922A')
  .attr('stroke-width', 0.8)
  .attr('stroke-dasharray', '3 3')
  .attr('opacity', 0.5);

// Main circle
nodeG.append('circle')
  .attr('r', d => d.central ? 30 : 18)
  .attr('fill', d => d.central
    ? 'rgba(184,146,42,0.18)'
    : 'rgba(28,25,20,0.06)')
  .attr('stroke', d => d.central ? '#B8922A' : '#4A4030')
  .attr('stroke-width', d => d.central ? 2 : 1)
  .attr('filter', d => d.central ? 'url(#glow-gold)' : null)
  .style('transition', 'r 0.2s');

// Label
nodeG.append('text')
  .text(d => d.id.length > 14 ? d.id.slice(0, 13) + '…' : d.id)
  .attr('text-anchor', 'middle')
  .attr('dominant-baseline', 'central')
  .attr('font-family', d => d.central ? "'Cinzel', serif" : "'JetBrains Mono', monospace")
  .attr('font-size', d => d.central ? 11 : 8.5)
  .attr('font-weight', d => d.central ? '600' : '400')
  .attr('fill', d => d.central ? '#B8922A' : '#2A2218')
  .attr('letter-spacing', d => d.central ? '0.06em' : '0')
  .style('pointer-events', 'none')
  .style('user-select', 'none');

// Tooltip
const tt = document.getElementById('tooltip');

nodeG
  .on('mouseover', (e, d) => {{
    const connected = new Set([d.id]);
    links.forEach(l => {{
      const s = typeof l.source === 'object' ? l.source.id : l.source;
      const t = typeof l.target === 'object' ? l.target.id : l.target;
      if (s === d.id) connected.add(t);
      if (t === d.id) connected.add(s);
    }});
    nodeG.selectAll('circle:last-of-type').attr('opacity', n => connected.has(n.id) ? 1 : 0.2);
    link.attr('stroke-opacity', l => {{
      const s = typeof l.source === 'object' ? l.source.id : l.source;
      const t = typeof l.target === 'object' ? l.target.id : l.target;
      return (s === d.id || t === d.id) ? relOpacity(l.rel) : 0.05;
    }});

    const rels = links
      .filter(l => {{
        const s = typeof l.source === 'object' ? l.source.id : l.source;
        const t = typeof l.target === 'object' ? l.target.id : l.target;
        return s === d.id || t === d.id;
      }})
      .map(l => l.rel);
    const uniqueRels = [...new Set(rels)].slice(0, 3).join(', ');

    tt.innerHTML = `
      <div style="font-weight:600">${{d.id}}</div>
      ${{uniqueRels ? `<div class="tt-rel">${{uniqueRels}}</div>` : ''}}
      <div class="tt-weight">${{connected.size - 1}} connections</div>
    `;
    tt.style.opacity = '1';
  }})
  .on('mousemove', e => {{
    const box = wrap.getBoundingClientRect();
    let x = e.clientX - box.left + 14;
    let y = e.clientY - box.top  - 40;
    if (x + 190 > W) x = x - 200;
    if (y < 0) y = 10;
    tt.style.left = x + 'px';
    tt.style.top  = y + 'px';
  }})
  .on('mouseout', () => {{
    nodeG.selectAll('circle:last-of-type').attr('opacity', 1);
    link.attr('stroke-opacity', d => relOpacity(d.rel));
    tt.style.opacity = '0';
  }});

// Click to isolate
nodeG.on('click', (e, d) => {{
  e.stopPropagation();
  const connected = new Set([d.id]);
  links.forEach(l => {{
    const s = typeof l.source === 'object' ? l.source.id : l.source;
    const t = typeof l.target === 'object' ? l.target.id : l.target;
    if (s === d.id) connected.add(t);
    if (t === d.id) connected.add(s);
  }});
  nodeG.selectAll('circle:last-of-type').attr('opacity', n => connected.has(n.id) ? 1 : 0.12);
  link.attr('stroke-opacity', l => {{
    const s = typeof l.source === 'object' ? l.source.id : l.source;
    const t = typeof l.target === 'object' ? l.target.id : l.target;
    return (s === d.id || t === d.id) ? relOpacity(l.rel) : 0.03;
  }});
}});

svg.on('click', () => {{
  nodeG.selectAll('circle:last-of-type').attr('opacity', 1);
  link.attr('stroke-opacity', d => relOpacity(d.rel));
}});

sim.on('tick', () => {{
  link
    .attr('x1', d => d.source.x).attr('y1', d => d.source.y)
    .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
  nodeG.attr('transform', d => `translate(${{d.x}},${{d.y)}}`);
}});

// Fade in
svg.style('opacity', 0).transition().duration(600).style('opacity', 1);
</script>
</body>
</html>"""


# ── Main App Factory ───────────────────────────────────────────────────────


def make_app(db_path: str) -> gr.Blocks:
    db           = NeuroGraphDB(db_path)
    router       = DaVinciRouter(db)
    orchestrator = DVNCOrchestrator(db=db)

    with gr.Blocks(
        title="DVNC.AI — Brain-Inspired Design Discovery",
        theme=gr.themes.Soft(primary_hue="violet"),
    ) as app:

        gr.HTML("""
        <div style="text-align:center; padding:20px;
                    background: linear-gradient(135deg, #1a0a2e, #16213e);
                    border-radius:12px; margin-bottom:20px;">
            <h1 style="color:#c084fc; font-size:2.2em; margin:0;">DVNC.AI</h1>
            <p style="color:#94a3b8; margin:8px 0 0;">Brain-Inspired Polymathic Design Discovery System</p>
            <p style="color:#64748b; font-size:0.85em;">
                Da Vinci Routing · 6-Agent Debate · Hebbian Connectome · Visible Reasoning
            </p>
        </div>
        """)

        with gr.Tabs():

            # ══════════════════════════════════════════════════════════════
            # TAB 1 — Discovery Engine
            # ══════════════════════════════════════════════════════════════
            with gr.TabItem("Discovery Engine"):
                gr.Markdown("""
                Enter a design challenge. The system routes through the DVNC Connectome,
                runs 6 specialised AI agents in debate, and produces an evidence-anchored
                Innovation Card with full provenance.
                *After running, switch to the Head-to-Head tab — it will already be populated.*
                """)
                with gr.Row():
                    with gr.Column(scale=2):
                        brief_input = gr.Textbox(
                            label="Design Brief",
                            placeholder="e.g. Design a lightweight load-bearing structure inspired by biological tissue architecture...",
                            lines=4,
                        )
                    with gr.Column(scale=1):
                        api_key_input = gr.Textbox(
                            label="API Key (Groq / Gemini / DeepSeek / Anthropic)",
                            placeholder="gsk_... or AIza... or sk-...",
                            type="password",
                        )
                        steps_slider  = gr.Slider(2, 6, value=4, step=1, label="Routing Steps")
                        fanout_slider = gr.Slider(5, 40, value=20, step=5, label="Fanout per Step")
                        run_btn = gr.Button("Run DVNC Discovery", variant="primary", size="lg")

                route_panel     = gr.Textbox(label="Da Vinci Routing Panel", lines=25, max_lines=35, interactive=False)
                final_card      = gr.Textbox(label="Final Innovation Card", lines=30, max_lines=50, interactive=False)
                with gr.Accordion("Agent Debate Log", open=False):
                    agent_log_out = gr.Textbox(lines=40, max_lines=80, interactive=False)
                overall_score_out = gr.Number(label="Overall Innovation Score (0-100)")
                status_out        = gr.Textbox(label="Status", lines=2, interactive=False)

            # ══════════════════════════════════════════════════════════════
            # TAB 2 — Head-to-Head
            # ══════════════════════════════════════════════════════════════
            with gr.TabItem("Head-to-Head"):
                gr.HTML("""
                <div style="padding:10px 0 16px;">
                    <h3 style="margin:0 0 6px;">DVNC.AI vs Plain LLM</h3>
                    <p style="color:#64748b; margin:0; font-size:0.9em;">
                        Auto-populated when you run the Discovery Engine.
                        Same brief sent directly to the LLM — no routing, no knowledge graph, no agents.
                    </p>
                </div>
                """)
                h2h_brief_display = gr.Textbox(label="Brief being compared", lines=2, interactive=False,
                    placeholder="Run a query in Discovery Engine — results appear here automatically.")
                h2h_scores = gr.Textbox(label="Score Comparison", lines=9, interactive=False)
                gr.Markdown("---")
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### DVNC.AI\n*6-Agent Debate · Evidence-grounded*")
                        dvnc_out           = gr.Textbox(lines=30, max_lines=60, interactive=False, show_label=False)
                        dvnc_score_display = gr.Number(label="DVNC.AI Score (0-100)")
                    with gr.Column(scale=1):
                        gr.Markdown("### Plain LLM\n*No graph · No citations · No routing*")
                        plain_out           = gr.Textbox(lines=30, max_lines=60, interactive=False, show_label=False)
                        plain_score_display = gr.Number(label="Plain LLM Score (0-100)")

            # ══════════════════════════════════════════════════════════════
            # TAB 3 — Connectome Explorer
            # ══════════════════════════════════════════════════════════════
            with gr.TabItem("Connectome Explorer"):
                gr.Markdown("Explore concepts in the knowledge graph. Enter a concept to see its neighbours and the network diagram.")

                with gr.Row():
                    concept_input = gr.Textbox(
                        label="Concept",
                        value="bone",
                        placeholder="e.g. auxetic, raman, scaffold, cardiomyocyte",
                    )
                    limit_slider = gr.Slider(5, 40, value=20, step=5, label="Max neighbours")
                    explore_btn  = gr.Button("Explore", variant="primary")

                # ── Premium D3 graph ──────────────────────────────────────
                graph_out = gr.HTML(
                    value="""
                    <div style="height:520px; background:radial-gradient(ellipse at 50% 40%,#FFFDF7,#F0EAE0 55%,#E6DDD0);
                                border:1px solid rgba(184,146,42,0.2); border-radius:12px;
                                display:flex; align-items:center; justify-content:center;">
                        <div style="text-align:center; font-family:'Cinzel',serif; color:#8B7B62;">
                            <div style="font-size:13px; letter-spacing:0.12em; text-transform:uppercase; margin-bottom:8px;">
                                Knowledge Graph
                            </div>
                            <div style="font-size:11px; opacity:0.6;">
                                Enter a concept and click Explore to visualise connections
                            </div>
                        </div>
                    </div>
                    """,
                    label="Network Diagram",
                )

                gr.Markdown("---")
                gr.Markdown("#### Neighbour Table")
                neighbours_out = gr.Dataframe(
                    headers=["From", "To", "Relation", "Weight", "Evidence Count"],
                    label="Top Neighbours",
                    interactive=False,
                )

                gr.Markdown("---")
                gr.Markdown("#### Spreading Activation")
                with gr.Row():
                    prop_steps  = gr.Slider(1, 6, value=3, step=1, label="Propagation Steps")
                    prop_fanout = gr.Slider(5, 30, value=15, step=5, label="Fanout")
                    prop_btn    = gr.Button("Propagate Activation", variant="secondary")

                propagation_out = gr.Dataframe(
                    headers=["Node ID", "Label", "Activation Score"],
                    label="Spreading Activation Result",
                    interactive=False,
                )

                # ── Handlers ──────────────────────────────────────────────
                def explore_concept(concept, limit):
                    concept = concept.strip().lower()
                    node_id   = f"concept::{concept}"
                    neighbors = db.top_neighbors(node_id, limit=int(limit))

                    # Table rows
                    rows = [
                        [
                            n["pre"].split("::")[-1],
                            n["post"].split("::")[-1],
                            n["rel"],
                            round(n["weight"], 3),
                            n["evidence_count"],
                        ]
                        for n in neighbors
                    ]

                    # Graph HTML
                    if neighbors:
                        graph_html = _build_graph_html(concept, neighbors)
                    else:
                        graph_html = f"""
                        <div style="height:520px; background:#F0EAE0; border:1px solid rgba(184,146,42,0.2);
                                    border-radius:12px; display:flex; align-items:center; justify-content:center;">
                            <div style="font-family:'Cinzel',serif; color:#8B7B62; text-align:center;">
                                <div style="font-size:13px; letter-spacing:0.1em;">No connections found for "{concept}"</div>
                                <div style="font-size:11px; margin-top:6px; opacity:0.6;">Try a different concept</div>
                            </div>
                        </div>
                        """

                    return graph_html, rows

                def propagate_concept(concept, steps, fanout):
                    node_id = f"concept::{concept.strip().lower()}"
                    results = db.propagate(node_id, steps=int(steps), fanout=int(fanout))
                    return [
                        [r[0], r[0].split("::")[-1], round(r[1], 5)]
                        for r in results[:30]
                    ]

                explore_btn.click(
                    fn=explore_concept,
                    inputs=[concept_input, limit_slider],
                    outputs=[graph_out, neighbours_out],
                )
                prop_btn.click(
                    fn=propagate_concept,
                    inputs=[concept_input, prop_steps, prop_fanout],
                    outputs=[propagation_out],
                )

            # ══════════════════════════════════════════════════════════════
            # TAB 4 — Database Inspector
            # ══════════════════════════════════════════════════════════════
            with gr.TabItem("Database Inspector"):
                gr.Markdown("### Connectome Statistics")
                stats_btn = gr.Button("Refresh Stats", variant="secondary")
                stats_out = gr.Textbox(label="Database Statistics", lines=20, interactive=False)

                with gr.Row():
                    node_kind       = gr.Dropdown(choices=["all","concept","document","domain"], value="all", label="Node Kind")
                    node_search     = gr.Textbox(label="Search nodes (label contains)", placeholder="e.g. cardiac")
                    node_search_btn = gr.Button("Search Nodes", variant="secondary")
                nodes_out = gr.Dataframe(headers=["ID","Kind","Label","Domain"], label="Nodes", interactive=False)

                top_syn_btn = gr.Button("Show Top 50 Synapses by Weight", variant="secondary")
                syn_out     = gr.Dataframe(headers=["From","To","Relation","Weight","Evidence","LMM Tags"], label="Top Synapses", interactive=False)

                def get_stats(): return _db_stats(db_path)

                def search_nodes(kind, search_term):
                    nodes = db.search_nodes(query=search_term or "", kind=None if kind == "all" else kind, limit=100)
                    rows = []
                    for n in nodes:
                        props = n.get("props", {})
                        if isinstance(props, str):
                            try: props = json.loads(props)
                            except Exception: props = {}
                        rows.append([n["id"], n["kind"], n["label"], props.get("domain", props.get("domain_hint", ""))])
                    return rows

                def get_top_synapses(): return _db_top_synapses(db_path, limit=50)

                stats_btn.click(fn=get_stats, outputs=[stats_out])
                node_search_btn.click(fn=search_nodes, inputs=[node_kind, node_search], outputs=[nodes_out])
                top_syn_btn.click(fn=get_top_synapses, outputs=[syn_out])

            # ══════════════════════════════════════════════════════════════
            # TAB 5 — Add Papers
            # ══════════════════════════════════════════════════════════════
            with gr.TabItem("Add Papers"):
                gr.Markdown("""
                ### Add Papers to the Connectome
                1. **Fetch by DOI** · 2. **Search by Topic** · 3. **Paste Directly**
                """)
                gr.Markdown("#### Method 1 — Fetch by DOI")
                with gr.Row():
                    doi_input = gr.Textbox(label="DOI", placeholder="e.g. 10.1126/sciadv.1601007")
                    doi_btn   = gr.Button("Fetch & Ingest", variant="secondary")
                doi_out = gr.Textbox(label="Result", lines=4, interactive=False)

                gr.Markdown("#### Method 2 — Search by Topic")
                with gr.Row():
                    search_query_input = gr.Textbox(label="Search Query", placeholder="e.g. auxetic cardiac patch")
                    search_papers_btn  = gr.Button("Search & Ingest Top 3", variant="secondary")
                search_out = gr.Textbox(label="Result", lines=6, interactive=False)

                gr.Markdown("#### Method 3 — Paste Directly")
                paste_title  = gr.Textbox(label="Title", lines=1)
                paste_text   = gr.Textbox(label="Abstract / Key Text", lines=5)
                with gr.Row():
                    paste_source = gr.Textbox(label="Source / Author", value="manual")
                    paste_domain = gr.Textbox(label="Domain", value="general")
                paste_btn = gr.Button("Ingest Paper", variant="secondary")
                paste_out = gr.Textbox(label="Result", lines=3, interactive=False)

                def fetch_doi(doi):
                    try:
                        from ..curation.pipeline import ingest_docs
                        from ..curation.fetchers import fetch_by_doi
                        docs = fetch_by_doi(doi)
                        if not docs: return f"No data found for DOI: {doi}"
                        ingest_docs(db, docs, verbose=False)
                        return f"Ingested {len(docs)} document(s) for DOI {doi}"
                    except Exception as e: return f"Error: {e}"

                def search_papers(query):
                    try:
                        from ..curation.pipeline import ingest_docs
                        from ..curation.fetchers import fetch_openalex
                        docs = fetch_openalex(query, max_results=3)
                        if not docs: return f"No results for: {query}"
                        ingest_docs(db, docs, verbose=False)
                        return f"Ingested {len(docs)} paper(s):\n" + "\n".join(f"  - {d.get('title','?')}" for d in docs)
                    except Exception as e: return f"Error: {e}"

                def paste_paper(title, text, source, domain):
                    try:
                        from ..curation.pipeline import ingest_docs
                        if not title.strip() or not text.strip(): return "Title and text are required."
                        ingest_docs(db, [{"doc_id": f"manual_{hash(title)%99999:05d}", "title": title.strip(), "text": text.strip(), "source": source.strip() or "manual", "domain": domain.strip() or "general"}], verbose=False)
                        return f"Ingested: {title}"
                    except Exception as e: return f"Error: {e}"

                doi_btn.click(fn=fetch_doi, inputs=[doi_input], outputs=[doi_out])
                search_papers_btn.click(fn=search_papers, inputs=[search_query_input], outputs=[search_out])
                paste_btn.click(fn=paste_paper, inputs=[paste_title, paste_text, paste_source, paste_domain], outputs=[paste_out])

        # ── Discovery Engine click — writes to both tabs ──────────────────
        def run_discovery(brief, api_key, steps, fanout):
            if not brief.strip():
                return "No route", "Please enter a design brief.", "", 0, "Enter a brief first.", "", "", "", 0, "", 0
            if api_key.strip():
                _set_api_key(api_key)
            try:
                router.steps  = int(steps)
                router.fanout = int(fanout)
                route_result     = router.route(brief)
                route_panel_text = _format_route_panel(route_result)
                result           = orchestrator.run(brief=brief, route_result=route_result)
                log_text         = _format_agent_log(result["agent_log"])
                dvnc_card        = result["final_card"]
                pipeline_score   = round(result["overall_score"] * 100)
                sources          = len(route_result.evidence_nodes)
                status = f"Complete | Score: {pipeline_score}/100 | Agents: 6 | Evidence sources: {sources} | Head-to-Head tab updated"
            except Exception as e:
                route_panel_text = f"Error: {e}"
                log_text = dvnc_card = f"[DVNC error: {e}]"
                pipeline_score = 0
                status = f"Error: {e}"

            try:
                plain_card = _call_plain_llm(brief)
            except Exception as e:
                plain_card = f"[Plain LLM error: {e}]"

            dvnc_scores  = _score_output(dvnc_card)
            plain_scores = _score_output(plain_card)
            dvnc_scores["overall"] = max(dvnc_scores["overall"], pipeline_score)
            score_table = _build_score_table(dvnc_scores, plain_scores)

            return (
                route_panel_text, dvnc_card, log_text, pipeline_score, status,
                brief, score_table, dvnc_card, dvnc_scores["overall"], plain_card, plain_scores["overall"],
            )

        run_btn.click(
            fn=run_discovery,
            inputs=[brief_input, api_key_input, steps_slider, fanout_slider],
            outputs=[
                route_panel, final_card, agent_log_out, overall_score_out, status_out,
                h2h_brief_display, h2h_scores, dvnc_out, dvnc_score_display, plain_out, plain_score_display,
            ],
        )

    return app
