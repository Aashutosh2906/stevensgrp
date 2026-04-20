"""
DVNC.AI Gradio Application
Built on the DVNC.AI Design System (Cinzel · Crimson Pro · JetBrains Mono · Parchment palette)

Tabs:
  1. Discovery Engine     — full 6-agent pipeline with visible routing
  2. Head-to-Head         — auto-populated when Discovery Engine runs
  3. Connectome Explorer  — SVG node graph + neighbour table
  4. Database Inspector   — stats, node/synapse browser
  5. Add Papers           — DOI · search · paste
"""

from __future__ import annotations
import json
import math
import os
import re
import sqlite3

import gradio as gr

from ..db.neurographdb import NeuroGraphDB, LMM_LABELS
from ..routing.davinci_router import DaVinciRouter
from ..agents.orchestrator import DVNCOrchestrator


# ══════════════════════════════════════════════════════════════════════════════
# DESIGN SYSTEM CSS
# Injected via gr.Blocks(css=...) — overrides Gradio's Soft theme
# Matches DVNC.AI Design System (colors_and_type.css + component kit)
# ══════════════════════════════════════════════════════════════════════════════

DVNC_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;500;600;700&family=Crimson+Pro:ital,wght@0,300;0,400;0,500;1,300;1,400&family=JetBrains+Mono:wght@300;400;500&display=swap');

/* ── Root ── */
:root {
  --dvnc-parchment:       #F7F3EC;
  --dvnc-parchment-light: #FFFDF7;
  --dvnc-parchment-mid:   #F0EAE0;
  --dvnc-parchment-edge:  #E6DDD0;
  --dvnc-ink:             #2A2218;
  --dvnc-ink-header:      #4A4030;
  --dvnc-ink-meta:        #8B7B62;
  --dvnc-ink-disabled:    #B5A898;
  --dvnc-gold:            #B8922A;
  --dvnc-gold-deep:       #9A7822;
  --dvnc-navy:            #1B3A5C;
  --dvnc-sage:            #6B8F71;
  --dvnc-walnut:          #9C8E7E;
  --dvnc-border:          rgba(184,146,42,0.25);
  --dvnc-border-strong:   rgba(184,146,42,0.50);
  --dvnc-font-display:    'Cinzel', Georgia, serif;
  --dvnc-font-body:       'Crimson Pro', Georgia, serif;
  --dvnc-font-mono:       'JetBrains Mono', monospace;
}

/* ── Page background ── */
body, .gradio-container, .main, footer { background: var(--dvnc-parchment) !important; }
footer { display: none !important; }

/* ── Gradio tab bar — override with design system tabs ── */
.tabs > .tab-nav {
  border-bottom: 1px solid var(--dvnc-border) !important;
  background: transparent !important;
  padding: 0 !important;
  gap: 0 !important;
}
.tabs > .tab-nav > button {
  font-family: var(--dvnc-font-display) !important;
  font-size: 10px !important;
  font-weight: 500 !important;
  letter-spacing: 0.14em !important;
  text-transform: uppercase !important;
  color: var(--dvnc-ink-meta) !important;
  padding: 12px 20px !important;
  border: none !important;
  border-bottom: 2px solid transparent !important;
  border-radius: 0 !important;
  background: transparent !important;
  transition: color 150ms !important;
}
.tabs > .tab-nav > button.selected {
  color: var(--dvnc-gold) !important;
  border-bottom: 2px solid var(--dvnc-gold) !important;
  background: transparent !important;
}
.tabs > .tab-nav > button:hover:not(.selected) {
  color: var(--dvnc-ink-header) !important;
  background: transparent !important;
}

/* ── Inputs / Textareas ── */
input[type=text], input[type=password], textarea, .block.svelte-1ed2p3z {
  font-family: var(--dvnc-font-body) !important;
  font-size: 14px !important;
  color: var(--dvnc-ink) !important;
  background: var(--dvnc-parchment-light) !important;
  border: 1px solid var(--dvnc-border) !important;
  border-radius: 6px !important;
}
input[type=text]:focus, input[type=password]:focus, textarea:focus {
  border-color: var(--dvnc-gold) !important;
  box-shadow: 0 0 0 3px rgba(184,146,42,0.15) !important;
  outline: none !important;
}

/* ── Labels on Gradio components ── */
label > span, .form > span {
  font-family: var(--dvnc-font-display) !important;
  font-size: 10px !important;
  font-weight: 500 !important;
  letter-spacing: 0.14em !important;
  text-transform: uppercase !important;
  color: var(--dvnc-ink-header) !important;
}

/* ── Primary button ── */
.primary {
  font-family: var(--dvnc-font-display) !important;
  font-size: 11px !important;
  font-weight: 500 !important;
  letter-spacing: 0.14em !important;
  text-transform: uppercase !important;
  background: var(--dvnc-gold) !important;
  color: var(--dvnc-parchment-light) !important;
  border: 1px solid var(--dvnc-gold) !important;
  border-radius: 4px !important;
  transition: background 150ms !important;
}
.primary:hover { background: var(--dvnc-gold-deep) !important; border-color: var(--dvnc-gold-deep) !important; }

/* ── Secondary button ── */
.secondary {
  font-family: var(--dvnc-font-display) !important;
  font-size: 10px !important;
  letter-spacing: 0.12em !important;
  text-transform: uppercase !important;
  background: transparent !important;
  color: var(--dvnc-gold) !important;
  border: 1px solid var(--dvnc-border) !important;
  border-radius: 4px !important;
}
.secondary:hover { background: rgba(184,146,42,0.08) !important; }

/* ── Sliders ── */
input[type=range] { accent-color: var(--dvnc-gold) !important; }

/* ── Dataframe table ── */
.table-wrap table thead tr th {
  font-family: var(--dvnc-font-display) !important;
  font-size: 9.5px !important;
  letter-spacing: 0.12em !important;
  text-transform: uppercase !important;
  color: var(--dvnc-ink-header) !important;
  background: var(--dvnc-parchment-mid) !important;
  border-bottom: 1px solid var(--dvnc-border) !important;
  padding: 8px 12px !important;
}
.table-wrap table tbody tr td {
  font-family: var(--dvnc-font-mono) !important;
  font-size: 11px !important;
  color: var(--dvnc-ink) !important;
  border-bottom: 1px solid rgba(184,146,42,0.1) !important;
  padding: 7px 12px !important;
}
.table-wrap { border: 1px solid var(--dvnc-border) !important; border-radius: 8px !important; overflow: hidden !important; background: var(--dvnc-parchment-light) !important; }

/* ── Accordion ── */
.accordion > .label-wrap {
  font-family: var(--dvnc-font-display) !important;
  font-size: 10px !important;
  letter-spacing: 0.1em !important;
  text-transform: uppercase !important;
  color: var(--dvnc-ink-header) !important;
  background: var(--dvnc-parchment-mid) !important;
  border: 1px solid var(--dvnc-border) !important;
  border-radius: 6px !important;
  padding: 10px 14px !important;
}
.accordion > .label-wrap:hover { background: var(--dvnc-parchment-edge) !important; }

/* ── Dropdown ── */
.wrap-inner { background: var(--dvnc-parchment-light) !important; border: 1px solid var(--dvnc-border) !important; }

/* ── Number input ── */
input[type=number] {
  font-family: var(--dvnc-font-mono) !important;
  font-size: 13px !important;
  color: var(--dvnc-ink) !important;
  background: var(--dvnc-parchment-light) !important;
  border: 1px solid var(--dvnc-border) !important;
  border-radius: 6px !important;
}

/* ── gr.HTML blocks — no wrapper padding ── */
.html-container { padding: 0 !important; background: transparent !important; }
"""


# ══════════════════════════════════════════════════════════════════════════════
# HTML COMPONENT GENERATORS
# Python equivalents of the JSX components in ui_kits/gradio_app/
# ══════════════════════════════════════════════════════════════════════════════

def _html_hero() -> str:
    """Hero banner — matches Hero.jsx exactly."""
    return """
<div style="
  background: linear-gradient(135deg, #1a0a2e, #16213e);
  border-radius: 12px; padding: 28px 24px; text-align: center;
  position: relative; overflow: hidden; margin-bottom: 4px;
">
  <div style="
    position:absolute; inset:0; pointer-events:none;
    background-image:
      radial-gradient(circle at 20% 30%, rgba(192,132,252,0.08) 0%, transparent 40%),
      radial-gradient(circle at 80% 70%, rgba(184,146,42,0.06) 0%, transparent 40%);
  "></div>
  <div style="font-family:'Cinzel',serif; font-size:38px; font-weight:600;
              letter-spacing:0.14em; color:#c084fc; position:relative;">
    DVNC<span style="color:#e9d5ff">.</span>AI
  </div>
  <div style="font-family:'Crimson Pro',Georgia,serif; font-size:15px; font-style:italic;
              color:#cbd5e1; margin-top:6px; position:relative;">
    Brain-Inspired Polymathic Design Discovery System
  </div>
  <div style="font-family:'Cinzel',serif; font-size:10px; letter-spacing:0.18em;
              text-transform:uppercase; color:#94a3b8; margin-top:10px; position:relative;">
    Da Vinci Routing · 6-Agent Debate · Hebbian Connectome · Visible Reasoning
  </div>
</div>"""


def _html_label(text: str) -> str:
    return f"""<div style="font-family:'Cinzel',serif; font-size:10px; font-weight:500;
  letter-spacing:0.14em; text-transform:uppercase; color:#4A4030;
  margin-bottom:6px; margin-top:14px;">{text}</div>"""


def _html_routing_panel(route_result) -> str:
    """Styled routing panel — matches RoutingPanel.jsx with colored spans."""
    if route_result is None:
        return _html_mono_block("No route computed. Enter a brief and run the pipeline.")

    REL_COLOR = {"EVOKES": "#B8922A", "PRIMES": "#1B3A5C", "MENTIONS": "#6B8F71", "CO_OCCURS": "#9C8E7E"}
    lines = []

    def row(text, color="#4A4030"):
        return f'<div style="color:{color}">{text}</div>'

    lines.append(row('╔════════════════════════════════════════════╗', '#8B7B62'))
    lines.append(row('║   <span style="color:#B8922A;font-weight:600">DA VINCI ROUTING PANEL</span>                ║', '#8B7B62'))
    lines.append(row('╚════════════════════════════════════════════╝', '#8B7B62'))
    lines.append(row(''))
    lines.append(row('── <span style="color:#B8922A">PRIMARY ROUTE</span> ──────────────────────────'))

    max_score = max((s.score for s in route_result.primary_route), default=1) or 1
    for i, step in enumerate(route_result.primary_route):
        norm = step.score / max_score
        bar_n = max(1, int(norm * 10))
        bar = f'<span style="color:#B8922A">{"█" * bar_n}</span>{"░" * (10 - bar_n)}'
        rel_str = ""
        if step.rel_from_prev:
            c = REL_COLOR.get(step.rel_from_prev, "#6B8F71")
            rel_str = f'<span style="color:{c}">─[{step.rel_from_prev}]→ </span>'
        cross = f' <span style="color:#B8922A"> ⟳ CROSS-DOMAIN</span>' if step.is_cross_domain else ''
        label_col = '#B8922A' if step.is_cross_domain else '#2A2218'
        lines.append(
            f'<div>  {str(i).zfill(2)}. {rel_str}'
            f'<span style="color:{label_col}">{step.label:<22}</span>'
            f' {step.score:.3f} [{bar}]{cross}</div>'
        )
        if step.lmm_tags:
            names = [LMM_LABELS.get(t, t) for t in step.lmm_tags[:2]]
            lines.append(f'<div style="color:#8B7B62">       LMM: {", ".join(names)}</div>')

    if route_result.alternative_routes:
        lines.append(row(''))
        lines.append(row('── <span style="color:#B8922A">ALTERNATIVE ROUTES</span> ──────────────────────'))
        for i, alt in enumerate(route_result.alternative_routes, 1):
            path = ' → '.join(s.label for s in alt)
            lines.append(f'<div style="color:#8B7B62">  Alt {i}: {path}</div>')

    if route_result.suppressed_hubs:
        lines.append(row(''))
        lines.append(row('── <span style="color:#B8922A">SUPPRESSED HUBS</span> ──────────────────────────'))
        for hub in route_result.suppressed_hubs[:5]:
            lines.append(f'<div style="color:#9C8E7E">  ✕ {hub.split("::")[-1]}</div>')

    lmm = route_result.lmm_activations
    if lmm:
        lines.append(row(''))
        lines.append(row('── <span style="color:#B8922A">DA VINCI MENTAL MODEL ACTIVATIONS</span> ────'))
        for code, score in sorted(lmm.items(), key=lambda x: x[1], reverse=True)[:5]:
            name = LMM_LABELS.get(code, code)
            bar_n = max(1, int(score * 5))
            bar = f'<span style="color:#B8922A">{"█" * bar_n}</span>{"░" * (5 - bar_n)}'
            lines.append(f'<div>  <span style="color:#B8922A">{code}</span>'
                         f' <span style="color:#4A4030">{name:<28}</span>'
                         f' {score:.2f} [{bar}]</div>')

    lines.append(row(''))
    lines.append(row('── <span style="color:#B8922A">SPREADING ACTIVATION</span> (top 15 nodes) ──'))
    max_act = max((s for _, s in route_result.activation_map[:15]), default=1)
    for node_id, score in route_result.activation_map[:15]:
        label = node_id.split("::")[-1][:24]
        norm = score / max_act if max_act else 0
        bar_n = max(1, int(norm * 16))
        bar = f'<span style="color:#B8922A">{"█" * bar_n}</span>{"░" * (16 - bar_n)}'
        lines.append(f'<div>  <span style="color:#2A2218">{label:<24}</span> {score:.4f} [{bar}]</div>')

    lines.append(row(''))
    lines.append(
        f'<div>Novelty score: <span style="color:#B8922A">{route_result.novelty_score:.3f}</span>'
        f'&nbsp;&nbsp;&nbsp;Cross-domain leaps: <span style="color:#B8922A">{route_result.cross_domain_count}</span></div>'
    )

    inner = "\n".join(lines)
    return f"""
<div style="
  font-family:'JetBrains Mono',monospace; font-size:11px; line-height:1.6;
  color:#4A4030; background:#FFFDF7;
  border:1px solid rgba(184,146,42,0.25); border-radius:8px;
  padding:14px 16px; white-space:pre; overflow-x:auto;
">{inner}</div>"""


def _html_innovation_card(card_text: str, score: int, lmm_hint: str = "") -> str:
    """InnovationCard.jsx — parchment gradient background with grid overlay."""
    if not card_text or card_text.startswith("["):
        return f"""
<div style="
  background: radial-gradient(ellipse at 50% 40%, #FFFDF7 0%, #F0EAE0 55%, #E6DDD0 100%);
  border:1px dashed rgba(184,146,42,0.35); border-radius:12px;
  padding:32px 24px; text-align:center; min-height:120px;
  display:flex; align-items:center; justify-content:center;
">
  <div style="font-family:'Crimson Pro',Georgia,serif; font-size:14px; font-style:italic; color:#8B7B62;">
    Run the pipeline to populate this card.
  </div>
</div>"""

    # Highlight [S#] citations
    styled = re.sub(
        r'\[S(\d+)\]',
        r'<span style="font-family:\'JetBrains Mono\',monospace;font-size:9px;'
        r'color:#1B3A5C;background:#EBF3FB;border:1px solid #C5DCF0;'
        r'border-radius:3px;padding:1px 4px;margin-left:1px">[S\1]</span>',
        card_text
    )
    # Convert newlines to <br>
    styled = styled.replace("\n", "<br>")

    lmm_tag = f" · {lmm_hint}" if lmm_hint else ""

    return f"""
<div style="
  position:relative; overflow:hidden;
  background: radial-gradient(ellipse at 50% 40%, #FFFDF7 0%, #F0EAE0 55%, #E6DDD0 100%);
  border:1px solid rgba(184,146,42,0.35); border-radius:12px; padding:20px 24px;
">
  <div style="
    position:absolute; inset:0; pointer-events:none;
    background-image: linear-gradient(rgba(184,146,42,0.04) 1px, transparent 1px),
                      linear-gradient(90deg, rgba(184,146,42,0.04) 1px, transparent 1px);
    background-size: 36px 36px;
  "></div>
  <div style="position:relative; display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:10px;">
    <div style="font-family:'Cinzel',serif; font-size:10px; font-weight:500;
                letter-spacing:0.16em; text-transform:uppercase; color:#B8922A;">
      Innovation Card{lmm_tag}
    </div>
    <div style="font-family:'JetBrains Mono',monospace; font-size:11px; color:#8B7B62;">
      {score} / 100
    </div>
  </div>
  <div style="
    position:relative;
    font-family:'Crimson Pro',Georgia,serif; font-size:14px;
    color:#2A2218; line-height:1.65;
  ">{styled}</div>
</div>"""


def _html_agent_log(agent_log: list[dict]) -> str:
    """Styled agent debate log."""
    if not agent_log:
        return ""
    parts = []
    for entry in agent_log:
        agent_name = entry.get("agent", "")
        role = entry.get("role", "")
        output = entry.get("output", "").replace("\n", "<br>")
        parts.append(f"""
<div style="margin-bottom:18px;">
  <div style="font-family:'Cinzel',serif; font-size:10px; font-weight:500;
              letter-spacing:0.12em; text-transform:uppercase; color:#B8922A;
              border-bottom:1px solid rgba(184,146,42,0.2); padding-bottom:5px; margin-bottom:8px;">
    ══ {agent_name}
    <span style="font-family:'JetBrains Mono',monospace; font-size:9px;
                 color:#8B7B62; font-weight:400; text-transform:none;
                 letter-spacing:0.04em;"> · {role}</span>
  </div>
  <div style="font-family:'JetBrains Mono',monospace; font-size:10.5px;
              color:#4A4030; line-height:1.65;">{output}</div>
</div>""")
    inner = "".join(parts)
    return f"""
<div style="
  background:#FFFDF7; border:1px solid rgba(184,146,42,0.25);
  border-radius:8px; padding:16px 18px; max-height:480px; overflow-y:auto;
">{inner}</div>"""


def _html_status(text: str, ok: bool = True) -> str:
    color = "#6B8F71" if ok else "#8B3A2A"
    return f"""
<div style="font-family:'JetBrains Mono',monospace; font-size:11px; color:{color};
            padding:8px 12px; background:rgba(107,143,113,0.06);
            border:1px solid rgba(107,143,113,0.2); border-radius:6px; margin-top:4px;">
  {text}
</div>"""


def _html_mono_block(text: str) -> str:
    """Plain JetBrains Mono block — for fallback displays."""
    return f"""
<div style="font-family:'JetBrains Mono',monospace; font-size:11px; line-height:1.6;
            color:#4A4030; background:#FFFDF7; border:1px solid rgba(184,146,42,0.2);
            border-radius:8px; padding:14px 16px; white-space:pre-wrap;">{text}</div>"""


def _html_score_table(dvnc_scores: dict, plain_scores: dict) -> str:
    """Score comparison — matches the box-drawing table style."""
    dims = [("Citations","citations"),("Specificity","specificity"),
            ("Structure","structure"),("Concision","concision"),("OVERALL","overall")]
    rows = []
    for label, key in dims:
        d, p = dvnc_scores[key], plain_scores[key]
        d_bar = "█" * (d // 10) + "░" * (10 - d // 10)
        p_bar = "█" * (p // 10) + "░" * (10 - p // 10)
        winner = "<-- DVNC" if d > p else ("<-- LLM " if p > d else "  TIE  ")
        w_color = "#B8922A" if d > p else "#1B3A5C" if p > d else "#9C8E7E"
        is_overall = label == "OVERALL"
        row_style = "font-weight:600;" if is_overall else ""
        rows.append(
            f'<div style="display:flex; gap:8px; padding:4px 0; {row_style}'
            f'border-top:{"1px solid rgba(184,146,42,0.2)" if is_overall else "none"}; margin-top:{"6px" if is_overall else "0"};">'
            f'<span style="font-family:\'Cinzel\',serif;font-size:9.5px;letter-spacing:0.08em;'
            f'text-transform:uppercase;color:#4A4030;width:90px;flex-shrink:0;">{label}</span>'
            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:10px;color:#2A2218;">'
            f'{d:3d} [<span style="color:#B8922A">{d_bar}</span>]</span>'
            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:10px;color:#4A4030;">'
            f'{p:3d} [<span style="color:#9C8E7E">{p_bar}</span>]</span>'
            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:9.5px;color:{w_color};">'
            f'{winner}</span>'
            f'</div>'
        )
    header = (
        '<div style="display:flex; gap:8px; padding:0 0 6px; '
        'border-bottom:1px solid rgba(184,146,42,0.25); margin-bottom:4px;">'
        '<span style="font-family:\'Cinzel\',serif;font-size:9px;letter-spacing:0.14em;'
        'text-transform:uppercase;color:#8B7B62;width:90px;">Dimension</span>'
        '<span style="font-family:\'Cinzel\',serif;font-size:9px;letter-spacing:0.1em;'
        'text-transform:uppercase;color:#B8922A;">DVNC.AI</span>'
        '<span style="font-family:\'Cinzel\',serif;font-size:9px;letter-spacing:0.1em;'
        'text-transform:uppercase;color:#9C8E7E;margin-left:8px;">Plain LLM</span>'
        '<span style="font-family:\'Cinzel\',serif;font-size:9px;letter-spacing:0.1em;'
        'text-transform:uppercase;color:#8B7B62;margin-left:auto;">Winner</span>'
        '</div>'
    )
    return f"""
<div style="background:#FFFDF7; border:1px solid rgba(184,146,42,0.25);
            border-radius:8px; padding:14px 16px;">
  {header}{"".join(rows)}
</div>"""


# ══════════════════════════════════════════════════════════════════════════════
# NODE GRAPH SVG — matches NodeGraph.jsx exactly
# ══════════════════════════════════════════════════════════════════════════════

def _truncate(text: str, n: int = 12) -> str:
    return text if len(text) <= n else text[:n - 1] + "…"


def _html_node_graph(concept: str, neighbors: list[dict]) -> str:
    """
    Pure SVG knowledge graph — direct Python port of NodeGraph.jsx.
    Uses the same 720×380 viewBox, radial layout, color scheme.
    """
    REL_COLOR   = {"EVOKES": "#B8922A", "PRIMES": "#1B3A5C", "MENTIONS": "#6B8F71", "CO_OCCURS": "#9C8E7E"}
    REL_WIDTH   = {"EVOKES": 2.2, "PRIMES": 1.8, "MENTIONS": 1.3, "CO_OCCURS": 0.9}
    REL_OPACITY = {"EVOKES": 0.85, "PRIMES": 0.8, "MENTIONS": 0.65, "CO_OCCURS": 0.45}

    cx, cy = 360, 190
    r_ring = 145

    # Deduplicate neighbors — keep highest weight per label
    seen: dict[str, dict] = {}
    for n in neighbors:
        lbl = n["post"].split("::")[-1]
        if lbl == concept:
            lbl = n["pre"].split("::")[-1]
        rel = n.get("rel", "CO_OCCURS")
        w   = float(n.get("weight", 1.0))
        if lbl not in seen or w > seen[lbl]["weight"]:
            seen[lbl] = {"weight": w, "rel": rel}

    node_list = sorted(seen.items(), key=lambda x: x[1]["weight"], reverse=True)
    n = len(node_list)

    if n == 0:
        return _html_node_graph_empty(concept)

    # Two rings if many nodes
    if n <= 8:
        rings = [(node_list, r_ring)]
    else:
        rings = [(node_list[:8], 115), (node_list[8:], 195)]

    positions: dict[str, tuple[float, float]] = {}
    for ring_nodes, radius in rings:
        count = len(ring_nodes)
        for i, (lbl, _) in enumerate(ring_nodes):
            angle = (2 * math.pi * i / count) - math.pi / 2
            positions[lbl] = (cx + math.cos(angle) * radius, cy + math.sin(angle) * radius)

    max_w = max((v["weight"] for v in seen.values()), default=1) or 1
    total_nodes = n + 1
    total_edges = len(neighbors)

    parts = [
        f'<div style="position:relative;height:380px;'
        f'background:radial-gradient(ellipse at 50% 40%,#FFFDF7 0%,#F0EAE0 55%,#E6DDD0 100%);'
        f'border:1px solid rgba(184,146,42,0.25);border-radius:12px;overflow:hidden;">',

        # Grid overlay
        '<div style="position:absolute;inset:0;pointer-events:none;'
        'background-image:linear-gradient(rgba(184,146,42,0.04) 1px,transparent 1px),'
        'linear-gradient(90deg,rgba(184,146,42,0.04) 1px,transparent 1px);'
        'background-size:36px 36px;"></div>',

        # Title
        f'<div style="position:absolute;top:14px;left:18px;'
        f'font-family:\'Cinzel\',serif;font-size:11px;font-weight:500;'
        f'letter-spacing:0.12em;text-transform:uppercase;color:#4A4030;">'
        f'Connectome · <span style="color:#B8922A;font-weight:600">{concept.upper()}</span></div>',

        # Meta
        f'<div style="position:absolute;top:14px;right:16px;'
        f'font-family:\'JetBrains Mono\',monospace;font-size:9.5px;color:#8B7B62;">'
        f'{total_nodes} nodes · {total_edges} edges</div>',

        # SVG
        '<svg width="100%" height="100%" viewBox="0 0 720 380">'
        '<defs>'
        '<filter id="gg" x="-50%" y="-50%" width="200%" height="200%">'
        '<feGaussianBlur in="SourceGraphic" stdDeviation="5" result="b"/>'
        '<feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>'
        '</filter>'
        '</defs>',
    ]

    # Edges — straight lines from NodeGraph.jsx
    for lbl, info in node_list:
        if lbl not in positions:
            continue
        x2, y2 = positions[lbl]
        rel = info["rel"]
        color   = REL_COLOR.get(rel, "#9C8E7E")
        width   = REL_WIDTH.get(rel, 1.0)
        opacity = REL_OPACITY.get(rel, 0.5)
        parts.append(
            f'<line x1="{cx}" y1="{cy}" x2="{x2:.1f}" y2="{y2:.1f}"'
            f' stroke="{color}" stroke-width="{width}" opacity="{opacity}"/>'
        )

    # Central node — matches NodeGraph.jsx exactly
    parts.append(
        f'<circle cx="{cx}" cy="{cy}" r="40" fill="none"'
        f' stroke="#B8922A" stroke-width="0.8" stroke-dasharray="3 3" opacity="0.5"/>'
        f'<circle cx="{cx}" cy="{cy}" r="30"'
        f' fill="rgba(184,146,42,0.18)" stroke="#B8922A" stroke-width="2" filter="url(#gg)"/>'
        f'<text x="{cx}" y="{cy + 4}" text-anchor="middle"'
        f' font-family="\'Cinzel\',serif" font-size="11" font-weight="600"'
        f' fill="#B8922A" letter-spacing="0.06em">{_truncate(concept.upper(), 12)}</text>'
    )

    # Neighbor nodes — matches NodeGraph.jsx
    for lbl, info in node_list:
        if lbl not in positions:
            continue
        x, y = positions[lbl]
        rel   = info["rel"]
        w     = info["weight"]
        # Scale radius 16–22 by weight
        node_r = round(16 + 6 * min(1.0, w / max_w), 1)
        parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{node_r}"'
            f' fill="rgba(28,25,20,0.06)" stroke="#4A4030" stroke-width="1"/>'
            f'<text x="{x:.1f}" y="{y + 3:.1f}" text-anchor="middle"'
            f' font-family="\'JetBrains Mono\',monospace" font-size="8.5" fill="#2A2218"'
            f' style="user-select:none">{_truncate(lbl, 12)}</text>'
        )

    # Legend — matches NodeGraph.jsx
    lx, ly = 620, 270
    parts.append(
        f'<rect x="{lx - 8}" y="{ly - 12}" width="114" height="88" rx="5"'
        f' fill="rgba(255,253,247,0.88)" stroke="rgba(184,146,42,0.2)" stroke-width="0.8"/>'
    )
    legend_items = [("EVOKES","#B8922A",2),("PRIMES","#1B3A5C",2),
                    ("MENTIONS","#6B8F71",1),("CO_OCCURS","#9C8E7E",1)]
    for i, (name, color, lw) in enumerate(legend_items):
        ry = ly + i * 18
        parts.append(
            f'<line x1="{lx}" y1="{ry}" x2="{lx + 22}" y2="{ry}"'
            f' stroke="{color}" stroke-width="{lw}" opacity="0.9"/>'
            f'<text x="{lx + 28}" y="{ry + 4}"'
            f' font-family="\'Cinzel\',serif" font-size="9" letter-spacing="0.1em" fill="#4A4030"'
            f' style="user-select:none">{name}</text>'
        )

    parts.append('</svg></div>')
    return "\n".join(parts)


def _html_node_graph_empty(concept: str) -> str:
    return f"""
<div style="height:380px;
  background:radial-gradient(ellipse at 50% 40%,#FFFDF7 0%,#F0EAE0 55%,#E6DDD0 100%);
  border:1px solid rgba(184,146,42,0.25);border-radius:12px;
  display:flex;align-items:center;justify-content:center;">
  <div style="text-align:center;">
    <div style="font-family:'Cinzel',serif;font-size:13px;letter-spacing:0.12em;
                text-transform:uppercase;color:#8B7B62;">Connectome · {concept.upper()}</div>
    <div style="font-family:'Crimson Pro',Georgia,serif;font-size:13px;
                color:#B5A898;font-style:italic;margin-top:6px;">No connections found</div>
  </div>
</div>"""


def _html_node_graph_placeholder() -> str:
    cx, cy = 360, 190
    return f"""
<div style="position:relative;height:380px;
  background:radial-gradient(ellipse at 50% 40%,#FFFDF7 0%,#F0EAE0 55%,#E6DDD0 100%);
  border:1px solid rgba(184,146,42,0.25);border-radius:12px;overflow:hidden;">
  <div style="position:absolute;inset:0;pointer-events:none;
    background-image:linear-gradient(rgba(184,146,42,0.04) 1px,transparent 1px),
    linear-gradient(90deg,rgba(184,146,42,0.04) 1px,transparent 1px);
    background-size:36px 36px;"></div>
  <svg width="100%" height="100%" viewBox="0 0 720 380">
    <circle cx="{cx}" cy="{cy}" r="40" fill="none" stroke="#B8922A"
            stroke-width="0.8" stroke-dasharray="3 3" opacity="0.4"/>
    <circle cx="{cx}" cy="{cy}" r="30"
            fill="rgba(184,146,42,0.08)" stroke="#B8922A" stroke-width="1.5" opacity="0.5"/>
    <text x="{cx}" y="{cy - 12}" text-anchor="middle"
          font-family="'Cinzel',serif" font-size="11" letter-spacing="0.12em"
          fill="#B8922A" opacity="0.6">KNOWLEDGE</text>
    <text x="{cx}" y="{cy + 6}" text-anchor="middle"
          font-family="'Cinzel',serif" font-size="11" letter-spacing="0.12em"
          fill="#B8922A" opacity="0.6">GRAPH</text>
    <text x="{cx}" y="{cy + 60}" text-anchor="middle"
          font-family="'Crimson Pro',Georgia,serif" font-size="13"
          fill="#8B7B62" font-style="italic">Enter a concept and click Explore</text>
  </svg>
</div>"""


# ══════════════════════════════════════════════════════════════════════════════
# BACKEND HELPERS
# ══════════════════════════════════════════════════════════════════════════════

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
    return {"citations": cit_score, "specificity": spec_score, "structure": struct_score,
            "concision": concision, "overall": overall}


def _db_stats(db_path: str) -> str:
    try:
        conn = sqlite3.connect(db_path)
        cur  = conn.cursor()
        lines = [
            "╔══════════════════════════════════════════════╗",
            "║         DVNC CONNECTOME — DB STATS          ║",
            "╚══════════════════════════════════════════════╝", ""
        ]
        for sql, label in [("SELECT COUNT(*) FROM nodes","Total Nodes"),("SELECT COUNT(*) FROM synapses","Total Synapses")]:
            try: cur.execute(sql); lines.append(f"  {label:<30} {cur.fetchone()[0]}")
            except Exception: pass
        for sql, header in [
            ("SELECT kind, COUNT(*) FROM nodes GROUP BY kind ORDER BY COUNT(*) DESC","Node breakdown:"),
            ("SELECT rel, COUNT(*) FROM synapses GROUP BY rel ORDER BY COUNT(*) DESC","Synapse breakdown:"),
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
        rows = conn.execute(
            "SELECT pre, post, rel, weight, evidence_count, lmm_tags FROM synapses ORDER BY weight DESC LIMIT ?", (limit,)
        ).fetchall()
        conn.close()
        result = []
        for r in rows:
            try: tags_str = ", ".join(json.loads(r["lmm_tags"])[:2])
            except Exception: tags_str = ""
            result.append([r["pre"].split("::")[-1], r["post"].split("::")[-1],
                           r["rel"], round(r["weight"], 3), r["evidence_count"], tags_str])
        return result
    except Exception as e:
        return [[f"Error: {e}", "", "", "", "", ""]]


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP FACTORY
# ══════════════════════════════════════════════════════════════════════════════

def make_app(db_path: str) -> gr.Blocks:
    db           = NeuroGraphDB(db_path)
    router       = DaVinciRouter(db)
    orchestrator = DVNCOrchestrator(db=db)

    with gr.Blocks(
        title="DVNC.AI — Brain-Inspired Design Discovery",
        theme=gr.themes.Base(),
        css=DVNC_CSS,
    ) as app:

        # ── Hero ──────────────────────────────────────────────────────────
        gr.HTML(_html_hero())

        with gr.Tabs():

            # ══════════════════════════════════════════════════════════════
            # TAB 1 — Discovery Engine
            # ══════════════════════════════════════════════════════════════
            with gr.TabItem("Discovery Engine"):
                gr.HTML("""
                <div style="font-family:'Crimson Pro',Georgia,serif; font-size:14px;
                            font-style:italic; color:#4A4030; line-height:1.55;
                            margin:14px 0 18px;">
                  Enter a design challenge. The system routes through the DVNC Connectome,
                  runs 6 specialised AI agents in debate, and produces an evidence-anchored
                  Innovation Card with full provenance.
                  After running, the Head-to-Head tab auto-populates.
                </div>""")

                with gr.Row():
                    with gr.Column(scale=2):
                        brief_input = gr.Textbox(
                            label="Design Brief",
                            placeholder="e.g. Design a lightweight load-bearing structure inspired by biological tissue architecture that can be additively manufactured...",
                            lines=4,
                        )
                    with gr.Column(scale=1):
                        api_key_input = gr.Textbox(
                            label="API Key — Groq / Gemini / DeepSeek / Anthropic",
                            placeholder="gsk_… or AIza… or sk-ant-…",
                            type="password",
                        )
                        steps_slider  = gr.Slider(2, 6, value=4, step=1,  label="Routing Steps")
                        fanout_slider = gr.Slider(5, 40, value=20, step=5, label="Fanout per Step")
                        run_btn = gr.Button("Run DVNC Discovery", variant="primary", size="lg")

                # Routing Panel
                gr.HTML(_html_label("Da Vinci Routing Panel"))
                route_panel_out = gr.HTML(_html_routing_panel(None))

                # Innovation Card
                gr.HTML(_html_label("Final Innovation Card"))
                final_card_out = gr.HTML(_html_innovation_card("", 0))

                # Agent log
                with gr.Accordion("Agent Debate Log", open=False):
                    agent_log_out = gr.HTML("")

                # Status + score
                with gr.Row():
                    status_out        = gr.HTML("")
                    overall_score_out = gr.Number(label="Score (0–100)", value=0)

            # ══════════════════════════════════════════════════════════════
            # TAB 2 — Head-to-Head
            # ══════════════════════════════════════════════════════════════
            with gr.TabItem("Head-to-Head"):
                gr.HTML("""
                <div style="padding:10px 0 16px;">
                  <div style="font-family:'Cinzel',serif;font-size:13px;font-weight:500;
                              letter-spacing:0.06em;color:#4A4030;margin-bottom:6px;">
                    DVNC.AI vs Plain LLM
                  </div>
                  <div style="font-family:'Crimson Pro',Georgia,serif;font-size:14px;
                              font-style:italic;color:#8B7B62;line-height:1.55;">
                    Auto-populated when you run the Discovery Engine.
                    The same brief is sent directly to the LLM — no routing, no knowledge
                    graph, no 6-agent pipeline — so you can see exactly what the connectome adds.
                  </div>
                </div>""")

                h2h_brief_display = gr.Textbox(
                    label="Brief being compared",
                    lines=2, interactive=False,
                    placeholder="Run a query in the Discovery Engine — results appear here automatically.",
                )

                gr.HTML(_html_label("Score Comparison"))
                h2h_scores_out = gr.HTML("")

                gr.HTML('<hr style="border:none;border-top:1px solid rgba(184,146,42,0.2);margin:16px 0;">')

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.HTML("""
                        <div style="font-family:'Cinzel',serif;font-size:12px;font-weight:500;
                                    letter-spacing:0.1em;color:#4A4030;margin-bottom:4px;">
                          DVNC.AI
                        </div>
                        <div style="font-family:'Crimson Pro',Georgia,serif;font-size:12px;
                                    font-style:italic;color:#8B7B62;margin-bottom:10px;">
                          6-Agent Debate · Evidence-grounded
                        </div>""")
                        dvnc_out           = gr.HTML(_html_innovation_card("", 0))
                        dvnc_score_display = gr.Number(label="DVNC.AI Score (0–100)", value=0)
                    with gr.Column(scale=1):
                        gr.HTML("""
                        <div style="font-family:'Cinzel',serif;font-size:12px;font-weight:500;
                                    letter-spacing:0.1em;color:#4A4030;margin-bottom:4px;">
                          Plain LLM
                        </div>
                        <div style="font-family:'Crimson Pro',Georgia,serif;font-size:12px;
                                    font-style:italic;color:#8B7B62;margin-bottom:10px;">
                          No graph · No citations · No routing
                        </div>""")
                        plain_out           = gr.HTML(_html_innovation_card("", 0))
                        plain_score_display = gr.Number(label="Plain LLM Score (0–100)", value=0)

            # ══════════════════════════════════════════════════════════════
            # TAB 3 — Connectome Explorer
            # ══════════════════════════════════════════════════════════════
            with gr.TabItem("Connectome Explorer"):
                gr.HTML("""
                <div style="font-family:'Crimson Pro',Georgia,serif;font-size:14px;
                            font-style:italic;color:#4A4030;line-height:1.55;margin:14px 0 18px;">
                  Explore concepts in the knowledge graph.
                  Enter a concept to see its neighbours and the network diagram.
                </div>""")

                with gr.Row():
                    concept_input = gr.Textbox(
                        label="Concept",
                        value="bone",
                        placeholder="e.g. auxetic, raman, scaffold, cardiomyocyte",
                    )
                    limit_slider = gr.Slider(5, 40, value=12, step=4, label="Max Neighbours")
                    explore_btn  = gr.Button("Explore", variant="primary")

                # Node graph
                graph_out = gr.HTML(_html_node_graph_placeholder())

                # Neighbour table
                gr.HTML(_html_label("Neighbour Table"))
                neighbours_out = gr.Dataframe(
                    headers=["From", "To", "Relation", "Weight", "Evidence"],
                    interactive=False,
                )

                gr.HTML('<hr style="border:none;border-top:1px solid rgba(184,146,42,0.15);margin:16px 0;">')
                gr.HTML(_html_label("Spreading Activation"))
                with gr.Row():
                    prop_steps  = gr.Slider(1, 6, value=3, step=1, label="Steps")
                    prop_fanout = gr.Slider(5, 30, value=15, step=5, label="Fanout")
                    prop_btn    = gr.Button("Propagate", variant="secondary")
                propagation_out = gr.Dataframe(
                    headers=["Node ID", "Label", "Activation Score"],
                    interactive=False,
                )

                def explore_concept(concept, limit):
                    concept   = concept.strip().lower()
                    node_id   = f"concept::{concept}"
                    neighbors = db.top_neighbors(node_id, limit=int(limit))
                    rows = [
                        [n["pre"].split("::")[-1], n["post"].split("::")[-1],
                         n["rel"], round(n["weight"], 3), n["evidence_count"]]
                        for n in neighbors
                    ]
                    graph = _html_node_graph(concept, neighbors) if neighbors else _html_node_graph_empty(concept)
                    return graph, rows

                def propagate_concept(concept, steps, fanout):
                    node_id = f"concept::{concept.strip().lower()}"
                    results = db.propagate(node_id, steps=int(steps), fanout=int(fanout))
                    return [[r[0], r[0].split("::")[-1], round(r[1], 5)] for r in results[:30]]

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
                gr.HTML(_html_label("Connectome Statistics"))
                stats_btn = gr.Button("Refresh Stats", variant="secondary")
                stats_out = gr.Textbox(label="Statistics", lines=18, interactive=False)
                with gr.Row():
                    node_kind       = gr.Dropdown(choices=["all","concept","document","domain"], value="all", label="Node Kind")
                    node_search     = gr.Textbox(label="Search Nodes (label contains)", placeholder="e.g. cardiac")
                    node_search_btn = gr.Button("Search", variant="secondary")
                nodes_out   = gr.Dataframe(headers=["ID","Kind","Label","Domain"], interactive=False)
                top_syn_btn = gr.Button("Top 50 Synapses by Weight", variant="secondary")
                syn_out     = gr.Dataframe(headers=["From","To","Relation","Weight","Evidence","LMM Tags"], interactive=False)

                def get_stats(): return _db_stats(db_path)

                def search_nodes(kind, search_term):
                    nodes = db.search_nodes(query=search_term or "", kind=None if kind=="all" else kind, limit=100)
                    rows = []
                    for nd in nodes:
                        props = nd.get("props", {})
                        if isinstance(props, str):
                            try: props = json.loads(props)
                            except Exception: props = {}
                        rows.append([nd["id"], nd["kind"], nd["label"], props.get("domain", props.get("domain_hint", ""))])
                    return rows

                def get_top_synapses(): return _db_top_synapses(db_path, limit=50)

                stats_btn.click(fn=get_stats, outputs=[stats_out])
                node_search_btn.click(fn=search_nodes, inputs=[node_kind, node_search], outputs=[nodes_out])
                top_syn_btn.click(fn=get_top_synapses, outputs=[syn_out])

            # ══════════════════════════════════════════════════════════════
            # TAB 5 — Add Papers
            # ══════════════════════════════════════════════════════════════
            with gr.TabItem("Add Papers"):
                gr.HTML("""
                <div style="font-family:'Crimson Pro',Georgia,serif;font-size:14px;
                            color:#4A4030;line-height:1.55;margin:14px 0 18px;">
                  Add new papers to enrich the knowledge graph.
                  Three methods: Fetch by DOI · Search by Topic · Paste Directly.
                </div>""")

                gr.HTML(_html_label("Method 1 — Fetch by DOI"))
                with gr.Row():
                    doi_input = gr.Textbox(label="DOI", placeholder="e.g. 10.1126/sciadv.1601007")
                    doi_btn   = gr.Button("Fetch & Ingest", variant="secondary")
                doi_out = gr.Textbox(label="Result", lines=3, interactive=False)

                gr.HTML(_html_label("Method 2 — Search by Topic"))
                with gr.Row():
                    search_query_input = gr.Textbox(label="Query", placeholder="e.g. auxetic cardiac patch biomaterial")
                    search_papers_btn  = gr.Button("Search & Ingest Top 3", variant="secondary")
                search_out = gr.Textbox(label="Result", lines=4, interactive=False)

                gr.HTML(_html_label("Method 3 — Paste Directly"))
                paste_title = gr.Textbox(label="Title", lines=1)
                paste_text  = gr.Textbox(label="Abstract / Key Text", lines=5)
                with gr.Row():
                    paste_source = gr.Textbox(label="Source", value="manual")
                    paste_domain = gr.Textbox(label="Domain", value="general")
                paste_btn = gr.Button("Ingest Paper", variant="secondary")
                paste_out = gr.Textbox(label="Result", lines=2, interactive=False)

                def fetch_doi(doi):
                    try:
                        from ..curation.pipeline import ingest_docs
                        from ..curation.fetchers import fetch_by_doi
                        docs = fetch_by_doi(doi)
                        if not docs: return f"No data found for DOI: {doi}"
                        ingest_docs(db, docs, verbose=False)
                        return f"✓ Ingested {len(docs)} document(s) — DOI {doi}"
                    except Exception as e: return f"Error: {e}"

                def search_papers(query):
                    try:
                        from ..curation.pipeline import ingest_docs
                        from ..curation.fetchers import fetch_openalex
                        docs = fetch_openalex(query, max_results=3)
                        if not docs: return f"No results for: {query}"
                        ingest_docs(db, docs, verbose=False)
                        return "✓ Ingested:\n" + "\n".join(f"  — {d.get('title','?')}" for d in docs)
                    except Exception as e: return f"Error: {e}"

                def paste_paper(title, text, source, domain):
                    try:
                        from ..curation.pipeline import ingest_docs
                        if not title.strip() or not text.strip(): return "Title and text are required."
                        ingest_docs(db, [{"doc_id": f"manual_{hash(title)%99999:05d}",
                                          "title": title.strip(), "text": text.strip(),
                                          "source": source.strip() or "manual",
                                          "domain": domain.strip() or "general"}], verbose=False)
                        return f"✓ Ingested: {title}"
                    except Exception as e: return f"Error: {e}"

                doi_btn.click(fn=fetch_doi, inputs=[doi_input], outputs=[doi_out])
                search_papers_btn.click(fn=search_papers, inputs=[search_query_input], outputs=[search_out])
                paste_btn.click(fn=paste_paper, inputs=[paste_title, paste_text, paste_source, paste_domain], outputs=[paste_out])

        # ══════════════════════════════════════════════════════════════════
        # DISCOVERY ENGINE CLICK HANDLER
        # Writes to Tab 1 AND Tab 2 simultaneously
        # ══════════════════════════════════════════════════════════════════

        def run_discovery(brief, api_key, steps, fanout):
            blank_route  = _html_routing_panel(None)
            blank_card   = _html_innovation_card("", 0)

            if not brief.strip():
                return (blank_route, blank_card, "", 0,
                        _html_status("Enter a brief and run the pipeline.", ok=False),
                        "", blank_card, blank_card, 0, 0, "")

            if api_key.strip():
                _set_api_key(api_key)

            # ── DVNC pipeline ─────────────────────────────────────────────
            try:
                router.steps  = int(steps)
                router.fanout = int(fanout)
                route_result   = router.route(brief)
                result         = orchestrator.run(brief=brief, route_result=route_result)
                dvnc_card_text = result["final_card"]
                pipeline_score = round(result["overall_score"] * 100)
                sources        = len(route_result.evidence_nodes)

                # Extract top LMM for card subtitle
                lmm_hint = ""
                if route_result.lmm_activations:
                    top_lmm = max(route_result.lmm_activations, key=route_result.lmm_activations.get)
                    lmm_hint = LMM_LABELS.get(top_lmm, top_lmm)

                route_html  = _html_routing_panel(route_result)
                card_html   = _html_innovation_card(dvnc_card_text, pipeline_score, lmm_hint)
                log_html    = _html_agent_log(result["agent_log"])
                status_html = _html_status(
                    f"Complete | Score: {pipeline_score}/100 | Agents: 6 | Evidence sources: {sources} | Head-to-Head tab updated"
                )
            except Exception as e:
                route_html  = _html_mono_block(f"Error: {e}")
                card_html   = _html_innovation_card("", 0)
                log_html    = ""
                status_html = _html_status(f"Error: {e}", ok=False)
                dvnc_card_text = f"[DVNC error: {e}]"
                pipeline_score = 0

            # ── Plain LLM ─────────────────────────────────────────────────
            try:
                plain_card_text = _call_plain_llm(brief)
            except Exception as e:
                plain_card_text = f"[Plain LLM error: {e}]"

            # ── Scores ────────────────────────────────────────────────────
            dvnc_scores  = _score_output(dvnc_card_text)
            plain_scores = _score_output(plain_card_text)
            dvnc_scores["overall"] = max(dvnc_scores["overall"], pipeline_score)

            scores_html   = _html_score_table(dvnc_scores, plain_scores)
            dvnc_h2h_html = _html_innovation_card(dvnc_card_text, dvnc_scores["overall"])
            plain_h2h_html= _html_innovation_card(plain_card_text, plain_scores["overall"])

            return (
                # Tab 1
                route_html, card_html, log_html, pipeline_score, status_html,
                # Tab 2
                brief, scores_html, dvnc_h2h_html, dvnc_scores["overall"],
                plain_h2h_html, plain_scores["overall"],
            )

        run_btn.click(
            fn=run_discovery,
            inputs=[brief_input, api_key_input, steps_slider, fanout_slider],
            outputs=[
                route_panel_out, final_card_out, agent_log_out,
                overall_score_out, status_out,
                h2h_brief_display, h2h_scores_out, dvnc_out,
                dvnc_score_display, plain_out, plain_score_display,
            ],
        )

    return app
