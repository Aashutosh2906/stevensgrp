"""
DVNC.AI Node Graph
Generates the premium knowledge-graph SVG from the DVNC.AI Design System.

Usage (from gradio_app.py):
    from .node_graph import build_node_graph_html, node_graph_placeholder

    # In the Explore button handler:
    html = build_node_graph_html(concept, neighbors)

    # Default empty state:
    html = node_graph_placeholder()
"""

from __future__ import annotations
import math

# ── Design-system constants ────────────────────────────────────────────────

_FONTS = (
    "@import url('https://fonts.googleapis.com/css2?"
    "family=Cinzel:wght@400;500;600&"
    "family=Crimson+Pro:ital,wght@0,400;0,500;1,400&"
    "family=JetBrains+Mono:wght@300;400&display=swap');"
)

_REL_COLOR = {
    "EVOKES":    "#B8922A",
    "PRIMES":    "#1B3A5C",
    "MENTIONS":  "#6B8F71",
    "CO_OCCURS": "#9C8E7E",
}
_REL_LABEL_COLOR = {
    "EVOKES":    "#B8922A",
    "PRIMES":    "#1B3A5C",
    "MENTIONS":  "#4A6B52",
    "CO_OCCURS": "#7A6E5E",
}
_REL_EDGE_WIDTH   = {"EVOKES": 1.6, "PRIMES": 1.3, "MENTIONS": 1.2, "CO_OCCURS": 1.0}
_REL_EDGE_OPACITY = {"EVOKES": 0.50, "PRIMES": 0.45, "MENTIONS": 0.45, "CO_OCCURS": 0.40}

# Fixed scatter-field dots — same positions as design system node-graph.html
_SCATTER = [
    (70,  40,  3.0, "#B8922A", 0.18), (230, 25,  2.0, "#6B8F71", 0.25),
    (380, 35,  2.5, "#9C8E7E", 0.30), (660, 55,  3.0, "#1B3A5C", 0.20),
    (90,  340, 2.5, "#B8922A", 0.22), (190, 355, 2.0, "#6B8F71", 0.30),
    (620, 335, 3.0, "#9C8E7E", 0.25), (690, 285, 2.0, "#1B3A5C", 0.22),
    (30,  200, 2.0, "#B8922A", 0.20), (700, 190, 2.5, "#6B8F71", 0.20),
    (430, 340, 2.0, "#B8922A", 0.28), (300, 60,  1.8, "#9C8E7E", 0.35),
    (575, 40,  2.0, "#B8922A", 0.20), (485, 60,  1.5, "#6B8F71", 0.30),
    (140, 310, 1.8, "#1B3A5C", 0.22), (360, 310, 1.5, "#9C8E7E", 0.30),
    (520, 310, 2.0, "#B8922A", 0.25),
]

# Canvas — matches node-graph.html exactly
_W, _H = 720, 380
_CX, _CY = 360, 200


# ── Helpers ────────────────────────────────────────────────────────────────

def _trunc(text: str, n: int = 14) -> str:
    return text if len(text) <= n else text[:n - 1] + "…"


def _node_radius(weight: float, max_w: float) -> float:
    """Scale radius 6.5–13 by weight — matches design system range."""
    ratio = min(1.0, weight / max_w) if max_w else 0
    return round(6.5 + 6.5 * ratio, 1)


def _label_font_size(r: float) -> int:
    """Font size for neighbor label — larger dots get larger text."""
    if r >= 12:  return 15
    if r >= 10:  return 14
    if r >= 8:   return 13
    return 12


def _label_placement(x: float, y: float, r: float) -> tuple[float, float, str]:
    """
    Returns (lx, anchor) — label x-position and text-anchor.
    Nodes on the right half: label to the right.
    Nodes on the left half:  label to the left.
    Gap of 7px from dot edge.
    """
    if x >= _CX:
        return x + r + 7, y, "start"
    else:
        return x - r - 7, y, "end"


# ── SVG builder ────────────────────────────────────────────────────────────

def _svg_scatter() -> str:
    parts = ["  <g>"]
    for sx, sy, sr, sc, so in _SCATTER:
        parts.append(f'    <circle cx="{sx}" cy="{sy}" r="{sr}" fill="{sc}" opacity="{so}"/>')
    parts.append("  </g>")
    return "\n".join(parts)


def _svg_web(node_positions: list[tuple[float, float]]) -> str:
    """Faint connective web — curved paths between adjacent nodes + lines to center."""
    parts = ['  <g stroke="#C9B88E" stroke-width="1" opacity="0.35" fill="none">']
    n = len(node_positions)
    for i in range(n):
        x1, y1 = node_positions[i]
        # Line to center
        parts.append(f'    <line x1="{x1:.0f}" y1="{y1:.0f}" x2="{_CX}" y2="{_CY}"/>')
        # Curved path to next node in ring
        x2, y2 = node_positions[(i + 1) % n]
        mx = (x1 + x2) / 2
        my = (y1 + y2) / 2 - 22
        parts.append(f'    <path d="M{x1:.0f} {y1:.0f} Q {mx:.0f} {my:.0f}, {x2:.0f} {y2:.0f}"/>')
    parts.append("  </g>")
    return "\n".join(parts)


def _svg_edges(node_list: list[tuple], positions: dict) -> str:
    """Colored primary edges from center to each neighbor."""
    parts = ['  <g fill="none" stroke-linecap="round">']
    for lbl, info in node_list:
        if lbl not in positions:
            continue
        x2, y2  = positions[lbl]
        rel      = info["rel"]
        color    = _REL_COLOR.get(rel, "#9C8E7E")
        width    = _REL_EDGE_WIDTH.get(rel, 1.0)
        opacity  = _REL_EDGE_OPACITY.get(rel, 0.4)
        dash     = ' stroke-dasharray="3 3"' if rel == "CO_OCCURS" else ""
        parts.append(
            f'    <path d="M{_CX} {_CY} L{x2:.1f} {y2:.1f}"'
            f' stroke="{color}" stroke-width="{width}" opacity="{opacity}"{dash}/>'
        )
    parts.append("  </g>")
    return "\n".join(parts)


def _svg_central(concept: str) -> str:
    """Central concept — plain Cinzel text, no circle. Matches design system exactly."""
    label = concept.upper()
    return (
        f'  <text x="{_CX}" y="{_CY - 5}" text-anchor="middle"'
        f' font-family="\'Cinzel\', serif" font-size="22" font-weight="600"'
        f' fill="#4A4030" letter-spacing="0.15em">{label}</text>\n'
        f'  <text x="{_CX}" y="{_CY + 18}" text-anchor="middle"'
        f' font-family="\'Cinzel\', serif" font-size="12" font-weight="500"'
        f' fill="#8B7B62" letter-spacing="0.22em">CONNECTOME</text>'
    )


def _svg_nodes(node_list: list[tuple], positions: dict, max_w: float) -> str:
    """
    Neighbor nodes — filled dot + floating label beside it.
    Matches node-graph.html exactly:
      - Filled circle, no stroke
      - Name in Crimson Pro beside the dot
      - REL · weight in JetBrains Mono below (omitted for very small nodes)
    """
    parts = []
    for lbl, info in node_list:
        if lbl not in positions:
            continue
        x, y    = positions[lbl]
        rel     = info["rel"]
        w       = info["weight"]
        color   = _REL_COLOR.get(rel, "#9C8E7E")
        lcolor  = _REL_LABEL_COLOR.get(rel, "#4A4030")
        r       = _node_radius(w, max_w)
        fs      = _label_font_size(r)
        lx, ly, anchor = _label_placement(x, y, r)

        # Filled dot — no stroke, matches design system
        parts.append(
            f'  <circle cx="{x:.1f}" cy="{y:.1f}" r="{r}"'
            f' fill="{color}"/>'
        )
        # Name label
        parts.append(
            f'  <text x="{lx:.1f}" y="{ly - 1:.1f}" text-anchor="{anchor}"'
            f' font-family="\'Crimson Pro\', Georgia, serif"'
            f' font-size="{fs}" font-weight="500" fill="{lcolor}"'
            f' style="user-select:none">{_trunc(lbl, 14)}</text>'
        )


    return "\n".join(parts)


def _svg_legend() -> str:
    """Pill-shaped legend at bottom-right — matches .legend CSS in design system."""
    return """  <!-- legend handled in HTML layer -->"""


# ── Public API ─────────────────────────────────────────────────────────────

def build_node_graph_html(concept: str, neighbors: list[dict]) -> str:
    """
    Build the premium node-graph HTML string from live neighbor data.
    Drop-in for gr.HTML() in the Connectome Explorer tab.

    Args:
        concept:   the central concept label (e.g. "bone")
        neighbors: list of dicts with keys: pre, post, rel, weight, evidence_count
    """

    # ── Deduplicate neighbors — keep highest weight per label ──────────────
    seen: dict[str, dict] = {}
    for n in neighbors:
        lbl = n["post"].split("::")[-1]
        if lbl == concept:
            lbl = n["pre"].split("::")[-1]
        if lbl == concept:
            continue
        rel = n.get("rel", "CO_OCCURS")
        w   = float(n.get("weight", 1.0))
        if lbl not in seen or w > seen[lbl]["weight"]:
            seen[lbl] = {"weight": w, "rel": rel}

    node_list = sorted(seen.items(), key=lambda x: x[1]["weight"], reverse=True)
    n_nodes   = len(node_list)

    if n_nodes == 0:
        return node_graph_empty(concept)

    max_w = max(v["weight"] for v in seen.values()) or 1.0

    # ── Radial layout ──────────────────────────────────────────────────────
    # Use same 720×380 canvas as design system
    # Inner ring (up to 6 nodes) at radius 110, outer at 170
    if n_nodes <= 6:
        rings = [(node_list, 155)]
    elif n_nodes <= 10:
        inner_n = min(6, n_nodes // 2 + 1)
        rings = [(node_list[:inner_n], 115), (node_list[inner_n:], 185)]
    else:
        rings = [(node_list[:6], 105), (node_list[6:], 175)]

    positions: dict[str, tuple[float, float]] = {}
    all_positions: list[tuple[float, float]] = []

    for ring_nodes, radius in rings:
        count = len(ring_nodes)
        for i, (lbl, _) in enumerate(ring_nodes):
            # Start from top (-π/2), go clockwise
            angle = (2 * math.pi * i / count) - math.pi / 2
            x = _CX + math.cos(angle) * radius
            y = _CY + math.sin(angle) * radius
            # Keep nodes within canvas bounds (accounting for labels ~80px wide)
            x = max(80, min(_W - 80, x))
            y = max(30, min(_H - 40, y))
            positions[lbl] = (x, y)
            all_positions.append((x, y))

    # ── Assemble SVG ───────────────────────────────────────────────────────
    total_nodes = n_nodes + 1

    svg_body = "\n".join([
        _svg_scatter(),
        _svg_web(all_positions),
        _svg_edges(node_list, positions),
        _svg_central(concept),
        _svg_nodes(node_list, positions, max_w),
    ])

    svg = (
        f'<svg viewBox="0 0 {_W} {_H}" width="100%" height="100%"'
        f' style="position:absolute;inset:0;display:block;">\n'
        f'{svg_body}\n'
        f'</svg>'
    )

    # ── Legend (HTML layer — matches .legend CSS exactly) ──────────────────
    legend_items = [("Evokes","#B8922A"), ("Primes","#1B3A5C"), ("Mentions","#6B8F71"), ("Co-occurs","#9C8E7E")]
    legend_dots = "".join(
        f'<div style="display:flex;align-items:center;gap:5px;'
        f'font-family:\'Cinzel\',serif;font-size:8.5px;letter-spacing:0.1em;'
        f'color:#6B5E4A;text-transform:uppercase;">'
        f'<span style="width:8px;height:8px;border-radius:50%;background:{c};'
        f'display:inline-block;flex-shrink:0;"></span>{name}</div>'
        for name, c in legend_items
    )

    return f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;500;600&family=Crimson+Pro:ital,wght@0,400;0,500;1,400&family=JetBrains+Mono:wght@300;400&display=swap');
.dvnc-graph-wrap {{
  position: relative;
  height: 380px;
  background: radial-gradient(ellipse at 50% 45%, #FFFDF7 0%, #F4EEE2 60%, #EAE0CF 100%);
  border: 1px solid rgba(184,146,42,0.22);
  border-radius: 12px;
  overflow: hidden;
}}
.dvnc-graph-wrap::before {{
  content: '';
  position: absolute;
  inset: 0;
  background-image:
    linear-gradient(rgba(184,146,42,0.03) 1px, transparent 1px),
    linear-gradient(90deg, rgba(184,146,42,0.03) 1px, transparent 1px);
  background-size: 40px 40px;
  pointer-events: none;
}}
.dvnc-graph-title {{
  position: absolute; top: 14px; left: 18px; z-index: 2;
  font-family: 'Cinzel', serif; font-size: 10.5px; font-weight: 500;
  letter-spacing: 0.14em; text-transform: uppercase; color: #4A4030;
}}
.dvnc-graph-meta {{
  position: absolute; top: 14px; right: 16px; z-index: 2;
  font-family: 'JetBrains Mono', monospace; font-size: 9.5px; color: #8B7B62;
}}
.dvnc-graph-legend {{
  position: absolute; bottom: 12px; right: 14px; z-index: 2;
  display: flex; gap: 12px;
  padding: 7px 12px;
  background: rgba(255,253,247,0.75);
  border: 1px solid rgba(184,146,42,0.15);
  border-radius: 999px;
}}
</style>
<div class="dvnc-graph-wrap">
  <div class="dvnc-graph-title">Connectome · Knowledge Graph</div>
  <div class="dvnc-graph-meta">{total_nodes} concepts</div>
  {svg}
  <div class="dvnc-graph-legend">{legend_dots}</div>
</div>"""


def node_graph_empty(concept: str) -> str:
    """Shown when a concept has no neighbors in the DB."""
    return f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;500&family=Crimson+Pro:ital,wght@0,400&display=swap');
</style>
<div style="
  position:relative; height:380px;
  background:radial-gradient(ellipse at 50% 45%,#FFFDF7 0%,#F4EEE2 60%,#EAE0CF 100%);
  border:1px solid rgba(184,146,42,0.22); border-radius:12px; overflow:hidden;
  display:flex; align-items:center; justify-content:center; flex-direction:column; gap:8px;
">
  <div style="font-family:'Cinzel',serif;font-size:10.5px;font-weight:500;
              letter-spacing:0.14em;text-transform:uppercase;color:#4A4030;">
    Connectome · Knowledge Graph</div>
  <div style="font-family:'Cinzel',serif;font-size:18px;font-weight:600;
              letter-spacing:0.15em;color:#8B7B62;">{concept.upper()}</div>
  <div style="font-family:'Crimson Pro',Georgia,serif;font-size:13px;
              font-style:italic;color:#B5A898;">No connections found in the database</div>
</div>"""


def node_graph_placeholder() -> str:
    """Default state before any concept has been explored."""
    return f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;500;600&family=Crimson+Pro:ital,wght@0,400&display=swap');
</style>
<div style="
  position:relative; height:380px;
  background:radial-gradient(ellipse at 50% 45%,#FFFDF7 0%,#F4EEE2 60%,#EAE0CF 100%);
  border:1px solid rgba(184,146,42,0.22); border-radius:12px; overflow:hidden;
">
  <div style="position:absolute;inset:0;pointer-events:none;
    background-image:linear-gradient(rgba(184,146,42,0.03) 1px,transparent 1px),
    linear-gradient(90deg,rgba(184,146,42,0.03) 1px,transparent 1px);
    background-size:40px 40px;"></div>
  <div style="position:absolute;top:14px;left:18px;
    font-family:'Cinzel',serif;font-size:10.5px;font-weight:500;
    letter-spacing:0.14em;text-transform:uppercase;color:#4A4030;">
    Connectome · Knowledge Graph</div>
  <svg viewBox="0 0 720 380" width="100%" height="100%"
       style="position:absolute;inset:0;display:block;">
    <!-- Scatter dots for atmosphere -->
    <circle cx="70"  cy="40"  r="3"   fill="#B8922A" opacity="0.18"/>
    <circle cx="230" cy="25"  r="2"   fill="#6B8F71" opacity="0.25"/>
    <circle cx="380" cy="35"  r="2.5" fill="#9C8E7E" opacity="0.3"/>
    <circle cx="660" cy="55"  r="3"   fill="#1B3A5C" opacity="0.2"/>
    <circle cx="90"  cy="340" r="2.5" fill="#B8922A" opacity="0.22"/>
    <circle cx="620" cy="335" r="3"   fill="#9C8E7E" opacity="0.25"/>
    <circle cx="30"  cy="200" r="2"   fill="#B8922A" opacity="0.2"/>
    <circle cx="700" cy="190" r="2.5" fill="#6B8F71" opacity="0.2"/>
    <!-- Central placeholder text -->
    <text x="360" y="185" text-anchor="middle"
      font-family="'Cinzel', serif" font-size="13" font-weight="500"
      fill="#8B7B62" letter-spacing="0.12em" opacity="0.6">KNOWLEDGE GRAPH</text>
    <text x="360" y="215" text-anchor="middle"
      font-family="'Crimson Pro', Georgia, serif" font-size="13"
      fill="#B5A898" font-style="italic">Enter a concept and click Explore</text>
  </svg>
</div>"""
