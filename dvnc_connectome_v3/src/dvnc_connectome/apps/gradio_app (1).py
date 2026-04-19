"""
DVNC.AI Gradio Application

Four tabs:
  1. Discovery Engine  — full 6-agent pipeline with visible routing
  2. Head-to-Head      — DVNC.AI vs plain LLM side-by-side comparison
  3. Connectome Explorer — browse the knowledge graph interactively
  4. Database Inspector  — raw stats, node/synapse browser
"""

from __future__ import annotations
import json
import os
import re
from pathlib import Path

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
        lines.append(
            f"  {i:2d}. {rel}{step.label:<25} score={step.score:.3f} [{score_bar}]{cross}"
        )
        if step.lmm_tags:
            lmm_names = [LMM_LABELS.get(t, t) for t in step.lmm_tags[:2]]
            lines.append(f"       LMM: {', '.join(lmm_names)}")

    lines.append("")

    if route_result.alternative_routes:
        lines.append("── ALTERNATIVE ROUTES ─────────────────────────────────")
        for i, alt in enumerate(route_result.alternative_routes, 1):
            alt_path = " → ".join(s.label for s in alt)
            lines.append(f"  Alt {i}: {alt_path}")
        lines.append("")

    if route_result.suppressed_hubs:
        lines.append("── SUPPRESSED HUBS (hub suppression active) ──────────")
        for hub in route_result.suppressed_hubs[:5]:
            label = hub.split("::")[-1]
            lines.append(f"  ✕ {label}")
        lines.append("")

    lmm = route_result.lmm_activations
    if lmm:
        lines.append("── DA VINCI MENTAL MODEL ACTIVATIONS ─────────────────")
        sorted_lmm = sorted(lmm.items(), key=lambda x: x[1], reverse=True)
        for code, score in sorted_lmm[:5]:
            name = LMM_LABELS.get(code, code)
            bar = "█" * max(1, int(score * 5))
            lines.append(f"  {code} {name:<28} {score:.2f} [{bar}]")
        lines.append("")

    lines.append("── SPREADING ACTIVATION MAP (top 15 nodes) ───────────")
    max_score = max((s for _, s in route_result.activation_map[:15]), default=1)
    for node_id, score in route_result.activation_map[:15]:
        label = node_id.split("::")[-1][:25]
        norm = score / max_score if max_score else 0
        bar = "█" * max(1, int(norm * 20))
        lines.append(f"  {label:<26} {score:.4f} [{bar}]")

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


def _call_plain_llm(brief: str) -> str:
    """
    Send the brief directly to the LLM with no knowledge graph, no routing,
    no evidence pack. This is the 'vanilla LLM' baseline for Head-to-Head.
    Uses whatever LLM backend is configured in agents/base.py via env vars.
    """
    system = (
        "You are a research scientist and innovation consultant. "
        "Given a design brief, produce a detailed innovation proposal. "
        "Include: a cross-domain insight, a specific testable hypothesis, "
        "a step-by-step experimental programme, and a commercial/IP lens. "
        "Be as specific as possible. Cite any papers you can recall by name."
    )
    try:
        # Reuse whichever LLM backend is active (Groq / Gemini / DeepSeek)
        from ..agents.base import _call_claude
        return _call_claude(system, brief, max_tokens=1200)
    except Exception as e:
        return f"[Plain LLM error: {e}]"


def _score_output(text: str) -> dict:
    """
    Heuristic scoring of an output text on four dimensions (0–100 each).
    Returns dict with keys: citations, specificity, structure, overall.
    """
    citations = len(re.findall(r'\[S\d+\]', text))
    numbers   = len(re.findall(r'\d+\.?\d*\s*(?:kPa|MPa|GPa|S/cm|nm|μm|mm|cm|mg|mL|wt%|%|Hz|°C)', text))
    steps     = len(re.findall(r'(?:Step\s*\d|IF\s|THEN\s|→)', text, re.IGNORECASE))
    words     = len(text.split())

    cit_score  = min(100, citations * 14)
    spec_score = min(100, numbers   * 10 + steps * 6)
    struct_score = min(100, steps * 12 + (20 if len(text) > 400 else 0))
    concision  = 100 if 120 <= words <= 600 else max(0, 100 - abs(words - 360) // 4)

    overall = int(
        0.35 * cit_score +
        0.30 * spec_score +
        0.20 * struct_score +
        0.15 * concision
    )
    return {
        "citations":   cit_score,
        "specificity": spec_score,
        "structure":   struct_score,
        "concision":   concision,
        "overall":     overall,
    }


def _render_score_bars(dvnc_scores: dict, plain_scores: dict) -> str:
    """Render a text-based score comparison table."""
    dims = [
        ("Citations",   "citations"),
        ("Specificity", "specificity"),
        ("Structure",   "structure"),
        ("Concision",   "concision"),
        ("OVERALL",     "overall"),
    ]
    lines = []
    lines.append("╔══════════════════════════════════════════════════════════════╗")
    lines.append("║  DIMENSION        DVNC.AI  ████████░░  Plain LLM            ║")
    lines.append("╠══════════════════════════════════════════════════════════════╣")
    for label, key in dims:
        d = dvnc_scores[key]
        p = plain_scores[key]
        d_bar = "█" * (d // 10) + "░" * (10 - d // 10)
        p_bar = "█" * (p // 10) + "░" * (10 - p // 10)
        winner = " ◀ DVNC" if d > p else (" ◀ LLM" if p > d else " TIE")
        lines.append(f"  {label:<14} {d:3d}  [{d_bar}]  {p:3d}  [{p_bar}]{winner}")
    lines.append("╚══════════════════════════════════════════════════════════════╝")
    return "\n".join(lines)


# ── Main App Factory ───────────────────────────────────────────────────────


def make_app(db_path: str) -> gr.Blocks:
    db = NeuroGraphDB(db_path)
    router = DaVinciRouter(db)
    orchestrator = DVNCOrchestrator(db=db)

    with gr.Blocks(
        title="DVNC.AI — Brain-Inspired Design Discovery",
        theme=gr.themes.Soft(primary_hue="violet"),
    ) as app:

        gr.HTML("""
        <div style="text-align:center; padding:20px;
                    background: linear-gradient(135deg, #1a0a2e, #16213e);
                    border-radius:12px; margin-bottom:20px;">
            <h1 style="color:#c084fc; font-size:2.2em; margin:0;">🧠 DVNC.AI</h1>
            <p style="color:#94a3b8; margin:8px 0 0;">
                Brain-Inspired Polymathic Design Discovery System
            </p>
            <p style="color:#64748b; font-size:0.85em;">
                Da Vinci Routing · 6-Agent Debate · Hebbian Connectome · Visible Reasoning
            </p>
        </div>
        """)

        with gr.Tabs():

            # ══════════════════════════════════════════════════════════════
            # TAB 1 — Discovery Engine
            # ══════════════════════════════════════════════════════════════
            with gr.TabItem("🔬 Discovery Engine"):
                gr.Markdown("""
                Enter a design challenge. The system routes through the DVNC Connectome,
                runs 6 specialised AI agents in debate, and produces an evidence-anchored
                Innovation Card with full provenance.
                """)

                with gr.Row():
                    with gr.Column(scale=2):
                        brief_input = gr.Textbox(
                            label="Design Brief",
                            placeholder=(
                                "e.g. Design a lightweight load-bearing structure inspired by "
                                "biological tissue architecture that can be additively manufactured "
                                "and adapts its stiffness under varying loads"
                            ),
                            lines=4,
                        )
                    with gr.Column(scale=1):
                        api_key_input = gr.Textbox(
                            label="API Key (optional — enables AI synthesis)",
                            placeholder="Paste your Groq / Gemini / DeepSeek key",
                            type="password",
                        )
                        steps_slider = gr.Slider(2, 6, value=4, step=1, label="Routing Steps")
                        fanout_slider = gr.Slider(5, 40, value=20, step=5, label="Fanout per Step")
                        run_btn = gr.Button("▶ Run DVNC Discovery", variant="primary", size="lg")

                with gr.Row():
                    route_panel = gr.Textbox(
                        label="Da Vinci Routing Panel (explicit visible routing)",
                        lines=25, max_lines=35,
                        interactive=False,
                    )

                with gr.Row():
                    final_card = gr.Textbox(
                        label="Final Innovation Card",
                        lines=30, max_lines=50,
                        interactive=False,
                    )

                with gr.Accordion("Agent Debate Log (full 6-agent pipeline)", open=False):
                    agent_log_out = gr.Textbox(
                        label="Agent-by-Agent Debate",
                        lines=40, max_lines=80,
                        interactive=False,
                    )

                overall_score_out = gr.Number(label="Overall Innovation Score (0–100)")
                status_out = gr.Textbox(label="Status", lines=2, interactive=False)

                def run_discovery(brief, api_key, steps, fanout):
                    if not brief.strip():
                        return "No route", "Please enter a design brief.", "", 0, "Enter a brief first."

                    if api_key.strip():
                        # Detect key type and set correct env var
                        key = api_key.strip()
                        if key.startswith("gsk_"):
                            os.environ["GROQ_API_KEY"] = key
                        elif key.startswith("AIza"):
                            os.environ["GEMINI_API_KEY"] = key
                        elif key.startswith("sk-ant"):
                            os.environ["ANTHROPIC_API_KEY"] = key
                        else:
                            # DeepSeek or other OpenAI-compatible
                            os.environ["DEEPSEEK_API_KEY"] = key

                    try:
                        router.steps = int(steps)
                        router.fanout = int(fanout)

                        route_result = router.route(brief)
                        route_panel_text = _format_route_panel(route_result)

                        result = orchestrator.run(brief=brief, route_result=route_result)

                        log_text = _format_agent_log(result["agent_log"])
                        score = round(result["overall_score"] * 100)
                        status = (
                            f"✓ Complete | Score: {score}/100 | "
                            f"Agents: 6 | Evidence sources: {len(route_result.evidence_nodes)}"
                        )

                        return (
                            route_panel_text,
                            result["final_card"],
                            log_text,
                            score,
                            status,
                        )
                    except Exception as e:
                        return f"Error: {e}", "", "", 0, f"Error: {e}"

                run_btn.click(
                    fn=run_discovery,
                    inputs=[brief_input, api_key_input, steps_slider, fanout_slider],
                    outputs=[route_panel, final_card, agent_log_out, overall_score_out, status_out],
                )

            # ══════════════════════════════════════════════════════════════
            # TAB 2 — Head-to-Head Comparison
            # ══════════════════════════════════════════════════════════════
            with gr.TabItem("⚔️ Head-to-Head"):

                gr.HTML("""
                <div style="padding:14px 0 4px;">
                    <h3 style="margin:0 0 6px;font-size:1.1em;">DVNC.AI vs Plain LLM</h3>
                    <p style="color:#64748b;font-size:0.9em;margin:0;">
                        Same brief. Same model. One goes through the Da Vinci Connectome
                        with 6 specialised agents and a knowledge graph.
                        The other goes directly to the LLM with no routing, no evidence pack.
                        See the difference provenance makes.
                    </p>
                </div>
                """)

                with gr.Row():
                    h2h_brief = gr.Textbox(
                        label="Design Brief",
                        placeholder=(
                            "e.g. Design a novel composite biomaterial for myocardial infarction "
                            "combining auxetic mechanics, electrical conductivity, and immunomodulation"
                        ),
                        lines=3,
                    )

                with gr.Row():
                    with gr.Column(scale=1):
                        h2h_api_key = gr.Textbox(
                            label="API Key",
                            placeholder="Paste your Groq / Gemini / DeepSeek key",
                            type="password",
                        )
                    with gr.Column(scale=1):
                        h2h_steps = gr.Slider(2, 6, value=4, step=1, label="DVNC Routing Steps")
                    with gr.Column(scale=1):
                        h2h_fanout = gr.Slider(5, 40, value=20, step=5, label="DVNC Fanout")
                    with gr.Column(scale=1):
                        h2h_run_btn = gr.Button(
                            "⚔️ Run Head-to-Head", variant="primary", size="lg"
                        )

                h2h_status = gr.Textbox(label="Status", lines=1, interactive=False)

                # ── Score comparison bar ──────────────────────────────────
                h2h_scores = gr.Textbox(
                    label="Score Comparison (Citations · Specificity · Structure · Concision · Overall)",
                    lines=10,
                    interactive=False,
                )

                # ── Side-by-side outputs ──────────────────────────────────
                gr.HTML("<hr style='margin:18px 0;border-color:#e2e8f0'>")

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.HTML("""
                        <div style="background:#1a0a2e;border-radius:8px 8px 0 0;
                                    padding:10px 16px;display:flex;align-items:center;
                                    justify-content:space-between;">
                            <div>
                                <span style="color:#c084fc;font-weight:600;font-size:1em;">
                                    🧠 DVNC.AI
                                </span>
                                <span style="color:#64748b;font-size:0.78em;margin-left:10px;">
                                    Da Vinci Routing · 6-Agent Debate · Connectome-grounded
                                </span>
                            </div>
                        </div>
                        """)
                        dvnc_out = gr.Textbox(
                            label="",
                            lines=30,
                            max_lines=50,
                            interactive=False,
                            show_label=False,
                        )
                        dvnc_score_display = gr.Number(
                            label="DVNC.AI Overall Score (0–100)"
                        )

                    with gr.Column(scale=1):
                        gr.HTML("""
                        <div style="background:#f1f5f9;border-radius:8px 8px 0 0;
                                    padding:10px 16px;display:flex;align-items:center;
                                    justify-content:space-between;">
                            <div>
                                <span style="color:#334155;font-weight:600;font-size:1em;">
                                    🤖 Plain LLM
                                </span>
                                <span style="color:#94a3b8;font-size:0.78em;margin-left:10px;">
                                    Same model · No knowledge graph · No citations · No routing
                                </span>
                            </div>
                        </div>
                        """)
                        plain_out = gr.Textbox(
                            label="",
                            lines=30,
                            max_lines=50,
                            interactive=False,
                            show_label=False,
                        )
                        plain_score_display = gr.Number(
                            label="Plain LLM Overall Score (0–100)"
                        )

                # ── Route trace (collapsed) ───────────────────────────────
                with gr.Accordion("DVNC.AI Routing Trace", open=False):
                    h2h_route = gr.Textbox(
                        label="Da Vinci Routing Panel",
                        lines=20,
                        interactive=False,
                    )

                with gr.Accordion("DVNC.AI Agent Debate Log", open=False):
                    h2h_agent_log = gr.Textbox(
                        label="6-Agent Pipeline Log",
                        lines=30,
                        interactive=False,
                    )

                # ── Handler ───────────────────────────────────────────────
                def run_h2h(brief, api_key, steps, fanout):
                    if not brief.strip():
                        return (
                            "Please enter a design brief.",
                            "", "", 0, 0,
                            "No route", "",
                        )

                    if api_key.strip():
                        key = api_key.strip()
                        if key.startswith("gsk_"):
                            os.environ["GROQ_API_KEY"] = key
                        elif key.startswith("AIza"):
                            os.environ["GEMINI_API_KEY"] = key
                        elif key.startswith("sk-ant"):
                            os.environ["ANTHROPIC_API_KEY"] = key
                        else:
                            os.environ["DEEPSEEK_API_KEY"] = key

                    # ── Step 1: DVNC.AI ──────────────────────────────────
                    try:
                        router.steps = int(steps)
                        router.fanout = int(fanout)

                        route_result = router.route(brief)
                        route_text = _format_route_panel(route_result)

                        dvnc_result = orchestrator.run(
                            brief=brief, route_result=route_result
                        )
                        dvnc_card = dvnc_result["final_card"]
                        dvnc_log  = _format_agent_log(dvnc_result["agent_log"])
                        dvnc_pipeline_score = round(dvnc_result["overall_score"] * 100)

                    except Exception as e:
                        dvnc_card = f"[DVNC error: {e}]"
                        dvnc_log  = ""
                        route_text = f"Error: {e}"
                        dvnc_pipeline_score = 0
                        route_result = None

                    # ── Step 2: Plain LLM ────────────────────────────────
                    try:
                        plain_card = _call_plain_llm(brief)
                    except Exception as e:
                        plain_card = f"[Plain LLM error: {e}]"

                    # ── Step 3: Score both ───────────────────────────────
                    dvnc_scores  = _score_output(dvnc_card)
                    plain_scores = _score_output(plain_card)

                    # Blend pipeline score with heuristic score for DVNC
                    dvnc_scores["overall"] = max(
                        dvnc_scores["overall"],
                        dvnc_pipeline_score,
                    )

                    score_bars = _render_score_bars(dvnc_scores, plain_scores)

                    sources = len(route_result.evidence_nodes) if route_result else 0
                    status = (
                        f"✓ Complete | DVNC: {dvnc_scores['overall']}/100 | "
                        f"Plain LLM: {plain_scores['overall']}/100 | "
                        f"Evidence sources: {sources}"
                    )

                    return (
                        status,
                        score_bars,
                        dvnc_card,
                        plain_card,
                        dvnc_scores["overall"],
                        plain_scores["overall"],
                        route_text,
                        dvnc_log,
                    )

                h2h_run_btn.click(
                    fn=run_h2h,
                    inputs=[h2h_brief, h2h_api_key, h2h_steps, h2h_fanout],
                    outputs=[
                        h2h_status,
                        h2h_scores,
                        dvnc_out,
                        plain_out,
                        dvnc_score_display,
                        plain_score_display,
                        h2h_route,
                        h2h_agent_log,
                    ],
                )

            # ══════════════════════════════════════════════════════════════
            # TAB 3 — Connectome Explorer
            # ══════════════════════════════════════════════════════════════
            with gr.TabItem("🕸 Connectome Explorer"):
                gr.Markdown("""
                Browse the knowledge graph directly. Explore concepts, see their neighbours,
                and watch spreading activation propagate through the connectome.
                """)

                with gr.Row():
                    concept_input = gr.Textbox(
                        label="Concept to explore",
                        value="bone",
                        placeholder="e.g. auxetic, raman, scaffold, cardiomyocyte",
                    )
                    limit_slider = gr.Slider(5, 50, value=20, step=5, label="Max neighbours")
                    explore_btn = gr.Button("Explore", variant="secondary")

                neighbours_out = gr.Dataframe(
                    headers=["From", "To", "Relation", "Weight", "Evidence Count"],
                    label="Top Neighbours",
                    interactive=False,
                )

                with gr.Row():
                    prop_steps = gr.Slider(1, 6, value=3, step=1, label="Propagation Steps")
                    prop_fanout = gr.Slider(5, 30, value=15, step=5, label="Fanout")
                    prop_btn = gr.Button("Propagate Activation", variant="secondary")

                propagation_out = gr.Dataframe(
                    headers=["Node ID", "Label", "Activation Score"],
                    label="Spreading Activation Result",
                    interactive=False,
                )

                def explore_concept(concept, limit):
                    node_id = f"concept::{concept.strip().lower()}"
                    neighbors = db.top_neighbors(node_id, limit=int(limit))
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
                    return rows

                def propagate_concept(concept, steps, fanout):
                    node_id = f"concept::{concept.strip().lower()}"
                    results = db.propagate(node_id, steps=int(steps), fanout=int(fanout))
                    rows = [
                        [r[0], r[0].split("::")[-1], round(r[1], 5)]
                        for r in results[:30]
                    ]
                    return rows

                explore_btn.click(
                    fn=explore_concept,
                    inputs=[concept_input, limit_slider],
                    outputs=[neighbours_out],
                )
                prop_btn.click(
                    fn=propagate_concept,
                    inputs=[concept_input, prop_steps, prop_fanout],
                    outputs=[propagation_out],
                )

            # ══════════════════════════════════════════════════════════════
            # TAB 4 — Database Inspector
            # ══════════════════════════════════════════════════════════════
            with gr.TabItem("📊 Database Inspector"):
                gr.Markdown("### Connectome Statistics")

                stats_btn = gr.Button("Refresh Stats", variant="secondary")
                stats_out = gr.Textbox(
                    label="Database Statistics",
                    lines=20,
                    interactive=False,
                )

                with gr.Row():
                    node_kind = gr.Dropdown(
                        choices=["all", "concept", "document", "domain"],
                        value="all",
                        label="Node Kind",
                    )
                    node_search = gr.Textbox(
                        label="Search nodes (label contains)",
                        placeholder="e.g. cardiac",
                    )
                    node_search_btn = gr.Button("Search Nodes", variant="secondary")

                nodes_out = gr.Dataframe(
                    headers=["ID", "Kind", "Label", "Domain"],
                    label="Nodes",
                    interactive=False,
                )

                top_syn_btn = gr.Button("Show Top 50 Synapses", variant="secondary")
                syn_out = gr.Dataframe(
                    headers=["From", "To", "Relation", "Weight", "Evidence", "LMM Tags"],
                    label="Top Synapses by Weight",
                    interactive=False,
                )

                def get_stats():
                    stats = db.get_stats()
                    lines = [
                        "╔══════════════════════════════════════════════╗",
                        "║         DVNC CONNECTOME — DB STATS          ║",
                        "╚══════════════════════════════════════════════╝",
                        "",
                    ]
                    for k, v in stats.items():
                        lines.append(f"  {k:<30} {v}")
                    return "\n".join(lines)

                def search_nodes(kind, search_term):
                    nodes = db.search_nodes(
                        kind=None if kind == "all" else kind,
                        label_contains=search_term or None,
                        limit=100,
                    )
                    rows = []
                    for n in nodes:
                        try:
                            props = json.loads(n.get("props", "{}"))
                        except Exception:
                            props = {}
                        rows.append([
                            n["id"],
                            n["kind"],
                            n["label"],
                            props.get("domain", props.get("domain_hint", "")),
                        ])
                    return rows

                def get_top_synapses():
                    syns = db.get_all_synapses(limit=50)
                    rows = [
                        [
                            s["pre"].split("::")[-1],
                            s["post"].split("::")[-1],
                            s["rel"],
                            round(s["weight"], 3),
                            s["evidence_count"],
                            ", ".join(s.get("lmm_tags", [])[:2]),
                        ]
                        for s in syns
                    ]
                    return rows

                stats_btn.click(fn=get_stats, outputs=[stats_out])
                node_search_btn.click(
                    fn=search_nodes,
                    inputs=[node_kind, node_search],
                    outputs=[nodes_out],
                )
                top_syn_btn.click(fn=get_top_synapses, outputs=[syn_out])

                app.load(fn=get_stats, outputs=[stats_out])

            # ══════════════════════════════════════════════════════════════
            # TAB 5 — Add Papers
            # ══════════════════════════════════════════════════════════════
            with gr.TabItem("📄 Add Papers"):
                gr.Markdown("""
                ### Add Papers to the Connectome

                Add new papers to enrich the knowledge graph. Three methods:
                1. **Fetch by DOI** — enter a DOI and we pull title + abstract from Semantic Scholar
                2. **Search** — search for papers by topic and add the best matches
                3. **Paste directly** — paste the title, abstract, and key details of a paper

                New papers are immediately ingested into the live connectome.
                """)

                gr.Markdown("### Method 1: Fetch by DOI")
                with gr.Row():
                    doi_input = gr.Textbox(
                        label="DOI",
                        placeholder="e.g. 10.1126/sciadv.1601007",
                    )
                    doi_btn = gr.Button("Fetch & Ingest", variant="secondary")
                doi_out = gr.Textbox(label="Result", lines=4, interactive=False)

                gr.Markdown("### Method 2: Search by Topic")
                with gr.Row():
                    search_query_input = gr.Textbox(
                        label="Search Query",
                        placeholder="e.g. auxetic cardiac patch biomaterial",
                    )
                    search_papers_btn = gr.Button("Search & Ingest Top 3", variant="secondary")
                search_out = gr.Textbox(label="Result", lines=6, interactive=False)

                gr.Markdown("### Method 3: Paste Directly")
                with gr.Row():
                    with gr.Column():
                        paste_title  = gr.Textbox(label="Title", lines=1)
                        paste_text   = gr.Textbox(label="Abstract / Key Text", lines=5)
                        paste_source = gr.Textbox(label="Source / Author", value="manual")
                        paste_domain = gr.Textbox(label="Domain", value="general")
                        paste_btn    = gr.Button("Ingest Paper", variant="secondary")
                paste_out = gr.Textbox(label="Result", lines=3, interactive=False)

                def fetch_doi(doi):
                    try:
                        from ..curation.pipeline import ingest_docs
                        from ..curation.fetchers import fetch_by_doi
                        docs = fetch_by_doi(doi)
                        if not docs:
                            return f"No data found for DOI: {doi}"
                        ingest_docs(db, docs, verbose=False)
                        return f"✓ Ingested {len(docs)} document(s) for DOI {doi}"
                    except Exception as e:
                        return f"Error: {e}"

                def search_papers(query):
                    try:
                        from ..curation.pipeline import ingest_docs
                        from ..curation.fetchers import fetch_openalex
                        docs = fetch_openalex(query, max_results=3)
                        if not docs:
                            return f"No results for: {query}"
                        ingest_docs(db, docs, verbose=False)
                        titles = "\n".join(f"  • {d.get('title','?')}" for d in docs)
                        return f"✓ Ingested {len(docs)} paper(s):\n{titles}"
                    except Exception as e:
                        return f"Error: {e}"

                def paste_paper(title, text, source, domain):
                    try:
                        from ..curation.pipeline import ingest_docs
                        if not title.strip() or not text.strip():
                            return "Title and text are required."
                        doc = {
                            "doc_id": f"manual_{hash(title) % 99999:05d}",
                            "title":  title.strip(),
                            "text":   text.strip(),
                            "source": source.strip() or "manual",
                            "domain": domain.strip() or "general",
                        }
                        ingest_docs(db, [doc], verbose=False)
                        return f"✓ Ingested: {title}"
                    except Exception as e:
                        return f"Error: {e}"

                doi_btn.click(fn=fetch_doi, inputs=[doi_input], outputs=[doi_out])
                search_papers_btn.click(
                    fn=search_papers, inputs=[search_query_input], outputs=[search_out]
                )
                paste_btn.click(
                    fn=paste_paper,
                    inputs=[paste_title, paste_text, paste_source, paste_domain],
                    outputs=[paste_out],
                )

    return app
