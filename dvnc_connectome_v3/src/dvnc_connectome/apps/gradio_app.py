"""
DVNC.AI Gradio Application
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
from .node_graph import build_node_graph_html, node_graph_empty, node_graph_placeholder


# ── Routing panel formatter ────────────────────────────────────────────────

def _format_route_panel(route_result) -> str:
    if route_result is None:
        return "No route computed."
    lines = []
    lines.append("╔══════════════════════════════════════════════════════╗")
    lines.append("║          DA VINCI ROUTING PANEL                     ║")
    lines.append("╚══════════════════════════════════════════════════════╝")
    lines.append("")
    lines.append("── PRIMARY ROUTE ──────────────────────────────────────")
    max_score = max((s.score for s in route_result.primary_route), default=1) or 1
    for i, step in enumerate(route_result.primary_route):
        norm = step.score / max_score
        bar = "█" * max(1, int(norm * 10)) + "░" * (10 - max(1, int(norm * 10)))
        rel = f" ─[{step.rel_from_prev}]→ " if step.rel_from_prev else ""
        cross = " ⟳ [CROSS-DOMAIN]" if step.is_cross_domain else ""
        lines.append(f"  {i:2d}. {rel}{step.label:<25} score={step.score:.3f} [{bar}]{cross}")
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
        lines.append("── SUPPRESSED HUBS ───────────────────────────────────")
        for hub in route_result.suppressed_hubs[:5]:
            lines.append(f"  ✕ {hub.split('::')[-1]}")
        lines.append("")
    lmm = route_result.lmm_activations
    if lmm:
        lines.append("── DA VINCI MENTAL MODEL ACTIVATIONS ─────────────────")
        for code, score in sorted(lmm.items(), key=lambda x: x[1], reverse=True)[:5]:
            name = LMM_LABELS.get(code, code)
            bar = "█" * max(1, int(score * 5)) + "░" * (5 - max(1, int(score * 5)))
            lines.append(f"  {code} {name:<28} {score:.2f} [{bar}]")
        lines.append("")
    lines.append("── SPREADING ACTIVATION MAP (top 15 nodes) ───────────")
    max_act = max((s for _, s in route_result.activation_map[:15]), default=1)
    for node_id, score in route_result.activation_map[:15]:
        label = node_id.split("::")[-1][:25]
        norm = score / max_act if max_act else 0
        bar = "█" * max(1, int(norm * 20)) + "░" * (20 - max(1, int(norm * 20)))
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


# ── Score helper ───────────────────────────────────────────────────────────

def _score_output(text: str) -> dict:
    citations    = len(re.findall(r'\[S\d+\]', text))
    numbers      = len(re.findall(r'\d+\.?\d*\s*(?:kPa|MPa|GPa|S/cm|nm|um|mm|cm|mg|mL|wt%|%|Hz|C)', text))
    steps        = len(re.findall(r'(?:Step\s*\d|IF\s|THEN\s|->)', text, re.IGNORECASE))
    words        = len(text.split())
    cit_score    = min(100, citations * 14)
    spec_score   = min(100, numbers * 10 + steps * 6)
    struct_score = min(100, steps * 12 + (20 if len(text) > 400 else 0))
    concision    = 100 if 120 <= words <= 600 else max(0, 100 - abs(words - 360) // 4)
    overall = int(0.35*cit_score + 0.30*spec_score + 0.20*struct_score + 0.15*concision)
    return {"citations": cit_score, "specificity": spec_score, "structure": struct_score,
            "concision": concision, "overall": overall}


def _build_score_table(dvnc_scores: dict, plain_scores: dict) -> str:
    dims = [("Citations","citations"),("Specificity","specificity"),
            ("Structure","structure"),("Concision","concision"),("OVERALL","overall")]
    lines = [f"{'DIMENSION':<14}  {'DVNC.AI':^16}  {'Plain LLM':^16}  WINNER", "─" * 62]
    for label, key in dims:
        d, p = dvnc_scores[key], plain_scores[key]
        d_bar = "█" * (d // 10) + "░" * (10 - d // 10)
        p_bar = "█" * (p // 10) + "░" * (10 - p // 10)
        winner = "<-- DVNC" if d > p else ("<-- LLM " if p > d else "  TIE  ")
        lines.append(f"{label:<14}  {d:3d} [{d_bar}]  {p:3d} [{p_bar}]  {winner}")
    return "\n".join(lines)


# ── Plain LLM ──────────────────────────────────────────────────────────────

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


# ── API key ────────────────────────────────────────────────────────────────

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


# ── DB helpers ─────────────────────────────────────────────────────────────

def _db_stats(db_path: str) -> str:
    try:
        conn = sqlite3.connect(db_path)
        cur  = conn.cursor()
        lines = ["╔══════════════════════════════════════════════╗",
                 "║         DVNC CONNECTOME — DB STATS          ║",
                 "╚══════════════════════════════════════════════╝", ""]
        for sql, label in [("SELECT COUNT(*) FROM nodes","Total Nodes"),
                            ("SELECT COUNT(*) FROM synapses","Total Synapses")]:
            try:
                cur.execute(sql)
                lines.append(f"  {label:<30} {cur.fetchone()[0]}")
            except Exception:
                pass
        for sql, header in [
            ("SELECT kind, COUNT(*) FROM nodes GROUP BY kind ORDER BY COUNT(*) DESC","Node breakdown:"),
            ("SELECT rel, COUNT(*) FROM synapses GROUP BY rel ORDER BY COUNT(*) DESC","Synapse breakdown:"),
        ]:
            try:
                cur.execute(sql)
                lines.append("")
                lines.append(f"  {header}")
                for row in cur.fetchall():
                    lines.append(f"    {row[0]:<26} {row[1]}")
            except Exception:
                pass
        conn.close()
        return "\n".join(lines)
    except Exception as e:
        return f"Stats error: {e}"


def _db_top_synapses(db_path: str, limit: int = 50) -> list:
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT pre, post, rel, weight, evidence_count, lmm_tags "
            "FROM synapses ORDER BY weight DESC LIMIT ?", (limit,)
        ).fetchall()
        conn.close()
        result = []
        for r in rows:
            try:
                tags_str = ", ".join(json.loads(r["lmm_tags"])[:2])
            except Exception:
                tags_str = ""
            result.append([r["pre"].split("::")[-1], r["post"].split("::")[-1],
                           r["rel"], round(r["weight"], 3), r["evidence_count"], tags_str])
        return result
    except Exception as e:
        return [[f"Error: {e}", "", "", "", "", ""]]


# ── App factory ────────────────────────────────────────────────────────────

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

            # ── Tab 1: Discovery Engine ───────────────────────────────────
            with gr.TabItem("Discovery Engine"):
                gr.Markdown("""
                Enter a design challenge. The system routes through the DVNC Connectome,
                runs 6 specialised AI agents in debate, and produces an evidence-anchored
                Innovation Card with full provenance.
                *After running, the Head-to-Head tab auto-populates.*
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
                            label="API Key (Groq / Gemini / DeepSeek / Anthropic)",
                            placeholder="gsk_... or AIza... or sk-ant-...",
                            type="password",
                        )
                        steps_slider  = gr.Slider(2, 6, value=4, step=1,  label="Routing Steps")
                        fanout_slider = gr.Slider(5, 40, value=20, step=5, label="Fanout per Step")
                        run_btn = gr.Button("Run DVNC Discovery", variant="primary", size="lg")

                route_panel = gr.Textbox(
                    label="Da Vinci Routing Panel",
                    lines=25, max_lines=35, interactive=False,
                )
                final_card = gr.Textbox(
                    label="Final Innovation Card",
                    lines=30, max_lines=50, interactive=False,
                )
                with gr.Accordion("Agent Debate Log (full 6-agent pipeline)", open=False):
                    agent_log_out = gr.Textbox(lines=40, max_lines=80, interactive=False)
                overall_score_out = gr.Number(label="Overall Innovation Score (0-100)")
                status_out        = gr.Textbox(label="Status", lines=2, interactive=False)

            # ── Tab 2: Head-to-Head ───────────────────────────────────────
            with gr.TabItem("Head-to-Head"):
                gr.HTML("""
                <div style="padding:10px 0 16px;">
                    <h3 style="margin:0 0 6px;">DVNC.AI vs Plain LLM</h3>
                    <p style="color:#64748b; margin:0; font-size:0.9em;">
                        Auto-populated when you run the Discovery Engine.
                        Same brief sent directly to the LLM — no routing, no graph, no agents.
                    </p>
                </div>
                """)
                h2h_brief_display = gr.Textbox(
                    label="Brief being compared", lines=2, interactive=False,
                    placeholder="Run a query in the Discovery Engine — results appear here automatically.",
                )
                h2h_scores = gr.Textbox(
                    label="Score Comparison  (Citations / Specificity / Structure / Concision / Overall)",
                    lines=9, interactive=False,
                )
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

            # ── Tab 3: Connectome Explorer ────────────────────────────────
            with gr.TabItem("Connectome Explorer"):
                gr.Markdown("""
                Explore concepts in the knowledge graph.
                Enter a concept to see its neighbours and the network diagram.
                """)
                with gr.Row():
                    concept_input = gr.Textbox(
                        label="Concept",
                        value="bone",
                        placeholder="e.g. auxetic, raman, scaffold, cardiomyocyte",
                    )
                    limit_slider = gr.Slider(5, 40, value=12, step=4, label="Max Neighbours")
                    explore_btn  = gr.Button("Explore", variant="primary")

                # ── Premium node graph — powered by node_graph.py ─────────
                graph_out = gr.HTML(node_graph_placeholder())

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

                def explore_concept(concept, limit):
                    concept   = concept.strip().lower()
                    node_id   = f"concept::{concept}"
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
                    # Use node_graph.py for the premium graph
                    graph = (
                        build_node_graph_html(concept, neighbors)
                        if neighbors
                        else node_graph_empty(concept)
                    )
                    return graph, rows

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

            # ── Tab 4: Database Inspector ─────────────────────────────────
            with gr.TabItem("Database Inspector"):
                gr.Markdown("### Connectome Statistics")
                stats_btn = gr.Button("Refresh Stats", variant="secondary")
                stats_out = gr.Textbox(label="Database Statistics", lines=20, interactive=False)
                with gr.Row():
                    node_kind       = gr.Dropdown(
                        choices=["all","concept","document","domain"],
                        value="all", label="Node Kind",
                    )
                    node_search     = gr.Textbox(
                        label="Search nodes (label contains)",
                        placeholder="e.g. cardiac",
                    )
                    node_search_btn = gr.Button("Search Nodes", variant="secondary")
                nodes_out   = gr.Dataframe(headers=["ID","Kind","Label","Domain"], interactive=False)
                top_syn_btn = gr.Button("Show Top 50 Synapses by Weight", variant="secondary")
                syn_out     = gr.Dataframe(
                    headers=["From","To","Relation","Weight","Evidence","LMM Tags"],
                    interactive=False,
                )

                def get_stats():
                    return _db_stats(db_path)

                def search_nodes(kind, search_term):
                    nodes = db.search_nodes(
                        query=search_term or "",
                        kind=None if kind == "all" else kind,
                        limit=100,
                    )
                    rows = []
                    for nd in nodes:
                        props = nd.get("props", {})
                        if isinstance(props, str):
                            try:
                                props = json.loads(props)
                            except Exception:
                                props = {}
                        rows.append([nd["id"], nd["kind"], nd["label"],
                                     props.get("domain", props.get("domain_hint", ""))])
                    return rows

                def get_top_synapses():
                    return _db_top_synapses(db_path, limit=50)

                stats_btn.click(fn=get_stats, outputs=[stats_out])
                node_search_btn.click(
                    fn=search_nodes, inputs=[node_kind, node_search], outputs=[nodes_out]
                )
                top_syn_btn.click(fn=get_top_synapses, outputs=[syn_out])

            # ── Tab 5: Add Papers ─────────────────────────────────────────
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
                    search_query_input = gr.Textbox(
                        label="Search Query", placeholder="e.g. auxetic cardiac patch biomaterial"
                    )
                    search_papers_btn = gr.Button("Search & Ingest Top 3", variant="secondary")
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
                        return f"✓ Ingested {len(docs)} document(s) for DOI {doi}"
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
                        ingest_docs(db, [{
                            "doc_id": f"manual_{hash(title) % 99999:05d}",
                            "title":  title.strip(), "text":   text.strip(),
                            "source": source.strip() or "manual",
                            "domain": domain.strip() or "general",
                        }], verbose=False)
                        return f"✓ Ingested: {title}"
                    except Exception as e: return f"Error: {e}"

                doi_btn.click(fn=fetch_doi, inputs=[doi_input], outputs=[doi_out])
                search_papers_btn.click(fn=search_papers, inputs=[search_query_input], outputs=[search_out])
                paste_btn.click(
                    fn=paste_paper,
                    inputs=[paste_title, paste_text, paste_source, paste_domain],
                    outputs=[paste_out],
                )

        # ── Discovery Engine click — writes Tab 1 + Tab 2 ────────────────
        def run_discovery(brief, api_key, steps, fanout):
            if not brief.strip():
                return (
                    "No route", "Please enter a design brief.", "", 0,
                    "Enter a brief first.",
                    "", "", "", 0, "", 0,
                )
            if api_key.strip():
                _set_api_key(api_key)

            # DVNC pipeline
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
                status = (
                    f"Complete | Score: {pipeline_score}/100 | Agents: 6 | "
                    f"Evidence sources: {sources} | Head-to-Head tab updated"
                )
            except Exception as e:
                route_panel_text = f"Error: {e}"
                log_text = dvnc_card = f"[DVNC error: {e}]"
                pipeline_score = 0
                status = f"Error: {e}"

            # Plain LLM
            try:
                plain_card = _call_plain_llm(brief)
            except Exception as e:
                plain_card = f"[Plain LLM error: {e}]"

            # Scores
            dvnc_scores  = _score_output(dvnc_card)
            plain_scores = _score_output(plain_card)
            dvnc_scores["overall"] = max(dvnc_scores["overall"], pipeline_score)
            score_table = _build_score_table(dvnc_scores, plain_scores)

            return (
                # Tab 1
                route_panel_text, dvnc_card, log_text, pipeline_score, status,
                # Tab 2
                brief, score_table,
                dvnc_card, dvnc_scores["overall"],
                plain_card, plain_scores["overall"],
            )

        run_btn.click(
            fn=run_discovery,
            inputs=[brief_input, api_key_input, steps_slider, fanout_slider],
            outputs=[
                route_panel, final_card, agent_log_out, overall_score_out, status_out,
                h2h_brief_display, h2h_scores,
                dvnc_out, dvnc_score_display,
                plain_out, plain_score_display,
            ],
        )

    return app
