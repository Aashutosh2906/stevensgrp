"""
DVNC.AI Gradio Application

Three tabs:
  1. Discovery Engine — full 6-agent pipeline with visible routing
  2. Connectome Explorer — browse the knowledge graph interactively
  3. Database Inspector — raw stats, node/synapse browser

Visual routing panel shows:
  - Primary route with relation types and scores
  - Alternative routes
  - Suppressed hub nodes
  - LMM activations
  - Spreading activation heatmap (text-based)
"""

from __future__ import annotations
import json
import os
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

    # Primary route — normalise score bars to max 10 blocks
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

    # Alternative routes
    if route_result.alternative_routes:
        lines.append("── ALTERNATIVE ROUTES ─────────────────────────────────")
        for i, alt in enumerate(route_result.alternative_routes, 1):
            alt_path = " → ".join(s.label for s in alt)
            lines.append(f"  Alt {i}: {alt_path}")
        lines.append("")

    # Suppressed hubs
    if route_result.suppressed_hubs:
        lines.append("── SUPPRESSED HUBS (hub suppression active) ──────────")
        for hub in route_result.suppressed_hubs[:5]:
            label = hub.split("::")[-1]
            lines.append(f"  ✕ {label}")
        lines.append("")

    # LMM Activations — normalise bars to max 20 blocks
    lmm = route_result.lmm_activations
    if lmm:
        lines.append("── DA VINCI MENTAL MODEL ACTIVATIONS ─────────────────")
        sorted_lmm = sorted(lmm.items(), key=lambda x: x[1], reverse=True)
        max_lmm = sorted_lmm[0][1] if sorted_lmm else 1
        max_lmm = max_lmm or 1
        for code, score in sorted_lmm[:5]:
            name = LMM_LABELS.get(code, code)
            norm = score / max_lmm
            bar = "█" * max(1, int(norm * 20))
            lines.append(f"  {code} {name:<28} {score:.2f} [{bar}]")
        lines.append("")

    # Spreading activation top nodes
    lines.append("── SPREADING ACTIVATION MAP (top 15 nodes) ───────────")
    max_score = max((s for _, s in route_result.activation_map[:15]), default=1) or 1
    for node_id, score in route_result.activation_map[:15]:
        label = node_id.split("::")[-1][:25]
        norm = score / max_score
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


# ── Main App Factory ───────────────────────────────────────────────────────


def make_app(db_path: str) -> gr.Blocks:
    db = NeuroGraphDB(db_path)
    router = DaVinciRouter(db)
    orchestrator = DVNCOrchestrator(db=db)

    # ─────────────────────────────────────────────────────────────────────
    # Tab 1: Discovery Engine
    # ─────────────────────────────────────────────────────────────────────
    with gr.Blocks(
        title="DVNC.AI — Brain-Inspired Design Discovery",
        theme=gr.themes.Soft(primary_hue="violet"),
    ) as app:
        gr.HTML("""
        <div style="text-align:center; padding:20px; background: linear-gradient(135deg, #1a0a2e, #16213e);
                    border-radius:12px; margin-bottom:20px;">
            <h1 style="color:#c084fc; font-size:2.2em; margin:0;">🧠 DVNC.AI</h1>
            <p style="color:#94a3b8; margin:8px 0 0;">Brain-Inspired Polymathic Design Discovery System</p>
            <p style="color:#64748b; font-size:0.85em;">
                Da Vinci Routing · 6-Agent Debate · Hebbian Connectome · Visible Reasoning
            </p>
        </div>
        """)

        with gr.Tabs():

            # ── Tab 1: Discovery Engine ───────────────────────────────────
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
                            label="Anthropic API Key (optional — enables Claude Opus synthesis)",
                            placeholder="sk-ant-...",
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

                    # Inject API key if provided
                    if api_key.strip():
                        os.environ["ANTHROPIC_API_KEY"] = api_key.strip()

                    try:
                        # Update router settings
                        router.steps = int(steps)
                        router.fanout = int(fanout)

                        # Run Da Vinci routing
                        route_result = router.route(brief)
                        route_panel_text = _format_route_panel(route_result)

                        # Run 6-agent pipeline
                        result = orchestrator.run(brief=brief, route_result=route_result)

                        log_text = _format_agent_log(result["agent_log"])
                        score = round(result["overall_score"] * 100)
                        status = f"✓ Complete | Score: {score}/100 | Agents: 6 | Evidence sources: {len(route_result.evidence_nodes)}"

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

            # ── Tab 2: Connectome Explorer ────────────────────────────────
            with gr.TabItem("🕸 Connectome Explorer"):
                gr.Markdown("""
                Browse the knowledge graph directly. Explore concepts, see their neighbours,
                and watch spreading activation propagate through the connectome.
                """)

                with gr.Row():
                    concept_input = gr.Textbox(
                        label="Concept to explore",
                        value="bone",
                        placeholder="e.g. carbon, muscle, topology, flow ...",
                    )
                    limit_slider = gr.Slider(5, 60, value=20, step=5, label="Neighbour limit")
                    explore_btn = gr.Button("Explore", variant="primary")

                with gr.Row():
                    neighbours_out = gr.Dataframe(
                        label="Strongest Synapses (neighbours)",
                        headers=["From", "To", "Relation", "Weight", "Evidence Count"],
                        interactive=False,
                    )

                gr.Markdown("### Spreading Activation")
                with gr.Row():
                    prop_steps = gr.Slider(1, 6, value=3, step=1, label="Propagation steps")
                    prop_fanout = gr.Slider(5, 40, value=15, step=5, label="Fanout")
                    prop_btn = gr.Button("Propagate", variant="secondary")

                propagation_out = gr.Dataframe(
                    label="Activation scores (top activated nodes)",
                    headers=["Node", "Label", "Score"],
                    interactive=False,
                )

                def explore_concept(concept, limit):
                    node_id = f"concept::{concept.strip().lower()}"
                    neighbors = db.top_neighbors(node_id, limit=int(limit))
                    if not neighbors:
                        # Try fuzzy search
                        matches = db.search_nodes(concept, kind="Concept", limit=5)
                        if matches:
                            node_id = f"concept::{matches[0]['label']}"
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

            # ── Tab 3: Database Inspector ─────────────────────────────────
            with gr.TabItem("📊 Database Inspector"):
                gr.Markdown("### Connectome Statistics")

                stats_btn = gr.Button("Refresh Stats", variant="secondary")
                stats_out = gr.JSON(label="Database Statistics")

                gr.Markdown("### Node Browser")
                with gr.Row():
                    node_kind = gr.Dropdown(
                        choices=["Concept", "Document", "Domain"],
                        value="Concept",
                        label="Node kind",
                    )
                    node_search = gr.Textbox(label="Search nodes (label contains)", placeholder="flow")
                    node_search_btn = gr.Button("Search", variant="secondary")

                nodes_out = gr.Dataframe(
                    label="Nodes",
                    headers=["ID", "Kind", "Label", "Domain"],
                    interactive=False,
                )

                gr.Markdown("### Top Synapses by Weight")
                top_syn_btn = gr.Button("Show Top Synapses", variant="secondary")
                syn_out = gr.Dataframe(
                    label="Top 50 synapses by weight",
                    headers=["From", "To", "Relation", "Weight", "Evidence", "LMM Tags"],
                    interactive=False,
                )

                def get_stats():
                    return db.stats()

                def search_nodes(kind, query):
                    nodes = db.search_nodes(query, kind=kind, limit=50)
                    rows = []
                    for n in nodes:
                        props = n.get("props", {})
                        if isinstance(props, str):
                            try:
                                props = json.loads(props)
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
                node_search_btn.click(fn=search_nodes, inputs=[node_kind, node_search], outputs=[nodes_out])
                top_syn_btn.click(fn=get_top_synapses, outputs=[syn_out])

                # Auto-load stats on startup
                app.load(fn=get_stats, outputs=[stats_out])

            # ── Tab 4: Add Papers ──────────────────────────────────────────
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
                        placeholder="e.g. 10.1002/adfm.201800618",
                    )
                    doi_domain = gr.Dropdown(
                        choices=["cardiac_biomaterials", "raman_spectroscopy",
                                 "conducting_polymers", "biomineralization",
                                 "biomechanics", "materials", "general"],
                        value="general",
                        label="Domain",
                    )
                    doi_fetch_btn = gr.Button("Fetch & Ingest", variant="primary")
                doi_status = gr.Textbox(label="Status", interactive=False)

                gr.Markdown("### Method 2: Search for Papers")
                with gr.Row():
                    search_input = gr.Textbox(
                        label="Search Query",
                        placeholder="e.g. auxetic cardiac patch conductive polymer",
                    )
                    search_max = gr.Slider(1, 20, value=5, step=1, label="Max results")
                    search_btn = gr.Button("Search & Ingest", variant="primary")
                search_status = gr.Textbox(label="Status", interactive=False)

                gr.Markdown("### Method 3: Paste Paper Directly")
                with gr.Row():
                    with gr.Column():
                        paste_title = gr.Textbox(label="Paper Title")
                        paste_authors = gr.Textbox(label="Authors")
                        paste_year = gr.Number(label="Year", value=2024)
                        paste_domain_sel = gr.Dropdown(
                            choices=["cardiac_biomaterials", "raman_spectroscopy",
                                     "conducting_polymers", "biomineralization",
                                     "biomechanics", "materials", "general"],
                            value="general",
                            label="Domain",
                        )
                    with gr.Column():
                        paste_text = gr.Textbox(
                            label="Abstract + Key Findings (paste full text here)",
                            lines=12,
                            placeholder="Paste the abstract, key findings, measurements, and methods...",
                        )
                        paste_url = gr.Textbox(label="URL / DOI (optional)")
                paste_btn = gr.Button("Save & Ingest", variant="primary")
                paste_status = gr.Textbox(label="Status", interactive=False)

                def fetch_doi(doi_val, domain_val):
                    if not doi_val.strip():
                        return "Please enter a DOI."
                    try:
                        from ..curation.stevens_loader import fetch_paper_by_doi
                        from ..curation.pipeline import ingest_docs
                        paper = fetch_paper_by_doi(doi_val.strip())
                        if not paper:
                            return f"Could not fetch paper for DOI: {doi_val}"
                        paper["domain"] = domain_val
                        ingest_docs(db, [paper], verbose=False)
                        return f"✓ Ingested: {paper['title']} ({len(paper['text'])} chars, domain: {domain_val})"
                    except Exception as e:
                        return f"Error: {e}"

                def search_and_ingest(query, max_results):
                    if not query.strip():
                        return "Please enter a search query."
                    try:
                        from ..curation.stevens_loader import search_papers
                        from ..curation.pipeline import ingest_docs
                        papers = search_papers(query.strip(), max_results=int(max_results))
                        if not papers:
                            return f"No papers found for: {query}"
                        ingest_docs(db, papers, verbose=False)
                        titles = [p['title'][:60] for p in papers]
                        return f"✓ Ingested {len(papers)} papers:\n" + "\n".join(f"  • {t}" for t in titles)
                    except Exception as e:
                        return f"Error: {e}"

                def paste_and_ingest(title, authors, year, domain, text, url):
                    if not title.strip() or not text.strip():
                        return "Please enter at least a title and text."
                    try:
                        from ..curation.stevens_loader import save_uploaded_paper
                        from ..curation.pipeline import ingest_docs
                        doc = save_uploaded_paper(
                            title=title.strip(),
                            text=text.strip(),
                            domain=domain,
                            authors=authors.strip(),
                            year=int(year) if year else None,
                            url=url.strip(),
                        )
                        ingest_docs(db, [doc], verbose=False)
                        return f"✓ Saved and ingested: {title} ({len(text)} chars, domain: {domain})"
                    except Exception as e:
                        return f"Error: {e}"

                doi_fetch_btn.click(
                    fn=fetch_doi,
                    inputs=[doi_input, doi_domain],
                    outputs=[doi_status],
                )
                search_btn.click(
                    fn=search_and_ingest,
                    inputs=[search_input, search_max],
                    outputs=[search_status],
                )
                paste_btn.click(
                    fn=paste_and_ingest,
                    inputs=[paste_title, paste_authors, paste_year,
                            paste_domain_sel, paste_text, paste_url],
                    outputs=[paste_status],
                )

    return app
