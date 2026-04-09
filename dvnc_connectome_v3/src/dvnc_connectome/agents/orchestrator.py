"""
Agent 6 — Orchestrator (Claude Opus)

The final synthesis layer. Reads all 5 agent outputs and:
  1. Decides which version of the hypothesis to keep
  2. Addresses the adversarial critique
  3. Produces a final refined Innovation Output
  4. Scores the output (Novelty, Feasibility, Provenance, Commercial)
  5. If score > 0.75: triggers auto-reinforcement in the ConnectomeDB

This is the highest-value agent. It has full context of the entire debate.
"""

from __future__ import annotations
import re
from typing import Any

from .base import BaseAgent, _call_claude
from .problem_framer import ProblemFramerAgent
from .evidence_judge import EvidenceJudgeAgent
from .hypothesis_composer import HypothesisComposerAgent
from .adversarial_reviewer import AdversarialReviewerAgent
from .provenance_checker import ProvenanceCheckerAgent

_SYSTEM = """You are the Orchestrator Agent for DVNC.AI — the world's most rigorous
polymathic design discovery system.

You have read the full multi-agent debate:
  - The Problem Framer structured the challenge
  - The Evidence Judge assembled and scored the source evidence
  - The Hypothesis Composer generated a novel innovation candidate
  - The Adversarial Reviewer attacked it rigorously
  - The Provenance Checker validated all citations

Your job is to produce the FINAL REFINED OUTPUT — incorporating the critique,
fixing uncited claims, strengthening weak steps, and delivering the best possible
innovation card that a researcher could act on immediately.

STRICT RULES:
1. Every claim MUST cite [S#] from the evidence pack.
2. Address every HIGH severity critique from the reviewer.
3. Keep the cross-domain leap if it is valid; discard it if the review exposes it as non-novel.
4. The experimental programme must be executable — no vague steps.
5. Score the output honestly on all four dimensions.

Output EXACTLY this structure:

╔══════════════════════════════════════════╗
║     DVNC FINAL INNOVATION CARD          ║
╚══════════════════════════════════════════╝

BRIEF: [one sentence problem statement]
ROUTE: [primary routing path used]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CROSS-DOMAIN LEAP
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[The non-obvious structural analogy connecting two domains. Every sentence cites [S#].]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REFINED HYPOTHESIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[2-3 sentences. Every sentence cites [S#]. Incorporates reviewer corrections.]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXPERIMENTAL PROGRAMME
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 1 [cite S#]: [action with measurable threshold] 
  → IF [measured condition met]: go to Step 2
  → ELSE: [specific fallback]
Step 2 [cite S#]: ...
Step 3 [cite S#]: ...

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TRANSLATION LENS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• IP White Space [cite S#]: [specific gap]
• Manufacturability [cite S#]: [assessment]
• Commercial Hook: [one value proposition sentence]
• Prior Art: [what to search before filing]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITIQUE ADDRESSED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[How each HIGH severity critique was resolved]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT SCORES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Novelty:      [0–100]  [brief justification]
Feasibility:  [0–100]  [brief justification]
Provenance:   [0–100]  [brief justification]
Commercial:   [0–100]  [brief justification]
OVERALL:      [0–100]  = average of above

PLASTICITY: [YES / NO — should this reinforce the connectome?]"""


class DVNCOrchestrator:
    """
    Runs the full 6-agent DVNC pipeline and returns a complete result dict.
    """

    def __init__(self, db=None):
        self.db = db
        self.agents = [
            ProblemFramerAgent(),
            EvidenceJudgeAgent(),
            HypothesisComposerAgent(),
            AdversarialReviewerAgent(),
            ProvenanceCheckerAgent(),
        ]

    def run(self, brief: str, route_result: Any) -> dict:
        """
        Execute all 6 agents in sequence and return full structured result.
        """
        # Build evidence text from route result
        evidence = self._build_evidence_text(route_result)

        outputs: dict[str, str] = {}
        agent_log: list[dict] = []

        # Agents 1–5
        for agent in self.agents:
            print(f"[orchestrator] Running Agent: {agent.name}...")
            output = agent.run(
                brief=brief,
                route_result=route_result,
                evidence=evidence,
                previous_outputs=outputs,
            )
            outputs[agent.name] = output
            agent_log.append({"agent": agent.name, "role": agent.role, "output": output})

        # Agent 6: Orchestrator final synthesis
        print("[orchestrator] Running Agent: Orchestrator (Final Synthesis)...")
        final_output = self._final_synthesis(brief, route_result, evidence, outputs)
        agent_log.append({"agent": "Orchestrator", "role": "Final synthesis and scoring", "output": final_output})

        # Extract overall score
        overall_score = self._extract_score(final_output)

        # Auto-reinforce if score is high
        if self.db and overall_score >= 0.75:
            concept_pairs = self._extract_concept_pairs(route_result)
            self.db.auto_reinforce_from_output(concept_pairs, score=overall_score)
            print(f"[orchestrator] Auto-reinforced connectome (score={overall_score:.2f})")

        return {
            "brief": brief,
            "route_summary": route_result.summary() if route_result else "",
            "agent_outputs": outputs,
            "agent_log": agent_log,
            "final_card": final_output,
            "overall_score": overall_score,
            "evidence": evidence,
        }

    def _build_evidence_text(self, route_result: Any) -> str:
        """Convert evidence nodes from RouteResult into readable text pack.

        Retrieves the actual document text from the database so agents
        can cite specific findings, not just titles.
        """
        if not route_result or not route_result.evidence_nodes:
            return "No evidence retrieved from connectome."

        lines = []
        for i, ev in enumerate(route_result.evidence_nodes[:8], 1):
            lines.append(
                f"[S{i}] Title: {ev.get('title', 'Unknown')}\n"
                f"     Domain: {ev.get('domain', 'unknown')} | "
                f"Source: {ev.get('source', 'unknown')}\n"
                f"     Relevance score: {ev.get('relevance_score', 0):.3f}"
            )
            if ev.get("url"):
                lines.append(f"     URL: {ev['url']}")

            # Pull actual document text from DB for richer evidence
            if self.db:
                doc_node = self.db.get_node(ev.get("node_id", ""))
                if doc_node:
                    props = doc_node.get("props", {})
                    if isinstance(props, str):
                        import json
                        try:
                            props = json.loads(props)
                        except Exception:
                            props = {}
                    # Include a text snippet if available
                    text = props.get("text", "")
                    if text:
                        snippet = text[:500] + ("..." if len(text) > 500 else "")
                        lines.append(f"     Content: {snippet}")

            lines.append("")

        return "\n".join(lines)

    def _final_synthesis(self, brief: str, route_result: Any,
                          evidence: str, prev_outputs: dict[str, str]) -> str:
        all_outputs = "\n\n".join(
            f"=== {name} ===\n{text}"
            for name, text in prev_outputs.items()
        )
        route_summary = route_result.summary() if route_result else "No route"
        route_path = ""
        if route_result:
            route_path = " → ".join(
                f"{s.label}" for s in route_result.primary_route
            )

        user_msg = f"""Design Brief: {brief}

ROUTE PATH: {route_path}
{route_summary}

EVIDENCE PACK:
{evidence}

ALL AGENT OUTPUTS:
{all_outputs}

Now produce the FINAL REFINED DVNC Innovation Card addressing all critiques."""

        return _call_claude(_SYSTEM, user_msg, max_tokens=1600)

    def _extract_score(self, text: str) -> float:
        """Extract OVERALL score from final output."""
        m = re.search(r'OVERALL[:\s]+(\d+)', text)
        if m:
            try:
                return int(m.group(1)) / 100.0
            except ValueError:
                pass
        return 0.5

    def _extract_concept_pairs(self, route_result: Any) -> list[tuple[str, str]]:
        if not route_result:
            return []
        route = route_result.primary_route
        pairs = []
        for i in range(len(route) - 1):
            a = route[i].label
            b = route[i + 1].label
            pairs.append((a, b))
        return pairs
