"""
Agent 3 — Hypothesis Composer (Claude Opus)

The synthesis heart of DVNC. Reads the evidence pack and generates a novel
innovation hypothesis using ONLY evidence from the pack.

Output follows the 4-part Da Vinci Innovation Card:
  1. Cross-Domain Leap
  2. Evidence-Anchored Hypothesis
  3. Experimental Programme (3–5 conditional steps)
  4. Translation Lens (IP, commercial, manufacturability)
"""

from .base import BaseAgent, _call_claude
from typing import Any

_SYSTEM = """You are the Hypothesis Composer for DVNC.AI — a brain-inspired polymathic design system.

You read the evidence pack assembled from a domain-spanning connectome and generate
a genuinely novel design innovation hypothesis.

RULES (non-negotiable):
1. Every sentence in your hypothesis MUST end with a [S#] citation from the evidence pack.
2. Do NOT introduce any knowledge not present in the evidence pack.
3. The cross-domain leap must connect at least TWO different domains from the route.
4. The experimental programme must have 3–5 steps with conditional logic (IF/THEN).
5. Every measurement must have units.

Output EXACTLY this structure:

═══════════════════════════════════════════
DVNC INNOVATION CARD
═══════════════════════════════════════════

CROSS-DOMAIN LEAP:
[Describe the structural analogy connecting two non-obvious domains]
Route path: [Concept A] → [EVOKES/PRIMES] → [Concept B] → [domain bridge] → [Concept C]

HYPOTHESIS:
[2–3 sentences. Every sentence ends with [S#].]

EXPERIMENTAL PROGRAMME:
Step 1: [Action with measurable threshold] → IF [condition]: proceed to Step 2 | ELSE: [fallback]
Step 2: [Action with measurable threshold] → IF [condition]: proceed to Step 3 | ELSE: [fallback]
Step 3: [Action with measurable threshold] → IF [condition]: proceed to Step 4 | ELSE: stop
[Step 4 and 5 if needed]

Each step cites [S#].

TRANSLATION LENS:
• IP Opportunity: [specific white space, cite [S#]]
• Manufacturability: [assessment with constraint, cite [S#]]
• Commercial Hook: [one sentence value proposition]
• Prior Art Risk: [what to check]

═══════════════════════════════════════════"""


class HypothesisComposerAgent(BaseAgent):
    name = "Hypothesis Composer"
    role = "Generates evidence-anchored innovation hypotheses"
    model = "claude-opus-4-5"  # Best model for this critical synthesis step

    def run(self, brief: str, route_result: Any, evidence: str,
            previous_outputs: dict[str, str] | None = None) -> str:
        prev = previous_outputs or {}
        framing = prev.get("Problem Framer", "")
        evidence_pack = prev.get("Evidence Judge", evidence)

        route_trace = ""
        if route_result:
            steps = route_result.primary_route
            route_trace = " → ".join(
                f"{s.label}[{s.rel_from_prev}]" if s.rel_from_prev else s.label
                for s in steps
            )
            alt_routes = []
            for i, alt in enumerate(route_result.alternative_routes[:2]):
                alt_trace = " → ".join(
                    f"{s.label}[{s.rel_from_prev}]" if s.rel_from_prev else s.label
                    for s in alt
                )
                alt_routes.append(f"Alternative {i+1}: {alt_trace}")
            if alt_routes:
                route_trace += "\n" + "\n".join(alt_routes)

        user_msg = f"""Design Brief:
{brief}

Problem Framing:
{framing}

Da Vinci Routing Path:
{route_trace}

Cross-domain leaps in this route: {route_result.cross_domain_count if route_result else 0}
Novelty score: {route_result.novelty_score if route_result else 0}

EVIDENCE PACK:
{evidence_pack}

Generate the DVNC Innovation Card following the strict template.
Use ONLY information from the evidence pack. Cite every sentence."""

        return _call_claude(_SYSTEM, user_msg, max_tokens=1400)
