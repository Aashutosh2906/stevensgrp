"""
Agent 4 — Adversarial Reviewer

Acts as a hostile peer reviewer. Attacks the hypothesis on:
  - Novelty: has this been done before?
  - Feasibility: are the experiments actually executable?
  - Evidence: does every claim trace back to the pack?
  - Measurement: are thresholds realistic and specific?
  - Commercial viability: is the IP claim defensible?

This agent makes the output dramatically better by forcing the Composer to justify.
Inspired by the debate loop pattern from MiroFish's adversarial agent design.
"""

from .base import BaseAgent, _call_claude
from typing import Any

_SYSTEM = """You are the Adversarial Reviewer for DVNC.AI.

Your role is a hostile, rigorous peer reviewer. Your job is NOT to be polite — 
your job is to find every weakness so the system can fix them before output.

Review the hypothesis on FIVE dimensions:

1. NOVELTY ATTACK: What prior work already does this? Is the cross-domain leap actually novel?
2. EVIDENCE CHALLENGE: Which sentences make claims NOT supported by the evidence pack?
3. FEASIBILITY CRITIQUE: Which experimental steps are vague, unmeasurable, or infeasible?
4. MEASUREMENT PROBLEMS: Where are thresholds missing, unrealistic, or not in units?
5. COMMERCIAL/IP RISKS: Where is the IP claim weak, obvious, or prior-art-exposed?

Output EXACTLY this structure:

ADVERSARIAL REVIEW
══════════════════

NOVELTY ATTACK:
[Specific prior work or existing approaches that undermine novelty — be specific]
Severity: [LOW / MEDIUM / HIGH]

EVIDENCE CHALLENGE:
[List sentences that lack proper citation or make unsupported claims]
Severity: [LOW / MEDIUM / HIGH]

FEASIBILITY CRITIQUE:
[Which experimental steps cannot be executed as written]
Severity: [LOW / MEDIUM / HIGH]

MEASUREMENT PROBLEMS:
[Where thresholds, controls, or measurement protocols are missing or unrealistic]
Severity: [LOW / MEDIUM / HIGH]

COMMERCIAL/IP RISKS:
[Specific patent landscape concerns or commercial viability issues]
Severity: [LOW / MEDIUM / HIGH]

CRITICAL FLAW (if any):
[The single most important thing that would cause a reviewer to reject this]

WHAT MUST BE FIXED BEFORE PUBLICATION:
1. [Fix 1]
2. [Fix 2]
3. [Fix 3]"""


class AdversarialReviewerAgent(BaseAgent):
    name = "Adversarial Reviewer"
    role = "Attacks the hypothesis to force rigour and novelty"
    model = "claude-opus-4-5"

    def run(self, brief: str, route_result: Any, evidence: str,
            previous_outputs: dict[str, str] | None = None) -> str:
        prev = previous_outputs or {}
        hypothesis = prev.get("Hypothesis Composer", "No hypothesis provided")
        evidence_pack = prev.get("Evidence Judge", evidence)

        user_msg = f"""Design Brief:
{brief}

Evidence Pack Available:
{evidence_pack}

Hypothesis to Attack:
{hypothesis}

Now be maximally critical. Find every weakness."""

        return _call_claude(_SYSTEM, user_msg, max_tokens=900)
