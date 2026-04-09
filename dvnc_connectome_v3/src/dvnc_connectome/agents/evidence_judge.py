"""
Agent 2 — Evidence Judge

Takes the route result and evidence nodes, scores and ranks evidence snippets,
then assembles a clean evidence pack with confidence ratings.

Each snippet must be traceable to a source document.
"""

from .base import BaseAgent, _call_claude
from typing import Any

_SYSTEM = """You are the Evidence Judge for DVNC.AI.

Your job is to read a set of source snippets retrieved from the DVNC Connectome
and assess which are most relevant to the design brief. You score each snippet
on two dimensions:

  RELEVANCE (0-10): how directly it addresses the design brief
  CREDIBILITY (0-10): how specific, measurable, and source-grounded it is

Output EXACTLY this structure:

EVIDENCE PACK (ranked by relevance × credibility):

[S1] Source: [document title / domain]
Snippet: [key passage, max 2 sentences]
Relevance: [X/10] | Credibility: [Y/10]
Reason: [one sentence why this matters]

[S2] ...
[S3] ...
[S4] ...
[S5] ...

EVIDENCE GAPS:
[What important evidence is missing from this pack?]

CONFIDENCE SUMMARY:
Overall evidence strength for this query: [LOW / MEDIUM / HIGH]
Reasoning: [one sentence]

Do NOT add any information not present in the provided snippets.
Every claim in later agents MUST cite a [S#] from this pack."""


class EvidenceJudgeAgent(BaseAgent):
    name = "Evidence Judge"
    role = "Scores, ranks, and assembles the evidence pack"

    def run(self, brief: str, route_result: Any, evidence: str,
            previous_outputs: dict[str, str] | None = None) -> str:
        framing = (previous_outputs or {}).get("Problem Framer", "")
        user_msg = f"""Design Brief:
{brief}

Problem Framing:
{framing}

Evidence snippets retrieved from DVNC Connectome:
{evidence}

Route summary:
{route_result.summary() if route_result else "No route available"}

Assess, score, and assemble the evidence pack."""

        return _call_claude(_SYSTEM, user_msg, max_tokens=900)
