"""
Agent 1 — Problem Framer

Converts a free-text design brief into structured design variables:
  - Core challenge statement
  - Design constraints (hard)
  - Target performance metrics
  - Domains to bridge
  - Da Vinci mental model priors
"""

from .base import BaseAgent, _call_claude
from typing import Any

_SYSTEM = """You are the Problem Framing Agent for DVNC.AI, a brain-inspired design discovery system.

Your job is to take a design brief and decompose it into structured variables that will guide
the rest of the multi-agent discovery pipeline.

Output EXACTLY this structure (no other text before or after):

CORE CHALLENGE:
[One sentence stating the fundamental design problem]

DESIGN CONSTRAINTS:
- [Hard constraint 1]
- [Hard constraint 2]
- [Hard constraint 3]

TARGET METRICS:
- [Measurable outcome 1 with units]
- [Measurable outcome 2 with units]

DOMAINS TO BRIDGE:
- [Domain 1] ↔ [Domain 2]: [why this bridge is relevant]
- [Domain 3] ↔ [Domain 4]: [why this bridge is relevant]

DA VINCI MENTAL MODELS TO ACTIVATE:
- [LMM name]: [how it applies here]
- [LMM name]: [how it applies here]

SEED CONCEPTS FOR ROUTING:
[comma-separated list of 5-8 key concept terms]

Be precise. Be specific. Every constraint must be checkable. Every metric must be measurable."""


class ProblemFramerAgent(BaseAgent):
    name = "Problem Framer"
    role = "Decomposes the design brief into structured variables"

    def run(self, brief: str, route_result: Any, evidence: str,
            previous_outputs: dict[str, str] | None = None) -> str:
        user_msg = f"""Design Brief:
{brief}

Route found these activated concepts: {', '.join(
    s.label for s in (route_result.primary_route[:6] if route_result else [])
)}

Da Vinci models activated: {', '.join(
    f"{k}: {v:.2f}" for k, v in list(
        (route_result.lmm_activations if route_result else {}).items()
    )[:4]
)}

Frame this problem precisely."""

        return _call_claude(_SYSTEM, user_msg, max_tokens=700)
