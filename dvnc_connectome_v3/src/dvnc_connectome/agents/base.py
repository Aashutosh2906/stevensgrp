"""
Base agent class and shared utilities.

All agents receive:
  - A RouteResult from the Da Vinci Router
  - The evidence pack (text snippets from source documents)
  - A brief (the user's problem statement)

All agents return structured text that gets passed to the next agent.
"""

from __future__ import annotations
import os
from typing import Any


def _call_claude(system: str, user: str, model: str = "claude-opus-4-5",
                 max_tokens: int = 1200) -> str:
    """
    Call Anthropic Claude API. Falls back to a template if no API key.
    Set ANTHROPIC_API_KEY in your environment or .env file.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return f"[API key not set — provide ANTHROPIC_API_KEY to enable AI synthesis]\n\n{user[:500]}"

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return message.content[0].text
    except Exception as e:
        return f"[Claude API error: {e}]"


class BaseAgent:
    name: str = "BaseAgent"
    role: str = ""
    model: str = "claude-opus-4-5"

    def run(self, brief: str, route_result: Any, evidence: str,
            previous_outputs: dict[str, str] | None = None) -> str:
        raise NotImplementedError
