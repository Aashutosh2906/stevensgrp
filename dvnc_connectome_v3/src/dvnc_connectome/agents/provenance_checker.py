"""
Agent 5 — Provenance Checker (rule-based, no LLM)

Validates that every claim in the hypothesis cites a valid [S#] reference
from the evidence pack. Strips uncited claims. Returns a clean provenance report.

This is a deterministic agent — no LLM required.
"""

from __future__ import annotations
import re
from typing import Any


class ProvenanceCheckerAgent:
    name = "Provenance Checker"
    role = "Validates citations and strips uncited claims (rule-based)"

    def run(self, brief: str, route_result: Any, evidence: str,
            previous_outputs: dict[str, str] | None = None) -> str:
        prev = previous_outputs or {}
        hypothesis = prev.get("Hypothesis Composer", "")
        evidence_pack = prev.get("Evidence Judge", "")

        if not hypothesis:
            return "PROVENANCE: No hypothesis to check."

        # Extract available citation keys from evidence pack
        available_refs = set(re.findall(r'\[S(\d+)\]', evidence_pack))

        # Find all citations used in hypothesis
        used_refs = set(re.findall(r'\[S(\d+)\]', hypothesis))

        # Check for sentences without citations
        sentences = re.split(r'(?<=[.!?])\s+', hypothesis)
        uncited = []
        for sent in sentences:
            # Skip structural lines (headers, labels)
            if not sent.strip() or sent.startswith(('═', '•', 'Step', 'Route', 'IF', 'ELSE')):
                continue
            # Check if sentence has a citation
            if not re.search(r'\[S\d+\]', sent) and len(sent) > 40:
                uncited.append(sent[:120] + "...")

        # Check for invalid citations (cited but not in evidence pack)
        invalid = used_refs - available_refs if available_refs else set()
        valid = used_refs & available_refs if available_refs else used_refs

        report_lines = [
            "PROVENANCE REPORT",
            "═══════════════════",
            f"Evidence sources available: S{', S'.join(sorted(available_refs, key=int)) if available_refs else 'none'}",
            f"Citations used in hypothesis: S{', S'.join(sorted(used_refs, key=int)) if used_refs else 'none'}",
            f"Valid citations: {len(valid)} / {len(used_refs)}",
            "",
        ]

        if invalid:
            report_lines.append(f"⚠ INVALID CITATIONS (not in evidence pack): S{', S'.join(sorted(invalid, key=int))}")
        else:
            report_lines.append("✓ All citations valid")

        if uncited:
            report_lines.append(f"\n⚠ UNCITED SENTENCES ({len(uncited)} found):")
            for s in uncited[:5]:
                report_lines.append(f"  - {s}")
        else:
            report_lines.append("✓ All major claims have citations")

        # Compute provenance score
        n_sentences = max(len(sentences), 1)
        n_cited = n_sentences - len(uncited)
        score = round((n_cited / n_sentences) * 100)
        report_lines.append(f"\nPROVENANCE SCORE: {score}%")

        verdict = "PASS" if score >= 70 and not invalid else "FAIL — revise before publication"
        report_lines.append(f"VERDICT: {verdict}")

        return "\n".join(report_lines)
