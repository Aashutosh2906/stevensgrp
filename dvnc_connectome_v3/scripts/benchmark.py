#!/usr/bin/env python3
"""
DVNC.AI Head-to-Head Benchmark
================================

Runs the same design brief through:
  1. DVNC.AI (full 6-agent pipeline with connectome routing)
  2. GPT-4o (raw, no knowledge graph)
  3. Claude 3.5 Sonnet (raw, no knowledge graph)

Then scores each output on 6 dimensions using a blind evaluator.

Usage:
    python scripts/benchmark.py \
        --db ./data/dvnc.db \
        --openai-key sk-... \
        --anthropic-key sk-ant-... \
        [--brief "Design a novel composite biomaterial..."] \
        [--output ./benchmark_results.json]

Requirements:
    pip install openai anthropic
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ── Benchmark Queries ────────────────────────────────────────────────────────

DEFAULT_BRIEF = (
    "Design a novel composite biomaterial for myocardial infarction combining "
    "auxetic mechanics, electrical conductivity, and immunomodulation"
)

# Additional test briefs for a broader benchmark
ADDITIONAL_BRIEFS = [
    "Design a Raman spectroscopy-based quality control system for real-time "
    "monitoring of conducting polymer cardiac patch manufacturing",

    "Propose a biomineralization-inspired approach to create electrically "
    "conductive bone scaffolds with graded porosity for load-bearing applications",

    "Design a wearable biosensor combining SERS-based molecular detection "
    "with auxetic substrate mechanics for continuous cardiac biomarker monitoring",
]


# ── Raw Model Calls ──────────────────────────────────────────────────────────

def call_gpt4o(brief: str, api_key: str) -> str:
    """Run the brief through GPT-4o with no knowledge graph."""
    try:
        from openai import OpenAI
    except ImportError:
        return "[SKIPPED] openai package not installed. pip install openai"

    client = OpenAI(api_key=api_key)

    system = """You are an expert materials scientist and biomedical engineer.
Given a design brief, produce a detailed innovation proposal including:
1. A cross-domain insight connecting at least two fields
2. A specific, testable hypothesis
3. A step-by-step experimental programme with decision gates
4. Key prior art to differentiate from
5. Translation potential (IP, manufacturability, commercial hook)

Be specific. Cite real papers, real measurements, real materials.
Do NOT make vague claims. Every number should be justified."""

    t0 = time.time()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": brief},
        ],
        max_tokens=2000,
        temperature=0.7,
    )
    elapsed = time.time() - t0
    text = response.choices[0].message.content
    return f"[GPT-4o | {elapsed:.1f}s]\n\n{text}"


def call_claude(brief: str, api_key: str) -> str:
    """Run the brief through Claude 3.5 Sonnet with no knowledge graph."""
    try:
        import anthropic
    except ImportError:
        return "[SKIPPED] anthropic package not installed. pip install anthropic"

    client = anthropic.Anthropic(api_key=api_key)

    system = """You are an expert materials scientist and biomedical engineer.
Given a design brief, produce a detailed innovation proposal including:
1. A cross-domain insight connecting at least two fields
2. A specific, testable hypothesis
3. A step-by-step experimental programme with decision gates
4. Key prior art to differentiate from
5. Translation potential (IP, manufacturability, commercial hook)

Be specific. Cite real papers, real measurements, real materials.
Do NOT make vague claims. Every number should be justified."""

    t0 = time.time()
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        system=system,
        messages=[{"role": "user", "content": brief}],
    )
    elapsed = time.time() - t0
    text = response.content[0].text
    return f"[Claude Sonnet | {elapsed:.1f}s]\n\n{text}"


def call_dvnc(brief: str, db_path: str) -> str:
    """Run the brief through the full DVNC.AI pipeline."""
    from dvnc_connectome.db.neurographdb import NeuroGraphDB
    from dvnc_connectome.routing.davinci_router import DaVinciRouter
    from dvnc_connectome.agents.orchestrator import DVNCOrchestrator
    from dvnc_connectome.apps.gradio_app import _format_route_panel

    db = NeuroGraphDB(db_path)
    router = DaVinciRouter(db, steps=4, fanout=20)
    orchestrator = DVNCOrchestrator(db=db)

    t0 = time.time()
    route_result = router.route(brief)
    result = orchestrator.run(brief=brief, route_result=route_result)
    elapsed = time.time() - t0

    route_panel = _format_route_panel(route_result)
    final_card = result["final_card"]

    return (
        f"[DVNC.AI | {elapsed:.1f}s | Score: {result['overall_score']:.2f}]\n\n"
        f"{route_panel}\n\n{final_card}"
    )


# ── Blind Evaluator ──────────────────────────────────────────────────────────

EVALUATOR_SYSTEM = """You are a senior research review panel evaluating three
innovation proposals for the SAME design brief. You do NOT know which system
produced which output — they are labelled Output A, Output B, Output C.

Score EACH output on these 6 dimensions (0–100):

1. SPECIFICITY: Are materials, measurements, and parameters named precisely?
   (100 = every claim has specific numbers, materials, conditions)
   (0 = vague generalities like "a conductive polymer")

2. NOVELTY: Does the proposal identify a genuinely unexplored combination?
   (100 = identifies a gap no existing paper fills, with evidence)
   (0 = restates known approaches without differentiation)

3. EVIDENCE GROUNDING: Are claims backed by cited real sources?
   (100 = every claim traces to a named paper/measurement)
   (0 = no citations, or citations appear fabricated)

4. EXPERIMENTAL RIGOUR: Could a PhD student execute the protocol tomorrow?
   (100 = exact steps, reagents, equipment, decision gates, controls)
   (0 = hand-wavy "future work" steps)

5. INTELLECTUAL HONESTY: Does it flag what it doesn't know?
   (100 = explicitly marks speculation vs evidence, flags gaps)
   (0 = presents speculation as established fact)

6. CROSS-DOMAIN INSIGHT: Does it make a non-obvious connection between fields?
   (100 = structural analogy between distant domains, justified)
   (0 = stays within a single discipline)

Output EXACTLY this JSON:
{
  "evaluations": {
    "A": {"specificity": X, "novelty": X, "evidence": X, "rigour": X, "honesty": X, "cross_domain": X, "total": X, "strengths": "...", "weaknesses": "..."},
    "B": {"specificity": X, "novelty": X, "evidence": X, "rigour": X, "honesty": X, "cross_domain": X, "total": X, "strengths": "...", "weaknesses": "..."},
    "C": {"specificity": X, "novelty": X, "evidence": X, "rigour": X, "honesty": X, "cross_domain": X, "total": X, "strengths": "...", "weaknesses": "..."}
  },
  "ranking": ["X", "Y", "Z"],
  "commentary": "..."
}

Where "total" is the average of the 6 scores. Be harsh but fair. Do not inflate."""


def run_blind_evaluation(brief: str, outputs: dict[str, str], api_key: str) -> dict:
    """Run blind evaluation using Claude as judge."""
    try:
        import anthropic
    except ImportError:
        return {"error": "anthropic not installed"}

    client = anthropic.Anthropic(api_key=api_key)

    # Shuffle outputs so the evaluator can't guess from order
    import random
    labels = ["A", "B", "C"]
    systems = list(outputs.keys())
    random.shuffle(systems)
    label_map = dict(zip(labels, systems))

    user_msg = f"""DESIGN BRIEF: {brief}

=== OUTPUT A ===
{outputs[label_map['A']]}

=== OUTPUT B ===
{outputs[label_map['B']]}

=== OUTPUT C ===
{outputs[label_map['C']]}

Score all three outputs. Return ONLY the JSON."""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1500,
        system=EVALUATOR_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
    )

    text = response.content[0].text

    # Parse JSON from response
    try:
        # Find JSON in response
        start = text.index("{")
        end = text.rindex("}") + 1
        result = json.loads(text[start:end])
    except (ValueError, json.JSONDecodeError):
        result = {"raw": text}

    # Map labels back to system names
    result["label_map"] = {k: v for k, v in label_map.items()}
    return result


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DVNC.AI Head-to-Head Benchmark")
    parser.add_argument("--db", default="./data/dvnc.db", help="DVNC database path")
    parser.add_argument("--openai-key", default=None, help="OpenAI API key")
    parser.add_argument("--anthropic-key", default=None, help="Anthropic API key")
    parser.add_argument("--brief", default=DEFAULT_BRIEF, help="Design brief to test")
    parser.add_argument("--all-briefs", action="store_true", help="Run all test briefs")
    parser.add_argument("--output", default="./benchmark_results.json", help="Output path")
    parser.add_argument("--skip-eval", action="store_true", help="Skip blind evaluation")
    args = parser.parse_args()

    # Resolve API keys from args or environment
    openai_key = args.openai_key or os.environ.get("OPENAI_API_KEY", "")
    anthropic_key = args.anthropic_key or os.environ.get("ANTHROPIC_API_KEY", "")

    briefs = [args.brief]
    if args.all_briefs:
        briefs.extend(ADDITIONAL_BRIEFS)

    all_results = []

    for i, brief in enumerate(briefs):
        print(f"\n{'='*60}")
        print(f"BENCHMARK {i+1}/{len(briefs)}")
        print(f"Brief: {brief[:80]}...")
        print(f"{'='*60}\n")

        outputs = {}

        # 1. DVNC.AI
        print("[1/3] Running DVNC.AI...")
        try:
            outputs["DVNC.AI"] = call_dvnc(brief, args.db)
            print("  ✓ DVNC.AI complete")
        except Exception as e:
            outputs["DVNC.AI"] = f"[ERROR] {e}"
            print(f"  ✗ DVNC.AI error: {e}")

        # 2. GPT-4o
        if openai_key:
            print("[2/3] Running GPT-4o...")
            try:
                outputs["GPT-4o"] = call_gpt4o(brief, openai_key)
                print("  ✓ GPT-4o complete")
            except Exception as e:
                outputs["GPT-4o"] = f"[ERROR] {e}"
                print(f"  ✗ GPT-4o error: {e}")
        else:
            outputs["GPT-4o"] = "[SKIPPED] No OpenAI API key provided"
            print("[2/3] Skipping GPT-4o (no API key)")

        # 3. Claude Sonnet
        if anthropic_key:
            print("[3/3] Running Claude Sonnet...")
            try:
                outputs["Claude"] = call_claude(brief, anthropic_key)
                print("  ✓ Claude complete")
            except Exception as e:
                outputs["Claude"] = f"[ERROR] {e}"
                print(f"  ✗ Claude error: {e}")
        else:
            outputs["Claude"] = "[SKIPPED] No Anthropic API key provided"
            print("[3/3] Skipping Claude (no API key)")

        # Print all outputs
        for name, output in outputs.items():
            print(f"\n{'─'*40}")
            print(f"  {name}")
            print(f"{'─'*40}")
            print(output[:2000])
            if len(output) > 2000:
                print(f"\n  ... ({len(output)} chars total)")

        # Blind evaluation
        evaluation = None
        if not args.skip_eval and anthropic_key:
            print("\n[EVAL] Running blind evaluation...")
            try:
                evaluation = run_blind_evaluation(brief, outputs, anthropic_key)
                print("  ✓ Evaluation complete")

                # Print summary
                if "evaluations" in evaluation:
                    print("\n  SCORES:")
                    label_map = evaluation.get("label_map", {})
                    for label, scores in evaluation["evaluations"].items():
                        system = label_map.get(label, label)
                        total = scores.get("total", "?")
                        print(f"    {system:12s}: {total}/100")
                    print(f"\n  RANKING: {evaluation.get('ranking', '?')}")
                    print(f"  Commentary: {evaluation.get('commentary', '')[:200]}")
            except Exception as e:
                evaluation = {"error": str(e)}
                print(f"  ✗ Evaluation error: {e}")

        all_results.append({
            "brief": brief,
            "outputs": outputs,
            "evaluation": evaluation,
        })

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n[benchmark] Results saved to: {output_path}")


if __name__ == "__main__":
    main()
