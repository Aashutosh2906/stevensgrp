#!/usr/bin/env python3
"""
Build the DVNC Connectome database.

Usage:
    python scripts/build_db.py --db ./data/dvnc.db [--no-network] [--no-stevens]
                               [--codex path] [--masterclass path] [--triplets path]
                               [--dois 10.1234/... 10.5678/...]
                               [--search "cardiac biomaterials" "Raman spectroscopy"]

Options:
    --db            Path to output SQLite database (created if absent)
    --no-network    Skip network dataset fetching (use seed corpus only)
    --no-stevens    Skip Stevens Group corpus loading
    --codex         Path to a codex_data.json file (Leonardo notebooks)
    --masterclass   Path to a Masterclass_data.json file
    --triplets      Path to a dvnc_kg_triplets.csv file (pre-existing triplets)
    --dois          Space-separated DOIs to fetch from Semantic Scholar
    --search        Space-separated search queries for additional papers
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dvnc_connectome.db.neurographdb import NeuroGraphDB
from dvnc_connectome.curation.pipeline import run_full_pipeline, ingest_docs


def load_json_docs(path: str, source: str) -> list[dict]:
    """Load a JSON file as a list of documents."""
    p = Path(path)
    if not p.exists():
        print(f"[build_db] Warning: {path} not found, skipping.")
        return []
    with open(p) as f:
        data = json.load(f)
    docs = []
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        items = data.get("data", [data])
    else:
        items = [data]
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        text = item.get("text") or item.get("content") or item.get("transcription", "")
        if not text:
            continue
        docs.append({
            "doc_id": item.get("id") or item.get("doc_id") or f"{source}_{i}",
            "title": item.get("title") or f"{source} document {i}",
            "text": text,
            "source": source,
            "domain": item.get("domain", "general"),
        })
    print(f"[build_db] Loaded {len(docs)} docs from {path}")
    return docs


def load_triplets_csv(path: str) -> list[dict]:
    """Load pre-existing triplets CSV and convert to document format."""
    import csv
    p = Path(path)
    if not p.exists():
        print(f"[build_db] Warning: {path} not found, skipping.")
        return []
    docs = []
    with open(p) as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            subj = row.get("subject", "")
            pred = row.get("predicate", "")
            obj = row.get("object", "")
            if subj and pred and obj:
                text = f"{subj} {pred.replace('_', ' ')} {obj}."
                docs.append({
                    "doc_id": f"triplet_{i}",
                    "title": f"{subj} — {pred} — {obj}",
                    "text": text,
                    "source": "triplets",
                    "domain": "general",
                })
    print(f"[build_db] Loaded {len(docs)} triplets from {path}")
    return docs


def main():
    parser = argparse.ArgumentParser(description="Build DVNC Connectome database")
    parser.add_argument("--db", default="./data/dvnc.db", help="Output database path")
    parser.add_argument("--no-network", action="store_true", help="Skip network fetching")
    parser.add_argument("--no-stevens", action="store_true", help="Skip Stevens corpus")
    parser.add_argument("--codex", default=None, help="Path to codex_data.json")
    parser.add_argument("--masterclass", default=None, help="Path to Masterclass_data.json")
    parser.add_argument("--triplets", default=None, help="Path to dvnc_kg_triplets.csv")
    parser.add_argument("--dois", nargs="*", default=None, help="DOIs to fetch")
    parser.add_argument("--search", nargs="*", default=None, help="Search queries")
    args = parser.parse_args()

    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[build_db] Building connectome at: {db_path}")
    db = NeuroGraphDB(db_path)
    db.init()

    # Collect extra docs from user-provided files
    extra_docs: list[dict] = []
    if args.codex:
        extra_docs.extend(load_json_docs(args.codex, "codex"))
    if args.masterclass:
        extra_docs.extend(load_json_docs(args.masterclass, "masterclass"))
    if args.triplets:
        extra_docs.extend(load_triplets_csv(args.triplets))

    t0 = time.time()
    stats = run_full_pipeline(
        db=db,
        include_network=not args.no_network,
        include_stevens=not args.no_stevens,
        extra_docs=extra_docs if extra_docs else None,
        extra_dois=args.dois,
        search_queries=args.search,
        verbose=True,
    )
    elapsed = time.time() - t0

    print(f"\n[build_db] Complete in {elapsed:.1f}s")
    print(f"[build_db] Final stats: {json.dumps(stats, indent=2)}")


if __name__ == "__main__":
    main()
