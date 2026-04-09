"""
Data curation pipeline: ingest documents → build connectome.

For each document:
  1. Extract concept tokens
  2. Create Document node and Concept nodes
  3. Reinforce MENTIONS synapses (doc → concept)
  4. Reinforce CO_OCCURS synapses (concept ↔ concept) from co-occurrence
  5. Detect cross-domain EVOKES edges (cross-domain co-occurrence)
  6. Tag nodes with Da Vinci LMM labels based on domain heuristics
"""

import time
from typing import Iterable

from ..db.neurographdb import (
    NeuroGraphDB,
    REL_CO_OCCURS, REL_MENTIONS, REL_EVOKES, REL_DOMAIN_OF,
    LMM_LABELS,
)
from .noun_assoc import concepts, cooccurrence_edges


# Domain → LMM mapping heuristic
_DOMAIN_LMM = {
    "biomechanics":         ["LMM_007", "LMM_010", "LMM_002"],
    "ergonomics":           ["LMM_010", "LMM_005", "LMM_009"],
    "materials":            ["LMM_001", "LMM_007", "LMM_004"],
    "design":               ["LMM_003", "LMM_009", "LMM_006"],
    "neuroscience":         ["LMM_007", "LMM_004", "LMM_008"],
    "bionics":              ["LMM_007", "LMM_003", "LMM_001"],
    # Stevens Group domains
    "raman_spectroscopy":   ["LMM_005", "LMM_001", "LMM_004"],
    "cardiac_biomaterials": ["LMM_007", "LMM_003", "LMM_005"],
    "conducting_polymers":  ["LMM_001", "LMM_007", "LMM_002"],
    "biomineralization":    ["LMM_001", "LMM_003", "LMM_007"],
    "general":              ["LMM_004"],
}

# Concept nodes per domain for cross-domain bridge detection
_DOMAIN_CONCEPTS: dict[str, set[str]] = {}


def _domain_lmm(domain: str) -> list[str]:
    return _DOMAIN_LMM.get(domain, ["LMM_004"])


def _ensure_domain_node(db: NeuroGraphDB, domain: str):
    domain_id = f"domain::{domain}"
    db.upsert_node(domain_id, kind="Domain", label=domain,
                   props={"lmm_tags": _domain_lmm(domain)})
    return domain_id


def ingest_docs(db: NeuroGraphDB, docs: Iterable[dict],
                window: int = 8, max_terms: int = 100,
                verbose: bool = True) -> dict:
    """
    Main ingestion function. Builds the full connectome from a stream of docs.

    Returns stats dict with node/synapse counts.
    """
    n_docs = 0
    n_concepts = 0
    n_synapses = 0

    for doc in docs:
        doc_id = doc.get("doc_id", f"doc_{n_docs}")
        title = doc.get("title", doc_id)
        text = doc.get("text", "")
        source = doc.get("source", "unknown")
        domain = doc.get("domain", "general")

        if len(text) < 20:
            continue

        # Document node — store a text snippet for evidence retrieval
        node_id = f"doc::{source}::{doc_id}"
        snippet = text[:600] + ("..." if len(text) > 600 else "")
        db.upsert_node(node_id, kind="Document", label=title,
                       props={"source": source, "domain": domain,
                              "year": doc.get("year"),
                              "url": doc.get("url", ""),
                              "text": snippet,
                              "lmm_tags": _domain_lmm(domain)})

        # Domain node
        domain_id = _ensure_domain_node(db, domain)
        db.reinforce(node_id, domain_id, REL_DOMAIN_OF, inc=1.0)

        # Extract concepts
        terms = concepts(text, max_terms=max_terms)
        if not terms:
            continue

        # Track per-domain concepts for cross-domain edges
        _DOMAIN_CONCEPTS.setdefault(domain, set()).update(terms)

        # Concept nodes + MENTIONS edges
        term_nodes = []
        for term in terms:
            t_node = f"concept::{term}"
            db.upsert_node(t_node, kind="Concept", label=term,
                           props={"domain_hint": domain,
                                  "lmm_tags": _domain_lmm(domain)})
            db.reinforce(node_id, t_node, REL_MENTIONS, inc=1.0,
                         lmm_tags=_domain_lmm(domain),
                         last_seen=time.time())
            term_nodes.append(term)
            n_concepts += 1

        # CO_OCCURS edges (within document)
        for (a, b, w) in cooccurrence_edges(term_nodes, window=window):
            lmm = _domain_lmm(domain)
            db.reinforce(f"concept::{a}", f"concept::{b}", REL_CO_OCCURS,
                         inc=w, lmm_tags=lmm, last_seen=time.time())
            db.reinforce(f"concept::{b}", f"concept::{a}", REL_CO_OCCURS,
                         inc=w * 0.8, lmm_tags=lmm, last_seen=time.time())
            n_synapses += 2

        n_docs += 1
        if verbose and n_docs % 10 == 0:
            print(f"[pipeline] Ingested {n_docs} docs, {n_synapses} synapses")

    # Cross-domain EVOKES edges (structural analogies between domains)
    _add_cross_domain_bridges(db)

    stats = db.stats()
    if verbose:
        print(f"[pipeline] Done. {stats}")
    return stats


def _add_cross_domain_bridges(db: NeuroGraphDB):
    """
    Add cross-domain EVOKES edges where the same concept appears in multiple domains.
    This is the core of the Da Vinci structural analogy layer.
    """
    domains = list(_DOMAIN_CONCEPTS.keys())
    for i, d1 in enumerate(domains):
        for d2 in domains[i + 1:]:
            shared = _DOMAIN_CONCEPTS[d1] & _DOMAIN_CONCEPTS[d2]
            for concept in shared:
                # A concept shared across domains EVOKES cross-domain thinking
                node = f"concept::{concept}"
                db.reinforce(node, f"domain::{d1}", REL_EVOKES, inc=1.5,
                             lmm_tags=["LMM_003"],  # Analogical Bridging
                             cross_domain=True)
                db.reinforce(node, f"domain::{d2}", REL_EVOKES, inc=1.5,
                             lmm_tags=["LMM_003"],
                             cross_domain=True)


def run_full_pipeline(db: NeuroGraphDB, include_network: bool = True,
                       include_stevens: bool = True,
                       extra_docs: list[dict] | None = None,
                       extra_dois: list[str] | None = None,
                       search_queries: list[str] | None = None,
                       verbose: bool = True) -> dict:
    """
    Top-level pipeline runner. Fetches all data sources and builds the graph.

    Sources:
      - Seed corpus + Wikipedia + OpenAlex (generic)
      - Stevens Group corpus + uploaded papers (domain-specific)
      - Extra DOIs fetched from Semantic Scholar
      - Papers found via search queries
      - User-provided extra_docs
    """
    from .datasets import iter_all_documents
    db.init()

    docs = list(iter_all_documents(include_network=include_network))

    # Stevens Group corpus (core + uploads + dynamic fetches)
    if include_stevens:
        try:
            from .stevens_loader import iter_all_stevens_docs
            stevens_docs = list(iter_all_stevens_docs(
                include_uploads=True,
                extra_dois=extra_dois,
                search_queries=search_queries,
            ))
            docs.extend(stevens_docs)
            if verbose:
                print(f"[pipeline] Stevens corpus: {len(stevens_docs)} documents")
        except Exception as e:
            if verbose:
                print(f"[pipeline] Stevens loader not available: {e}")

    if extra_docs:
        docs.extend(extra_docs)

    if verbose:
        print(f"[pipeline] Total documents to ingest: {len(docs)}")

    return ingest_docs(db, docs, verbose=verbose)
