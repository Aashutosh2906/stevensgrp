"""
Da Vinci Routing System — Explicit visible routing through the DVNC Connectome.

Inspired by:
  - Jeff Lichtman's connectome traversal
  - MiroFish's agent-based network propagation
  - OpenClaw's multi-path adversarial search

Principles:
  1. PRIMES relations: direct causal/activation paths
  2. EVOKES relations: cross-domain analogical bridges (highest value)
  3. Hub suppression: avoid obvious overconnected nodes
  4. Route diversity: maintain multiple competing paths
  5. Full transparency: every routing decision is logged and visible

The router outputs a RouteResult with:
  - trace: step-by-step path with relation types and scores
  - branches: alternative route branches not taken
  - suppressed: hub nodes that were deliberately avoided
  - evidence_nodes: source document nodes found along the route
  - lmm_activations: which Da Vinci mental models fired
"""

from __future__ import annotations

import dataclasses
import math
from collections import defaultdict
from typing import Any

from ..db.neurographdb import (
    NeuroGraphDB,
    REL_CO_OCCURS, REL_EVOKES, REL_PRIMES, REL_MENTIONS,
    REL_CONSTRAINS, REL_DOMAIN_OF, LMM_LABELS,
)


@dataclasses.dataclass
class RouteStep:
    node_id: str
    label: str
    score: float
    rel_from_prev: str
    weight: float
    lmm_tags: list[str]
    is_cross_domain: bool = False
    suppressed: bool = False


@dataclasses.dataclass
class RouteResult:
    seed: str
    query: str
    primary_route: list[RouteStep]       # Best route
    alternative_routes: list[list[RouteStep]]  # Other paths
    suppressed_hubs: list[str]           # Hub nodes bypassed
    evidence_nodes: list[dict]           # Source documents found
    lmm_activations: dict[str, float]   # LMM model scores
    activation_map: list[tuple[str, float]]  # Full spreading activation
    novelty_score: float                 # How non-obvious is this route?
    cross_domain_count: int             # Number of domain bridges crossed

    def summary(self) -> str:
        """Human-readable route summary."""
        path = " → ".join(
            f"{s.label}[{s.rel_from_prev}]" if s.rel_from_prev else s.label
            for s in self.primary_route
        )
        lmm_str = ", ".join(
            f"{k}:{LMM_LABELS[k]}" for k, v in sorted(
                self.lmm_activations.items(), key=lambda x: x[1], reverse=True
            )[:3]
        )
        return (
            f"Route: {path}\n"
            f"Novelty: {self.novelty_score:.2f} | Cross-domain leaps: {self.cross_domain_count}\n"
            f"Da Vinci Models activated: {lmm_str}\n"
            f"Evidence sources: {len(self.evidence_nodes)}"
        )


class DaVinciRouter:
    """
    The explicit Da Vinci Routing System.

    This is not a black box. Every routing decision is visible and traceable.
    """

    # Relation weights for activation propagation
    _REL_MULTIPLIERS = {
        REL_EVOKES: 2.5,      # Cross-domain analogies are most valuable
        REL_PRIMES: 2.0,      # Causal/activation edges
        REL_CO_OCCURS: 1.0,   # Co-occurrence (standard)
        REL_MENTIONS: 0.5,    # Document mentions (weaker)
        REL_CONSTRAINS: 0.4,  # Constraints dampen propagation
        REL_DOMAIN_OF: 0.3,
    }

    def __init__(self, db: NeuroGraphDB,
                 steps: int = 4,
                 fanout: int = 20,
                 decay: float = 0.82,
                 hub_threshold: int = 40,
                 n_alternative_routes: int = 3):
        self.db = db
        self.steps = steps
        self.fanout = fanout
        self.decay = decay
        self.hub_threshold = hub_threshold
        self.n_alternatives = n_alternative_routes

    # ------------------------------------------------------------------
    # Main routing entry point
    # ------------------------------------------------------------------

    def route(self, query: str, domain_bias: str | None = None) -> RouteResult:
        """
        Route from a natural-language query through the connectome.

        1. Resolve query to seed concept nodes
        2. Run spreading activation (visible, logged)
        3. Extract primary route + alternatives
        4. Score novelty and cross-domain count
        5. Pull evidence documents
        6. Return full RouteResult
        """
        # Step 1: Seed concept resolution
        seeds = self._resolve_seeds(query)
        if not seeds:
            # Fallback: use raw query terms as concept IDs
            seeds = [f"concept::{t.lower()}" for t in query.split()[:3]]

        primary_seed = seeds[0]

        # Step 2: Full spreading activation map
        activation_map = self.db.propagate(
            start=primary_seed,
            steps=self.steps,
            fanout=self.fanout,
            decay=self.decay,
            rel_weights=self._REL_MULTIPLIERS,
            hub_threshold=self.hub_threshold,
        )

        # Step 3: Identify suppressed hubs
        suppressed = self._find_suppressed_hubs()

        # Step 4: Build primary route (greedy best-path)
        primary_route = self._build_route(primary_seed, activation_map)

        # Step 5: Alternative routes (diverse, non-overlapping)
        alt_routes = self._build_alternative_routes(
            primary_seed, activation_map, primary_route
        )

        # Step 6: Pull evidence documents
        evidence_nodes = self._gather_evidence(activation_map[:15])

        # Step 7: LMM activation scores
        lmm_activations = self._score_lmm(primary_route + sum(alt_routes, []))

        # Step 8: Novelty score
        novelty = self._score_novelty(primary_route, activation_map)

        # Step 9: Count cross-domain leaps
        cross_domain_count = sum(1 for s in primary_route if s.is_cross_domain)

        return RouteResult(
            seed=primary_seed,
            query=query,
            primary_route=primary_route,
            alternative_routes=alt_routes,
            suppressed_hubs=suppressed,
            evidence_nodes=evidence_nodes,
            lmm_activations=lmm_activations,
            activation_map=activation_map[:30],
            novelty_score=novelty,
            cross_domain_count=cross_domain_count,
        )

    # ------------------------------------------------------------------
    # Seed resolution
    # ------------------------------------------------------------------

    # Generic words that make poor routing seeds — suppress them so the
    # router starts from domain-specific concepts instead.
    _STOP_SEEDS = {
        "novel", "new", "design", "create", "develop", "build", "make",
        "propose", "combine", "combining", "use", "using", "based",
        "approach", "method", "system", "model", "analysis", "study",
        "research", "investigate", "explore", "evaluate", "assess",
        "the", "for", "and", "with", "that", "this", "from", "into",
        "application", "applications", "material", "materials",
        "structure", "process", "technology", "technique",
    }

    def _resolve_seeds(self, query: str) -> list[str]:
        """
        Find concept nodes in the DB matching query terms.

        Strategy: rank candidates by specificity (out-degree) so rare,
        domain-specific concepts (e.g. 'auxetic', 'PEDOT', 'immunomodulation')
        are preferred over generic hubs ('novel', 'design', 'material').
        """
        # --- 1. Extract candidate terms (single words + bigrams) ----------
        words = [t.lower() for t in query.split() if len(t) >= 3]
        bigrams = [
            f"{words[i]} {words[i+1]}"
            for i in range(len(words) - 1)
        ]
        candidates = bigrams + words  # bigrams first (more specific)

        # --- 2. Resolve each candidate to a node, with specificity score --
        scored: list[tuple[str, float]] = []  # (node_id, specificity)
        seen_ids: set[str] = set()

        for term in candidates:
            # Skip generic stop-seeds
            if term in self._STOP_SEEDS:
                continue

            # Exact match
            node_id = f"concept::{term}"
            node = self.db.get_node(node_id)
            if node and node_id not in seen_ids:
                seen_ids.add(node_id)
                # Specificity = inverse of out-degree (fewer connections = rarer = better seed)
                out_deg = self.db._conn.execute(
                    "SELECT COUNT(*) FROM synapses WHERE pre=?", (node_id,)
                ).fetchone()[0]
                specificity = 1.0 / max(out_deg, 1)
                scored.append((node_id, specificity))
                continue

            # Fuzzy search
            matches = self.db.search_nodes(term, kind="Concept", limit=3)
            for m in matches:
                mid = f"concept::{m['label']}"
                if mid not in seen_ids:
                    seen_ids.add(mid)
                    out_deg = self.db._conn.execute(
                        "SELECT COUNT(*) FROM synapses WHERE pre=?", (mid,)
                    ).fetchone()[0]
                    specificity = 1.0 / max(out_deg, 1)
                    scored.append((mid, specificity))

        # --- 3. Sort by specificity (most specific first) -----------------
        scored.sort(key=lambda x: x[1], reverse=True)
        return [node_id for node_id, _ in scored[:5]]

    # ------------------------------------------------------------------
    # Route construction
    # ------------------------------------------------------------------

    def _build_route(self, seed: str, activation_map: list[tuple[str, float]],
                     exclude: set[str] | None = None) -> list[RouteStep]:
        """
        Construct a route by following the highest-activation path from seed.
        Each step picks the strongest neighbour not yet in the route.
        """
        exclude = exclude or set()
        route: list[RouteStep] = []
        visited = {seed} | exclude

        # Seed step
        seed_node = self.db.get_node(seed)
        seed_label = seed_node["label"] if seed_node else seed.split("::")[-1]
        route.append(RouteStep(
            node_id=seed,
            label=seed_label,
            score=1.0,
            rel_from_prev="",
            weight=1.0,
            lmm_tags=json_parse(seed_node.get("props", "{}")).get("lmm_tags", []) if seed_node else [],
        ))

        # Build activation lookup
        score_map = {k: v for k, v in activation_map}

        current = seed
        for _ in range(self.steps):
            neighbors = self.db.top_neighbors(current, limit=self.fanout)
            best = None
            best_score = -1.0
            best_syn = None

            for syn in neighbors:
                target = syn["post"]
                if target in visited:
                    continue
                # Score = activation_map score × relation multiplier
                a_score = score_map.get(target, 0.0)
                rel_mult = self._REL_MULTIPLIERS.get(syn["rel"], 1.0)
                combined = a_score * rel_mult
                if combined > best_score:
                    best_score = combined
                    best = target
                    best_syn = syn

            if best is None:
                break

            visited.add(best)
            node = self.db.get_node(best)
            label = node["label"] if node else best.split("::")[-1]
            node_props = json_parse(node.get("props", "{}")) if node else {}

            # Detect cross-domain leap
            is_cross_domain = best_syn["rel"] == "EVOKES" or best.startswith("domain::")

            route.append(RouteStep(
                node_id=best,
                label=label,
                score=best_score,
                rel_from_prev=best_syn["rel"],
                weight=best_syn["weight"],
                lmm_tags=best_syn.get("lmm_tags", []),
                is_cross_domain=is_cross_domain,
            ))
            current = best

        return route

    def _build_alternative_routes(
        self,
        seed: str,
        activation_map: list[tuple[str, float]],
        primary_route: list[RouteStep],
    ) -> list[list[RouteStep]]:
        """
        Build diverse alternative routes by excluding primary route nodes.
        Each alternative starts from the seed but avoids primary route nodes.
        """
        primary_nodes = {s.node_id for s in primary_route}
        alts = []
        excluded = set(primary_nodes)

        for _ in range(self.n_alternatives):
            alt = self._build_route(seed, activation_map, exclude=excluded)
            if len(alt) > 1:
                alts.append(alt)
                excluded.update(s.node_id for s in alt)
            else:
                break

        return alts

    # ------------------------------------------------------------------
    # Hub suppression
    # ------------------------------------------------------------------

    def _find_suppressed_hubs(self) -> list[str]:
        """Return node IDs that were suppressed due to high out-degree."""
        rows = self.db._conn.execute(
            """SELECT pre, COUNT(*) as deg FROM synapses
               GROUP BY pre HAVING deg > ?
               ORDER BY deg DESC LIMIT 10""",
            (self.hub_threshold,)
        ).fetchall()
        return [r["pre"] for r in rows]

    # ------------------------------------------------------------------
    # Evidence gathering
    # ------------------------------------------------------------------

    def _gather_evidence(self, top_nodes: list[tuple[str, float]]) -> list[dict]:
        """
        Find source documents connected to activated concept nodes.

        Strategy:
          1. Look at the full activation map (not just top 15 umbrella nodes)
          2. For each activated concept, find MENTIONS edges pointing to it
          3. Also look for documents that MENTION domain nodes (e.g. cardiac_biomaterials)
          4. Score each doc by sum of (concept_activation * edge_weight)
        """
        evidence = []
        seen_docs: set[str] = set()
        doc_scores: dict[str, float] = defaultdict(float)  # accumulate relevance
        doc_cache: dict[str, dict] = {}

        # Widen the search: use top 40 activated nodes (not just 15)
        search_nodes = top_nodes[:40]

        for node_id, score in search_nodes:
            # --- Concept nodes: find docs that MENTION this concept --------
            rows = self.db._conn.execute(
                """SELECT pre, weight FROM synapses
                   WHERE post=? AND rel='MENTIONS'
                   ORDER BY weight DESC LIMIT 5""",
                (node_id,)
            ).fetchall()
            for row in rows:
                doc_id = row["pre"]
                doc_scores[doc_id] += score * row["weight"]
                if doc_id not in doc_cache:
                    doc_node = self.db.get_node(doc_id)
                    if doc_node:
                        doc_cache[doc_id] = doc_node

            # --- Domain nodes: find docs in this domain -------------------
            if node_id.startswith("domain::"):
                rows = self.db._conn.execute(
                    """SELECT pre, weight FROM synapses
                       WHERE post=? AND rel='DOMAIN_OF'
                       ORDER BY weight DESC LIMIT 5""",
                    (node_id,)
                ).fetchall()
                for row in rows:
                    doc_id = row["pre"]
                    doc_scores[doc_id] += score * row["weight"]
                    if doc_id not in doc_cache:
                        doc_node = self.db.get_node(doc_id)
                        if doc_node:
                            doc_cache[doc_id] = doc_node

        # Build final evidence list sorted by accumulated relevance
        for doc_id, total_score in sorted(doc_scores.items(), key=lambda x: x[1], reverse=True):
            if doc_id in doc_cache:
                doc_node = doc_cache[doc_id]
                evidence.append({
                    "node_id": doc_id,
                    "title": doc_node["label"],
                    "domain": doc_node["props"].get("domain", ""),
                    "source": doc_node["props"].get("source", ""),
                    "url": doc_node["props"].get("url", ""),
                    "relevance_score": total_score,
                })
            if len(evidence) >= 12:
                break

        return evidence

    # ------------------------------------------------------------------
    # LMM scoring
    # ------------------------------------------------------------------

    def _score_lmm(self, all_steps: list[RouteStep]) -> dict[str, float]:
        """Aggregate Da Vinci Mental Model activation scores across route steps."""
        scores: dict[str, float] = defaultdict(float)
        for step in all_steps:
            for tag in step.lmm_tags:
                scores[tag] += step.score
        return dict(scores)

    # ------------------------------------------------------------------
    # Novelty scoring
    # ------------------------------------------------------------------

    def _score_novelty(self, route: list[RouteStep],
                        activation_map: list[tuple[str, float]]) -> float:
        """
        Novelty = how non-obvious is this route?

        Higher novelty if:
          - Route uses EVOKES relations (cross-domain)
          - Route traverses low-weight synapses (less trodden paths)
          - Route includes concepts with low overall activation scores
        """
        if len(route) < 2:
            return 0.0

        evokes_count = sum(1 for s in route if s.rel_from_prev == "EVOKES")
        avg_weight = sum(s.weight for s in route[1:]) / max(len(route) - 1, 1)
        avg_score = sum(s.score for s in route[1:]) / max(len(route) - 1, 1)

        # Penalty for using only common high-weight paths
        commonness = avg_weight / max(avg_weight, 1e-9)
        evokes_bonus = evokes_count * 0.3

        # Normalise: lower commonness + more evokes = higher novelty
        novelty = min(1.0, (1.0 - math.tanh(commonness * 0.5)) + evokes_bonus)
        return round(novelty, 3)


def json_parse(s: str | dict) -> dict:
    """Safe JSON parse — handles both str and dict inputs."""
    if isinstance(s, dict):
        return s
    import json
    try:
        return json.loads(s)
    except Exception:
        return {}
