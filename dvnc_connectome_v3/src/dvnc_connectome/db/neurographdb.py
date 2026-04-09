"""
NeuroGraphDB — SQLite-backed brain-inspired property graph database.

Design principles:
  - Nodes are "neurons" with typed kinds (Concept, Document, Domain, LMM)
  - Edges are "synapses" with weights, evidence counts, and relation types
  - Hebbian plasticity: reinforce() increases synapse weight on repeated co-activation
  - Spreading activation: propagate() simulates neural firing through the graph
  - LMM tagging: every node can be labelled with Da Vinci's 10 Mental Models
"""

import json
import sqlite3
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


# Da Vinci's 10 Leonardo Mental Models
LMM_LABELS = {
    "LMM_001": "Geometric Structure",
    "LMM_002": "Dynamic Flow",
    "LMM_003": "Analogical Bridging",
    "LMM_004": "Systems Thinking",
    "LMM_005": "Sensory Detail",
    "LMM_006": "Spatial Reasoning",
    "LMM_007": "Living Machine",
    "LMM_008": "Sfumato (Ambiguity)",
    "LMM_009": "Art/Science Fusion",
    "LMM_010": "Corporalita (Body-Mind)",
}

# Relation type taxonomy
REL_CO_OCCURS = "CO_OCCURS"        # Two concepts appear together
REL_MENTIONS = "MENTIONS"          # Document mentions a concept
REL_PRIMES = "PRIMES"             # Concept activates another (causal)
REL_EVOKES = "EVOKES"             # Cross-domain structural analogy
REL_CONSTRAINS = "CONSTRAINS"      # Hard constraint relationship
REL_DERIVED_FROM = "DERIVED_FROM"  # Provenance chain
REL_OPPOSES = "OPPOSES"            # Adversarial / contradictory
REL_DOMAIN_OF = "DOMAIN_OF"        # Concept belongs to domain


class NeuroGraphDB:
    """
    A compact property-graph database designed for brain-inspired connectome storage.

    Each edge carries:
      weight        — synaptic strength (increases with Hebbian reinforcement)
      evidence_count— how many documents supported this edge
      rel           — relation type from taxonomy above
      last_seen     — Unix timestamp of last reinforcement
      lmm_tags      — Da Vinci mental model labels (JSON list)
    """

    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def init(self):
        """Create tables and indices. Safe to call multiple times."""
        c = self._conn
        c.executescript("""
        CREATE TABLE IF NOT EXISTS nodes (
            id       TEXT PRIMARY KEY,
            kind     TEXT NOT NULL DEFAULT 'Concept',
            label    TEXT NOT NULL,
            props    TEXT NOT NULL DEFAULT '{}'
        );

        CREATE TABLE IF NOT EXISTS synapses (
            pre           TEXT NOT NULL,
            post          TEXT NOT NULL,
            rel           TEXT NOT NULL,
            weight        REAL NOT NULL DEFAULT 1.0,
            evidence_count INTEGER NOT NULL DEFAULT 1,
            last_seen     REAL NOT NULL,
            lmm_tags      TEXT NOT NULL DEFAULT '[]',
            props         TEXT NOT NULL DEFAULT '{}',
            PRIMARY KEY (pre, post, rel)
        );

        CREATE INDEX IF NOT EXISTS idx_synapses_pre  ON synapses(pre);
        CREATE INDEX IF NOT EXISTS idx_synapses_post ON synapses(post);
        CREATE INDEX IF NOT EXISTS idx_synapses_rel  ON synapses(rel);
        CREATE INDEX IF NOT EXISTS idx_synapses_w    ON synapses(weight DESC);
        CREATE INDEX IF NOT EXISTS idx_nodes_kind    ON nodes(kind);
        CREATE INDEX IF NOT EXISTS idx_nodes_label   ON nodes(label);
        """)
        c.commit()

    # ------------------------------------------------------------------
    # Node operations
    # ------------------------------------------------------------------

    def upsert_node(self, node_id: str, kind: str = "Concept",
                    label: str = "", props: dict | None = None):
        props = props or {}
        label = label or node_id.split("::")[-1]
        self._conn.execute(
            """INSERT INTO nodes(id, kind, label, props)
               VALUES(?,?,?,?)
               ON CONFLICT(id) DO UPDATE SET
                 kind=excluded.kind,
                 label=excluded.label,
                 props=excluded.props""",
            (node_id, kind, label, json.dumps(props))
        )
        self._conn.commit()

    def get_node(self, node_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM nodes WHERE id=?", (node_id,)
        ).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["props"] = json.loads(d["props"])
        return d

    def search_nodes(self, query: str, kind: str | None = None, limit: int = 30) -> list[dict]:
        """Full-text search over node labels."""
        sql = "SELECT * FROM nodes WHERE label LIKE ?"
        params: list = [f"%{query}%"]
        if kind:
            sql += " AND kind=?"
            params.append(kind)
        sql += " LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Synapse operations (Hebbian plasticity)
    # ------------------------------------------------------------------

    def reinforce(self, pre: str, post: str, rel: str,
                  inc: float = 1.0, lmm_tags: list[str] | None = None,
                  **extra):
        """
        Hebbian reinforcement: if the synapse exists, strengthen it.
        If not, create it. Weight grows with every co-activation.
        """
        lmm_tags = lmm_tags or []
        now = time.time()
        existing = self._conn.execute(
            "SELECT weight, evidence_count, props FROM synapses WHERE pre=? AND post=? AND rel=?",
            (pre, post, rel)
        ).fetchone()

        if existing:
            new_weight = existing["weight"] + inc
            new_count = existing["evidence_count"] + 1
            old_props = json.loads(existing["props"])
            old_props.update(extra)
            self._conn.execute(
                """UPDATE synapses
                   SET weight=?, evidence_count=?, last_seen=?, lmm_tags=?, props=?
                   WHERE pre=? AND post=? AND rel=?""",
                (new_weight, new_count, now, json.dumps(lmm_tags),
                 json.dumps(old_props), pre, post, rel)
            )
        else:
            self._conn.execute(
                """INSERT INTO synapses(pre, post, rel, weight, evidence_count, last_seen, lmm_tags, props)
                   VALUES(?,?,?,?,1,?,?,?)""",
                (pre, post, rel, inc, now, json.dumps(lmm_tags), json.dumps(extra))
            )
        self._conn.commit()

    def top_neighbors(self, node_id: str, limit: int = 30,
                      rel_filter: list[str] | None = None) -> list[dict]:
        """Return strongest outgoing synapses for a node."""
        sql = "SELECT * FROM synapses WHERE pre=?"
        params: list = [node_id]
        if rel_filter:
            placeholders = ",".join("?" * len(rel_filter))
            sql += f" AND rel IN ({placeholders})"
            params.extend(rel_filter)
        sql += " ORDER BY weight DESC LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(sql, params).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["props"] = json.loads(d["props"])
            d["lmm_tags"] = json.loads(d["lmm_tags"])
            result.append(d)
        return result

    def get_synapse(self, pre: str, post: str, rel: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM synapses WHERE pre=? AND post=? AND rel=?",
            (pre, post, rel)
        ).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["props"] = json.loads(d["props"])
        d["lmm_tags"] = json.loads(d["lmm_tags"])
        return d

    # ------------------------------------------------------------------
    # Spreading activation (brain-inspired routing)
    # ------------------------------------------------------------------

    def propagate(self, start: str, steps: int = 4, fanout: int = 20,
                  decay: float = 0.80, rel_weights: dict[str, float] | None = None,
                  hub_threshold: int = 50) -> list[tuple[str, float]]:
        """
        Spreading activation from a seed concept.

        Parameters
        ----------
        start         : starting node id
        steps         : number of hops
        fanout        : max frontier size per step
        decay         : activation decay per hop (0–1)
        rel_weights   : per-relation type multipliers (e.g. EVOKES gets 2x)
        hub_threshold : nodes with more than this many outgoing synapses are
                        penalised (hub suppression — makes routing non-obvious)
        """
        rel_weights = rel_weights or {
            REL_EVOKES: 2.0,
            REL_PRIMES: 1.8,
            REL_CO_OCCURS: 1.0,
            REL_CONSTRAINS: 0.6,
            REL_OPPOSES: 0.3,
        }

        scores: dict[str, float] = defaultdict(float)
        scores[start] = 1.0
        visited: set[str] = {start}
        frontier: list[str] = [start]

        for step in range(steps):
            new_scores: dict[str, float] = defaultdict(float)
            for node in frontier:
                node_score = scores[node]
                neighbors = self.top_neighbors(node, limit=fanout)
                for syn in neighbors:
                    target = syn["post"]
                    rel = syn["rel"]
                    weight = syn["weight"]
                    rel_mult = rel_weights.get(rel, 1.0)

                    # Hub suppression: penalise overly-connected nodes
                    out_degree = self._conn.execute(
                        "SELECT COUNT(*) FROM synapses WHERE pre=?", (target,)
                    ).fetchone()[0]
                    hub_penalty = 1.0 if out_degree < hub_threshold else (hub_threshold / out_degree)

                    activation = node_score * weight * rel_mult * decay * hub_penalty
                    new_scores[target] += activation

            # Merge and pick top fanout
            for k, v in new_scores.items():
                scores[k] += v
            frontier = sorted(
                [k for k in new_scores if k not in visited],
                key=lambda k: scores[k],
                reverse=True
            )[:fanout]
            visited.update(frontier)

        # Exclude the start node from results
        result = [(k, v) for k, v in scores.items() if k != start]
        result.sort(key=lambda x: x[1], reverse=True)
        return result

    # ------------------------------------------------------------------
    # Plasticity: self-growing from high-scoring outputs
    # ------------------------------------------------------------------

    def auto_reinforce_from_output(self, concept_pairs: list[tuple[str, str]],
                                   score: float, threshold: float = 0.75):
        """
        If an innovation output scores above threshold, reinforce the concept
        pairs that led to it. The graph gets smarter with every use.
        """
        if score < threshold:
            return
        for (a, b) in concept_pairs:
            pre = f"concept::{a}"
            post = f"concept::{b}"
            inc = 0.5 * score  # Proportional to output quality
            self.reinforce(pre, post, REL_PRIMES, inc=inc)
            self.reinforce(post, pre, REL_PRIMES, inc=inc * 0.5)

    # ------------------------------------------------------------------
    # Stats and introspection
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        nodes = self._conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
        synapses = self._conn.execute("SELECT COUNT(*) FROM synapses").fetchone()[0]
        kinds = self._conn.execute(
            "SELECT kind, COUNT(*) as c FROM nodes GROUP BY kind"
        ).fetchall()
        rels = self._conn.execute(
            "SELECT rel, COUNT(*) as c, AVG(weight) as avg_w FROM synapses GROUP BY rel"
        ).fetchall()
        return {
            "nodes": nodes,
            "synapses": synapses,
            "node_kinds": {r["kind"]: r["c"] for r in kinds},
            "relation_types": {r["rel"]: {"count": r["c"], "avg_weight": round(r["avg_w"], 3)}
                               for r in rels},
        }

    def get_all_nodes(self, kind: str | None = None) -> list[dict]:
        sql = "SELECT * FROM nodes"
        params = []
        if kind:
            sql += " WHERE kind=?"
            params.append(kind)
        rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def get_all_synapses(self, limit: int = 5000) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM synapses ORDER BY weight DESC LIMIT ?", (limit,)
        ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["props"] = json.loads(d["props"])
            d["lmm_tags"] = json.loads(d["lmm_tags"])
            result.append(d)
        return result

    def close(self):
        self._conn.close()
