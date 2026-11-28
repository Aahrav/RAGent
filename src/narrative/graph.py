"""Narrative Knowledge Graph Builder for TRACE (Issue #56).

Constructs a NetworkX MultiDiGraph from extracted facts (ExtractedFact /
FactTriple), NarrativeEvent records, and KnowledgeAssertion records. Provides:
  - Typed nodes: Character, Location, Object, Event, Scene, Concept
  - Multi-edge relationships preserving scene provenance & confidence
  - Query utilities: entity fact lookup, conflict detection, k-hop traversal
  - Frontend-ready JSON serialization (`to_graph_json`) and disk persistence
"""

from __future__ import annotations

import json
import re
from collections import deque
from pathlib import Path
from typing import Any, Literal

import networkx as nx
from pydantic import BaseModel, Field

from src.narrative.extractor import (
    DocumentExtraction,
    FactTriple,
    NarrativeEvent,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

NodeType = Literal["Character", "Location", "Object", "Event", "Scene", "Concept"]

_SINGLE_VALUED_PREDICATES: frozenset[str] = frozenset(
    {
        "has_status",
        "status",
        "eye_color",
        "hair_color",
        "age",
        "located_at",
        "lives_in",
        "born_in",
        "died_in",
    }
)

_POSSESSION_PREDICATES: frozenset[str] = frozenset(
    {"holds", "owns", "carries", "has_item", "wields", "possesses"}
)


def _normalize_id(name: str) -> str:
    """Normalize an entity name into a canonical lowercase snake_case node ID."""
    cleaned = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
    return cleaned or name.strip().lower()


# ── Pydantic v2 Graph Schemas ──────────────────────────────────────────────────


class GraphNode(BaseModel):
    """A typed entity node in the narrative knowledge graph."""

    id: str = Field(..., min_length=1)
    label: str = Field(..., min_length=1)
    node_type: NodeType = "Concept"
    aliases: list[str] = Field(default_factory=list)
    attributes: dict[str, Any] = Field(default_factory=dict)


class GraphEdge(BaseModel):
    """A directed relationship edge anchored to a scene/segment."""

    source: str = Field(..., min_length=1)
    target: str = Field(..., min_length=1)
    predicate: str = Field(..., min_length=1)
    scene_id: str | None = Field(default=None)
    source_segment: str | None = Field(default=None)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    evidence: str | None = Field(default=None)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConflictingEdgePair(BaseModel):
    """Two edges on the same entity/predicate that contradict each other."""

    entity_id: str = Field(..., min_length=1)
    predicate: str = Field(..., min_length=1)
    edge_a: GraphEdge
    edge_b: GraphEdge
    reason: str = Field(..., min_length=1)


# ── Knowledge Graph Engine ─────────────────────────────────────────────────────


class NarrativeKnowledgeGraph:
    """NetworkX MultiDiGraph-backed knowledge graph for narrative continuity."""

    def __init__(self, title: str = "Untitled") -> None:
        self.title = title
        self.graph: nx.MultiDiGraph = nx.MultiDiGraph()
        self._alias_index: dict[str, str] = {}

    # ── Construction ───────────────────────────────────────────────────────────

    @classmethod
    def from_extraction(cls, extraction: DocumentExtraction) -> "NarrativeKnowledgeGraph":
        """Build a complete knowledge graph from a DocumentExtraction."""
        kg = cls(title=extraction.title)
        kg.add_extraction(extraction)
        return kg

    @classmethod
    def from_db(cls, document_id: Any, db: Any | None = None) -> "NarrativeKnowledgeGraph":
        """Reconstruct a knowledge graph from persisted SQL/PostgreSQL records."""
        import uuid as _uuid

        from src.storage.db.database import SessionLocal
        from src.storage.db.models import Document, NarrativeEventRecord, NarrativeFact

        doc_uuid = (
            document_id if isinstance(document_id, _uuid.UUID) else _uuid.UUID(str(document_id))
        )
        owns_session = db is None
        session = db if db is not None else SessionLocal()

        try:
            doc = session.query(Document).filter(Document.id == doc_uuid).first()
            kg = cls(title=doc.filename if doc else str(doc_uuid))

            facts = session.query(NarrativeFact).filter(NarrativeFact.document_id == doc_uuid).all()
            for f in facts:
                kg.add_fact(
                    FactTriple(
                        subject=f.subject,
                        predicate=f.predicate,
                        object=f.object,
                        scene_id=f.scene_id or "unknown",
                        source_segment=f.source_segment,
                        confidence=f.confidence,
                        source=f.source if f.source in ("system1_deterministic", "laya", "llm") else "llm",
                        evidence=f.evidence,
                    )
                )

            events = (
                session.query(NarrativeEventRecord)
                .filter(NarrativeEventRecord.document_id == doc_uuid)
                .all()
            )
            for idx, ev in enumerate(events):
                kg.add_event(
                    NarrativeEvent(
                        description=ev.description,
                        characters=list(ev.characters or []),
                        location=ev.location,
                        temporal_order=ev.temporal_order,
                        source_segment=ev.source_segment,
                        scene_id=ev.scene_id,
                        confidence=ev.confidence,
                    ),
                    event_id=f"event_{idx:04d}",
                )

            return kg
        finally:
            if owns_session:
                session.close()

    def add_extraction(self, extraction: DocumentExtraction) -> None:
        """Ingest all facts and events from a DocumentExtraction."""
        for fact in extraction.all_triples:
            self.add_fact(fact)

        for idx, event in enumerate(extraction.all_events):
            self.add_event(event, event_id=f"event_{idx:04d}_{event.scene_id or 'na'}")

        logger.info(
            "Knowledge graph updated from extraction",
            extra={
                "title": self.title,
                "nodes": self.graph.number_of_nodes(),
                "edges": self.graph.number_of_edges(),
            },
        )

    def add_node(
        self,
        label: str,
        node_type: NodeType = "Concept",
        node_id: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> str:
        """Upsert a node into the MultiDiGraph and register its canonical alias."""
        nid = node_id or _normalize_id(label)
        if nid in self.graph:
            existing = self.graph.nodes[nid]
            # Upgrade generic Concept type when a more specific type is observed
            if existing.get("node_type") == "Concept" and node_type != "Concept":
                existing["node_type"] = node_type
            aliases: list[str] = existing.setdefault("aliases", [])
            if label not in aliases and label != existing.get("label"):
                aliases.append(label)
            if attributes:
                existing.setdefault("attributes", {}).update(attributes)
        else:
            self.graph.add_node(
                nid,
                id=nid,
                label=label,
                node_type=node_type,
                aliases=[label],
                attributes=dict(attributes or {}),
            )

        self._alias_index[_normalize_id(label)] = nid
        self._alias_index[nid] = nid
        return nid

    def add_fact(self, fact: FactTriple) -> None:
        """Add an ExtractedFact / FactTriple as typed nodes and a directed edge."""
        subj_type, obj_type = self._infer_node_types(fact.predicate)
        src_id = self.add_node(fact.subject, node_type=subj_type)
        tgt_id = self.add_node(fact.object, node_type=obj_type)

        self.graph.add_edge(
            src_id,
            tgt_id,
            source=src_id,
            target=tgt_id,
            predicate=fact.predicate.lower(),
            scene_id=fact.scene_id,
            source_segment=fact.source_segment,
            confidence=fact.confidence,
            evidence=fact.evidence,
            metadata={"extractor_source": fact.source},
        )

    def add_event(self, event: NarrativeEvent, event_id: str | None = None) -> str:
        """Add a NarrativeEvent node linked to its participating characters and location."""
        eid = event_id or _normalize_id(f"event_{event.scene_id or ''}_{event.description[:32]}")
        self.add_node(
            label=event.description,
            node_type="Event",
            node_id=eid,
            attributes={
                "temporal_order": event.temporal_order,
                "scene_id": event.scene_id,
                "source_segment": event.source_segment,
            },
        )

        for char in event.characters:
            char_id = self.add_node(char, node_type="Character")
            self.graph.add_edge(
                char_id,
                eid,
                source=char_id,
                target=eid,
                predicate="participated_in",
                scene_id=event.scene_id,
                source_segment=event.source_segment,
                confidence=event.confidence,
                evidence=None,
                metadata={},
            )

        if event.location:
            loc_id = self.add_node(event.location, node_type="Location")
            self.graph.add_edge(
                eid,
                loc_id,
                source=eid,
                target=loc_id,
                predicate="occurred_at",
                scene_id=event.scene_id,
                source_segment=event.source_segment,
                confidence=event.confidence,
                evidence=None,
                metadata={},
            )

        return eid

    # ── Query Utilities ────────────────────────────────────────────────────────

    def resolve_entity(self, entity: str) -> str | None:
        """Resolve a display name, alias, or node ID to its canonical graph node ID."""
        if entity in self.graph:
            return entity
        return self._alias_index.get(_normalize_id(entity))

    def get_node(self, entity: str) -> GraphNode | None:
        """Return the GraphNode for an entity if present."""
        nid = self.resolve_entity(entity)
        if nid is None:
            return None
        data = self.graph.nodes[nid]
        return GraphNode(**data)

    def get_entity_facts(self, entity: str, include_incoming: bool = True) -> list[GraphEdge]:
        """Return all outgoing (and optionally incoming) edges for an entity."""
        nid = self.resolve_entity(entity)
        if nid is None:
            return []

        edges: list[GraphEdge] = []
        for _, _, data in self.graph.out_edges(nid, data=True):
            edges.append(self._edge_from_data(data))

        if include_incoming:
            for _, _, data in self.graph.in_edges(nid, data=True):
                edges.append(self._edge_from_data(data))

        return edges

    def find_conflicting_edges(self) -> list[ConflictingEdgePair]:
        """Detect contradictory edges on single-valued predicates across scenes."""
        conflicts: list[ConflictingEdgePair] = []

        for nid in self.graph.nodes:
            by_pred: dict[str, list[GraphEdge]] = {}
            for _, _, data in self.graph.out_edges(nid, data=True):
                edge = self._edge_from_data(data)
                if edge.predicate in _SINGLE_VALUED_PREDICATES:
                    by_pred.setdefault(edge.predicate, []).append(edge)

            for pred, edges in by_pred.items():
                if len(edges) < 2:
                    continue
                # Compare distinct target values for the same subject + single-valued predicate
                for i in range(len(edges)):
                    for j in range(i + 1, len(edges)):
                        ea, eb = edges[i], edges[j]
                        if ea.target != eb.target:
                            conflicts.append(
                                ConflictingEdgePair(
                                    entity_id=nid,
                                    predicate=pred,
                                    edge_a=ea,
                                    edge_b=eb,
                                    reason=(
                                        f"Entity '{nid}' has conflicting '{pred}' values: "
                                        f"'{ea.target}' ({ea.scene_id}) vs '{eb.target}' ({eb.scene_id})."
                                    ),
                                )
                            )

        return conflicts

    def traverse_relationships(
        self,
        entity: str,
        max_hops: int = 2,
        predicates: set[str] | None = None,
    ) -> dict[str, Any]:
        """BFS traversal up to max_hops from an entity, returning reachable nodes and edges."""
        start_id = self.resolve_entity(entity)
        if start_id is None:
            return {"root": None, "nodes": [], "edges": []}

        visited: set[str] = {start_id}
        queue: deque[tuple[str, int]] = deque([(start_id, 0)])
        collected_edges: list[GraphEdge] = []
        pred_filter = {p.lower() for p in predicates} if predicates else None

        while queue:
            curr, depth = queue.popleft()
            if depth >= max_hops:
                continue

            neighbors = list(self.graph.out_edges(curr, data=True)) + list(
                self.graph.in_edges(curr, data=True)
            )
            for u, v, data in neighbors:
                edge = self._edge_from_data(data)
                if pred_filter and edge.predicate not in pred_filter:
                    continue
                collected_edges.append(edge)
                nxt = v if u == curr else u
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append((nxt, depth + 1))

        nodes = [GraphNode(**self.graph.nodes[n]).model_dump() for n in sorted(visited)]
        return {
            "root": start_id,
            "nodes": nodes,
            "edges": [e.model_dump() for e in collected_edges],
        }

    # ── Serialization & Persistence ────────────────────────────────────────────

    def to_graph_json(self) -> dict[str, Any]:
        """Export frontend-ready JSON ({nodes, edges, stats}) for React Flow / Cytoscape."""
        nodes = [GraphNode(**data).model_dump() for _, data in self.graph.nodes(data=True)]
        edges = [self._edge_from_data(data).model_dump() for _, _, data in self.graph.edges(data=True)]
        return {
            "title": self.title,
            "nodes": nodes,
            "edges": edges,
            "stats": {
                "node_count": len(nodes),
                "edge_count": len(edges),
                "conflict_count": len(self.find_conflicting_edges()),
            },
        }

    def save_json(self, path: str | Path) -> Path:
        """Persist the knowledge graph to a JSON file on disk."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_graph_json(), indent=2), encoding="utf-8")
        return target

    @classmethod
    def load_json(cls, path: str | Path) -> "NarrativeKnowledgeGraph":
        """Load a previously saved knowledge graph JSON file."""
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        kg = cls(title=raw.get("title", "Untitled"))
        for n in raw.get("nodes", []):
            node = GraphNode(**n)
            kg.graph.add_node(node.id, **node.model_dump())
            kg._alias_index[node.id] = node.id
            for alias in node.aliases:
                kg._alias_index[_normalize_id(alias)] = node.id
        for e in raw.get("edges", []):
            edge = GraphEdge(**e)
            kg.graph.add_edge(edge.source, edge.target, **edge.model_dump())
        return kg

    # ── Internal Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _infer_node_types(predicate: str) -> tuple[NodeType, NodeType]:
        pred = predicate.lower()
        if pred == "present_in":
            return "Character", "Scene"
        if pred == "located_at":
            return "Scene", "Location"
        if pred == "occurs_at":
            return "Scene", "Concept"
        if pred in _POSSESSION_PREDICATES:
            return "Character", "Object"
        if pred in ("has_status", "has_injury", "knows"):
            return "Character", "Concept"
        return "Concept", "Concept"

    @staticmethod
    def _edge_from_data(data: dict[str, Any]) -> GraphEdge:
        return GraphEdge(
            source=data["source"],
            target=data["target"],
            predicate=data["predicate"],
            scene_id=data.get("scene_id"),
            source_segment=data.get("source_segment"),
            confidence=data.get("confidence", 1.0),
            evidence=data.get("evidence"),
            metadata=data.get("metadata", {}),
        )
