"""Unit tests for TRACE Knowledge Graph Builder (Issue #56)."""

from __future__ import annotations

import uuid
from pathlib import Path

from src.narrative.extractor import (
    DocumentExtraction,
    FactTriple,
    NarrativeEvent,
    SegmentExtraction,
    save_extraction,
)
from src.narrative.graph import NarrativeKnowledgeGraph
from src.storage.db.models import Document


def _sample_extraction() -> DocumentExtraction:
    return DocumentExtraction(
        title="Test Script",
        segments=[
            SegmentExtraction(
                segment_id="scene_0001_seg_0",
                scene_id="scene_0001",
                triples=[
                    FactTriple(
                        subject="BRUCE",
                        predicate="present_in",
                        object="scene_0001",
                        scene_id="scene_0001",
                        source_segment="scene_0001_seg_0",
                    ),
                    FactTriple(
                        subject="BRUCE",
                        predicate="holds",
                        object="silver_key",
                        scene_id="scene_0001",
                        source_segment="scene_0001_seg_0",
                    ),
                    FactTriple(
                        subject="BRUCE",
                        predicate="has_status",
                        object="dead",
                        scene_id="scene_0001",
                        source_segment="scene_0001_seg_0",
                    ),
                ],
                events=[
                    NarrativeEvent(
                        description="Bruce drops the silver key.",
                        characters=["BRUCE"],
                        location="VAULT",
                        temporal_order=0,
                        source_segment="scene_0001_seg_0",
                        scene_id="scene_0001",
                    )
                ],
            ),
            SegmentExtraction(
                segment_id="scene_0002_seg_0",
                scene_id="scene_0002",
                triples=[
                    FactTriple(
                        subject="Bruce",
                        predicate="has_status",
                        object="alive",
                        scene_id="scene_0002",
                        source_segment="scene_0002_seg_0",
                    ),
                ],
            ),
        ],
    )


def test_graph_builds_typed_nodes_and_queries_entity_facts() -> None:
    kg = NarrativeKnowledgeGraph.from_extraction(_sample_extraction())

    bruce_node = kg.get_node("BRUCE")
    assert bruce_node is not None
    assert bruce_node.node_type == "Character"
    assert "Bruce" in bruce_node.aliases

    key_node = kg.get_node("silver_key")
    assert key_node is not None
    assert key_node.node_type == "Object"

    facts = kg.get_entity_facts("bruce")
    predicates = {e.predicate for e in facts}
    assert {"present_in", "holds", "has_status", "participated_in"} <= predicates


def test_graph_detects_conflicting_edges_and_traverses() -> None:
    kg = NarrativeKnowledgeGraph.from_extraction(_sample_extraction())

    conflicts = kg.find_conflicting_edges()
    assert len(conflicts) == 1
    assert conflicts[0].entity_id == "bruce"
    assert conflicts[0].predicate == "has_status"
    assert {conflicts[0].edge_a.target, conflicts[0].edge_b.target} == {"dead", "alive"}

    sub = kg.traverse_relationships("BRUCE", max_hops=2)
    node_ids = {n["id"] for n in sub["nodes"]}
    assert "bruce" in node_ids
    assert "silver_key" in node_ids
    assert "vault" in node_ids


def test_graph_json_roundtrip_and_db_reconstruction(tmp_path: Path, db_session) -> None:
    extraction = _sample_extraction()
    kg = NarrativeKnowledgeGraph.from_extraction(extraction)

    out_path = kg.save_json(tmp_path / "graph.json")
    loaded = NarrativeKnowledgeGraph.load_json(out_path)
    assert loaded.to_graph_json()["stats"]["node_count"] == kg.to_graph_json()["stats"]["node_count"]
    assert len(loaded.find_conflicting_edges()) == 1

    doc_id = uuid.uuid4()
    db_session.add(Document(id=doc_id, filename="test.fountain", raw_text="..."))
    db_session.commit()
    save_extraction(doc_id, extraction, db=db_session)

    db_kg = NarrativeKnowledgeGraph.from_db(doc_id, db=db_session)
    assert db_kg.get_node("BRUCE") is not None
    assert len(db_kg.find_conflicting_edges()) == 1
