"""Unit tests for TRACE Entity & Fact Extraction Pipeline (Issue #55)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from src.narrative.extractor import (
    LayaSegmentTriage,
    NarrativeExtractor,
    _LlmEvent,
    _LlmKnowledge,
    _LlmSegmentPayload,
    _LlmTemporal,
    _LlmTriple,
    _LlmWorldRule,
    save_extraction,
)
from src.narrative.parsers import FountainParser
from src.narrative.segmenter import NarrativeSegment
from src.storage.db.models import (
    Document,
    NarrativeEventRecord,
    NarrativeFact,
    NarrativeKnowledgeRecord,
    WorldRuleRecord,
)


class _FakeLayaRunner:
    """In-memory stub matching Laya's runner.predict(state, questions) contract."""

    def __init__(self, answers: dict[str, dict[str, Any]]) -> None:
        self._answers = answers
        self.calls: int = 0

    def predict(self, state: Any, questions: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        self.calls += 1
        return {"answers": self._answers}


def test_deterministic_system1_extracts_structural_and_lexical_facts() -> None:
    seg = NarrativeSegment(
        segment_id="scene_0002_seg_0",
        scene_id="scene_0002",
        scene_number=2,
        segment_index=1,
        heading="INT. HOSPITAL - NIGHT",
        location="HOSPITAL",
        time_anchor="FLASHBACK",
        characters_present=["MICHAEL", "SONNY"],
        text="FLASHBACK. Sollozzo is dead on the floor. MICHAEL checks his coat.",
    )

    ext = NarrativeExtractor(mode="system1").extract_segment(seg, prev_scene_id="scene_0001")

    assert ext.used_system2 is False
    assert ext.triage is None
    triple_tuples = {(t.subject, t.predicate, t.object) for t in ext.triples}
    assert ("MICHAEL", "present_in", "scene_0002") in triple_tuples
    assert ("scene_0002", "located_at", "HOSPITAL") in triple_tuples
    assert ("scene_0002", "occurs_at", "FLASHBACK") in triple_tuples
    assert ("Sollozzo", "has_status", "dead") in triple_tuples

    assert len(ext.temporal_markers) == 1
    assert ext.temporal_markers[0].relation == "before"


@patch("src.narrative.extractor.get_llm")
def test_laya_hybrid_skips_system2_on_low_salience_segment(mock_get_llm: MagicMock) -> None:
    fake_laya = _FakeLayaRunner(
        {
            "has_physical_state_change": {"noul": 0.04, "confidence": 0.96},
            "has_epistemic_event": {"noul": 0.06, "confidence": 0.94},
            "temporal_order": {"choice": "chronological", "confidence": 0.92},
            "continuity_salience": {"score": 0.0, "confidence": 0.90},
        }
    )
    seg = NarrativeSegment(
        segment_id="scene_0001_seg_0",
        scene_id="scene_0001",
        scene_number=1,
        segment_index=0,
        heading="EXT. STREET - DAY",
        characters_present=["TOM"],
        text="TOM walks down the street and buys a newspaper.",
    )

    ext = NarrativeExtractor(mode="hybrid", laya_agent=fake_laya).extract_segment(seg)

    assert fake_laya.calls == 1
    assert isinstance(ext.triage, LayaSegmentTriage)
    assert ext.triage.continuity_salience == 1
    assert ext.used_system2 is False
    mock_get_llm.assert_not_called()


@patch("src.narrative.extractor.get_llm")
def test_laya_hybrid_escalates_to_system2_on_epistemic_or_state_signal(
    mock_get_llm: MagicMock,
) -> None:
    fake_laya = _FakeLayaRunner(
        {
            "has_physical_state_change": {"noul": 0.91, "confidence": 0.91},
            "has_epistemic_event": {"noul": 0.89, "confidence": 0.89},
            "temporal_order": {"choice": "flashback", "confidence": 0.88},
            "continuity_salience": {"score": 4.0, "confidence": 0.95},
        }
    )

    mock_structured = MagicMock()
    mock_structured.invoke.return_value = _LlmSegmentPayload(
        triples=[
            _LlmTriple(
                subject="BRUCE",
                predicate="has_injury",
                object="broken_leg",
                evidence="Bruce limps with a broken leg.",
            )
        ],
        events=[
            _LlmEvent(
                description="Bruce warns the team not to inform Michael about the ambush.",
                characters=["BRUCE", "MICHAEL"],
                location="SAFEHOUSE",
                temporal_order="1",
            )
        ],
        world_rules=[
            _LlmWorldRule(
                rule="No family member may enter the safehouse armed.",
                evidence="Leave your guns at the door.",
            )
        ],
        knowledge_assertions=[
            _LlmKnowledge(
                character="MICHAEL",
                fact="ambush_at_restaurant",
                knows=False,
                evidence="Don't tell Michael about the ambush yet.",
            )
        ],
        temporal_markers=[
            _LlmTemporal(
                source_event="bruce_injury",
                relation="before",
                target_event="scene_0002",
                cue="earlier",
            )
        ],
    )
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured
    mock_get_llm.return_value = mock_llm

    script = """INT. SAFEHOUSE - NIGHT

BRUCE
Don't tell Michael about the ambush yet.
"""
    doc = FountainParser().parse_text(script, source_path="test.fountain")
    result = NarrativeExtractor(mode="hybrid", laya_agent=fake_laya).extract_document(doc)

    assert result.title == "Untitled"
    assert len(result.segments) == 1
    assert isinstance(result.segments[0].triage, LayaSegmentTriage)
    assert result.segments[0].triage.continuity_salience == 5
    assert result.segments[0].used_system2 is True
    assert any(
        t.subject == "BRUCE" and t.predicate == "has_injury" and t.source == "llm"
        and t.source_segment == "scene_0001_seg_0"
        for t in result.all_triples
    )
    assert len(result.all_events) == 1
    assert result.all_events[0].location == "SAFEHOUSE"
    assert len(result.all_world_rules) == 1
    assert result.all_world_rules[0].established_in == "scene_0001_seg_0"
    assert len(result.all_knowledge_assertions) == 1
    assert result.all_knowledge_assertions[0].knows is False
    assert any(m.relation == "before" for m in result.all_temporal_markers)


@patch("src.narrative.extractor.get_llm")
def test_save_extraction_persists_to_database(mock_get_llm: MagicMock, db_session) -> None:
    import uuid

    fake_laya = _FakeLayaRunner(
        {
            "has_physical_state_change": {"noul": 0.90, "confidence": 0.90},
            "has_epistemic_event": {"noul": 0.85, "confidence": 0.85},
            "temporal_order": {"choice": "chronological", "confidence": 0.92},
            "continuity_salience": {"score": 4.0, "confidence": 0.90},
        }
    )
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = _LlmSegmentPayload(
        triples=[_LlmTriple(subject="ALICE", predicate="holds", object="key")],
        events=[_LlmEvent(description="Alice unlocks the vault.", characters=["ALICE"], location="VAULT")],
        world_rules=[_LlmWorldRule(rule="Only the keyholder may open the vault.")],
        knowledge_assertions=[_LlmKnowledge(character="BOB", fact="vault_is_open", knows=False)],
    )
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured
    mock_get_llm.return_value = mock_llm

    doc_id = uuid.uuid4()
    db_session.add(Document(id=doc_id, filename="vault.fountain", raw_text="INT. VAULT - NIGHT"))
    db_session.commit()

    doc = FountainParser().parse_text("INT. VAULT - NIGHT\n\nALICE\nI have the key.\n")
    extraction = NarrativeExtractor(mode="hybrid", laya_agent=fake_laya).extract_document(doc)
    counts = save_extraction(doc_id, extraction, db=db_session)

    assert counts["facts"] >= 1
    assert counts["events"] == 1
    assert counts["world_rules"] == 1
    assert counts["knowledge"] == 1
    assert db_session.query(NarrativeFact).filter_by(document_id=doc_id).count() == counts["facts"]
    assert db_session.query(NarrativeEventRecord).filter_by(document_id=doc_id).count() == 1
    assert db_session.query(WorldRuleRecord).filter_by(document_id=doc_id).count() == 1
    assert db_session.query(NarrativeKnowledgeRecord).filter_by(document_id=doc_id).count() == 1
