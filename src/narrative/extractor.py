"""Entity, fact, knowledge, and temporal extraction pipeline for TRACE (Issue #55).

Combines a fast, non-autoregressive System 1 decision engine (Laya) with
deterministic structural extraction and selective System 2 LLM escalation
(via LangChain structured outputs) to extract:
  1. Subject-Predicate-Object (SPO) fact triples
  2. Character epistemic knowledge assertions (who learns/knows what)
  3. Relative temporal markers (before, after, simultaneous_with)
"""

from __future__ import annotations

import re
import threading
from typing import Any, Literal

from pydantic import BaseModel, Field

from src.ml.llm import get_llm
from src.narrative.parsers.base import NarrativeDocument
from src.narrative.segmenter import NarrativeSegment, segment_narrative
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Pydantic v2 Extraction Schemas ─────────────────────────────────────────────


class FactTriple(BaseModel):
    """A Subject-Predicate-Object factual claim anchored to a scene."""

    subject: str = Field(..., min_length=1)
    predicate: str = Field(..., min_length=1)
    object: str = Field(..., min_length=1)
    scene_id: str = Field(..., min_length=1)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source: Literal["system1_deterministic", "laya", "llm"] = "system1_deterministic"
    evidence: str | None = Field(default=None)


class KnowledgeAssertion(BaseModel):
    """An epistemic assertion about what a character knows or does not know."""

    character: str = Field(..., min_length=1)
    fact: str = Field(..., min_length=1)
    knows: bool = Field(
        default=True,
        description="True if the character learns/knows the fact; False if explicitly unaware.",
    )
    scene_id: str = Field(..., min_length=1)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    evidence: str | None = Field(default=None)


class TemporalMarker(BaseModel):
    """A relative temporal ordering link between events or scenes."""

    source_event: str = Field(..., min_length=1)
    relation: Literal["before", "after", "simultaneous_with"]
    target_event: str = Field(..., min_length=1)
    scene_id: str = Field(..., min_length=1)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    cue: str | None = Field(default=None)


class LayaSegmentTriage(BaseModel):
    """Typed System 1 schema evaluated in a single pass by Laya."""

    has_physical_state_change: bool = Field(
        ...,
        description="Does a character's physical state, injury, survival, or possession change?",
    )
    has_epistemic_event: bool = Field(
        ...,
        description="Does a character learn, reveal, conceal, or act upon a secret or key fact?",
    )
    temporal_order: Literal["chronological", "flashback", "flash_forward", "simultaneous"] = Field(
        ...,
        description="How this scene relates in time to the preceding narrative.",
    )
    continuity_salience: int = Field(
        ...,
        ge=1,
        le=5,
        description="1 (filler/trivial) to 5 (critical plot/state/timeline event).",
    )


class SegmentExtraction(BaseModel):
    """All extracted facts, knowledge assertions, and temporal markers for one segment."""

    segment_id: str = Field(..., min_length=1)
    scene_id: str = Field(..., min_length=1)
    triples: list[FactTriple] = Field(default_factory=list)
    knowledge_assertions: list[KnowledgeAssertion] = Field(default_factory=list)
    temporal_markers: list[TemporalMarker] = Field(default_factory=list)
    triage: LayaSegmentTriage | None = Field(default=None)
    used_system2: bool = Field(default=False)


class DocumentExtraction(BaseModel):
    """Aggregated extractions across an entire NarrativeDocument."""

    title: str = Field(default="Untitled")
    segments: list[SegmentExtraction] = Field(default_factory=list)

    @property
    def all_triples(self) -> list[FactTriple]:
        return [t for seg in self.segments for t in seg.triples]

    @property
    def all_knowledge_assertions(self) -> list[KnowledgeAssertion]:
        return [k for seg in self.segments for k in seg.knowledge_assertions]

    @property
    def all_temporal_markers(self) -> list[TemporalMarker]:
        return [m for seg in self.segments for m in seg.temporal_markers]


# ── Internal LLM Structured Output Schema ──────────────────────────────────────


class _LlmTriple(BaseModel):
    subject: str = Field(..., min_length=1)
    predicate: str = Field(..., min_length=1)
    object: str = Field(..., min_length=1)
    evidence: str | None = None


class _LlmKnowledge(BaseModel):
    character: str = Field(..., min_length=1)
    fact: str = Field(..., min_length=1)
    knows: bool = True
    evidence: str | None = None


class _LlmTemporal(BaseModel):
    source_event: str = Field(..., min_length=1)
    relation: Literal["before", "after", "simultaneous_with"]
    target_event: str = Field(..., min_length=1)
    cue: str | None = None


class _LlmSegmentPayload(BaseModel):
    triples: list[_LlmTriple] = Field(default_factory=list)
    knowledge_assertions: list[_LlmKnowledge] = Field(default_factory=list)
    temporal_markers: list[_LlmTemporal] = Field(default_factory=list)


# ── Thread-Safe Laya Singleton ─────────────────────────────────────────────────

_laya_agent: Any = None
_laya_lock = threading.Lock()


def get_laya_agent(model_id: str | None = None) -> Any:
    """Lazy-load and cache the Laya System 1 agent using double-checked locking."""
    global _laya_agent
    if _laya_agent is not None:
        return _laya_agent

    with _laya_lock:
        if _laya_agent is not None:
            return _laya_agent

        import laya

        logger.info("Loading Laya System 1 model", extra={"model_id": model_id or "default"})
        _laya_agent = laya.load(model_id)
        logger.info("Laya System 1 model loaded")
        return _laya_agent


# ── Deterministic System 1 Lexical Rules ───────────────────────────────────────

_BEFORE_CUE_RE = re.compile(r"\b(FLASHBACK|EARLIER|BEFORE|YEARS?\s+AGO|PREVIOUSLY)\b", re.IGNORECASE)
_AFTER_CUE_RE = re.compile(r"\b(FLASH\s+FORWARD|LATER|THE\s+NEXT\s+(?:DAY|MORNING|NIGHT)|AFTERWARDS)\b", re.IGNORECASE)
_SIMUL_CUE_RE = re.compile(r"\b(MEANWHILE|SIMULTANEOUSLY|AT\s+THE\s+SAME\s+TIME|CONTINUOUS)\b", re.IGNORECASE)

_DEATH_RE = re.compile(
    r"\b([A-Z][A-Za-z0-9_'\-]{1,25})\s+(?:is\s+dead|was\s+killed|dies|died)\b",
)


def _extract_deterministic_system1(
    segment: NarrativeSegment,
    prev_scene_id: str | None,
) -> tuple[list[FactTriple], list[TemporalMarker]]:
    """Extract zero-cost structural triples and explicit lexical temporal/state cues."""
    triples: list[FactTriple] = []
    markers: list[TemporalMarker] = []

    for char in segment.characters_present:
        triples.append(
            FactTriple(
                subject=char,
                predicate="present_in",
                object=segment.scene_id,
                scene_id=segment.scene_id,
                source="system1_deterministic",
            )
        )

    if segment.location:
        triples.append(
            FactTriple(
                subject=segment.scene_id,
                predicate="located_at",
                object=segment.location,
                scene_id=segment.scene_id,
                source="system1_deterministic",
            )
        )

    if segment.time_anchor:
        triples.append(
            FactTriple(
                subject=segment.scene_id,
                predicate="occurs_at",
                object=segment.time_anchor,
                scene_id=segment.scene_id,
                source="system1_deterministic",
            )
        )

    for m in _DEATH_RE.finditer(segment.text):
        triples.append(
            FactTriple(
                subject=m.group(1),
                predicate="has_status",
                object="dead",
                scene_id=segment.scene_id,
                confidence=0.85,
                source="system1_deterministic",
                evidence=m.group(0),
            )
        )

    if prev_scene_id and prev_scene_id != segment.scene_id:
        anchor_text = f"{segment.time_anchor or ''} {segment.heading}".strip()
        if _BEFORE_CUE_RE.search(anchor_text):
            rel: Literal["before", "after", "simultaneous_with"] = "before"
        elif _SIMUL_CUE_RE.search(anchor_text):
            rel = "simultaneous_with"
        else:
            rel = "after"

        markers.append(
            TemporalMarker(
                source_event=segment.scene_id,
                relation=rel,
                target_event=prev_scene_id,
                scene_id=segment.scene_id,
                confidence=0.9 if _BEFORE_CUE_RE.search(anchor_text) or _AFTER_CUE_RE.search(anchor_text) else 0.75,
                cue=segment.time_anchor,
            )
        )

    return triples, markers


# ── Main Extractor Orchestrator ────────────────────────────────────────────────


class NarrativeExtractor:
    """Two-tier Hybrid Extractor combining Laya (System 1) + LLM (System 2)."""

    def __init__(
        self,
        mode: Literal["hybrid", "laya", "system1", "llm"] = "hybrid",
        laya_agent: Any | None = None,
        laya_model_id: str | None = None,
        min_confidence: float | None = None,
        salience_threshold: int = 3,
    ) -> None:
        self.mode = mode
        self._laya_agent = laya_agent
        self.laya_model_id = laya_model_id
        self.min_confidence = min_confidence
        self.salience_threshold = salience_threshold

    def extract_segment(
        self,
        segment: NarrativeSegment,
        prev_scene_id: str | None = None,
    ) -> SegmentExtraction:
        """Extract facts, knowledge assertions, and temporal markers for one segment."""
        triples, markers = _extract_deterministic_system1(segment, prev_scene_id)
        knowledge: list[KnowledgeAssertion] = []
        triage: LayaSegmentTriage | None = None
        used_system2 = False

        # Tier 1: Laya System 1 triage
        if self.mode in ("hybrid", "laya"):
            triage = self._run_laya_triage(segment)
            if triage is not None and prev_scene_id and prev_scene_id != segment.scene_id:
                laya_rel = self._triage_to_temporal(triage)
                if laya_rel is not None:
                    markers.append(
                        TemporalMarker(
                            source_event=segment.scene_id,
                            relation=laya_rel,
                            target_event=prev_scene_id,
                            scene_id=segment.scene_id,
                            confidence=0.9,
                            cue=f"laya:{triage.temporal_order}",
                        )
                    )

        # Tier 2: Selective System 2 LLM escalation
        should_escalate = self.mode == "llm" or (
            self.mode == "hybrid" and self._should_escalate(triage)
        )
        if should_escalate:
            llm_payload = self._run_llm_extraction(segment)
            if llm_payload is not None:
                used_system2 = True
                for t in llm_payload.triples:
                    triples.append(
                        FactTriple(
                            subject=t.subject,
                            predicate=t.predicate,
                            object=t.object,
                            scene_id=segment.scene_id,
                            confidence=0.95,
                            source="llm",
                            evidence=t.evidence,
                        )
                    )
                for k in llm_payload.knowledge_assertions:
                    knowledge.append(
                        KnowledgeAssertion(
                            character=k.character,
                            fact=k.fact,
                            knows=k.knows,
                            scene_id=segment.scene_id,
                            confidence=0.95,
                            evidence=k.evidence,
                        )
                    )
                for m in llm_payload.temporal_markers:
                    markers.append(
                        TemporalMarker(
                            source_event=m.source_event,
                            relation=m.relation,
                            target_event=m.target_event,
                            scene_id=segment.scene_id,
                            confidence=0.95,
                            cue=m.cue,
                        )
                    )

        deduped_triples = self._dedupe_triples(triples)
        return SegmentExtraction(
            segment_id=segment.segment_id,
            scene_id=segment.scene_id,
            triples=deduped_triples,
            knowledge_assertions=knowledge,
            temporal_markers=markers,
            triage=triage,
            used_system2=used_system2,
        )

    def extract_document(
        self,
        document: NarrativeDocument,
        segments: list[NarrativeSegment] | None = None,
    ) -> DocumentExtraction:
        """Run extraction sequentially across all segments of a NarrativeDocument."""
        segs = segments if segments is not None else segment_narrative(document)
        results: list[SegmentExtraction] = []
        prev_scene_id: str | None = None

        for seg in segs:
            results.append(self.extract_segment(seg, prev_scene_id=prev_scene_id))
            prev_scene_id = seg.scene_id

        logger.info(
            "Narrative extraction complete",
            extra={
                "title": document.title,
                "mode": self.mode,
                "segments": len(results),
                "system2_calls": sum(1 for r in results if r.used_system2),
            },
        )
        return DocumentExtraction(title=document.title, segments=results)

    def _run_laya_triage(self, segment: NarrativeSegment) -> LayaSegmentTriage | None:
        """Evaluate segment using Laya's non-autoregressive System 1 decide() API."""
        try:
            import laya

            agent = self._laya_agent or get_laya_agent(self.laya_model_id)
            decision = laya.decide(
                agent,
                segment.text,
                schema=LayaSegmentTriage,
                min_confidence=self.min_confidence,
                return_details=True,
            )
            values = decision.values if hasattr(decision, "values") else decision
            return LayaSegmentTriage.model_validate(values)
        except Exception as exc:
            logger.warning(
                "Laya System 1 triage failed; continuing with fallback",
                extra={"segment_id": segment.segment_id, "error": str(exc)},
            )
            return None

    def _should_escalate(self, triage: LayaSegmentTriage | None) -> bool:
        """Gate System 2 LLM calls: only escalate when Laya detects salient plot/state/epistemic signals."""
        if triage is None:
            return True
        return (
            triage.has_physical_state_change
            or triage.has_epistemic_event
            or triage.continuity_salience >= self.salience_threshold
        )

    @staticmethod
    def _triage_to_temporal(
        triage: LayaSegmentTriage,
    ) -> Literal["before", "after", "simultaneous_with"] | None:
        mapping: dict[str, Literal["before", "after", "simultaneous_with"]] = {
            "flashback": "before",
            "flash_forward": "after",
            "simultaneous": "simultaneous_with",
        }
        return mapping.get(triage.temporal_order)

    @staticmethod
    def _run_llm_extraction(segment: NarrativeSegment) -> _LlmSegmentPayload | None:
        """System 2 extraction via LangChain structured output."""
        try:
            llm = get_llm()
            structured_llm = llm.with_structured_output(_LlmSegmentPayload)
            system_prompt = (
                "You are a narrative continuity analyst. Extract only explicitly supported:\n"
                "1. Subject-Predicate-Object fact triples (injuries, possessions, deaths, locations, relationships).\n"
                "2. Character knowledge assertions (who learns, knows, or is explicitly kept unaware of a fact).\n"
                "3. Temporal ordering markers (before, after, simultaneous_with) relative to other events."
            )
            human_prompt = (
                f"Scene ID: {segment.scene_id}\n"
                f"Heading: {segment.heading}\n"
                f"Characters: {', '.join(segment.characters_present) or 'Unknown'}\n\n"
                f"Text:\n{segment.text}"
            )
            return structured_llm.invoke([("system", system_prompt), ("human", human_prompt)])
        except Exception as exc:
            logger.warning(
                "System 2 LLM extraction failed",
                extra={"segment_id": segment.segment_id, "error": str(exc)},
            )
            return None

    @staticmethod
    def _dedupe_triples(triples: list[FactTriple]) -> list[FactTriple]:
        seen: set[tuple[str, str, str, str]] = set()
        unique: list[FactTriple] = []
        for t in triples:
            key = (t.subject.lower(), t.predicate.lower(), t.object.lower(), t.scene_id)
            if key not in seen:
                seen.add(key)
                unique.append(t)
        return unique
