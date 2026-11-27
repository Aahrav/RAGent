"""Narrative-aware segmentation for TRACE (Issue #54).

Splits parsed NarrativeDocument objects along scene and element boundaries
rather than arbitrary character/token counts, attaching rich narrative metadata
({scene_id, act, chapter, characters_present, time_anchor}) to each segment.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

from src.narrative.parsers.base import (
    Action,
    CharacterDialogue,
    NarrativeDocument,
    NarrativeElement,
    Scene,
)
from src.services.rag.models import Chunk
from src.utils.logger import get_logger

logger = get_logger(__name__)

_ACT_RE = re.compile(
    r"\b(ACT\s+(?:[IVX]+|\d+|ONE|TWO|THREE|FOUR|FIVE)|PROLOGUE|EPILOGUE)\b",
    re.IGNORECASE,
)

_TEMPORAL_CUE_RE = re.compile(
    r"\b("
    r"FLASHBACK|FLASH\s+FORWARD|PRESENT\s+DAY|"
    r"(?:\d+|ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN|SEVERAL|MANY)\s+"
    r"(?:SECONDS?|MINUTES?|HOURS?|DAYS?|WEEKS?|MONTHS?|YEARS?)\s+(?:LATER|EARLIER|BEFORE|AGO)|"
    r"THE\s+NEXT\s+(?:DAY|MORNING|NIGHT|WEEK)|"
    r"MOMENTS?\s+LATER|LATER\s+THAT\s+(?:DAY|NIGHT|EVENING)|"
    r"CONTINUOUS|MEANWHILE|SIMULTANEOUSLY"
    r")\b",
    re.IGNORECASE,
)


class NarrativeSegment(BaseModel):
    """A retrieval- and extraction-ready narrative unit bounded by scene/element structure."""

    segment_id: str = Field(..., min_length=1)
    scene_id: str = Field(..., min_length=1)
    scene_number: int = Field(..., ge=1)
    segment_index: int = Field(..., ge=0)
    sub_index: int = Field(default=0, ge=0)
    act: str | None = Field(default=None)
    chapter: str | None = Field(default=None)
    heading: str = Field(default="")
    location: str | None = Field(default=None)
    time_of_day: str | None = Field(default=None)
    characters_present: list[str] = Field(default_factory=list)
    time_anchor: str | None = Field(default=None)
    text: str = Field(..., min_length=1)
    page_start: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)
    timestamp_start: str | None = Field(default=None)
    timestamp_end: str | None = Field(default=None)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_rag_chunk(self, doc_id: str = "", source: str | None = None) -> Chunk:
        """Convert this segment into a RAG Chunk with narrative metadata in its payload."""
        return Chunk(
            text=self.text,
            source=source or self.metadata.get("source_path") or self.metadata.get("document_title", "narrative"),
            page=self.page_start or 1,
            chunk_index=self.segment_index,
            doc_id=doc_id,
            metadata={
                "scene_id": self.scene_id,
                "segment_id": self.segment_id,
                "act": self.act,
                "chapter": self.chapter,
                "characters_present": self.characters_present,
                "time_anchor": self.time_anchor,
                "heading": self.heading,
                "location": self.location,
                "time_of_day": self.time_of_day,
            },
        )


def _format_element(el: NarrativeElement) -> str:
    """Render a CharacterDialogue or Action element into clean, readable text."""
    if isinstance(el, CharacterDialogue):
        header = el.character
        if el.extension:
            header = f"{header} ({el.extension})"
        parts = [f"{header}:"]
        if el.parenthetical:
            parts.append(el.parenthetical)
        parts.append(el.dialogue)
        return " ".join(parts)
    return el.text


def _detect_act(scene: Scene, current_act: str | None) -> str | None:
    """Detect an act marker in the scene heading, chapter, or opening elements."""
    for candidate in (scene.heading, scene.chapter or ""):
        m = _ACT_RE.search(candidate)
        if m:
            return m.group(1).upper()

    for el in scene.elements[:3]:
        if isinstance(el, Action):
            m = _ACT_RE.search(el.text)
            if m:
                return m.group(1).upper()

    return current_act


def _extract_time_anchor(scene: Scene, elements: list[NarrativeElement]) -> str | None:
    """Determine the best temporal anchor for a scene or sub-segment."""
    # 1. Explicit element/scene timestamps (e.g. SRT)
    for el in elements:
        if el.timestamp_start:
            return el.timestamp_start
    if scene.timestamp_start:
        return scene.timestamp_start

    # 2. Explicit temporal cue in heading or opening action lines
    m = _TEMPORAL_CUE_RE.search(scene.heading)
    if m:
        return m.group(1).upper()

    for el in elements[:2]:
        if isinstance(el, Action):
            m = _TEMPORAL_CUE_RE.search(el.text)
            if m:
                return m.group(1).upper()

    # 3. Screenplay slugline time of day (e.g. NIGHT, DAWN)
    if scene.time_of_day:
        return scene.time_of_day.upper()

    return None


def _characters_in_elements(elements: list[NarrativeElement]) -> list[str]:
    """Return unique character names appearing in the given elements, preserving order."""
    seen: dict[str, None] = {}
    for el in elements:
        if isinstance(el, CharacterDialogue):
            seen.setdefault(el.character, None)
    return list(seen.keys())


def _split_oversized_text(text: str, max_chars: int) -> list[str]:
    """Fallback splitter for a single prose/action block that exceeds max_chars."""
    if len(text) <= max_chars:
        return [text]

    sentences = re.split(r"(?<=[.!?])\s+", text)
    pieces: list[str] = []
    buf: list[str] = []
    buf_len = 0

    for sent in sentences:
        if len(sent) > max_chars:
            if buf:
                pieces.append(" ".join(buf))
                buf = []
                buf_len = 0
            # Hard word/char wrap as last resort
            for i in range(0, len(sent), max_chars):
                pieces.append(sent[i : i + max_chars].strip())
            continue

        added = len(sent) + (1 if buf else 0)
        if buf_len + added > max_chars and buf:
            pieces.append(" ".join(buf))
            buf = [sent]
            buf_len = len(sent)
        else:
            buf.append(sent)
            buf_len += added

    if buf:
        pieces.append(" ".join(buf))
    return [p for p in pieces if p]


class NarrativeSegmenter:
    """Segments a NarrativeDocument along scene and element boundaries."""

    def __init__(self, max_chars: int = 2000, overlap_elements: int = 1) -> None:
        if max_chars <= 0:
            raise ValueError("max_chars must be positive")
        if overlap_elements < 0:
            raise ValueError("overlap_elements cannot be negative")
        self.max_chars = max_chars
        self.overlap_elements = overlap_elements

    def segment(self, document: NarrativeDocument) -> list[NarrativeSegment]:
        """Convert a parsed NarrativeDocument into ordered NarrativeSegment objects."""
        segments: list[NarrativeSegment] = []
        current_act: str | None = None

        for scene in document.scenes:
            current_act = _detect_act(scene, current_act)
            scene_id = f"scene_{scene.scene_number:04d}"

            element_groups = self._group_scene_elements(scene)
            if not element_groups:
                continue

            for sub_idx, (group_els, group_text) in enumerate(element_groups):
                chars = _characters_in_elements(group_els)
                time_anchor = _extract_time_anchor(scene, group_els)
                segment_id = f"{scene_id}_seg_{sub_idx}"

                meta: dict[str, Any] = {
                    "scene_id": scene_id,
                    "act": current_act,
                    "chapter": scene.chapter,
                    "characters_present": chars,
                    "time_anchor": time_anchor,
                    "format": document.format,
                    "document_title": document.title,
                    "source_path": document.source_path,
                    "scene_label": scene.scene_label,
                    "int_ext": scene.int_ext,
                    "location": scene.location,
                    "time_of_day": scene.time_of_day,
                }

                segments.append(
                    NarrativeSegment(
                        segment_id=segment_id,
                        scene_id=scene_id,
                        scene_number=scene.scene_number,
                        segment_index=len(segments),
                        sub_index=sub_idx,
                        act=current_act,
                        chapter=scene.chapter,
                        heading=scene.heading,
                        location=scene.location,
                        time_of_day=scene.time_of_day,
                        characters_present=chars,
                        time_anchor=time_anchor,
                        text=group_text,
                        page_start=scene.page_start,
                        page_end=scene.page_end,
                        timestamp_start=scene.timestamp_start,
                        timestamp_end=scene.timestamp_end,
                        metadata=meta,
                    )
                )

        logger.info(
            "Narrative segmentation complete",
            extra={
                "title": document.title,
                "scenes": document.total_scenes,
                "segments": len(segments),
            },
        )
        return segments

    def _group_scene_elements(
        self, scene: Scene
    ) -> list[tuple[list[NarrativeElement], str]]:
        """Group a scene's elements into chunks <= max_chars without breaking elements."""
        if not scene.elements:
            raw = scene.raw_text.strip() or scene.heading.strip()
            if not raw:
                return []
            return [([], p) for p in _split_oversized_text(raw, self.max_chars)]

        header_prefix = f"[{scene.heading}]\n" if scene.heading else ""
        groups: list[tuple[list[NarrativeElement], str]] = []
        current_els: list[NarrativeElement] = []
        current_parts: list[str] = []
        current_len = len(header_prefix)

        def flush() -> None:
            nonlocal current_els, current_parts, current_len
            if current_parts:
                body = "\n".join(current_parts)
                text = f"{header_prefix}{body}".strip()
                groups.append((list(current_els), text))
            current_els = []
            current_parts = []
            current_len = len(header_prefix)

        for el in scene.elements:
            rendered = _format_element(el)

            # Handle a single element that itself exceeds max_chars
            if len(header_prefix) + len(rendered) > self.max_chars:
                flush()
                budget = max(100, self.max_chars - len(header_prefix))
                for piece in _split_oversized_text(rendered, budget):
                    groups.append(([el], f"{header_prefix}{piece}".strip()))
                continue

            added_len = len(rendered) + (1 if current_parts else 0)
            if current_parts and current_len + added_len > self.max_chars:
                prev_els = list(current_els)
                flush()
                if self.overlap_elements > 0 and prev_els:
                    overlap = prev_els[-self.overlap_elements :]
                    for ov in overlap:
                        ov_text = _format_element(ov)
                        if current_len + len(ov_text) + 1 + len(rendered) <= self.max_chars:
                            current_els.append(ov)
                            current_parts.append(ov_text)
                            current_len += len(ov_text) + 1

            current_els.append(el)
            current_parts.append(rendered)
            current_len += len(rendered) + (1 if len(current_parts) > 1 else 0)

        flush()
        return groups


def segment_narrative(
    document: NarrativeDocument,
    max_chars: int = 2000,
    overlap_elements: int = 1,
) -> list[NarrativeSegment]:
    """Convenience function to segment a NarrativeDocument with default settings."""
    return NarrativeSegmenter(
        max_chars=max_chars, overlap_elements=overlap_elements
    ).segment(document)
