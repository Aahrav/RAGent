"""Unit tests for TRACE narrative-aware segmentation (Issue #54)."""

from __future__ import annotations

import pytest

from src.narrative.parsers import Action, CharacterDialogue, FountainParser, NarrativeDocument, Scene
from src.narrative.segmenter import NarrativeSegmenter, segment_narrative


def test_segmenter_preserves_scene_boundaries_and_metadata() -> None:
    script = """Title: Broken Alibi
Author: Ada

ACT I

INT. OFFICE - NIGHT

ALICE enters holding a sealed envelope.

ALICE
We need to talk about the ledger.

EXT. ROOFTOP - DAWN

THREE DAYS LATER. BOB stands alone by the ledge.

BOB (V.O.)
I never saw her again.
"""
    doc = FountainParser().parse_text(script, source_path="alibi.fountain")
    segments = segment_narrative(doc)

    # Opening ACT I line + 2 scenes = 3 segments; scenes are never merged
    assert len(segments) >= 2
    office_seg = next(s for s in segments if s.location == "OFFICE")
    rooftop_seg = next(s for s in segments if s.location == "ROOFTOP")

    assert office_seg.scene_id != rooftop_seg.scene_id
    assert office_seg.act == "ACT I"
    assert office_seg.characters_present == ["ALICE"]
    assert office_seg.time_anchor == "NIGHT"
    assert office_seg.metadata["scene_id"] == office_seg.scene_id
    assert office_seg.metadata["act"] == "ACT I"
    assert office_seg.metadata["characters_present"] == ["ALICE"]

    assert rooftop_seg.act == "ACT I"
    assert rooftop_seg.characters_present == ["BOB"]
    assert rooftop_seg.time_anchor == "THREE DAYS LATER"
    assert "BOB (V.O.): I never saw her again." in rooftop_seg.text


def test_segmenter_subdivides_long_scene_along_elements_with_overlap() -> None:
    scene = Scene(
        scene_number=1,
        heading="INT. INTERROGATION ROOM - NIGHT",
        location="INTERROGATION ROOM",
        time_of_day="NIGHT",
        elements=[
            CharacterDialogue(character="DETECTIVE", dialogue="Where were you at 9 PM? " * 4),
            CharacterDialogue(character="SUSPECT", dialogue="I was at the diner on 4th Street. " * 4),
            CharacterDialogue(character="DETECTIVE", dialogue="The diner closed at 8 PM. " * 4),
            CharacterDialogue(character="WITNESS", dialogue="I saw him leave the alley! " * 4),
        ],
    )
    doc = NarrativeDocument(title="Test", format="fountain", scenes=[scene])

    segments = NarrativeSegmenter(max_chars=260, overlap_elements=1).segment(doc)

    assert len(segments) >= 2
    assert all(seg.scene_id == "scene_0001" for seg in segments)
    assert [seg.sub_index for seg in segments] == list(range(len(segments)))
    assert [seg.segment_index for seg in segments] == list(range(len(segments)))
    # First segment shouldn't claim WITNESS is present if WITNESS only appears later
    assert "WITNESS" not in segments[0].characters_present
    assert "WITNESS" in segments[-1].characters_present


def test_segment_to_rag_chunk_conversion() -> None:
    scene = Scene(
        scene_number=2,
        heading="Chapter 3: The Vault",
        chapter="Chapter 3: The Vault",
        page_start=42,
        elements=[Action(text="The vault door swung open.")],
    )
    doc = NarrativeDocument(title="Vault", format="epub", source_path="vault.epub", scenes=[scene])
    seg = segment_narrative(doc)[0]

    chunk = seg.to_rag_chunk(doc_id="doc-123")
    assert chunk.doc_id == "doc-123"
    assert chunk.source == "vault.epub"
    assert chunk.page == 42
    assert chunk.metadata["scene_id"] == "scene_0002"
    assert chunk.metadata["chapter"] == "Chapter 3: The Vault"


def test_segmenter_validates_init_params() -> None:
    with pytest.raises(ValueError, match="max_chars"):
        NarrativeSegmenter(max_chars=0)
    with pytest.raises(ValueError, match="overlap_elements"):
        NarrativeSegmenter(overlap_elements=-1)
