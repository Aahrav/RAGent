"""Base Pydantic v2 models and abstract parser for TRACE narrative ingestion."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field


class CharacterDialogue(BaseModel):
    """A single spoken or captioned line attributed to a character."""

    kind: Literal["dialogue"] = "dialogue"
    character: str = Field(..., min_length=1, description="Normalized speaker name.")
    dialogue: str = Field(..., description="Spoken or captioned text.")
    parenthetical: str | None = Field(
        default=None, description="Stage direction or delivery note, e.g. '(whispering)'."
    )
    extension: str | None = Field(
        default=None, description="Screenplay extension such as 'V.O.', 'O.S.', or 'CONT'D'."
    )
    is_dual: bool = Field(
        default=False, description="True if spoken simultaneously (Fountain '^' dual dialogue)."
    )
    timestamp_start: str | None = Field(
        default=None, description="Optional start timecode (e.g. SRT '00:01:12,300')."
    )
    timestamp_end: str | None = Field(
        default=None, description="Optional end timecode (e.g. SRT '00:01:15,100')."
    )


class Action(BaseModel):
    """A narrative action, description, or prose paragraph within a scene."""

    kind: Literal["action"] = "action"
    text: str = Field(..., min_length=1, description="Action or descriptive prose.")
    timestamp_start: str | None = Field(default=None)
    timestamp_end: str | None = Field(default=None)


NarrativeElement = CharacterDialogue | Action


class Scene(BaseModel):
    """A contiguous narrative unit (screenplay scene, chapter, or subtitle segment)."""

    scene_number: int = Field(..., ge=1, description="1-based sequential index in the document.")
    scene_label: str | None = Field(
        default=None, description="Explicit scene number/label from source (e.g. '12A')."
    )
    heading: str = Field(default="", description="Scene slugline or chapter title.")
    int_ext: str | None = Field(
        default=None, description="Interior/Exterior marker (e.g. 'INT.', 'EXT.', 'INT./EXT.')."
    )
    location: str | None = Field(default=None, description="Parsed scene location.")
    time_of_day: str | None = Field(
        default=None, description="Parsed time of day (e.g. 'DAY', 'NIGHT', 'LATER')."
    )
    chapter: str | None = Field(default=None, description="Chapter title or identifier, if any.")
    elements: list[NarrativeElement] = Field(
        default_factory=list, description="Ordered sequence of dialogue and action elements."
    )
    raw_text: str = Field(default="", description="Original unparsed text for this scene.")
    page_start: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)
    timestamp_start: str | None = Field(default=None)
    timestamp_end: str | None = Field(default=None)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def dialogues(self) -> list[CharacterDialogue]:
        """Return all dialogue elements in this scene in order."""
        return [el for el in self.elements if isinstance(el, CharacterDialogue)]

    @property
    def actions(self) -> list[Action]:
        """Return all action elements in this scene in order."""
        return [el for el in self.elements if isinstance(el, Action)]

    @property
    def characters_present(self) -> list[str]:
        """Return unique character names appearing in dialogue, preserving first-seen order."""
        seen: dict[str, None] = {}
        for d in self.dialogues:
            seen.setdefault(d.character, None)
        return list(seen.keys())


class NarrativeDocument(BaseModel):
    """Structured representation of a parsed narrative work."""

    title: str = Field(default="Untitled", description="Document or screenplay title.")
    author: str | None = Field(default=None, description="Author or screenwriter, if known.")
    format: str = Field(..., description="Source format identifier (e.g. 'fountain', 'srt').")
    source_path: str | None = Field(default=None, description="Filesystem path of the source.")
    scenes: list[Scene] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def total_scenes(self) -> int:
        """Total number of parsed scenes or chapters."""
        return len(self.scenes)

    @property
    def all_characters(self) -> list[str]:
        """Unique character names across all scenes, preserving first-seen order."""
        seen: dict[str, None] = {}
        for scene in self.scenes:
            for char in scene.characters_present:
                seen.setdefault(char, None)
        return list(seen.keys())


class BaseNarrativeParser(ABC):
    """Abstract base class for format-specific narrative parsers."""

    format_name: ClassVar[str]
    supported_extensions: ClassVar[frozenset[str]]

    def parse_file(self, path: str | Path) -> NarrativeDocument:
        """Read a narrative file from disk and parse it into a NarrativeDocument."""
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Narrative file not found: {file_path}")
        content = file_path.read_text(encoding="utf-8", errors="replace")
        return self.parse_text(content, source_path=str(file_path))

    @abstractmethod
    def parse_text(self, content: str, source_path: str | None = None) -> NarrativeDocument:
        """Parse raw text content into a NarrativeDocument."""
