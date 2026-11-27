"""Fountain (.fountain) screenplay parser for TRACE."""

from __future__ import annotations

import re
from typing import Any, ClassVar

from src.narrative.parsers.base import (
    Action,
    BaseNarrativeParser,
    CharacterDialogue,
    NarrativeDocument,
    Scene,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

_TITLE_KEY_RE = re.compile(r"^([A-Za-z][A-Za-z0-9 _-]{0,30}):\s*(.*)$")
_SCENE_HEADING_RE = re.compile(
    r"^(?:(\.)(?!\.)|(?=(?:INT\.?/EXT\.?|EXT\.?/INT\.?|I/E\.?|INT\.?|EXT\.?|EST\.?)(?:\s|$)))"
    r"(.+?)(?:\s*#([A-Za-z0-9.-]+)#)?\s*$",
    re.IGNORECASE,
)
_SLUG_PARTS_RE = re.compile(
    r"^(INT\.?/EXT\.?|EXT\.?/INT\.?|I/E\.?|INT\.?|EXT\.?|EST\.?)\s+(.+)$",
    re.IGNORECASE,
)
_CHARACTER_EXT_RE = re.compile(r"^(.+?)\s*\(([^)]+)\)\s*$")
_TRANSITION_RE = re.compile(r"^(?:>\s*.+|[A-Z0-9\s]+TO:)$")


def _strip_boneyard(text: str) -> str:
    """Remove Fountain /* boneyard */ comments."""
    return re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)


def _parse_slugline(heading: str) -> tuple[str | None, str | None, str | None]:
    """Split a screenplay slugline into (int_ext, location, time_of_day)."""
    clean = heading.lstrip(".").strip()
    int_ext: str | None = None
    rest = clean

    m = _SLUG_PARTS_RE.match(clean)
    if m:
        int_ext = m.group(1).upper()
        rest = m.group(2).strip()

    if " - " in rest:
        loc, tod = rest.rsplit(" - ", 1)
        return int_ext, loc.strip() or None, tod.strip() or None
    return int_ext, rest or None, None


class FountainParser(BaseNarrativeParser):
    """Parser for standard Fountain (.fountain) screenplay files."""

    format_name: ClassVar[str] = "fountain"
    supported_extensions: ClassVar[frozenset[str]] = frozenset({".fountain"})

    def parse_text(self, content: str, source_path: str | None = None) -> NarrativeDocument:
        """Parse Fountain screenplay text into structured scenes, dialogue, and action."""
        normalized = _strip_boneyard(content.replace("\r\n", "\n").replace("\r", "\n"))
        lines = normalized.split("\n")

        title_meta, body_start = self._extract_title_page(lines)
        title = title_meta.get("title") or "Untitled"
        author = title_meta.get("author") or title_meta.get("authors")

        scenes: list[Scene] = []
        current_scene: Scene | None = None
        current_raw_lines: list[str] = []

        def ensure_scene() -> Scene:
            nonlocal current_scene
            if current_scene is None:
                current_scene = Scene(
                    scene_number=len(scenes) + 1,
                    heading="OPENING",
                )
            return current_scene

        def flush_scene() -> None:
            nonlocal current_scene, current_raw_lines
            if current_scene is not None:
                current_scene.raw_text = "\n".join(current_raw_lines).strip()
                scenes.append(current_scene)
                current_scene = None
                current_raw_lines = []

        i = body_start
        n = len(lines)
        while i < n:
            line = lines[i]
            stripped = line.strip()

            if not stripped:
                if current_scene is not None:
                    current_raw_lines.append(line)
                i += 1
                continue

            # Page breaks, sections (#), synopses (=), notes ([[...]])
            if stripped == "===" or stripped.startswith("#") or stripped.startswith("="):
                if current_scene is not None:
                    current_raw_lines.append(line)
                i += 1
                continue

            # Scene heading (must be preceded by blank line or start of body)
            prev_blank = i == body_start or not lines[i - 1].strip()
            heading_match = _SCENE_HEADING_RE.match(stripped) if prev_blank else None
            if heading_match:
                flush_scene()
                heading_text = heading_match.group(2).strip()
                scene_label = heading_match.group(3)
                int_ext, location, time_of_day = _parse_slugline(heading_text)
                current_scene = Scene(
                    scene_number=len(scenes) + 1,
                    scene_label=scene_label,
                    heading=heading_text,
                    int_ext=int_ext,
                    location=location,
                    time_of_day=time_of_day,
                )
                current_raw_lines = [line]
                i += 1
                continue

            # Transitions (e.g. CUT TO: or > FADE OUT.)
            if prev_blank and _TRANSITION_RE.match(stripped) and not stripped.endswith("^"):
                scene = ensure_scene()
                current_raw_lines.append(line)
                scene.elements.append(Action(text=stripped.lstrip("> ").strip()))
                i += 1
                continue

            # Character cue -> Dialogue block
            next_line = lines[i + 1].strip() if i + 1 < n else ""
            if prev_blank and next_line and self._is_character_cue(stripped):
                scene = ensure_scene()
                current_raw_lines.append(line)
                char_name, extension, is_dual = self._parse_character_cue(stripped)

                parentheticals: list[str] = []
                dialogue_parts: list[str] = []
                i += 1
                while i < n:
                    d_line = lines[i]
                    d_stripped = d_line.strip()
                    if not d_stripped:
                        break
                    current_raw_lines.append(d_line)
                    if d_stripped.startswith("(") and d_stripped.endswith(")"):
                        parentheticals.append(d_stripped)
                    else:
                        dialogue_parts.append(d_stripped)
                    i += 1

                if dialogue_parts:
                    scene.elements.append(
                        CharacterDialogue(
                            character=char_name,
                            dialogue=" ".join(dialogue_parts),
                            parenthetical=" ".join(parentheticals) if parentheticals else None,
                            extension=extension,
                            is_dual=is_dual,
                        )
                    )
                continue

            # Default: Action paragraph (collect contiguous non-blank lines)
            scene = ensure_scene()
            action_lines: list[str] = []
            while i < n and lines[i].strip():
                # Stop if next line looks like nothing special; Fountain action is contiguous
                current_raw_lines.append(lines[i])
                action_lines.append(lines[i].strip().lstrip("!"))
                i += 1
            if action_lines:
                scene.elements.append(Action(text=" ".join(action_lines)))

        flush_scene()

        logger.debug(
            "Parsed Fountain document",
            extra={
                "source_path": source_path,
                "scene_count": len(scenes),
                "character_count": len({c for s in scenes for c in s.characters_present}),
            },
        )

        return NarrativeDocument(
            title=title,
            author=author,
            format=self.format_name,
            source_path=source_path,
            scenes=scenes,
            metadata={"title_page": title_meta} if title_meta else {},
        )

    @staticmethod
    def _extract_title_page(lines: list[str]) -> tuple[dict[str, Any], int]:
        """Parse optional Fountain key-value title page at the top of the file."""
        meta: dict[str, Any] = {}
        i = 0
        n = len(lines)
        while i < n and not lines[i].strip():
            i += 1

        if i >= n or not _TITLE_KEY_RE.match(lines[i]):
            return meta, 0

        current_key: str | None = None
        while i < n:
            line = lines[i]
            if not line.strip():
                i += 1
                break
            m = _TITLE_KEY_RE.match(line)
            if m and not line.startswith((" ", "\t")):
                current_key = m.group(1).strip().lower().replace(" ", "_")
                val = m.group(2).strip()
                meta[current_key] = val
            elif current_key and line.startswith((" ", "\t")):
                existing = meta.get(current_key, "")
                addition = line.strip()
                meta[current_key] = f"{existing} {addition}".strip() if existing else addition
            else:
                break
            i += 1

        return meta, i

    @staticmethod
    def _is_character_cue(line: str) -> bool:
        """Return True if a line is a Fountain character cue."""
        if line.startswith("@"):
            return True
        if line.startswith((".", ">", "!", "~")):
            return False
        candidate = line.rstrip("^").strip()
        # Strip trailing parenthetical extension like (V.O.) before checking case
        candidate_no_ext = _CHARACTER_EXT_RE.sub(r"\1", candidate).strip()
        if not candidate_no_ext or not any(ch.isalpha() for ch in candidate_no_ext):
            return False
        return candidate_no_ext == candidate_no_ext.upper()

    @staticmethod
    def _parse_character_cue(line: str) -> tuple[str, str | None, bool]:
        """Extract (character_name, extension, is_dual) from a character cue."""
        is_dual = line.endswith("^")
        clean = line.lstrip("@").rstrip("^").strip()
        extension: str | None = None
        m = _CHARACTER_EXT_RE.match(clean)
        if m:
            clean = m.group(1).strip()
            extension = m.group(2).strip()
        return clean, extension, is_dual
