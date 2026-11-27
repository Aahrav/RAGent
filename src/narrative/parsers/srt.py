"""SubRip (.srt) subtitle parser for TRACE."""

from __future__ import annotations

import re
from pathlib import Path
from typing import ClassVar

from src.narrative.parsers.base import (
    Action,
    BaseNarrativeParser,
    CharacterDialogue,
    NarrativeDocument,
    Scene,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

_TIMECODE_RE = re.compile(
    r"^(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})"
)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_SPEAKER_PREFIX_RE = re.compile(r"^-?\s*([A-Z][A-Z0-9 .'\-]{0,30}):\s+(.+)$")


class SrtParser(BaseNarrativeParser):
    """Parser for SubRip (.srt) subtitle files."""

    format_name: ClassVar[str] = "srt"
    supported_extensions: ClassVar[frozenset[str]] = frozenset({".srt"})

    def parse_text(self, content: str, source_path: str | None = None) -> NarrativeDocument:
        """Parse SRT subtitle blocks into a NarrativeDocument."""
        normalized = content.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
        blocks = [b.strip() for b in re.split(r"\n\s*\n", normalized) if b.strip()]

        elements: list[CharacterDialogue | Action] = []
        first_ts: str | None = None
        last_ts: str | None = None

        for block in blocks:
            lines = [ln.strip() for ln in block.split("\n") if ln.strip()]
            if len(lines) < 2:
                continue

            ts_idx = 1 if lines[0].isdigit() else 0
            if ts_idx >= len(lines):
                continue

            ts_match = _TIMECODE_RE.match(lines[ts_idx])
            if not ts_match:
                continue

            start_ts, end_ts = ts_match.group(1), ts_match.group(2)
            if first_ts is None:
                first_ts = start_ts
            last_ts = end_ts

            text_lines = lines[ts_idx + 1 :]
            if not text_lines:
                continue

            clean_text = _HTML_TAG_RE.sub("", " ".join(text_lines)).strip()
            if not clean_text:
                continue

            speaker_match = _SPEAKER_PREFIX_RE.match(clean_text)
            if speaker_match:
                elements.append(
                    CharacterDialogue(
                        character=speaker_match.group(1).strip(),
                        dialogue=speaker_match.group(2).strip(),
                        timestamp_start=start_ts,
                        timestamp_end=end_ts,
                    )
                )
            else:
                elements.append(
                    Action(
                        text=clean_text,
                        timestamp_start=start_ts,
                        timestamp_end=end_ts,
                    )
                )

        scenes: list[Scene] = []
        if elements:
            scenes.append(
                Scene(
                    scene_number=1,
                    heading="SUBTITLES",
                    elements=elements,
                    raw_text=normalized.strip(),
                    timestamp_start=first_ts,
                    timestamp_end=last_ts,
                )
            )

        title = Path(source_path).stem if source_path else "Untitled"
        logger.debug(
            "Parsed SRT document",
            extra={"source_path": source_path, "element_count": len(elements)},
        )

        return NarrativeDocument(
            title=title,
            format=self.format_name,
            source_path=source_path,
            scenes=scenes,
        )
