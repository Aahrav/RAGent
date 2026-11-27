"""Final Draft (.fdx) XML screenplay parser for TRACE."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import ClassVar

from src.narrative.parsers.base import (
    Action,
    BaseNarrativeParser,
    CharacterDialogue,
    NarrativeDocument,
    Scene,
)
from src.narrative.parsers.fountain import _parse_slugline
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _paragraph_text(paragraph: ET.Element) -> str:
    """Concatenate all <Text> child elements inside an FDX <Paragraph>."""
    parts = [node.text or "" for node in paragraph.findall(".//Text")]
    if not parts and paragraph.text:
        return paragraph.text.strip()
    return "".join(parts).strip()


class FdxParser(BaseNarrativeParser):
    """Parser for Final Draft (.fdx) XML screenplay files."""

    format_name: ClassVar[str] = "fdx"
    supported_extensions: ClassVar[frozenset[str]] = frozenset({".fdx"})

    def parse_text(self, content: str, source_path: str | None = None) -> NarrativeDocument:
        """Parse Final Draft XML content into a NarrativeDocument."""
        try:
            root = ET.fromstring(content)
        except ET.ParseError as exc:
            raise ValueError(f"Invalid FDX XML content: {exc}") from exc

        scenes: list[Scene] = []
        current_scene: Scene | None = None
        current_character: str | None = None
        current_extension: str | None = None
        pending_parenthetical: str | None = None

        def ensure_scene() -> Scene:
            nonlocal current_scene
            if current_scene is None:
                current_scene = Scene(scene_number=len(scenes) + 1, heading="OPENING")
            return current_scene

        def flush_scene() -> None:
            nonlocal current_scene
            if current_scene is not None:
                scenes.append(current_scene)
                current_scene = None

        for para in root.findall(".//Content/Paragraph"):
            ptype = para.attrib.get("Type", "").strip()
            text = _paragraph_text(para)
            if not text:
                continue

            if ptype == "Scene Heading":
                flush_scene()
                current_character = None
                pending_parenthetical = None
                int_ext, location, time_of_day = _parse_slugline(text)
                current_scene = Scene(
                    scene_number=len(scenes) + 1,
                    scene_label=para.attrib.get("Number"),
                    heading=text,
                    int_ext=int_ext,
                    location=location,
                    time_of_day=time_of_day,
                )
            elif ptype == "Character":
                pending_parenthetical = None
                if "(" in text and text.endswith(")"):
                    name, ext = text.rsplit("(", 1)
                    current_character = name.strip()
                    current_extension = ext.rstrip(")").strip()
                else:
                    current_character = text.strip()
                    current_extension = None
            elif ptype == "Parenthetical":
                pending_parenthetical = (
                    f"{pending_parenthetical} {text}".strip()
                    if pending_parenthetical
                    else text
                )
            elif ptype == "Dialogue" and current_character:
                scene = ensure_scene()
                scene.elements.append(
                    CharacterDialogue(
                        character=current_character,
                        dialogue=text,
                        parenthetical=pending_parenthetical,
                        extension=current_extension,
                    )
                )
                pending_parenthetical = None
            else:
                # Action, Transition, Shot, General, etc.
                current_character = None
                pending_parenthetical = None
                scene = ensure_scene()
                scene.elements.append(Action(text=text))

        flush_scene()

        title = Path(source_path).stem if source_path else "Untitled"
        logger.debug(
            "Parsed FDX document",
            extra={"source_path": source_path, "scene_count": len(scenes)},
        )

        return NarrativeDocument(
            title=title,
            format=self.format_name,
            source_path=source_path,
            scenes=scenes,
        )
