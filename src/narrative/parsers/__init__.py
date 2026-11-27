"""Narrative format parser registry and convenience entrypoint."""

from __future__ import annotations

from pathlib import Path

from src.narrative.parsers.base import (
    Action,
    BaseNarrativeParser,
    CharacterDialogue,
    NarrativeDocument,
    Scene,
)
from src.narrative.parsers.epub_pdf import EpubParser, PdfNarrativeParser
from src.narrative.parsers.fdx import FdxParser
from src.narrative.parsers.fountain import FountainParser
from src.narrative.parsers.srt import SrtParser

_PARSERS: tuple[BaseNarrativeParser, ...] = (
    FountainParser(),
    SrtParser(),
    FdxParser(),
    EpubParser(),
    PdfNarrativeParser(),
)

_EXT_TO_PARSER: dict[str, BaseNarrativeParser] = {
    ext: parser
    for parser in _PARSERS
    for ext in parser.supported_extensions
}


def get_parser(path_or_ext: str | Path) -> BaseNarrativeParser:
    """Return the appropriate parser for a file path or extension (e.g. '.fountain')."""
    ext = (
        path_or_ext.lower()
        if isinstance(path_or_ext, str) and path_or_ext.startswith(".")
        else Path(path_or_ext).suffix.lower()
    )
    parser = _EXT_TO_PARSER.get(ext)
    if parser is None:
        supported = ", ".join(sorted(_EXT_TO_PARSER.keys()))
        raise ValueError(
            f"Unsupported narrative format '{ext}'. Supported extensions: {supported}"
        )
    return parser


def parse_narrative(path: str | Path) -> NarrativeDocument:
    """Auto-detect format by file extension and parse a narrative file."""
    return get_parser(path).parse_file(path)


__all__ = [
    "Action",
    "BaseNarrativeParser",
    "CharacterDialogue",
    "EpubParser",
    "FdxParser",
    "FountainParser",
    "NarrativeDocument",
    "PdfNarrativeParser",
    "Scene",
    "SrtParser",
    "get_parser",
    "parse_narrative",
]
