"""TRACE narrative ingestion, representation, and parsing package."""

from src.narrative.extractor import (
    DocumentExtraction,
    FactTriple,
    KnowledgeAssertion,
    LayaSegmentTriage,
    NarrativeExtractor,
    SegmentExtraction,
    TemporalMarker,
    get_laya_agent,
)
from src.narrative.parsers import (
    Action,
    BaseNarrativeParser,
    CharacterDialogue,
    EpubParser,
    FdxParser,
    FountainParser,
    NarrativeDocument,
    PdfNarrativeParser,
    Scene,
    SrtParser,
    get_parser,
    parse_narrative,
)
from src.narrative.segmenter import (
    NarrativeSegment,
    NarrativeSegmenter,
    segment_narrative,
)

__all__ = [
    "Action",
    "BaseNarrativeParser",
    "CharacterDialogue",
    "DocumentExtraction",
    "EpubParser",
    "FactTriple",
    "FdxParser",
    "FountainParser",
    "KnowledgeAssertion",
    "LayaSegmentTriage",
    "NarrativeDocument",
    "NarrativeExtractor",
    "NarrativeSegment",
    "NarrativeSegmenter",
    "PdfNarrativeParser",
    "Scene",
    "SegmentExtraction",
    "SrtParser",
    "TemporalMarker",
    "get_laya_agent",
    "get_parser",
    "parse_narrative",
    "segment_narrative",
]
