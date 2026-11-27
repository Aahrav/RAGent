"""EPUB (.epub) and PDF (.pdf) narrative parsers for TRACE."""

from __future__ import annotations

import posixpath
import re
import xml.etree.ElementTree as ET
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from typing import ClassVar

from src.narrative.parsers.base import (
    Action,
    BaseNarrativeParser,
    NarrativeDocument,
    Scene,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

_CHAPTER_HEADING_RE = re.compile(
    r"^(?:CHAPTER\s+[A-Z0-9]+(?:\s*[:.\-]\s*.*)?|PROLOGUE|EPILOGUE)\s*$",
    re.IGNORECASE,
)


class _XhtmlTextExtractor(HTMLParser):
    """Minimal stdlib HTML/XHTML text extractor preserving paragraph breaks."""

    _BLOCK_TAGS = frozenset({"p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "br"})
    _SKIP_TAGS = frozenset({"script", "style"})

    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self._chunks: list[str] = []
        self.title: str | None = None
        self._in_heading = False
        self._heading_buf: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag_l = tag.lower()
        if tag_l in self._SKIP_TAGS:
            self._skip_depth += 1
        elif tag_l in ("h1", "h2") and self.title is None:
            self._in_heading = True
            self._heading_buf = []
        if tag_l in self._BLOCK_TAGS:
            self._chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag_l = tag.lower()
        if tag_l in self._SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
        elif tag_l in ("h1", "h2") and self._in_heading:
            self._in_heading = False
            heading = "".join(self._heading_buf).strip()
            if heading and self.title is None:
                self.title = heading
        if tag_l in self._BLOCK_TAGS:
            self._chunks.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            if self._in_heading:
                self._heading_buf.append(data)
            else:
                self._chunks.append(data)

    def get_paragraphs(self) -> list[str]:
        raw = "".join(self._chunks)
        return [
            re.sub(r"\s+", " ", para).strip()
            for para in re.split(r"\n\s*\n", raw)
            if para.strip()
        ]


class EpubParser(BaseNarrativeParser):
    """Parser for EPUB (.epub) e-book archives using Python stdlib."""

    format_name: ClassVar[str] = "epub"
    supported_extensions: ClassVar[frozenset[str]] = frozenset({".epub"})

    def parse_file(self, path: str | Path) -> NarrativeDocument:
        """Read an EPUB archive from disk and extract chapters as Scene objects."""
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Narrative file not found: {file_path}")

        scenes: list[Scene] = []
        title = file_path.stem
        author: str | None = None

        with zipfile.ZipFile(file_path, "r") as zf:
            opf_path = self._find_opf_path(zf)
            opf_root = ET.fromstring(zf.read(opf_path))

            # Extract DC metadata if present
            title_el = opf_root.find(".//{http://purl.org/dc/elements/1.1/}title")
            if title_el is not None and title_el.text:
                title = title_el.text.strip()
            creator_el = opf_root.find(".//{http://purl.org/dc/elements/1.1/}creator")
            if creator_el is not None and creator_el.text:
                author = creator_el.text.strip()

            opf_dir = posixpath.dirname(opf_path)
            manifest: dict[str, str] = {}
            for item in opf_root.findall(".//{*}manifest/{*}item"):
                item_id = item.attrib.get("id")
                href = item.attrib.get("href")
                if item_id and href:
                    manifest[item_id] = posixpath.normpath(
                        posixpath.join(opf_dir, href) if opf_dir else href
                    )

            for itemref in opf_root.findall(".//{*}spine/{*}itemref"):
                idref = itemref.attrib.get("idref")
                if not idref or idref not in manifest:
                    continue
                chapter_path = manifest[idref]
                if chapter_path not in zf.namelist():
                    continue

                html_text = zf.read(chapter_path).decode("utf-8", errors="replace")
                extractor = _XhtmlTextExtractor()
                extractor.feed(html_text)
                paragraphs = extractor.get_paragraphs()
                if not paragraphs:
                    continue

                chapter_title = extractor.title or f"Chapter {len(scenes) + 1}"
                scenes.append(
                    Scene(
                        scene_number=len(scenes) + 1,
                        heading=chapter_title,
                        chapter=chapter_title,
                        elements=[Action(text=p) for p in paragraphs],
                        raw_text="\n\n".join(paragraphs),
                    )
                )

        logger.debug(
            "Parsed EPUB document",
            extra={"source_path": str(file_path), "chapter_count": len(scenes)},
        )

        return NarrativeDocument(
            title=title,
            author=author,
            format=self.format_name,
            source_path=str(file_path),
            scenes=scenes,
        )

    def parse_text(self, content: str, source_path: str | None = None) -> NarrativeDocument:
        """EPUB is a binary ZIP format; parse plain XHTML/text fallback if provided."""
        extractor = _XhtmlTextExtractor()
        extractor.feed(content)
        paragraphs = extractor.get_paragraphs()
        scenes = (
            [
                Scene(
                    scene_number=1,
                    heading=extractor.title or "Chapter 1",
                    chapter=extractor.title or "Chapter 1",
                    elements=[Action(text=p) for p in paragraphs],
                    raw_text="\n\n".join(paragraphs),
                )
            ]
            if paragraphs
            else []
        )
        return NarrativeDocument(
            title=extractor.title or "Untitled",
            format=self.format_name,
            source_path=source_path,
            scenes=scenes,
        )

    @staticmethod
    def _find_opf_path(zf: zipfile.ZipFile) -> str:
        """Locate the OPF package file inside META-INF/container.xml."""
        container_xml = zf.read("META-INF/container.xml")
        root = ET.fromstring(container_xml)
        rootfile = root.find(".//{*}rootfile")
        if rootfile is None or "full-path" not in rootfile.attrib:
            raise ValueError("Invalid EPUB: missing rootfile in META-INF/container.xml")
        return rootfile.attrib["full-path"]


class PdfNarrativeParser(BaseNarrativeParser):
    """Parser for PDF (.pdf) manuscripts and novels using pypdf."""

    format_name: ClassVar[str] = "pdf"
    supported_extensions: ClassVar[frozenset[str]] = frozenset({".pdf"})

    def parse_file(self, path: str | Path) -> NarrativeDocument:
        """Read a PDF from disk and group extracted pages into chapter/scene units."""
        from pypdf import PdfReader

        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Narrative file not found: {file_path}")

        reader = PdfReader(str(file_path))
        pages: list[tuple[int, str]] = []
        for idx, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append((idx, text))

        scenes = self._pages_to_scenes(pages)
        title = file_path.stem
        if reader.metadata and reader.metadata.title:
            title = str(reader.metadata.title).strip() or title
        author = str(reader.metadata.author).strip() if reader.metadata and reader.metadata.author else None

        logger.debug(
            "Parsed PDF narrative document",
            extra={"source_path": str(file_path), "scene_count": len(scenes)},
        )

        return NarrativeDocument(
            title=title,
            author=author,
            format=self.format_name,
            source_path=str(file_path),
            scenes=scenes,
        )

    def parse_text(self, content: str, source_path: str | None = None) -> NarrativeDocument:
        """Parse already-extracted text into chapter/scene units."""
        scenes = self._pages_to_scenes([(1, content)] if content.strip() else [])
        title = Path(source_path).stem if source_path else "Untitled"
        return NarrativeDocument(
            title=title,
            format=self.format_name,
            source_path=source_path,
            scenes=scenes,
        )

    @staticmethod
    def _pages_to_scenes(pages: list[tuple[int, str]]) -> list[Scene]:
        """Split pages on detected chapter headings, or fall back to one Scene per page."""
        scenes: list[Scene] = []
        current_heading: str | None = None
        current_paras: list[str] = []
        page_start: int | None = None
        page_end: int | None = None

        def flush() -> None:
            nonlocal current_heading, current_paras, page_start, page_end
            if current_paras:
                heading = current_heading or f"Page {page_start}"
                scenes.append(
                    Scene(
                        scene_number=len(scenes) + 1,
                        heading=heading,
                        chapter=current_heading,
                        elements=[Action(text=p) for p in current_paras],
                        raw_text="\n\n".join(current_paras),
                        page_start=page_start,
                        page_end=page_end,
                    )
                )
            current_heading = None
            current_paras = []
            page_start = None
            page_end = None

        for page_num, text in pages:
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            for line in lines:
                if _CHAPTER_HEADING_RE.match(line):
                    flush()
                    current_heading = line
                    page_start = page_num
                    page_end = page_num
                else:
                    if page_start is None:
                        page_start = page_num
                    page_end = page_num
                    current_paras.append(line)

        flush()
        return scenes
