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
    CharacterDialogue,
    NarrativeDocument,
    Scene,
)
from src.narrative.parsers.fountain import _CHARACTER_EXT_RE, _parse_slugline
from src.utils.logger import get_logger

logger = get_logger(__name__)

_CHAPTER_HEADING_RE = re.compile(
    r"^(?:CHAPTER\s+[A-Z0-9]+(?:\s*[:.\-]\s*.*)?|PROLOGUE|EPILOGUE)\s*$",
    re.IGNORECASE,
)
_SCREENPLAY_HEADING_RE = re.compile(
    r"^(INT\.?/EXT\.?|EXT\.?/INT\.?|I/E\.?|INT\.?|EXT\.?|EST\.?)"
    r"(?:\s+(DAY|NIGHT|NITE|DAWN|DUSK|MORNING|EVENING|LATER))?\s*[:.\-\s]+(.+)$",
    re.IGNORECASE,
)
_TRANSITION_RE = re.compile(r"^(?:[A-Z0-9\s]+TO:|FADE\s+(?:IN|OUT)\.?)$")
_PAGE_NUM_RE = re.compile(r"^\d+\.?$")


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

    @classmethod
    def _pages_to_scenes(cls, pages: list[tuple[int, str]]) -> list[Scene]:
        """Auto-detect screenplay vs. prose PDFs and split into Scene objects."""
        if cls._looks_like_screenplay(pages):
            return cls._pages_to_screenplay_scenes(pages)
        return cls._pages_to_chapter_scenes(pages)

    @staticmethod
    def _looks_like_screenplay(pages: list[tuple[int, str]]) -> bool:
        """Return True if the extracted PDF text contains screenplay sluglines."""
        heading_hits = 0
        for _, text in pages[:10]:
            for line in text.splitlines():
                if _SCREENPLAY_HEADING_RE.match(line.strip()):
                    heading_hits += 1
                    if heading_hits >= 2:
                        return True
        return False

    @staticmethod
    def _parse_screenplay_slugline(heading: str) -> tuple[str | None, str | None, str | None]:
        """Parse both standard ('INT. X - DAY') and colon-style ('INT DAY: X') sluglines."""
        m = _SCREENPLAY_HEADING_RE.match(heading.strip())
        if m and m.group(2) and ":" in heading:
            int_ext = m.group(1).upper().rstrip(".") + "."
            tod = m.group(2).upper()
            loc = m.group(3).strip()
            return int_ext, loc or None, tod
        return _parse_slugline(heading)

    @staticmethod
    def _is_character_cue(line: str) -> bool:
        """Heuristic for uppercase screenplay character cues in PDF text."""
        if (
            line.startswith("(")
            or _SCREENPLAY_HEADING_RE.match(line)
            or _TRANSITION_RE.match(line)
            or _PAGE_NUM_RE.match(line)
        ):
            return False
        base = _CHARACTER_EXT_RE.sub(r"\1", line).strip()
        if not base or len(base) > 35 or len(base.split()) > 4:
            return False
        if not any(ch.isalpha() for ch in base):
            return False
        if base.endswith((".", "!", "?", ",")):
            return False
        return base == base.upper()

    @classmethod
    def _pages_to_screenplay_scenes(cls, pages: list[tuple[int, str]]) -> list[Scene]:
        """Parse screenplay-formatted PDF pages into Scene objects with dialogue."""
        scenes: list[Scene] = []
        current_scene: Scene | None = None
        raw_lines: list[str] = []

        def ensure_scene(page_num: int) -> Scene:
            nonlocal current_scene
            if current_scene is None:
                current_scene = Scene(
                    scene_number=len(scenes) + 1,
                    heading="TITLE / OPENING",
                    page_start=page_num,
                    page_end=page_num,
                )
            return current_scene

        def flush_scene() -> None:
            nonlocal current_scene, raw_lines
            if current_scene is not None:
                current_scene.raw_text = "\n".join(raw_lines).strip()
                scenes.append(current_scene)
                current_scene = None
                raw_lines = []

        # Flatten lines while tracking page numbers
        stream: list[tuple[int, str]] = []
        for page_num, text in pages:
            for line in text.splitlines():
                s = line.strip()
                if s and not _PAGE_NUM_RE.match(s):
                    stream.append((page_num, s))

        i = 0
        n = len(stream)
        while i < n:
            page_num, line = stream[i]

            if _SCREENPLAY_HEADING_RE.match(line):
                flush_scene()
                int_ext, location, time_of_day = cls._parse_screenplay_slugline(line)
                current_scene = Scene(
                    scene_number=len(scenes) + 1,
                    heading=line,
                    int_ext=int_ext,
                    location=location,
                    time_of_day=time_of_day,
                    page_start=page_num,
                    page_end=page_num,
                )
                raw_lines = [line]
                i += 1
                continue

            in_opening = current_scene is None
            scene = ensure_scene(page_num)
            scene.page_end = page_num

            if not in_opening and scene.heading != "TITLE / OPENING" and i + 1 < n and cls._is_character_cue(line):
                raw_lines.append(line)
                m = _CHARACTER_EXT_RE.match(line)
                char_name = m.group(1).strip() if m else line
                extension = m.group(2).strip() if m else None
                parentheticals: list[str] = []
                dialogue_parts: list[str] = []
                i += 1
                while i < n:
                    p_num, d_line = stream[i]
                    if _SCREENPLAY_HEADING_RE.match(d_line) or _TRANSITION_RE.match(d_line) or cls._is_character_cue(d_line):
                        break
                    scene.page_end = p_num
                    raw_lines.append(d_line)
                    if d_line.startswith("(") and d_line.endswith(")"):
                        parentheticals.append(d_line)
                    else:
                        dialogue_parts.append(d_line)
                        # Stop after a sentence-ending line if the next line starts with a new speaker or heading
                    i += 1
                if dialogue_parts:
                    scene.elements.append(
                        CharacterDialogue(
                            character=char_name,
                            dialogue=" ".join(dialogue_parts),
                            parenthetical=" ".join(parentheticals) if parentheticals else None,
                            extension=extension,
                        )
                    )
                continue

            raw_lines.append(line)
            scene.elements.append(Action(text=line))
            i += 1

        flush_scene()
        return scenes

    @staticmethod
    def _pages_to_chapter_scenes(pages: list[tuple[int, str]]) -> list[Scene]:
        """Split prose pages on detected chapter headings, or fall back to one Scene per page."""
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
