"""Unit tests for TRACE narrative format parsers (Issue #53)."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from src.narrative.parsers import (
    Action,
    CharacterDialogue,
    EpubParser,
    FdxParser,
    FountainParser,
    PdfNarrativeParser,
    SrtParser,
    get_parser,
    parse_narrative,
)


def test_fountain_parser_extracts_title_scenes_and_dialogue() -> None:
    script = """Title: The Midnight Train
Author: Ada Lovelace

/* Secret boneyard note that should be ignored */

INT. TRAIN CAR - NIGHT #1A#

Rain lashes the window. BRUCE checks his watch.

BRUCE (V.O.)
(whispering)
We're already ten minutes behind.

@McCLANE^
Hold on!

EXT. STATION PLATFORM - DAWN

The train screeches to a halt.

CUT TO:
"""
    doc = FountainParser().parse_text(script, source_path="sample.fountain")

    assert doc.title == "The Midnight Train"
    assert doc.author == "Ada Lovelace"
    assert doc.format == "fountain"
    assert doc.total_scenes == 2

    s1 = doc.scenes[0]
    assert s1.scene_number == 1
    assert s1.scene_label == "1A"
    assert s1.int_ext == "INT."
    assert s1.location == "TRAIN CAR"
    assert s1.time_of_day == "NIGHT"
    assert s1.characters_present == ["BRUCE", "McCLANE"]
    assert len(s1.actions) == 1
    assert "Rain lashes the window." in s1.actions[0].text

    d1, d2 = s1.dialogues
    assert d1.character == "BRUCE"
    assert d1.extension == "V.O."
    assert d1.parenthetical == "(whispering)"
    assert d1.dialogue == "We're already ten minutes behind."
    assert d1.is_dual is False

    assert d2.character == "McCLANE"
    assert d2.is_dual is True
    assert d2.dialogue == "Hold on!"

    s2 = doc.scenes[1]
    assert s2.int_ext == "EXT."
    assert s2.location == "STATION PLATFORM"
    assert s2.time_of_day == "DAWN"
    assert [a.text for a in s2.actions] == [
        "The train screeches to a halt.",
        "CUT TO:",
    ]
    assert doc.all_characters == ["BRUCE", "McCLANE"]


def test_fountain_forced_scene_heading_and_file_roundtrip(tmp_path: Path) -> None:
    fountain_file = tmp_path / "forced.fountain"
    fountain_file.write_text(
        ".FLASHBACK - THE LABORATORY\n\nELENA\nIt worked.\n",
        encoding="utf-8",
    )

    doc = parse_narrative(fountain_file)
    assert doc.format == "fountain"
    assert doc.total_scenes == 1
    assert doc.scenes[0].heading == "FLASHBACK - THE LABORATORY"
    assert doc.scenes[0].dialogues[0].character == "ELENA"


def test_srt_parser_extracts_speakers_actions_and_timestamps() -> None:
    srt_content = """1
00:00:01,000 --> 00:00:03,500
<i>ALICE:</i> Where did you hide the key?

2
00:00:04,000 --> 00:00:06,200
[Thunder crashes outside]

3
00:00:07,000 --> 00:00:09,000
- BOB: Behind the bookshelf.
"""
    doc = SrtParser().parse_text(srt_content, source_path="episode_01.srt")

    assert doc.title == "episode_01"
    assert doc.format == "srt"
    assert doc.total_scenes == 1

    scene = doc.scenes[0]
    assert scene.timestamp_start == "00:00:01,000"
    assert scene.timestamp_end == "00:00:09,000"
    assert len(scene.elements) == 3

    assert isinstance(scene.elements[0], CharacterDialogue)
    assert scene.elements[0].character == "ALICE"
    assert scene.elements[0].dialogue == "Where did you hide the key?"

    assert isinstance(scene.elements[1], Action)
    assert scene.elements[1].text == "[Thunder crashes outside]"

    assert isinstance(scene.elements[2], CharacterDialogue)
    assert scene.elements[2].character == "BOB"
    assert scene.elements[2].dialogue == "Behind the bookshelf."


def test_fdx_parser_parses_final_draft_xml() -> None:
    fdx_xml = """<?xml version="1.0" encoding="UTF-8" ?>
<FinalDraft DocumentType="Script" Template="No" Version="1">
  <Content>
    <Paragraph Type="Scene Heading" Number="4">
      <Text>INT. ARCHIVE - NIGHT</Text>
    </Paragraph>
    <Paragraph Type="Action">
      <Text>Dust floats in the flashlight beam.</Text>
    </Paragraph>
    <Paragraph Type="Character">
      <Text>MARCUS (O.S.)</Text>
    </Paragraph>
    <Paragraph Type="Parenthetical">
      <Text>(urgent)</Text>
    </Paragraph>
    <Paragraph Type="Dialogue">
      <Text>Don't touch the ledger.</Text>
    </Paragraph>
  </Content>
</FinalDraft>
"""
    doc = FdxParser().parse_text(fdx_xml, source_path="archive.fdx")

    assert doc.title == "archive"
    assert doc.format == "fdx"
    assert doc.total_scenes == 1

    scene = doc.scenes[0]
    assert scene.scene_label == "4"
    assert scene.int_ext == "INT."
    assert scene.location == "ARCHIVE"
    assert scene.time_of_day == "NIGHT"
    assert len(scene.actions) == 1
    assert len(scene.dialogues) == 1
    assert scene.dialogues[0].character == "MARCUS"
    assert scene.dialogues[0].extension == "O.S."
    assert scene.dialogues[0].parenthetical == "(urgent)"
    assert scene.dialogues[0].dialogue == "Don't touch the ledger."


def test_fdx_parser_rejects_invalid_xml() -> None:
    with pytest.raises(ValueError, match="Invalid FDX XML"):
        FdxParser().parse_text("<FinalDraft><Broken")


def test_epub_parser_reads_archive_chapters(tmp_path: Path) -> None:
    epub_path = tmp_path / "novel.epub"
    container_xml = """<?xml version="1.0"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>"""
    opf_xml = """<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:title>Clockwork Glass</dc:title>
    <dc:creator>Nora Vance</dc:creator>
  </metadata>
  <manifest>
    <item id="ch1" href="ch1.xhtml" media-type="application/xhtml+xml"/>
    <item id="ch2" href="ch2.xhtml" media-type="application/xhtml+xml"/>
  </manifest>
  <spine>
    <itemref idref="ch1"/>
    <itemref idref="ch2"/>
  </spine>
</package>"""
    ch1_html = "<html><body><h1>Chapter 1: The Watchmaker</h1><p>The clock stopped at noon.</p></body></html>"
    ch2_html = "<html><body><h1>Chapter 2: The Alibi</h1><p>He was miles away.</p></body></html>"

    with zipfile.ZipFile(epub_path, "w") as zf:
        zf.writestr("META-INF/container.xml", container_xml)
        zf.writestr("OEBPS/content.opf", opf_xml)
        zf.writestr("OEBPS/ch1.xhtml", ch1_html)
        zf.writestr("OEBPS/ch2.xhtml", ch2_html)

    doc = EpubParser().parse_file(epub_path)
    assert doc.title == "Clockwork Glass"
    assert doc.author == "Nora Vance"
    assert doc.format == "epub"
    assert doc.total_scenes == 2
    assert doc.scenes[0].chapter == "Chapter 1: The Watchmaker"
    assert doc.scenes[0].actions[0].text == "The clock stopped at noon."
    assert doc.scenes[1].chapter == "Chapter 2: The Alibi"


def test_pdf_parser_splits_chapters_and_registry_errors() -> None:
    text = "PROLOGUE\nA cold wind blew.\nCHAPTER 1: ARRIVAL\nThe ship docked."
    doc = PdfNarrativeParser().parse_text(text, source_path="book.pdf")

    assert doc.title == "book"
    assert doc.total_scenes == 2
    assert doc.scenes[0].chapter == "PROLOGUE"
    assert doc.scenes[1].chapter == "CHAPTER 1: ARRIVAL"

    with pytest.raises(ValueError, match="Unsupported narrative format"):
        get_parser(".docx")


def test_pdf_parser_detects_screenplay_sluglines_and_dialogue() -> None:
    text = """THE GODFATHER
MARIO PUZO
INT DAY: DON'S OFFICE (SUMMER 1945)
(CONT. )
BONASERA
(pause)
I believe in America.
EXT NITE: STREET (WINTER 1945)
MICHAEL (V.O.)
It was never personal.
"""
    doc = PdfNarrativeParser().parse_text(text, source_path="godfather.pdf")

    assert doc.total_scenes == 3
    # Title page uppercase lines stay Action, not CharacterDialogue
    assert doc.scenes[0].heading == "TITLE / OPENING"
    assert doc.scenes[0].dialogues == []

    # Standalone (CONT. ) is not treated as a character cue
    assert doc.scenes[1].int_ext == "INT."
    assert doc.scenes[1].location == "DON'S OFFICE (SUMMER 1945)"
    assert doc.scenes[1].time_of_day == "DAY"
    assert doc.scenes[1].characters_present == ["BONASERA"]
    assert doc.scenes[1].dialogues[0].character == "BONASERA"
    assert doc.scenes[1].dialogues[0].parenthetical == "(pause)"

    assert doc.scenes[2].int_ext == "EXT."
    assert doc.scenes[2].time_of_day == "NITE"
    assert doc.scenes[2].dialogues[0].character == "MICHAEL"
    assert doc.scenes[2].dialogues[0].extension == "V.O."
