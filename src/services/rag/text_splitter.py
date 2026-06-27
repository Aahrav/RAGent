"""Text chunking: splits Documents into smaller Chunks for embedding and retrieval.

Why chunking?
  LLMs have a token/context limit — you can't feed a whole document into a prompt.
  Smaller chunks also improve retrieval precision: instead of returning a whole page,
  Qdrant returns only the 3-5 paragraphs most relevant to the query.

Algorithm — Recursive Character Splitting:
  Try separators in order of preference:
    1. Double newline  ("\n\n") — paragraph boundary  ← preferred
    2. Single newline  ("\n")   — line boundary
    3. Period + space  (". ")   — sentence boundary
    4. Space           (" ")    — word boundary
    5. Empty string    ("")     — character boundary   ← last resort

  For each separator, split the text and check if the pieces are small enough.
  If a piece is still too large, recursively split it with the next separator.
  Small pieces are merged together (greedy) to stay under chunk_size without
  wasting space.

Why overlap?
  The last `chunk_overlap` characters of each chunk are repeated at the start
  of the next chunk. This ensures that a sentence split across a boundary can
  still be understood in either chunk.

Default settings (from config):
  chunk_size    = 500  characters
  chunk_overlap = 50   characters

Usage:
    from src.services.rag.text_splitter import split_documents
    from src.services.rag.models import Document

    docs = [Document(text="Long text...", source="report.pdf", page=1)]
    chunks = split_documents(docs)
    # → list of Chunk objects, each ≤ 500 characters
"""

from __future__ import annotations

from src.config import get_settings
from src.services.rag.models import Chunk, Document
from src.storage.document_store import make_doc_id
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Separator priority list ────────────────────────────────────────────────────
# Ordered from most-preferred (keeps semantic units intact) to least-preferred.

_SEPARATORS: list[str] = ["\n\n", "\n", ". ", " ", ""]


# ── Core splitting logic ───────────────────────────────────────────────────────

def _split_text(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    separators: list[str] | None = None,
) -> list[str]:
    """Recursively split text into pieces no larger than chunk_size.

    This is the core algorithm. It is called recursively — pieces that are
    still too large get passed back in with a shorter separator list.

    Args:
        text:          The text to split.
        chunk_size:    Maximum number of characters per output piece.
        chunk_overlap: How many characters to repeat at the start of each
                       new piece (overlap window).
        separators:    Separator priority list. Defaults to ``_SEPARATORS``.

    Returns:
        List of text strings, each at most ``chunk_size`` characters long.
    """
    if separators is None:
        separators = _SEPARATORS

    # Base case: text already fits in one chunk — return it as-is
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    # Try each separator in priority order
    chosen_sep: str = separators[-1]   # fallback: character-level split
    remaining_seps: list[str] = []

    for i, sep in enumerate(separators):
        if sep == "" or sep in text:
            chosen_sep = sep
            remaining_seps = separators[i + 1:]
            break

    # Split the text by the chosen separator
    raw_pieces: list[str] = text.split(chosen_sep) if chosen_sep else list(text)

    # Merge small pieces together (greedy) so we don't create hundreds of
    # tiny chunks when the text has many short lines.
    merged: list[str] = _merge_pieces(raw_pieces, chosen_sep, chunk_size)

    # Recursively split any merged piece that is still too large
    final_pieces: list[str] = []
    for piece in merged:
        if len(piece) <= chunk_size:
            if piece.strip():
                final_pieces.append(piece)
        else:
            # This piece is still too big — recurse with the next separator tier
            sub_pieces = _split_text(piece, chunk_size, chunk_overlap, remaining_seps)
            final_pieces.extend(sub_pieces)

    # Apply overlap window
    return _apply_overlap(final_pieces, chunk_size, chunk_overlap)


def _merge_pieces(pieces: list[str], separator: str, chunk_size: int) -> list[str]:
    """Greedily merge short pieces into larger ones up to chunk_size.

    Without this step, splitting on "\n\n" for a document with many short
    paragraphs would produce hundreds of tiny chunks — wasteful and imprecise.

    Example (chunk_size=100):
      Input pieces: ["Hello", "world", "this is a longer paragraph that uses most of the budget"]
      After merge:  ["Hello world", "this is a longer paragraph that uses most of the budget"]

    Args:
        pieces:     Raw split pieces (not yet size-checked).
        separator:  The separator that was used to create the pieces.
                    Used to rejoin them correctly.
        chunk_size: Maximum character count for each merged group.

    Returns:
        List of merged pieces. Each merged piece fits within chunk_size.
        Pieces that are already too large are returned unchanged (will be
        recursively split later).
    """
    merged: list[str] = []
    current_parts: list[str] = []
    current_len: int = 0
    sep_len = len(separator)

    for piece in pieces:
        piece_len = len(piece)

        # Would adding this piece (plus the re-join separator) exceed the limit?
        addition = piece_len + (sep_len if current_parts else 0)

        if current_len + addition > chunk_size and current_parts:
            # Flush the current group
            merged.append(separator.join(current_parts))
            current_parts = []
            current_len = 0

        current_parts.append(piece)
        current_len += addition

    # Flush whatever is left
    if current_parts:
        merged.append(separator.join(current_parts))

    return merged


def _apply_overlap(pieces: list[str], chunk_size: int, chunk_overlap: int) -> list[str]:
    """Prepend the tail of the previous chunk to each chunk (overlap window).

    This prevents hard breaks in context between adjacent chunks.

    Example (chunk_overlap=20):
      Chunk 1: "...the revenue grew in Q3"
      Chunk 2 (without overlap): "due to product sales"
      Chunk 2 (with overlap):    "grew in Q3 due to product sales"
                                  ^^^^^^^^^^^ ← overlap from chunk 1

    Args:
        pieces:        Flat list of text strings, each ≤ chunk_size.
        chunk_size:    Max characters per chunk (used to guard the merge).
        chunk_overlap: Number of characters to carry over from the previous chunk.

    Returns:
        List of overlapped strings. Length is the same as ``pieces``.
    """
    if chunk_overlap <= 0 or len(pieces) <= 1:
        return pieces

    overlapped: list[str] = [pieces[0]]

    for i in range(1, len(pieces)):
        prev = pieces[i - 1]
        curr = pieces[i]

        # Take the last `chunk_overlap` characters of the previous piece
        tail = prev[-chunk_overlap:].strip()

        if tail and not curr.startswith(tail):
            candidate = tail + " " + curr
            # Guard: don't exceed chunk_size due to the prepended tail
            if len(candidate) <= chunk_size + chunk_overlap:
                curr = candidate

        overlapped.append(curr)

    return overlapped


# ── Public API ─────────────────────────────────────────────────────────────────

def split_document(
    doc: Document,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Chunk]:
    """Split a single Document into a list of Chunks.

    Args:
        doc:          The source Document to split.
        chunk_size:   Max characters per chunk. Defaults to ``settings.chunk_size``.
        chunk_overlap: Overlap characters. Defaults to ``settings.chunk_overlap``.

    Returns:
        List of Chunk objects derived from this Document.
        Returns an empty list if the document text is empty.
    """
    settings = get_settings()
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    if not doc.text.strip():
        logger.warning(
            "Document has no text — skipping",
            extra={"source": doc.source, "page": doc.page},
        )
        return []

    # Run the recursive splitting algorithm
    pieces = _split_text(doc.text, chunk_size, chunk_overlap)

    # Convert raw text pieces into Chunk objects with full metadata
    doc_id = make_doc_id(doc.source)

    chunks: list[Chunk] = []
    for idx, piece in enumerate(pieces):
        if not piece.strip():
            continue   # skip whitespace-only pieces

        chunks.append(
            Chunk(
                text=piece.strip(),
                source=doc.source,
                page=doc.page,
                chunk_index=idx,
                doc_id=doc_id,
                score=0.0,
                metadata=doc.metadata.copy(),
            )
        )

    logger.debug(
        "Document split into chunks",
        extra={
            "source": doc.source,
            "page": doc.page,
            "chunks": len(chunks),
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
        },
    )
    return chunks


def split_documents(
    docs: list[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Chunk]:
    """Split a list of Documents into a flat list of Chunks.

    This is the main entry point used by the ingest pipeline.

    Args:
        docs:         List of Documents to split (can be from multiple files).
        chunk_size:   Max characters per chunk. Defaults to ``settings.chunk_size``.
        chunk_overlap: Overlap characters. Defaults to ``settings.chunk_overlap``.

    Returns:
        Flat list of all Chunks across all Documents, in order.

    Example:
        >>> docs = load_documents(["data/report.pdf"])  # 10 pages → 10 Documents
        >>> chunks = split_documents(docs)              # → ~80 Chunks
        >>> len(chunks)
        83
        >>> chunks[0].source
        'data/report.pdf'
        >>> chunks[0].page
        1
        >>> len(chunks[0].text)   # ≤ chunk_size
        487
    """
    if not docs:
        return []

    all_chunks: list[Chunk] = []
    for doc in docs:
        chunks = split_document(doc, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        all_chunks.extend(chunks)

    logger.info(
        "Splitting complete",
        extra={
            "input_documents": len(docs),
            "output_chunks": len(all_chunks),
            "chunk_size": chunk_size or get_settings().chunk_size,
            "chunk_overlap": chunk_overlap or get_settings().chunk_overlap,
        },
    )
    return all_chunks
