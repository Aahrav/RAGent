"""Document loaders for PDF, plain text, and Markdown files.

Converts raw files into a list of Document objects.
Each Document holds the text of one page (PDF) or the whole file (txt/md),
plus the source path and page number.

Why one Document per page for PDFs?
  - Preserves page context — citations can reference "page 4"
  - Avoids huge single-document strings that exceed token limits
  - Lets the text splitter work on smaller, coherent units

Supported formats:
  - .pdf  → extracted via pypdf, one Document per page
  - .txt  → read as UTF-8, one Document for the whole file
  - .md   → read as UTF-8, one Document for the whole file

Usage:
    from src.services.rag.document_loader import load_documents

    docs = load_documents(["data/report.pdf", "data/notes/"])
    # Returns a flat list of Document objects across all input paths
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

from src.services.rag.models import Document
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Supported file extensions ──────────────────────────────────────────────────

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".pdf", ".txt", ".md"})


# ── Per-format loaders ─────────────────────────────────────────────────────────

def _load_pdf(path: Path) -> list[Document]:
    """Extract text from a PDF, one Document per page.

    Uses pypdf — pure Python, no system dependencies.

    Args:
        path: Path to the PDF file.

    Returns:
        List of Documents, one per page. Pages with no extractable text
        are skipped (scanned images without OCR, for example).
    """
    try:
        from pypdf import PdfReader  # lazy import — only needed for PDFs
    except ImportError as exc:
        raise ImportError(
            "pypdf is required to load PDF files. "
            "Run: pip install pypdf"
        ) from exc

    docs: list[Document] = []

    reader = PdfReader(str(path))
    total_pages = len(reader.pages)

    logger.debug(
        "Loading PDF",
        extra={"path": str(path), "total_pages": total_pages},
    )

    for page_num, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:
            logger.warning(
                "Could not extract text from PDF page — skipping",
                extra={"path": str(path), "page": page_num, "error": str(exc)},
            )
            continue

        text = text.strip()
        if not text:
            # Page has no extractable text (e.g. scanned image page)
            logger.debug(
                "Skipping empty PDF page",
                extra={"path": str(path), "page": page_num},
            )
            continue

        docs.append(
            Document(
                text=text,
                source=str(path),
                page=page_num,
                metadata={"total_pages": total_pages},
            )
        )

    logger.info(
        "PDF loaded",
        extra={
            "path": str(path),
            "pages_loaded": len(docs),
            "pages_total": total_pages,
        },
    )
    return docs


def _load_text(path: Path) -> list[Document]:
    """Load a plain text file as a single Document.

    Args:
        path: Path to the .txt or .md file.

    Returns:
        A one-element list containing the whole file as a Document.
        Returns an empty list if the file is empty.
    """
    try:
        text = path.read_text(encoding="utf-8").strip()
    except UnicodeDecodeError:
        # Fallback to latin-1 for files with non-UTF-8 encoding
        logger.warning(
            "UTF-8 decode failed — retrying with latin-1",
            extra={"path": str(path)},
        )
        text = path.read_text(encoding="latin-1").strip()

    if not text:
        logger.warning("Text file is empty — skipping", extra={"path": str(path)})
        return []

    doc = Document(
        text=text,
        source=str(path),
        page=1,
        metadata={"file_type": path.suffix.lstrip(".")},
    )

    logger.info(
        "Text file loaded",
        extra={"path": str(path), "chars": len(text)},
    )
    return [doc]


# ── Extension → loader mapping ─────────────────────────────────────────────────

_LOADERS: dict[str, Callable[[Path], list[Document]]] = {
    ".pdf": _load_pdf,
    ".txt": _load_text,
    ".md": _load_text,   # Markdown is plain text — same loader
}


# ── Public API ─────────────────────────────────────────────────────────────────

def load_file(path: str | Path) -> list[Document]:
    """Load a single file and return its Documents.

    Args:
        path: Path to the file. Must be an existing file (not a directory).

    Returns:
        List of Document objects. Empty list if the file has no extractable text.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If the file extension is not supported.
    """
    path = Path(path).resolve()

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if not path.is_file():
        raise ValueError(f"Expected a file, got a directory: {path}")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{ext}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    loader = _LOADERS[ext]
    return loader(path)


def load_directory(directory: str | Path) -> list[Document]:
    """Recursively load all supported files from a directory.

    Args:
        directory: Path to the directory to scan.

    Returns:
        Flat list of Documents from all supported files found.
        Files with unsupported extensions are silently skipped.

    Raises:
        NotADirectoryError: If the path is not a directory.
    """
    directory = Path(directory).resolve()

    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    docs: list[Document] = []
    files_found = 0
    files_loaded = 0

    # rglob("*") walks the directory tree recursively
    for file_path in sorted(directory.rglob("*")):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        files_found += 1
        try:
            file_docs = load_file(file_path)
            docs.extend(file_docs)
            files_loaded += 1
        except Exception as exc:
            # Log the error but keep going — don't let one bad file stop everything
            logger.error(
                "Failed to load file — skipping",
                extra={"path": str(file_path), "error": str(exc)},
            )

    logger.info(
        "Directory loaded",
        extra={
            "directory": str(directory),
            "files_found": files_found,
            "files_loaded": files_loaded,
            "total_documents": len(docs),
        },
    )
    return docs


def load_documents(sources: list[str]) -> list[Document]:
    """Load documents from a mixed list of file paths and/or directory paths.

    This is the main entry point — the pipeline calls this with whatever
    the user passed to POST /ingest.

    Args:
        sources: List of paths. Each can be:
                 - A path to a supported file (.pdf, .txt, .md)
                 - A path to a directory (all supported files inside are loaded)

    Returns:
        Flat list of all Documents across all sources, in order.

    Example:
        >>> docs = load_documents([
        ...     "data/reports/Q3.pdf",
        ...     "data/notes/",          # loads all txt/md/pdf files inside
        ...     "data/readme.md",
        ... ])
    """
    if not sources:
        logger.warning("load_documents called with empty sources list")
        return []

    all_docs: list[Document] = []

    for source in sources:
        path = Path(source)

        if path.is_dir():
            docs = load_directory(path)
        elif path.is_file():
            docs = load_file(path)
        else:
            logger.error(
                "Source path does not exist — skipping",
                extra={"source": source},
            )
            continue

        all_docs.extend(docs)

    logger.info(
        "All sources loaded",
        extra={"sources": len(sources), "total_documents": len(all_docs)},
    )
    return all_docs
