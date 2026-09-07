"""CLI script for bulk document ingestion.

Usage: 
    python scripts/ingest_docs.py --source data/
    python scripts/ingest_docs.py --source data/test.txt
"""

import argparse
import sys
from pathlib import Path

# Add the project root (one directory up) to sys.path so we can import 'src'
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.config import get_settings
from src.ml.embedding import embed
from src.services.rag.document_loader import load_documents
from src.services.rag.text_splitter import split_documents
from src.storage import vector_db
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    """Ingest documents from a file or directory into Qdrant."""
    parser = argparse.ArgumentParser(description="Ingest documents into Qdrant.")
    parser.add_argument("--source", type=str, required=True, help="Path to file or directory")
    parser.add_argument("--clear", action="store_true", help="Clear the existing collection before ingesting")
    args = parser.parse_args()

    source = args.source
    settings = get_settings()
    collection = settings.qdrant_collection

    logger.info("Starting ingestion", extra={"source": source, "collection": collection})

    # 1. Load documents
    docs = load_documents([source])
    if not docs:
        logger.warning("No documents loaded. Check the source path and try again.")
        sys.exit(1)

    # 2. Split into chunks
    chunks = split_documents(docs)
    logger.info("Documents split", extra={"total_chunks": len(chunks)})
    
    if not chunks:
        logger.warning("No chunks generated. Documents might be empty.")
        sys.exit(1)

    # 3. Embed chunks
    logger.info("Embedding chunks...")
    texts = [chunk.text for chunk in chunks]
    # embed() expects a list of strings and returns a list of vectors
    vectors = embed(texts)

    # 4. Store in Qdrant
    logger.info("Upserting into Qdrant...", extra={"collection": collection})
    
    if args.clear:
        logger.warning("Clearing existing collection...", extra={"collection": collection})
        client = vector_db.get_client()
        client.delete_collection(collection_name=collection)
        
    vector_db.ensure_collection(collection, vector_size=settings.embedding_dim)
    
    payloads = []
    for i, chunk in enumerate(chunks):
        payload = {
            "text": chunk.text,
            "source": chunk.source,
            "page": chunk.page,
            "chunk_index": i
        }
        # Add any extra metadata the splitter/loader might have attached
        payload.update(chunk.metadata)
        payloads.append(payload)

    vector_db.upsert_points(collection=collection, vectors=vectors, payloads=payloads)

    logger.info("Ingestion complete! ✅", extra={"chunks_upserted": len(payloads)})


if __name__ == "__main__":
    main()
