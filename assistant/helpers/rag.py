"""RAG (Retrieval-Augmented Generation) helper functions.

This module handles document indexing and retrieval for Stage 5 of the workshop.
It uses:
- sentence-transformers for embeddings (runs locally, no API calls)
- ChromaDB for vector storage and similarity search
- PyMuPDF for PDF text extraction
"""

from __future__ import annotations

import os
from pathlib import Path

# Lazy imports to avoid slow startup when RAG isn't used
_chroma_client = None
_collection = None
_embedding_model = None

COLLECTION_NAME = "assistant_docs"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500  # characters per chunk
CHUNK_OVERLAP = 50  # overlap between chunks


def _get_embedding_model():
    """Lazy-load the sentence-transformers model."""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def _get_collection():
    """Lazy-load the ChromaDB collection."""
    global _chroma_client, _collection
    if _collection is None:
        import chromadb
        # Store the DB in the project directory
        db_path = Path(__file__).parent.parent.parent / ".chroma_db"
        _chroma_client = chromadb.PersistentClient(path=str(db_path))
        _collection = _chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks.

    Args:
        text: The text to split.
        chunk_size: Maximum characters per chunk.
        overlap: Number of characters to overlap between chunks.

    Returns:
        List of text chunks.
    """
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap

    return chunks


def _extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text content from a PDF file.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Extracted text content.
    """
    import fitz  # PyMuPDF

    text_parts = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text_parts.append(page.get_text())

    return "\n".join(text_parts)


def _extract_text_from_file(file_path: str) -> str:
    """Extract text from a file (PDF or plain text).

    Args:
        file_path: Path to the file.

    Returns:
        Extracted text content.
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return _extract_text_from_pdf(file_path)
    elif suffix in (".txt", ".md", ".py", ".js", ".json", ".csv"):
        return path.read_text(encoding="utf-8", errors="ignore")
    else:
        # Try to read as text
        try:
            return path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return ""


def index_documents(folder_path: str) -> int:
    """Index all documents in a folder for RAG retrieval.

    This function:
    1. Scans the folder for supported files (PDF, TXT, MD, etc.)
    2. Extracts text from each file
    3. Chunks the text into smaller pieces
    4. Embeds each chunk using sentence-transformers
    5. Stores embeddings in ChromaDB

    Args:
        folder_path: Path to the folder containing documents.

    Returns:
        Number of chunks indexed.

    Example:
        >>> count = index_documents("./docs")
        >>> print(f"Indexed {count} chunks")
    """
    folder = Path(folder_path)
    if not folder.exists():
        raise ValueError(f"Folder does not exist: {folder_path}")

    collection = _get_collection()
    model = _get_embedding_model()

    # Supported file extensions
    extensions = {".pdf", ".txt", ".md", ".py", ".js", ".json", ".csv"}

    all_chunks = []
    all_ids = []
    all_metadata = []

    for file_path in folder.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in extensions:
            text = _extract_text_from_file(str(file_path))
            if not text.strip():
                continue

            chunks = _chunk_text(text)
            for i, chunk in enumerate(chunks):
                chunk_id = f"{file_path.name}_{i}"
                all_chunks.append(chunk)
                all_ids.append(chunk_id)
                all_metadata.append({
                    "source": file_path.name,
                    "chunk_index": i,
                })

    if not all_chunks:
        return 0

    # Generate embeddings
    embeddings = model.encode(all_chunks, show_progress_bar=False).tolist()

    # Upsert into ChromaDB (handles duplicates)
    collection.upsert(
        ids=all_ids,
        embeddings=embeddings,
        documents=all_chunks,
        metadatas=all_metadata,
    )

    return len(all_chunks)


def retrieve_context(query: str, top_k: int = 3) -> list[str]:
    """Retrieve the most relevant document chunks for a query.

    This function:
    1. Embeds the query using the same model used for indexing
    2. Searches ChromaDB for similar chunks
    3. Returns the top-k most relevant chunks

    Args:
        query: The user's question.
        top_k: Number of chunks to retrieve.

    Returns:
        List of relevant text chunks.

    Example:
        >>> chunks = retrieve_context("What did Professor Smith say about recursion?")
        >>> for chunk in chunks:
        ...     print(chunk[:100] + "...")
    """
    collection = _get_collection()
    model = _get_embedding_model()

    # Check if collection is empty
    if collection.count() == 0:
        return []

    # Embed the query
    query_embedding = model.encode([query], show_progress_bar=False).tolist()[0]

    # Search for similar chunks
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
    )

    # Extract the document texts
    documents = results.get("documents", [[]])[0]
    return documents


def clear_index() -> None:
    """Clear all indexed documents from the vector store.

    Useful for re-indexing or testing.
    """
    global _collection
    if _collection is not None:
        _collection.delete(where={})
