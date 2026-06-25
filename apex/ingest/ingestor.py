"""
Document Ingestor — populate the knowledge base from local files.

Reads documents from the user's ApexVault, splits them into overlapping
text chunks, embeds each chunk via Ollama all-minilm, and adds them to
the RetrievalEngine (HNSW + BM25 indexes).

Privacy rule: document content stays on device.
The ingestor reads file content, but only to build the local index.
Nothing is sent to an external service.

Supported file types
---------------------
.py .js .ts  →  "debugging_python"
.md .txt .rst → "writing_document"
.pdf         →  "reading_reference"  (basic text extraction only)
others       →  "writing_document"   (default)
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import ollama
from loguru import logger

from apex.retrieval.rrf import Chunk, RetrievalEngine

EMBED_MODEL = "all-minilm"

# ── Extension → label mapping ─────────────────────────────────────────────────

_EXTENSION_LABEL: dict[str, str] = {
    ".py":   "debugging_python",
    ".js":   "debugging_python",
    ".ts":   "debugging_python",
    ".go":   "debugging_python",
    ".rs":   "debugging_python",
    ".java": "debugging_python",
    ".cpp":  "debugging_python",
    ".c":    "debugging_python",
    ".md":   "writing_document",
    ".txt":  "writing_document",
    ".rst":  "writing_document",
    ".tex":  "writing_document",
    ".pdf":  "reading_reference",
    ".html": "reading_reference",
}

_SUPPORTED_EXTENSIONS = set(_EXTENSION_LABEL.keys())


def label_from_path(path: str) -> str:
    """Return the task context label for a file based on its extension."""
    ext = Path(path).suffix.lower()
    return _EXTENSION_LABEL.get(ext, "writing_document")


# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_text(
    text: str,
    size: int = 512,
    overlap: int = 64,
) -> list[str]:
    """
    Split text into overlapping fixed-size character chunks.

    Parameters
    ----------
    text    Input text to chunk.
    size    Maximum characters per chunk.
    overlap Characters shared between consecutive chunks.

    Returns
    -------
    List of text chunks. Empty if text is empty.
    """
    if not text.strip():
        return []

    chunks: list[str] = []
    start = 0
    step = max(1, size - overlap)

    while start < len(text):
        end = min(start + size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start += step

    return chunks


# Maximum characters sent to all-minilm in a single embed call.
# all-MiniLM-L6-v2 has a 256-token context limit.  Dense text (all-caps,
# code, tables) can tokenize at 2–3 chars/token, so 350 chars ≈ 120–175
# tokens — safely under the cap for any content type.
_EMBED_CHAR_LIMIT = 350


# ── Embedding ─────────────────────────────────────────────────────────────────

def _embed(text: str) -> np.ndarray:
    """
    Embed text via Ollama all-minilm. Returns normalised float32 ndarray.

    If Ollama rejects the text for exceeding its context window (status 400),
    the text is halved and retried once.  This is a safety net for edge-case
    chunks that slip past the character limit (e.g. dense tables, all-caps
    military text).  Normal chunks are always under _EMBED_CHAR_LIMIT.
    """
    for attempt, t in enumerate((text, text[: len(text) // 2])):
        try:
            response = ollama.embed(model=EMBED_MODEL, input=t)
            vec = np.array(response.embeddings[0], dtype=np.float32)
            norm = np.linalg.norm(vec)
            if attempt > 0:
                logger.debug("Ingestor: embed retry succeeded (truncated to {} chars)", len(t))
            return vec / norm if norm > 0 else vec
        except Exception as exc:
            if attempt == 0 and "context length" in str(exc).lower():
                logger.warning(
                    "Ingestor: chunk too long for all-minilm ({} chars), retrying truncated",
                    len(text),
                )
                continue
            raise
    # unreachable — loop always returns or raises
    raise RuntimeError("embed failed")


# ── Ingestor ──────────────────────────────────────────────────────────────────

class Ingestor:
    """
    Document ingestor: files → chunks → embeddings → RetrievalEngine.

    Parameters
    ----------
    engine
        The RetrievalEngine to populate.
    chunk_size
        Maximum characters per chunk. Default 350 (_EMBED_CHAR_LIMIT).
        Sized to stay under all-minilm's 256-token context for any content type.
    overlap
        Overlapping characters between consecutive chunks. Default 40.
    """

    def __init__(
        self,
        engine: RetrievalEngine,
        chunk_size: int = _EMBED_CHAR_LIMIT,
        overlap: int = 40,
    ) -> None:
        self._engine = engine
        self._chunk_size = chunk_size
        self._overlap = overlap
        # chunk_id → (text, source, label) — persisted on save
        self._metadata: dict[str, dict] = {}

    # ── Public interface ─────────────────────────────────────────────────────

    def ingest_file(self, path: str, label: str = "") -> int:
        """
        Ingest a single file into the RetrievalEngine.

        Parameters
        ----------
        path    Absolute or relative file path.
        label   Task context label. If empty, inferred from file extension.

        Returns
        -------
        Number of chunks ingested. 0 if the file was skipped (binary/empty).
        """
        effective_label = label or label_from_path(path)

        try:
            text = Path(path).read_text(encoding="utf-8", errors="strict")
        except (UnicodeDecodeError, PermissionError) as exc:
            logger.debug("Ingestor: skipping '{}' ({})", path, exc)
            return 0

        # Control characters (null bytes) indicate a binary file even if
        # the bytes happen to be valid UTF-8 sequences.
        if "\x00" in text:
            logger.debug("Ingestor: skipping '{}' (null bytes — binary content)", path)
            return 0

        if not text.strip():
            return 0

        chunks = chunk_text(text, size=self._chunk_size, overlap=self._overlap)
        path_hash = hashlib.sha256(path.encode()).hexdigest()[:8]
        count = 0

        for i, chunk_text_content in enumerate(chunks):
            chunk_id = f"{path_hash}_{i}"
            chunk = Chunk(
                chunk_id=chunk_id,
                text=chunk_text_content,
                source=path,
                label=effective_label,
            )
            vec = _embed(chunk_text_content)
            self._engine.add_chunk(chunk, vec)
            self._metadata[chunk_id] = {
                "text": chunk_text_content,
                "source": path,
                "label": effective_label,
            }
            count += 1

        logger.info(
            "Ingestor: ingested '{}' → {} chunk(s) label='{}'",
            path, count, effective_label,
        )
        return count

    def ingest_directory(self, directory: str) -> int:
        """
        Recursively ingest all supported files in a directory.

        Parameters
        ----------
        directory   Root directory to scan.

        Returns
        -------
        Total number of chunks ingested across all files.
        """
        total = 0
        for root, _, files in os.walk(directory):
            for filename in files:
                ext = Path(filename).suffix.lower()
                if ext not in _SUPPORTED_EXTENSIONS:
                    logger.debug("Ingestor: skipping unsupported file '{}'", filename)
                    continue
                path = os.path.join(root, filename)
                total += self.ingest_file(path)
        logger.info("Ingestor: directory '{}' → {} total chunk(s)", directory, total)
        return total

    def save_index(self, index_path: str) -> None:
        """
        Persist the HNSW index and chunk metadata to disk.

        Parameters
        ----------
        index_path   Base path. HNSW saved to index_path + '.hnsw',
                     metadata to index_path + '.meta.json'.
        """
        self._engine._hnsw.save(index_path + ".hnsw")
        with open(index_path + ".meta.json", "w") as f:
            json.dump(self._metadata, f)
        logger.info("Ingestor: saved index to '{}'", index_path)

    def load_index(self, index_path: str) -> None:
        """
        Load a previously saved index from disk and restore the engine state.

        Parameters
        ----------
        index_path   Base path used in the corresponding save_index() call.
        """
        self._engine._hnsw.load(index_path + ".hnsw")
        with open(index_path + ".meta.json") as f:
            self._metadata = json.load(f)

        # Restore chunk store and BM25 index from metadata
        for chunk_id, meta in self._metadata.items():
            chunk = Chunk(
                chunk_id=chunk_id,
                text=meta["text"],
                source=meta["source"],
                label=meta["label"],
            )
            # Restore in-memory store (BM25 is rebuilt from metadata)
            self._engine._store[chunk_id] = chunk
            self._engine._bm25.add(chunk_id, meta["text"], label=meta["label"])

        logger.info(
            "Ingestor: loaded {} chunk(s) from '{}'",
            len(self._metadata), index_path,
        )


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Re-index the knowledge base from the ApexVault directory.

    Reads the following environment variables (with sensible defaults):
        APEX_VAULT_PATH   Directory containing documents to ingest.
                          Default: ~/Documents/ApexVault
        APEX_INDEX_PATH   Base path for the saved index files.
                          Default: ./apex_vault  (produces apex_vault.hnsw
                          and apex_vault.meta.json in the project directory)

    Usage:
        uv run python -m apex.ingest.ingestor
        APEX_VAULT_PATH=~/my/docs uv run python -m apex.ingest.ingestor
    """
    import os
    import sys

    vault_path = os.path.expanduser(
        os.environ.get("APEX_VAULT_PATH", "~/Documents/ApexVault")
    )
    index_path = os.environ.get("APEX_INDEX_PATH", "apex_vault")

    if not os.path.isdir(vault_path):
        logger.error(
            "ApexVault directory does not exist: '{}'\n"
            "Create it and add documents, or set APEX_VAULT_PATH.",
            vault_path,
        )
        sys.exit(1)

    logger.info("Ingestor: starting from vault='{}'", vault_path)
    engine = RetrievalEngine()
    ingestor = Ingestor(engine)
    total = ingestor.ingest_directory(vault_path)

    if total == 0:
        logger.warning(
            "Ingestor: no chunks produced from '{}'. "
            "Add supported files (.py, .md, .txt, .rst, .ts, .go, …).",
            vault_path,
        )
        sys.exit(0)

    ingestor.save_index(index_path)
    logger.info(
        "Ingestor: complete. {} chunk(s) indexed. "
        "Index saved to '{}.hnsw' + '{}.meta.json'.",
        total, index_path, index_path,
    )
