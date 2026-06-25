"""
BM25 Sparse Lexical Index.

Uses SQLite's built-in FTS5 extension, which implements BM25 ranking natively.
SQLite is already in the Python standard library — no external dependency.

Role in the pipeline
--------------------
Receives the task context label ℓ (e.g. "debugging_python") as the search
query. The label is tokenized by splitting on underscores before querying,
so "debugging_python" becomes "debugging python" and matches chunks containing
those words.

BM25 is complementary to HNSW:
- HNSW catches semantic similarity (paraphrases, synonyms)
- BM25 catches lexical overlap (exact keyword matches in the label)

The two results are fused by the RRF combiner in rrf.py.

Invariants
----------
- search() accepts only string labels — never numpy arrays.
- Results are returned as a list of chunk_ids (str) in BM25 rank order
  (most relevant first).
- The index is content-blind: it does not interpret the meaning of chunks.
"""
from __future__ import annotations

import sqlite3
from loguru import logger


def _label_to_query(label: str) -> str:
    """
    Convert a task context label to a BM25 FTS5 query string.
    "debugging_python" → "debugging python"
    """
    return " ".join(label.replace("_", " ").split())


class BM25Index:
    """
    BM25 lexical retrieval index backed by SQLite FTS5.

    Parameters
    ----------
    db_path
        SQLite database path. Use ":memory:" (default) for ephemeral indexes
        used during tests or live sessions. Pass a file path for persistence.
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._db_path = db_path
        self._setup()

    # ── Public interface ─────────────────────────────────────────────────────

    def add(self, chunk_id: str, text: str, label: str = "", *, commit: bool = True) -> None:
        """
        Index a chunk for lexical search.

        Parameters
        ----------
        chunk_id
            Unique string identifier. Must match the chunk_id in HNSWIndex.
        text
            The full text content of the chunk. This is the field searched
            by BM25 queries.
        label
            Optional domain label (e.g. "debugging_python"). Indexed alongside
            text so that label keywords also match in search results.
        commit
            If False, skip the per-row commit so callers can batch many inserts
            and call flush() once at the end.  Default True preserves the
            original single-row behaviour.
        """
        self._conn.execute(
            "INSERT INTO chunks(chunk_id, text, label) VALUES (?, ?, ?)",
            (chunk_id, text, label),
        )
        if commit:
            self._conn.commit()
        logger.debug("BM25Index: added chunk_id='{}'", chunk_id)

    def flush(self) -> None:
        """Commit any uncommitted inserts (call after bulk adds with commit=False)."""
        self._conn.commit()

    def search(self, query: str, k: int = 10) -> list[str]:
        """
        Return up to k chunk_ids ranked by BM25 relevance.

        Parameters
        ----------
        query
            Text query string. Typically the task context label ℓ with
            underscores split (e.g. "debugging python").
            Must be a string — numpy arrays are not accepted.
        k
            Maximum number of results.

        Returns
        -------
        List of chunk_ids in descending BM25 relevance order (best first).
        """
        if not query.strip():
            return []

        # Sanitize: FTS5 special characters that could cause syntax errors
        safe_query = _sanitize_fts5(query)
        if not safe_query:
            return []

        try:
            rows = self._conn.execute(
                "SELECT chunk_id FROM chunks WHERE chunks MATCH ? ORDER BY rank LIMIT ?",
                (safe_query, k),
            ).fetchall()
        except sqlite3.OperationalError as exc:
            logger.warning("BM25Index: FTS5 query error for '{}': {}", query, exc)
            return []

        return [row[0] for row in rows]

    def close(self) -> None:
        self._conn.close()

    # ── Internal ─────────────────────────────────────────────────────────────

    def _setup(self) -> None:
        """Create the FTS5 virtual table if it doesn't exist."""
        self._conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5(
                chunk_id UNINDEXED,
                text,
                label,
                tokenize = 'unicode61'
            )
        """)
        self._conn.commit()

    def __len__(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
        return row[0] if row else 0


def _sanitize_fts5(query: str) -> str:
    """
    Remove FTS5 special syntax characters that could cause OperationalError.
    Preserves alphanumeric words and spaces.
    """
    import re
    # Keep only word characters and spaces
    clean = re.sub(r'[^\w\s]', ' ', query)
    return " ".join(clean.split())
