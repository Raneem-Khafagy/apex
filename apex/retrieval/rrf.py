"""
Reciprocal Rank Fusion (RRF) combiner and RetrievalEngine.

RRF formula (Cormack et al., 2009):
    score(d) = Σ_r  1 / (k + rank_r(d))
where k=60 is a smoothing constant and rank_r(d) is the 0-based rank of
document d in retrieval system r.

Role in the pipeline
--------------------
RRF takes the ranked lists from HNSW (dense) and BM25 (sparse) and produces
a single fused ranking. Because it uses rank position (not raw scores), it is
immune to the scale mismatch between cosine distances and BM25 scores.

RetrievalEngine wraps HNSWIndex + BM25Index and exposes two methods:
    add_chunk(chunk, vector)    — populate the knowledge base
    search(q_hat, label, k)     → list[Chunk]  — query the knowledge base
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from loguru import logger

from apex.retrieval.bm25 import BM25Index, _label_to_query
from apex.retrieval.hnsw_index import HNSWIndex

# ── Chunk dataclass ──────────────────────────────────────────────────────────

@dataclass
class Chunk:
    """
    A retrieved text chunk with its origin metadata and RRF score.

    chunk_id
        Unique identifier shared across HNSW and BM25 indexes.
    text
        The raw text content of this chunk.
    source
        Origin document identifier (file path, URL, etc.).
    label
        Domain label used when the chunk was indexed (e.g. "debugging_python").
    score
        RRF fusion score assigned by rrf_fuse(). Higher = more relevant.
        Defaults to 0.0 before fusion.
    """
    chunk_id: str
    text: str
    source: str
    label: str
    score: float = 0.0


# ── RRF pure function ────────────────────────────────────────────────────────

def rrf_fuse(
    dense_ranked: list[str],
    sparse_ranked: list[str],
    k: int = 60,
) -> list[tuple[str, float]]:
    """
    Fuse two ranked lists of chunk_ids using Reciprocal Rank Fusion.

    Parameters
    ----------
    dense_ranked
        Chunk IDs in dense (HNSW) rank order, best first.
    sparse_ranked
        Chunk IDs in sparse (BM25) rank order, best first.
    k
        RRF smoothing constant. Default 60 per the original paper and CLAUDE.md.

    Returns
    -------
    List of (chunk_id, rrf_score) sorted by score descending.
    """
    scores: dict[str, float] = {}

    for rank, chunk_id in enumerate(dense_ranked):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)

    for rank, chunk_id in enumerate(sparse_ranked):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ── RetrievalEngine ──────────────────────────────────────────────────────────

class RetrievalEngine:
    """
    Hybrid retrieval engine: HNSW (dense) + BM25 (sparse) fused by RRF.

    The engine is content-blind — it does not interpret the meaning of chunks
    or branch on domain labels. The label is used only as the BM25 query string.

    Parameters
    ----------
    hnsw
        An HNSWIndex instance. Created internally if not provided.
    bm25
        A BM25Index instance. Created internally if not provided.
    """

    def __init__(
        self,
        hnsw: HNSWIndex | None = None,
        bm25: BM25Index | None = None,
    ) -> None:
        self._hnsw: HNSWIndex = hnsw or HNSWIndex()
        self._bm25: BM25Index = bm25 or BM25Index()
        # In-memory chunk store: chunk_id → Chunk (for text retrieval after ranking)
        self._store: dict[str, Chunk] = {}

    def add_chunk(self, chunk: Chunk, vector: np.ndarray) -> None:
        """
        Add a chunk to all indexes.

        Parameters
        ----------
        chunk
            The Chunk to store. chunk.text is indexed by BM25.
            chunk.label is indexed alongside text for lexical matching.
        vector
            Dense embedding for this chunk, shape (dim,).
            Produced by the ingestor via Ollama all-minilm.
        """
        self._store[chunk.chunk_id] = chunk
        self._hnsw.add(chunk.chunk_id, vector)
        self._bm25.add(chunk.chunk_id, chunk.text, label=chunk.label)
        logger.debug("RetrievalEngine: indexed chunk_id='{}'", chunk.chunk_id)

    def search(
        self,
        q_hat: np.ndarray,
        label: str,
        k: int = 5,
    ) -> list[Chunk]:
        """
        Retrieve the top-k most relevant chunks via RRF-fused hybrid search.

        Parameters
        ----------
        q_hat
            Dense intent vector from the IIE, shape (dim,).
            Passed directly to HNSW — never converted to text.
        label
            Task context label from the IIE, e.g. "debugging_python".
            Passed to BM25 as a lexical query (underscores split).
            Never passed to HNSW.
        k
            Maximum number of chunks to return.

        Returns
        -------
        List of Chunk objects ordered by RRF score descending.
        Each chunk has its .score field set to the fused RRF score.
        """
        retrieve_k = k * 3  # over-fetch so RRF has enough candidates to rerank

        # ── Dense retrieval (HNSW on q̂) ─────────────────────────────────────
        dense_results = self._hnsw.search(q_hat, k=retrieve_k)
        dense_ranked = [cid for cid, _ in dense_results]

        # ── Sparse retrieval (BM25 on ℓ) ─────────────────────────────────────
        bm25_query = _label_to_query(label)
        sparse_ranked = self._bm25.search(bm25_query, k=retrieve_k)

        logger.debug(
            "RetrievalEngine: dense={} sparse={} candidates before RRF",
            len(dense_ranked), len(sparse_ranked),
        )

        # ── RRF fusion ────────────────────────────────────────────────────────
        fused = rrf_fuse(dense_ranked, sparse_ranked, k=60)

        # ── Resolve chunk_ids → Chunk objects ────────────────────────────────
        results: list[Chunk] = []
        for chunk_id, rrf_score in fused[:k]:
            chunk = self._store.get(chunk_id)
            if chunk is not None:
                results.append(Chunk(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    source=chunk.source,
                    label=chunk.label,
                    score=rrf_score,
                ))

        logger.debug(
            "RetrievalEngine: returning {} chunks, top='{}'",
            len(results),
            results[0].chunk_id if results else "none",
        )
        return results

    def __len__(self) -> int:
        return len(self._store)
