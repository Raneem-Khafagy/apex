"""
Tests for the Hybrid Retrieval Engine.

Real components throughout — real Ollama embeddings, real hnswlib, real SQLite.
No mocks, no patches.

A module-level fixture builds a shared RetrievalEngine populated with five
domain-representative chunks. All embedding calls go through real Ollama
all-minilm (same model used in production). Building the engine once amortizes
the embedding cost across the test session.

Chunks in the shared fixture:
  chunk_debug   — Python debugging / error tracing content
  chunk_write   — document writing / prose drafting content
  chunk_read    — academic reading / literature review content
  chunk_factory — industrial anomaly / sensor telemetry content
  chunk_legal   — legal clause / contract drafting content

These five chunks exercise both semantic (HNSW) and lexical (BM25) retrieval.
"""
import sqlite3

import numpy as np
import pytest

from apex.retrieval.bm25 import BM25Index
from apex.retrieval.hnsw_index import HNSWIndex
from apex.retrieval.rrf import Chunk, RetrievalEngine, rrf_fuse

# ── Module-level shared engine ───────────────────────────────────────────────
# Built once; all tests share it. Embedding 5 chunks via Ollama takes ~1s.

EMBED_DIM = 384


def _embed_real(text: str) -> np.ndarray:
    """Embed via real Ollama all-minilm — same path as production."""
    import ollama
    response = ollama.embed(model="all-minilm", input=text)
    vec = np.array(response.embeddings[0], dtype=np.float32)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


_CHUNKS = [
    Chunk("chunk_debug",   "Python traceback error debugging stack trace exception TypeError", "test_doc", "debugging_python"),
    Chunk("chunk_write",   "drafting a technical specification document writing structured prose", "test_doc", "writing_document"),
    Chunk("chunk_read",    "reading academic papers literature review citations bibliography methodology", "test_doc", "reading_reference"),
    Chunk("chunk_factory", "industrial sensor telemetry anomaly detection machine failure alert", "test_doc", "anomaly_response"),
    Chunk("chunk_legal",   "drafting legal contract clause terms conditions liability indemnification", "test_doc", "writing_document"),
]

# Build shared engine at module load — real Ollama embeddings
_engine = RetrievalEngine()
for _chunk in _CHUNKS:
    _vec = _embed_real(_chunk.text)
    _engine.add_chunk(_chunk, _vec)


# ── Chunk dataclass ──────────────────────────────────────────────────────────

class TestChunk:
    def test_chunk_has_required_fields(self):
        c = Chunk("id1", "some text", "source.md", "debugging_python")
        assert c.chunk_id == "id1"
        assert c.text == "some text"
        assert c.source == "source.md"
        assert c.label == "debugging_python"

    def test_chunk_score_defaults_zero(self):
        c = Chunk("id1", "text", "src", "label")
        assert c.score == 0.0

    def test_chunk_score_settable(self):
        c = Chunk("id1", "text", "src", "label", score=0.85)
        assert c.score == 0.85


# ── rrf_fuse pure function ───────────────────────────────────────────────────

class TestRRFFuse:
    def test_returns_sorted_by_score_descending(self):
        dense  = ["a", "b", "c"]
        sparse = ["b", "a", "c"]
        results = rrf_fuse(dense, sparse)
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)

    def test_chunk_in_both_lists_scores_higher(self):
        # "a" appears rank-1 in both lists → highest RRF score
        dense  = ["a", "b"]
        sparse = ["a", "c"]
        results = dict(rrf_fuse(dense, sparse))
        assert results["a"] > results["b"]
        assert results["a"] > results["c"]

    def test_chunk_only_in_dense_still_scored(self):
        dense  = ["a", "b"]
        sparse = ["c"]
        results = dict(rrf_fuse(dense, sparse))
        assert "a" in results
        assert "b" in results

    def test_chunk_only_in_sparse_still_scored(self):
        dense  = ["a"]
        sparse = ["a", "b"]
        results = dict(rrf_fuse(dense, sparse))
        assert "b" in results

    def test_rrf_formula_k60(self):
        # score(d) = Σ 1 / (k + rank + 1),  k=60, rank is 0-based
        # rank-0 in dense only → 1/(60+0+1) = 1/61
        dense  = ["a"]
        sparse = []
        results = dict(rrf_fuse(dense, sparse, k=60))
        assert abs(results["a"] - 1.0 / 61) < 1e-9

    def test_rrf_both_rank0_sums_correctly(self):
        # rank-0 in both → 1/61 + 1/61 = 2/61
        dense  = ["a"]
        sparse = ["a"]
        results = dict(rrf_fuse(dense, sparse, k=60))
        assert abs(results["a"] - 2.0 / 61) < 1e-9

    def test_empty_inputs_return_empty(self):
        assert rrf_fuse([], []) == []

    def test_custom_k_parameter(self):
        dense  = ["a"]
        sparse = []
        results_k10  = dict(rrf_fuse(dense, sparse, k=10))
        results_k100 = dict(rrf_fuse(dense, sparse, k=100))
        # Smaller k → higher score for rank-0
        assert results_k10["a"] > results_k100["a"]


# ── HNSWIndex ────────────────────────────────────────────────────────────────

class TestHNSWIndex:
    def test_add_and_search_returns_results(self):
        idx = HNSWIndex(dim=EMBED_DIM)
        v = _embed_real("python error debugging")
        idx.add("c1", v)
        results = idx.search(v, k=1)
        assert len(results) == 1

    def test_search_result_is_list_of_tuples(self):
        idx = HNSWIndex(dim=EMBED_DIM)
        v = _embed_real("writing document")
        idx.add("c1", v)
        results = idx.search(v, k=1)
        chunk_id, distance = results[0]
        assert isinstance(chunk_id, str)
        assert isinstance(distance, float)

    def test_nearest_neighbour_is_self(self):
        idx = HNSWIndex(dim=EMBED_DIM)
        v = _embed_real("academic paper citation")
        idx.add("self_chunk", v)
        idx.add("other_chunk", _embed_real("industrial sensor anomaly"))
        results = idx.search(v, k=1)
        assert results[0][0] == "self_chunk"

    def test_search_k_limits_results(self):
        idx = HNSWIndex(dim=EMBED_DIM)
        for i in range(5):
            idx.add(f"c{i}", _embed_real(f"document chunk number {i}"))
        results = idx.search(_embed_real("document chunk"), k=3)
        assert len(results) == 3

    def test_search_results_ordered_by_distance_ascending(self):
        idx = HNSWIndex(dim=EMBED_DIM)
        query_vec = _embed_real("python error")
        idx.add("close", _embed_real("python stack trace error"))
        idx.add("far",   _embed_real("agricultural crop irrigation"))
        results = idx.search(query_vec, k=2)
        distances = [d for _, d in results]
        assert distances == sorted(distances)

    def test_save_and_load_preserves_results(self, tmp_path):
        idx = HNSWIndex(dim=EMBED_DIM)
        v = _embed_real("machine learning model training")
        idx.add("ml_chunk", v)

        path = str(tmp_path / "test.idx")
        idx.save(path)

        idx2 = HNSWIndex(dim=EMBED_DIM)
        idx2.load(path)

        results = idx2.search(v, k=1)
        assert results[0][0] == "ml_chunk"

    def test_index_does_not_accept_text_as_query(self):
        """HNSW must only be searched with a vector — never a string."""
        idx = HNSWIndex(dim=EMBED_DIM)
        v = _embed_real("test")
        idx.add("c1", v)
        with pytest.raises((TypeError, AttributeError, ValueError)):
            idx.search("debugging_python", k=1)  # type: ignore


# ── BM25Index ────────────────────────────────────────────────────────────────

class TestBM25Index:
    def test_add_and_search_returns_results(self):
        idx = BM25Index()
        idx.add("c1", "Python debugging error traceback", label="debugging_python")
        results = idx.search("debugging python", k=5)
        assert len(results) >= 1

    def test_search_result_is_list_of_chunk_ids(self):
        idx = BM25Index()
        idx.add("c1", "writing document prose", label="writing_document")
        results = idx.search("writing document", k=5)
        assert all(isinstance(cid, str) for cid in results)

    def test_relevant_chunk_appears_in_results(self):
        idx = BM25Index()
        idx.add("debug_chunk", "Python traceback exception debugging stack trace", label="debugging_python")
        idx.add("write_chunk", "drafting document writing prose specification",    label="writing_document")
        results = idx.search("debugging python", k=5)
        assert "debug_chunk" in results

    def test_most_relevant_chunk_ranks_first(self):
        idx = BM25Index()
        idx.add("exact",  "debugging python error traceback",      label="debugging_python")
        idx.add("irrelevant", "agricultural irrigation crop field", label="reading_reference")
        results = idx.search("debugging python", k=2)
        assert results[0] == "exact"

    def test_label_used_as_query_not_vector(self):
        """BM25 accepts a string label — never a numpy vector."""
        idx = BM25Index()
        idx.add("c1", "some text content", label="writing_document")
        # Must work with underscored label split into words
        results = idx.search("writing document", k=5)
        assert isinstance(results, list)

    def test_empty_index_returns_empty(self):
        idx = BM25Index()
        results = idx.search("debugging python", k=5)
        assert results == []

    def test_k_limits_results(self):
        idx = BM25Index()
        for i in range(10):
            idx.add(f"c{i}", f"python debugging error chunk {i}", label="debugging_python")
        results = idx.search("debugging python", k=3)
        assert len(results) <= 3


# ── RetrievalEngine integration ──────────────────────────────────────────────

class TestRetrievalEngine:
    def test_search_returns_chunk_objects(self):
        results = _engine.search(_embed_real("python error"), label="debugging_python", k=3)
        assert len(results) >= 1
        assert all(isinstance(c, Chunk) for c in results)

    def test_search_results_have_scores(self):
        results = _engine.search(_embed_real("python error"), label="debugging_python", k=3)
        assert all(c.score > 0.0 for c in results)

    def test_search_results_ordered_by_score_descending(self):
        results = _engine.search(_embed_real("python error"), label="debugging_python", k=5)
        scores = [c.score for c in results]
        assert scores == sorted(scores, reverse=True)

    def test_debugging_query_returns_debug_chunk_first(self):
        """Semantic relevance: a debugging query must surface the debugging chunk."""
        q = _embed_real("python stack trace error exception")
        results = _engine.search(q, label="debugging_python", k=5)
        assert results[0].chunk_id == "chunk_debug", (
            f"Expected chunk_debug first, got {[c.chunk_id for c in results]}"
        )

    def test_writing_query_returns_writing_chunk_first(self):
        """Semantic relevance: a writing query must surface a writing chunk."""
        q = _embed_real("drafting document prose specification")
        results = _engine.search(q, label="writing_document", k=5)
        assert results[0].chunk_id in ("chunk_write", "chunk_legal"), (
            f"Expected a writing chunk first, got {[c.chunk_id for c in results]}"
        )

    def test_factory_query_returns_factory_chunk_first(self):
        """Semantic relevance: anomaly query must surface the factory chunk."""
        q = _embed_real("sensor anomaly industrial machine failure")
        results = _engine.search(q, label="anomaly_response", k=5)
        assert results[0].chunk_id == "chunk_factory", (
            f"Expected chunk_factory first, got {[c.chunk_id for c in results]}"
        )

    def test_search_k_limits_output(self):
        q = _embed_real("document")
        results = _engine.search(q, label="writing_document", k=2)
        assert len(results) <= 2

    def test_chunks_contain_text(self):
        q = _embed_real("python error")
        results = _engine.search(q, label="debugging_python", k=3)
        assert all(len(c.text) > 0 for c in results)

    def test_chunks_contain_source(self):
        q = _embed_real("python error")
        results = _engine.search(q, label="debugging_python", k=3)
        assert all(isinstance(c.source, str) for c in results)

    def test_engine_is_domain_blind(self):
        """
        The engine must not branch on label — it passes label as BM25 query only.
        Two different labels on the same vector must both return results.
        """
        q = _embed_real("document text content")
        r1 = _engine.search(q, label="debugging_python",  k=3)
        r2 = _engine.search(q, label="writing_document",  k=3)
        assert len(r1) >= 1
        assert len(r2) >= 1
