"""
HNSW Dense Vector Index.

Wraps hnswlib to provide add/search/save/load with string chunk IDs.
hnswlib uses integer IDs internally; this class maintains a bidirectional
int↔str mapping so the rest of the pipeline never touches raw integers.

Invariants enforced here
------------------------
- search() accepts only numpy arrays — never strings or text.
- Results are returned as [(chunk_id: str, distance: float)] sorted
  by distance ascending (closest first).
- The index is content-blind: it stores and compares vectors only.
  It has no knowledge of what the vectors represent.

Embedding model: all-MiniLM-L6-v2 via Ollama (dim=384, INT8 quantized).
Space: cosine (distance = 1 − cosine_similarity, ∈ [0, 2]).
"""
from __future__ import annotations

import numpy as np
import hnswlib
from loguru import logger


class HNSWIndex:
    """
    Approximate Nearest Neighbour index for dense intent vectors.

    Parameters
    ----------
    dim
        Vector dimension. Must match the embedding model output.
        Default 384 for all-MiniLM-L6-v2.
    max_elements
        Maximum number of vectors the index can hold.
        Can be increased by rebuilding the index.
    ef_construction
        Build-time accuracy/speed trade-off. Higher = more accurate, slower build.
    M
        Number of bidirectional links per node. Higher = more accurate, more memory.
    ef_search
        Query-time accuracy parameter. Higher = more accurate, slower query.
    """

    def __init__(
        self,
        dim: int = 384,
        max_elements: int = 50_000,
        ef_construction: int = 200,
        M: int = 16,
        ef_search: int = 50,
    ) -> None:
        self._dim = dim
        self._max_elements = max_elements
        self._ef_search = ef_search

        self._index = hnswlib.Index(space="cosine", dim=dim)
        self._index.init_index(
            max_elements=max_elements,
            ef_construction=ef_construction,
            M=M,
        )
        self._index.set_ef(ef_search)

        # Bidirectional int↔str ID mapping
        self._int_to_str: dict[int, str] = {}
        self._str_to_int: dict[str, int] = {}
        self._next_id: int = 0

    # ── Public interface ─────────────────────────────────────────────────────

    def add(self, chunk_id: str, vector: np.ndarray) -> None:
        """
        Add a single vector to the index.

        Parameters
        ----------
        chunk_id
            String identifier for this chunk. Must be unique.
        vector
            Dense embedding, shape (dim,), dtype float32.
        """
        if chunk_id in self._str_to_int:
            logger.warning("HNSWIndex: chunk_id '{}' already indexed — skipping", chunk_id)
            return

        int_id = self._next_id
        self._next_id += 1
        self._int_to_str[int_id] = chunk_id
        self._str_to_int[chunk_id] = int_id

        vec = np.array(vector, dtype=np.float32).reshape(1, -1)

        # Auto-resize when approaching the cap (at 90% capacity).
        # hnswlib.resize_index() grows the allocation in-place without rebuilding.
        if self._next_id >= int(self._max_elements * 0.9):
            self._max_elements *= 2
            self._index.resize_index(self._max_elements)
            logger.info(
                "HNSWIndex: auto-resized to max_elements={}", self._max_elements
            )

        self._index.add_items(vec, [int_id])
        logger.debug("HNSWIndex: added chunk_id='{}' int_id={}", chunk_id, int_id)

    def search(self, query: np.ndarray, k: int = 10) -> list[tuple[str, float]]:
        """
        Find the k nearest neighbours to query in cosine space.

        Parameters
        ----------
        query
            Dense query vector, shape (dim,), dtype float32.
            Must be a numpy array — strings are not accepted.
        k
            Number of results to return.

        Returns
        -------
        List of (chunk_id, distance) sorted by distance ascending.
        Distance is cosine distance ∈ [0, 2] (0 = identical).
        """
        if not isinstance(query, np.ndarray):
            raise TypeError(
                f"HNSWIndex.search() requires a numpy array, got {type(query).__name__}. "
                "The HNSW index is searched with q̂ (dense vector), never with text."
            )

        n_indexed = len(self._int_to_str)
        if n_indexed == 0:
            return []

        k_actual = min(k, n_indexed)
        q = np.array(query, dtype=np.float32).reshape(1, -1)
        labels, distances = self._index.knn_query(q, k=k_actual)

        results: list[tuple[str, float]] = []
        for int_id, dist in zip(labels[0], distances[0]):
            chunk_id = self._int_to_str.get(int(int_id))
            if chunk_id is not None:
                results.append((chunk_id, float(dist)))

        return results  # already sorted ascending by hnswlib

    def save(self, path: str) -> None:
        """
        Persist the index and ID mapping to disk.

        Parameters
        ----------
        path
            File path for the hnswlib binary index.
            The ID mapping is saved to path + '.ids'.
        """
        import json
        self._index.save_index(path)
        with open(path + ".ids", "w") as f:
            json.dump({
                "int_to_str": {str(k): v for k, v in self._int_to_str.items()},
                "next_id": self._next_id,
            }, f)
        logger.info("HNSWIndex: saved {} vectors to '{}'", len(self._int_to_str), path)

    def load(self, path: str) -> None:
        """
        Load a previously saved index and ID mapping from disk.

        Parameters
        ----------
        path
            File path used in the corresponding save() call.
        """
        import json
        with open(path + ".ids") as f:
            data = json.load(f)
        self._int_to_str = {int(k): v for k, v in data["int_to_str"].items()}
        self._str_to_int = {v: int(k) for k, v in self._int_to_str.items()}
        self._next_id = data["next_id"]
        # Load with enough headroom for the stored vectors plus future additions.
        load_max = max(self._max_elements, self._next_id + 1000)
        self._index.load_index(path, max_elements=load_max)
        self._max_elements = load_max
        self._index.set_ef(self._ef_search)
        logger.info("HNSWIndex: loaded {} vectors from '{}'", len(self._int_to_str), path)

    def __len__(self) -> int:
        return len(self._int_to_str)
