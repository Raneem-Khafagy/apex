"""
Tests for the document Ingestor.

Real Ollama all-minilm embeddings — no mocks.
Tests create real temp files and verify the RetrievalEngine is populated.
"""
import os
import tempfile

import pytest

from apex.ingest.ingestor import Ingestor, chunk_text, label_from_path
from apex.retrieval.rrf import RetrievalEngine
import ollama
import numpy as np


def _embed(text: str) -> np.ndarray:
    r = ollama.embed(model="all-minilm", input=text)
    v = np.array(r.embeddings[0], dtype=np.float32)
    return v / (np.linalg.norm(v) or 1.0)


# ── chunk_text pure function ─────────────────────────────────────────────────

class TestChunkText:
    def test_short_text_is_single_chunk(self):
        chunks = chunk_text("hello world", size=512, overlap=64)
        assert len(chunks) == 1
        assert chunks[0] == "hello world"

    def test_long_text_splits_into_multiple_chunks(self):
        text = "word " * 300          # ~1500 chars
        chunks = chunk_text(text, size=200, overlap=40)
        assert len(chunks) > 1

    def test_chunks_cover_full_text(self):
        text = "abcdefghij" * 50      # 500 chars
        chunks = chunk_text(text, size=100, overlap=20)
        # Reconstruct: first chunk starts at 0, subsequent start after (size - overlap)
        assert chunks[0].startswith("abcd")
        assert text[-5:] in chunks[-1]

    def test_overlap_means_adjacent_chunks_share_content(self):
        text = "A" * 50 + "B" * 50 + "C" * 50
        chunks = chunk_text(text, size=60, overlap=20)
        assert len(chunks) >= 2
        # The boundary region appears in consecutive chunks
        assert any("A" in c and "B" in c for c in chunks)

    def test_empty_text_returns_empty_list(self):
        assert chunk_text("", size=512, overlap=64) == []

    def test_chunk_size_is_respected(self):
        text = "x" * 1000
        chunks = chunk_text(text, size=100, overlap=0)
        assert all(len(c) <= 100 for c in chunks)


# ── label_from_path ──────────────────────────────────────────────────────────

class TestLabelFromPath:
    def test_py_maps_to_debugging(self):
        assert label_from_path("main.py") == "debugging_python"

    def test_md_maps_to_writing(self):
        assert label_from_path("notes.md") == "writing_document"

    def test_txt_maps_to_writing(self):
        assert label_from_path("readme.txt") == "writing_document"

    def test_pdf_maps_to_reading(self):
        assert label_from_path("paper.pdf") == "reading_reference"

    def test_unknown_extension_returns_default(self):
        label = label_from_path("file.xyz")
        assert isinstance(label, str) and len(label) > 0

    def test_path_with_directory(self):
        label = label_from_path("/home/user/project/module.py")
        assert label == "debugging_python"


# ── Ingestor ─────────────────────────────────────────────────────────────────

class TestIngestor:
    def _make_engine_and_ingestor(self):
        engine = RetrievalEngine()
        ingestor = Ingestor(engine)
        return engine, ingestor

    def test_ingest_file_returns_chunk_count(self):
        engine, ingestor = self._make_engine_and_ingestor()
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w",
                                        delete=False) as f:
            f.write("def foo():\n    pass\n\n" * 10)
            path = f.name
        try:
            count = ingestor.ingest_file(path)
            assert count >= 1
        finally:
            os.unlink(path)

    def test_ingest_file_populates_engine(self):
        engine, ingestor = self._make_engine_and_ingestor()
        with tempfile.NamedTemporaryFile(suffix=".md", mode="w",
                                        delete=False) as f:
            f.write("# Guide\n\nThis document explains debugging Python errors.\n" * 5)
            path = f.name
        try:
            ingestor.ingest_file(path)
            q = _embed("python debugging error")
            results = engine.search(q, label="debugging_python", k=3)
            assert len(results) >= 1
        finally:
            os.unlink(path)

    def test_ingest_file_assigns_label_from_extension(self):
        engine, ingestor = self._make_engine_and_ingestor()
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w",
                                        delete=False) as f:
            f.write("# python module\ndef main(): pass\n")
            path = f.name
        try:
            ingestor.ingest_file(path)
            q = _embed("python code")
            results = engine.search(q, label="debugging_python", k=3)
            assert all(c.label == "debugging_python" for c in results)
        finally:
            os.unlink(path)

    def test_ingest_file_custom_label_overrides_extension(self):
        engine, ingestor = self._make_engine_and_ingestor()
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w",
                                        delete=False) as f:
            f.write("writing prose\n" * 5)
            path = f.name
        try:
            ingestor.ingest_file(path, label="writing_document")
            q = _embed("writing document")
            results = engine.search(q, label="writing_document", k=3)
            assert all(c.label == "writing_document" for c in results)
        finally:
            os.unlink(path)

    def test_ingest_directory_ingests_all_supported_files(self):
        engine, ingestor = self._make_engine_and_ingestor()
        with tempfile.TemporaryDirectory() as tmpdir:
            for name, content in [
                ("module.py",  "def debug(): raise TypeError('error')"),
                ("notes.md",   "# Notes\nThis documents the API design."),
                ("ignore.bin", "\x00\x01\x02"),  # binary — must be skipped
            ]:
                with open(os.path.join(tmpdir, name), "w",
                          errors="ignore") as f:
                    f.write(content)
            count = ingestor.ingest_directory(tmpdir)
        assert count >= 2  # .py and .md ingested; .bin skipped

    def test_ingest_empty_directory_returns_zero(self):
        engine, ingestor = self._make_engine_and_ingestor()
        with tempfile.TemporaryDirectory() as tmpdir:
            count = ingestor.ingest_directory(tmpdir)
        assert count == 0

    def test_save_and_load_index(self, tmp_path):
        engine, ingestor = self._make_engine_and_ingestor()
        with tempfile.NamedTemporaryFile(suffix=".md", mode="w",
                                        delete=False) as f:
            f.write("Python debugging traceback error analysis\n" * 5)
            doc_path = f.name
        try:
            ingestor.ingest_file(doc_path)
            index_path = str(tmp_path / "test_index")
            ingestor.save_index(index_path)

            # New engine + ingestor, load saved index
            engine2 = RetrievalEngine()
            ingestor2 = Ingestor(engine2)
            ingestor2.load_index(index_path)

            q = _embed("python error debugging")
            results = engine2.search(q, label="debugging_python", k=3)
            assert len(results) >= 1
        finally:
            os.unlink(doc_path)

    def test_ingest_skips_binary_files_gracefully(self):
        engine, ingestor = self._make_engine_and_ingestor()
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            f.write(b"\x00\x01\x02\x03" * 100)  # binary content
            path = f.name
        try:
            count = ingestor.ingest_file(path)  # must not raise
            assert count == 0
        finally:
            os.unlink(path)
