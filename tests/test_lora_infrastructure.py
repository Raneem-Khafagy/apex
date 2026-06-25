#!/usr/bin/env python3
"""
Test LoRA loading infrastructure for Intent Inference Engine.

Tests the LoRA adapter loading, domain mapping, and integration with the
IntentEngine without requiring actual trained adapter files.
"""

import os
import pickle
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from apex.inference.lora import LoRALoader, LoRAAdapter
from apex.inference.intent_engine import IntentEngine
from apex.adapters.base import SignalVector


class TestLoRAAdapter:
    """Test LoRAAdapter dataclass functionality."""

    def test_lora_adapter_creation(self):
        """Test LoRAAdapter creation and properties."""
        weights = {
            "layer_1.A": np.random.randn(128, 8),
            "layer_1.B": np.random.randn(8, 128),
            "layer_2.A": np.random.randn(64, 8),
            "layer_2.B": np.random.randn(8, 64)
        }

        adapter = LoRAAdapter(
            domain="productivity",
            weights=weights,
            rank=8,
            alpha=16.0
        )

        assert adapter.domain == "productivity"
        assert adapter.rank == 8
        assert adapter.alpha == 16.0
        assert adapter.scaling_factor == 2.0  # alpha / rank = 16 / 8

    def test_scaling_factor_calculation(self):
        """Test LoRA scaling factor calculation."""
        # Normal case
        adapter1 = LoRAAdapter("test", {}, rank=8, alpha=16.0)
        assert adapter1.scaling_factor == 2.0

        # High alpha
        adapter2 = LoRAAdapter("test", {}, rank=4, alpha=32.0)
        assert adapter2.scaling_factor == 8.0

        # Edge case: rank = 0 (should not divide by zero)
        adapter3 = LoRAAdapter("test", {}, rank=0, alpha=16.0)
        assert adapter3.scaling_factor == 1.0


class TestLoRALoader:
    """Test LoRALoader file scanning and loading functionality."""

    def test_loader_initialization(self):
        """Test LoRALoader initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = LoRALoader(lora_dir=tmpdir)
            assert loader.lora_dir == Path(tmpdir)
            assert loader.loaded_adapters == {}

    def test_scan_empty_directory(self):
        """Test scanning an empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = LoRALoader(lora_dir=tmpdir)
            available = loader.scan_available_adapters()
            assert available == {}

    def test_scan_with_adapter_files(self):
        """Test scanning directory with mock adapter files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock adapter files
            (Path(tmpdir) / "lora_productivity.bin").touch()
            (Path(tmpdir) / "lora_factory.safetensors").touch()
            (Path(tmpdir) / "lora_research.bin").touch()
            (Path(tmpdir) / "not_a_lora.txt").touch()  # Should be ignored
            (Path(tmpdir) / "lora_invalid.xyz").touch()  # Unsupported format

            loader = LoRALoader(lora_dir=tmpdir)
            available = loader.scan_available_adapters()

            assert "productivity" in available
            assert "factory" in available
            assert "research" in available
            assert available["productivity"].endswith("lora_productivity.bin")
            assert available["factory"].endswith("lora_factory.safetensors")
            assert available["research"].endswith("lora_research.bin")

            # Should ignore non-LoRA files
            assert len(available) == 3

    def test_scan_nonexistent_directory(self):
        """Test scanning a directory that doesn't exist."""
        loader = LoRALoader(lora_dir="/path/that/does/not/exist")
        available = loader.scan_available_adapters()
        assert available == {}

    def test_load_bin_adapter(self):
        """Test loading a .bin adapter file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a mock .bin file
            weights = {
                "layer_1.A": np.random.randn(128, 8).astype(np.float32),
                "layer_1.B": np.random.randn(8, 128).astype(np.float32),
                "lora_rank": 8,
                "lora_alpha": 16.0
            }

            bin_path = Path(tmpdir) / "lora_productivity.bin"
            with open(bin_path, "wb") as f:
                pickle.dump(weights, f)

            loader = LoRALoader(lora_dir=tmpdir)
            adapter = loader.load_adapter("productivity", str(bin_path))

            assert adapter.domain == "productivity"
            assert adapter.rank == 8
            assert adapter.alpha == 16.0
            assert adapter.scaling_factor == 2.0

            # Check that weights were loaded (excluding metadata)
            assert "layer_1.A" in adapter.weights
            assert "layer_1.B" in adapter.weights
            assert "lora_rank" not in adapter.weights  # Metadata should be filtered
            assert "lora_alpha" not in adapter.weights

    def test_load_safetensors_adapter_fallback(self):
        """Test loading safetensors adapter with invalid file (triggers fallback)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create an invalid .safetensors file (will trigger error handling)
            safetensors_path = Path(tmpdir) / "lora_factory.safetensors"
            with open(safetensors_path, "wb") as f:
                f.write(b"invalid safetensors data")  # Not a valid safetensors file

            loader = LoRALoader(lora_dir=tmpdir)

            # Should raise an error since the file is invalid and safetensors is available
            with pytest.raises(Exception):  # Could be SafetensorError or other exception
                loader.load_adapter("factory", str(safetensors_path))

    def test_load_safetensors_import_fallback(self):
        """Test safetensors import fallback (when safetensors not available)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            safetensors_path = Path(tmpdir) / "lora_factory.safetensors"
            safetensors_path.touch()

            loader = LoRALoader(lora_dir=tmpdir)

            # Mock ImportError for safetensors
            with patch('builtins.__import__', side_effect=ImportError("No module named 'safetensors'")):
                with pytest.warns(UserWarning, match="safetensors not available"):
                    adapter = loader.load_adapter("factory", str(safetensors_path))

                # Should create a stub adapter
                assert adapter.domain == "factory"
                assert adapter.weights == {}  # Empty weights for stub
                assert adapter.rank == 8
                assert adapter.alpha == 16.0

    def test_load_unsupported_format(self):
        """Test loading unsupported file format raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            unsupported_path = Path(tmpdir) / "lora_test.xyz"
            unsupported_path.touch()

            loader = LoRALoader(lora_dir=tmpdir)
            with pytest.raises(ValueError, match="Unsupported LoRA format"):
                loader.load_adapter("test", str(unsupported_path))

    def test_load_nonexistent_file(self):
        """Test loading nonexistent file raises error."""
        loader = LoRALoader()
        with pytest.raises(FileNotFoundError):
            loader.load_adapter("test", "/path/that/does/not/exist.bin")

    def test_get_adapter_loads_and_caches(self):
        """Test get_adapter loads and caches adapters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock adapter file
            weights = {"layer_1.A": np.random.randn(64, 4).astype(np.float32)}
            bin_path = Path(tmpdir) / "lora_test.bin"
            with open(bin_path, "wb") as f:
                pickle.dump(weights, f)

            loader = LoRALoader(lora_dir=tmpdir)

            # First call should load the adapter
            adapter1 = loader.get_adapter("test")
            assert adapter1 is not None
            assert adapter1.domain == "test"

            # Second call should return cached adapter
            adapter2 = loader.get_adapter("test")
            assert adapter2 is adapter1  # Same object reference

    def test_get_adapter_returns_none_for_unavailable(self):
        """Test get_adapter returns None for unavailable domains."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = LoRALoader(lora_dir=tmpdir)
            adapter = loader.get_adapter("nonexistent_domain")
            assert adapter is None

    def test_list_available_domains(self):
        """Test listing available domains."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock files
            (Path(tmpdir) / "lora_productivity.bin").touch()
            (Path(tmpdir) / "lora_factory.bin").touch()

            loader = LoRALoader(lora_dir=tmpdir)
            domains = loader.list_available_domains()

            assert "productivity" in domains
            assert "factory" in domains
            assert len(domains) == 2


class TestIntentEngineLoRAIntegration:
    """Test IntentEngine integration with LoRA adapters."""

    def test_intent_engine_lora_disabled(self):
        """Test IntentEngine with LoRA disabled."""
        # Create mock vector table to avoid Ollama dependency
        mock_vectors = {
            "writing_document": np.random.random(384).astype(np.float32),
            "debugging_python": np.random.random(384).astype(np.float32),
        }

        engine = IntentEngine(
            vector_table=mock_vectors,
            enable_lora=False
        )

        assert engine._enable_lora is False
        assert engine._lora_loader is None
        assert engine._domain_adapters == {}
        assert engine.list_available_domains() == []

    def test_intent_engine_lora_enabled_empty_dir(self):
        """Test IntentEngine with LoRA enabled but no adapters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_vectors = {
                "writing_document": np.random.random(384).astype(np.float32),
            }

            engine = IntentEngine(
                vector_table=mock_vectors,
                enable_lora=True,
                lora_dir=tmpdir
            )

            assert engine._enable_lora is True
            assert engine._lora_loader is not None
            assert engine._domain_adapters == {}
            assert engine.list_available_domains() == []

    def test_intent_engine_lora_enabled_with_adapters(self):
        """Test IntentEngine with LoRA adapters available."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock adapter files
            weights = {"layer_1.A": np.random.randn(64, 8).astype(np.float32)}
            for domain in ["productivity", "factory", "research"]:
                bin_path = Path(tmpdir) / f"lora_{domain}.bin"
                with open(bin_path, "wb") as f:
                    pickle.dump(weights, f)

            mock_vectors = {
                "writing_document": np.random.random(384).astype(np.float32),
            }

            engine = IntentEngine(
                vector_table=mock_vectors,
                enable_lora=True,
                lora_dir=tmpdir
            )

            assert engine._enable_lora is True
            assert engine._lora_loader is not None
            assert len(engine._domain_adapters) == 3

            domains = engine.list_available_domains()
            assert "productivity" in domains
            assert "factory" in domains
            assert "research" in domains

    def test_has_domain_adapter(self):
        """Test has_domain_adapter method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create one adapter
            weights = {"layer_1.A": np.random.randn(32, 4).astype(np.float32)}
            bin_path = Path(tmpdir) / "lora_productivity.bin"
            with open(bin_path, "wb") as f:
                pickle.dump(weights, f)

            mock_vectors = {"writing": np.random.random(384).astype(np.float32)}
            engine = IntentEngine(
                vector_table=mock_vectors,
                enable_lora=True,
                lora_dir=tmpdir
            )

            assert engine.has_domain_adapter("productivity") is True
            assert engine.has_domain_adapter("factory") is False
            assert engine.has_domain_adapter("nonexistent") is False

    def test_get_adapter_info(self):
        """Test get_adapter_info method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create adapter with specific metadata
            weights = {
                "layer_1.A": np.random.randn(64, 16).astype(np.float32),
                "lora_rank": 16,
                "lora_alpha": 32.0
            }
            bin_path = Path(tmpdir) / "lora_productivity.bin"
            with open(bin_path, "wb") as f:
                pickle.dump(weights, f)

            mock_vectors = {"writing": np.random.random(384).astype(np.float32)}
            engine = IntentEngine(
                vector_table=mock_vectors,
                enable_lora=True,
                lora_dir=tmpdir
            )

            info = engine.get_adapter_info("productivity")
            assert info is not None
            assert info["domain"] == "productivity"
            assert info["rank"] == 16
            assert info["alpha"] == 32.0
            assert info["scaling_factor"] == 2.0  # 32 / 16
            assert info["weights_loaded"] is True
            assert info["adapter_path"].endswith("lora_productivity.bin")

            # Test nonexistent domain
            assert engine.get_adapter_info("nonexistent") is None

    def test_extract_domain_from_label(self):
        """Test domain extraction from task labels."""
        mock_vectors = {"writing": np.random.random(384).astype(np.float32)}
        engine = IntentEngine(vector_table=mock_vectors, enable_lora=False)

        # Productivity patterns
        assert engine._extract_domain_from_label("debugging_python") == "productivity"
        assert engine._extract_domain_from_label("writing_document") == "productivity"
        assert engine._extract_domain_from_label("coding_javascript") == "productivity"

        # Factory patterns
        assert engine._extract_domain_from_label("factory_anomaly") == "factory"
        assert engine._extract_domain_from_label("sensor_monitoring") == "factory"
        assert engine._extract_domain_from_label("maintenance_alert") == "factory"

        # Research patterns
        assert engine._extract_domain_from_label("research_paper") == "research"
        assert engine._extract_domain_from_label("reading_reference") == "research"
        assert engine._extract_domain_from_label("academic_analysis") == "research"

        # Unknown pattern
        assert engine._extract_domain_from_label("unknown_activity") is None

    @pytest.mark.asyncio
    async def test_domain_adaptation_in_inference(self):
        """Test that domain adaptation is called during LLM inference."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create adapter
            weights = {"layer_1.A": np.random.randn(32, 8).astype(np.float32)}
            bin_path = Path(tmpdir) / "lora_productivity.bin"
            with open(bin_path, "wb") as f:
                pickle.dump(weights, f)

            mock_vectors = {"writing": np.random.random(384).astype(np.float32)}
            engine = IntentEngine(
                vector_table=mock_vectors,
                enable_lora=True,
                lora_dir=tmpdir
            )

            # Mock the LLM call to avoid Ollama dependency
            with patch.object(engine, '_llm_infer', return_value=(
                "debugging_python",  # Label that maps to productivity domain
                np.random.random(384).astype(np.float32)
            )):
                # Mock the domain adaptation method to verify it's called
                with patch.object(engine, '_apply_domain_adaptation',
                                return_value=np.random.random(384).astype(np.float32)) as mock_adapt:

                    signal = SignalVector(
                        source_id="test",
                        content_hash="hash",
                        activity_type="unknown",  # Will trigger LLM path
                        velocity_metric=1.0,
                        temporal_proximity=0.5,
                        urgency_flag=False
                    )

                    q_hat, c, label = await engine.infer(signal)

                    # Verify adaptation was called
                    mock_adapt.assert_called_once()
                    call_args = mock_adapt.call_args
                    assert call_args[0][0] == "debugging_python"  # Label
                    assert isinstance(call_args[0][1], np.ndarray)  # Base vector

    def test_apply_domain_adaptation_placeholder(self):
        """Test _apply_domain_adaptation placeholder behavior."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create adapter
            weights = {"layer_1.A": np.random.randn(32, 8).astype(np.float32)}
            bin_path = Path(tmpdir) / "lora_productivity.bin"
            with open(bin_path, "wb") as f:
                pickle.dump(weights, f)

            mock_vectors = {"writing": np.random.random(384).astype(np.float32)}
            engine = IntentEngine(
                vector_table=mock_vectors,
                enable_lora=True,
                lora_dir=tmpdir
            )

            base_vector = np.random.random(384).astype(np.float32)

            # Test with known domain
            adapted_vector = engine._apply_domain_adaptation("debugging_python", base_vector)
            # Currently returns base vector unchanged (placeholder)
            np.testing.assert_array_equal(adapted_vector, base_vector)

            # Test with unknown domain
            adapted_vector = engine._apply_domain_adaptation("unknown_task", base_vector)
            np.testing.assert_array_equal(adapted_vector, base_vector)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])