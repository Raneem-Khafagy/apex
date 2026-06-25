"""
LoRA Adapter Loader for Intent Inference Engine.

Loads domain-specific LoRA adapters and applies them to base IIE model weights.
Supports both .bin (PyTorch) and .safetensors (HuggingFace) formats.

The loader scans apex/inference/lora/ for adapter files following the naming
convention: lora_{domain}.{bin|safetensors}

Example files:
    apex/inference/lora/lora_productivity.bin
    apex/inference/lora/lora_factory.safetensors
    apex/inference/lora/lora_research.bin
"""
from __future__ import annotations

import os
import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger


@dataclass
class LoRAAdapter:
    """
    Represents a loaded LoRA adapter for a specific domain.

    Parameters
    ----------
    domain : str
        The domain label this adapter targets (e.g., "productivity", "factory")
    weights : dict
        The LoRA adapter weights (A and B matrices per layer)
    rank : int
        LoRA rank (dimensionality of the low-rank decomposition)
    alpha : float
        LoRA scaling parameter
    """
    domain: str
    weights: Dict[str, Any]
    rank: int = 8
    alpha: float = 16.0

    @property
    def scaling_factor(self) -> float:
        """Compute the LoRA scaling factor: alpha / rank."""
        return self.alpha / self.rank if self.rank > 0 else 1.0


class LoRALoader:
    """
    LoRA adapter loader and manager.

    Scans for available domain adapters, loads them on demand, and provides
    a unified interface for applying domain-specific adaptations to the IIE.

    Parameters
    ----------
    lora_dir : str, optional
        Directory containing LoRA adapter files.
        Defaults to "apex/inference/lora/"
    """

    def __init__(self, lora_dir: Optional[str] = None):
        if lora_dir is None:
            # Default to lora/ directory relative to this file
            current_dir = Path(__file__).parent
            lora_dir = str(current_dir)

        self.lora_dir = Path(lora_dir)
        self.loaded_adapters: Dict[str, LoRAAdapter] = {}

        # Supported file formats
        self.supported_formats = [".bin", ".safetensors"]

        logger.debug("LoRALoader initialized with dir: {}", self.lora_dir)

    def scan_available_adapters(self) -> Dict[str, str]:
        """
        Scan the LoRA directory for available adapter files.

        Returns
        -------
        Dict[str, str]
            Mapping from domain name to file path.
            e.g., {"productivity": "/path/to/lora_productivity.bin"}
        """
        available = {}

        if not self.lora_dir.exists():
            logger.debug("LoRA directory does not exist: {}", self.lora_dir)
            return available

        for file_path in self.lora_dir.glob("lora_*"):
            if file_path.suffix in self.supported_formats:
                # Extract domain from filename: lora_productivity.bin → productivity
                domain = file_path.stem.replace("lora_", "")
                available[domain] = str(file_path)
                logger.debug("Found LoRA adapter: {} → {}", domain, file_path.name)

        return available

    def load_adapter(self, domain: str, file_path: str) -> LoRAAdapter:
        """
        Load a LoRA adapter from file.

        Parameters
        ----------
        domain : str
            Domain label for this adapter
        file_path : str
            Path to the .bin or .safetensors file

        Returns
        -------
        LoRAAdapter
            Loaded adapter ready for application

        Raises
        ------
        FileNotFoundError
            If the adapter file doesn't exist
        ValueError
            If the file format is not supported
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"LoRA adapter file not found: {file_path}")

        if path.suffix == ".bin":
            return self._load_bin_adapter(domain, path)
        elif path.suffix == ".safetensors":
            return self._load_safetensors_adapter(domain, path)
        else:
            raise ValueError(f"Unsupported LoRA format: {path.suffix}")

    def _load_bin_adapter(self, domain: str, path: Path) -> LoRAAdapter:
        """Load LoRA adapter from PyTorch .bin file."""
        try:
            # Use pickle to load .bin files (standard PyTorch format)
            with open(path, "rb") as f:
                weights = pickle.load(f)

            # Extract LoRA metadata if present
            rank = weights.get("lora_rank", 8)
            alpha = weights.get("lora_alpha", 16.0)

            # Remove metadata keys from weights dict
            adapter_weights = {k: v for k, v in weights.items()
                             if not k.startswith(("lora_rank", "lora_alpha", "_metadata"))}

            logger.info("Loaded LoRA adapter: {} (rank={}, alpha={})", domain, rank, alpha)

            return LoRAAdapter(
                domain=domain,
                weights=adapter_weights,
                rank=rank,
                alpha=alpha
            )

        except Exception as e:
            logger.error("Failed to load .bin LoRA adapter {}: {}", path, e)
            raise

    def _load_safetensors_adapter(self, domain: str, path: Path) -> LoRAAdapter:
        """Load LoRA adapter from HuggingFace .safetensors file."""
        try:
            # Try to import safetensors
            try:
                from safetensors import safe_open
            except ImportError:
                # Fallback: create a stub adapter for development
                warnings.warn(
                    f"safetensors not available, creating stub adapter for {domain}. "
                    "Install with: uv add safetensors"
                )
                return self._create_stub_adapter(domain)

            weights = {}
            with safe_open(path, framework="pt") as f:
                # Load all tensors from the safetensors file
                for key in f.keys():
                    weights[key] = f.get_tensor(key)

            # Extract LoRA metadata from the weights
            rank = 8  # Default rank
            alpha = 16.0  # Default alpha

            # Look for metadata in tensor names or separate metadata
            for key in weights:
                if "lora_rank" in key:
                    rank = int(weights[key].item())
                elif "lora_alpha" in key:
                    alpha = float(weights[key].item())

            # Filter out metadata tensors
            adapter_weights = {k: v for k, v in weights.items()
                             if not any(meta in k for meta in ["lora_rank", "lora_alpha", "_metadata"])}

            logger.info("Loaded LoRA adapter: {} (rank={}, alpha={})", domain, rank, alpha)

            return LoRAAdapter(
                domain=domain,
                weights=adapter_weights,
                rank=rank,
                alpha=alpha
            )

        except Exception as e:
            logger.error("Failed to load .safetensors LoRA adapter {}: {}", path, e)
            raise

    def _create_stub_adapter(self, domain: str) -> LoRAAdapter:
        """Create a stub adapter for development when safetensors is not available."""
        logger.warning("Creating stub LoRA adapter for domain '{}' (safetensors not available)", domain)
        return LoRAAdapter(
            domain=domain,
            weights={},  # Empty weights - no actual adaptation
            rank=8,
            alpha=16.0
        )

    def get_adapter(self, domain: str) -> Optional[LoRAAdapter]:
        """
        Get a loaded LoRA adapter for the specified domain.

        Loads the adapter if not already loaded, or returns None if no
        adapter file exists for this domain.

        Parameters
        ----------
        domain : str
            Domain label to get adapter for

        Returns
        -------
        LoRAAdapter or None
            Loaded adapter, or None if not available
        """
        # Return already loaded adapter
        if domain in self.loaded_adapters:
            return self.loaded_adapters[domain]

        # Scan for available adapters and load if found
        available = self.scan_available_adapters()

        if domain not in available:
            logger.debug("No LoRA adapter available for domain '{}'", domain)
            return None

        try:
            adapter = self.load_adapter(domain, available[domain])
            self.loaded_adapters[domain] = adapter
            return adapter
        except Exception as e:
            logger.warning("Failed to load LoRA adapter for '{}': {}", domain, e)
            return None

    def list_available_domains(self) -> list[str]:
        """
        List all domains that have available LoRA adapters.

        Returns
        -------
        list[str]
            List of domain names with available adapters
        """
        return list(self.scan_available_adapters().keys())

    def apply_adapter(self, base_weights: Dict[str, Any], adapter: LoRAAdapter) -> Dict[str, Any]:
        """
        Apply a LoRA adapter to base model weights.

        This is a placeholder implementation. In a real system, this would:
        1. Apply LoRA low-rank matrices (A @ B) to the base weights
        2. Scale by the adapter's scaling_factor
        3. Return the adapted weights

        Parameters
        ----------
        base_weights : Dict[str, Any]
            Base model weights (typically from Ollama/HuggingFace)
        adapter : LoRAAdapter
            LoRA adapter to apply

        Returns
        -------
        Dict[str, Any]
            Adapted weights ready for inference
        """
        if not adapter.weights:
            # Empty adapter (stub) - return base weights unchanged
            logger.debug("Applying empty LoRA adapter for '{}' (no-op)", adapter.domain)
            return base_weights

        # Placeholder: In production, this would perform actual LoRA math:
        # adapted_weight = base_weight + scaling_factor * (A @ B)
        logger.debug(
            "Applying LoRA adapter '{}' (rank={}, α={}, scaling={:.2f})",
            adapter.domain, adapter.rank, adapter.alpha, adapter.scaling_factor
        )

        # For now, return base weights (infrastructure stub)
        # TODO: Implement actual LoRA weight application when model loading is available
        adapted_weights = base_weights.copy()

        # Add metadata to track that adapter was applied
        adapted_weights["_lora_domain"] = adapter.domain
        adapted_weights["_lora_scaling"] = adapter.scaling_factor

        return adapted_weights