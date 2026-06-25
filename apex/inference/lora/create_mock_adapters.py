#!/usr/bin/env python3
"""
Create mock LoRA adapter files for development and testing.

This script creates placeholder LoRA adapter files in the expected format.
These are development stubs that can be replaced with actual trained adapters
when domain-specific fine-tuning is complete.

Usage:
    python -m apex.inference.lora.create_mock_adapters
    python -m apex.inference.lora.create_mock_adapters --domain productivity
"""

import argparse
import os
import pickle
from pathlib import Path

import numpy as np
from loguru import logger

# Default LoRA parameters
DEFAULT_RANK = 8
DEFAULT_ALPHA = 16.0

# Mock layer dimensions (typical for transformer models)
LAYER_CONFIGS = [
    {"name": "embedding.lora_A", "shape": (384, DEFAULT_RANK)},
    {"name": "embedding.lora_B", "shape": (DEFAULT_RANK, 384)},
    {"name": "attention.q_proj.lora_A", "shape": (384, DEFAULT_RANK)},
    {"name": "attention.q_proj.lora_B", "shape": (DEFAULT_RANK, 384)},
    {"name": "attention.k_proj.lora_A", "shape": (384, DEFAULT_RANK)},
    {"name": "attention.k_proj.lora_B", "shape": (DEFAULT_RANK, 384)},
    {"name": "attention.v_proj.lora_A", "shape": (384, DEFAULT_RANK)},
    {"name": "attention.v_proj.lora_B", "shape": (DEFAULT_RANK, 384)},
    {"name": "feed_forward.up_proj.lora_A", "shape": (384, DEFAULT_RANK)},
    {"name": "feed_forward.up_proj.lora_B", "shape": (DEFAULT_RANK, 1536)},
    {"name": "feed_forward.down_proj.lora_A", "shape": (1536, DEFAULT_RANK)},
    {"name": "feed_forward.down_proj.lora_B", "shape": (DEFAULT_RANK, 384)},
]

# Domain configurations
DOMAINS = {
    "productivity": {
        "description": "Personal productivity (coding, writing, debugging)",
        "rank": 8,
        "alpha": 16.0,
        "seed": 42  # Reproducible weights
    },
    "factory": {
        "description": "Smart factory / industrial IoT",
        "rank": 16,  # Higher rank for complex industrial patterns
        "alpha": 32.0,
        "seed": 123
    },
    "research": {
        "description": "Research and academic analysis",
        "rank": 8,
        "alpha": 24.0,  # Higher alpha for stronger adaptation
        "seed": 456
    }
}


def create_mock_weights(rank: int, alpha: float, seed: int = 42) -> dict:
    """
    Create mock LoRA weights with realistic structure.

    Parameters
    ----------
    rank : int
        LoRA rank (low-rank dimension)
    alpha : float
        LoRA scaling parameter
    seed : int
        Random seed for reproducible weights

    Returns
    -------
    dict
        Dictionary of tensor names to weight arrays
    """
    np.random.seed(seed)
    weights = {}

    # Add LoRA metadata
    weights["lora_rank"] = rank
    weights["lora_alpha"] = alpha

    # Generate LoRA weights for each layer
    for layer_config in LAYER_CONFIGS:
        name = layer_config["name"]
        shape = layer_config["shape"]

        # Adjust shape if rank differs from default
        if "lora_A" in name:
            # A matrix: (input_dim, rank)
            adjusted_shape = (shape[0], rank)
        else:  # lora_B
            # B matrix: (rank, output_dim)
            adjusted_shape = (rank, shape[1])

        # Initialize with small random values (typical for LoRA)
        weight = np.random.normal(0, 0.01, adjusted_shape).astype(np.float32)
        weights[name] = weight

    logger.debug("Created mock weights: {} layers, rank={}, alpha={}",
                len([k for k in weights if "lora_" in k and k not in ["lora_rank", "lora_alpha"]]),
                rank, alpha)

    return weights


def save_adapter(domain: str, weights: dict, output_dir: Path, format: str = "bin") -> str:
    """
    Save LoRA adapter to file.

    Parameters
    ----------
    domain : str
        Domain name (e.g., "productivity")
    weights : dict
        LoRA weights dictionary
    output_dir : Path
        Output directory
    format : str
        File format: "bin" or "safetensors"

    Returns
    -------
    str
        Path to created file
    """
    filename = f"lora_{domain}.{format}"
    file_path = output_dir / filename

    if format == "bin":
        with open(file_path, "wb") as f:
            pickle.dump(weights, f)
        logger.info("Saved LoRA adapter: {}", file_path)
    elif format == "safetensors":
        try:
            from safetensors.numpy import save_file
            # Filter out non-tensor metadata for safetensors
            tensor_weights = {k: v for k, v in weights.items()
                            if isinstance(v, np.ndarray)}
            save_file(tensor_weights, file_path)
            logger.info("Saved LoRA adapter: {} (safetensors)", file_path)
        except ImportError:
            logger.warning("safetensors not available, creating .bin file instead")
            filename = f"lora_{domain}.bin"
            file_path = output_dir / filename
            with open(file_path, "wb") as f:
                pickle.dump(weights, f)
            logger.info("Saved LoRA adapter: {} (fallback to .bin)", file_path)
    else:
        raise ValueError(f"Unsupported format: {format}")

    return str(file_path)


def create_adapters(domains: list[str], output_dir: str, format: str = "bin") -> list[str]:
    """
    Create mock LoRA adapters for specified domains.

    Parameters
    ----------
    domains : list[str]
        Domain names to create adapters for
    output_dir : str
        Output directory path
    format : str
        File format: "bin" or "safetensors"

    Returns
    -------
    list[str]
        Paths to created files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    created_files = []

    for domain in domains:
        if domain not in DOMAINS:
            logger.warning("Unknown domain '{}', using default config", domain)
            config = {
                "description": f"Mock domain: {domain}",
                "rank": DEFAULT_RANK,
                "alpha": DEFAULT_ALPHA,
                "seed": 42
            }
        else:
            config = DOMAINS[domain]

        logger.info("Creating LoRA adapter for '{}': {}", domain, config["description"])

        # Create mock weights
        weights = create_mock_weights(
            rank=config["rank"],
            alpha=config["alpha"],
            seed=config["seed"]
        )

        # Save to file
        file_path = save_adapter(domain, weights, output_path, format)
        created_files.append(file_path)

    return created_files


def main():
    parser = argparse.ArgumentParser(
        description="Create mock LoRA adapter files for development",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available domains:
{chr(10).join(f"  {domain}: {config['description']}" for domain, config in DOMAINS.items())}

Examples:
  python -m apex.inference.lora.create_mock_adapters
  python -m apex.inference.lora.create_mock_adapters --domain productivity
  python -m apex.inference.lora.create_mock_adapters --format safetensors
        """
    )

    parser.add_argument(
        "--domain",
        choices=list(DOMAINS.keys()) + ["all"],
        default="all",
        help="Domain to create adapter for (default: all)"
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: apex/inference/lora/)"
    )

    parser.add_argument(
        "--format",
        choices=["bin", "safetensors"],
        default="bin",
        help="File format (default: bin)"
    )

    parser.add_argument(
        "--list-domains",
        action="store_true",
        help="List available domains and exit"
    )

    args = parser.parse_args()

    if args.list_domains:
        print("Available domains:")
        for domain, config in DOMAINS.items():
            print(f"  {domain}: {config['description']}")
            print(f"    rank={config['rank']}, alpha={config['alpha']}")
        return

    # Default output directory
    if args.output_dir is None:
        current_dir = Path(__file__).parent
        args.output_dir = str(current_dir)

    # Determine domains to create
    if args.domain == "all":
        domains_to_create = list(DOMAINS.keys())
    else:
        domains_to_create = [args.domain]

    logger.info("Creating mock LoRA adapters: {}", domains_to_create)
    logger.info("Output directory: {}", args.output_dir)
    logger.info("Format: {}", args.format)

    try:
        created_files = create_adapters(domains_to_create, args.output_dir, args.format)

        print(f"\n✅ Created {len(created_files)} LoRA adapter file(s):")
        for file_path in created_files:
            print(f"   {file_path}")

        print(f"\nTo test the infrastructure:")
        print(f"   uv run python -m pytest tests/test_lora_infrastructure.py -v")
        print(f"   uv run python -c \"from apex.inference.intent_engine import IntentEngine; print(IntentEngine().list_available_domains())\"")

    except Exception as e:
        logger.error("Failed to create adapters: {}", e)
        return 1

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())