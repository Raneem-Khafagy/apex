"""
LoRA Domain Adapters for Intent Inference Engine.

Provides lightweight domain-specific fine-tuning on top of the base IIE model
without retraining the full model. Each domain gets its own LoRA adapter that
can be loaded at runtime.

Architecture:
    Base IIE weights (universal, shared) + LoRA adapter (domain-specific, ~MB)

Supported formats:
    - .bin (PyTorch binary)
    - .safetensors (HuggingFace safe format)

Domain mapping:
    - productivity → lora_productivity.bin
    - factory → lora_factory.bin
    - research → lora_research.bin
"""

from .loader import LoRALoader, LoRAAdapter

__all__ = ["LoRALoader", "LoRAAdapter"]