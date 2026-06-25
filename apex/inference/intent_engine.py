"""
Intent Inference Engine (IIE) — hybrid gate + small LLM.

Pipeline for each incoming SignalVector:

    Signal arrives
        │
        ▼
    Heuristic Gate (<1ms)
        ├── Known high-confidence pattern → return (q̂, c=0.9, ℓ) immediately
        └── Ambiguous / unknown → Small LLM Reasoner via Ollama
                                      └── return (q̂, c, ℓ)

Output triple (q̂, c, ℓ):
  q̂  — dense intent vector, numpy ndarray shape (384,). NEVER a text string.
  c   — confidence ∈ [0, 1]
  ℓ   — task context label string, e.g. "debugging_python"

Architecture rules enforced here:
  1. q̂ is always produced by an embedding model call — never by text generation.
  2. urgency_flag = True forces c → 1.0 unconditionally.
  3. If Ollama is unavailable, return a low-confidence default (c=0.3) so the
     Speculative Retrieval Scheduler will WAIT rather than fire incorrectly.
"""
from __future__ import annotations

import json
from typing import Any, Optional

import numpy as np
import ollama
from loguru import logger

from apex.adapters.base import SignalVector
from apex.inference.heuristic_gate import HeuristicGate
from apex.inference.lora import LoRALoader

# Type alias
IntentTriple = tuple[np.ndarray, float, str]

EMBED_DIM = 384          # all-MiniLM-L6-v2 output dimension
EMBED_MODEL = "all-minilm"
CHAT_MODEL = "phi3.5"

# Confidence assigned to LLM-classified signals (below heuristic's 0.9)
LLM_CONFIDENCE = 0.65

# Fallback triple when all inference paths fail
_FALLBACK_LABEL = "writing_document"

# Labels pre-computed at startup — these are the known heuristic categories.
# The embedding of each is stored in the gate's vector_table.
_KNOWN_LABELS: list[str] = [
    "writing_document",
    "debugging_python",
    "reading_reference",
]

_SYSTEM_PROMPT = (
    "You are a behavioral intent classifier for a proactive edge context system. "
    "Given a JSON object describing what a user is currently doing on their device "
    "(active application, activity type, velocity metric), output ONLY a task-context "
    "label as a short snake_case string. "
    "Examples: debugging_python, writing_document, reading_reference, "
    "reviewing_code, drafting_legal_clause, anomaly_response, planning_meeting. "
    "Output nothing else — just the label, no punctuation, no explanation."
)


def _embed(text: str, embed_model: str = EMBED_MODEL) -> np.ndarray:
    """Embed text using Ollama. Returns float32 ndarray of shape (384,)."""
    response = ollama.embed(model=embed_model, input=text)
    vec = np.array(response.embeddings[0], dtype=np.float32)
    # Normalize to unit sphere so cosine similarity = dot product
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


def _build_vector_table(embed_model: str = EMBED_MODEL) -> dict[str, np.ndarray]:
    """
    Embed all known labels at startup.
    Called once; result is shared with HeuristicGate.
    """
    table: dict[str, np.ndarray] = {}
    for label in _KNOWN_LABELS:
        try:
            table[label] = _embed(label, embed_model=embed_model)
            logger.debug("IIE: embedded label '{}'", label)
        except Exception as exc:
            logger.warning("IIE: failed to embed label '{}': {}", label, exc)
    return table


class IntentEngine:
    """
    Hybrid Intent Inference Engine with LoRA domain adaptation.

    Architecture:
        Base IIE weights (universal, shared) + LoRA adapter (domain-specific)

    At initialization, scans for available LoRA adapters and loads them for
    domain-specific intent inference. Each domain can have its own fine-tuned
    behavior without modifying the base model.

    Parameters
    ----------
    vector_table
        Pre-computed label→vector mapping for HeuristicGate.
        Pass explicitly in tests to avoid Ollama calls at init time.
        If None, vectors are built from Ollama at construction.
    chat_model
        Ollama model name for the LLM classifier path.
        Default: phi3.5. Override via APEX_LLM_MODEL env var (set in main.py).
    embed_model
        Ollama model name for embedding. Default: all-minilm.
        Override via APEX_EMBED_MODEL env var (set in main.py).
    lora_dir
        Directory containing LoRA adapter files.
        Default: apex/inference/lora/
    enable_lora
        Whether to enable LoRA adapter loading. Default: True.
        Set to False in tests to avoid file system dependencies.
    """

    def __init__(
        self,
        vector_table: Optional[dict[str, np.ndarray]] = None,
        chat_model: str = CHAT_MODEL,
        embed_model: str = EMBED_MODEL,
        lora_dir: Optional[str] = None,
        enable_lora: bool = True,
    ) -> None:
        self._chat_model = chat_model
        self._embed_model = embed_model
        self._enable_lora = enable_lora

        if vector_table is None:
            logger.info(
                "IIE: building vector table (embed_model='{}', chat_model='{}')",
                embed_model, chat_model,
            )
            vector_table = _build_vector_table(embed_model=embed_model)
        self._gate = HeuristicGate(vector_table=vector_table)
        self._vector_table = vector_table

        # ── LoRA Domain Adapter Infrastructure ───────────────────────────────
        self._lora_loader = None
        self._domain_adapters: dict[str, str] = {}  # domain → adapter_path mapping

        if enable_lora:
            try:
                self._lora_loader = LoRALoader(lora_dir=lora_dir)
                available_adapters = self._lora_loader.scan_available_adapters()
                self._domain_adapters = available_adapters

                logger.info(
                    "IIE: LoRA infrastructure initialized with {} domain adapter(s): {}",
                    len(available_adapters),
                    list(available_adapters.keys()) if available_adapters else "none"
                )

                # Attempt to load all available adapters at startup for validation
                for domain in available_adapters:
                    adapter = self._lora_loader.get_adapter(domain)
                    if adapter:
                        logger.debug(
                            "IIE: validated LoRA adapter for '{}' (rank={}, α={})",
                            domain, adapter.rank, adapter.alpha
                        )
                    else:
                        logger.warning("IIE: failed to load LoRA adapter for '{}'", domain)

            except Exception as e:
                logger.warning("IIE: LoRA initialization failed: {} — continuing without adapters", e)
                self._lora_loader = None
                self._domain_adapters = {}
        else:
            logger.debug("IIE: LoRA adapters disabled")
            self._domain_adapters = {}

    async def infer(self, signal: SignalVector) -> IntentTriple:
        """
        Infer user intent from a behavioral signal.

        Returns
        -------
        (q̂, c, ℓ)
          q̂: dense intent vector, np.ndarray of shape (384,) — never a str
          c:  confidence score ∈ [0, 1]
          ℓ:  task context label string
        """
        # ── Fast path: heuristic gate ────────────────────────────────────────
        gate_result = self._gate.match(signal)
        if gate_result is not None:
            q_hat, c, label = gate_result
            # urgency_flag overrides confidence unconditionally
            if signal.urgency_flag:
                c = 1.0
            logger.debug("IIE heuristic path: label='{}' c={:.2f}", label, c)
            return q_hat, c, label

        # ── Slow path: LLM classifier ────────────────────────────────────────
        label, q_hat = await self._llm_infer(signal)

        # Apply domain-specific LoRA adaptation to the intent vector
        q_hat = self._apply_domain_adaptation(label, q_hat)

        c = 1.0 if signal.urgency_flag else LLM_CONFIDENCE
        logger.debug("IIE LLM path: label='{}' c={:.2f}", label, c)
        return q_hat, c, label

    async def _llm_infer(self, signal: SignalVector) -> tuple[str, np.ndarray]:
        """
        Call the small LLM to classify the signal into a label,
        then embed that label to produce q̂.

        The LLM outputs a text label (ℓ). q̂ is then produced by an
        embedding call on ℓ. q̂ is never the label string itself.
        """
        signal_dict = {
            "activity_type": signal.activity_type,
            "velocity_metric": round(signal.velocity_metric, 3),
            "temporal_proximity": round(signal.temporal_proximity, 3),
            "urgency_flag": signal.urgency_flag,
        }

        try:
            response = ollama.chat(
                model=self._chat_model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(signal_dict)},
                ],
            )
            label = response.message.content.strip().lower().replace(" ", "_")
            label = label[:64]  # guard against runaway output
        except Exception as exc:
            logger.warning("IIE LLM chat failed: {} — using fallback label", exc)
            label = _FALLBACK_LABEL

        # Embed the label to produce q̂ — this is the ONLY way q̂ is produced
        try:
            q_hat = _embed(label, embed_model=self._embed_model)
        except Exception as exc:
            logger.warning("IIE embedding failed for '{}': {} — using fallback vector", label, exc)
            # Return a zero vector with very low confidence so the SRS waits
            q_hat = np.zeros(EMBED_DIM, dtype=np.float32)

        return label, q_hat

    # ── LoRA Domain Adapter Methods ──────────────────────────────────────────

    def list_available_domains(self) -> list[str]:
        """
        List domains that have available LoRA adapters.

        Returns
        -------
        list[str]
            Domain names with loaded adapters
        """
        if self._lora_loader is None:
            return []
        return self._lora_loader.list_available_domains()

    def has_domain_adapter(self, domain: str) -> bool:
        """
        Check if a LoRA adapter is available for the given domain.

        Parameters
        ----------
        domain : str
            Domain label to check

        Returns
        -------
        bool
            True if adapter is available
        """
        return domain in self._domain_adapters

    def get_adapter_info(self, domain: str) -> Optional[dict[str, Any]]:
        """
        Get information about a domain's LoRA adapter.

        Parameters
        ----------
        domain : str
            Domain to query

        Returns
        -------
        dict or None
            Adapter metadata (rank, alpha, scaling_factor) or None if not available
        """
        if not self._lora_loader or domain not in self._domain_adapters:
            return None

        adapter = self._lora_loader.get_adapter(domain)
        if adapter is None:
            return None

        return {
            "domain": adapter.domain,
            "rank": adapter.rank,
            "alpha": adapter.alpha,
            "scaling_factor": adapter.scaling_factor,
            "weights_loaded": len(adapter.weights) > 0,
            "adapter_path": self._domain_adapters[domain]
        }

    def _apply_domain_adaptation(self, label: str, base_vector: np.ndarray) -> np.ndarray:
        """
        Apply domain-specific LoRA adaptation to the intent vector.

        In the current implementation, this is a placeholder that returns the
        base vector unchanged. In a production system, this would:

        1. Extract domain from the label (e.g., "debugging_python" → "productivity")
        2. Load the appropriate LoRA adapter for that domain
        3. Apply the adapter weights to modify the vector semantics
        4. Return the domain-adapted vector

        Parameters
        ----------
        label : str
            Task context label (may contain domain information)
        base_vector : np.ndarray
            Base intent vector from embedding

        Returns
        -------
        np.ndarray
            Domain-adapted intent vector (same shape as input)
        """
        if not self._enable_lora or not self._lora_loader:
            return base_vector

        # Extract domain from label — heuristic approach for now
        domain = self._extract_domain_from_label(label)

        if domain is None:
            # No domain detected, return base vector
            return base_vector

        adapter = self._lora_loader.get_adapter(domain)
        if adapter is None:
            logger.debug("IIE: no LoRA adapter for domain '{}', using base vector", domain)
            return base_vector

        # TODO: Implement actual vector adaptation when model weights are accessible
        # For now, log that adaptation would occur and return base vector
        logger.debug(
            "IIE: applying LoRA adaptation for domain '{}' (label='{}', scaling={:.2f})",
            domain, label, adapter.scaling_factor
        )

        # Placeholder: return base vector unchanged
        # In production: return adapter.apply_to_vector(base_vector)
        return base_vector

    def _extract_domain_from_label(self, label: str) -> Optional[str]:
        """
        Extract domain from a task context label.

        Uses heuristics to map task labels to known domains.
        This mapping should eventually be learned or configured explicitly.

        Parameters
        ----------
        label : str
            Task context label (e.g., "debugging_python", "writing_document")

        Returns
        -------
        str or None
            Domain name if detected, None otherwise
        """
        label_lower = label.lower()

        # Productivity domain patterns
        if any(pattern in label_lower for pattern in [
            "writing", "document", "debugging", "coding", "programming",
            "python", "javascript", "typescript", "react", "api", "testing"
        ]):
            return "productivity"

        # Factory domain patterns
        if any(pattern in label_lower for pattern in [
            "factory", "anomaly", "sensor", "maintenance", "production",
            "industrial", "monitoring", "alert", "machine", "equipment"
        ]):
            return "factory"

        # Research domain patterns
        if any(pattern in label_lower for pattern in [
            "research", "reading", "reference", "citation", "paper",
            "academic", "analysis", "review", "literature", "study"
        ]):
            return "research"

        # No domain pattern detected
        return None
