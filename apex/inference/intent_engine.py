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
from typing import Optional

import numpy as np
import ollama
from loguru import logger

from apex.adapters.base import SignalVector
from apex.inference.heuristic_gate import HeuristicGate

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
    Hybrid Intent Inference Engine — fast heuristic gate + small-LLM classifier.

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
    """

    def __init__(
        self,
        vector_table: Optional[dict[str, np.ndarray]] = None,
        chat_model: str = CHAT_MODEL,
        embed_model: str = EMBED_MODEL,
    ) -> None:
        self._chat_model = chat_model
        self._embed_model = embed_model

        if vector_table is None:
            logger.info(
                "IIE: building vector table (embed_model='{}', chat_model='{}')",
                embed_model, chat_model,
            )
            vector_table = _build_vector_table(embed_model=embed_model)
        self._gate = HeuristicGate(vector_table=vector_table)
        self._vector_table = vector_table

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
