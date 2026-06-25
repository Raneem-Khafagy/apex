"""
LLM Adapter Layer — semantic translator for subscriber-specific formatting.

The LLM Adapter is NOT a generator. It takes pre-retrieved Chunk objects and
reformats them into the shape each subscriber needs. It adds no new facts.
If chunks are empty, it returns an empty string immediately — no Ollama call.

Model: Phi-3.5 Mini (3.8B, INT4) via Ollama.
Backend: ollama.generate() with a profile-derived system prompt.

Architecture rule enforced here
---------------------------------
The adapter is a translator, not a generator. The system prompt instructs
Phi-3.5 to reformat ONLY the provided content and produce NOTHING that is
not present in the input chunks.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

import ollama
from loguru import logger

from apex.retrieval.rrf import Chunk

MODEL = "phi3.5"

# Approximate chars-per-token ratio for rough budget enforcement before
# sending to Ollama. Conservative estimate (1 token ≈ 4 chars in English).
_CHARS_PER_TOKEN = 4

# Reserve tokens for the system prompt and response headroom
_SYSTEM_PROMPT_TOKEN_BUDGET = 200


# ── ConsumerProfile ──────────────────────────────────────────────────────────

@dataclass
class ConsumerProfile:
    """
    Registered profile for a subscribing application.

    Persisted at subscription time and used by the LLM Adapter to determine
    how to format the same retrieved chunks differently for each consumer.

    Fields
    ------
    subscriber_id
        Unique identifier for this subscriber. Matches the ContextBuffer key.
    autonomy_level
        How much initiative APEX takes:
        "suggestive" | "assistive" | "autonomous"
    goal_horizon
        Time horizon of user goals: "short" | "mid" | "long"
    interaction_style
        How the consumer surfaces context:
        "ambient" | "soft-interrupt" | "hard-interrupt" | "conversational"
    output_format
        Wire format: "json" | "markdown" | "plain-text" | "voice" | "structured-alert"
    vocabulary_level
        Terminology complexity: "technical" | "domain-expert" | "layman"
    verbosity
        Response length: "concise" | "standard" | "detailed"
    citation_style
        How sources are cited: "inline" | "footnote" | "none"
    max_context_tokens
        Hard token budget for the formatted output. Chunk text is truncated
        before the Ollama call to stay within this budget.
    domain_schema
        Optional JSON Schema dict. When present, the adapter formats output
        as a JSON object conforming to this schema.
    """
    subscriber_id: str
    autonomy_level: str
    goal_horizon: str
    interaction_style: str
    output_format: str
    vocabulary_level: str
    verbosity: str
    citation_style: str
    max_context_tokens: int
    domain_schema: Optional[dict] = field(default=None)


# ── Prompt builders ──────────────────────────────────────────────────────────

def _build_system_prompt(profile: ConsumerProfile) -> str:
    """Construct the system prompt from the subscriber's profile."""
    format_instruction = {
        "json":             "Respond with valid JSON only. No prose outside JSON.",
        "markdown":         "Respond in Markdown with headers and bullet points.",
        "plain-text":       "Respond in plain text. No markup.",
        "structured-alert": "Respond as a terse structured alert. Lead with severity.",
        "voice":            "Respond in natural spoken language, short sentences.",
    }.get(profile.output_format, "Respond in plain text.")

    verbosity_instruction = {
        "concise":  "Be brief. One to three sentences maximum.",
        "standard": "Provide a clear, moderate-length response.",
        "detailed": "Provide a thorough explanation with relevant details.",
    }.get(profile.verbosity, "")

    vocabulary_instruction = {
        "technical":      "Use precise technical terminology appropriate for an expert.",
        "domain-expert":  "Use domain-specific language; assume professional background.",
        "layman":         "Use simple language. Avoid jargon. Explain terms.",
    }.get(profile.vocabulary_level, "")

    citation_instruction = {
        "inline":   "Cite sources inline using [source_name] notation.",
        "footnote": "Add numbered footnotes for sources at the end.",
        "none":     "Do not include citations or source references.",
    }.get(profile.citation_style, "")

    schema_instruction = ""
    if profile.domain_schema:
        schema_instruction = (
            f"\nYour response MUST conform to this JSON Schema:\n"
            f"{json.dumps(profile.domain_schema, indent=2)}"
        )

    return (
        "You are a context formatting assistant for a proactive edge AI system. "
        "You will be given retrieved knowledge chunks. "
        "Your ONLY job is to reformat and summarise that content for the user. "
        "Do NOT add facts, opinions, or information not present in the provided chunks. "
        "If the chunks are empty or irrelevant, respond with an empty string.\n\n"
        f"Format: {format_instruction}\n"
        f"Verbosity: {verbosity_instruction}\n"
        f"Vocabulary: {vocabulary_instruction}\n"
        f"Citations: {citation_instruction}"
        f"{schema_instruction}"
    )


def _build_user_message(chunks: list[Chunk], char_budget: int) -> str:
    """
    Serialise chunks into a user message, respecting the character budget.

    Chunks are included in RRF score order (highest first). Each chunk's
    text is truncated independently to stay within the per-chunk budget,
    and the loop stops once the total budget is consumed.
    """
    parts: list[str] = []
    remaining = char_budget

    for chunk in chunks:
        if remaining <= 0:
            break
        header = f"[{chunk.chunk_id} | {chunk.label} | src:{chunk.source}]\n"
        available = remaining - len(header)
        if available <= 0:
            break
        text = chunk.text[:available]
        parts.append(header + text)
        remaining -= len(header) + len(text)

    return "\n\n".join(parts)


# ── Post-processing ──────────────────────────────────────────────────────────

def _strip_code_fences(text: str, output_format: str) -> str:
    """
    Remove markdown code fences that LLMs often wrap around structured output.
    E.g. ```json ... ``` → the inner JSON string.
    Only applied when output_format requires clean structured text (json).
    """
    import re
    if output_format == "json":
        # Match ```json ... ``` or ``` ... ``` blocks
        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if match:
            return match.group(1).strip()
    return text


# ── LLMAdapter ───────────────────────────────────────────────────────────────

class LLMAdapter:
    """
    Semantic translator: same chunks → different formatted output per subscriber.

    Parameters
    ----------
    model
        Ollama model name. Defaults to "phi3.5" (Phi-3.5 Mini, INT4).
    """

    def __init__(self, model: str = MODEL) -> None:
        self._model = model

    def format(self, chunks: list[Chunk], profile: ConsumerProfile) -> str:
        """
        Reformat retrieved chunks according to the subscriber's profile.

        Parameters
        ----------
        chunks
            Retrieved Chunk objects from ContextBuffer.get(). If empty,
            returns "" immediately without calling Ollama.
        profile
            The subscriber's ConsumerProfile — controls output shape.

        Returns
        -------
        Formatted string ready to push to the subscriber.
        Empty string if chunks is empty (no hallucination).
        """
        # ── Translator invariant: empty input → empty output ─────────────────
        if not chunks:
            logger.debug("LLMAdapter: no chunks — returning empty (no Ollama call)")
            return ""

        system_prompt = _build_system_prompt(profile)

        # ── Token budget: compute char budget for chunk content ───────────────
        # Reserve tokens for system prompt and response; remainder goes to chunks
        chunk_token_budget = max(
            64,
            profile.max_context_tokens - _SYSTEM_PROMPT_TOKEN_BUDGET,
        )
        char_budget = chunk_token_budget * _CHARS_PER_TOKEN

        user_message = _build_user_message(chunks, char_budget)

        logger.debug(
            "LLMAdapter: formatting {} chunk(s) for subscriber='{}' format='{}'",
            len(chunks), profile.subscriber_id, profile.output_format,
        )

        try:
            response = ollama.chat(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_message},
                ],
            )
            result = _strip_code_fences(response.message.content.strip(),
                                        profile.output_format)
        except Exception as exc:
            logger.error("LLMAdapter: Ollama call failed: {}", exc)
            # Return raw chunk text as fallback — better than empty
            result = "\n\n".join(c.text for c in chunks)[:char_budget]

        # ── JSON validation: ensure parseable output when format="json" ───────
        if profile.output_format == "json":
            import json as _json
            try:
                _json.loads(result)
            except (_json.JSONDecodeError, ValueError):
                logger.warning(
                    "LLMAdapter: JSON output was malformed — wrapping as fallback"
                )
                result = _json.dumps({"context": result})

        logger.debug(
            "LLMAdapter: formatted {} chars for subscriber='{}'",
            len(result), profile.subscriber_id,
        )
        return result
