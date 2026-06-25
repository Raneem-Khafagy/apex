# ADR-001: Replace Mocked Ollama With Real Local Models in Tests

**Status:** Accepted
**Date:** 2026-04-07
**Deciders:** Raneem Khafagy (thesis author)

---

## Context

APEX currently has three test files. Two of them (`test_adapters.py`, `test_monitor.py`)
test pure logic and OS wiring with no external service dependencies. The third
(`test_intent_engine.py`) mocks all Ollama calls — both `ollama.embed()` and
`ollama.chat()` — using `unittest.mock.patch` and `MagicMock`.

The motivation for those mocks was stated in the file header:
> "Ollama is mocked in all tests — no live LLM required."

This approach has a fundamental flaw: the mocks test mock behavior, not production
behavior. The critical architectural invariant of APEX — that `q̂` is always a real
dense vector produced by a real embedding model, and never a text string — can only
be verified by calling the real model.

### What the Mocks Currently Hide

| Mocked call | What it hides |
|---|---|
| `ollama.embed()` returns a hardcoded vector | Whether `all-minilm` actually produces a shape-(384,) float32 unit vector |
| `ollama.chat()` returns `"reading_reference"` | Whether `phi3.5` actually classifies behavioral signals into valid snake_case labels |
| Patching the whole `ollama` module at init | Whether `_build_vector_table()` actually builds a valid vector table at startup |

The mocks in `TestIntentEngine` are all Ollama mocks. A passing mock-based test tells
us the mock works. It tells us nothing about whether the real Ollama pipeline works.

### APEX's On-Device Architecture Enables Real Testing

APEX's core design principle — **everything runs on-device, no cloud** — means the
"external dependency" that usually justifies mocks (an external API, a remote
database) does not exist here. Ollama runs at `localhost:11434`. It is a local process.
Running it in tests is no different from running SQLite or `watchfiles` in tests — and
those are never mocked.

This makes APEX unusually well-suited for the no-mock principle: the right choice is
also the architecturally consistent choice.

---

## Decision

**All Ollama calls in `test_intent_engine.py` will be replaced with real calls to a
locally-running Ollama instance.**

The test suite is restructured into two layers with pytest marks:

```
pytest.mark.unit        — pure logic, no Ollama, always fast, always runnable
pytest.mark.integration — real Ollama required, marked for CI gating
```

---

## Mock Audit: What Changes and What Stays

This is the full inventory of every mock-like pattern in the test suite and the
decision for each.

### ❌ REMOVE — Ollama mocks in `test_intent_engine.py`

```python
# BEFORE (wrong)
with patch("apex.inference.intent_engine.ollama") as mock_ollama:
    mock_ollama.embed = MagicMock(return_value=mock_embed_response)
    mock_ollama.chat = MagicMock(return_value=...)
```

These are removed entirely. The `TestIntentEngine` class is replaced with
`TestIntentEngineIntegration` that calls real Ollama.

**Why:** These mocks test that `MagicMock` returns what you told it to return. They
cannot detect a broken embedding pipeline, a changed Ollama API, or a model that
starts emitting the wrong output shape.

---

### ✅ KEEP — Constructor injection in `TestProductivityAdapter`

```python
# KEEP (correct)
adapter = ProductivityAdapter(
    watch_path="/tmp",
    app_detector=lambda: app_name,    # <— injected callable, not a mock
)
```

This is **dependency injection**, not mocking. `app_detector` is a constructor
parameter defined as part of the `ProductivityAdapter` API for exactly this purpose —
to allow tests to control which app name is returned without OS interaction. There is
no `patch()` call. The real `ProductivityAdapter` code runs; only the OS-detection
leaf is controlled by a pure Python lambda.

This is analogous to injecting a clock into a scheduler so you can test timeout
behavior without sleeping. It is valid, it is clean, and it stays.

---

### ✅ KEEP — Privacy sentinel in `TestProductivityAdapter`

```python
# KEEP (correct)
with patch("builtins.open", side_effect=AssertionError("file content read!")):
    sv = adapter.observe()  # must not raise
```

This is not a mock that substitutes a dependency to make a test pass. It is a
**guard rail that makes the test fail if `observe()` ever reads a file**. It is
testing a negative: *this code path must never call `open()`*. Patching `open` is
the correct mechanism for that invariant. No alternative achieves the same guarantee.

**Architecture rule preserved:** The Behavioral Signal Monitor must never read file
content — only structural metadata.

---

### ✅ KEEP — `MagicMock` adapter in `test_monitor.py`

```python
# KEEP (correct)
adapter = MagicMock()
adapter.observe.return_value = _make_signal()
monitor = SignalMonitor(adapter=adapter, watch_path=tmpdir)
```

`SignalMonitor` takes a `SignalAdapter` via constructor injection. The monitor's
job is to call `adapter.observe()` when `watchfiles` detects a file change, then
fire registered callbacks. The tests verify this wiring: "does the callback receive
the signal that the adapter produced?"

The `MagicMock` here is a **controlled stub at a true interface boundary** (the
`SignalAdapter` ABC). The monitor tests are testing the monitor, not the adapter.
The adapter itself has its own test class (`TestProductivityAdapter`). These are
separate units with separate responsibilities; one does not need to run inside
the other's test.

This is the correct unit-of-test isolation. It stays.

---

### ✅ NO MOCKS NEEDED — `TestHeuristicGate`

These tests already have zero mocks. `HeuristicGate` is pure Python — a dict lookup
over numpy arrays. The `vector_table` is built from `np.random.default_rng` in
tests, which is not a mock; it is a synthetic but real numpy ndarray.

These tests are already correct and require no changes.

---

## Options Considered

### Option A: Keep all Ollama mocks (status quo)
**Pros:** Fast test suite, no Ollama dependency for CI
**Cons:** Tests prove nothing about the actual pipeline. The critical invariant
(`q̂` is real ndarray) is tested against a fake. Any regression in the Ollama
integration is invisible.
**Decision: Rejected.**

### Option B: Remove all mocks everywhere, including monitor adapter
**Pros:** Maximally strict.
**Cons:** Forces `SignalMonitor` tests to run `ProductivityAdapter.observe()` with
real OS APIs (NSWorkspace) — this makes monitor tests platform-specific and slow.
The monitor's behavior doesn't change based on what the adapter returns; the mock
correctly isolates the unit under test.
**Decision: Rejected.**

### Option C (Chosen): Remove Ollama mocks, keep interface-boundary stubs
Replace Ollama mocks with real calls. Keep constructor-injected test doubles and
privacy sentinels. Use `pytest.mark.integration` to gate Ollama-dependent tests in CI.

**Pros:**
- Real behavior tested where it matters (IIE pipeline)
- Interface-boundary isolation preserved where correct
- Privacy invariants still enforced
- CI can run `pytest -m unit` without Ollama, `pytest -m integration` when Ollama is available
- Architecturally consistent with APEX's on-device-only design

**Decision: Accepted.**

---

## Security Analysis

Running Ollama locally in tests raises no security concerns for APEX:

| Concern | Reality |
|---|---|
| Data egress | Ollama runs at `localhost:11434`. No network calls leave the machine. |
| PII in test signals | Test signals are synthetic: `{"activity_type": "writing", "velocity_metric": 0.8}`. No real user data. |
| Model access | Models are pulled once (`ollama pull all-minilm`, `ollama pull phi3.5`) and cached locally. |
| Alignment with APEX principle | "No data leaves the device" applies to tests too. Real Ollama in tests is the correct implementation of this principle. |

---

## Implementation

### 1. `tests/conftest.py` (new file)

```python
"""
Pytest configuration and shared fixtures for APEX tests.

Integration tests (pytest.mark.integration) require:
  - Ollama running: ollama serve
  - Models pulled:  ollama pull all-minilm && ollama pull phi3.5
"""
import pytest
import ollama as _ollama


def _ollama_running() -> bool:
    try:
        _ollama.list()
        return True
    except Exception:
        return False


def _model_available(model_prefix: str) -> bool:
    try:
        models = _ollama.list()
        return any(m.model.startswith(model_prefix) for m in models.models)
    except Exception:
        return False


# ── Marks ────────────────────────────────────────────────────────────────────

requires_ollama = pytest.mark.skipif(
    not _ollama_running(),
    reason=(
        "Ollama not running. Start with: ollama serve\n"
        "Then pull models: ollama pull all-minilm && ollama pull phi3.5"
    ),
)

requires_embed_model = pytest.mark.skipif(
    not _model_available("all-minilm"),
    reason="all-minilm not available. Run: ollama pull all-minilm",
)

requires_chat_model = pytest.mark.skipif(
    not _model_available("phi3.5"),
    reason="phi3.5 not available. Run: ollama pull phi3.5",
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def real_intent_engine():
    """
    Build a real IntentEngine against live Ollama all-minilm.
    Session-scoped: vector table is built once per test session (expensive).
    """
    from apex.inference.intent_engine import IntentEngine
    return IntentEngine()  # calls real ollama.embed() at init
```

### 2. `tests/test_intent_engine.py` — rewritten integration section

```python
# ── IntentEngine (real Ollama) ────────────────────────────────────────────────

@pytest.mark.integration
@requires_ollama
@requires_embed_model
class TestIntentEngineIntegration:
    """
    IntentEngine tests against real Ollama.
    These tests verify the actual pipeline behavior, not mock behavior.

    Run with:  pytest -m integration
    Skip with: pytest -m "not integration"  (CI without Ollama)
    """

    async def test_output_is_triple(self, real_intent_engine):
        result = await real_intent_engine.infer(_signal("writing", velocity=0.8))
        assert len(result) == 3

    async def test_q_hat_is_real_ndarray_not_str(self, real_intent_engine):
        """THE critical invariant against the real model."""
        q_hat, c, label = await real_intent_engine.infer(_signal("writing", velocity=0.8))
        assert isinstance(q_hat, np.ndarray), (
            f"q̂ must be np.ndarray from real all-minilm — got {type(q_hat).__name__}. "
            "The IIE must never emit a text string as a retrieval query."
        )
        assert q_hat.shape == (EMBED_DIM,)
        assert q_hat.dtype == np.float32

    async def test_heuristic_path_confirmed_by_confidence(self, real_intent_engine):
        """
        Known signal → heuristic gate hits → confidence >= 0.85.
        LLM path would return LLM_CONFIDENCE = 0.65.
        We verify the heuristic ran by observing the confidence level.
        No mocking needed: the confidence values are architecturally distinct.
        """
        _, c, _ = await real_intent_engine.infer(_signal("writing", velocity=0.9))
        assert c >= 0.85, (
            f"Expected heuristic path (c >= 0.85), got c={c}. "
            "If c ≈ 0.65, the LLM path ran instead — check HeuristicGate pattern table."
        )

    async def test_llm_path_for_ambiguous_signal(self, real_intent_engine):
        """
        Exotic unknown activity → gate misses → real phi3.5 classifies it.
        We verify: output is valid triple, q̂ is ndarray, label is snake_case string.
        """
        q_hat, c, label = await real_intent_engine.infer(
            _signal("unknown_exotic_activity_xyz", velocity=0.4)
        )
        assert isinstance(q_hat, np.ndarray)
        assert q_hat.shape == (EMBED_DIM,)
        assert 0.0 <= c <= 1.0
        assert isinstance(label, str)
        assert len(label) > 0
        assert " " not in label, f"Label must be snake_case, got: '{label}'"

    async def test_urgency_flag_forces_confidence_to_one(self, real_intent_engine):
        """urgency_flag=True must yield c=1.0 regardless of LLM output."""
        _, c, _ = await real_intent_engine.infer(
            _signal("anomaly_event", velocity=1.0, urgency=True)
        )
        assert c >= 0.95, f"urgency_flag=True must yield c>=0.95, got {c}"

    async def test_embedding_is_unit_normalized(self, real_intent_engine):
        """all-minilm vectors must be L2-normalized (cosine sim = dot product)."""
        q_hat, _, _ = await real_intent_engine.infer(_signal("writing", velocity=0.8))
        norm = float(np.linalg.norm(q_hat))
        assert abs(norm - 1.0) < 1e-5, f"q̂ must be unit vector, got norm={norm:.6f}"
```

### 3. `pyproject.toml` addition

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
markers = [
    "unit: pure logic tests, no external services required",
    "integration: requires Ollama running locally (ollama serve)",
]
```

### 4. `justfile` additions

```makefile
# Run unit tests only (no Ollama needed)
test-unit:
    uv run pytest tests/ -m "not integration" -v

# Run full suite including Ollama integration tests
test-integration:
    uv run pytest tests/ -m integration -v

# Run everything
test:
    uv run pytest tests/ -v
```

---

## Consequences

### What becomes easier
- Regressions in the Ollama integration surface immediately in CI (integration tier)
- The critical invariant `q̂ is ndarray` is verified against the real model
- Test failures mean something real broke, not that a mock was set up wrong
- Architecture stays consistent: on-device testing mirrors on-device deployment

### What becomes harder
- Developers must have Ollama running and models pulled to run integration tests
- Integration tests are slower (~100–500ms per test vs <1ms for mock tests)
- CI requires an Ollama-enabled runner for the integration tier

### What we'll need to revisit
- If `phi3.5` is replaced with another model (per CLAUDE.md trigger), integration
  tests must be rerun to verify the new model still classifies signals correctly
- The `requires_chat_model` fixture must be updated to the new model name
- CLAUDE.md must be updated (model change trigger already defined there)

---

## Updated Testing Rule for CLAUDE.md

```
# OLD (removed)
Tests hit real FastAPI, no mocks for the server. Ollama must be running for
integration tests. Unit tests for individual components can mock Ollama responses.

# NEW
Tests must use real dependencies, not mocks. No unit.mock.patch on Ollama.

Three categories are permitted:
1. Pure logic (no external services) — test pure Python/numpy. No marks needed.
2. Integration (Ollama required) — mark with @pytest.mark.integration.
   Requires: ollama serve + ollama pull all-minilm + ollama pull phi3.5
3. Interface boundary stubs — constructor-injected callables at SignalAdapter
   boundaries are permitted (not mocks; they are the defined API seam).

Privacy sentinels — patch("builtins.open", side_effect=AssertionError(...))
is permitted to verify that observe() never reads file content.
```
