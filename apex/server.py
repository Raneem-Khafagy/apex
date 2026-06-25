"""
APEX MCP Server — FastAPI publish-subscribe endpoint.

Extends MCP from request-response to publish-subscribe. Subscribing
applications register a ConsumerProfile and receive proactively pushed
context over WebSocket without ever issuing a retrieval query.

Endpoints
---------
POST   /subscribe          Register a subscriber profile → subscriber_id
GET    /context/{id}       Pull-mode fallback: return current buffer context
WS     /stream/{id}        Primary push channel (WebSocket)
DELETE /subscribe/{id}     Unregister subscriber, clean up buffer partition
GET    /                   Glass cockpit web dashboard
GET    /events             Server-Sent Events stream for dashboard updates
GET    /state              Snapshot of current pipeline state (JSON)

Module-level singletons
------------------------
_profiles  dict[subscriber_id → ConsumerProfile]
_buffer    ContextBuffer — per-subscriber isolated TTL cache
_adapter   LLMAdapter — Phi-3.5 Mini formatter
_connections dict[subscriber_id → set[WebSocket]] — active WS connections

Privacy rule: no document content is logged. Only subscriber_id and
metadata fields appear in log output. Context previews in the dashboard
are truncated to 120 characters.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from typing import Any, Callable

from fastapi import Depends, FastAPI, HTTPException, WebSocket, WebSocketDisconnect, status
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import BaseModel, field_validator

from apex.adapter.llm_adapter import ConsumerProfile, LLMAdapter
from apex.auth import (
    UserDB,
    create_token,
    get_current_user,
    get_user_db,
    _get_secret,
)
from apex.buffer.context_buffer import ContextBuffer
from apex.retrieval.rrf import Chunk

# ── Module-level singletons ───────────────────────────────────────────────────

app = FastAPI(title="APEX MCP Server", version="0.1.0")

_profiles: dict[str, ConsumerProfile] = {}
_buffer: ContextBuffer = ContextBuffer()
_adapter: LLMAdapter = LLMAdapter()
_connections: dict[str, set[WebSocket]] = {}
# Tracks which WebSocket connections requested ?format=json (opt-in by SPA)
_ws_wants_json: set[WebSocket] = set()

# Optional coordinator — set by init_app() when main.py assembles the pipeline.
# None in standalone server mode (just serve) or tests.
_coordinator = None  # type: ignore[assignment]

# ── Runtime vault config ──────────────────────────────────────────────────────
# Mutable at runtime via POST /config/vault + POST /config/reindex.
_vault_config: dict[str, Any] = {
    "vault_path":  os.path.expanduser("~/Documents/ApexVault"),
    "index_path":  "apex_vault",
    "doc_count":   0,
    "reindexing":  False,
}
_engine_ref = None   # RetrievalEngine | None
_monitor_ref = None  # SignalMonitor | None

# ── SSE broadcast infrastructure ──────────────────────────────────────────────
# Live state snapshot — updated by _broadcast_sse() from the coordinator.
# Sent to new SSE clients immediately on connect so the dashboard is warm.
_sse_state: dict[str, Any] = {
    "signal":   {},   # latest signal info (activity, velocity, label, confidence)
    "pipeline": {},   # latest scheduler decision (action, label, tau, reason)
    "buffer":   {},   # per-subscriber chunk counts
    "metrics":  {},   # PRP, LtC, DPS
    "context":  "",   # last pushed context preview (≤120 chars)
}
_sse_queues: set[asyncio.Queue] = set()


def _broadcast_sse(event_type: str, data: Any) -> None:
    """
    Update the live state snapshot and push an SSE event to all connected dashboard clients.

    Called by coordinator via the injected _sse_broadcast callable.
    Non-blocking: drops events if a client queue is full (slow consumer).
    """
    _sse_state[event_type] = data
    for q in list(_sse_queues):
        try:
            q.put_nowait({"type": event_type, "data": data})
        except asyncio.QueueFull:
            pass  # slow client — drop rather than block the pipeline


def init_app(
    buffer: ContextBuffer,
    adapter: LLMAdapter,
    coordinator=None,   # type: ignore[assignment]  PipelineCoordinator | None
    engine=None,        # type: ignore[assignment]  RetrievalEngine | None
    monitor=None,       # type: ignore[assignment]  SignalMonitor | None
    vault_path: str = "",
    index_path: str = "",
) -> None:
    """
    Inject shared pipeline instances into the MCP server.

    Call this BEFORE starting uvicorn so the server uses the same
    ContextBuffer and LLMAdapter as the running pipeline.
    """
    global _buffer, _adapter, _coordinator, _engine_ref, _monitor_ref
    _buffer = buffer
    _adapter = adapter
    _coordinator = coordinator
    _engine_ref = engine
    _monitor_ref = monitor
    if vault_path:
        _vault_config["vault_path"] = vault_path
    if index_path:
        _vault_config["index_path"] = index_path
    if coordinator is not None:
        coordinator._push_callback = _push_context
        coordinator._sse_broadcast = _broadcast_sse
    logger.info("MCP: server singletons injected from main.py")


# ── Request / response models ─────────────────────────────────────────────────

class SubscribeRequest(BaseModel):
    subscriber_id: str = ""
    autonomy_level: str = "assistive"
    goal_horizon: str = "short"
    interaction_style: str = "ambient"
    output_format: str = "plain-text"
    vocabulary_level: str = "technical"
    verbosity: str = "concise"
    citation_style: str = "none"
    max_context_tokens: int = 512
    domain_schema: Any = None


class SubscribeResponse(BaseModel):
    subscriber_id: str


class ContextResponse(BaseModel):
    subscriber_id: str
    context: str


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _format_context(subscriber_id: str) -> str:
    """
    Pull warm chunks from the buffer and format them for the subscriber.

    The LLM adapter call (phi3.5 via Ollama) is synchronous and can take
    5-15 seconds. Running it via asyncio.to_thread() keeps the event loop
    free to handle concurrent requests (e.g. vault_agent pull requests)
    while phi3.5 generates.
    """
    profile = _profiles.get(subscriber_id)
    if profile is None:
        return ""
    chunks = _buffer.get(subscriber_id)
    if not chunks:
        return ""
    return await asyncio.to_thread(_adapter.format, chunks, profile)


async def _format_context_with_ids(subscriber_id: str) -> tuple[str, list[str]]:
    """
    Like _format_context but also returns the chunk_ids for the SPA's ?format=json path.

    Returns (formatted_text, chunk_ids). chunk_ids preserves buffer order
    (highest-priority chunk first) so chunk_ids[0] is the primary card ID.
    """
    profile = _profiles.get(subscriber_id)
    if profile is None:
        return "", []
    chunks = _buffer.get(subscriber_id)
    if not chunks:
        return "", []
    chunk_ids = [c.chunk_id for c in chunks]
    text = await asyncio.to_thread(_adapter.format, chunks, profile)
    return text, chunk_ids


async def _push_context(subscriber_id: str) -> bool:
    """
    Format current buffer context and push it to all active WebSocket connections
    for this subscriber.

    Sockets are captured AFTER formatting completes — phi3.5 can take 5-85s,
    and WS connections that weren't established at the start of the call will
    be ready by the time formatting finishes.

    Returns True if at least one message was successfully delivered (subscriber
    was connected), False if no connections were active (unclaimed prefetch).
    """
    if subscriber_id not in _profiles:
        return False

    # Format first (slow — phi3.5 via asyncio.to_thread).
    # Do NOT check sockets before this: the WS connection race means sockets
    # may not be in _connections yet when the call starts.
    # Use _format_context_with_ids so we have chunk_ids for JSON-format sockets.
    formatted, chunk_ids = await _format_context_with_ids(subscriber_id)
    if not formatted:
        logger.debug("_push_context: empty formatted text for sub='{}'", subscriber_id)
        return False

    # Re-capture sockets AFTER format completes — connections established during
    # the phi3.5 wait are now visible.
    sockets = _connections.get(subscriber_id, set())
    logger.debug(
        "_push_context: sub='{}' active_sockets={} formatted_len={}",
        subscriber_id, len(sockets), len(formatted),
    )
    if not sockets:
        logger.warning(
            "_push_context: no WS connections for sub='{}' — push missed (buffer warm, unclaimed)",
            subscriber_id,
        )
        return False

    # Broadcast context preview to dashboard (privacy-safe: ≤120 chars)
    _broadcast_sse("context", formatted[:120])

    # Build JSON payload once (reused for all JSON-format sockets)
    json_payload = json.dumps({
        "chunk_id": chunk_ids[0] if chunk_ids else str(uuid.uuid4()),
        "text": formatted,
        "ts": time.time(),
    })

    dead: set[WebSocket] = set()
    delivered = False
    for ws in sockets:
        try:
            payload = json_payload if ws in _ws_wants_json else formatted
            await ws.send_text(payload)
            delivered = True
            logger.debug("_push_context: delivered to sub='{}' json={}", subscriber_id, ws in _ws_wants_json)
        except Exception as exc:
            logger.debug("_push_context: dead socket for sub='{}': {}", subscriber_id, exc)
            dead.add(ws)
    for ws in dead:
        _ws_wants_json.discard(ws)
        sockets.discard(ws)

    # DPS buffer dump — append delivery record for annotation harness
    dump_path = os.environ.get("APEX_BUFFER_DUMP_PATH", "")
    if dump_path and delivered:
        _append_buffer_dump(dump_path, subscriber_id, formatted, chunk_ids)

    return delivered


def _append_buffer_dump(
    dump_path: str,
    subscriber_id: str,
    formatted: str,
    chunk_ids: list[str],
) -> None:
    """Append one push record to the JSONL buffer dump file for DPS annotation."""
    import pathlib
    profile = _profiles.get(subscriber_id)
    record = {
        "subscriber_id":  subscriber_id,
        "ts":             time.time(),
        "formatted":      formatted,
        "chunk_ids":      chunk_ids,
        "signal_context": "",   # populated by coordinator in future; empty for now
        "profile": {
            "output_format":    getattr(profile, "output_format",    "plain-text") if profile else "plain-text",
            "verbosity":        getattr(profile, "verbosity",         "standard")  if profile else "standard",
            "vocabulary_level": getattr(profile, "vocabulary_level",  "technical") if profile else "technical",
        },
    }
    try:
        path = pathlib.Path(dump_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
    except OSError as exc:
        logger.debug("_append_buffer_dump: write failed: {}", exc)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/subscribe", response_model=SubscribeResponse)
async def subscribe(request: SubscribeRequest) -> SubscribeResponse:
    """
    Register a new subscriber.

    The server assigns a unique subscriber_id. The caller stores this ID
    and uses it for /context/{id}, /stream/{id}, and /unsubscribe/{id}.
    """
    subscriber_id = str(uuid.uuid4())

    profile = ConsumerProfile(
        subscriber_id=subscriber_id,
        autonomy_level=request.autonomy_level,
        goal_horizon=request.goal_horizon,
        interaction_style=request.interaction_style,
        output_format=request.output_format,
        vocabulary_level=request.vocabulary_level,
        verbosity=request.verbosity,
        citation_style=request.citation_style,
        max_context_tokens=request.max_context_tokens,
        domain_schema=request.domain_schema,
    )
    _profiles[subscriber_id] = profile
    _connections[subscriber_id] = set()

    if _coordinator is not None:
        _coordinator.add_subscriber(subscriber_id)

    logger.info("MCP: subscriber registered id='{}'", subscriber_id)
    return SubscribeResponse(subscriber_id=subscriber_id)


@app.get("/context/{subscriber_id}", response_model=ContextResponse)
async def get_context(subscriber_id: str) -> ContextResponse:
    """
    Pull-mode fallback: return the current formatted buffer context.

    When the buffer has warm content for this subscriber, the request is
    treated as a pull-mode supervision event (the user "would have searched
    here"). claim_via_pull() records this as a proactive claim with
    LtC = t_available - t_now (negative = APEX was ready before the need).

    Returns 404 if the subscriber is not registered.
    Returns an empty context string if the buffer has no warm chunks yet.
    """
    if subscriber_id not in _profiles:
        raise HTTPException(status_code=404, detail="Subscriber not found")

    # Pull-mode supervision: attempt to claim the most recent unclaimed proactive
    # prefetch event for this subscriber, regardless of whether the buffer is
    # currently warm. The claim window (60 s) is what defines "proactive" — if
    # APEX had content ready within the last minute, that counts as a proactive
    # hit even if the buffer TTL has since expired.
    #
    # This MUST happen before _format_context() so the claim timestamp (t_need)
    # is recorded at the moment the user would have searched, not after the
    # potentially slow LLM formatting call.
    t_need = time.time()
    if _coordinator is not None:
        _coordinator.store.claim_via_pull(
            session_id=_coordinator.session_id,
            subscriber_id=subscriber_id,
            t_need=t_need,
        )

    formatted = await _format_context(subscriber_id)
    return ContextResponse(subscriber_id=subscriber_id, context=formatted)


@app.websocket("/stream/{subscriber_id}")
async def stream(websocket: WebSocket, subscriber_id: str) -> None:
    """
    Primary push channel. The subscriber connects and waits for APEX to push
    formatted context. The subscriber may also send "pull" to trigger an
    immediate push of current buffer contents.

    Query params
    ------------
    format=json  Opt-in for the SPA. Pushes {chunk_id, text, ts} JSON payloads
                 instead of plain text. Legacy subscribers omit this param and
                 continue to receive plain text (unchanged behavior).

    Closes with code 4004 if subscriber_id is not registered.
    """
    if subscriber_id not in _profiles:
        await websocket.close(code=4004)
        return

    send_json = websocket.query_params.get("format", "text") == "json"

    await websocket.accept()
    _connections.setdefault(subscriber_id, set()).add(websocket)
    if send_json:
        _ws_wants_json.add(websocket)
    logger.info(
        "MCP: WebSocket opened for subscriber='{}' (total={} connections, json={})",
        subscriber_id, len(_connections[subscriber_id]), send_json,
    )

    try:
        while True:
            msg = await websocket.receive_text()
            if msg.strip().lower() in ("pull", "ping"):
                # Subscriber explicitly requests current context
                if send_json:
                    text, chunk_ids = await _format_context_with_ids(subscriber_id)
                    payload = json.dumps({
                        "chunk_id": chunk_ids[0] if chunk_ids else str(uuid.uuid4()),
                        "text": text,
                        "ts": time.time(),
                    }) if text else json.dumps({"chunk_id": "", "text": "", "ts": time.time()})
                else:
                    text = await _format_context(subscriber_id)
                    payload = text if text else ""
                await websocket.send_text(payload)
    except WebSocketDisconnect:
        logger.info("MCP: WebSocket closed for subscriber='{}'", subscriber_id)
    finally:
        sockets = _connections.get(subscriber_id, set())
        sockets.discard(websocket)
        _ws_wants_json.discard(websocket)


@app.delete("/subscribe/{subscriber_id}")
async def unsubscribe(subscriber_id: str) -> dict:
    """
    Unregister a subscriber. Cleans up profile, buffer partition, and
    any active WebSocket connections.
    """
    if subscriber_id not in _profiles:
        raise HTTPException(status_code=404, detail="Subscriber not found")

    del _profiles[subscriber_id]
    _buffer.clear_subscriber(subscriber_id)

    if _coordinator is not None:
        _coordinator.remove_subscriber(subscriber_id)

    # Close active WebSocket connections
    for ws in list(_connections.pop(subscriber_id, set())):
        try:
            await ws.close()
        except Exception:
            pass

    logger.info("MCP: subscriber unregistered id='{}'", subscriber_id)
    return {"unsubscribed": subscriber_id}


# ── Auth request / response models ───────────────────────────────────────────

class RegisterRequest(BaseModel):
    username: str
    password: str
    domain: str = "general"

    @field_validator("password")
    @classmethod
    def password_min_length(cls, v: str) -> str:
        if len(v) < 6:
            raise ValueError("Password must be at least 6 characters")
        return v

    @field_validator("domain")
    @classmethod
    def domain_normalise(cls, v: str) -> str:
        v = v.strip().lower()
        return v or "general"


class LoginRequest(BaseModel):
    username: str
    password: str


class AuthResponse(BaseModel):
    token: str
    user_id: str
    username: str
    domain: str
    onboarded: bool


class OnboardRequest(BaseModel):
    domain: str
    profile: dict


# ── Auth endpoints ────────────────────────────────────────────────────────────

@app.post("/auth/register", response_model=AuthResponse)
async def auth_register(request: RegisterRequest) -> AuthResponse:
    """
    Create a new user account.

    Returns a JWT token immediately so the client can proceed without
    a separate login step. Domain defaults to 'writing'; users can
    change it during onboarding.
    """
    db = get_user_db()
    try:
        user = db.create_user(request.username, request.password, request.domain)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT,
                            detail=f"Username '{request.username}' already exists")
    token = create_token(
        {"user_id": user["user_id"], "username": user["username"]},
        _get_secret(),
    )
    logger.info("Auth: registered username='{}'", request.username)
    return AuthResponse(
        token=token,
        user_id=user["user_id"],
        username=user["username"],
        domain=user["domain"],
        onboarded=bool(user.get("onboarded", 0)),
    )


@app.post("/auth/login", response_model=AuthResponse)
async def auth_login(request: LoginRequest) -> AuthResponse:
    """
    Authenticate with username + password. Returns a JWT token.
    """
    db = get_user_db()
    user = db.authenticate(request.username, request.password)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Invalid username or password")
    token = create_token(
        {"user_id": user["user_id"], "username": user["username"]},
        _get_secret(),
    )
    logger.info("Auth: login username='{}'", request.username)
    return AuthResponse(
        token=token,
        user_id=user["user_id"],
        username=user["username"],
        domain=user["domain"],
        onboarded=bool(user.get("onboarded", 0)),
    )


@app.get("/auth/me")
async def auth_me(
    current_user: dict = Depends(get_current_user),
) -> dict:
    """
    Return the authenticated user's profile and a live subscriber_id.

    If the daemon was restarted since the user last logged in, the stored
    subscriber_id will no longer be in _profiles. This endpoint transparently
    re-registers the user's ConsumerProfile and updates the DB so the client
    always receives a working subscriber_id.
    """
    import json as _json

    db = get_user_db()
    user_id = current_user["user_id"]
    subscriber_id = current_user.get("subscriber_id")

    # Re-hydrate: if subscriber_id is missing or stale, re-register
    if not subscriber_id or subscriber_id not in _profiles:
        profile_data = {}
        raw = current_user.get("profile_json", "{}")
        if raw:
            try:
                profile_data = _json.loads(raw)
            except Exception:
                profile_data = {}

        new_id = str(uuid.uuid4())
        _profiles[new_id] = ConsumerProfile(
            subscriber_id=new_id,
            autonomy_level=profile_data.get("autonomy_level", "assistive"),
            goal_horizon=profile_data.get("goal_horizon", "short"),
            interaction_style=profile_data.get("interaction_style", "ambient"),
            output_format=profile_data.get("output_format", "markdown"),
            vocabulary_level=profile_data.get("vocabulary_level", "domain-expert"),
            verbosity=profile_data.get("verbosity", "concise"),
            citation_style=profile_data.get("citation_style", "none"),
            max_context_tokens=int(profile_data.get("max_context_tokens", 512)),
            domain_schema=profile_data.get("domain_schema"),
        )
        _connections[new_id] = set()
        if _coordinator is not None:
            _coordinator.add_subscriber(new_id)
        db.update_subscriber_id(user_id, new_id)
        subscriber_id = new_id
        logger.info("Auth: re-hydrated subscriber_id='{}' for user='{}'",
                    subscriber_id, current_user["username"])

    return {
        "user_id": user_id,
        "username": current_user["username"],
        "domain": current_user["domain"],
        "subscriber_id": subscriber_id,
        "onboarded": bool(current_user.get("onboarded", 0)),
        "profile_json": current_user.get("profile_json", "{}"),
    }


@app.post("/auth/onboard")
async def auth_onboard(
    request: OnboardRequest,
    current_user: dict = Depends(get_current_user),
) -> dict:
    """
    Set the user's domain and ConsumerProfile on first login (or settings update).

    Replaces any existing subscriber registration with a fresh one using
    the new profile, then persists the choice to the DB.
    """
    import json as _json

    db = get_user_db()
    user_id = current_user["user_id"]

    # Tear down old subscriber if it exists
    old_sub = current_user.get("subscriber_id")
    if old_sub and old_sub in _profiles:
        del _profiles[old_sub]
        _buffer.clear_subscriber(old_sub)
        if _coordinator is not None:
            _coordinator.remove_subscriber(old_sub)
        _connections.pop(old_sub, None)

    # Register fresh subscriber with new profile
    new_id = str(uuid.uuid4())
    p = request.profile
    _profiles[new_id] = ConsumerProfile(
        subscriber_id=new_id,
        autonomy_level=p.get("autonomy_level", "assistive"),
        goal_horizon=p.get("goal_horizon", "short"),
        interaction_style=p.get("interaction_style", "ambient"),
        output_format=p.get("output_format", "markdown"),
        vocabulary_level=p.get("vocabulary_level", "domain-expert"),
        verbosity=p.get("verbosity", "concise"),
        citation_style=p.get("citation_style", "none"),
        max_context_tokens=int(p.get("max_context_tokens", 512)),
        domain_schema=p.get("domain_schema"),
    )
    _connections[new_id] = set()
    if _coordinator is not None:
        _coordinator.add_subscriber(new_id)

    # Persist to DB
    db.update_profile(user_id, request.domain, p)
    db.update_subscriber_id(user_id, new_id)

    logger.info("Auth: onboarded user='{}' domain='{}' sub='{}'",
                current_user["username"], request.domain, new_id)
    return {"subscriber_id": new_id, "domain": request.domain}


# ── Vault / knowledge-base config endpoints ──────────────────────────────────

class VaultConfigRequest(BaseModel):
    vault_path: str = ""
    fresh: bool = False


@app.get("/config/vault")
async def get_vault_config(
    current_user: dict = Depends(get_current_user),
) -> dict:
    """Return the current vault path, index path, and document count."""
    return {
        "vault_path": _vault_config["vault_path"],
        "index_path": _vault_config["index_path"],
        "doc_count":  _vault_config["doc_count"],
        "reindexing": _vault_config["reindexing"],
    }


@app.post("/config/vault")
async def set_vault_config(
    request: VaultConfigRequest,
    current_user: dict = Depends(get_current_user),
) -> dict:
    """
    Update the vault path (knowledge base folder).

    - fresh=True: reset to an empty temporary directory so APEX has no prior data.
    - vault_path: absolute path to an existing directory on this machine.

    Returns the new effective path. Re-indexing does NOT happen automatically —
    call POST /config/reindex after this to rebuild the index.
    """
    import tempfile

    if request.fresh:
        tmp = tempfile.mkdtemp(prefix="apex_vault_")
        _vault_config["vault_path"] = tmp
        _vault_config["doc_count"] = 0
        logger.info("config: vault reset to fresh tmp dir '{}'", tmp)
        return {"vault_path": tmp, "status": "ready — vault is empty, add documents then reindex"}

    path = request.vault_path.strip()
    if not path:
        raise HTTPException(status_code=400, detail="vault_path must not be empty")
    expanded = os.path.expanduser(path)
    if not os.path.isdir(expanded):
        raise HTTPException(
            status_code=400,
            detail=f"Path does not exist or is not a directory: {expanded}",
        )
    _vault_config["vault_path"] = expanded
    _vault_config["doc_count"] = 0
    # Update monitor watch path if available
    if _monitor_ref is not None:
        try:
            _monitor_ref.set_watch_path(expanded)
        except Exception:
            pass  # monitor may not support runtime path change
    logger.info("config: vault path updated to '{}'", expanded)
    return {"vault_path": expanded, "status": "path set — run reindex to rebuild the index"}


@app.post("/config/reindex")
async def trigger_reindex(
    current_user: dict = Depends(get_current_user),
) -> dict:
    """
    Rebuild the HNSW + BM25 index from the current vault path in the background.
    Poll GET /config/vault to watch doc_count and reindexing flag.
    """
    if _engine_ref is None:
        raise HTTPException(status_code=503, detail="Pipeline not running — start daemon first")
    if _vault_config["reindexing"]:
        return {"status": "already reindexing"}

    async def _do_reindex() -> None:
        from apex.ingest.ingestor import Ingestor
        _vault_config["reindexing"] = True
        try:
            path = _vault_config["vault_path"]
            idx  = _vault_config["index_path"]
            ingestor = Ingestor(_engine_ref)
            await asyncio.to_thread(ingestor.ingest, path)
            await asyncio.to_thread(ingestor.save_index, idx)
            _vault_config["doc_count"] = len(ingestor._metadata)
            logger.info("config: reindex complete — {} chunks", _vault_config["doc_count"])
        except Exception as exc:
            logger.error("config: reindex failed — {}", exc)
        finally:
            _vault_config["reindexing"] = False

    asyncio.create_task(_do_reindex())
    return {"status": "reindexing started", "vault_path": _vault_config["vault_path"]}


# ── Dashboard endpoints ───────────────────────────────────────────────────────

@app.get("/state")
async def get_state() -> dict:
    """Current pipeline state snapshot — for dashboard initial load."""
    return _sse_state


@app.get("/events")
async def sse_events() -> StreamingResponse:
    """
    Server-Sent Events stream for the glass cockpit dashboard.
    Sends the current state snapshot on connect, then streams updates.
    Keepalive comment sent every 30 s to prevent proxy timeouts.
    """
    q: asyncio.Queue = asyncio.Queue(maxsize=100)
    _sse_queues.add(q)

    async def generate():
        # Warm start: send current snapshot to newly connected dashboard
        for event_type, data in _sse_state.items():
            if data:
                yield f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
        # Stream live updates
        try:
            while True:
                try:
                    event = await asyncio.wait_for(q.get(), timeout=30.0)
                    yield f"event: {event['type']}\ndata: {json.dumps(event['data'])}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            _sse_queues.discard(q)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/", response_class=HTMLResponse)
async def dashboard() -> HTMLResponse:
    """Glass cockpit — real-time proactive AI layer visualization."""
    return HTMLResponse(content=_DASHBOARD_HTML)


# ── Dashboard HTML ────────────────────────────────────────────────────────────

_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>APEX — Proactive AI Layer</title>
<style>
  :root {
    --bg:       #0d1117;
    --surface:  #161b22;
    --border:   #30363d;
    --text:     #c9d1d9;
    --muted:    #8b949e;
    --green:    #3fb950;
    --blue:     #58a6ff;
    --yellow:   #d29922;
    --red:      #f85149;
    --orange:   #e3b341;
    --purple:   #bc8cff;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'SF Mono', 'Fira Code', 'Cascadia Code', monospace;
    font-size: 13px;
    height: 100vh;
    display: flex;
    flex-direction: column;
  }
  header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 20px;
    border-bottom: 1px solid var(--border);
    background: var(--surface);
  }
  header h1 { font-size: 15px; color: var(--blue); letter-spacing: 0.05em; }
  #status { display: flex; align-items: center; gap: 8px; color: var(--muted); }
  #status-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--red); transition: background 0.5s;
  }
  #status-dot.live { background: var(--green); animation: pulse 2s infinite; }
  @keyframes pulse {
    0%, 100% { opacity: 1; } 50% { opacity: 0.4; }
  }
  .grid {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    grid-template-rows: 1fr auto;
    gap: 1px;
    background: var(--border);
    flex: 1;
    overflow: hidden;
  }
  .panel {
    background: var(--bg);
    padding: 16px;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    gap: 10px;
  }
  .panel-title {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--muted);
    border-bottom: 1px solid var(--border);
    padding-bottom: 8px;
  }
  .context-panel {
    grid-column: 1 / -1;
    max-height: 140px;
  }
  .row { display: flex; justify-content: space-between; align-items: center; }
  .label { color: var(--muted); }
  .value { color: var(--text); font-weight: 600; }
  .value.green  { color: var(--green); }
  .value.blue   { color: var(--blue); }
  .value.yellow { color: var(--yellow); }
  .value.red    { color: var(--red); }
  .value.purple { color: var(--purple); }
  .badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.05em;
  }
  .badge-retrieve { background: #1c3a2a; color: var(--green); }
  .badge-wait     { background: #2a2315; color: var(--yellow); }
  .badge-idle     { background: #1f2428; color: var(--muted); }
  .vel-bar-wrap {
    background: var(--surface);
    border-radius: 3px;
    height: 6px;
    width: 100%;
    overflow: hidden;
  }
  .vel-bar {
    height: 100%;
    background: var(--blue);
    transition: width 0.4s ease;
    border-radius: 3px;
  }
  .context-text {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 10px;
    color: var(--text);
    font-size: 12px;
    line-height: 1.5;
    word-break: break-word;
    flex: 1;
    overflow-y: auto;
    white-space: pre-wrap;
  }
  .metric-big {
    font-size: 26px;
    font-weight: 700;
    line-height: 1;
  }
  .metric-label { font-size: 10px; color: var(--muted); margin-top: 2px; }
  .metrics-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
  }
  .metric-item { display: flex; flex-direction: column; }
  .ts { font-size: 10px; color: var(--muted); margin-top: auto; }
</style>
</head>
<body>

<header>
  <h1>⬡ APEX &mdash; Proactive AI Layer</h1>
  <div id="status">
    <div id="status-dot"></div>
    <span id="status-text">Connecting…</span>
  </div>
</header>

<div class="grid">

  <!-- Sensing panel -->
  <div class="panel">
    <div class="panel-title">Sensing</div>
    <div class="row">
      <span class="label">Activity</span>
      <span class="value blue" id="s-activity">—</span>
    </div>
    <div class="row">
      <span class="label">Velocity</span>
      <span class="value" id="s-vel-num">0.00</span>
    </div>
    <div class="vel-bar-wrap"><div class="vel-bar" id="s-vel-bar" style="width:0%"></div></div>
    <div class="row">
      <span class="label">Inferred label</span>
      <span class="value purple" id="s-label">—</span>
    </div>
    <div class="row">
      <span class="label">Confidence</span>
      <span class="value" id="s-conf">—</span>
    </div>
    <div class="row">
      <span class="label">Urgency</span>
      <span class="value" id="s-urgency">—</span>
    </div>
    <div class="ts" id="s-ts"></div>
  </div>

  <!-- Pipeline panel -->
  <div class="panel">
    <div class="panel-title">Pipeline Decision</div>
    <div class="row">
      <span class="label">Action</span>
      <span id="p-action"><span class="badge badge-idle">—</span></span>
    </div>
    <div class="row">
      <span class="label">Label</span>
      <span class="value purple" id="p-label">—</span>
    </div>
    <div class="row">
      <span class="label">τ (threshold)</span>
      <span class="value" id="p-tau">—</span>
    </div>
    <div class="row">
      <span class="label">Reason</span>
      <span></span>
    </div>
    <div style="color:var(--muted);font-size:11px;line-height:1.4" id="p-reason">—</div>
    <div class="ts" id="p-ts"></div>
  </div>

  <!-- Metrics panel -->
  <div class="panel">
    <div class="panel-title">Thesis Metrics</div>
    <div class="metrics-grid">
      <div class="metric-item">
        <div class="metric-big" id="m-prp">—</div>
        <div class="metric-label">PRP &gt; 0.65</div>
      </div>
      <div class="metric-item">
        <div class="metric-big" id="m-ltc">—</div>
        <div class="metric-label">LtC (ms) &lt; 0</div>
      </div>
      <div class="metric-item">
        <div class="metric-big" id="m-dps">—</div>
        <div class="metric-label">DPS &gt; 0.75</div>
      </div>
      <div class="metric-item">
        <div class="metric-big" id="m-buf" style="color:var(--blue)">0</div>
        <div class="metric-label">Buffer chunks</div>
      </div>
    </div>
  </div>

  <!-- Context preview -->
  <div class="panel context-panel">
    <div class="panel-title">Last Pushed Context <span style="font-size:9px;color:var(--muted)">(≤120 chars — privacy truncated)</span></div>
    <div class="context-text" id="ctx-text">Waiting for first retrieval…</div>
  </div>

</div>

<script>
const $ = id => document.getElementById(id);
const fmt = (v, d=2) => v == null ? '—' : Number(v).toFixed(d);
const now  = () => new Date().toLocaleTimeString();

function setLive(ok) {
  $('status-dot').className = ok ? 'live' : '';
  $('status-text').textContent = ok ? 'Live' : 'Disconnected — retrying…';
}

function applySignal(d) {
  $('s-activity').textContent = d.activity_type || '—';
  $('s-label').textContent    = d.label || '—';
  $('s-conf').textContent     = d.confidence != null ? fmt(d.confidence) : '—';
  const vel = d.velocity || 0;
  $('s-vel-num').textContent  = fmt(vel);
  $('s-vel-bar').style.width  = (vel * 100).toFixed(1) + '%';
  $('s-urgency').textContent  = d.urgency ? '🔴 YES' : 'no';
  $('s-urgency').className    = 'value ' + (d.urgency ? 'red' : 'green');
  $('s-ts').textContent       = 'updated ' + now();
}

function applyPipeline(d) {
  const a = (d.action || '').toUpperCase();
  const badge = a === 'RETRIEVE'
    ? '<span class="badge badge-retrieve">RETRIEVE</span>'
    : a === 'WAIT'
    ? '<span class="badge badge-wait">WAIT</span>'
    : '<span class="badge badge-idle">—</span>';
  $('p-action').innerHTML  = badge;
  $('p-label').textContent = d.label || '—';
  $('p-tau').textContent   = d.tau != null ? fmt(d.tau) : '—';
  $('p-reason').textContent = d.reason || '—';
  $('p-ts').textContent    = 'updated ' + now();
}

function applyMetrics(d) {
  const prp = d.prp;
  $('m-prp').textContent  = prp != null ? fmt(prp) : '—';
  $('m-prp').className    = 'metric-big ' + (prp == null ? '' : prp >= 0.65 ? 'green' : 'red');
  const ltc = d.ltc;
  $('m-ltc').textContent  = ltc != null ? Math.round(ltc).toLocaleString() : '—';
  $('m-ltc').className    = 'metric-big ' + (ltc == null ? '' : ltc < 0 ? 'green' : 'red');
  const dps = d.dps;
  $('m-dps').textContent  = dps != null ? fmt(dps) : '—';
  $('m-dps').className    = 'metric-big ' + (dps == null ? '' : dps >= 0.75 ? 'green' : 'yellow');
}

function applyBuffer(d) {
  const total = Object.values(d).reduce((s, v) => s + (v || 0), 0);
  $('m-buf').textContent = total;
}

function applyContext(text) {
  if (text) $('ctx-text').textContent = text;
}

const handlers = {
  signal:   applySignal,
  pipeline: applyPipeline,
  metrics:  applyMetrics,
  buffer:   applyBuffer,
  context:  applyContext,
};

function connect() {
  const src = new EventSource('/events');

  src.onopen = () => setLive(true);
  src.onerror = () => {
    setLive(false);
    src.close();
    setTimeout(connect, 3000);
  };

  Object.entries(handlers).forEach(([type, fn]) => {
    src.addEventListener(type, e => {
      try { fn(JSON.parse(e.data)); } catch {}
    });
  });
}

connect();
</script>
</body>
</html>
"""

# ── SPA static file serving ───────────────────────────────────────────────────
# Serve the React SPA at /app. The build output (vite build → apex/static/app/)
# must exist before this is useful; a .gitkeep placeholder keeps the directory
# in version control until the first build.
_spa_dir = os.path.join(os.path.dirname(__file__), "static", "app")
if os.path.isdir(_spa_dir) and any(
    f != ".gitkeep" for f in os.listdir(_spa_dir) if not f.startswith(".")
):
    app.mount("/app", StaticFiles(directory=_spa_dir, html=True), name="app")
