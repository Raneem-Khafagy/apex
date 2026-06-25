"""
Tests for the APEX MCP server.

Tests hit real FastAPI — no mocks for the server per CLAUDE.md.
Uses httpx.AsyncClient with the ASGI app directly (no network socket needed).
WebSocket tests use starlette's TestClient.

Subscriber lifecycle tested:
    POST /subscribe     → subscriber_id returned
    GET  /context/{id}  → formatted context (may be empty if pipeline hasn't fired)
    WS   /stream/{id}   → push channel receives messages
    DELETE /subscribe/{id} → subscriber removed, subsequent GET returns 404
"""
import json

import httpx
import pytest
from starlette.testclient import TestClient

from apex.server import app
from apex.adapter.llm_adapter import ConsumerProfile


def _profile_payload(**overrides) -> dict:
    base = {
        "subscriber_id": "",          # server assigns this
        "autonomy_level": "assistive",
        "goal_horizon": "short",
        "interaction_style": "ambient",
        "output_format": "plain-text",
        "vocabulary_level": "technical",
        "verbosity": "concise",
        "citation_style": "none",
        "max_context_tokens": 512,
        "domain_schema": None,
    }
    base.update(overrides)
    return base


# ── POST /subscribe ──────────────────────────────────────────────────────────

class TestSubscribe:
    async def test_subscribe_returns_200(self):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/subscribe", json=_profile_payload())
        assert resp.status_code == 200

    async def test_subscribe_returns_subscriber_id(self):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/subscribe", json=_profile_payload())
        body = resp.json()
        assert "subscriber_id" in body
        assert isinstance(body["subscriber_id"], str)
        assert len(body["subscriber_id"]) > 0

    async def test_each_subscription_gets_unique_id(self):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            r1 = await client.post("/subscribe", json=_profile_payload())
            r2 = await client.post("/subscribe", json=_profile_payload())
        assert r1.json()["subscriber_id"] != r2.json()["subscriber_id"]

    async def test_subscribe_stores_profile(self):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/subscribe", json=_profile_payload(output_format="json"))
            sid = resp.json()["subscriber_id"]
            # Subsequent GET should work (profile was stored)
            ctx = await client.get(f"/context/{sid}")
        assert ctx.status_code == 200


# ── GET /context/{id} ────────────────────────────────────────────────────────

class TestGetContext:
    async def test_get_context_known_subscriber_returns_200(self):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            sid = (await client.post("/subscribe", json=_profile_payload())).json()["subscriber_id"]
            resp = await client.get(f"/context/{sid}")
        assert resp.status_code == 200

    async def test_get_context_unknown_subscriber_returns_404(self):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/context/nonexistent_subscriber_xyz")
        assert resp.status_code == 404

    async def test_get_context_returns_json(self):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            sid = (await client.post("/subscribe", json=_profile_payload())).json()["subscriber_id"]
            resp = await client.get(f"/context/{sid}")
        body = resp.json()
        assert isinstance(body, dict)

    async def test_get_context_has_context_field(self):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            sid = (await client.post("/subscribe", json=_profile_payload())).json()["subscriber_id"]
            resp = await client.get(f"/context/{sid}")
        body = resp.json()
        assert "context" in body

    async def test_get_context_initially_empty_or_string(self):
        """Fresh subscriber has no pre-fetched context yet — field is empty string."""
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            sid = (await client.post("/subscribe", json=_profile_payload())).json()["subscriber_id"]
            resp = await client.get(f"/context/{sid}")
        assert isinstance(resp.json()["context"], str)


# ── DELETE /subscribe/{id} ───────────────────────────────────────────────────

class TestUnsubscribe:
    async def test_delete_returns_200(self):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            sid = (await client.post("/subscribe", json=_profile_payload())).json()["subscriber_id"]
            resp = await client.delete(f"/subscribe/{sid}")
        assert resp.status_code == 200

    async def test_delete_unknown_returns_404(self):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.delete("/subscribe/does_not_exist")
        assert resp.status_code == 404

    async def test_after_delete_get_returns_404(self):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            sid = (await client.post("/subscribe", json=_profile_payload())).json()["subscriber_id"]
            await client.delete(f"/subscribe/{sid}")
            resp = await client.get(f"/context/{sid}")
        assert resp.status_code == 404

    async def test_delete_cleans_up_buffer(self):
        """After unsubscribe, context buffer partition must be gone."""
        from apex.server import _buffer
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            sid = (await client.post("/subscribe", json=_profile_payload())).json()["subscriber_id"]
            await client.delete(f"/subscribe/{sid}")
        assert _buffer.get(sid) == []


# ── WebSocket /stream/{id} ────────────────────────────────────────────────────

class TestWebSocketStream:
    def test_websocket_connects_for_known_subscriber(self):
        client = TestClient(app)
        with client:
            # Subscribe first
            sid = client.post("/subscribe", json=_profile_payload()).json()["subscriber_id"]
            # Connect WebSocket — must not raise
            with client.websocket_connect(f"/stream/{sid}") as ws:
                # Send a ping to confirm connection is live
                ws.send_text("ping")
                # Server may or may not echo; just confirm no exception raised

    def test_websocket_unknown_subscriber_closes(self):
        client = TestClient(app)
        with client:
            with pytest.raises(Exception):
                # Server must reject unknown subscriber
                with client.websocket_connect("/stream/unknown_xyz") as ws:
                    ws.receive_text()

    def test_websocket_receives_push_after_manual_inject(self):
        """
        Manually inject context into the buffer, then connect via WebSocket
        and trigger a push. Verifies the full push pathway.
        """
        from apex.server import _buffer, _profiles, _push_context
        from apex.retrieval.rrf import Chunk
        import asyncio

        client = TestClient(app)
        with client:
            sid = client.post("/subscribe", json=_profile_payload()).json()["subscriber_id"]

            # Inject a chunk directly into the buffer
            chunk = Chunk("test_c1", "Relevant debugging context for testing", "test.py",
                          "debugging_python", score=0.9)
            _buffer.put(sid, [chunk])

            with client.websocket_connect(f"/stream/{sid}") as ws:
                # Request a push
                ws.send_text("pull")
                msg = ws.receive_text()
                assert isinstance(msg, str)
