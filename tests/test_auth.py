"""
Tests for apex/auth.py — UserDB, password hashing, JWT, and the
/auth/* endpoint behaviours.

All tests use an in-memory / temp-file UserDB so they are isolated
from each other and from any real apex_users.db on disk.

No mocks are used.  UserDB accepts a custom db_path so real SQLite
is exercised without touching production state.
"""
from __future__ import annotations

import os
import tempfile
import time

import pytest
from fastapi.testclient import TestClient

from apex.auth import UserDB, hash_password, verify_password, create_token, decode_token


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def db(tmp_path):
    """Fresh UserDB backed by a temp file for each test."""
    return UserDB(db_path=str(tmp_path / "test_users.db"))


@pytest.fixture()
def jwt_secret():
    return "test-secret-not-for-production"


# ── Password hashing ─────────────────────────────────────────────────────────

class TestPasswordHashing:
    def test_hash_is_not_plaintext(self):
        h = hash_password("hunter2")
        assert h != "hunter2"

    def test_verify_correct_password(self):
        h = hash_password("correct-horse")
        assert verify_password("correct-horse", h) is True

    def test_verify_wrong_password(self):
        h = hash_password("correct-horse")
        assert verify_password("wrong-horse", h) is False

    def test_two_hashes_of_same_password_differ(self):
        """bcrypt salts each hash — deterministic equality would be a bug."""
        h1 = hash_password("same")
        h2 = hash_password("same")
        assert h1 != h2

    def test_empty_password_hashes(self):
        """Even empty strings must hash without raising."""
        h = hash_password("")
        assert verify_password("", h) is True
        assert verify_password("x", h) is False


# ── JWT ───────────────────────────────────────────────────────────────────────

class TestJWT:
    def test_create_and_decode(self, jwt_secret):
        token = create_token({"user_id": "u1", "username": "raneem"}, jwt_secret)
        payload = decode_token(token, jwt_secret)
        assert payload["user_id"] == "u1"
        assert payload["username"] == "raneem"

    def test_wrong_secret_raises(self, jwt_secret):
        token = create_token({"user_id": "u1"}, jwt_secret)
        with pytest.raises(Exception):
            decode_token(token, "wrong-secret")

    def test_expired_token_raises(self, jwt_secret):
        token = create_token({"user_id": "u1"}, jwt_secret, expires_in_seconds=-1)
        with pytest.raises(Exception):
            decode_token(token, jwt_secret)

    def test_token_contains_exp_claim(self, jwt_secret):
        token = create_token({"user_id": "u1"}, jwt_secret)
        payload = decode_token(token, jwt_secret)
        assert "exp" in payload
        assert payload["exp"] > time.time()


# ── UserDB ────────────────────────────────────────────────────────────────────

class TestUserDB:
    def test_create_user(self, db):
        user = db.create_user("alice", "pass123", "writing")
        assert user["username"] == "alice"
        assert user["domain"] == "writing"
        assert "user_id" in user
        assert "password_hash" not in user  # must never leak hash

    def test_duplicate_username_raises(self, db):
        db.create_user("alice", "pass123", "writing")
        with pytest.raises(ValueError, match="already exists"):
            db.create_user("alice", "different", "research")

    def test_get_by_username(self, db):
        db.create_user("bob", "secret", "factory")
        user = db.get_by_username("bob")
        assert user is not None
        assert user["username"] == "bob"

    def test_get_nonexistent_returns_none(self, db):
        assert db.get_by_username("nobody") is None

    def test_get_by_id(self, db):
        created = db.create_user("carol", "pw", "research")
        fetched = db.get_by_id(created["user_id"])
        assert fetched is not None
        assert fetched["username"] == "carol"

    def test_authenticate_correct(self, db):
        db.create_user("dave", "goodpass", "writing")
        user = db.authenticate("dave", "goodpass")
        assert user is not None
        assert user["username"] == "dave"

    def test_authenticate_wrong_password(self, db):
        db.create_user("eve", "goodpass", "writing")
        assert db.authenticate("eve", "badpass") is None

    def test_authenticate_unknown_user(self, db):
        assert db.authenticate("nobody", "anything") is None

    def test_update_subscriber_id(self, db):
        user = db.create_user("frank", "pw", "factory")
        db.update_subscriber_id(user["user_id"], "sub-abc")
        updated = db.get_by_id(user["user_id"])
        assert updated["subscriber_id"] == "sub-abc"

    def test_update_profile(self, db):
        user = db.create_user("grace", "pw", "research")
        profile = {
            "autonomy_level": "suggestive",
            "verbosity": "detailed",
            "max_context_tokens": 1024,
        }
        db.update_profile(user["user_id"], "research", profile)
        updated = db.get_by_id(user["user_id"])
        import json
        stored = json.loads(updated["profile_json"])
        assert stored["verbosity"] == "detailed"

    def test_db_file_permissions(self, tmp_path):
        """DB file must be owner-only readable (0o600)."""
        db_path = str(tmp_path / "perms_test.db")
        UserDB(db_path=db_path)
        mode = oct(os.stat(db_path).st_mode & 0o777)
        assert mode == oct(0o600), f"Expected 0o600, got {mode}"

    def test_sql_injection_username_safe(self, db):
        """SQL injection in username must not corrupt the DB."""
        malicious = "'; DROP TABLE users; --"
        result = db.get_by_username(malicious)
        assert result is None
        # DB should still be intact — can still create normal user
        user = db.create_user("safe_user", "pw", "writing")
        assert user["username"] == "safe_user"


# ── /auth endpoint integration (via TestClient) ───────────────────────────────

@pytest.fixture()
def client(tmp_path, monkeypatch):
    """
    TestClient with auth endpoints wired.
    Uses a fresh temp DB and a test JWT secret to avoid touching
    production state or real apex_users.db.
    """
    monkeypatch.setenv("APEX_USERS_DB", str(tmp_path / "auth_test.db"))
    monkeypatch.setenv("APEX_JWT_SECRET", "test-jwt-secret-fixture")

    # Import server after env vars set — server reads them at module level via lazy init
    from apex.server import app
    from apex import auth as auth_module
    # Reset the module-level singletons so they pick up monkeypatched env vars
    auth_module._user_db = None
    auth_module._jwt_secret = None

    return TestClient(app)


class TestAuthEndpoints:
    def test_register_success(self, client):
        r = client.post("/auth/register", json={
            "username": "alice", "password": "pass1234", "domain": "writing"
        })
        assert r.status_code == 200
        data = r.json()
        assert "token" in data
        assert "user_id" in data

    def test_register_duplicate_username(self, client):
        client.post("/auth/register", json={
            "username": "bob", "password": "pass1234", "domain": "writing"
        })
        r = client.post("/auth/register", json={
            "username": "bob", "password": "other_pass", "domain": "research"
        })
        assert r.status_code == 409

    def test_register_short_password_rejected(self, client):
        r = client.post("/auth/register", json={
            "username": "carol", "password": "abc", "domain": "writing"
        })
        assert r.status_code == 422

    def test_login_success(self, client):
        client.post("/auth/register", json={
            "username": "dave", "password": "pass1234", "domain": "factory"
        })
        r = client.post("/auth/login", json={
            "username": "dave", "password": "pass1234"
        })
        assert r.status_code == 200
        assert "token" in r.json()

    def test_login_wrong_password(self, client):
        client.post("/auth/register", json={
            "username": "eve", "password": "correct", "domain": "writing"
        })
        r = client.post("/auth/login", json={
            "username": "eve", "password": "wrong"
        })
        assert r.status_code == 401

    def test_login_unknown_user(self, client):
        r = client.post("/auth/login", json={
            "username": "nobody", "password": "anything"
        })
        assert r.status_code == 401

    def test_me_requires_auth(self, client):
        r = client.get("/auth/me")
        assert r.status_code == 401

    def test_me_returns_user(self, client):
        reg = client.post("/auth/register", json={
            "username": "frank", "password": "pass1234", "domain": "research"
        })
        token = reg.json()["token"]
        r = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 200
        data = r.json()
        assert data["username"] == "frank"
        assert data["domain"] == "research"

    def test_me_invalid_token(self, client):
        r = client.get("/auth/me", headers={"Authorization": "Bearer bad.token.here"})
        assert r.status_code == 401

    def test_onboard_sets_domain_and_profile(self, client):
        reg = client.post("/auth/register", json={
            "username": "grace", "password": "pass1234", "domain": "writing"
        })
        token = reg.json()["token"]
        r = client.post("/auth/onboard", json={
            "domain": "research",
            "profile": {
                "autonomy_level": "suggestive",
                "goal_horizon": "long",
                "interaction_style": "conversational",
                "output_format": "markdown",
                "vocabulary_level": "domain-expert",
                "verbosity": "detailed",
                "citation_style": "footnote",
                "max_context_tokens": 1024,
            }
        }, headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 200
        # Verify /auth/me reflects the onboarded domain
        me = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert me.json()["domain"] == "research"
