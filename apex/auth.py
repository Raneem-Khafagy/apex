"""
APEX Auth Module — local-user authentication for the web UI.

Provides:
  - UserDB       SQLite-backed user store (apex_users.db)
  - hash_password / verify_password   bcrypt via passlib
  - create_token / decode_token       JWT via python-jose
  - get_current_user                  FastAPI dependency (Bearer token)

Security design
---------------
- Passwords stored as bcrypt hashes only — plaintext never persisted.
- DB file created with 0o600 permissions (owner read/write only).
- All SQL queries use parameterised placeholders (no string interpolation).
- JWT secret loaded from env var APEX_JWT_SECRET, or from
  apex_jwt_secret.key on disk (created on first run). This ensures
  tokens survive daemon restarts without requiring manual config.
- Server binds to 127.0.0.1 by default — no remote exposure.

Limitations (acceptable for research prototype on localhost)
------------------------------------------------------------
- Single-instance only; no horizontal scaling.
- Shared knowledge-base index across all users (see CLAUDE.md §Security).
- No rate limiting on login attempts.
- Token revocation is client-side only (delete from localStorage).
"""
from __future__ import annotations

import json
import os
import secrets
import sqlite3
import time
import uuid
from typing import Any, Optional

import bcrypt as _bcrypt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from loguru import logger

# ── Constants ─────────────────────────────────────────────────────────────────

_SECRET_FILE = "apex_jwt_secret.key"
_TOKEN_EXPIRE_SECONDS = 7 * 24 * 3600   # 7 days

_bearer = HTTPBearer(auto_error=False)

# ── Module-level lazy singletons (reset in tests via monkeypatch) ─────────────

_user_db: Optional["UserDB"] = None
_jwt_secret: Optional[str] = None


def _get_secret() -> str:
    """
    Load JWT secret from env → disk file → generate new one.
    Written to apex_jwt_secret.key so tokens survive daemon restarts.
    """
    global _jwt_secret
    if _jwt_secret is not None:
        return _jwt_secret

    env_secret = os.environ.get("APEX_JWT_SECRET")
    if env_secret:
        _jwt_secret = env_secret
        return _jwt_secret

    if os.path.exists(_SECRET_FILE):
        with open(_SECRET_FILE) as f:
            _jwt_secret = f.read().strip()
        return _jwt_secret

    _jwt_secret = secrets.token_hex(32)
    with open(_SECRET_FILE, "w") as f:
        f.write(_jwt_secret)
    os.chmod(_SECRET_FILE, 0o600)
    logger.info("Auth: generated new JWT secret → {}", _SECRET_FILE)
    return _jwt_secret


def get_user_db() -> "UserDB":
    """Return the module-level UserDB singleton, creating it if needed."""
    global _user_db
    if _user_db is None:
        db_path = os.environ.get("APEX_USERS_DB", "apex_users.db")
        _user_db = UserDB(db_path=db_path)
    return _user_db


# ── Password helpers ──────────────────────────────────────────────────────────

def hash_password(plain: str) -> str:
    """Return a bcrypt hash of plain. Each call produces a unique salt."""
    return _bcrypt.hashpw(plain.encode(), _bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    """Return True iff plain matches the bcrypt hash."""
    try:
        return _bcrypt.checkpw(plain.encode(), hashed.encode())
    except Exception:
        return False


# ── JWT helpers ───────────────────────────────────────────────────────────────

def create_token(
    payload: dict[str, Any],
    secret: str,
    expires_in_seconds: int = _TOKEN_EXPIRE_SECONDS,
) -> str:
    """
    Create a signed JWT.

    Parameters
    ----------
    payload
        Claims to embed (user_id, username, etc.).
    secret
        Signing secret.
    expires_in_seconds
        Validity window. Use a negative value in tests to get an expired token.
    """
    data = dict(payload)
    data["exp"] = time.time() + expires_in_seconds
    return jwt.encode(data, secret, algorithm="HS256")


def decode_token(token: str, secret: str) -> dict[str, Any]:
    """
    Decode and verify a JWT.

    Raises JWTError (or subclass) on invalid signature, expiry, or malformed input.
    """
    return jwt.decode(token, secret, algorithms=["HS256"])


# ── FastAPI dependency ────────────────────────────────────────────────────────

async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> dict[str, Any]:
    """
    FastAPI dependency — decode Bearer token and return the user dict.

    Raises HTTP 401 on missing / invalid / expired token.
    The returned dict matches UserDB.get_by_id() — safe fields only,
    no password_hash.
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication token",
        )
    try:
        payload = decode_token(credentials.credentials, _get_secret())
        user_id: str = payload["user_id"]
    except (JWTError, KeyError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )

    db = get_user_db()
    user = db.get_by_id(user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )
    return user


# ── UserDB ────────────────────────────────────────────────────────────────────

class UserDB:
    """
    SQLite-backed user store for APEX local authentication.

    All public methods return safe dicts (no password_hash field).
    SQL queries use parameterised placeholders to prevent injection.

    Parameters
    ----------
    db_path
        Path to the SQLite file. Created with 0o600 permissions if new.
    """

    def __init__(self, db_path: str = "apex_users.db") -> None:
        self._db_path = db_path
        self._con = sqlite3.connect(db_path, check_same_thread=False)
        self._con.row_factory = sqlite3.Row
        self._create_schema()
        # Restrict file permissions after creation
        try:
            os.chmod(db_path, 0o600)
        except OSError:
            pass  # in-memory or test path that doesn't exist on disk

    def _create_schema(self) -> None:
        self._con.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id       TEXT PRIMARY KEY,
                username      TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                domain        TEXT NOT NULL DEFAULT 'writing',
                profile_json  TEXT NOT NULL DEFAULT '{}',
                subscriber_id TEXT,
                onboarded     INTEGER NOT NULL DEFAULT 0,
                created_at    REAL NOT NULL
            )
        """)
        self._con.commit()

    # ── Write operations ─────────────────────────────────────────────────────

    def create_user(
        self,
        username: str,
        password: str,
        domain: str = "writing",
    ) -> dict[str, Any]:
        """
        Create a new user account.

        Returns a safe user dict (no password_hash).
        Raises ValueError if the username already exists.
        """
        existing = self.get_by_username(username)
        if existing is not None:
            raise ValueError(f"Username '{username}' already exists")

        user_id = str(uuid.uuid4())
        password_hash = hash_password(password)
        now = time.time()

        self._con.execute(
            """
            INSERT INTO users
                (user_id, username, password_hash, domain, profile_json,
                 subscriber_id, onboarded, created_at)
            VALUES (?, ?, ?, ?, ?, NULL, 0, ?)
            """,
            [user_id, username, password_hash, domain, "{}", now],
        )
        self._con.commit()
        logger.info("Auth: user created username='{}' domain='{}'", username, domain)
        return self._safe(self._con.execute(
            "SELECT * FROM users WHERE user_id = ?", [user_id]
        ).fetchone())

    def update_subscriber_id(self, user_id: str, subscriber_id: str) -> None:
        """Persist a new subscriber_id for the user (called on re-hydration)."""
        self._con.execute(
            "UPDATE users SET subscriber_id = ? WHERE user_id = ?",
            [subscriber_id, user_id],
        )
        self._con.commit()

    def update_profile(
        self,
        user_id: str,
        domain: str,
        profile: dict[str, Any],
    ) -> None:
        """Persist updated domain + ConsumerProfile settings."""
        self._con.execute(
            "UPDATE users SET domain = ?, profile_json = ?, onboarded = 1 WHERE user_id = ?",
            [domain, json.dumps(profile), user_id],
        )
        self._con.commit()

    # ── Read / auth operations ────────────────────────────────────────────────

    def authenticate(self, username: str, password: str) -> Optional[dict[str, Any]]:
        """
        Verify credentials.

        Returns a safe user dict on success, None on failure.
        Timing-safe: always runs verify_password even for unknown users.
        """
        row = self._con.execute(
            "SELECT * FROM users WHERE username = ?", [username]
        ).fetchone()
        if row is None:
            # Run a dummy verify to prevent username-enumeration via timing
            verify_password(password, hash_password("dummy"))
            return None
        if not verify_password(password, row["password_hash"]):
            return None
        return self._safe(row)

    def get_by_username(self, username: str) -> Optional[dict[str, Any]]:
        row = self._con.execute(
            "SELECT * FROM users WHERE username = ?", [username]
        ).fetchone()
        return self._safe(row) if row else None

    def get_by_id(self, user_id: str) -> Optional[dict[str, Any]]:
        row = self._con.execute(
            "SELECT * FROM users WHERE user_id = ?", [user_id]
        ).fetchone()
        return self._safe(row) if row else None

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _safe(row: sqlite3.Row) -> dict[str, Any]:
        """Convert a Row to a dict, stripping the password_hash field."""
        d = dict(row)
        d.pop("password_hash", None)
        return d
