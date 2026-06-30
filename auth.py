"""
Reelcrate auth — minimal in-house email/password + JWT.

Storage: a single users.json on disk (alongside the jobs/ folder). Fine for the
first few hundred users; migrate to Postgres when we need it.

Routes:
  POST /api/auth/signup   {email, password, name?}   → {token, email, name}
  POST /api/auth/signin   {email, password}          → {token, email, name}
  GET  /api/auth/me       (Authorization: Bearer …)  → {email, name}

Other routes that need a logged-in user import the `current_user` dependency.
"""

import json
import os
import re
import time
from pathlib import Path
from typing import Optional

import bcrypt
import jwt
from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel


# -------------------- config --------------------

DATA_ROOT = Path(os.environ.get("REELCRATE_DATA", "/tmp/reelcrate"))
DATA_ROOT.mkdir(parents=True, exist_ok=True)
USERS_FILE = DATA_ROOT / "users.json"

# JWT_SECRET is read from env in prod (Railway env var). Falls back to a static
# dev string locally so tests work without setup. RENAMED in prod = all sessions
# invalidated.
JWT_SECRET = os.environ.get("JWT_SECRET", "reelcrate-dev-secret-please-set-in-railway")
JWT_ALGO   = "HS256"
TOKEN_TTL  = 60 * 60 * 24 * 30   # 30 days

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


# -------------------- storage helpers --------------------

def _load_users() -> dict:
    if not USERS_FILE.exists():
        return {}
    try:
        return json.loads(USERS_FILE.read_text())
    except Exception:
        return {}


def _save_users(users: dict) -> None:
    tmp = USERS_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(users, indent=2))
    tmp.replace(USERS_FILE)


def _hash(pw: str) -> str:
    return bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()


def _check(pw: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(pw.encode(), hashed.encode())
    except Exception:
        return False


def make_token(email: str) -> str:
    now = int(time.time())
    payload = {"sub": email, "iat": now, "exp": now + TOKEN_TTL}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGO)


def verify_token(token: str) -> Optional[str]:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGO])
        return payload.get("sub")
    except Exception:
        return None


# -------------------- routes --------------------

router = APIRouter(prefix="/api/auth", tags=["auth"])


class SignupReq(BaseModel):
    email: str
    password: str
    name: str = ""


class SigninReq(BaseModel):
    email: str
    password: str


def _validate_credentials(email: str, password: str):
    if not EMAIL_RE.match(email):
        raise HTTPException(400, "Please enter a valid email")
    if len(password) < 8:
        raise HTTPException(400, "Password must be at least 8 characters")


@router.post("/signup")
async def signup(req: SignupReq):
    email = req.email.strip().lower()
    _validate_credentials(email, req.password)
    users = _load_users()
    if email in users:
        raise HTTPException(409, "An account with this email already exists")
    name = req.name.strip() or email.split("@")[0]
    users[email] = {
        "email": email,
        "name": name,
        "password_hash": _hash(req.password),
        "created_at": int(time.time()),
    }
    _save_users(users)
    return {"token": make_token(email), "email": email, "name": name}


@router.post("/signin")
async def signin(req: SigninReq):
    email = req.email.strip().lower()
    users = _load_users()
    u = users.get(email)
    if not u or not _check(req.password, u["password_hash"]):
        # Same error for both cases to avoid leaking which emails exist.
        raise HTTPException(401, "Wrong email or password")
    name = u.get("name") or email.split("@")[0]
    return {"token": make_token(email), "email": email, "name": name}


async def current_user(authorization: Optional[str] = Header(None)) -> str:
    """FastAPI dependency. Returns the email of the signed-in user or 401."""
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "Sign in required")
    token = authorization.split(" ", 1)[1].strip()
    email = verify_token(token)
    if not email:
        raise HTTPException(401, "Session expired — please sign in again")
    return email


@router.get("/me")
async def me(email: str = Depends(current_user)):
    users = _load_users()
    u = users.get(email)
    if not u:
        raise HTTPException(401, "Account not found")
    return {"email": u["email"], "name": u.get("name", u["email"].split("@")[0])}
