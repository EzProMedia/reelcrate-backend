"""
Reelcrate auth — email/password + email verification + password reset + TOTP MFA.

Storage: users.json on disk (alongside jobs/). Each user record holds:
    email, name, password_hash, created_at,
    verified, verify_token, verify_token_exp,
    reset_token, reset_token_exp,
    mfa_enabled, mfa_secret

Routes:
  POST /api/auth/signup       {email, password, name?}      → {token, email, name, verified}
  POST /api/auth/signin       {email, password}              → either {token, email, name, verified}
                                                              or {mfa_required: true, mfa_token}
  POST /api/auth/mfa/verify   {mfa_token, code}              → {token, email, name, verified}
  POST /api/auth/forgot       {email}                        → {ok: true}      (always 200 to prevent enumeration)
  POST /api/auth/reset        {token, password}              → {ok: true}
  GET  /api/auth/verify       ?token=…                       → 302 redirect to /app/?verified=1
  GET  /api/auth/me           (Authorization: Bearer …)      → {email, name, verified, mfa_enabled}
  POST /api/auth/mfa/setup    (Authorization)                → {secret, otpauth_url, qr_png_b64}
  POST /api/auth/mfa/confirm  (Authorization) {code}         → {ok: true, mfa_enabled: true}
  POST /api/auth/mfa/disable  (Authorization) {password}     → {ok: true, mfa_enabled: false}
  POST /api/auth/resend       (Authorization)                → {ok: true}      (resend verification email)
"""

import io
import json
import os
import re
import secrets
import time
from base64 import b64encode
from pathlib import Path
from typing import Optional

import bcrypt
import jwt
import pyotp
import qrcode
from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from email_service import send_verify_email, send_reset_email


# -------------------- config --------------------

DATA_ROOT  = Path(os.environ.get("REELCRATE_DATA", "/tmp/reelcrate"))
DATA_ROOT.mkdir(parents=True, exist_ok=True)
USERS_FILE = DATA_ROOT / "users.json"

JWT_SECRET = os.environ.get("JWT_SECRET", "reelcrate-dev-secret-please-set-in-railway")
JWT_ALGO   = "HS256"
TOKEN_TTL          = 60 * 60 * 24 * 30   # 30 days for the real token
MFA_TOKEN_TTL      = 60 * 10             # 10 min for the temp token between signin and MFA verify
VERIFY_TOKEN_TTL   = 60 * 60 * 24        # 24 hours for email verification
RESET_TOKEN_TTL    = 60 * 60             # 1 hour for password reset

APP_URL = os.environ.get("APP_URL", "https://reelcrate.app").rstrip("/")

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
MFA_ISSUER = "Reelcrate"


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


# -------------------- token helpers --------------------

def make_token(email: str, ttl: int = TOKEN_TTL, kind: str = "session") -> str:
    now = int(time.time())
    payload = {"sub": email, "iat": now, "exp": now + ttl, "kind": kind}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGO)


def verify_token(token: str, expect_kind: str = "session") -> Optional[str]:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGO])
        if payload.get("kind") != expect_kind:
            return None
        return payload.get("sub")
    except Exception:
        return None


def random_token() -> str:
    """Short URL-safe token for reset/verify links."""
    return secrets.token_urlsafe(32)


# -------------------- user record helpers --------------------

def _public(u: dict) -> dict:
    return {
        "email":    u["email"],
        "name":     u.get("name", u["email"].split("@")[0]),
        "verified": bool(u.get("verified")),
        "mfa_enabled": bool(u.get("mfa_enabled")),
    }


# -------------------- routes --------------------

router = APIRouter(prefix="/api/auth", tags=["auth"])


class SignupReq(BaseModel):
    email: str
    password: str
    name: str = ""


class SigninReq(BaseModel):
    email: str
    password: str


class ForgotReq(BaseModel):
    email: str


class ResetReq(BaseModel):
    token: str
    password: str


class MfaVerifyReq(BaseModel):
    mfa_token: str
    code: str


class MfaConfirmReq(BaseModel):
    code: str


class MfaDisableReq(BaseModel):
    password: str


def _validate_credentials(email: str, password: str):
    if not EMAIL_RE.match(email):
        raise HTTPException(400, "Please enter a valid email")
    if len(password) < 8:
        raise HTTPException(400, "Password must be at least 8 characters")


# ---- signup / signin / me --------------------------------------------------

@router.post("/signup")
async def signup(req: SignupReq):
    email = req.email.strip().lower()
    _validate_credentials(email, req.password)
    users = _load_users()
    if email in users:
        raise HTTPException(409, "An account with this email already exists")
    name = req.name.strip() or email.split("@")[0]

    verify_tok = random_token()
    users[email] = {
        "email": email,
        "name":  name,
        "password_hash": _hash(req.password),
        "created_at":    int(time.time()),
        "verified":      False,
        "verify_token":  verify_tok,
        "verify_token_exp": int(time.time()) + VERIFY_TOKEN_TTL,
        "mfa_enabled":   False,
        "mfa_secret":    None,
    }
    _save_users(users)

    # Fire-and-forget the verification email; sign-up succeeds even if email errors out.
    send_verify_email(email, name, verify_tok)

    return {**_public(users[email]), "token": make_token(email)}


@router.post("/signin")
async def signin(req: SigninReq):
    email = req.email.strip().lower()
    users = _load_users()
    u = users.get(email)
    if not u or not _check(req.password, u["password_hash"]):
        raise HTTPException(401, "Wrong email or password")

    # If MFA is on, return a short-lived mfa_token instead of the real session token.
    if u.get("mfa_enabled"):
        return {"mfa_required": True,
                "mfa_token": make_token(email, ttl=MFA_TOKEN_TTL, kind="mfa")}
    return {**_public(u), "token": make_token(email)}


@router.post("/mfa/verify")
async def mfa_verify(req: MfaVerifyReq):
    email = verify_token(req.mfa_token, expect_kind="mfa")
    if not email:
        raise HTTPException(401, "MFA challenge expired — sign in again")
    users = _load_users()
    u = users.get(email)
    if not u or not u.get("mfa_enabled") or not u.get("mfa_secret"):
        raise HTTPException(400, "MFA not enabled on this account")
    totp = pyotp.TOTP(u["mfa_secret"])
    if not totp.verify(req.code.strip(), valid_window=1):
        raise HTTPException(401, "Wrong code — try again")
    return {**_public(u), "token": make_token(email)}


async def current_user(authorization: Optional[str] = Header(None)) -> str:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "Sign in required")
    token = authorization.split(" ", 1)[1].strip()
    email = verify_token(token, expect_kind="session")
    if not email:
        raise HTTPException(401, "Session expired — please sign in again")
    return email


@router.get("/me")
async def me(email: str = Depends(current_user)):
    users = _load_users()
    u = users.get(email)
    if not u:
        raise HTTPException(401, "Account not found")
    return _public(u)


# ---- email verification ----------------------------------------------------

@router.get("/verify")
async def verify_email_route(token: str):
    """Browser GET via the link in the email. Redirects back to /app/."""
    users = _load_users()
    for email, u in users.items():
        if u.get("verify_token") == token:
            if int(time.time()) > u.get("verify_token_exp", 0):
                return RedirectResponse(url=f"{APP_URL}/app/?verified=expired")
            u["verified"]         = True
            u["verify_token"]     = None
            u["verify_token_exp"] = 0
            _save_users(users)
            return RedirectResponse(url=f"{APP_URL}/app/?verified=1")
    return RedirectResponse(url=f"{APP_URL}/app/?verified=bad")


@router.post("/resend")
async def resend_verify(email: str = Depends(current_user)):
    users = _load_users()
    u = users.get(email)
    if not u:
        raise HTTPException(401, "Account not found")
    if u.get("verified"):
        return {"ok": True, "already_verified": True}
    tok = random_token()
    u["verify_token"]     = tok
    u["verify_token_exp"] = int(time.time()) + VERIFY_TOKEN_TTL
    _save_users(users)
    send_verify_email(email, u.get("name", ""), tok)
    return {"ok": True}


# ---- password reset --------------------------------------------------------

@router.post("/forgot")
async def forgot(req: ForgotReq):
    email = req.email.strip().lower()
    users = _load_users()
    u = users.get(email)
    if u:
        tok = random_token()
        u["reset_token"]     = tok
        u["reset_token_exp"] = int(time.time()) + RESET_TOKEN_TTL
        _save_users(users)
        send_reset_email(email, u.get("name", ""), tok)
    # Always return 200 — don't leak whether the email is registered.
    return {"ok": True}


@router.post("/reset")
async def reset(req: ResetReq):
    if len(req.password) < 8:
        raise HTTPException(400, "Password must be at least 8 characters")
    users = _load_users()
    for email, u in users.items():
        if u.get("reset_token") == req.token:
            if int(time.time()) > u.get("reset_token_exp", 0):
                raise HTTPException(400, "Reset link expired — request a new one")
            u["password_hash"]   = _hash(req.password)
            u["reset_token"]     = None
            u["reset_token_exp"] = 0
            _save_users(users)
            return {"ok": True}
    raise HTTPException(400, "Reset link is invalid")


# ---- MFA (TOTP) ------------------------------------------------------------

@router.post("/mfa/setup")
async def mfa_setup(email: str = Depends(current_user)):
    users = _load_users()
    u = users.get(email)
    if not u:
        raise HTTPException(401, "Account not found")
    if u.get("mfa_enabled"):
        raise HTTPException(409, "MFA is already enabled — disable it first to re-set up")
    # Generate a fresh secret and stash it pending /mfa/confirm.
    secret = pyotp.random_base32()
    u["mfa_secret"]  = secret
    u["mfa_enabled"] = False
    _save_users(users)

    otpauth = pyotp.totp.TOTP(secret).provisioning_uri(
        name=email, issuer_name=MFA_ISSUER)

    # Build a QR code PNG of the otpauth URL, return base64 so frontend can show it inline.
    img = qrcode.make(otpauth)
    buf = io.BytesIO(); img.save(buf, format="PNG")
    qr_b64 = b64encode(buf.getvalue()).decode()

    return {"secret": secret, "otpauth_url": otpauth,
            "qr_png_b64": f"data:image/png;base64,{qr_b64}"}


@router.post("/mfa/confirm")
async def mfa_confirm(req: MfaConfirmReq, email: str = Depends(current_user)):
    users = _load_users()
    u = users.get(email)
    if not u or not u.get("mfa_secret"):
        raise HTTPException(400, "Run /mfa/setup first")
    if u.get("mfa_enabled"):
        return {"ok": True, "mfa_enabled": True}
    totp = pyotp.TOTP(u["mfa_secret"])
    if not totp.verify(req.code.strip(), valid_window=1):
        raise HTTPException(401, "Wrong code — try again")
    u["mfa_enabled"] = True
    _save_users(users)
    return {"ok": True, "mfa_enabled": True}


@router.post("/mfa/disable")
async def mfa_disable(req: MfaDisableReq, email: str = Depends(current_user)):
    users = _load_users()
    u = users.get(email)
    if not u:
        raise HTTPException(401, "Account not found")
    if not _check(req.password, u["password_hash"]):
        raise HTTPException(401, "Wrong password")
    u["mfa_enabled"] = False
    u["mfa_secret"]  = None
    _save_users(users)
    return {"ok": True, "mfa_enabled": False}
