"""
Reelcrate billing — Stripe Checkout Sessions + Customer Portal + Webhook handler.

Flow:
  POST /api/billing/checkout   → creates a Stripe Checkout Session, returns URL
  POST /api/billing/portal     → creates a Customer Portal Session, returns URL
  POST /api/billing/webhook    → receives Stripe subscription events (RAW body)
  GET  /api/billing/status     → returns the caller's subscription tier

User record additions:
  stripe_customer_id       : "cus_..."         (created on first checkout)
  subscription_status      : "trialing" | "active" | "past_due" | "canceled" | None
  subscription_id          : "sub_..."         (Stripe subscription id)
  subscription_current_period_end : int (epoch)

`is_paying()` returns True when the subscription is trialing OR active — used by the
upload gate in main.py.
"""

import json
import os
import time
from typing import Optional

import stripe
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from auth import current_user, _load_users, _save_users


STRIPE_SECRET_KEY     = os.environ.get("STRIPE_SECRET_KEY", "")
STRIPE_PRICE_ID       = os.environ.get("STRIPE_PRICE_ID", "")
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "")
APP_URL               = os.environ.get("APP_URL", "https://reelcrate.app").rstrip("/")
TRIAL_DAYS            = int(os.environ.get("STRIPE_TRIAL_DAYS", "14"))

if STRIPE_SECRET_KEY:
    stripe.api_key = STRIPE_SECRET_KEY


router = APIRouter(prefix="/api/billing", tags=["billing"])


# -------------------- helpers --------------------

def is_paying(email: str) -> bool:
    """The upload gate calls this to decide whether to accept an upload."""
    u = _load_users().get(email) or {}
    status = u.get("subscription_status")
    if status in ("trialing", "active"):
        # Also check current_period_end if we have it — Stripe events may lag.
        end = u.get("subscription_current_period_end") or 0
        if end == 0 or end > int(time.time()) - 3600:
            return True
    return False


def _billing_public(u: dict) -> dict:
    return {
        "subscription_status":              u.get("subscription_status"),
        "subscription_current_period_end":  u.get("subscription_current_period_end"),
        "has_stripe_customer":              bool(u.get("stripe_customer_id")),
    }


def _get_or_create_customer(email: str, name: str = "") -> str:
    """Fetch existing Stripe customer id or create one, and cache it on the user."""
    users = _load_users()
    u = users.get(email)
    if not u:
        raise HTTPException(401, "Account not found")
    cid = u.get("stripe_customer_id")
    if cid:
        return cid
    customer = stripe.Customer.create(
        email=email,
        name=name or u.get("name", ""),
        metadata={"reelcrate_email": email},
    )
    u["stripe_customer_id"] = customer.id
    _save_users(users)
    return customer.id


# -------------------- routes --------------------

@router.post("/checkout")
async def checkout(email: str = Depends(current_user)):
    if not STRIPE_SECRET_KEY:
        raise HTTPException(503, "Billing not configured on the server")
    if not STRIPE_PRICE_ID:
        raise HTTPException(503, "Billing price not configured")

    users = _load_users()
    u = users.get(email) or {}
    if not u.get("verified"):
        raise HTTPException(403, "Please verify your email before subscribing")

    customer_id = _get_or_create_customer(email, u.get("name", ""))

    session = stripe.checkout.Session.create(
        mode="subscription",
        customer=customer_id,
        line_items=[{"price": STRIPE_PRICE_ID, "quantity": 1}],
        subscription_data={
            "trial_period_days": TRIAL_DAYS,
            "metadata": {"reelcrate_email": email},
        },
        allow_promotion_codes=True,
        success_url=f"{APP_URL}/app/?checkout=success&session_id={{CHECKOUT_SESSION_ID}}",
        cancel_url =f"{APP_URL}/app/?checkout=cancel",
    )
    return {"url": session.url, "session_id": session.id}


@router.post("/portal")
async def portal(email: str = Depends(current_user)):
    if not STRIPE_SECRET_KEY:
        raise HTTPException(503, "Billing not configured on the server")
    users = _load_users()
    u = users.get(email) or {}
    cid = u.get("stripe_customer_id")
    if not cid:
        raise HTTPException(400, "No Stripe customer yet — start a subscription first")
    session = stripe.billing_portal.Session.create(
        customer=cid,
        return_url=f"{APP_URL}/app/",
    )
    return {"url": session.url}


@router.get("/status")
async def status_(email: str = Depends(current_user)):
    users = _load_users()
    u = users.get(email) or {}
    return _billing_public(u)


# -------- webhook (Stripe → us) --------

def _apply_subscription(sub: dict):
    """Given a Stripe Subscription object, update the user record if we can
    match it back to a Reelcrate account via metadata."""
    email = (sub.get("metadata") or {}).get("reelcrate_email")
    if not email:
        # Fall back to customer lookup — the subscription's customer id was
        # created with our metadata so we can search local users.
        cid = sub.get("customer")
        if cid:
            users = _load_users()
            for e, ur in users.items():
                if ur.get("stripe_customer_id") == cid:
                    email = e; break
    if not email:
        return
    users = _load_users()
    u = users.get(email)
    if not u:
        return
    u["subscription_id"] = sub.get("id")
    u["subscription_status"] = sub.get("status")
    u["subscription_current_period_end"] = sub.get("current_period_end") or 0
    _save_users(users)


@router.post("/webhook")
async def webhook(request: Request):
    payload = await request.body()
    sig = request.headers.get("Stripe-Signature", "")
    if not STRIPE_WEBHOOK_SECRET:
        # In dev, accept parsed JSON directly so we can test without a webhook secret.
        try:
            event = json.loads(payload.decode())
        except Exception:
            raise HTTPException(400, "invalid payload")
    else:
        try:
            event = stripe.Webhook.construct_event(
                payload, sig, STRIPE_WEBHOOK_SECRET
            )
        except stripe.error.SignatureVerificationError:
            raise HTTPException(400, "bad signature")
        except Exception:
            raise HTTPException(400, "invalid payload")

    et = event.get("type", "")
    obj = (event.get("data") or {}).get("object") or {}

    if et in (
        "customer.subscription.created",
        "customer.subscription.updated",
        "customer.subscription.trial_will_end",
    ):
        _apply_subscription(obj)
    elif et == "customer.subscription.deleted":
        obj["status"] = "canceled"
        _apply_subscription(obj)
    elif et == "checkout.session.completed":
        # Fetch the subscription and apply — Session doesn't include full status.
        sub_id = obj.get("subscription")
        if sub_id:
            try:
                sub = stripe.Subscription.retrieve(sub_id)
                _apply_subscription(sub)
            except Exception:
                pass

    return {"received": True}
