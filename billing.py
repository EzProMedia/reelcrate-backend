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

import asyncio
import json
import os
import time
from typing import Optional

import stripe
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from reliability import queue_email

from auth import current_user, _load_users, _save_users


STRIPE_SECRET_KEY     = os.environ.get("STRIPE_SECRET_KEY", "")
STRIPE_PRICE_ID       = os.environ.get("STRIPE_PRICE_ID", "")        # monthly $19
STRIPE_PRICE_ID_YEAR  = os.environ.get("STRIPE_PRICE_ID_YEAR", "")   # annual  $190
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "")
APP_URL               = os.environ.get("APP_URL", "https://reelcrate.app").rstrip("/")
TRIAL_DAYS            = int(os.environ.get("STRIPE_TRIAL_DAYS", "14"))

# Map plan name → env price id
_PRICE_BY_PLAN = {
    "monthly": STRIPE_PRICE_ID,
    "yearly":  STRIPE_PRICE_ID_YEAR,
}

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


async def _get_or_create_customer(email: str, name: str = "") -> str:
    """Fetch existing Stripe customer id or create one, and cache it on the user."""
    users = _load_users()
    u = users.get(email)
    if not u:
        raise HTTPException(401, "Account not found")
    cid = u.get("stripe_customer_id")
    if cid:
        return cid
    customer = await asyncio.to_thread(stripe.Customer.create,
        email=email,
        name=name or u.get("name", ""),
        metadata={"reelcrate_email": email},
        idempotency_key="customer:" + email,
    )
    users = _load_users()
    u = users[email]
    u["stripe_customer_id"] = customer.id
    _save_users(users)
    return customer.id


# -------------------- routes --------------------

def _resolve_promotion_code(code: str) -> Optional[str]:
    """Turn a customer-facing promo code string (e.g. 'FOUNDER40') into its
    Stripe promotion_code id ('promo_...'), or None if it isn't a currently
    valid, active code. Never raises — a bad/expired code just falls through to
    the manual promo field at checkout."""
    code = (code or "").strip()
    if not code:
        return None
    try:
        found = stripe.PromotionCode.list(code=code, active=True, limit=1)
        data = found.get("data") if isinstance(found, dict) else found.data
        if data:
            return data[0]["id"] if isinstance(data[0], dict) else data[0].id
    except Exception as e:
        print(f"[billing] promo resolve failed for {code!r}: {e}")
    return None


class CheckoutReq(BaseModel):
    plan: str = "monthly"          # "monthly" or "yearly"
    promo: Optional[str] = Field(default=None, max_length=100)    # optional promo code to auto-apply (e.g. FOUNDER40)


@router.post("/checkout")
async def checkout(req: CheckoutReq = CheckoutReq(), email: str = Depends(current_user)):
    if not STRIPE_SECRET_KEY:
        raise HTTPException(503, "Billing not configured on the server")

    price_id = _PRICE_BY_PLAN.get(req.plan)
    if not price_id:
        raise HTTPException(503, f"Plan '{req.plan}' not configured")

    users = _load_users()
    u = users.get(email) or {}
    if not u.get("verified"):
        raise HTTPException(403, "Please verify your email before subscribing")

    customer_id = await _get_or_create_customer(email, u.get("name", ""))

    session_kwargs = dict(
        mode="subscription",
        customer=customer_id,
        line_items=[{"price": price_id, "quantity": 1}],
        subscription_data={
            "trial_period_days": TRIAL_DAYS,
            "metadata": {"reelcrate_email": email, "reelcrate_plan": req.plan},
        },
        success_url=f"{APP_URL}/app/?checkout=success&session_id={{CHECKOUT_SESSION_ID}}",
        cancel_url =f"{APP_URL}/app/?checkout=cancel",
    )

    # Auto-apply a promo code (e.g. the ?promo=FOUNDER40 deep link) when one is
    # supplied and resolves to a live Stripe promotion code. Stripe rejects
    # `discounts` and `allow_promotion_codes` together, so it's one or the other:
    #   - valid promo  -> pre-fill the discount, no manual field needed
    #   - no/bad promo -> show the manual promo field so it can still be typed
    promo_id = await asyncio.to_thread(_resolve_promotion_code, req.promo) if req.promo else None
    applied_promo = None
    if promo_id:
        session_kwargs["discounts"] = [{"promotion_code": promo_id}]
        applied_promo = (req.promo or "").strip().upper()
    else:
        session_kwargs["allow_promotion_codes"] = True

    try:
        session = await asyncio.to_thread(stripe.checkout.Session.create, **session_kwargs)
    except stripe.error.InvalidRequestError as exc:
        # Only retry a rejected discount; never hide unrelated payment errors.
        if not promo_id or not str(getattr(exc, 'param', '') or '').startswith(('discounts', 'promotion_code')):
            raise HTTPException(502, "Checkout could not be opened. Please try again.") from exc
        session_kwargs.pop('discounts', None)
        session_kwargs['allow_promotion_codes'] = True
        applied_promo = None
        session = await asyncio.to_thread(stripe.checkout.Session.create, **session_kwargs)
    return {"url": session.url, "session_id": session.id, "promo_applied": applied_promo}


@router.post("/portal")
async def portal(email: str = Depends(current_user)):
    if not STRIPE_SECRET_KEY:
        raise HTTPException(503, "Billing not configured on the server")
    users = _load_users()
    u = users.get(email) or {}
    cid = u.get("stripe_customer_id")
    if not cid:
        raise HTTPException(400, "No Stripe customer yet — start a subscription first")
    session = await asyncio.to_thread(stripe.billing_portal.Session.create,
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

    new_status = sub.get("status")
    u["subscription_id"] = sub.get("id")
    u["subscription_status"] = new_status
    period_end = sub.get("current_period_end") or sub.get("trial_end") or max(
        (item.get("current_period_end") or 0 for item in (sub.get("items") or {}).get("data", [])), default=0)
    u["subscription_current_period_end"] = period_end

    # Fire the welcome email exactly once — the first time a user transitions
    # to a paying/trialing state. We stamp welcome_sent_at so we never re-send
    # on later Stripe update events (which fire every time anything changes).
    if new_status in ("trialing", "active") and not u.get("subscription_welcome_sent_at"):
        try:
            import time as _time
            from email_service import send_subscription_welcome
            plan_label = (sub.get("metadata") or {}).get("reelcrate_plan", "").capitalize()
            plan_label = f"Reelcrate {plan_label}" if plan_label else "Reelcrate"
            # For trials we prefer trial_end; for direct activations we use
            # current_period_end so the "first bill" line still makes sense.
            trial_end = sub.get("trial_end") or period_end or 0
            queue_email(send_subscription_welcome,
                delivery_id="subscription-welcome:" + email,
                to=email,
                name=u.get("name", ""),
                plan_label=plan_label,
                trial_end_epoch=int(trial_end) if trial_end else 0,
                is_trialing=(new_status == "trialing"),
            )
            u["subscription_welcome_sent_at"] = int(_time.time())
        except Exception as e:
            print(f"[billing] welcome email failed for {email}: {e}")

    _save_users(users)


@router.post("/webhook")
async def webhook(request: Request):
    payload = await request.body()
    sig = request.headers.get("Stripe-Signature", "")
    if not STRIPE_WEBHOOK_SECRET:
        raise HTTPException(503, "Payment webhook is not configured")
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
                sub = await asyncio.to_thread(stripe.Subscription.retrieve, sub_id)
                _apply_subscription(sub)
            except Exception as exc:
                raise HTTPException(503, "Subscription update unavailable; retry delivery") from exc

    return {"received": True}
