"""
Email sender for Reelcrate. Thin wrapper around Resend's HTTP API.

Resend gives 3,000 emails/month free on the sandbox sender (onboarding@resend.dev).
Once we verify the reelcrate.app domain we can flip the FROM address.

If RESEND_API_KEY isn't set the sender prints to stdout instead of sending — useful
for local dev so we never accidentally email real users from a dev machine.
"""

import json
import os
import urllib.request
import urllib.error
from typing import Optional


RESEND_API_KEY = os.environ.get("RESEND_API_KEY", "")
RESEND_URL     = "https://api.resend.com/emails"
FROM_ADDRESS   = os.environ.get("RESEND_FROM", "Reelcrate <onboarding@resend.dev>")
APP_URL        = os.environ.get("APP_URL", "https://reelcrate.app").rstrip("/")
# Where the backend lives — verify links point here so /api/auth/verify can
# run and then 307 the user back to APP_URL/app/?verified=1.
BACKEND_URL    = os.environ.get(
    "BACKEND_URL",
    "https://reelcrate-backend-production.up.railway.app",
).rstrip("/")


def send_email(to: str, subject: str, html: str, text: Optional[str] = None) -> bool:
    """Send an email via Resend. Returns True on success. Never throws on network errors."""
    if not RESEND_API_KEY:
        print(f"[email] (no RESEND_API_KEY — would send) to={to} subject={subject!r}")
        return True

    body = {
        "from":    FROM_ADDRESS,
        "to":      [to],
        "subject": subject,
        "html":    html,
    }
    if text:
        body["text"] = text

    req = urllib.request.Request(
        RESEND_URL,
        data=json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {RESEND_API_KEY}",
            "Content-Type":  "application/json",
            # Cloudflare in front of api.resend.com rejects the default Python
            # urllib User-Agent (Cloudflare error 1010). Set a real-looking one.
            "User-Agent":    "Reelcrate/1.0 (+https://reelcrate.app)",
            "Accept":        "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            ok = 200 <= resp.status < 300
            if not ok:
                print(f"[email] non-2xx from Resend: {resp.status}")
            return ok
    except urllib.error.HTTPError as e:
        try:
            err_body = e.read().decode()[:400]
        except Exception:
            err_body = ""
        print(f"[email] HTTPError {e.code}: {err_body}")
        return False
    except Exception as e:
        print(f"[email] send failed: {type(e).__name__}: {e}")
        return False


# ----- ready-made templates --------------------------------------------------

def _wrap(title_html: str, body_html: str, cta_html: str = "") -> str:
    return f"""<!doctype html>
<html><body style="margin:0;padding:0;background:#0a0a0a;font-family:-apple-system,BlinkMacSystemFont,Arial,sans-serif;color:#fff">
<table width="100%" cellpadding="0" cellspacing="0" style="background:#0a0a0a"><tr><td align="center">
  <table width="520" cellpadding="0" cellspacing="0" style="margin:30px auto;background:#111;border-radius:16px;overflow:hidden">
    <tr><td style="padding:34px 36px 8px 36px">
      <div style="font-size:14px;color:#f5c518;font-weight:900;letter-spacing:.18em">REELCRATE.</div>
    </td></tr>
    <tr><td style="padding:8px 36px 16px 36px">
      <div style="font-size:26px;font-weight:900;line-height:1.15">{title_html}</div>
    </td></tr>
    <tr><td style="padding:0 36px 24px 36px;font-size:15px;color:#cfcfcf;line-height:1.55">
      {body_html}
    </td></tr>
    {cta_html and f'<tr><td style="padding:0 36px 28px 36px" align="left">{cta_html}</td></tr>' or ''}
    <tr><td style="padding:18px 36px 30px 36px;border-top:1px solid #222;font-size:11px;color:#666">
      Reelcrate · DJ set → ready-to-post clips · <a href="{APP_URL}" style="color:#f5c518;text-decoration:none">reelcrate.app</a>
    </td></tr>
  </table>
</td></tr></table>
</body></html>"""


def _btn(label: str, href: str) -> str:
    return (f'<a href="{href}" style="display:inline-block;background:#f5c518;color:#000;'
            f'font-weight:900;font-size:14px;text-decoration:none;padding:14px 22px;'
            f'border-radius:10px;letter-spacing:.06em">{label}</a>')


def send_verify_email(to: str, name: str, token: str) -> bool:
    # Link goes to the backend — it marks the account verified, then 307-redirects
    # the browser to APP_URL/app/?verified=1 which the frontend renders as ✓.
    link = f"{BACKEND_URL}/api/auth/verify?token={token}"
    title = f"Verify your email to start uploading"
    body = (f'Hey {name or "DJ"} —<br><br>'
            f'Tap the button below to verify {to} and unlock uploads. '
            f'The link is good for 24 hours.<br><br>'
            f'<span style="color:#888;font-size:12px">If you didn\'t create a Reelcrate account, you can ignore this.</span>')
    return send_email(to, "Verify your Reelcrate email",
                      _wrap(title, body, _btn("Verify email →", link)))


def send_reset_email(to: str, name: str, token: str) -> bool:
    link = f"{APP_URL}/app/?reset={token}"
    title = "Reset your Reelcrate password"
    body = (f'Hey {name or "DJ"} —<br><br>'
            f'Tap below to pick a new password. The link is good for 1 hour.<br><br>'
            f'<span style="color:#888;font-size:12px">Didn\'t request this? Someone may have typed your email by mistake. You can ignore this — your password won\'t change.</span>')
    return send_email(to, "Reset your Reelcrate password",
                      _wrap(title, body, _btn("Reset password →", link)))


def send_waitlist_welcome(to: str, name: str = "") -> bool:
    """Sent to DJs who submit the email form on the landing page. The app is
    now live and taking cards — so this isn't really a waitlist anymore, it's
    a delivery mechanism for the FOUNDER40 code + a nudge to try the app.
    Kept the function name for backwards compatibility with existing callers."""
    title = "Your Reelcrate founder code is inside."
    body = (f'Hey {name or "DJ"} —<br><br>'
            f'Reelcrate is live and you\'re one of the first 40 DJs eligible for the '
            f'founder deal: <b>$9.50/mo forever</b> (half off the regular $19/mo, locked in for life).<br><br>'
            f'<b>Your code:</b> '
            f'<span style="display:inline-block;background:#f5c518;color:#000;padding:6px 12px;'
            f'border-radius:6px;font-weight:900;letter-spacing:.08em;font-size:15px">FOUNDER40</span><br><br>'
            f'<b>How to redeem:</b><br>'
            f'&nbsp;1. Tap the button below → sign up with this email<br>'
            f'&nbsp;2. Start your 14-day free trial (no card charged for 14 days)<br>'
            f'&nbsp;3. At Stripe checkout, enter <b>FOUNDER40</b> in the promo field<br>'
            f'&nbsp;4. You\'re locked in at $9.50/mo for life<br><br>'
            f'<b>What Reelcrate does:</b> drop a set (audio or video, up to 4 hrs), pick a BPM range and a visualizer, '
            f'and we cut it into 5 ready-to-post 9:16 clips with captions baked in — Reels, TikTok, Shorts. About 3 minutes end-to-end.<br><br>'
            f'<b>Heads up:</b> only 40 founder spots. Once they\'re gone, the code expires and it\'s regular pricing from there.<br><br>'
            f'— DJ EZ1 (founder)')
    return send_email(to, "Your Reelcrate founder code (FOUNDER40) is inside",
                      _wrap(title, body, _btn("Claim your spot →", APP_URL + "/app/?promo=FOUNDER40")))


def send_subscription_welcome(to: str, name: str = "", plan_label: str = "",
                              trial_end_epoch: int = 0, is_trialing: bool = True) -> bool:
    """Sent the first time a user transitions to trialing/active on Stripe.
    Confirms the trial started + tells them the next billing date."""
    import datetime as _dt
    next_bill = ""
    if trial_end_epoch:
        try:
            d = _dt.datetime.utcfromtimestamp(int(trial_end_epoch))
            next_bill = d.strftime("%b %d, %Y")
        except Exception:
            next_bill = ""

    plan = plan_label or "Reelcrate"
    title = "You're in — trial started" if is_trialing else "You're subscribed"
    parts = [f'Hey {name or "DJ"} —<br><br>',
             (f'Your <b>{plan}</b> free trial is live. '
              f'You have full access to Reelcrate right now — upload sets, '
              f'pick BPM ranges, get 5 ready-to-post clips in ~3 minutes.<br><br>'
              if is_trialing else
              f'Your <b>{plan}</b> subscription is active. Thanks for backing Reelcrate.<br><br>')]
    if is_trialing and next_bill:
        parts.append(f'<b>First bill:</b> {next_bill} — you can cancel any time before then '
                     f'from Settings → Manage subscription and you won\'t be charged.<br><br>')
    parts.append(
        '<b>What to do first:</b><br>'
        '&nbsp;• Sign in at <a href="' + APP_URL + '/app/" style="color:#f5c518">reelcrate.app/app</a><br>'
        '&nbsp;• Upload your most recent set (video or audio, up to 4 hrs)<br>'
        '&nbsp;• Pick a BPM range that matches the section you want to clip<br>'
        '&nbsp;• Post the clips to Reels / TikTok / Shorts<br><br>'
        '— DJ EZ1 (founder)'
    )
    body = "".join(parts)
    subject = f"Your Reelcrate trial is live" if is_trialing else "Welcome to Reelcrate"
    return send_email(to, subject,
                      _wrap(title, body, _btn("Open Reelcrate →", APP_URL + "/app/")))


def send_waitlist_alert(to_admin: str, submitter_email: str, name: str = "") -> bool:
    """Alerts the founder inbox that a new DJ joined the waitlist."""
    title = "New waitlist signup"
    body = (f'<b>{submitter_email}</b>'
            + (f' ({name})' if name else '')
            + f'<br><br>Sent from the CLAIM SPOT form on reelcrate.app.')
    return send_email(to_admin, f"[Reelcrate] Waitlist: {submitter_email}",
                      _wrap(title, body))
