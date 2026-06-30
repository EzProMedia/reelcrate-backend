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
    link = f"{APP_URL}/app/?verify={token}"
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
