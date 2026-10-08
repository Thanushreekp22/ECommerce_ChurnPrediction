"""
backend/notifications.py

Outreach side of the retention loop.

When a retention strategy is APPLIED to a customer, this module lets the product
actually reach out to that customer:

  * Email  -> real SMTP delivery when CHURNIQ_SMTP_* env vars are configured,
              otherwise "demo mode": the message is written to an outbox JSON
              (artifacts/notifications_outbox.json) so the whole flow can be
              demonstrated with zero credentials.
  * WhatsApp -> a wa.me deep link with the strategy text pre-filled. Opening it
              on a phone with WhatsApp installed composes the message for you
              (WhatsApp Business API requires a paid account, so a link is the
              practical free path for a demo).

The module never raises on delivery problems: outreach is best-effort so the
core "apply strategy" flow keeps working even if a mail server is down.
"""

from __future__ import annotations

import json
import smtplib
import urllib.parse
from datetime import datetime, timezone
from email.mime.text import MIMEText
from utils.config import (
    SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, SMTP_FROM, SMTP_USE_TLS,
    NOTIFICATIONS_OUTBOX, smtp_configured,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def looks_like_email(value: str) -> bool:
    return "@" in str(value or "") and "." in str(value or "").split("@")[-1]


def customer_email(customer_id: str) -> str | None:
    """Use a customer identifier as an email when it has email shape."""
    cid = str(customer_id or "").strip()
    if looks_like_email(cid):
        return cid
    return None


def build_whatsapp_link(phone: str, text: str) -> str:
    """wa.me deep link — opens WhatsApp with the message pre-filled."""
    phone = "".join(ch for ch in str(phone or "") if ch.isdigit())
    if not phone:
        return ""
    return "https://wa.me/{}?text={}".format(phone, urllib.parse.quote(text))


def _record_outbox(channel: str, recipient: str, subject: str, body: str, ok: bool, note: str) -> None:
    """Demo-mode record of an outgoing message (no SMTP required)."""
    try:
        entry = {
            "channel": channel, "recipient": recipient, "subject": subject,
            "body": body, "ok": ok, "note": note, "sent_at": _now(),
        }
        records = []
        if NOTIFICATIONS_OUTBOX.exists():
            try:
                records = json.loads(NOTIFICATIONS_OUTBOX.read_text(encoding="utf-8"))
            except Exception:
                records = []
        records.insert(0, entry)
        NOTIFICATIONS_OUTBOX.write_text(json.dumps(records[:200], indent=2), encoding="utf-8")
    except Exception:
        pass


def send_email(to: str, subject: str, body: str) -> dict:
    """
    Send an HTML-free plain-text email over SMTP.

    Falls back to demo mode (outbox log) when SMTP is not configured.
    """
    if not to:
        return {"sent": False, "delivered": False, "channel": "email", "status": "unavailable", "error": "No email address available."}

    if not smtp_configured():
        _record_outbox("email", to, subject, body, True, "demo-mode (SMTP not configured — message logged to outbox)")
        return {
            "sent": False, "delivered": False, "channel": "email", "mode": "demo",
            "status": "demo_logged",
            "note": "SMTP is not configured; the message was only logged locally.",
        }

    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = SMTP_FROM or SMTP_USER
    msg["To"] = to
    try:
        if SMTP_PORT == 465:
            server = smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, timeout=30)
        else:
            server = smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30)
            if SMTP_USE_TLS:
                server.starttls()
        try:
            server.login(SMTP_USER, SMTP_PASS)
            server.send_message(msg)
        finally:
            try:
                server.quit()
            except Exception:
                pass
        _record_outbox("email", to, subject, body, True, f"delivered via SMTP as {SMTP_FROM or SMTP_USER}")
        return {"sent": True, "delivered": True, "channel": "email", "mode": "smtp", "status": "delivered", "to": to}
    except Exception as exc:
        _record_outbox("email", to, subject, body, False, f"SMTP delivery failed: {exc}")
        return {"sent": False, "delivered": False, "channel": "email", "mode": "smtp", "status": "failed", "error": str(exc)}


def notify_retention(
    customer_id: str,
    strategy_key: str,
    action_text: str,
    channel: str = "email",
    email: str | None = None,
    phone: str | None = None,
) -> dict:
    """
    Send one retention-strategy outreach to a customer.

    channel: "email" | "whatsapp" | "both"
    Returns a structured result the frontend can render (wa_link included).
    """
    recipient = email if looks_like_email(email or "") else customer_email(customer_id)
    result: dict = {
        "customer_id": customer_id,
        "strategy_key": strategy_key,
        "channel": channel,
        "sent": False,
        "delivered": False,
        "messages": [],
    }

    subject = "Your personalised retention offer — {}".format(strategy_key.replace("_", " ").title())
    body = action_text or "We value you as a customer and would love to keep you. Reply to get a special offer."
    body = "Hi there!\n\n" + body + "\n\n— ChurnIQ Retention Team"

    channels = ["email", "whatsapp"] if channel == "both" else [channel]

    for ch in channels:
        if ch == "email":
            if not recipient:
                result["messages"].append({"channel": "email", "sent": False, "delivered": False, "status": "unavailable", "error": "No email address available."})
                continue
            res = send_email(recipient, subject, body)
            result["messages"].append({"channel": "email", "sent": res.get("sent"), "delivered": res.get("delivered", False), "status": res.get("status"), "mode": res.get("mode"),
                                       "to": recipient, "error": res.get("error"), "note": res.get("note")})
            if res.get("delivered"):
                result["sent"] = True
                result["delivered"] = True
                result.setdefault("recipients", []).append({"channel": "email", "to": recipient})
        elif ch == "whatsapp":
            link = build_whatsapp_link(phone or recipient, body)
            if link:
                _record_outbox("whatsapp", phone or recipient, "WhatsApp outreach", body, True,
                               "wa.me link generated (opens WhatsApp with the message pre-filled)")
            result["messages"].append({"channel": "whatsapp", "sent": False, "delivered": False, "status": "link_ready" if link else "unavailable",
                                       "wa_link": link or None,
                                       "error": None if link else "No phone number available."})
            if link:
                result["wa_link"] = link

    return result