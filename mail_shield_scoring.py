from __future__ import annotations

from pathlib import Path
from typing import Any

from mail_shield_header_tools import (
    detect_authentication_risk,
    extract_domain_from_email,
    has_lookalike_domain,
    looks_like_brand_spoof,
)
from mail_shield_url_tools import (
    domain_distance_hint,
    extract_urls,
    is_punycode_or_suspicious_tld,
    is_shortened_url,
    normalize_domain,
)


EXECUTABLE_EXTENSIONS = {
    ".exe", ".scr", ".bat", ".cmd", ".js", ".vbs", ".ps1", ".msi", ".com", ".jar", ".hta"
}
MACRO_OFFICE_EXTENSIONS = {".docm", ".xlsm", ".pptm"}
ARCHIVE_EXTENSIONS = {".zip", ".rar", ".7z"}

HIGH_RISK_PHRASES = [
    "wire transfer", "gift card", "verify your account", "account locked",
    "update banking", "new bank account", "ach change", "unusual login",
    "password reset", "immediate action", "final notice"
]

MEDIUM_RISK_PHRASES = [
    "urgent payment", "invoice overdue", "past due", "click here",
    "login to confirm", "update payment", "payment information",
    "bank details", "confirm your identity"
]


def _level_from_penalty(penalty: int) -> str:
    if penalty >= 25:
        return "HIGH"
    if penalty > 0:
        return "REVIEW"
    return "LOW"


def _add_reason(reasons: list[str], reason: str) -> None:
    if reason and reason not in reasons and len(reasons) < 5:
        reasons.append(reason)


def _attachment_penalty(attachments: list[Any]) -> tuple[int, str, list[str]]:
    penalty = 0
    reasons: list[str] = []

    for att in attachments or []:
        filename = ""
        size_bytes = None

        if isinstance(att, dict):
            filename = str(att.get("filename") or "")
            size_bytes = att.get("size_bytes")
        else:
            filename = str(getattr(att, "filename", "") or "")
            size_bytes = getattr(att, "size_bytes", None)

        lower = filename.lower().strip()
        suffixes = [s.lower() for s in Path(lower).suffixes]
        ext = suffixes[-1] if suffixes else ""

        if len(suffixes) >= 2 and suffixes[-2] in {".pdf", ".doc", ".docx", ".xls", ".xlsx", ".jpg", ".jpeg", ".png"} and suffixes[-1] in EXECUTABLE_EXTENSIONS:
            penalty += 35
            _add_reason(reasons, f"Attachment has a dangerous double extension: {filename}")
        elif ext in EXECUTABLE_EXTENSIONS:
            penalty += 35
            _add_reason(reasons, f"Attachment type is high risk: {ext}")
        elif ext in MACRO_OFFICE_EXTENSIONS:
            penalty += 25
            _add_reason(reasons, f"Macro-enabled Office attachment detected: {filename}")
        elif ext in ARCHIVE_EXTENSIONS:
            penalty += 10
            _add_reason(reasons, f"Archive attachment should be reviewed: {filename}")

        try:
            if size_bytes and int(size_bytes) > 25 * 1024 * 1024:
                _add_reason(reasons, f"Large attachment detected: {filename}")
        except Exception:
            pass

    return min(penalty, 60), _level_from_penalty(penalty), reasons


def _content_penalty(subject: str, body_text: str) -> tuple[int, str, list[str]]:
    text = f"{subject or ''}\n{body_text or ''}".lower()
    penalty = 0
    hits: list[str] = []

    for phrase in HIGH_RISK_PHRASES:
        if phrase in text:
            penalty += 10
            hits.append(phrase)

    for phrase in MEDIUM_RISK_PHRASES:
        if phrase in text:
            penalty += 5
            hits.append(phrase)

    penalty = min(penalty, 35)
    reasons = []
    if hits:
        reasons.append("Message contains payment, login, urgency, or account-risk language")

    return penalty, _level_from_penalty(penalty), reasons


def _sender_penalty(from_email: str, from_name: str, reply_to: str) -> tuple[int, str, list[str], str]:
    penalty = 0
    reasons: list[str] = []
    from_domain = extract_domain_from_email(from_email)
    reply_domain = extract_domain_from_email(reply_to)

    if not from_email or not from_domain:
        penalty += 20
        _add_reason(reasons, "Sender email is missing or invalid")

    if reply_domain and from_domain and reply_domain != from_domain:
        penalty += 15
        _add_reason(reasons, "Reply-To domain differs from sender domain")

    if from_domain and looks_like_brand_spoof(from_name, from_domain):
        penalty += 20
        _add_reason(reasons, "Display name may not match sender domain")

    if from_domain and has_lookalike_domain(from_domain):
        penalty += 15
        _add_reason(reasons, "Sender domain has lookalike or suspicious patterns")

    return min(penalty, 60), _level_from_penalty(penalty), reasons, from_domain


def _link_penalty(links: list[str], body_text: str, body_html: str, sender_domain: str, subject: str) -> tuple[int, str, list[str]]:
    urls = list(links or [])
    if not urls:
        urls = extract_urls(body_text or "", body_html or "")

    penalty = 0
    reasons: list[str] = []

    if len(urls) > 10:
        penalty += 8
        _add_reason(reasons, "Message contains many links")

    risky_link_count = 0
    for url in urls:
        host = normalize_domain(url)
        if not host:
            continue

        if is_shortened_url(url):
            penalty += 10
            risky_link_count += 1
        if is_punycode_or_suspicious_tld(url):
            penalty += 15
            risky_link_count += 1

        intent_text = f"{subject or ''} {body_text or ''}".lower()
        asks_login_or_payment = any(x in intent_text for x in ["login", "password", "payment", "bank", "verify", "account"])
        if asks_login_or_payment and domain_distance_hint(sender_domain, host):
            penalty += 15
            risky_link_count += 1

    if risky_link_count:
        _add_reason(reasons, "One or more links appear shortened, unusual, or unrelated to the sender")

    return min(penalty, 60), _level_from_penalty(penalty), reasons


def score_mail_shield(payload: Any) -> dict[str, Any]:
    from_email = getattr(payload, "from_email", None) or ""
    from_name = getattr(payload, "from_name", None) or ""
    reply_to = getattr(payload, "reply_to", None) or ""
    subject = getattr(payload, "subject", None) or ""
    body_text = getattr(payload, "body_text", None) or ""
    body_html = getattr(payload, "body_html", None) or ""
    headers = getattr(payload, "headers", None) or {}
    links = getattr(payload, "links", None) or []
    attachments = getattr(payload, "attachments", None) or []

    reasons: list[str] = []

    sender_penalty, sender_level, sender_reasons, sender_domain = _sender_penalty(from_email, from_name, reply_to)
    auth_penalty, auth_level, auth_reasons = detect_authentication_risk(headers)
    link_penalty, link_level, link_reasons = _link_penalty(links, body_text, body_html, sender_domain, subject)
    attachment_penalty, attachment_level, attachment_reasons = _attachment_penalty(attachments)
    content_penalty, content_level, content_reasons = _content_penalty(subject, body_text)

    total_penalty = sender_penalty + auth_penalty + link_penalty + attachment_penalty + content_penalty
    trust_score = max(0, min(100, 100 - total_penalty))

    if trust_score >= 80:
        risk_level = "LOW"
        color = "GREEN"
        label = "Low Risk"
        recommended_action = "No immediate action required"
    elif trust_score >= 55:
        risk_level = "REVIEW"
        color = "YELLOW"
        label = "Review"
        recommended_action = "Review before trusting this email"
    else:
        risk_level = "HIGH"
        color = "RED"
        label = "High Risk"
        recommended_action = "Verify further before opening links or attachments"

    for group in [sender_reasons, auth_reasons, link_reasons, attachment_reasons, content_reasons]:
        for reason in group:
            _add_reason(reasons, reason)

    if not reasons:
        reasons.append("No obvious sender, link, attachment, or content risk indicators found")

    return {
        "trust_score": int(trust_score),
        "risk_level": risk_level,
        "color": color,
        "label": label,
        "recommended_action": recommended_action,
        "signals": {
            "sender": sender_level,
            "authentication": auth_level,
            "links": link_level,
            "attachments": attachment_level,
            "content": content_level,
        },
        "reasons": reasons[:5],
        "verify_further_available": True,
    }