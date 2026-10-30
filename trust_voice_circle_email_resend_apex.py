"""Validate an apex Vfvid envelope recipient returned by Resend's receiving API.

Never use the visible To header of a forwarded message. Only the actual
recipient array on the authenticated Resend receiving API response is eligible,
and only after an explicit, separate apex cutover gate is enabled.
"""
from __future__ import annotations

import os
import re
from email.utils import getaddresses

_HANDLE = re.compile(r"^[a-z][a-z0-9_]{2,23}$")
_RESERVED = frozenset({
    "reply", "admin", "administrator", "system", "postmaster", "abuse",
    "support", "security", "billing", "sales", "help", "info", "contact",
    "noreply", "no-reply", "mail", "team", "staff", "hostmaster", "webmaster",
    "privacy", "legal", "dmarc", "dkim", "spf", "bounce", "bounces",
})


def _on(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in ("1", "true", "yes", "on")


def resend_apex_enabled() -> bool:
    """Opt-in receiving route, independent from Hostinger polling and replies."""
    return (_on("VERIFYD_CIRCLE_EMAIL_RESEND_APEX_ENABLED")
            and _on("VERIFYD_CIRCLE_EMAIL_PUBLIC_HANDLE_ENABLED"))


def resend_apex_recipient_handle(to_list) -> str | None:
    """Get exactly one eligible public handle from Resend's canonical To list.

    A multi-recipient event is not split here; avoid ambiguously delivering
    untrusted mail to any registered user. Operational names are never routed.
    """
    if not resend_apex_enabled() or not isinstance(to_list, list) or len(to_list) != 1:
        return None
    address_raw = to_list[0]
    if not isinstance(address_raw, str) or len(address_raw) > 254:
        return None
    parsed = getaddresses([address_raw])
    if len(parsed) != 1:
        return None
    address = parsed[0][1].strip().lower()
    if not address.endswith("@vfvid.com"):
        return None
    handle = address[:-len("@vfvid.com")]
    if not _HANDLE.fullmatch(handle):
        return None
    custom = set()
    for candidate in os.getenv("VERIFYD_CIRCLE_EMAIL_RESERVED_LOCALPARTS", "").split(","):
        name = candidate.strip().lower()
        if _HANDLE.fullmatch(name):
            custom.add(name)
    if handle in (_RESERVED | custom):
        return None
    return handle
