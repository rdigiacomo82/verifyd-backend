"""Read-only VeriFYD Circle/Trust Voice administrative reporting.

Attach to the existing /trust-voice identity router. This module does not touch
message bodies, attachments, or device-local accessibility/activity settings.
"""
from __future__ import annotations

import hmac
import os
from datetime import datetime, timezone

from fastapi import APIRouter, Header, HTTPException, Response
from database import get_db

router = APIRouter(tags=["Trust Voice Admin"])

# All figures are aggregates of existing database records. No tables are created.
_REPORT_SQL = """
SELECT
    i.id, i.email_lower AS email, i.handle, i.display_name, i.status, i.created_at,
    COALESCE(s.login_sessions, 0) AS login_sessions,
    s.last_login_at, s.last_session_activity,
    COALESCE(m.messages_sent, 0) AS messages_sent,
    COALESCE(c.calls_placed, 0) AS calls_placed,
    COALESCE(r.calls_received, 0) AS calls_received,
    COALESCE(g.group_videos_started, 0) AS group_videos_started
FROM verifyd_identities AS i
LEFT JOIN (
    SELECT identity_id,
           COUNT(*) AS login_sessions,
           MAX(created_at) AS last_login_at,
           MAX(COALESCE(last_seen_at, created_at)) AS last_session_activity
    FROM verifyd_voice_sessions
    GROUP BY identity_id
) AS s ON s.identity_id = i.id
LEFT JOIN (
    SELECT sender_identity_id, COUNT(*) AS messages_sent
    FROM verifyd_voice_messages
    GROUP BY sender_identity_id
) AS m ON m.sender_identity_id = i.id
LEFT JOIN (
    SELECT caller_identity_id, COUNT(*) AS calls_placed
    FROM verifyd_voice_calls
    GROUP BY caller_identity_id
) AS c ON c.caller_identity_id = i.id
LEFT JOIN (
    SELECT callee_identity_id, COUNT(*) AS calls_received
    FROM verifyd_voice_calls
    GROUP BY callee_identity_id
) AS r ON r.callee_identity_id = i.id
LEFT JOIN (
    SELECT started_by_identity_id, COUNT(*) AS group_videos_started
    FROM verifyd_voice_group_video_sessions
    GROUP BY started_by_identity_id
) AS g ON g.started_by_identity_id = i.id
ORDER BY i.created_at DESC
"""


def _require_admin(supplied: str) -> None:
    """Match existing ADMIN_KEY, but never accept a missing key or a URL query key."""
    configured = (os.environ.get("ADMIN_KEY") or "").strip()
    provided = (supplied or "").strip()
    if not configured or not provided or not hmac.compare_digest(configured, provided):
        raise HTTPException(status_code=401, detail="unauthorized")


@router.get("/admin/overview")
def trust_voice_admin_overview(
    response: Response,
    x_verifyd_admin_key: str = Header(default="", alias="X-VeriFYD-Admin-Key"),
):
    """Administrative identity/usage counts only. Excludes private communication content."""
    _require_admin(x_verifyd_admin_key)
    response.headers["Cache-Control"] = "no-store, private"
    response.headers["Pragma"] = "no-cache"

    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(_REPORT_SQL)
        users = [dict(row) for row in cur.fetchall()]

    summary = {
        "registered_accounts": len(users),
        "active_accounts": sum(1 for u in users if u["status"] == "active"),
        "accounts_with_login_history": sum(1 for u in users if u["login_sessions"] > 0),
        "login_sessions_created": sum(int(u["login_sessions"]) for u in users),
        "messages_sent": sum(int(u["messages_sent"]) for u in users),
        "call_records": sum(int(u["calls_placed"]) for u in users),
        "group_video_sessions_started": sum(int(u["group_videos_started"]) for u in users),
    }
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "users": users,
        "notes": {
            "login_sessions_created": "Cumulative sessions, not simultaneous online users.",
            "call_records": "Calls initiated, including unanswered and failed attempts.",
            "messages_sent": "Message records, not private message content.",
            "group_video_sessions_started": "Sessions initiated, not necessarily completed.",
            "activity_progress": "Device-local activities are not reported here.",
        },
    }
