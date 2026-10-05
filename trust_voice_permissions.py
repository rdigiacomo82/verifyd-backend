# ============================================================
# VeriFYD Trust Voice — Trust Circle Permissions
# VERIFYD_TRUST_VOICE_TRUST_CIRCLE_PERMISSIONS_V1
#
# Server-side Trust Circle permission foundation.
#
# IMPORTANT:
# - Does not modify existing VeriFYD tables.
# - Creates only the isolated Trust Circle membership table.
# - Does not affect calling unless the signaling layer invokes it.
# - Disabled unless VERIFYD_TRUST_VOICE_TRUST_CIRCLE_ENABLED=1.
# ============================================================

from __future__ import annotations

import os
from typing import Tuple

from database import get_db


ALLOWED_CALL_PRIVACY = {
    "anyone",
    "verified_users",
    "trust_circle",
    "nobody",
}


def trust_circle_enabled() -> bool:
    value = (
        os.environ.get(
            "VERIFYD_TRUST_VOICE_TRUST_CIRCLE_ENABLED",
            "",
        )
        or ""
    ).strip().lower()

    return value in {
        "1",
        "true",
        "yes",
        "on",
    }


def ensure_trust_circle_schema() -> None:
    """
    Create the isolated Trust Circle membership table.

    One row means:

        member_identity_id

    is a member of:

        owner_identity_id's Trust Circle

    Safe and idempotent.
    """

    with get_db() as conn:
        cur = conn.cursor()

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS verifyd_trust_circle_members (
                owner_identity_id  TEXT NOT NULL,
                member_identity_id TEXT NOT NULL,

                display_label      TEXT NOT NULL DEFAULT '',
                relationship_type  TEXT NOT NULL DEFAULT 'custom',
                contact_type       TEXT NOT NULL DEFAULT 'person',
                description        TEXT NOT NULL DEFAULT '',

                image_source       TEXT NOT NULL DEFAULT 'profile',
                builtin_image_key  TEXT NOT NULL DEFAULT '',

                sort_order         INTEGER NOT NULL DEFAULT 0,

                created_at         TEXT NOT NULL,
                updated_at         TEXT NOT NULL,

                PRIMARY KEY (
                    owner_identity_id,
                    member_identity_id
                )
            )
            """
        )

        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_verifyd_trust_circle_owner
            ON verifyd_trust_circle_members (
                owner_identity_id,
                sort_order
            )
            """
        )

        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_verifyd_trust_circle_member
            ON verifyd_trust_circle_members (
                member_identity_id
            )
            """
        )


def trust_circle_contains(
    owner_identity_id: str,
    member_identity_id: str,
) -> bool:
    """
    Return True when member_identity_id belongs to the owner's circle.
    """

    if not owner_identity_id or not member_identity_id:
        return False

    ensure_trust_circle_schema()

    with get_db() as conn:
        cur = conn.cursor()

        cur.execute(
            """
            SELECT 1
            FROM verifyd_trust_circle_members
            WHERE owner_identity_id = %s
              AND member_identity_id = %s
            LIMIT 1
            """,
            (
                owner_identity_id,
                member_identity_id,
            ),
        )

        return bool(cur.fetchone())


def call_permission_decision(
    caller: dict,
    callee: dict,
) -> Tuple[bool, str]:
    """
    Determine whether an authenticated Trust Voice caller is permitted
    to cause an incoming call to the target identity.

    Returns:

        (allowed, reason)

    The reason is intended for backend diagnostics.

    The caller-facing UI should receive only a generic "not_allowed"
    result so private account settings are not exposed.
    """

    # Preserve existing Trust Voice behavior until we deliberately enable
    # Trust Circle enforcement in Render.
    if not trust_circle_enabled():
        return True, "feature_disabled"

    privacy = (
        callee.get("call_privacy")
        or "verified_users"
    ).strip().lower()

    # Safe backwards-compatible fallback.
    if privacy not in ALLOWED_CALL_PRIVACY:
        privacy = "verified_users"

    # Any authenticated Trust Voice identity may call.
    if privacy == "anyone":
        return True, "allowed_anyone"

    # No incoming calls.
    if privacy == "nobody":
        return False, "blocked_by_privacy"

    # Caller must have an email-verified VeriFYD identity.
    if privacy == "verified_users":
        allowed = bool(
            caller.get("email_verified")
        )

        return (
            allowed,
            "verified_user_required",
        )

    # Caller must explicitly belong to the recipient's Trust Circle.
    if privacy == "trust_circle":
        allowed = trust_circle_contains(
            str(
                callee.get("id")
                or ""
            ),
            str(
                caller.get("id")
                or ""
            ),
        )

        return (
            allowed,
            "trust_circle_required",
        )

    return False, "blocked_by_privacy"