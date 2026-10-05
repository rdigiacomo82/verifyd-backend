# ============================================================
# VeriFYD Trust Voice — Trust Circle API
# VERIFYD_TRUST_VOICE_TRUST_CIRCLE_API_V1
#
# Authenticated API for:
#   - Trust Circle members
#   - calling privacy
#   - relationship / role catalog
#
# IMPORTANT:
# - Uses existing Trust Voice session token.
# - Uses existing verifyd_identities table without altering schema.
# - Uses isolated verifyd_trust_circle_members table.
# - Does not expose email addresses.
# - Generic role images are represented by keys only.
# ============================================================

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

from database import get_db
from trust_voice_permissions import (
    ALLOWED_CALL_PRIVACY,
    ensure_trust_circle_schema,
    trust_circle_enabled,
)
from trust_voice_signaling import (
    _identity_by_handle,
    _identity_by_id,
    _normalize_handle,
    _public_identity,
    _verify_token,
)


router = APIRouter(
    prefix="/trust-voice",
    tags=["Trust Voice Trust Circle"],
)

FEATURE_VERSION = "0.1.0"


ROLE_CATALOG = {
    "mom": {
        "label": "Mom",
        "contact_type": "person",
        "builtin_image_key": "family_mom",
    },
    "dad": {
        "label": "Dad",
        "contact_type": "person",
        "builtin_image_key": "family_dad",
    },
    "parent": {
        "label": "Parent",
        "contact_type": "person",
        "builtin_image_key": "family_parent",
    },
    "grandparent": {
        "label": "Grandparent",
        "contact_type": "person",
        "builtin_image_key": "family_grandparent",
    },
    "sibling": {
        "label": "Sibling",
        "contact_type": "person",
        "builtin_image_key": "family_sibling",
    },
    "family": {
        "label": "Family",
        "contact_type": "person",
        "builtin_image_key": "family_other",
    },
    "friend": {
        "label": "Friend",
        "contact_type": "person",
        "builtin_image_key": "friend",
    },
    "neighbor": {
        "label": "Neighbor",
        "contact_type": "person",
        "builtin_image_key": "neighbor",
    },
    "doctor": {
        "label": "Doctor",
        "contact_type": "person",
        "builtin_image_key": "care_doctor",
    },
    "nurse": {
        "label": "Nurse",
        "contact_type": "person",
        "builtin_image_key": "care_nurse",
    },
    "speech_therapist": {
        "label": "Speech Therapist",
        "contact_type": "person",
        "builtin_image_key": "care_speech_therapist",
    },
    "physical_therapist": {
        "label": "Physical Therapist",
        "contact_type": "person",
        "builtin_image_key": "care_physical_therapist",
    },
    "occupational_therapist": {
        "label": "Occupational Therapist",
        "contact_type": "person",
        "builtin_image_key": "care_occupational_therapist",
    },
    "caregiver": {
        "label": "Caregiver",
        "contact_type": "person",
        "builtin_image_key": "care_caregiver",
    },
    "teacher": {
        "label": "Teacher",
        "contact_type": "person",
        "builtin_image_key": "school_teacher",
    },
    "pharmacy": {
        "label": "Pharmacy",
        "contact_type": "organization",
        "builtin_image_key": "service_pharmacy",
    },
    "school": {
        "label": "School",
        "contact_type": "organization",
        "builtin_image_key": "service_school",
    },
    "medical_office": {
        "label": "Medical Office",
        "contact_type": "organization",
        "builtin_image_key": "service_medical_office",
    },
    "emergency_contact": {
        "label": "Emergency Contact",
        "contact_type": "person",
        "builtin_image_key": "emergency_contact",
    },
    "custom": {
        "label": "Other",
        "contact_type": "person",
        "builtin_image_key": "generic_person",
    },
}


class TrustCircleMemberCreate(BaseModel):
    handle: str = Field(min_length=3, max_length=25)

    display_label: str = Field(
        default="",
        max_length=80,
    )

    relationship_type: str = Field(
        default="custom",
        max_length=50,
    )

    description: str = Field(
        default="",
        max_length=160,
    )

    image_source: Literal[
        "profile",
        "builtin",
    ] = "profile"


class TrustCircleMemberUpdate(BaseModel):
    display_label: Optional[str] = Field(
        default=None,
        max_length=80,
    )

    relationship_type: Optional[str] = Field(
        default=None,
        max_length=50,
    )

    description: Optional[str] = Field(
        default=None,
        max_length=160,
    )

    image_source: Optional[
        Literal[
            "profile",
            "builtin",
        ]
    ] = None

    sort_order: Optional[int] = Field(
        default=None,
        ge=0,
        le=10000,
    )


class CallSettingsUpdate(BaseModel):
    call_privacy: Literal[
        "anyone",
        "verified_users",
        "trust_circle",
        "nobody",
    ]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean_short_text(
    value: str,
    max_length: int,
) -> str:
    cleaned = " ".join(
        (value or "").strip().split()
    )

    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length].rstrip()

    return cleaned


def _identity_from_bearer(
    authorization: Optional[str],
) -> dict:
    if (
        not authorization
        or not authorization.lower().startswith("bearer ")
    ):
        raise HTTPException(
            status_code=401,
            detail={
                "error": "session_required",
            },
        )

    token = authorization.split(
        " ",
        1,
    )[1].strip()

    if not token:
        raise HTTPException(
            status_code=401,
            detail={
                "error": "session_required",
            },
        )

    payload = _verify_token(token)

    identity = _identity_by_id(
        payload.get(
            "identity_id",
            "",
        )
    )

    if not identity:
        raise HTTPException(
            status_code=401,
            detail={
                "error": "identity_not_found",
            },
        )

    return identity


def _validate_role(
    relationship_type: str,
) -> dict:
    role = (
        relationship_type
        or "custom"
    ).strip().lower()

    if role not in ROLE_CATALOG:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_relationship_type",
                "message": "Choose a supported Trust Circle relationship.",
            },
        )

    return {
        "key": role,
        **ROLE_CATALOG[role],
    }


def _member_response(
    row: dict,
    member_identity: dict,
) -> dict:
    role = ROLE_CATALOG.get(
        row.get("relationship_type"),
        ROLE_CATALOG["custom"],
    )

    image_source = (
        row.get("image_source")
        or "profile"
    )

    builtin_image_key = (
        row.get("builtin_image_key")
        or role["builtin_image_key"]
    )

    public_identity = _public_identity(
        member_identity
    )

    return {
        "identity": public_identity,
        "presentation": {
            "display_label": (
                row.get("display_label")
                or public_identity.get("display_name")
                or public_identity.get("handle")
            ),
            "relationship_type": (
                row.get("relationship_type")
                or "custom"
            ),
            "relationship_label": role["label"],
            "contact_type": (
                row.get("contact_type")
                or role["contact_type"]
            ),
            "description": (
                row.get("description")
                or ""
            ),
            "image_source": image_source,
            "builtin_image_key": (
                builtin_image_key
                if image_source == "builtin"
                else ""
            ),
            "profile_media_url": (
                f"/trust-voice/profile-media/"
                f"{member_identity['id']}"
            ),
            "sort_order": int(
                row.get("sort_order")
                or 0
            ),
        },
        "created_at": (
            row.get("created_at")
            or ""
        ),
        "updated_at": (
            row.get("updated_at")
            or ""
        ),
    }


@router.get(
    "/trust-circle/health"
)
def trust_circle_health():
    return {
        "status": "ok",
        "feature": "trust_voice_trust_circle",
        "version": FEATURE_VERSION,
        "enforcement_enabled": trust_circle_enabled(),
    }


@router.get(
    "/trust-circle/roles"
)
def trust_circle_roles():
    return {
        "roles": [
            {
                "key": key,
                **value,
            }
            for key, value
            in ROLE_CATALOG.items()
        ],
        "note": (
            "Built-in image keys represent generic role artwork. "
            "They do not identify or verify a real person."
        ),
    }


@router.get(
    "/trust-circle"
)
def get_trust_circle(
    authorization: Optional[str] = Header(
        default=None
    ),
):
    owner = _identity_from_bearer(
        authorization
    )

    ensure_trust_circle_schema()

    with get_db() as conn:
        cur = conn.cursor()

        cur.execute(
            """
            SELECT *
            FROM verifyd_trust_circle_members
            WHERE owner_identity_id = %s
            ORDER BY
                sort_order ASC,
                created_at ASC
            """,
            (
                owner["id"],
            ),
        )

        rows = cur.fetchall() or []

    members = []

    for raw_row in rows:
        row = dict(raw_row)

        member_identity = _identity_by_id(
            row["member_identity_id"]
        )

        if not member_identity:
            continue

        members.append(
            _member_response(
                row,
                member_identity,
            )
        )

    return {
        "owner": _public_identity(owner),
        "count": len(members),
        "members": members,
    }


@router.post(
    "/trust-circle/members"
)
def add_trust_circle_member(
    payload: TrustCircleMemberCreate,
    authorization: Optional[str] = Header(
        default=None
    ),
):
    owner = _identity_from_bearer(
        authorization
    )

    handle_lower = _normalize_handle(
        payload.handle
    )

    if not handle_lower:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "handle_required",
            },
        )

    member = _identity_by_handle(
        handle_lower
    )

    if not member:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "handle_not_found",
            },
        )

    if member["id"] == owner["id"]:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "cannot_add_self",
            },
        )

    role = _validate_role(
        payload.relationship_type
    )

    display_label = _clean_short_text(
        payload.display_label,
        80,
    )

    if not display_label:
        display_label = (
            member.get("display_name")
            or f"@{member.get('handle', '')}"
        )

    description = _clean_short_text(
        payload.description,
        160,
    )

    builtin_image_key = (
        role["builtin_image_key"]
        if payload.image_source == "builtin"
        else ""
    )

    ensure_trust_circle_schema()

    now = _now_iso()

    with get_db() as conn:
        cur = conn.cursor()

        cur.execute(
            """
            SELECT COALESCE(
                MAX(sort_order),
                -1
            ) + 1 AS next_order
            FROM verifyd_trust_circle_members
            WHERE owner_identity_id = %s
            """,
            (
                owner["id"],
            ),
        )

        order_row = cur.fetchone()

        next_order = int(
            (
                order_row.get("next_order")
                if order_row
                else 0
            )
            or 0
        )

        cur.execute(
            """
            INSERT INTO verifyd_trust_circle_members (
                owner_identity_id,
                member_identity_id,
                display_label,
                relationship_type,
                contact_type,
                description,
                image_source,
                builtin_image_key,
                sort_order,
                created_at,
                updated_at
            )
            VALUES (
                %s, %s,
                %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s
            )
            ON CONFLICT (
                owner_identity_id,
                member_identity_id
            )
            DO UPDATE SET
                display_label = EXCLUDED.display_label,
                relationship_type = EXCLUDED.relationship_type,
                contact_type = EXCLUDED.contact_type,
                description = EXCLUDED.description,
                image_source = EXCLUDED.image_source,
                builtin_image_key = EXCLUDED.builtin_image_key,
                updated_at = EXCLUDED.updated_at
            RETURNING *
            """,
            (
                owner["id"],
                member["id"],
                display_label,
                role["key"],
                role["contact_type"],
                description,
                payload.image_source,
                builtin_image_key,
                next_order,
                now,
                now,
            ),
        )

        saved = dict(
            cur.fetchone()
        )

    return {
        "ok": True,
        "member": _member_response(
            saved,
            member,
        ),
    }


@router.patch(
    "/trust-circle/members/{handle}"
)
def update_trust_circle_member(
    handle: str,
    payload: TrustCircleMemberUpdate,
    authorization: Optional[str] = Header(
        default=None
    ),
):
    owner = _identity_from_bearer(
        authorization
    )

    member = _identity_by_handle(
        _normalize_handle(handle)
    )

    if not member:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "handle_not_found",
            },
        )

    ensure_trust_circle_schema()

    with get_db() as conn:
        cur = conn.cursor()

        cur.execute(
            """
            SELECT *
            FROM verifyd_trust_circle_members
            WHERE owner_identity_id = %s
              AND member_identity_id = %s
            LIMIT 1
            """,
            (
                owner["id"],
                member["id"],
            ),
        )

        existing = cur.fetchone()

        if not existing:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "trust_circle_member_not_found",
                },
            )

        existing = dict(existing)

        relationship_type = (
            payload.relationship_type
            if payload.relationship_type is not None
            else existing["relationship_type"]
        )

        role = _validate_role(
            relationship_type
        )

        display_label = (
            _clean_short_text(
                payload.display_label,
                80,
            )
            if payload.display_label is not None
            else existing["display_label"]
        )

        description = (
            _clean_short_text(
                payload.description,
                160,
            )
            if payload.description is not None
            else existing["description"]
        )

        image_source = (
            payload.image_source
            if payload.image_source is not None
            else existing["image_source"]
        )

        sort_order = (
            payload.sort_order
            if payload.sort_order is not None
            else existing["sort_order"]
        )

        builtin_image_key = (
            role["builtin_image_key"]
            if image_source == "builtin"
            else ""
        )

        cur.execute(
            """
            UPDATE verifyd_trust_circle_members
            SET
                display_label = %s,
                relationship_type = %s,
                contact_type = %s,
                description = %s,
                image_source = %s,
                builtin_image_key = %s,
                sort_order = %s,
                updated_at = %s
            WHERE owner_identity_id = %s
              AND member_identity_id = %s
            RETURNING *
            """,
            (
                display_label,
                role["key"],
                role["contact_type"],
                description,
                image_source,
                builtin_image_key,
                sort_order,
                _now_iso(),
                owner["id"],
                member["id"],
            ),
        )

        saved = dict(
            cur.fetchone()
        )

    return {
        "ok": True,
        "member": _member_response(
            saved,
            member,
        ),
    }


@router.delete(
    "/trust-circle/members/{handle}"
)
def delete_trust_circle_member(
    handle: str,
    authorization: Optional[str] = Header(
        default=None
    ),
):
    owner = _identity_from_bearer(
        authorization
    )

    member = _identity_by_handle(
        _normalize_handle(handle)
    )

    if not member:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "handle_not_found",
            },
        )

    ensure_trust_circle_schema()

    with get_db() as conn:
        cur = conn.cursor()

        cur.execute(
            """
            DELETE FROM verifyd_trust_circle_members
            WHERE owner_identity_id = %s
              AND member_identity_id = %s
            RETURNING member_identity_id
            """,
            (
                owner["id"],
                member["id"],
            ),
        )

        deleted = cur.fetchone()

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "trust_circle_member_not_found",
            },
        )

    return {
        "ok": True,
        "deleted": True,
        "handle": f"@{member['handle']}",
    }


@router.get(
    "/call-settings"
)
def get_call_settings(
    authorization: Optional[str] = Header(
        default=None
    ),
):
    identity = _identity_from_bearer(
        authorization
    )

    privacy = (
        identity.get("call_privacy")
        or "verified_users"
    )

    if privacy not in ALLOWED_CALL_PRIVACY:
        privacy = "verified_users"

    return {
        "call_privacy": privacy,
        "options": [
            {
                "value": "anyone",
                "label": "Anyone",
            },
            {
                "value": "verified_users",
                "label": "Verified VeriFYD users",
            },
            {
                "value": "trust_circle",
                "label": "My Trust Circle",
            },
            {
                "value": "nobody",
                "label": "Nobody",
            },
        ],
    }


@router.put(
    "/call-settings"
)
def update_call_settings(
    payload: CallSettingsUpdate,
    authorization: Optional[str] = Header(
        default=None
    ),
):
    identity = _identity_from_bearer(
        authorization
    )

    privacy = payload.call_privacy

    if privacy not in ALLOWED_CALL_PRIVACY:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_call_privacy",
            },
        )

    now = _now_iso()

    with get_db() as conn:
        cur = conn.cursor()

        cur.execute(
            """
            UPDATE verifyd_identities
            SET
                call_privacy = %s,
                updated_at = %s
            WHERE id = %s
            RETURNING call_privacy
            """,
            (
                privacy,
                now,
                identity["id"],
            ),
        )

        row = cur.fetchone()

    if not row:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "identity_not_found",
            },
        )

    return {
        "ok": True,
        "call_privacy": row["call_privacy"],
    }