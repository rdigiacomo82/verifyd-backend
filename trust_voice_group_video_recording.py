# ============================================================
# VeriFYD Trust Voice — Group Video Recording
# VERIFYD_TRUST_VOICE_GROUP_VIDEO_RECORDING_V1
# Additive server-side recording for the existing LiveKit Group Video.
# ============================================================

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Optional

import boto3
from botocore.config import Config
from fastapi import APIRouter, BackgroundTasks, Header, HTTPException
from fastapi.responses import StreamingResponse
from livekit import api
from pydantic import BaseModel

from database import get_db
from trust_voice_groups import (
    _group_member_ids,
    _group_member_rows,
    _require_group_member,
    ensure_group_schema,
)
from trust_voice_messages import _identity_from_bearer, _public_contact
from trust_voice_signaling import _identity_by_id, _send_to_identity

log = logging.getLogger("verifyd.trust_voice.group_video_recording")
router = APIRouter(
    prefix="/trust-voice",
    tags=["Trust Voice Group Video Recording"],
)

FEATURE_VERSION = "0.1.2"
_schema_ready = False

OPEN_STATUSES = {
    "consent_pending",
    "ready",
    "starting",
    "active",
    "stopping",
}
FINAL_STATUSES = {
    "complete",
    "failed",
    "declined",
    "canceled",
}


class RecordingConsentPayload(BaseModel):
    allowed: bool


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _now_iso() -> str:
    return _now().isoformat()


def _env_int(
    name: str,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    try:
        value = int(
            (os.environ.get(name, str(default)) or "").strip()
        )
    except Exception:
        value = default
    return max(minimum, min(value, maximum))


def _start_api_timeout_seconds() -> int:
    return _env_int(
        "VERIFYD_TRUST_VOICE_GROUP_RECORDING_START_TIMEOUT_SECONDS",
        45,
        15,
        120,
    )


def _stop_api_timeout_seconds() -> int:
    return _env_int(
        "VERIFYD_TRUST_VOICE_GROUP_RECORDING_STOP_TIMEOUT_SECONDS",
        30,
        10,
        120,
    )


def _finalize_timeout_seconds() -> int:
    return _env_int(
        "VERIFYD_TRUST_VOICE_GROUP_RECORDING_FINALIZE_TIMEOUT_SECONDS",
        90,
        30,
        300,
    )


def _parse_iso(value: Optional[str]) -> Optional[datetime]:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(
            raw.replace("Z", "+00:00")
        )
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _seconds_since(value: Optional[str]) -> Optional[float]:
    dt = _parse_iso(value)
    if not dt:
        return None
    return max(
        0.0,
        (_now() - dt).total_seconds(),
    )


def _enabled() -> bool:
    return (
        os.environ.get(
            "VERIFYD_TRUST_VOICE_GROUP_RECORDING_ENABLED",
            "",
        )
        or ""
    ).strip().lower() in {"1", "true", "yes", "on"}


def _livekit_url() -> str:
    return (os.environ.get("LIVEKIT_URL", "") or "").strip()


def _livekit_api_url() -> str:
    value = _livekit_url()
    if value.startswith("wss://"):
        return "https://" + value[len("wss://"):]
    if value.startswith("ws://"):
        return "http://" + value[len("ws://"):]
    return value


def _livekit_key() -> str:
    return (os.environ.get("LIVEKIT_API_KEY", "") or "").strip()


def _livekit_secret() -> str:
    return (os.environ.get("LIVEKIT_API_SECRET", "") or "").strip()


def _r2_account_id() -> str:
    return (
        os.environ.get("R2_TRUST_VOICE_RECORDINGS_ACCOUNT_ID", "")
        or os.environ.get("R2_TRUST_VOICE_ACCOUNT_ID", "")
        or os.environ.get("R2_ACCOUNT_ID", "")
        or ""
    ).strip()


def _r2_access_key() -> str:
    return (
        os.environ.get("R2_TRUST_VOICE_RECORDINGS_ACCESS_KEY_ID", "")
        or os.environ.get("R2_TRUST_VOICE_ACCESS_KEY_ID", "")
        or ""
    ).strip()


def _r2_secret() -> str:
    return (
        os.environ.get("R2_TRUST_VOICE_RECORDINGS_SECRET_KEY", "")
        or os.environ.get("R2_TRUST_VOICE_SECRET_KEY", "")
        or ""
    ).strip()


def _r2_bucket() -> str:
    return (
        os.environ.get("R2_TRUST_VOICE_RECORDINGS_BUCKET", "")
        or os.environ.get("R2_TRUST_VOICE_BUCKET", "")
        or ""
    ).strip()


def _r2_endpoint() -> str:
    account_id = _r2_account_id()
    return (
        f"https://{account_id}.r2.cloudflarestorage.com"
        if account_id
        else ""
    )


def _storage_configured() -> bool:
    return bool(
        _r2_account_id()
        and _r2_access_key()
        and _r2_secret()
        and _r2_bucket()
        and _r2_endpoint()
    )


def _configured() -> bool:
    return bool(
        _livekit_url()
        and _livekit_key()
        and _livekit_secret()
        and _storage_configured()
    )


def _require_recording_available() -> None:
    if not _enabled():
        raise HTTPException(
            status_code=404,
            detail={"error": "not_found"},
        )
    if not _configured():
        raise HTTPException(
            status_code=503,
            detail={
                "error": "group_video_recording_not_configured",
                "message": "Group video recording is not fully configured.",
            },
        )


def _r2_client():
    return boto3.client(
        "s3",
        endpoint_url=_r2_endpoint(),
        aws_access_key_id=_r2_access_key(),
        aws_secret_access_key=_r2_secret(),
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )


def _participant_identity(identity_id: str) -> str:
    return "tvp_" + hashlib.sha256(
        str(identity_id or "").encode("utf-8")
    ).hexdigest()[:32]


def _duration_seconds(
    started_at: Optional[str],
    ended_at: Optional[str],
) -> Optional[int]:
    if not started_at:
        return None
    try:
        start = datetime.fromisoformat(
            str(started_at).replace("Z", "+00:00")
        )
        end = (
            datetime.fromisoformat(
                str(ended_at).replace("Z", "+00:00")
            )
            if ended_at
            else _now()
        )
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)
        return max(
            0,
            int(
                (
                    end.astimezone(timezone.utc)
                    - start.astimezone(timezone.utc)
                ).total_seconds()
            ),
        )
    except Exception:
        return None


def ensure_recording_schema() -> None:
    global _schema_ready
    if _schema_ready:
        return

    ensure_group_schema()

    with get_db() as conn:
        cur = conn.cursor()

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS verifyd_voice_group_video_recordings (
                recording_id                TEXT PRIMARY KEY,
                session_id                  TEXT NOT NULL,
                conversation_id             TEXT NOT NULL,
                requested_by_identity_id     TEXT NOT NULL,
                status                      TEXT NOT NULL,
                requested_at                TEXT NOT NULL,
                ready_at                    TEXT,
                started_at                  TEXT,
                ended_at                    TEXT,
                egress_id                   TEXT,
                storage_key                 TEXT,
                stop_reason                 TEXT,
                error_text                  TEXT,
                egress_status               INTEGER,
                file_size_bytes             BIGINT
            )
            """
        )

        cur.execute(
            """
            ALTER TABLE verifyd_voice_group_video_recordings
            ADD COLUMN IF NOT EXISTS status_updated_at TEXT
            """
        )

        cur.execute(
            """
            UPDATE verifyd_voice_group_video_recordings
            SET status_updated_at = COALESCE(
                status_updated_at,
                ended_at,
                started_at,
                ready_at,
                requested_at
            )
            WHERE status_updated_at IS NULL
            """
        )

        cur.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS
            uq_verifyd_voice_group_video_open_recording
            ON verifyd_voice_group_video_recordings (session_id)
            WHERE status IN (
                'consent_pending',
                'ready',
                'starting',
                'active',
                'stopping'
            )
            """
        )

        cur.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS
            uq_verifyd_voice_group_video_egress
            ON verifyd_voice_group_video_recordings (egress_id)
            WHERE egress_id IS NOT NULL
            """
        )

        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_verifyd_voice_group_video_recording_history
            ON verifyd_voice_group_video_recordings (
                conversation_id,
                requested_at DESC
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS
            verifyd_voice_group_video_recording_consents (
                recording_id   TEXT NOT NULL,
                identity_id    TEXT NOT NULL,
                required       BOOLEAN NOT NULL DEFAULT TRUE,
                allowed        BOOLEAN,
                responded_at   TEXT,
                PRIMARY KEY (recording_id, identity_id)
            )
            """
        )

    _schema_ready = True


def _active_session(conversation_id: str) -> Optional[dict]:
    ensure_recording_schema()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT *
            FROM verifyd_voice_group_video_sessions
            WHERE conversation_id = %s
              AND status = 'active'
            ORDER BY started_at DESC
            LIMIT 1
            """,
            (conversation_id,),
        )
        row = cur.fetchone()
    return dict(row) if row else None


def _session_by_room(room_name: str) -> Optional[dict]:
    if not room_name:
        return None
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT *
            FROM verifyd_voice_group_video_sessions
            WHERE room_name = %s
            ORDER BY started_at DESC
            LIMIT 1
            """,
            (room_name,),
        )
        row = cur.fetchone()
    return dict(row) if row else None


def _recording_by_id(recording_id: str) -> Optional[dict]:
    ensure_recording_schema()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT *
            FROM verifyd_voice_group_video_recordings
            WHERE recording_id = %s
            LIMIT 1
            """,
            (recording_id,),
        )
        row = cur.fetchone()
    return dict(row) if row else None


def _recording_by_egress_id(egress_id: str) -> Optional[dict]:
    if not egress_id:
        return None
    ensure_recording_schema()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT *
            FROM verifyd_voice_group_video_recordings
            WHERE egress_id = %s
            LIMIT 1
            """,
            (egress_id,),
        )
        row = cur.fetchone()
    return dict(row) if row else None


def _open_recording(session_id: str) -> Optional[dict]:
    ensure_recording_schema()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT *
            FROM verifyd_voice_group_video_recordings
            WHERE session_id = %s
              AND status IN (
                  'consent_pending',
                  'ready',
                  'starting',
                  'active',
                  'stopping'
              )
            ORDER BY requested_at DESC
            LIMIT 1
            """,
            (session_id,),
        )
        row = cur.fetchone()
    return dict(row) if row else None


def _latest_recording(
    session_id: str,
) -> Optional[dict]:
    ensure_recording_schema()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT *
            FROM verifyd_voice_group_video_recordings
            WHERE session_id = %s
            ORDER BY requested_at DESC
            LIMIT 1
            """,
            (session_id,),
        )
        row = cur.fetchone()
    return dict(row) if row else None


def _public_recording(
    recording: Optional[dict],
) -> Optional[dict]:
    if not recording:
        return None

    requester_id = str(
        recording.get("requested_by_identity_id") or ""
    )
    requester = (
        _identity_by_id(requester_id)
        if requester_id
        else None
    )
    recording_id = str(
        recording.get("recording_id") or ""
    )

    complete = bool(
        recording.get("status") == "complete"
        and recording.get("storage_key")
    )

    return {
        "recording_id": recording_id,
        "session_id": recording.get("session_id") or "",
        "group_id": recording.get("conversation_id") or "",
        "status": recording.get("status") or "",
        "requested_by": (
            _public_contact(requester)
            if requester
            else None
        ),
        "requested_at": recording.get("requested_at") or "",
        "ready_at": recording.get("ready_at"),
        "started_at": recording.get("started_at"),
        "ended_at": recording.get("ended_at"),
        "duration_seconds": _duration_seconds(
            recording.get("started_at"),
            recording.get("ended_at"),
        ),
        "stop_reason": recording.get("stop_reason"),
        "egress_status": (
            int(recording.get("egress_status"))
            if recording.get("egress_status") is not None
            else None
        ),
        "error": recording.get("error_text") or None,
        "status_updated_at": recording.get("status_updated_at"),
        "file_size_bytes": (
            int(recording.get("file_size_bytes") or 0)
            if recording.get("file_size_bytes") is not None
            else None
        ),
        "available": complete,
        "media_path": (
            f"/trust-voice/group-video/recordings/"
            f"{recording_id}/media"
            if complete
            else None
        ),
        "download_path": (
            f"/trust-voice/group-video/recordings/"
            f"{recording_id}/download"
            if complete
            else None
        ),
    }


def recording_history_for_session(
    session_id: str,
) -> list[dict]:
    ensure_recording_schema()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT *
            FROM verifyd_voice_group_video_recordings
            WHERE session_id = %s
            ORDER BY requested_at ASC
            """,
            (session_id,),
        )
        rows = [
            dict(row)
            for row in (cur.fetchall() or [])
        ]
    return [
        _public_recording(row)
        for row in rows
    ]


def recording_count_for_session(
    session_id: str,
) -> int:
    ensure_recording_schema()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT COUNT(*) AS recording_count
            FROM verifyd_voice_group_video_recordings
            WHERE session_id = %s
              AND status = 'complete'
            """,
            (session_id,),
        )
        row = cur.fetchone() or {}
    return int(
        row.get("recording_count") or 0
    )


def _consent_row(
    recording_id: str,
    identity_id: str,
) -> Optional[dict]:
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT *
            FROM verifyd_voice_group_video_recording_consents
            WHERE recording_id = %s
              AND identity_id = %s
            LIMIT 1
            """,
            (
                recording_id,
                identity_id,
            ),
        )
        row = cur.fetchone()
    return dict(row) if row else None


def _consent_summary(recording_id: str) -> dict:
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT
                COUNT(*) FILTER (
                    WHERE required = TRUE
                ) AS required_count,
                COUNT(*) FILTER (
                    WHERE required = TRUE
                      AND allowed = TRUE
                ) AS allowed_count,
                COUNT(*) FILTER (
                    WHERE required = TRUE
                      AND allowed IS NULL
                ) AS pending_count,
                COUNT(*) FILTER (
                    WHERE required = TRUE
                      AND allowed = FALSE
                ) AS declined_count
            FROM verifyd_voice_group_video_recording_consents
            WHERE recording_id = %s
            """,
            (recording_id,),
        )
        row = cur.fetchone() or {}

    return {
        "required_count": int(
            row.get("required_count") or 0
        ),
        "allowed_count": int(
            row.get("allowed_count") or 0
        ),
        "pending_count": int(
            row.get("pending_count") or 0
        ),
        "declined_count": int(
            row.get("declined_count") or 0
        ),
    }


def _upsert_consent(
    recording_id: str,
    identity_id: str,
    required: bool,
    allowed: Optional[bool],
    *,
    preserve_answer: bool = False,
) -> None:
    existing = _consent_row(
        recording_id,
        identity_id,
    )

    if preserve_answer and existing:
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                UPDATE verifyd_voice_group_video_recording_consents
                SET required = %s
                WHERE recording_id = %s
                  AND identity_id = %s
                """,
                (
                    required,
                    recording_id,
                    identity_id,
                ),
            )
        return

    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO verifyd_voice_group_video_recording_consents (
                recording_id,
                identity_id,
                required,
                allowed,
                responded_at
            )
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (recording_id, identity_id)
            DO UPDATE SET
                required = EXCLUDED.required,
                allowed = EXCLUDED.allowed,
                responded_at = EXCLUDED.responded_at
            """,
            (
                recording_id,
                identity_id,
                required,
                allowed,
                (
                    _now_iso()
                    if allowed is not None
                    else None
                ),
            ),
        )


def _identity_map_for_group(
    conversation_id: str,
) -> dict[str, str]:
    result: dict[str, str] = {}
    for row in _group_member_rows(
        conversation_id
    ):
        identity_id = str(
            row.get("identity_id")
            or row.get("id")
            or ""
        )
        if identity_id:
            result[
                _participant_identity(identity_id)
            ] = identity_id
    return result


def _identity_id_from_participant(
    conversation_id: str,
    participant_identity: str,
) -> Optional[str]:
    return _identity_map_for_group(
        conversation_id
    ).get(
        participant_identity
    )


async def _connected_identity_ids(
    session: dict,
) -> list[str]:
    room_name = str(
        session.get("room_name") or ""
    )
    conversation_id = str(
        session.get("conversation_id") or ""
    )
    if not room_name:
        return []

    identity_map = _identity_map_for_group(
        conversation_id
    )

    async with api.LiveKitAPI(
        _livekit_api_url(),
        api_key=_livekit_key(),
        api_secret=_livekit_secret(),
    ) as lkapi:
        response = await lkapi.room.list_participants(
            api.ListParticipantsRequest(
                room=room_name
            )
        )

    result = []
    seen = set()

    for participant in response.participants:
        participant_identity = str(
            getattr(
                participant,
                "identity",
                "",
            )
            or ""
        )
        identity_id = identity_map.get(
            participant_identity
        )
        if (
            identity_id
            and identity_id not in seen
        ):
            seen.add(identity_id)
            result.append(identity_id)

    return result


async def _notify_group(
    conversation_id: str,
    payload: dict,
) -> None:
    for identity_id in _group_member_ids(
        conversation_id
    ):
        await _send_to_identity(
            identity_id,
            payload,
        )


def _storage_key(
    session_id: str,
    recording_id: str,
) -> str:
    return (
        "trust-voice/group-video-recordings/"
        f"{session_id}/{recording_id}.mp4"
    )


async def _start_livekit_egress(
    session: dict,
    storage_key: str,
):
    room_name = str(
        session.get("room_name") or ""
    )

    s3 = api.S3Upload(
        access_key=_r2_access_key(),
        secret=_r2_secret(),
        endpoint=_r2_endpoint(),
        region="auto",
        bucket=_r2_bucket(),
        force_path_style=True,
    )

    async with api.LiveKitAPI(
        _livekit_api_url(),
        api_key=_livekit_key(),
        api_secret=_livekit_secret(),
    ) as lkapi:
        if (
            hasattr(api, "StartEgressRequest")
            and hasattr(api, "TemplateSource")
            and hasattr(api, "FileOutput")
            and hasattr(api, "Output")
            and hasattr(api, "StorageConfig")
            and hasattr(
                lkapi.egress,
                "start_egress",
            )
        ):
            request = api.StartEgressRequest(
                room_name=room_name,
                template=api.TemplateSource(
                    layout="grid"
                ),
                preset=api.EncodingOptionsPreset.H264_720P_30,
                outputs=[
                    api.Output(
                        file=api.FileOutput(
                            file_type=api.EncodedFileType.MP4,
                            filepath=storage_key,
                        )
                    )
                ],
                storage=api.StorageConfig(
                    s3=s3
                ),
            )
            return await asyncio.wait_for(
                lkapi.egress.start_egress(
                    request
                ),
                timeout=_start_api_timeout_seconds(),
            )

        request = api.RoomCompositeEgressRequest(
            room_name=room_name,
            layout="grid",
            preset=api.EncodingOptionsPreset.H264_720P_30,
            audio_only=False,
            file_outputs=[
                api.EncodedFileOutput(
                    file_type=api.EncodedFileType.MP4,
                    filepath=storage_key,
                    s3=s3,
                )
            ],
        )
        return await asyncio.wait_for(
            lkapi.egress.start_room_composite_egress(
                request
            ),
            timeout=_start_api_timeout_seconds(),
        )


async def _stop_livekit_egress(
    egress_id: str,
):
    async with api.LiveKitAPI(
        _livekit_api_url(),
        api_key=_livekit_key(),
        api_secret=_livekit_secret(),
    ) as lkapi:
        return await asyncio.wait_for(
            lkapi.egress.stop_egress(
                api.StopEgressRequest(
                    egress_id=egress_id
                )
            ),
            timeout=_stop_api_timeout_seconds(),
        )


def _egress_status_name(status: int) -> str:
    return {
        0: "STARTING",
        1: "ACTIVE",
        2: "ENDING",
        3: "COMPLETE",
        4: "FAILED",
        5: "ABORTED",
        6: "LIMIT_REACHED",
    }.get(int(status or 0), f"UNKNOWN_{status}")


def _egress_file_size(info) -> Optional[int]:
    try:
        results = list(
            getattr(info, "file_results", None)
            or []
        )
        if results:
            return int(
                getattr(results[0], "size", 0)
                or 0
            )
    except Exception:
        return None
    return None


def _r2_object_metadata(
    storage_key: str,
) -> Optional[dict]:
    if not storage_key:
        return None
    try:
        obj = _r2_client().head_object(
            Bucket=_r2_bucket(),
            Key=storage_key,
        )
        return {
            "size": int(
                obj.get("ContentLength") or 0
            ),
            "etag": str(
                obj.get("ETag") or ""
            ),
        }
    except Exception:
        return None


async def _list_livekit_egress(
    *,
    egress_id: str = "",
    room_name: str = "",
):
    request = api.ListEgressRequest(
        egress_id=egress_id or "",
        room_name=room_name or "",
    )
    async with api.LiveKitAPI(
        _livekit_api_url(),
        api_key=_livekit_key(),
        api_secret=_livekit_secret(),
    ) as lkapi:
        response = await asyncio.wait_for(
            lkapi.egress.list_egress(
                request
            ),
            timeout=10,
        )
    return list(
        getattr(response, "items", None)
        or []
    )


async def _mark_complete_from_storage(
    recording: dict,
    *,
    source: str,
    notify: bool,
) -> dict:
    recording_id = str(
        recording.get("recording_id") or ""
    )
    metadata = await asyncio.to_thread(
        _r2_object_metadata,
        str(recording.get("storage_key") or ""),
    )
    if not metadata:
        return recording

    now = _now_iso()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE verifyd_voice_group_video_recordings
            SET
                status = 'complete',
                ended_at = COALESCE(ended_at, %s),
                file_size_bytes = COALESCE(%s, file_size_bytes),
                error_text = NULL,
                status_updated_at = %s
            WHERE recording_id = %s
              AND status NOT IN (
                  'complete',
                  'declined',
                  'canceled'
              )
            """,
            (
                now,
                metadata.get("size"),
                now,
                recording_id,
            ),
        )

    current = (
        _recording_by_id(recording_id)
        or recording
    )
    log.info(
        "Group recording reconciled from R2: recording_id=%s "
        "source=%s size=%s",
        recording_id,
        source,
        metadata.get("size"),
    )

    if notify:
        await _notify_group(
            str(current.get("conversation_id") or ""),
            {
                "type": "group_video_recording_ended",
                "group_id": current.get("conversation_id") or "",
                "recording": _public_recording(
                    current
                ),
            },
        )

    return current


async def _apply_egress_info(
    recording_id: str,
    info,
    *,
    source: str,
    notify: bool = True,
) -> dict:
    current = (
        _recording_by_id(recording_id)
        or {}
    )
    if not current:
        return {}

    prior_status = str(
        current.get("status") or ""
    )
    if prior_status in {
        "complete",
        "declined",
        "canceled",
    }:
        return current

    egress_id = str(
        getattr(info, "egress_id", "") or ""
    )
    egress_status = int(
        getattr(info, "status", 0) or 0
    )
    status_name = _egress_status_name(
        egress_status
    )
    file_size = _egress_file_size(
        info
    )
    error_text = str(
        getattr(info, "error", "") or ""
    )[:500]
    now = _now_iso()

    log.info(
        "Group recording Egress state: recording_id=%s "
        "egress_id=%s source=%s egress_status=%s prior_status=%s",
        recording_id,
        egress_id or "-",
        source,
        status_name,
        prior_status,
    )

    if egress_status == 0:
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                UPDATE verifyd_voice_group_video_recordings
                SET
                    egress_id = COALESCE(NULLIF(%s, ''), egress_id),
                    egress_status = %s,
                    status = CASE
                        WHEN status IN ('ready', 'consent_pending')
                        THEN 'starting'
                        ELSE status
                    END,
                    status_updated_at = CASE
                        WHEN status IN ('ready', 'consent_pending')
                        THEN %s
                        ELSE status_updated_at
                    END
                WHERE recording_id = %s
                  AND status NOT IN (
                      'complete',
                      'failed',
                      'declined',
                      'canceled'
                  )
                """,
                (
                    egress_id,
                    egress_status,
                    now,
                    recording_id,
                ),
            )

    elif egress_status == 1:
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                UPDATE verifyd_voice_group_video_recordings
                SET
                    egress_id = COALESCE(NULLIF(%s, ''), egress_id),
                    egress_status = %s,
                    status = CASE
                        WHEN status IN ('ready', 'starting')
                        THEN 'active'
                        ELSE status
                    END,
                    started_at = CASE
                        WHEN status IN ('ready', 'starting')
                        THEN COALESCE(started_at, %s)
                        ELSE started_at
                    END,
                    status_updated_at = CASE
                        WHEN status IN ('ready', 'starting')
                        THEN %s
                        ELSE status_updated_at
                    END,
                    error_text = NULL
                WHERE recording_id = %s
                  AND status NOT IN (
                      'complete',
                      'failed',
                      'declined',
                      'canceled'
                  )
                """,
                (
                    egress_id,
                    egress_status,
                    now,
                    now,
                    recording_id,
                ),
            )

    elif egress_status == 2:
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                UPDATE verifyd_voice_group_video_recordings
                SET
                    egress_id = COALESCE(NULLIF(%s, ''), egress_id),
                    egress_status = %s,
                    status = CASE
                        WHEN status IN ('starting', 'active')
                        THEN 'stopping'
                        ELSE status
                    END,
                    status_updated_at = CASE
                        WHEN status IN ('starting', 'active')
                        THEN %s
                        ELSE status_updated_at
                    END
                WHERE recording_id = %s
                  AND status NOT IN (
                      'complete',
                      'failed',
                      'declined',
                      'canceled'
                  )
                """,
                (
                    egress_id,
                    egress_status,
                    now,
                    recording_id,
                ),
            )

    elif egress_status in {3, 6}:
        stop_reason = current.get(
            "stop_reason"
        )
        if (
            egress_status == 6
            and not stop_reason
        ):
            stop_reason = "livekit_limit_reached"

        with get_db() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                UPDATE verifyd_voice_group_video_recordings
                SET
                    egress_id = COALESCE(NULLIF(%s, ''), egress_id),
                    status = 'complete',
                    ended_at = COALESCE(ended_at, %s),
                    stop_reason = %s,
                    error_text = NULLIF(%s, ''),
                    egress_status = %s,
                    file_size_bytes = COALESCE(%s, file_size_bytes),
                    status_updated_at = %s
                WHERE recording_id = %s
                  AND status NOT IN (
                      'declined',
                      'canceled'
                  )
                """,
                (
                    egress_id,
                    now,
                    stop_reason,
                    error_text,
                    egress_status,
                    file_size,
                    now,
                    recording_id,
                ),
            )

    elif egress_status in {4, 5}:
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                UPDATE verifyd_voice_group_video_recordings
                SET
                    egress_id = COALESCE(NULLIF(%s, ''), egress_id),
                    status = 'failed',
                    ended_at = COALESCE(ended_at, %s),
                    error_text = %s,
                    egress_status = %s,
                    status_updated_at = %s
                WHERE recording_id = %s
                  AND status NOT IN (
                      'complete',
                      'declined',
                      'canceled'
                  )
                """,
                (
                    egress_id,
                    now,
                    error_text or (
                        "livekit_egress_"
                        + status_name.lower()
                    ),
                    egress_status,
                    now,
                    recording_id,
                ),
            )

    refreshed = (
        _recording_by_id(recording_id)
        or current
    )
    new_status = str(
        refreshed.get("status") or ""
    )

    if notify and new_status != prior_status:
        conversation_id = str(
            refreshed.get("conversation_id") or ""
        )
        if new_status == "active":
            await _notify_group(
                conversation_id,
                {
                    "type": "group_video_recording_started",
                    "group_id": conversation_id,
                    "recording": _public_recording(
                        refreshed
                    ),
                },
            )
        elif new_status == "complete":
            await _notify_group(
                conversation_id,
                {
                    "type": "group_video_recording_ended",
                    "group_id": conversation_id,
                    "recording": _public_recording(
                        refreshed
                    ),
                },
            )
        elif new_status == "failed":
            await _notify_group(
                conversation_id,
                {
                    "type": "group_video_recording_failed",
                    "group_id": conversation_id,
                    "recording": _public_recording(
                        refreshed
                    ),
                },
            )

    return refreshed


async def _reconcile_recording_once(
    recording: dict,
    *,
    source: str,
    notify: bool = True,
) -> dict:
    recording_id = str(
        recording.get("recording_id") or ""
    )
    current = (
        _recording_by_id(recording_id)
        or recording
    )
    status = str(
        current.get("status") or ""
    )

    if status in FINAL_STATUSES:
        return current

    if (
        status == "stopping"
        and current.get("storage_key")
    ):
        completed = await _mark_complete_from_storage(
            current,
            source=source,
            notify=notify,
        )
        if str(
            completed.get("status") or ""
        ) == "complete":
            return completed

    egress_id = str(
        current.get("egress_id") or ""
    )
    room_name = ""
    session = _active_session(
        str(current.get("conversation_id") or "")
    )
    if session:
        room_name = str(
            session.get("room_name") or ""
        )

    infos = []
    try:
        if egress_id:
            infos = await _list_livekit_egress(
                egress_id=egress_id
            )
        elif status == "starting" and room_name:
            infos = await _list_livekit_egress(
                room_name=room_name
            )
    except Exception:
        log.warning(
            "Group recording Egress reconciliation lookup failed: "
            "recording_id=%s source=%s",
            recording_id,
            source,
            exc_info=True,
        )

    if infos:
        current = await _apply_egress_info(
            recording_id,
            infos[-1],
            source=source,
            notify=notify,
        )
        status = str(
            current.get("status") or ""
        )
        if (
            status in FINAL_STATUSES
            or status == "active"
        ):
            return current

    status_clock = (
        current.get("status_updated_at")
        or (
            current.get("ready_at")
            if status == "starting"
            else current.get("started_at")
        )
        or current.get("requested_at")
    )
    elapsed = _seconds_since(
        status_clock
    )

    if status == "starting":
        limit = max(
            60,
            _start_api_timeout_seconds() + 15,
        )
        if (
            elapsed is not None
            and elapsed >= limit
        ):
            log.error(
                "Group recording start timed out: recording_id=%s "
                "elapsed=%.1f",
                recording_id,
                elapsed,
            )
            _mark_failed(
                recording_id,
                "egress_start_timeout",
            )

    elif status == "stopping":
        limit = _finalize_timeout_seconds()
        if (
            elapsed is not None
            and elapsed >= limit
        ):
            completed = await _mark_complete_from_storage(
                current,
                source=source + ":timeout_r2",
                notify=notify,
            )
            if str(
                completed.get("status") or ""
            ) == "complete":
                return completed

            log.error(
                "Group recording finalization timed out: recording_id=%s "
                "elapsed=%.1f",
                recording_id,
                elapsed,
            )
            _mark_failed(
                recording_id,
                "egress_finalize_timeout",
            )

    refreshed = (
        _recording_by_id(recording_id)
        or current
    )
    if (
        notify
        and str(refreshed.get("status") or "") == "failed"
        and status != "failed"
    ):
        await _notify_group(
            str(refreshed.get("conversation_id") or ""),
            {
                "type": "group_video_recording_failed",
                "group_id": refreshed.get("conversation_id") or "",
                "recording": _public_recording(
                    refreshed
                ),
            },
        )
    return refreshed


async def _reconcile_until_settled(
    recording_id: str,
    *,
    phase: str,
) -> None:
    max_seconds = (
        max(60, _start_api_timeout_seconds() + 15)
        if phase == "start"
        else _finalize_timeout_seconds()
    )
    deadline = _now().timestamp() + max_seconds + 6

    while _now().timestamp() < deadline:
        recording = _recording_by_id(
            recording_id
        )
        if not recording:
            return

        status = str(
            recording.get("status") or ""
        )
        if status in FINAL_STATUSES:
            return
        if (
            phase == "start"
            and status == "active"
        ):
            return

        try:
            await _reconcile_recording_once(
                recording,
                source=f"background_{phase}",
                notify=True,
            )
        except Exception:
            log.warning(
                "Group recording background reconciliation failed: "
                "recording_id=%s phase=%s",
                recording_id,
                phase,
                exc_info=True,
            )

        await asyncio.sleep(3)


async def _begin_recording_background(
    recording_id: str,
) -> None:
    try:
        await _begin_recording(
            recording_id
        )
    except Exception:
        log.exception(
            "Background LiveKit group recording start failed"
        )

    await _reconcile_until_settled(
        recording_id,
        phase="start",
    )


def _mark_recording_stopping(
    recording_id: str,
    reason: str,
) -> dict:
    now = _now_iso()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE verifyd_voice_group_video_recordings
            SET
                status = 'stopping',
                stop_reason = %s,
                status_updated_at = %s
            WHERE recording_id = %s
              AND status IN ('starting', 'active')
            """,
            (
                reason,
                now,
                recording_id,
            ),
        )

    current = (
        _recording_by_id(recording_id)
        or {}
    )
    log.info(
        "Group recording stopping: recording_id=%s reason=%s "
        "egress_id=%s",
        recording_id,
        reason,
        current.get("egress_id") or "-",
    )
    return current


async def _stop_recording_background(
    recording_id: str,
    prior_status: str,
) -> None:
    recording = _recording_by_id(
        recording_id
    )
    if not recording:
        return

    egress_id = str(
        recording.get("egress_id") or ""
    )

    if egress_id:
        try:
            info = await _stop_livekit_egress(
                egress_id
            )
            await _apply_egress_info(
                recording_id,
                info,
                source="stop_response",
                notify=True,
            )
        except Exception:
            log.warning(
                "LiveKit StopEgress did not settle immediately: "
                "recording_id=%s egress_id=%s",
                recording_id,
                egress_id,
                exc_info=True,
            )
            with get_db() as conn:
                cur = conn.cursor()
                cur.execute(
                    """
                    UPDATE verifyd_voice_group_video_recordings
                    SET error_text = 'egress_stop_api_pending'
                    WHERE recording_id = %s
                      AND status = 'stopping'
                    """,
                    (recording_id,),
                )
    else:
        log.info(
            "Group recording stop requested during Egress startup: "
            "recording_id=%s prior_status=%s",
            recording_id,
            prior_status,
        )

    await _reconcile_until_settled(
        recording_id,
        phase="stop",
    )


async def _remove_livekit_participant(
    room_name: str,
    participant_identity: str,
) -> None:
    async with api.LiveKitAPI(
        _livekit_api_url(),
        api_key=_livekit_key(),
        api_secret=_livekit_secret(),
    ) as lkapi:
        await lkapi.room.remove_participant(
            api.RoomParticipantIdentity(
                room=room_name,
                identity=participant_identity,
            )
        )


def _mark_failed(
    recording_id: str,
    error_text: str,
) -> None:
    now = _now_iso()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE verifyd_voice_group_video_recordings
            SET
                status = 'failed',
                ended_at = COALESCE(ended_at, %s),
                error_text = %s,
                status_updated_at = %s
            WHERE recording_id = %s
              AND status NOT IN (
                  'complete',
                  'failed',
                  'declined',
                  'canceled'
              )
            """,
            (
                now,
                str(error_text or "")[:500],
                now,
                recording_id,
            ),
        )


async def _stop_internal(
    recording: dict,
    reason: str,
    *,
    raise_on_failure: bool = True,
) -> dict:
    recording_id = str(
        recording.get("recording_id") or ""
    )
    prior = str(
        recording.get("status") or ""
    )

    if prior == "stopping":
        return (
            _recording_by_id(recording_id)
            or recording
        )

    if prior not in {
        "starting",
        "active",
    }:
        return recording

    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE verifyd_voice_group_video_recordings
            SET
                status = 'stopping',
                stop_reason = %s
            WHERE recording_id = %s
              AND status IN ('starting', 'active')
            """,
            (
                reason,
                recording_id,
            ),
        )

    current = (
        _recording_by_id(recording_id)
        or recording
    )

    egress_id = str(
        current.get("egress_id") or ""
    )

    # During the brief starting race, _begin_recording handles this
    # after the Egress ID becomes available.
    if not egress_id:
        return current

    try:
        await _stop_livekit_egress(
            egress_id
        )
    except Exception as exc:
        log.exception(
            "LiveKit group recording stop failed"
        )
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                UPDATE verifyd_voice_group_video_recordings
                SET
                    status = %s,
                    error_text = 'egress_stop_failed'
                WHERE recording_id = %s
                  AND status = 'stopping'
                """,
                (
                    prior,
                    recording_id,
                ),
            )
        if raise_on_failure:
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "group_video_recording_stop_failed",
                },
            ) from exc

    return (
        _recording_by_id(recording_id)
        or current
    )


async def _notify_recording_request(
    conversation_id: str,
    recording: dict,
    identities: list[str],
) -> None:
    requester_id = str(
        recording.get("requested_by_identity_id") or ""
    )
    requester = (
        _identity_by_id(requester_id)
        if requester_id
        else None
    )

    for identity_id in identities:
        if identity_id == requester_id:
            continue

        await _send_to_identity(
            identity_id,
            {
                "type": "group_video_recording_request",
                "group_id": conversation_id,
                "recording": _public_recording(
                    recording
                ),
                "requester": (
                    _public_contact(requester)
                    if requester
                    else None
                ),
            },
        )


def _set_required_connected(
    recording_id: str,
    connected_ids: list[str],
    requester_id: str,
) -> None:
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE verifyd_voice_group_video_recording_consents
            SET required = FALSE
            WHERE recording_id = %s
            """,
            (recording_id,),
        )

    for identity_id in connected_ids:
        _upsert_consent(
            recording_id,
            identity_id,
            True,
            None,
            preserve_answer=True,
        )

    # Requesting a recording is the requester's affirmative consent.
    _upsert_consent(
        recording_id,
        requester_id,
        True,
        True,
    )


def _pending_required_ids(
    recording_id: str,
) -> list[str]:
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT identity_id
            FROM verifyd_voice_group_video_recording_consents
            WHERE recording_id = %s
              AND required = TRUE
              AND allowed IS DISTINCT FROM TRUE
            ORDER BY identity_id
            """,
            (recording_id,),
        )
        rows = cur.fetchall() or []

    return [
        str(row.get("identity_id") or "")
        for row in rows
        if row.get("identity_id")
    ]


async def _begin_recording(
    recording_id: str,
) -> dict:
    recording = _recording_by_id(
        recording_id
    )
    if not recording:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "group_video_recording_not_found"
            },
        )

    status = str(
        recording.get("status") or ""
    )
    if status in {
        "starting",
        "active",
        "stopping",
        "complete",
        "failed",
        "declined",
        "canceled",
    }:
        return recording

    now = _now_iso()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE verifyd_voice_group_video_recordings
            SET
                status = 'starting',
                ready_at = COALESCE(ready_at, %s),
                error_text = NULL,
                status_updated_at = %s
            WHERE recording_id = %s
              AND status IN ('consent_pending', 'ready')
            RETURNING *
            """,
            (
                now,
                now,
                recording_id,
            ),
        )
        claimed = cur.fetchone()

    if not claimed:
        return (
            _recording_by_id(recording_id)
            or recording
        )

    recording = dict(claimed)
    conversation_id = str(
        recording.get("conversation_id") or ""
    )
    requester_id = str(
        recording.get("requested_by_identity_id") or ""
    )
    session = _active_session(
        conversation_id
    )

    if (
        not session
        or str(session.get("session_id") or "")
        != str(recording.get("session_id") or "")
    ):
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                UPDATE verifyd_voice_group_video_recordings
                SET
                    status = 'canceled',
                    ended_at = %s,
                    stop_reason = 'room_finished',
                    status_updated_at = %s
                WHERE recording_id = %s
                  AND status = 'starting'
                """,
                (
                    _now_iso(),
                    _now_iso(),
                    recording_id,
                ),
            )
        return (
            _recording_by_id(recording_id)
            or recording
        )

    try:
        connected_ids = await _connected_identity_ids(
            session
        )
    except Exception as exc:
        log.exception(
            "Unable to recheck recording participants"
        )
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                UPDATE verifyd_voice_group_video_recordings
                SET
                    status = 'consent_pending',
                    error_text = 'participant_lookup_failed',
                    status_updated_at = %s
                WHERE recording_id = %s
                  AND status = 'starting'
                """,
                (
                    _now_iso(),
                    recording_id,
                ),
            )
        raise HTTPException(
            status_code=503,
            detail={
                "error": "group_video_participant_lookup_failed"
            },
        ) from exc

    if requester_id not in connected_ids:
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                UPDATE verifyd_voice_group_video_recordings
                SET
                    status = 'consent_pending',
                    status_updated_at = %s
                WHERE recording_id = %s
                  AND status = 'starting'
                """,
                (
                    _now_iso(),
                    recording_id,
                ),
            )
        raise HTTPException(
            status_code=409,
            detail={
                "error": "recording_requester_not_connected"
            },
        )

    _set_required_connected(
        recording_id,
        connected_ids,
        requester_id,
    )

    summary = _consent_summary(
        recording_id
    )
    pending = _pending_required_ids(
        recording_id
    )
    log.info(
        "Group recording consent gate: recording_id=%s connected=%s "
        "required=%s allowed=%s pending=%s declined=%s",
        recording_id,
        len(connected_ids),
        summary.get("required_count"),
        summary.get("allowed_count"),
        summary.get("pending_count"),
        summary.get("declined_count"),
    )

    if pending:
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                UPDATE verifyd_voice_group_video_recordings
                SET
                    status = 'consent_pending',
                    status_updated_at = %s
                WHERE recording_id = %s
                  AND status = 'starting'
                """,
                (
                    _now_iso(),
                    recording_id,
                ),
            )

        waiting = (
            _recording_by_id(recording_id)
            or recording
        )
        await _notify_recording_request(
            conversation_id,
            waiting,
            pending,
        )
        return waiting

    storage_key = _storage_key(
        str(session.get("session_id") or ""),
        recording_id,
    )

    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE verifyd_voice_group_video_recordings
            SET storage_key = %s
            WHERE recording_id = %s
              AND status = 'starting'
            """,
            (
                storage_key,
                recording_id,
            ),
        )

    log.info(
        "Starting LiveKit group recording: recording_id=%s "
        "room=%s start_timeout=%ss",
        recording_id,
        session.get("room_name") or "",
        _start_api_timeout_seconds(),
    )

    try:
        info = await _start_livekit_egress(
            session,
            storage_key,
        )
    except asyncio.TimeoutError as exc:
        log.error(
            "LiveKit StartEgress timed out: recording_id=%s",
            recording_id,
        )
        try:
            infos = await _list_livekit_egress(
                room_name=str(
                    session.get("room_name") or ""
                )
            )
        except Exception:
            infos = []

        if infos:
            return await _apply_egress_info(
                recording_id,
                infos[-1],
                source="start_timeout_room_lookup",
                notify=True,
            )

        _mark_failed(
            recording_id,
            "egress_start_timeout",
        )
        await _notify_group(
            conversation_id,
            {
                "type": "group_video_recording_failed",
                "group_id": conversation_id,
                "recording": _public_recording(
                    _recording_by_id(recording_id)
                ),
            },
        )
        raise HTTPException(
            status_code=503,
            detail={
                "error": "group_video_recording_start_timeout"
            },
        ) from exc
    except Exception as exc:
        log.exception(
            "LiveKit group recording start failed"
        )
        _mark_failed(
            recording_id,
            "egress_start_failed",
        )
        await _notify_group(
            conversation_id,
            {
                "type": "group_video_recording_failed",
                "group_id": conversation_id,
                "recording": _public_recording(
                    _recording_by_id(recording_id)
                ),
            },
        )
        raise HTTPException(
            status_code=503,
            detail={
                "error": "group_video_recording_start_failed"
            },
        ) from exc

    current = await _apply_egress_info(
        recording_id,
        info,
        source="start_response",
        notify=True,
    )

    if str(current.get("status") or "") == "stopping":
        egress_id = str(
            current.get("egress_id") or ""
        )
        if egress_id:
            try:
                stop_info = await _stop_livekit_egress(
                    egress_id
                )
                current = await _apply_egress_info(
                    recording_id,
                    stop_info,
                    source="start_stop_race",
                    notify=True,
                )
            except Exception:
                log.warning(
                    "Unable to settle Egress after startup/stop race: "
                    "recording_id=%s",
                    recording_id,
                    exc_info=True,
                )

    return (
        _recording_by_id(recording_id)
        or current
    )


def enforce_recording_join_policy(
    conversation_id: str,
    session_id: str,
    identity_id: str,
) -> None:
    recording = _open_recording(
        session_id
    )
    if not recording:
        return

    status = str(
        recording.get("status") or ""
    )

    if status == "stopping":
        raise HTTPException(
            status_code=409,
            detail={
                "error": "group_video_recording_stopping",
                "recording": _public_recording(
                    recording
                ),
            },
        )

    if status not in {
        "starting",
        "active",
    }:
        return

    consent = _consent_row(
        str(recording.get("recording_id") or ""),
        identity_id,
    )

    if (
        consent
        and consent.get("allowed") is True
    ):
        return

    raise HTTPException(
        status_code=409,
        detail={
            "error": "group_video_recording_consent_required",
            "message": (
                "This group video is being recorded. "
                "Consent is required before joining."
            ),
            "group_id": conversation_id,
            "recording": _public_recording(
                recording
            ),
        },
    )


async def handle_group_video_recording_presence(
    session: dict,
    participant_identity: str,
    connected: bool,
) -> None:
    """
    Enforces consent even if a previously issued LiveKit token is reused
    without hitting the backend /join route.
    """
    session_id = str(
        session.get("session_id") or ""
    )
    conversation_id = str(
        session.get("conversation_id") or ""
    )
    recording = _open_recording(
        session_id
    )
    if not recording:
        return

    recording_id = str(
        recording.get("recording_id") or ""
    )
    status = str(
        recording.get("status") or ""
    )
    identity_id = _identity_id_from_participant(
        conversation_id,
        participant_identity,
    )

    # Ignore non-member identities such as LiveKit recorder internals.
    if not identity_id:
        return

    consent = _consent_row(
        recording_id,
        identity_id,
    )

    if not connected:
        if consent:
            with get_db() as conn:
                cur = conn.cursor()
                cur.execute(
                    """
                    UPDATE verifyd_voice_group_video_recording_consents
                    SET required = FALSE
                    WHERE recording_id = %s
                      AND identity_id = %s
                    """,
                    (
                        recording_id,
                        identity_id,
                    ),
                )
        return

    if status not in {
        "starting",
        "active",
    }:
        return

    if (
        consent
        and consent.get("allowed") is True
    ):
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                UPDATE verifyd_voice_group_video_recording_consents
                SET required = TRUE
                WHERE recording_id = %s
                  AND identity_id = %s
                """,
                (
                    recording_id,
                    identity_id,
                ),
            )
        return

    try:
        await _remove_livekit_participant(
            str(session.get("room_name") or ""),
            participant_identity,
        )
    except Exception:
        log.exception(
            "Unable to remove unconsented participant"
        )
        # Fail closed: stop recording if removal fails.
        await _stop_internal(
            recording,
            "unconsented_participant_join",
            raise_on_failure=False,
        )

    await _send_to_identity(
        identity_id,
        {
            "type": "group_video_recording_join_blocked",
            "group_id": conversation_id,
            "recording": _public_recording(
                _recording_by_id(recording_id)
                or recording
            ),
        },
    )


async def handle_group_room_finished_recording(
    session: dict,
    event_time: str,
) -> None:
    recording = _open_recording(
        str(session.get("session_id") or "")
    )
    if not recording:
        return

    recording_id = str(
        recording.get("recording_id") or ""
    )
    status = str(
        recording.get("status") or ""
    )

    if status in {
        "consent_pending",
        "ready",
    }:
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                UPDATE verifyd_voice_group_video_recordings
                SET
                    status = 'canceled',
                    ended_at = %s,
                    stop_reason = 'room_finished'
                WHERE recording_id = %s
                  AND status IN ('consent_pending', 'ready')
                """,
                (
                    event_time,
                    recording_id,
                ),
            )
        return

    if status in {
        "starting",
        "active",
    }:
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                UPDATE verifyd_voice_group_video_recordings
                SET
                    status = 'stopping',
                    stop_reason = COALESCE(
                        stop_reason,
                        'room_finished'
                    )
                WHERE recording_id = %s
                  AND status IN ('starting', 'active')
                """,
                (recording_id,),
            )


def _recording_for_egress_info(info) -> Optional[dict]:
    egress_id = str(
        getattr(info, "egress_id", "") or ""
    )
    recording = _recording_by_egress_id(
        egress_id
    )
    if recording:
        return recording

    session = _session_by_room(
        str(getattr(info, "room_name", "") or "")
    )
    if not session:
        return None

    recording = _open_recording(
        str(session.get("session_id") or "")
    )
    if (
        recording
        and egress_id
        and not recording.get("egress_id")
    ):
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                UPDATE verifyd_voice_group_video_recordings
                SET egress_id = %s
                WHERE recording_id = %s
                  AND egress_id IS NULL
                """,
                (
                    egress_id,
                    recording.get("recording_id"),
                ),
            )
        recording = _recording_by_id(
            str(recording.get("recording_id") or "")
        )

    return recording


async def handle_livekit_recording_webhook(
    event,
    event_type: str,
    event_time: str,
) -> None:
    if event_type not in {
        "egress_started",
        "egress_updated",
        "egress_ended",
    }:
        return

    info = getattr(
        event,
        "egress_info",
        None,
    )
    if not info:
        return

    recording = _recording_for_egress_info(
        info
    )
    if not recording:
        log.warning(
            "LiveKit Egress webhook did not match a Trust Voice recording: "
            "event=%s egress_id=%s room=%s",
            event_type,
            getattr(info, "egress_id", "") or "",
            getattr(info, "room_name", "") or "",
        )
        return

    recording_id = str(
        recording.get("recording_id") or ""
    )
    log.info(
        "LiveKit recording webhook: recording_id=%s event=%s "
        "egress_id=%s status=%s",
        recording_id,
        event_type,
        getattr(info, "egress_id", "") or "",
        _egress_status_name(
            int(getattr(info, "status", 0) or 0)
        ),
    )

    await _apply_egress_info(
        recording_id,
        info,
        source=f"webhook:{event_type}",
        notify=True,
    )


def _require_recording_member(
    recording_id: str,
    authorization: Optional[str],
) -> tuple[dict, dict]:
    identity = _identity_from_bearer(
        authorization
    )
    recording = _recording_by_id(
        recording_id
    )
    if not recording:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "group_video_recording_not_found"
            },
        )

    _require_group_member(
        str(recording.get("conversation_id") or ""),
        str(identity["id"]),
    )
    return identity, recording


def _stream_private_recording(
    recording: dict,
    *,
    disposition: str,
    range_header: Optional[str] = None,
):
    storage_key = str(
        recording.get("storage_key") or ""
    )
    if (
        recording.get("status") != "complete"
        or not storage_key
    ):
        raise HTTPException(
            status_code=404,
            detail={
                "error": "group_video_recording_file_not_found"
            },
        )

    args = {
        "Bucket": _r2_bucket(),
        "Key": storage_key,
    }
    if (
        range_header
        and str(range_header).lower().startswith("bytes=")
    ):
        args["Range"] = str(
            range_header
        )

    try:
        obj = _r2_client().get_object(
            **args
        )
    except Exception as exc:
        log.exception(
            "Unable to open private group video recording"
        )
        raise HTTPException(
            status_code=404,
            detail={
                "error": "group_video_recording_file_not_found"
            },
        ) from exc

    body = obj["Body"]

    def iterator():
        try:
            while True:
                chunk = body.read(
                    1024 * 1024
                )
                if not chunk:
                    break
                yield chunk
        finally:
            try:
                body.close()
            except Exception:
                pass

    recording_id = str(
        recording.get("recording_id") or "recording"
    )
    filename = (
        "VeriFYD-Group-Video-"
        f"{recording_id}.mp4"
    )

    headers = {
        "Accept-Ranges": "bytes",
        "Cache-Control": "private, no-store",
        "Content-Disposition": (
            f'{disposition}; filename="{filename}"'
        ),
    }
    if obj.get("ContentLength") is not None:
        headers["Content-Length"] = str(
            obj["ContentLength"]
        )
    if obj.get("ContentRange"):
        headers["Content-Range"] = str(
            obj["ContentRange"]
        )

    return StreamingResponse(
        iterator(),
        status_code=(
            206
            if obj.get("ContentRange")
            else 200
        ),
        media_type="video/mp4",
        headers=headers,
    )


@router.get("/group-video/recording/health")
def recording_health():
    ensure_recording_schema()
    return {
        "status": "ok",
        "feature": "trust_voice_group_video_recording",
        "version": FEATURE_VERSION,
        "enabled": _enabled(),
        "configured": _configured(),
        "transport": "livekit_egress_v1",
        "capture": "room_composite_mp4_v1",
        "storage": (
            "private_r2_v1"
            if _storage_configured()
            else "not_configured"
        ),
        "consent": "all_connected_participants_v1",
        "late_join_policy": "explicit_consent_before_join_v1",
        "token_reuse_protection": "livekit_remove_participant_v1",
        "start_completion": "bounded_background_egress_v2",
        "stop_completion": "bounded_background_finalize_v2",
        "reconciliation": "webhook_plus_http_status_plus_r2_v2",
        "terminal_status_visibility": "latest_recording_v1",
        "start_api_timeout_seconds": _start_api_timeout_seconds(),
        "stop_api_timeout_seconds": _stop_api_timeout_seconds(),
        "finalize_timeout_seconds": _finalize_timeout_seconds(),
        "working_group_video_structure": "preserved",
    }


@router.get(
    "/groups/{conversation_id}/video/recording/status"
)
async def recording_status(
    conversation_id: str,
    authorization: Optional[str] = Header(default=None),
):
    identity = _identity_from_bearer(
        authorization
    )
    identity_id = str(
        identity["id"]
    )
    _require_group_member(
        conversation_id,
        identity_id,
    )

    session = _active_session(
        conversation_id
    )
    if not session:
        return {
            "active_session": False,
            "recording_enabled": _enabled(),
            "recording": None,
            "my_consent": None,
            "consent": None,
        }

    recording = _latest_recording(
        str(session.get("session_id") or "")
    )

    if recording and str(
        recording.get("status") or ""
    ) in {"starting", "stopping"}:
        try:
            recording = await _reconcile_recording_once(
                recording,
                source="http_status",
                notify=True,
            )
        except Exception:
            log.warning(
                "Recording status reconciliation failed: recording_id=%s",
                recording.get("recording_id") or "",
                exc_info=True,
            )
            recording = (
                _recording_by_id(
                    str(recording.get("recording_id") or "")
                )
                or recording
            )

    consent = (
        _consent_row(
            str(recording.get("recording_id") or ""),
            identity_id,
        )
        if recording
        else None
    )

    return {
        "active_session": True,
        "recording_enabled": _enabled(),
        "recording": _public_recording(
            recording
        ),
        "my_consent": (
            consent.get("allowed")
            if consent
            else None
        ),
        "consent": (
            _consent_summary(
                str(recording.get("recording_id") or "")
            )
            if recording
            else None
        ),
    }


@router.post(
    "/groups/{conversation_id}/video/recording/request"
)
async def request_recording(
    conversation_id: str,
    background_tasks: BackgroundTasks,
    authorization: Optional[str] = Header(default=None),
):
    _require_recording_available()

    identity = _identity_from_bearer(
        authorization
    )
    identity_id = str(
        identity["id"]
    )
    _require_group_member(
        conversation_id,
        identity_id,
    )

    session = _active_session(
        conversation_id
    )
    if not session:
        raise HTTPException(
            status_code=409,
            detail={"error": "no_active_group_video"},
        )

    existing = _open_recording(
        str(session.get("session_id") or "")
    )
    if existing:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "group_video_recording_already_open",
                "recording": _public_recording(
                    existing
                ),
            },
        )

    try:
        connected = await _connected_identity_ids(
            session
        )
    except Exception as exc:
        log.exception(
            "Unable to list participants for recording"
        )
        raise HTTPException(
            status_code=503,
            detail={
                "error": "group_video_participant_lookup_failed"
            },
        ) from exc

    if identity_id not in connected:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "recording_requester_not_connected"
            },
        )

    recording_id = (
        "tvgvr_" + uuid.uuid4().hex
    )

    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO verifyd_voice_group_video_recordings (
                recording_id,
                session_id,
                conversation_id,
                requested_by_identity_id,
                status,
                requested_at
            )
            VALUES (
                %s, %s, %s, %s,
                'consent_pending',
                %s
            )
            """,
            (
                recording_id,
                session.get("session_id"),
                conversation_id,
                identity_id,
                _now_iso(),
            ),
        )

    for participant_id in connected:
        _upsert_consent(
            recording_id,
            participant_id,
            True,
            (
                True
                if participant_id == identity_id
                else None
            ),
        )

    recording = (
        _recording_by_id(recording_id)
        or {}
    )
    pending = _pending_required_ids(
        recording_id
    )
    request_summary = _consent_summary(
        recording_id
    )
    log.info(
        "Group recording requested: recording_id=%s connected=%s "
        "required=%s allowed=%s pending=%s",
        recording_id,
        len(connected),
        request_summary.get("required_count"),
        request_summary.get("allowed_count"),
        request_summary.get("pending_count"),
    )

    if pending:
        await _notify_recording_request(
            conversation_id,
            recording,
            pending,
        )
    else:
        # Single-participant room: consent is already complete. Return READY
        # immediately and let LiveKit Egress start after the HTTP response.
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                UPDATE verifyd_voice_group_video_recordings
                SET
                    status = 'ready',
                    ready_at = COALESCE(ready_at, %s)
                WHERE recording_id = %s
                  AND status = 'consent_pending'
                """,
                (
                    _now_iso(),
                    recording_id,
                ),
            )

        recording = (
            _recording_by_id(recording_id)
            or recording
        )
        await _notify_group(
            conversation_id,
            {
                "type": "group_video_recording_ready",
                "group_id": conversation_id,
                "recording": _public_recording(
                    recording
                ),
            },
        )
        background_tasks.add_task(
            _begin_recording_background,
            recording_id,
        )

    return {
        "ok": True,
        "recording": _public_recording(
            recording
        ),
        "consent": _consent_summary(
            recording_id
        ),
    }


@router.post(
    "/groups/{conversation_id}/video/recording/{recording_id}/consent"
)
async def recording_consent(
    conversation_id: str,
    recording_id: str,
    payload: RecordingConsentPayload,
    background_tasks: BackgroundTasks,
    authorization: Optional[str] = Header(default=None),
):
    identity = _identity_from_bearer(
        authorization
    )
    identity_id = str(
        identity["id"]
    )
    _require_group_member(
        conversation_id,
        identity_id,
    )

    recording = _recording_by_id(
        recording_id
    )
    if (
        not recording
        or str(recording.get("conversation_id") or "")
        != conversation_id
    ):
        raise HTTPException(
            status_code=404,
            detail={
                "error": "group_video_recording_not_found"
            },
        )

    status = str(
        recording.get("status") or ""
    )
    if status in FINAL_STATUSES:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "group_video_recording_closed"
            },
        )

    existing = _consent_row(
        recording_id,
        identity_id,
    )

    if status in {
        "starting",
        "active",
        "stopping",
    }:
        if status == "stopping":
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "group_video_recording_stopping"
                },
            )

        if payload.allowed:
            # Late joiner can consent before retrying /video/join.
            _upsert_consent(
                recording_id,
                identity_id,
                bool(
                    existing
                    and existing.get("required")
                ),
                True,
            )
            return {
                "ok": True,
                "join_allowed": True,
                "recording": _public_recording(
                    _recording_by_id(recording_id)
                ),
                "my_consent": True,
            }

        # Declining before joining does not stop the existing recording.
        current_recorded = bool(
            existing
            and existing.get("required") is True
            and existing.get("allowed") is True
        )

        _upsert_consent(
            recording_id,
            identity_id,
            False,
            False,
        )

        if not current_recorded:
            return {
                "ok": True,
                "join_allowed": False,
                "recording": _public_recording(
                    _recording_by_id(recording_id)
                ),
                "my_consent": False,
            }

        stopping = _mark_recording_stopping(
            recording_id,
            "consent_withdrawn",
        )
        background_tasks.add_task(
            _stop_recording_background,
            recording_id,
            status,
        )
        await _notify_group(
            conversation_id,
            {
                "type": "group_video_recording_stopping",
                "group_id": conversation_id,
                "recording": _public_recording(
                    stopping
                ),
                "reason": "consent_withdrawn",
                "withdrawn_by": _public_contact(
                    identity
                ),
            },
        )
        return {
            "ok": True,
            "stopping": True,
            "recording": _public_recording(
                stopping
            ),
        }

    _upsert_consent(
        recording_id,
        identity_id,
        True,
        payload.allowed,
    )
    consent_summary = _consent_summary(
        recording_id
    )
    log.info(
        "Group recording consent response: recording_id=%s allowed=%s "
        "required=%s allowed_count=%s pending=%s declined=%s",
        recording_id,
        bool(payload.allowed),
        consent_summary.get("required_count"),
        consent_summary.get("allowed_count"),
        consent_summary.get("pending_count"),
        consent_summary.get("declined_count"),
    )

    if not payload.allowed:
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                UPDATE verifyd_voice_group_video_recordings
                SET
                    status = 'declined',
                    ended_at = %s,
                    stop_reason = 'consent_declined'
                WHERE recording_id = %s
                  AND status IN (
                      'consent_pending',
                      'ready'
                  )
                """,
                (
                    _now_iso(),
                    recording_id,
                ),
            )

        closed = (
            _recording_by_id(recording_id)
            or recording
        )
        await _notify_group(
            conversation_id,
            {
                "type": "group_video_recording_declined",
                "group_id": conversation_id,
                "recording": _public_recording(
                    closed
                ),
                "declined_by": _public_contact(
                    identity
                ),
            },
        )
        return {
            "ok": True,
            "declined": True,
            "recording": _public_recording(
                closed
            ),
        }

    pending = _pending_required_ids(
        recording_id
    )
    if pending:
        return {
            "ok": True,
            "ready": False,
            "recording": _public_recording(
                _recording_by_id(recording_id)
            ),
            "consent": _consent_summary(
                recording_id
            ),
        }

    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE verifyd_voice_group_video_recordings
            SET
                status = 'ready',
                ready_at = COALESCE(ready_at, %s)
            WHERE recording_id = %s
              AND status = 'consent_pending'
            """,
            (
                _now_iso(),
                recording_id,
            ),
        )

    await _notify_group(
        conversation_id,
        {
            "type": "group_video_recording_ready",
            "group_id": conversation_id,
            "recording": _public_recording(
                _recording_by_id(recording_id)
            ),
        },
    )

    # The final required Allow returns immediately in READY state.
    # LiveKit Egress startup happens after the response so the approving
    # participant never sits on a long-running consent request.
    ready_recording = (
        _recording_by_id(recording_id)
        or recording
    )
    background_tasks.add_task(
        _begin_recording_background,
        recording_id,
    )
    return {
        "ok": True,
        "ready": True,
        "recording": _public_recording(
            ready_recording
        ),
        "consent": _consent_summary(
            recording_id
        ),
    }


@router.post(
    "/groups/{conversation_id}/video/recording/{recording_id}/start"
)
async def start_recording(
    conversation_id: str,
    recording_id: str,
    authorization: Optional[str] = Header(default=None),
):
    # Recovery/retry route; normal UX auto-starts on final Allow.
    _require_recording_available()

    identity = _identity_from_bearer(
        authorization
    )
    identity_id = str(
        identity["id"]
    )
    group = _require_group_member(
        conversation_id,
        identity_id,
    )

    recording = _recording_by_id(
        recording_id
    )
    if (
        not recording
        or str(recording.get("conversation_id") or "")
        != conversation_id
    ):
        raise HTTPException(
            status_code=404,
            detail={
                "error": "group_video_recording_not_found"
            },
        )

    requester_id = str(
        recording.get("requested_by_identity_id") or ""
    )
    owner_id = str(
        group.get("owner_identity_id") or ""
    )

    if identity_id not in {
        requester_id,
        owner_id,
    }:
        raise HTTPException(
            status_code=403,
            detail={
                "error": "group_video_recording_start_not_allowed"
            },
        )

    started = await _begin_recording(
        recording_id
    )
    return {
        "ok": True,
        "recording": _public_recording(
            started
        ),
        "consent": _consent_summary(
            recording_id
        ),
    }


@router.post(
    "/groups/{conversation_id}/video/recording/stop"
)
async def stop_recording(
    conversation_id: str,
    background_tasks: BackgroundTasks,
    authorization: Optional[str] = Header(default=None),
):
    identity = _identity_from_bearer(
        authorization
    )
    identity_id = str(
        identity["id"]
    )
    group = _require_group_member(
        conversation_id,
        identity_id,
    )

    session = _active_session(
        conversation_id
    )
    if not session:
        raise HTTPException(
            status_code=409,
            detail={"error": "no_active_group_video"},
        )

    recording = _open_recording(
        str(session.get("session_id") or "")
    )
    if not recording:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "group_video_recording_not_active"
            },
        )

    recording_id = str(
        recording.get("recording_id") or ""
    )
    status = str(
        recording.get("status") or ""
    )

    # Before Egress starts, requester/owner may cancel the request.
    if status in {
        "consent_pending",
        "ready",
    }:
        requester_id = str(
            recording.get("requested_by_identity_id") or ""
        )
        owner_id = str(
            group.get("owner_identity_id") or ""
        )
        if identity_id not in {
            requester_id,
            owner_id,
        }:
            raise HTTPException(
                status_code=403,
                detail={
                    "error": "group_video_recording_cancel_not_allowed"
                },
            )

        with get_db() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                UPDATE verifyd_voice_group_video_recordings
                SET
                    status = 'canceled',
                    ended_at = %s,
                    stop_reason = 'canceled_by_requester'
                WHERE recording_id = %s
                  AND status IN (
                      'consent_pending',
                      'ready'
                  )
                """,
                (
                    _now_iso(),
                    recording_id,
                ),
            )

        canceled = (
            _recording_by_id(recording_id)
            or recording
        )
        await _notify_group(
            conversation_id,
            {
                "type": "group_video_recording_ended",
                "group_id": conversation_id,
                "recording": _public_recording(
                    canceled
                ),
            },
        )
        return {
            "ok": True,
            "recording": _public_recording(
                canceled
            ),
        }

    if status == "stopping":
        return {
            "ok": True,
            "recording": _public_recording(
                recording
            ),
        }

    consent = _consent_row(
        recording_id,
        identity_id,
    )
    requester_id = str(
        recording.get("requested_by_identity_id") or ""
    )
    owner_id = str(
        group.get("owner_identity_id") or ""
    )

    if not (
        (
            consent
            and consent.get("allowed") is True
            and consent.get("required") is True
        )
        or identity_id in {
            requester_id,
            owner_id,
        }
    ):
        raise HTTPException(
            status_code=403,
            detail={
                "error": "group_video_recording_stop_not_allowed"
            },
        )

    prior_status = status
    stopping = _mark_recording_stopping(
        recording_id,
        "stopped_by_participant",
    )
    background_tasks.add_task(
        _stop_recording_background,
        recording_id,
        prior_status,
    )
    await _notify_group(
        conversation_id,
        {
            "type": "group_video_recording_stopping",
            "group_id": conversation_id,
            "recording": _public_recording(
                stopping
            ),
            "reason": "stopped_by_participant",
            "stopped_by": _public_contact(
                identity
            ),
        },
    )
    return {
        "ok": True,
        "recording": _public_recording(
            stopping
        ),
    }


@router.get(
    "/group-video/recordings/{recording_id}"
)
def recording_detail(
    recording_id: str,
    authorization: Optional[str] = Header(default=None),
):
    identity, recording = _require_recording_member(
        recording_id,
        authorization,
    )
    consent = _consent_row(
        recording_id,
        str(identity["id"]),
    )
    return {
        "recording": _public_recording(
            recording
        ),
        "my_consent": (
            consent.get("allowed")
            if consent
            else None
        ),
        "consent": _consent_summary(
            recording_id
        ),
    }


@router.get(
    "/group-video/recordings/{recording_id}/media"
)
def stream_recording(
    recording_id: str,
    authorization: Optional[str] = Header(default=None),
    range_header: Optional[str] = Header(
        default=None,
        alias="Range",
    ),
):
    _identity, recording = _require_recording_member(
        recording_id,
        authorization,
    )
    return _stream_private_recording(
        recording,
        disposition="inline",
        range_header=range_header,
    )


@router.get(
    "/group-video/recordings/{recording_id}/download"
)
def download_recording(
    recording_id: str,
    authorization: Optional[str] = Header(default=None),
):
    _identity, recording = _require_recording_member(
        recording_id,
        authorization,
    )
    return _stream_private_recording(
        recording,
        disposition="attachment",
    )
