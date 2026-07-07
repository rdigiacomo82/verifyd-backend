import os
import re
import uuid
import json
import hmac
import hashlib
import base64
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, Field

try:
    import psycopg2
    import psycopg2.extras
    from psycopg2 import sql
except Exception:  # pragma: no cover
    psycopg2 = None
    sql = None


router = APIRouter()

FRONTEND_BASE_URL = os.getenv("FRONTEND_BASE_URL", "https://vfvid.com").rstrip("/")
DATABASE_URL = os.getenv("DATABASE_URL")

PHONE_HASH_SECRET = (
    os.getenv("TRUST_MESSAGE_PHONE_HASH_SECRET")
    or os.getenv("VERIFYD_SECRET_KEY")
    or os.getenv("SECRET_KEY")
    or "verifyd-local-phone-hash-change-me"
)


class SendTrustMessageRequest(BaseModel):
    certificate_id: str = Field(..., min_length=1)
    recipient_phone: str = Field(..., min_length=5)
    recipient_name: Optional[str] = ""
    sender_name: str = Field(..., min_length=1)
    sender_email: Optional[str] = ""
    message: Optional[str] = "Please review this certified VeriFYD verification record."
    include_verify_link: bool = True
    include_certificate_id: bool = True
    include_sender_info: bool = True


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _db_connect():
    if not DATABASE_URL:
        return None
    if psycopg2 is None:
        return None
    return psycopg2.connect(DATABASE_URL, cursor_factory=psycopg2.extras.RealDictCursor)


def normalize_phone(phone: str) -> str:
    """
    MVP E.164-ish normalizer.
    - Removes spaces, dashes, parentheses, periods.
    - Allows a leading +.
    - If 10 digits, assumes +1.
    - If 11 digits starting with 1, assumes +.
    """
    raw = (phone or "").strip()
    if not raw:
        raise ValueError("Recipient phone number is required.")

    has_plus = raw.startswith("+")
    digits = re.sub(r"\D", "", raw)

    if not digits:
        raise ValueError("Invalid phone number.")

    if has_plus:
        normalized = "+" + digits
    elif len(digits) == 10:
        normalized = "+1" + digits
    elif len(digits) == 11 and digits.startswith("1"):
        normalized = "+" + digits
    else:
        raise ValueError("Invalid phone number. Use a full phone number such as +1 555 555 1212.")

    if not re.fullmatch(r"\+[1-9]\d{7,14}", normalized):
        raise ValueError("Invalid phone number format.")

    return normalized


def mask_phone(phone: str) -> str:
    digits = re.sub(r"\D", "", phone or "")
    if len(digits) < 4:
        return "****"
    return "****" + digits[-4:]


def phone_last4(phone: str) -> str:
    digits = re.sub(r"\D", "", phone or "")
    return digits[-4:] if len(digits) >= 4 else ""


def hash_phone(phone: str) -> str:
    normalized = normalize_phone(phone)
    digest = hmac.new(
        PHONE_HASH_SECRET.encode("utf-8"),
        normalized.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return digest


def _json_safe(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _row_to_dict(row: Any) -> Dict[str, Any]:
    if not row:
        return {}
    return {k: _json_safe(v) for k, v in dict(row).items()}


def init_trust_message_table() -> None:
    conn = _db_connect()
    if conn is None:
        print("[trust-message] DATABASE_URL missing or psycopg2 unavailable; demo responses still work but records will not persist.")
        return

    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS trust_messages (
                        id SERIAL PRIMARY KEY,
                        trust_message_id TEXT UNIQUE NOT NULL,
                        certificate_id TEXT NOT NULL,
                        recipient_phone_hash TEXT,
                        recipient_phone_last4 TEXT,
                        recipient_phone_masked TEXT,
                        recipient_name TEXT,
                        sender_name TEXT,
                        sender_email TEXT,
                        message TEXT,
                        sms_body TEXT,
                        status TEXT,
                        provider TEXT,
                        provider_message_id TEXT,
                        trust_message_url TEXT,
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        sent_at TIMESTAMPTZ,
                        opened_at TIMESTAMPTZ
                    );
                    """
                )
    except Exception as exc:
        print(f"[trust-message] table init failed: {exc}")
    finally:
        conn.close()


def _list_public_tables(cur) -> list:
    cur.execute(
        """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
          AND table_type = 'BASE TABLE'
        ORDER BY table_name;
        """
    )
    return [r["table_name"] for r in cur.fetchall()]


def _table_columns(cur, table_name: str) -> set:
    cur.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = %s;
        """,
        (table_name,),
    )
    return {r["column_name"] for r in cur.fetchall()}


def lookup_certificate(certificate_id: str) -> Optional[Dict[str, Any]]:
    """
    Flexible lookup to avoid depending on one exact certificate table schema.
    Tries likely tables/columns and compares as text.
    """
    cert_id = (certificate_id or "").strip()
    if not cert_id:
        return None

    conn = _db_connect()
    if conn is None:
        return None

    preferred_tables = [
        "certificates",
        "certificate_records",
        "cert",
        "results",
        "analysis_results",
        "jobs",
    ]

    id_columns = [
        "certificate_id",
        "certificateId",
        "cert_id",
        "certId",
        "verification_id",
        "verificationId",
        "job_id",
        "jobId",
        "id",
    ]

    try:
        with conn.cursor() as cur:
            all_tables = _list_public_tables(cur)
            ordered_tables = []
            for t in preferred_tables:
                if t in all_tables and t not in ordered_tables:
                    ordered_tables.append(t)
            for t in all_tables:
                if t not in ordered_tables and t != "trust_messages":
                    ordered_tables.append(t)

            for table_name in ordered_tables:
                columns = _table_columns(cur, table_name)
                matching_cols = [c for c in id_columns if c in columns]
                if not matching_cols:
                    continue

                for col_name in matching_cols:
                    try:
                        query = sql.SQL("SELECT * FROM {} WHERE {}::text = %s LIMIT 1").format(
                            sql.Identifier(table_name),
                            sql.Identifier(col_name),
                        )
                        cur.execute(query, (cert_id,))
                        row = cur.fetchone()
                        if row:
                            data = _row_to_dict(row)
                            data["_source_table"] = table_name
                            data["_source_column"] = col_name
                            return data
                    except Exception:
                        conn.rollback()
                        continue
    except Exception as exc:
        print(f"[trust-message] certificate lookup failed: {exc}")
        return None
    finally:
        conn.close()

    return None


def _first_value(data: Dict[str, Any], keys: list, default: Any = None) -> Any:
    for key in keys:
        if key in data and data[key] not in (None, ""):
            return data[key]
    return default


def _as_int_or_none(value: Any) -> Optional[int]:
    try:
        if value is None or value == "":
            return None
        return int(round(float(value)))
    except Exception:
        return None


def certificate_public_summary(certificate_id: str, cert: Dict[str, Any]) -> Dict[str, Any]:
    label = _first_value(
        cert,
        ["label", "result_label", "verdict", "classification", "status", "display_label"],
        "UNKNOWN",
    )

    authenticity = _as_int_or_none(
        _first_value(
            cert,
            ["authenticity_score", "authenticity", "score", "certification_score"],
            None,
        )
    )

    ai_score = _as_int_or_none(
        _first_value(
            cert,
            ["ai_score", "ai_probability", "ai_indicators", "combined_ai_score", "combined_score"],
            None,
        )
    )

    if authenticity is None and ai_score is not None:
        authenticity = max(0, min(100, 100 - ai_score))

    if ai_score is None and authenticity is not None:
        ai_score = max(0, min(100, 100 - authenticity))

    sha256 = _first_value(
        cert,
        ["sha256", "sha256_hash", "file_hash", "original_sha256", "hash"],
        None,
    )

    media_type = _first_value(
        cert,
        ["media_type", "file_type", "type", "content_type"],
        None,
    )

    filename = _first_value(
        cert,
        ["filename", "original_filename", "file_name", "name"],
        None,
    )

    return {
        "certificate_id": certificate_id,
        "label": str(label).upper() if label else "UNKNOWN",
        "authenticity_score": authenticity,
        "ai_score": ai_score,
        "sha256": sha256,
        "media_type": media_type,
        "filename": filename,
        "verify_url": f"{FRONTEND_BASE_URL}/verify-certificate/{certificate_id}",
    }


def build_trust_message_sms(
    sender_name: str,
    trust_message_url: str,
    certificate_id: str,
    include_certificate_id: bool = True,
) -> str:
    sender = (sender_name or "A VeriFYD user").strip()
    body = (
        f"VeriFYD Trust Message from {sender}: "
        f"A certified verification record was shared with you. "
        f"View: {trust_message_url}."
    )
    if include_certificate_id:
        body += f" Certificate ID: {certificate_id}."
    body += " Reply STOP to opt out."
    return body


def send_sms_via_twilio(to_phone: str, body: str) -> Dict[str, Any]:
    sid = os.getenv("TWILIO_ACCOUNT_SID")
    token = os.getenv("TWILIO_AUTH_TOKEN")
    from_number = os.getenv("TWILIO_FROM_NUMBER")

    if not sid or not token or not from_number:
        return {
            "sent": True,
            "provider": "demo",
            "status": "demo_created",
            "provider_message_id": None,
            "sms_preview": body,
        }

    url = f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json"
    payload = urllib.parse.urlencode(
        {
            "From": from_number,
            "To": to_phone,
            "Body": body,
        }
    ).encode("utf-8")

    auth_raw = f"{sid}:{token}".encode("utf-8")
    auth_header = base64.b64encode(auth_raw).decode("ascii")

    request = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Authorization": f"Basic {auth_header}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            text = response.read().decode("utf-8", errors="replace")
            data = json.loads(text) if text else {}
            return {
                "sent": True,
                "provider": "twilio",
                "status": "sent",
                "provider_message_id": data.get("sid"),
                "sms_preview": None,
            }
    except Exception as exc:
        return {
            "sent": False,
            "provider": "twilio",
            "status": "failed",
            "provider_message_id": None,
            "error": str(exc),
            "sms_preview": None,
        }


def create_trust_message_record(
    trust_message_id: str,
    certificate_id: str,
    normalized_phone: str,
    recipient_name: str,
    sender_name: str,
    sender_email: str,
    message: str,
    sms_body: str,
    status: str,
    provider: str,
    provider_message_id: Optional[str],
    trust_message_url: str,
) -> None:
    conn = _db_connect()
    if conn is None:
        return

    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO trust_messages (
                        trust_message_id,
                        certificate_id,
                        recipient_phone_hash,
                        recipient_phone_last4,
                        recipient_phone_masked,
                        recipient_name,
                        sender_name,
                        sender_email,
                        message,
                        sms_body,
                        status,
                        provider,
                        provider_message_id,
                        trust_message_url,
                        sent_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                    ON CONFLICT (trust_message_id) DO NOTHING;
                    """,
                    (
                        trust_message_id,
                        certificate_id,
                        hash_phone(normalized_phone),
                        phone_last4(normalized_phone),
                        mask_phone(normalized_phone),
                        recipient_name or "",
                        sender_name or "",
                        sender_email or "",
                        message or "",
                        sms_body,
                        status,
                        provider,
                        provider_message_id,
                        trust_message_url,
                    ),
                )
    except Exception as exc:
        print(f"[trust-message] record insert failed: {exc}")
    finally:
        conn.close()


def get_trust_message_record(trust_message_id: str) -> Optional[Dict[str, Any]]:
    conn = _db_connect()
    if conn is None:
        return None

    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT *
                    FROM trust_messages
                    WHERE trust_message_id = %s
                    LIMIT 1;
                    """,
                    (trust_message_id,),
                )
                row = cur.fetchone()
                if not row:
                    return None

                record = _row_to_dict(row)

                if not record.get("opened_at"):
                    cur.execute(
                        """
                        UPDATE trust_messages
                        SET opened_at = NOW()
                        WHERE trust_message_id = %s
                          AND opened_at IS NULL;
                        """,
                        (trust_message_id,),
                    )

                return record
    except Exception as exc:
        print(f"[trust-message] record lookup failed: {exc}")
        return None
    finally:
        conn.close()


@router.post("/send-trust-message/")
async def send_trust_message(payload: SendTrustMessageRequest):
    certificate_id = (payload.certificate_id or "").strip()
    if not certificate_id:
        raise HTTPException(status_code=400, detail="certificate_id is required.")

    cert = lookup_certificate(certificate_id)
    if not cert:
        raise HTTPException(
            status_code=404,
            detail={
                "sent": False,
                "status": "not_found",
                "error": "Certificate not found.",
            },
        )

    try:
        normalized_phone = normalize_phone(payload.recipient_phone)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    if is_phone_opted_out(normalized_phone):
        raise HTTPException(
            status_code=403,
            detail={
                "sent": False,
                "status": "opted_out",
                "error": "This recipient has opted out of Trust Message SMS delivery.",
            },
        )

    trust_message_id = "VFYD-TM-" + uuid.uuid4().hex[:10].upper()
    trust_message_url = f"{FRONTEND_BASE_URL}/trust-message/{trust_message_id}"

    sms_body = build_trust_message_sms(
        sender_name=payload.sender_name,
        trust_message_url=trust_message_url,
        certificate_id=certificate_id,
        include_certificate_id=payload.include_certificate_id,
    )

    print(f"[trust-message] creating {trust_message_id} for certificate {certificate_id}")

    send_result = send_sms_via_twilio(normalized_phone, sms_body)

    create_trust_message_record(
        trust_message_id=trust_message_id,
        certificate_id=certificate_id,
        normalized_phone=normalized_phone,
        recipient_name=payload.recipient_name or "",
        sender_name=payload.sender_name or "",
        sender_email=payload.sender_email or "",
        message=payload.message or "",
        sms_body=sms_body,
        status=send_result.get("status") or "unknown",
        provider=send_result.get("provider") or "unknown",
        provider_message_id=send_result.get("provider_message_id"),
        trust_message_url=trust_message_url,
    )

    if not send_result.get("sent"):
        print(f"[trust-message] failed {trust_message_id}: {send_result.get('error')}")
        raise HTTPException(
            status_code=502,
            detail={
                "sent": False,
                "status": "failed",
                "error": send_result.get("error") or "SMS provider failed.",
                "trust_message_id": trust_message_id,
            },
        )

    print(f"[trust-message] {send_result.get('status')} {trust_message_id}")

    response = {
        "sent": True,
        "status": send_result.get("status"),
        "provider": send_result.get("provider"),
        "trust_message_id": trust_message_id,
        "trust_message_url": trust_message_url,
        "certificate_id": certificate_id,
        "recipient_phone_masked": mask_phone(normalized_phone),
        "message": payload.message or "",
        "certificate": certificate_public_summary(certificate_id, cert),
    }

    if send_result.get("provider_message_id"):
        response["provider_message_id"] = send_result.get("provider_message_id")

    if send_result.get("sms_preview"):
        response["sms_preview"] = send_result.get("sms_preview")

    return response


@router.get("/trust-message/{trust_message_id}")
async def get_trust_message(trust_message_id: str):
    trust_message_id = (trust_message_id or "").strip()
    if not trust_message_id:
        raise HTTPException(status_code=400, detail="trust_message_id is required.")

    record = get_trust_message_record(trust_message_id)
    if not record:
        raise HTTPException(
            status_code=404,
            detail={
                "found": False,
                "status": "not_found",
                "error": "Trust Message not found.",
            },
        )

    certificate_id = record.get("certificate_id")
    cert = lookup_certificate(certificate_id) if certificate_id else None

    certificate_summary = (
        certificate_public_summary(certificate_id, cert)
        if cert and certificate_id
        else {
            "certificate_id": certificate_id,
            "verify_url": f"{FRONTEND_BASE_URL}/verify-certificate/{certificate_id}" if certificate_id else None,
        }
    )

    return {
        "found": True,
        "trust_message_id": record.get("trust_message_id"),
        "certificate_id": certificate_id,
        "sender_name": record.get("sender_name"),
        "sender_email": record.get("sender_email"),
        "recipient_name": record.get("recipient_name"),
        "recipient_phone_masked": record.get("recipient_phone_masked"),
        "message": record.get("message"),
        "status": record.get("status"),
        "provider": record.get("provider"),
        "trust_message_url": record.get("trust_message_url"),
        "created_at": _json_safe(record.get("created_at")),
        "sent_at": _json_safe(record.get("sent_at")),
        "opened_at": _json_safe(record.get("opened_at")),
        "certificate": certificate_summary,
        "safety_notice": (
            "VeriFYD will never ask for your password, bank login, or one-time code. "
            "Only trust links beginning with https://vfvid.com/. "
            "If this message asks you to make a payment or change banking details, "
            "independently verify through a known contact method."
        ),
    }




# --- VeriFYD Trust Message Twilio inbound/status support ---

def init_trust_message_opt_out_table() -> None:
    conn = _db_connect()
    if conn is None:
        print("[trust-message] opt-out table not initialized because DATABASE_URL is unavailable.")
        return

    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS trust_message_opt_outs (
                        id SERIAL PRIMARY KEY,
                        recipient_phone_hash TEXT UNIQUE NOT NULL,
                        recipient_phone_last4 TEXT,
                        recipient_phone_masked TEXT,
                        status TEXT DEFAULT 'opted_out',
                        source TEXT,
                        raw_body TEXT,
                        provider_message_id TEXT,
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        updated_at TIMESTAMPTZ DEFAULT NOW()
                    );
                    """
                )
    except Exception as exc:
        print(f"[trust-message] opt-out table init failed: {exc}")
    finally:
        conn.close()


def is_phone_opted_out(normalized_phone: str) -> bool:
    conn = _db_connect()
    if conn is None:
        return False

    try:
        phone_hash = hash_phone(normalized_phone)
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT 1
                FROM trust_message_opt_outs
                WHERE recipient_phone_hash = %s
                  AND status = 'opted_out'
                LIMIT 1;
                """,
                (phone_hash,),
            )
            return cur.fetchone() is not None
    except Exception as exc:
        print(f"[trust-message] opt-out lookup failed: {exc}")
        return False
    finally:
        conn.close()


def record_phone_opt_out(
    normalized_phone: str,
    raw_body: str = "",
    provider_message_id: str = "",
    source: str = "twilio",
) -> None:
    conn = _db_connect()
    if conn is None:
        print("[trust-message] opt-out not persisted because DATABASE_URL is unavailable.")
        return

    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO trust_message_opt_outs (
                        recipient_phone_hash,
                        recipient_phone_last4,
                        recipient_phone_masked,
                        status,
                        source,
                        raw_body,
                        provider_message_id,
                        updated_at
                    )
                    VALUES (%s, %s, %s, 'opted_out', %s, %s, %s, NOW())
                    ON CONFLICT (recipient_phone_hash)
                    DO UPDATE SET
                        status = 'opted_out',
                        source = EXCLUDED.source,
                        raw_body = EXCLUDED.raw_body,
                        provider_message_id = EXCLUDED.provider_message_id,
                        updated_at = NOW();
                    """,
                    (
                        hash_phone(normalized_phone),
                        phone_last4(normalized_phone),
                        mask_phone(normalized_phone),
                        source,
                        raw_body or "",
                        provider_message_id or "",
                    ),
                )
    except Exception as exc:
        print(f"[trust-message] opt-out insert failed: {exc}")
    finally:
        conn.close()


def update_twilio_message_status(
    provider_message_id: str,
    message_status: str,
    error_code: str = "",
    error_message: str = "",
) -> None:
    if not provider_message_id:
        return

    conn = _db_connect()
    if conn is None:
        return

    status_value = f"twilio:{message_status}" if message_status else "twilio:status_callback"

    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE trust_messages
                    SET status = %s
                    WHERE provider_message_id = %s;
                    """,
                    (status_value, provider_message_id),
                )
    except Exception as exc:
        print(
            f"[trust-message] Twilio status update failed: {exc} "
            f"error_code={error_code} error_message={error_message}"
        )
    finally:
        conn.close()


async def parse_twilio_form(request: Request) -> Dict[str, str]:
    body = await request.body()
    parsed = urllib.parse.parse_qs(body.decode("utf-8", errors="replace"))
    return {key: values[0] if values else "" for key, values in parsed.items()}


@router.post("/twilio/sms-webhook/")
async def twilio_sms_webhook(request: Request):
    """
    Twilio inbound SMS webhook for STOP/HELP handling.

    Later, configure this on the Twilio number:
    https://verifyd-backend.onrender.com/twilio/sms-webhook/
    """
    form = await parse_twilio_form(request)

    from_phone = form.get("From") or ""
    body = (form.get("Body") or "").strip()
    message_sid = form.get("MessageSid") or form.get("SmsMessageSid") or ""

    normalized_body = re.sub(r"\s+", " ", body.upper()).strip()

    stop_words = {"STOP", "STOPALL", "UNSUBSCRIBE", "CANCEL", "END", "QUIT"}
    help_words = {"HELP", "INFO"}

    print(f"[trust-message] inbound SMS from={mask_phone(from_phone)} body={normalized_body}")

    try:
        normalized_phone = normalize_phone(from_phone)
    except Exception:
        normalized_phone = ""

    if normalized_body in stop_words and normalized_phone:
        record_phone_opt_out(
            normalized_phone,
            raw_body=body,
            provider_message_id=message_sid,
            source="twilio",
        )
        return Response(
            content="You have been opted out of VeriFYD Trust Message alerts. Reply HELP for help.",
            media_type="text/plain",
        )

    if normalized_body in help_words:
        return Response(
            content=(
                "VeriFYD Trust Message shares certified verification links. "
                "We will never ask for passwords, bank logins, or one-time codes. "
                "Reply STOP to opt out."
            ),
            media_type="text/plain",
        )

    return Response(
        content="VeriFYD Trust Message received your reply. Reply STOP to opt out or HELP for help.",
        media_type="text/plain",
    )


@router.post("/twilio/status-callback/")
async def twilio_status_callback(request: Request):
    """
    Twilio delivery status callback.

    Optional later:
    TWILIO_STATUS_CALLBACK_URL=https://verifyd-backend.onrender.com/twilio/status-callback/
    """
    form = await parse_twilio_form(request)

    message_sid = form.get("MessageSid") or form.get("SmsMessageSid") or ""
    message_status = form.get("MessageStatus") or form.get("SmsStatus") or ""
    to_phone = form.get("To") or ""
    error_code = form.get("ErrorCode") or ""
    error_message = form.get("ErrorMessage") or ""

    print(
        f"[trust-message] Twilio status sid={message_sid} "
        f"status={message_status} to={mask_phone(to_phone)} error={error_code}"
    )

    update_twilio_message_status(
        provider_message_id=message_sid,
        message_status=message_status,
        error_code=error_code,
        error_message=error_message,
    )

    return {"ok": True}

# --- End VeriFYD Trust Message Twilio inbound/status support ---


init_trust_message_table()
init_trust_message_opt_out_table()
