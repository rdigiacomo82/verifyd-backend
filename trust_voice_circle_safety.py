"""VeriFYD Circle Safety v1: permission-bound range exits and time-limited distress.

Disabled by default. Does not change the existing location-sharing contract,
call signaling, messages or voicemail. No emergency-service integration.
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import math
import os
import secrets
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

from database import get_db
from trust_voice_blocking import identities_blocked
from trust_voice_permissions import trust_circle_contains
from trust_voice_signaling import _identity_by_id
from trust_voice_messages import _identity_from_bearer

router = APIRouter(prefix="/trust-voice/safety", tags=["Trust Voice Circle Safety"])
log = logging.getLogger("verifyd.circle.safety")
_schema_lock = threading.Lock()
_schema_ready = False
MIN_RADIUS_M = 100
MAX_RADIUS_M = 32187  # 20 miles
PIN_ROUNDS = 260000
MAX_RECIPIENTS = 50
# Isolate network push I/O from calling, location and calendar request threads.
# Best effort: active incidents remain queryable and the scheduler retries.
_PUSH_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="verifyd-circle-safety-push")
_PUSH_SLOTS = threading.BoundedSemaphore(200)


def _enabled():
    return os.environ.get("VERIFYD_CIRCLE_SAFETY_ENABLED", "0").lower().strip() in {"1", "true", "yes", "on"}


def _required():
    if not _enabled():
        raise HTTPException(404, detail={"error": "circle_safety_unavailable"})


def _pilot_handles():
    """Explicit pilot allowlist; the safety feature fails closed if unset."""
    return {
        value.strip().lower().lstrip("@")
        for value in os.environ.get("VERIFYD_CIRCLE_SAFETY_PILOT_HANDLES", "").split(",")
        if value.strip()
    }


def _pilot_allowed(user):
    handle = str(user.get("handle_lower") or user.get("handle") or "").strip().lower().lstrip("@")
    return bool(handle and handle in _pilot_handles())


def _owner(auth):
    _required()
    user = _identity_from_bearer(auth)
    if str(user.get("status") or "active") != "active":
        raise HTTPException(401, detail={"error": "session_required"})
    if not _pilot_allowed(user):
        raise HTTPException(404, detail={"error": "circle_safety_unavailable"})
    return user


def _now():
    return datetime.now(timezone.utc)


def _iso(value):
    return value.isoformat() if value else None


def _schema():
    global _schema_ready
    if _schema_ready:
        return
    with _schema_lock:
        if _schema_ready:
            return
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS verifyd_circle_safety_settings (
                    identity_id TEXT PRIMARY KEY REFERENCES verifyd_identities(id) ON DELETE CASCADE,
                    allow_range_alerts BOOLEAN NOT NULL DEFAULT FALSE,
                    pin_salt TEXT, pin_digest TEXT,
                    pin_change_fail_count INTEGER NOT NULL DEFAULT 0,
                    pin_change_locked_until TIMESTAMPTZ,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS verifyd_circle_safety_zones (
                    zone_id TEXT PRIMARY KEY,
                    watcher_id TEXT NOT NULL REFERENCES verifyd_identities(id) ON DELETE CASCADE,
                    target_id TEXT NOT NULL REFERENCES verifyd_identities(id) ON DELETE CASCADE,
                    center_lat DOUBLE PRECISION NOT NULL,
                    center_lon DOUBLE PRECISION NOT NULL,
                    radius_m INTEGER NOT NULL,
                    last_state TEXT NOT NULL DEFAULT 'unknown',
                    enabled BOOLEAN NOT NULL DEFAULT TRUE,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    UNIQUE (watcher_id, target_id)
                )
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_verifyd_safety_zones_target ON verifyd_circle_safety_zones(target_id,enabled)")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS verifyd_circle_safety_alerts (
                    alert_id TEXT PRIMARY KEY,
                    recipient_id TEXT NOT NULL REFERENCES verifyd_identities(id) ON DELETE CASCADE,
                    actor_id TEXT NOT NULL REFERENCES verifyd_identities(id) ON DELETE CASCADE,
                    zone_id TEXT REFERENCES verifyd_circle_safety_zones(zone_id) ON DELETE SET NULL,
                    kind TEXT NOT NULL,
                    latitude DOUBLE PRECISION,
                    longitude DOUBLE PRECISION,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    dismissed_at TIMESTAMPTZ
                )
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_verifyd_safety_alerts_inbox ON verifyd_circle_safety_alerts(recipient_id,created_at DESC)")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS verifyd_circle_safety_incidents (
                    incident_id TEXT PRIMARY KEY,
                    sender_id TEXT NOT NULL REFERENCES verifyd_identities(id) ON DELETE CASCADE,
                    status TEXT NOT NULL DEFAULT 'active',
                    latitude DOUBLE PRECISION, longitude DOUBLE PRECISION,
                    accuracy_m DOUBLE PRECISION,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    last_position_at TIMESTAMPTZ,
                    expires_at TIMESTAMPTZ NOT NULL,
                    next_push_at TIMESTAMPTZ NOT NULL,
                    ended_at TIMESTAMPTZ,
                    pin_fail_count INTEGER NOT NULL DEFAULT 0,
                    pin_locked_until TIMESTAMPTZ
                )
            """)
            cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_verifyd_safety_one_active_incident ON verifyd_circle_safety_incidents(sender_id) WHERE status='active'")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS verifyd_circle_safety_recipients (
                    incident_id TEXT NOT NULL REFERENCES verifyd_circle_safety_incidents(incident_id) ON DELETE CASCADE,
                    recipient_id TEXT NOT NULL REFERENCES verifyd_identities(id) ON DELETE CASCADE,
                    muted BOOLEAN NOT NULL DEFAULT FALSE,
                    muted_at TIMESTAMPTZ,
                    PRIMARY KEY (incident_id, recipient_id)
                )
            """)
        _schema_ready = True


def _settings(identity_id):
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT allow_range_alerts,pin_salt,pin_digest FROM verifyd_circle_safety_settings WHERE identity_id=%s", (identity_id,))
        return cur.fetchone() or {}


def _hash_pin(pin, salt):
    return hashlib.pbkdf2_hmac("sha256", pin.encode("utf-8"), bytes.fromhex(salt), PIN_ROUNDS).hex()


def _check_pin(pin, row):
    return bool(row.get("pin_salt") and row.get("pin_digest") and
                hmac.compare_digest(_hash_pin(pin, row["pin_salt"]), row["pin_digest"]))


def _geo(lat, lon):
    if lat is None or lon is None:
        return
    if not math.isfinite(float(lat)) or not math.isfinite(float(lon)) or not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise HTTPException(422, detail={"error": "invalid_coordinates"})


def _distance_m(lat1, lon1, lat2, lon2):
    a, b, c, d = (math.radians(v) for v in (lat1, lon1, lat2, lon2))
    part = math.sin((c-a)/2)**2 + math.cos(a)*math.cos(c)*math.sin((d-b)/2)**2
    return 6371008.8 * 2 * math.asin(min(1, math.sqrt(part)))


def _location_granted(target_id, viewer_id, *, require_precise=True):
    """Match existing location /visible access check, including target's consent."""
    # The pilot must not reveal a non-pilot user's position to a pilot viewer.
    if not _pilot_allowed(_identity_by_id(str(target_id)) or {}):
        return False
    if not _pilot_allowed(_identity_by_id(str(viewer_id)) or {}):
        return False
    if identities_blocked(target_id, viewer_id):
        return False
    if not trust_circle_contains(target_id, viewer_id) or not trust_circle_contains(viewer_id, target_id):
        return False
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT 1 FROM verifyd_circle_location_shares s
            JOIN verifyd_circle_safety_settings p ON p.identity_id=s.owner_identity_id
            WHERE s.owner_identity_id=%s AND s.enabled=TRUE
              AND (s.expires_at IS NULL OR s.expires_at>NOW())
              AND p.allow_range_alerts=TRUE
              AND (%s=FALSE OR s.precision_mode='precise')
              AND (s.share_scope='all' OR EXISTS (
                   SELECT 1 FROM verifyd_circle_location_grants g
                   WHERE g.owner_identity_id=s.owner_identity_id AND g.viewer_identity_id=%s))
        """, (target_id, require_precise, viewer_id))
        return bool(cur.fetchone())


def _safety_push_authorized(identity_id, kind, event_id):
    """Recheck recipient, consent and incident state immediately before delivery.

    A queued red notification must not escape after a sender stops distress,
    a recipient mutes, or a monitored contact revokes location permission.
    """
    if not _enabled():
        return False
    _schema()
    with get_db() as conn:
        cur = conn.cursor()
        if kind == "circle_range_exit":
            cur.execute("""
                SELECT actor_id FROM verifyd_circle_safety_alerts
                WHERE alert_id=%s AND recipient_id=%s AND dismissed_at IS NULL
                  AND created_at>NOW()-INTERVAL '24 hours'
            """, (str(event_id), str(identity_id)))
            row = cur.fetchone()
            return bool(row and _location_granted(str(row["actor_id"]), str(identity_id)))
        if kind in {"circle_distress", "circle_distress_clear"}:
            status_rule = (
                "i.status='active' AND i.expires_at>NOW()"
                if kind == "circle_distress" else "i.status='stopped'"
            )
            cur.execute(f"""
                SELECT i.sender_id FROM verifyd_circle_safety_incidents i
                JOIN verifyd_circle_safety_recipients r ON r.incident_id=i.incident_id
                WHERE i.incident_id=%s AND r.recipient_id=%s AND r.muted=FALSE
                  AND ({status_rule})
            """, (str(event_id), str(identity_id)))
            row = cur.fetchone()
            if not row:
                return False
            sender_id = str(row["sender_id"])
            return bool(_pilot_allowed(_identity_by_id(sender_id) or {})
                        and _pilot_allowed(_identity_by_id(str(identity_id)) or {})
                        and trust_circle_contains(sender_id, str(identity_id))
                        and not identities_blocked(sender_id, str(identity_id)))
    return False


def _push(identity_id, *, kind, title, body, event_id):
    """Queue bounded best-effort push without blocking calls or location updates."""
    if not _PUSH_SLOTS.acquire(blocking=False):
        log.warning("Circle Safety push capacity reached; recipient can still view the incident in app")
        return False

    def deliver():
        try:
            if not _safety_push_authorized(identity_id, kind, event_id):
                return
            from trust_voice_push import notify_identity_push
            notify_identity_push(identity_id, notification_type=kind, title=title,
                                 body=body, url="/voice", tag=f"circle-safety-{event_id}",
                                 data={"safety_event_id": event_id}, ttl=600)
        except Exception:
            log.warning("Safety push delivery unavailable", exc_info=True)
        finally:
            _PUSH_SLOTS.release()

    try:
        _PUSH_POOL.submit(deliver)
        return True
    except Exception:
        _PUSH_SLOTS.release()
        log.warning("Circle Safety push enqueue unavailable", exc_info=True)
        return False


def _public_actor(actor_id):
    who = _identity_by_id(str(actor_id)) or {}
    return {"identity_id": str(actor_id), "handle": str(who.get("handle") or ""),
            "display_name": str(who.get("display_name") or "")}


class RangeConsent(BaseModel):
    enabled: bool


class PinChange(BaseModel):
    pin: str = Field(min_length=4, max_length=8, pattern=r"^[0-9]{4,8}$")
    current_pin: Optional[str] = Field(default=None, max_length=8)


class ZoneUpdate(BaseModel):
    center_lat: float = Field(ge=-90, le=90)
    center_lon: float = Field(ge=-180, le=180)
    radius_m: int = Field(ge=MIN_RADIUS_M, le=MAX_RADIUS_M)


class DistressStart(BaseModel):
    latitude: Optional[float] = Field(default=None, ge=-90, le=90)
    longitude: Optional[float] = Field(default=None, ge=-180, le=180)
    accuracy_m: Optional[float] = Field(default=None, ge=0, le=100000)


class DistressStop(BaseModel):
    pin: str = Field(min_length=4, max_length=8)


@router.get("/settings")
def safety_settings(authorization: Optional[str] = Header(default=None)):
    owner = _owner(authorization)
    _schema()
    row = _settings(str(owner["id"]))
    return {"enabled": True, "allow_range_alerts": bool(row.get("allow_range_alerts")),
            "pin_configured": bool(row.get("pin_digest")), "max_distress_minutes": 60}


@router.put("/settings/range-consent")
def update_range_consent(data: RangeConsent, authorization: Optional[str] = Header(default=None)):
    owner = _owner(authorization)
    _schema()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO verifyd_circle_safety_settings(identity_id,allow_range_alerts)
            VALUES(%s,%s) ON CONFLICT(identity_id) DO UPDATE SET
            allow_range_alerts=EXCLUDED.allow_range_alerts,updated_at=NOW()
        """, (str(owner["id"]), data.enabled))
        if not data.enabled:
            cur.execute("UPDATE verifyd_circle_safety_zones SET last_state='unknown' WHERE target_id=%s", (str(owner["id"]),))
            cur.execute("DELETE FROM verifyd_circle_safety_alerts WHERE actor_id=%s AND kind='range_exit'", (str(owner["id"]),))
    return {"allow_range_alerts": data.enabled}


@router.put("/settings/pin")
def update_pin(data: PinChange, authorization: Optional[str] = Header(default=None)):
    owner = _owner(authorization)
    _schema()
    owner_id = str(owner["id"])
    denied = False
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT pin_salt,pin_digest,pin_change_fail_count,pin_change_locked_until FROM verifyd_circle_safety_settings WHERE identity_id=%s FOR UPDATE", (owner_id,))
        old = cur.fetchone() or {}
        if old.get("pin_change_locked_until") and old["pin_change_locked_until"] > _now():
            raise HTTPException(429, detail={"error": "pin_temporarily_locked"})
        if old.get("pin_digest") and not _check_pin(str(data.current_pin or ""), old):
            count = int(old.get("pin_change_fail_count") or 0) + 1
            until = _now() + timedelta(seconds=60) if count % 5 == 0 else None
            cur.execute("UPDATE verifyd_circle_safety_settings SET pin_change_fail_count=%s,pin_change_locked_until=%s WHERE identity_id=%s", (count, until, owner_id))
            denied = True
        else:
            salt = secrets.token_hex(16)
            digest = _hash_pin(data.pin, salt)
            cur.execute("""
                INSERT INTO verifyd_circle_safety_settings(identity_id,pin_salt,pin_digest)
                VALUES(%s,%s,%s) ON CONFLICT(identity_id) DO UPDATE SET
                pin_salt=EXCLUDED.pin_salt,pin_digest=EXCLUDED.pin_digest,
                pin_change_fail_count=0,pin_change_locked_until=NULL,updated_at=NOW()
            """, (owner_id, salt, digest))
    if denied:
        raise HTTPException(403, detail={"error": "current_pin_required"})
    return {"pin_configured": True}


@router.get("/eligible")
def eligible_range_contacts(authorization: Optional[str] = Header(default=None)):
    """Only list Trust Circle contacts who explicitly permit range monitoring.

    Existing location consent is necessary but not sufficient. No coordinates
    are returned by this endpoint. This endpoint is unavailable when Safety is off.
    """
    owner = _owner(authorization)
    _schema()
    try:
        from trust_voice_location import _enabled as location_enabled, _schema as location_schema
        if not location_enabled():
            return {"eligible": [], "location_enabled": False}
        location_schema()
    except ImportError:
        return {"eligible": [], "location_enabled": False}
    viewer_id = str(owner["id"])
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT member_identity_id FROM verifyd_trust_circle_members
            WHERE owner_identity_id=%s LIMIT 100
        """, (viewer_id,))
        ids = [str(row["member_identity_id"]) for row in (cur.fetchall() or [])]
    eligible = []
    for target_id in dict.fromkeys(ids):
        if target_id == viewer_id or not _location_granted(target_id, viewer_id):
            continue
        person = _identity_by_id(target_id)
        if not person or str(person.get("status") or "active") != "active":
            continue
        eligible.append({**_public_actor(target_id), "precision": "precise"})
    return {"eligible": eligible, "location_enabled": True}


@router.get("/zones")
def get_zones(authorization: Optional[str] = Header(default=None)):
    owner = _owner(authorization)
    _schema()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT zone_id,target_id,center_lat,center_lon,radius_m,last_state,enabled FROM verifyd_circle_safety_zones WHERE watcher_id=%s ORDER BY updated_at DESC", (str(owner["id"]),))
        zones = [dict(row) for row in (cur.fetchall() or [])]
    return {"zones": [{**z, "target": _public_actor(z["target_id"])} for z in zones]}


@router.put("/zones/{target_id}")
def set_zone(target_id: str, data: ZoneUpdate, authorization: Optional[str] = Header(default=None)):
    owner = _owner(authorization)
    _schema()
    watcher_id = str(owner["id"])
    _geo(data.center_lat, data.center_lon)
    if str(target_id) == watcher_id or not _location_granted(str(target_id), watcher_id):
        raise HTTPException(403, detail={"error": "target_must_opt_in_and_share_precise_location"})
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT count(*) AS n FROM verifyd_circle_safety_zones WHERE watcher_id=%s", (watcher_id,))
        if int(cur.fetchone()["n"]) >= 20:
            cur.execute("SELECT 1 FROM verifyd_circle_safety_zones WHERE watcher_id=%s AND target_id=%s", (watcher_id, target_id))
            if not cur.fetchone():
                raise HTTPException(429, detail={"error": "max_twenty_range_alerts"})
        cur.execute("""
            INSERT INTO verifyd_circle_safety_zones(zone_id,watcher_id,target_id,center_lat,center_lon,radius_m)
            VALUES(%s,%s,%s,%s,%s,%s) ON CONFLICT(watcher_id,target_id) DO UPDATE SET
            center_lat=EXCLUDED.center_lat,center_lon=EXCLUDED.center_lon,
            radius_m=EXCLUDED.radius_m,last_state='unknown',enabled=TRUE,updated_at=NOW()
            RETURNING zone_id
        """, (str(uuid.uuid4()), watcher_id, str(target_id), data.center_lat, data.center_lon, data.radius_m))
        zone_id = cur.fetchone()["zone_id"]
    return {"ok": True, "zone_id": zone_id, "state": "unknown"}


@router.delete("/zones/{target_id}")
def delete_zone(target_id: str, authorization: Optional[str] = Header(default=None)):
    owner = _owner(authorization)
    _schema()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM verifyd_circle_safety_zones WHERE watcher_id=%s AND target_id=%s", (str(owner["id"]), str(target_id)))
    return {"ok": True}


def invalidate_range_for_owner(owner_id: str):
    """Privacy cleanup when existing location-sharing settings change.

    Remove cached range-alert coordinates and require a fresh inside fix.
    The original location-sharing operation succeeds even if this optional
    cleanup fails; no new coordinates can be read without current consent.
    """
    if not _enabled():
        return
    _schema()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            "DELETE FROM verifyd_circle_safety_alerts WHERE actor_id=%s AND kind='range_exit'",
            (str(owner_id),),
        )
        cur.execute(
            "UPDATE verifyd_circle_safety_zones SET last_state='unknown',updated_at=NOW() WHERE target_id=%s",
            (str(owner_id),),
        )


def evaluate_range_position(target_id: str):
    """Best-effort on each accepted location upload; first observation arms zone."""
    if not _enabled():
        return
    _schema()
    pending = []
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT p.latitude,p.longitude,p.accuracy_m,s.precision_mode
            FROM verifyd_circle_latest_locations p
            JOIN verifyd_circle_location_shares s ON s.owner_identity_id=p.owner_identity_id
            WHERE p.owner_identity_id=%s AND s.enabled=TRUE
              AND (s.expires_at IS NULL OR s.expires_at>NOW())
              AND p.recorded_at>NOW()-INTERVAL '2 minutes'
        """, (target_id,))
        point = cur.fetchone()
        if not point or point["precision_mode"] != "precise" or point["accuracy_m"] is None:
            return
        # Accuracy > 100m is too uncertain for a meaningful boundary crossing.
        accuracy = float(point["accuracy_m"])
        if accuracy > 100:
            return
        cur.execute("SELECT zone_id,watcher_id,center_lat,center_lon,radius_m,last_state FROM verifyd_circle_safety_zones WHERE target_id=%s AND enabled=TRUE FOR UPDATE", (target_id,))
        for z in (cur.fetchall() or []):
            watcher_id = str(z["watcher_id"])
            if not _location_granted(target_id, watcher_id):
                cur.execute("UPDATE verifyd_circle_safety_zones SET last_state='unknown' WHERE zone_id=%s", (z["zone_id"],))
                continue
            distance = _distance_m(point["latitude"], point["longitude"], z["center_lat"], z["center_lon"])
            margin = 25  # damp GPS drift around radius; measured accuracy also considered
            outside = distance - accuracy > z["radius_m"] + margin
            inside = distance + accuracy < z["radius_m"] - margin
            state = "outside" if outside else "inside" if inside else z["last_state"]
            if state == z["last_state"]:
                continue
            cur.execute("UPDATE verifyd_circle_safety_zones SET last_state=%s,updated_at=NOW() WHERE zone_id=%s", (state, z["zone_id"]))
            if z["last_state"] == "inside" and state == "outside":
                alert_id = str(uuid.uuid4())
                cur.execute("""
                    INSERT INTO verifyd_circle_safety_alerts(alert_id,recipient_id,actor_id,zone_id,kind,latitude,longitude)
                    VALUES (%s,%s,%s,%s,'range_exit',%s,%s)
                """, (alert_id, watcher_id, target_id, z["zone_id"], point["latitude"], point["longitude"]))
                pending.append((watcher_id, alert_id))
    actor = _public_actor(target_id)
    for watcher_id, alert_id in pending:
        _push(watcher_id, kind="circle_range_exit", title="Circle range alert",
              body=f"@{actor['handle']} has moved outside your selected range. Open VeriFYD Circle to check their shared location.", event_id=alert_id)


@router.get("/alerts")
def safety_alerts(authorization: Optional[str] = Header(default=None)):
    owner = _owner(authorization)
    _schema()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM verifyd_circle_safety_alerts WHERE created_at < NOW() - INTERVAL '24 hours'")
        cur.execute("SELECT alert_id,actor_id,kind,latitude,longitude,created_at FROM verifyd_circle_safety_alerts WHERE recipient_id=%s AND dismissed_at IS NULL ORDER BY created_at DESC LIMIT 50", (str(owner["id"]),))
        rows = cur.fetchall() or []
    visible = []
    for row in rows:
        if not _location_granted(str(row["actor_id"]), str(owner["id"])):
            continue
        visible.append({"alert_id": row["alert_id"], "kind": row["kind"],
                        "actor": _public_actor(row["actor_id"]), "latitude": row["latitude"],
                        "longitude": row["longitude"], "created_at": _iso(row["created_at"])})
    return {"alerts": visible}


@router.post("/alerts/{alert_id}/dismiss")
def dismiss_alert(alert_id: str, authorization: Optional[str] = Header(default=None)):
    owner = _owner(authorization)
    _schema()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("UPDATE verifyd_circle_safety_alerts SET dismissed_at=NOW() WHERE alert_id=%s AND recipient_id=%s", (alert_id, str(owner["id"])))
    return {"ok": True}


def _active_incident_for(cur, sender_id):
    cur.execute("""
        UPDATE verifyd_circle_safety_incidents SET status='expired',ended_at=NOW(),
            latitude=NULL,longitude=NULL,accuracy_m=NULL
        WHERE sender_id=%s AND status='active' AND expires_at <= NOW()
    """, (sender_id,))
    cur.execute("SELECT * FROM verifyd_circle_safety_incidents WHERE sender_id=%s AND status='active' AND expires_at>NOW() ORDER BY created_at DESC LIMIT 1", (sender_id,))
    return cur.fetchone()


@router.get("/distress/current")
def current_distress(authorization: Optional[str] = Header(default=None)):
    owner = _owner(authorization)
    _schema()
    with get_db() as conn:
        cur = conn.cursor()
        incident = _active_incident_for(cur, str(owner["id"]))
    return {"incident": _sender_incident(incident) if incident else None}


def _sender_incident(row):
    return {"incident_id": row["incident_id"], "created_at": _iso(row["created_at"]),
            "expires_at": _iso(row["expires_at"]), "latitude": row["latitude"],
            "longitude": row["longitude"], "status": row["status"]}


@router.post("/distress")
def start_distress(data: DistressStart, authorization: Optional[str] = Header(default=None)):
    owner = _owner(authorization)
    _schema()
    sender_id = str(owner["id"])
    if (data.latitude is None) != (data.longitude is None):
        raise HTTPException(422, detail={"error": "both_coordinates_required"})
    _geo(data.latitude, data.longitude)
    if data.accuracy_m is not None and not math.isfinite(data.accuracy_m):
        raise HTTPException(422, detail={"error": "invalid_accuracy"})
    if not _settings(sender_id).get("pin_digest"):
        raise HTTPException(409, detail={"error": "configure_distress_pin_first"})
    incident_id = str(uuid.uuid4())
    with get_db() as conn:
        cur = conn.cursor()
        incident = _active_incident_for(cur, sender_id)
        if incident:
            return {"ok": True, "already_active": True, "incident": _sender_incident(incident)}
        cur.execute("SELECT created_at FROM verifyd_circle_safety_incidents WHERE sender_id=%s ORDER BY created_at DESC LIMIT 1", (sender_id,))
        latest = cur.fetchone()
        if latest and (_now()-latest["created_at"]).total_seconds() < 60:
            raise HTTPException(429, detail={"error": "distress_cooldown"})
        cur.execute("SELECT member_identity_id FROM verifyd_trust_circle_members WHERE owner_identity_id=%s LIMIT 100", (sender_id,))
        candidate_ids = list({str(r["member_identity_id"]) for r in (cur.fetchall() or [])})
        recipients = [rid for rid in candidate_ids if rid != sender_id and not identities_blocked(sender_id, rid)
                      and _pilot_allowed(_identity_by_id(rid) or {})
                       and (_identity_by_id(rid) or {}).get("status", "active") == "active"][:MAX_RECIPIENTS]
        if not recipients:
            raise HTTPException(409, detail={"error": "no_eligible_circle_recipients"})
        cur.execute("""
            INSERT INTO verifyd_circle_safety_incidents(incident_id,sender_id,latitude,longitude,accuracy_m,created_at,last_position_at,expires_at,next_push_at)
            VALUES(%s,%s,%s,%s,%s,NOW(),%s,NOW()+INTERVAL '1 hour',NOW()+INTERVAL '5 minutes')
        """, (incident_id, sender_id, data.latitude, data.longitude, data.accuracy_m,
              _now() if data.latitude is not None else None))
        for rid in recipients:
            cur.execute("INSERT INTO verifyd_circle_safety_recipients(incident_id,recipient_id) VALUES(%s,%s)", (incident_id, rid))
    for rid in recipients:
        if not trust_circle_contains(sender_id, rid) or identities_blocked(sender_id, rid):
            continue
        _push(rid, kind="circle_distress", title="DISTRESS — VeriFYD Circle",
              body=f"@{owner.get('handle') or ''} triggered a distress alert. Open Circle for their last reported location. Not a 911 call.", event_id=incident_id)
    return {"ok": True, "already_active": False, "incident": {
        "incident_id": incident_id, "status": "active", "created_at": _iso(_now()),
        "expires_at": _iso(_now()+timedelta(hours=1)), "latitude": data.latitude, "longitude": data.longitude}}


@router.put("/distress/{incident_id}/position")
def update_distress_position(incident_id: str, data: DistressStart, authorization: Optional[str] = Header(default=None)):
    owner = _owner(authorization)
    _schema()
    if data.latitude is None or data.longitude is None:
        raise HTTPException(422, detail={"error": "coordinates_required"})
    _geo(data.latitude, data.longitude)
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""
            UPDATE verifyd_circle_safety_incidents
            SET latitude=%s,longitude=%s,accuracy_m=%s,last_position_at=NOW()
            WHERE incident_id=%s AND sender_id=%s AND status='active' AND expires_at>NOW()
              AND (last_position_at IS NULL OR last_position_at<NOW()-INTERVAL '10 seconds')
            RETURNING incident_id
        """, (data.latitude, data.longitude, data.accuracy_m, incident_id, str(owner["id"])))
        saved = bool(cur.fetchone())
    return {"ok": True, "updated": saved}


@router.post("/distress/{incident_id}/stop")
def stop_distress(incident_id: str, data: DistressStop, authorization: Optional[str] = Header(default=None)):
    owner = _owner(authorization)
    _schema()
    sender_id = str(owner["id"])
    settings = _settings(sender_id)
    bad_pin = False
    ids = []
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM verifyd_circle_safety_incidents WHERE incident_id=%s AND sender_id=%s FOR UPDATE", (incident_id, sender_id))
        row = cur.fetchone()
        if not row:
            raise HTTPException(404, detail={"error": "distress_not_found"})
        if row["status"] != "active" or row["expires_at"] <= _now():
            return {"ok": True, "status": "expired_or_stopped"}
        if row["pin_locked_until"] and row["pin_locked_until"] > _now():
            raise HTTPException(429, detail={"error": "pin_temporarily_locked"})
        if not _check_pin(data.pin, settings):
            count = int(row["pin_fail_count"]) + 1
            until = _now()+timedelta(seconds=60) if count % 5 == 0 else None
            cur.execute("UPDATE verifyd_circle_safety_incidents SET pin_fail_count=%s,pin_locked_until=%s WHERE incident_id=%s", (count, until, incident_id))
            bad_pin = True
        else:
            cur.execute("UPDATE verifyd_circle_safety_incidents SET status='stopped',ended_at=NOW(),latitude=NULL,longitude=NULL,accuracy_m=NULL WHERE incident_id=%s", (incident_id,))
            cur.execute("SELECT recipient_id FROM verifyd_circle_safety_recipients WHERE incident_id=%s AND muted=FALSE", (incident_id,))
            ids = [r["recipient_id"] for r in (cur.fetchall() or [])]
    if bad_pin:
        raise HTTPException(403, detail={"error": "incorrect_distress_pin"})
    for rid in ids:
        if trust_circle_contains(sender_id, str(rid)) and not identities_blocked(sender_id, str(rid)):
            _push(str(rid), kind="circle_distress_clear", title="Distress ended",
                  body=f"@{owner.get('handle') or ''} stopped their VeriFYD Circle distress alert.", event_id=incident_id)
    return {"ok": True, "status": "stopped"}


@router.get("/distress/inbox")
def distress_inbox(authorization: Optional[str] = Header(default=None)):
    owner = _owner(authorization)
    _schema()
    recipient_id = str(owner["id"])
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT i.incident_id,i.sender_id,i.created_at,i.expires_at,i.latitude,i.longitude,
                   i.last_position_at
            FROM verifyd_circle_safety_incidents i
            JOIN verifyd_circle_safety_recipients r ON r.incident_id=i.incident_id
            WHERE r.recipient_id=%s AND r.muted=FALSE AND i.status='active' AND i.expires_at>NOW()
            ORDER BY i.created_at DESC LIMIT 25
        """, (recipient_id,))
        rows = cur.fetchall() or []
    incidents = []
    for r in rows:
        if trust_circle_contains(str(r["sender_id"]), recipient_id) and not identities_blocked(str(r["sender_id"]), recipient_id):
            incidents.append({"incident_id": r["incident_id"], "sender": _public_actor(r["sender_id"]),
                              "latitude": r["latitude"], "longitude": r["longitude"],
                              "last_position_at": _iso(r["last_position_at"]),
                              "created_at": _iso(r["created_at"]), "expires_at": _iso(r["expires_at"])})
    return {"incidents": incidents}


@router.post("/distress/{incident_id}/mute")
def mute_distress(incident_id: str, authorization: Optional[str] = Header(default=None)):
    owner = _owner(authorization)
    _schema()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("UPDATE verifyd_circle_safety_recipients SET muted=TRUE,muted_at=NOW() WHERE incident_id=%s AND recipient_id=%s", (incident_id, str(owner["id"])))
    return {"ok": True, "muted": True}


def run_safety_dispatch_once():
    """Called by existing web push scheduler. Atomic claims prevent repeat blasts."""
    if not _enabled() or not _pilot_handles():
        return 0
    _schema()
    due = []
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("UPDATE verifyd_circle_safety_incidents SET status='expired',ended_at=NOW(),latitude=NULL,longitude=NULL,accuracy_m=NULL WHERE status='active' AND expires_at<=NOW()")
        cur.execute("DELETE FROM verifyd_circle_safety_alerts WHERE created_at<NOW()-INTERVAL '24 hours'")
        cur.execute("""
            SELECT incident_id,sender_id FROM verifyd_circle_safety_incidents
            WHERE status='active' AND expires_at>NOW() AND next_push_at<=NOW()
            ORDER BY next_push_at LIMIT 25 FOR UPDATE SKIP LOCKED
        """)
        due = [(r["incident_id"],r["sender_id"]) for r in (cur.fetchall() or [])]
        for incident_id, _ in due:
            cur.execute("UPDATE verifyd_circle_safety_incidents SET next_push_at=NOW()+INTERVAL '5 minutes' WHERE incident_id=%s", (incident_id,))
    for incident_id, sender_id in due:
        sender = _public_actor(sender_id)
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT r.recipient_id FROM verifyd_circle_safety_recipients r
                JOIN verifyd_circle_safety_incidents i ON i.incident_id=r.incident_id
                WHERE r.incident_id=%s AND r.muted=FALSE AND i.status='active' AND i.expires_at>NOW()
            """, (incident_id,))
            ids = [str(r["recipient_id"]) for r in (cur.fetchall() or [])]
        for rid in ids:
            if trust_circle_contains(str(sender_id), rid) and not identities_blocked(str(sender_id), rid):
                _push(rid, kind="circle_distress", title="DISTRESS — VeriFYD Circle",
                      body=f"@{sender['handle']} still has an active distress alert. Open Circle for last location.", event_id=incident_id)
    return len(due)