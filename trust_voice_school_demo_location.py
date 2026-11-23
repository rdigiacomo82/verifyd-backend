"""Explicit opt-in location for *practice* school reports only.

No live alert, student tracking, 911 transmission, or background location.
Location expires two hours after upload; staff see only reports in their school.
This demo feature is OFF by default and must be scoped to an allowlisted school.
"""
from __future__ import annotations
import math
import os
import threading
from datetime import datetime, timedelta, timezone
from typing import Optional
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field
from database import get_db
from trust_voice_school_safety_training import _actor, _audit, _membership, _schema as _training_schema
from trust_voice_school_safety_registration import _school_admin

router=APIRouter(prefix="/trust-voice/school-safety",tags=["School Safety Demo Location"])
_ready=False
_lock=threading.Lock()

def _enabled_for(school_id:str):
    active=os.getenv("VERIFYD_SAFETY_DEMO_LOCATION_ENABLED","0").lower().strip() in {"1","true","on","yes"}
    schools={x.strip() for x in os.getenv("VERIFYD_SAFETY_DEMO_LOCATION_SCHOOL_IDS","").split(",") if x.strip()}
    if not active or school_id not in schools:
        raise HTTPException(403,detail={"error":"demo_location_not_enabled_for_school"})

def _schema():
    global _ready
    _training_schema()
    if _ready:return
    with _lock:
        if _ready:return
        with get_db() as conn:
            conn.cursor().execute("""CREATE TABLE IF NOT EXISTS verifyd_safety_demo_report_locations(
                report_id TEXT PRIMARY KEY REFERENCES verifyd_safety_training_reports(report_id) ON DELETE CASCADE,
                school_id TEXT NOT NULL REFERENCES verifyd_safety_training_schools(school_id),
                student_id TEXT NOT NULL REFERENCES verifyd_identities(id),
                latitude DOUBLE PRECISION NOT NULL CHECK(latitude BETWEEN -90 AND 90),
                longitude DOUBLE PRECISION NOT NULL CHECK(longitude BETWEEN -180 AND 180),
                accuracy_m DOUBLE PRECISION NOT NULL CHECK(accuracy_m BETWEEN 1 AND 5000),
                captured_at TIMESTAMPTZ NOT NULL,
                shared_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                expires_at TIMESTAMPTZ NOT NULL DEFAULT (NOW()+INTERVAL '2 hours')
            )""")
        _ready=True

class DemoLocation(BaseModel):
    latitude: float=Field(ge=-90,le=90)
    longitude: float=Field(ge=-180,le=180)
    accuracy_m: float=Field(ge=1,le=5000)
    captured_at: datetime
    explicit_opt_in: bool=False

def _purge(cur):
    cur.execute("DELETE FROM verifyd_safety_demo_report_locations WHERE expires_at<=NOW()")

@router.post("/reports/{report_id}/demo-location")
def share_demo_location(report_id:str,payload:DemoLocation,authorization:Optional[str]=Header(default=None)):
    actor=_actor(authorization);_schema()
    if not payload.explicit_opt_in:
        raise HTTPException(422,detail={"error":"explicit_location_permission_required"})
    if any(not math.isfinite(v) for v in (payload.latitude,payload.longitude,payload.accuracy_m)):
        raise HTTPException(422,detail={"error":"invalid_coordinates"})
    if payload.captured_at.tzinfo is None:
        raise HTTPException(422,detail={"error":"timestamp_timezone_required"})
    now=datetime.now(timezone.utc)
    when=payload.captured_at.astimezone(timezone.utc)
    if when<now-timedelta(minutes=5) or when>now+timedelta(seconds=30):
        raise HTTPException(422,detail={"error":"location_timestamp_out_of_range"})
    student_id=str(actor["id"])
    with get_db() as conn:
        cur=conn.cursor()
        _purge(cur)
        cur.execute("""SELECT r.school_id,r.student_id,r.closed_at,i.status AS incident_status
            FROM verifyd_safety_training_reports r
            JOIN verifyd_safety_training_incidents i ON i.incident_id=r.incident_id
            WHERE r.report_id=%s""",(report_id,))
        record=cur.fetchone()
        if not record or str(record["student_id"])!=student_id:
            raise HTTPException(404,detail={"error":"practice_report_unavailable"})
        _enabled_for(record["school_id"])
        _membership(cur,record["school_id"],student_id,{"student"})
        if record["closed_at"] is not None or record["incident_status"]!="active":
            raise HTTPException(409,detail={"error":"practice_report_closed"})
        cur.execute("""INSERT INTO verifyd_safety_demo_report_locations(
            report_id,school_id,student_id,latitude,longitude,accuracy_m,captured_at,shared_at,expires_at)
            VALUES(%s,%s,%s,%s,%s,%s,%s,NOW(),NOW()+INTERVAL '2 hours')
            ON CONFLICT(report_id) DO UPDATE SET
            latitude=EXCLUDED.latitude,longitude=EXCLUDED.longitude,
            accuracy_m=EXCLUDED.accuracy_m,captured_at=EXCLUDED.captured_at,
            shared_at=NOW(),expires_at=NOW()+INTERVAL '2 hours'""",
            (report_id,record["school_id"],student_id,payload.latitude,payload.longitude,
             payload.accuracy_m,when))
        _audit(cur,record["school_id"],student_id,"demo_location_explicitly_shared",report_id)
    return {"ok":True,"demo_only":True,"staff_can_view":True,"expires_in_minutes":120,
            "real_notifications_sent":0,"dispatch_confirmed":False}

@router.delete("/reports/{report_id}/demo-location")
def withdraw_demo_location(report_id:str,authorization:Optional[str]=Header(default=None)):
    actor=_actor(authorization);_schema();student_id=str(actor["id"])
    with get_db() as conn:
        cur=conn.cursor()
        cur.execute("""SELECT school_id FROM verifyd_safety_training_reports
            WHERE report_id=%s AND student_id=%s""",(report_id,student_id))
        row=cur.fetchone()
        if not row:raise HTTPException(404,detail={"error":"practice_report_unavailable"})
        cur.execute("DELETE FROM verifyd_safety_demo_report_locations WHERE report_id=%s AND student_id=%s",
                    (report_id,student_id))
        _audit(cur,row["school_id"],student_id,"demo_location_withdrawn",report_id)
    return {"ok":True,"location_withdrawn":True}

@router.get("/incidents/{incident_id}/demo-locations")
def incident_demo_locations(incident_id:str,authorization:Optional[str]=Header(default=None)):
    actor=_actor(authorization);_schema()
    with get_db() as conn:
        cur=conn.cursor()
        cur.execute("SELECT school_id FROM verifyd_safety_training_incidents WHERE incident_id=%s",(incident_id,))
        incident=cur.fetchone()
        if not incident:raise HTTPException(404,detail={"error":"incident_not_found"})
        _enabled_for(incident["school_id"])
        _school_admin(cur,incident["school_id"],str(actor["id"]))
        _purge(cur)
        cur.execute("""SELECT l.report_id,l.latitude,l.longitude,l.accuracy_m,l.captured_at,
            l.shared_at,l.expires_at,i.handle AS student_handle,i.display_name AS student_name,
            r.student_status
            FROM verifyd_safety_demo_report_locations l
            JOIN verifyd_safety_training_reports r ON r.report_id=l.report_id
            JOIN verifyd_identities i ON i.id=l.student_id
            WHERE r.incident_id=%s AND r.closed_at IS NULL
            ORDER BY l.shared_at DESC LIMIT 100""",(incident_id,))
        locations=[dict(row) for row in cur.fetchall()]
        _audit(cur,incident["school_id"],str(actor["id"]),"demo_incident_locations_viewed",incident_id)
    return {"ok":True,"demo_only":True,"locations":locations,"real_notifications_sent":0}
