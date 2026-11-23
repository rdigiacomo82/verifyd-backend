"""School Safety incident inbox, durable replies and opted-in test notifications.

School-specific test-alert gate. Never sends 911/E911 traffic. Push acceptance
is not device delivery, and does not authorize a life-safety promise.
"""
from __future__ import annotations
import os
import uuid
from typing import Optional
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field
from database import get_db
from trust_voice_school_safety_training import _actor,_schema as _school_schema,_audit,_membership
from trust_voice_school_safety_registration import _school_admin
from trust_voice_school_safety_consents import POLICY_VERSION
from trust_voice_push import notify_identity_push, _enabled as _web_push_enabled, _configured as _web_push_configured, _active_subscriptions

router=APIRouter(prefix="/trust-voice/school-safety",tags=["School Incident Conversations"])

def _allowed(school_id):
    enabled=os.getenv("VERIFYD_SAFETY_TEST_NOTIFICATIONS_ENABLED","0").lower() in ("1","yes","true","on")
    ids={x.strip() for x in os.getenv("VERIFYD_SAFETY_TEST_NOTIFICATION_SCHOOL_IDS","").split(",") if x.strip()}
    return enabled and school_id in ids

def _schema():
    _school_schema()
    with get_db() as conn:
        cur=conn.cursor()
        cur.execute("""CREATE TABLE IF NOT EXISTS verifyd_safety_incident_messages(
            message_id TEXT PRIMARY KEY, report_id TEXT NOT NULL REFERENCES verifyd_safety_training_reports(report_id),
            sender_id TEXT NOT NULL REFERENCES verifyd_identities(id),
            client_id TEXT NOT NULL, body TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            UNIQUE(report_id,sender_id,client_id))""")
        cur.execute("""CREATE TABLE IF NOT EXISTS verifyd_safety_incident_outbox(
            report_id TEXT NOT NULL REFERENCES verifyd_safety_training_reports(report_id),
            recipient_id TEXT NOT NULL REFERENCES verifyd_identities(id),
            event_key TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            attempted_at TIMESTAMPTZ,
            attempts INT NOT NULL DEFAULT 0,
            accepted INT NOT NULL DEFAULT 0,
            state TEXT NOT NULL DEFAULT 'pending' CHECK(state IN ('pending','attempted')),
            PRIMARY KEY(report_id,recipient_id,event_key))""")
        cur.execute("""CREATE TABLE IF NOT EXISTS verifyd_safety_incident_acknowledgments(
            report_id TEXT NOT NULL REFERENCES verifyd_safety_training_reports(report_id),
            recipient_id TEXT NOT NULL REFERENCES verifyd_identities(id),
            acknowledged_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY(report_id,recipient_id))""")
        cur.execute("""CREATE INDEX IF NOT EXISTS safety_alert_inbox_idx
            ON verifyd_safety_incident_outbox(recipient_id,created_at DESC)""")

def _authorize(cur,report_id,actor_id):
    cur.execute("""SELECT r.report_id,r.school_id,r.student_id,r.closed_at,
        i.status AS incident_status
        FROM verifyd_safety_training_reports r
        JOIN verifyd_safety_training_incidents i ON i.incident_id=r.incident_id
        WHERE r.report_id=%s""",(report_id,))
    report=cur.fetchone()
    if not report:raise HTTPException(404,detail={"error":"report_not_found"})
    if str(report["student_id"])==actor_id:
        _membership(cur,report["school_id"],actor_id,{"student"})
        return report
    role=_membership(cur,report["school_id"],actor_id)
    cur.execute("""SELECT 1 FROM verifyd_safety_training_recipients
        WHERE report_id=%s AND recipient_id=%s""",(report_id,actor_id))
    if not cur.fetchone():raise HTTPException(403,detail={"error":"not_incident_recipient"})
    if role in ("principal","responder"):
        _school_admin(cur,report["school_id"],actor_id)
    elif role=="guardian":
        cur.execute("""SELECT 1 FROM verifyd_safety_training_guardians g
          JOIN verifyd_safety_training_signoffs x ON x.school_id=g.school_id
            AND x.actor_id=g.guardian_id AND x.subject_id=g.child_id
            AND x.kind='guardian' AND x.policy_version=%s AND x.revoked_at IS NULL
          WHERE g.school_id=%s AND g.child_id=%s AND g.guardian_id=%s""",
          (POLICY_VERSION,report["school_id"],report["student_id"],actor_id))
        if not cur.fetchone():raise HTTPException(403,detail={"error":"guardian_relationship_not_active"})
    else:raise HTTPException(403,detail={"error":"not_authorized_for_incident"})
    return report

def queue_report(cur,report_id,school_id,student_id,recipients):
    """Called inside report creation transaction; never enables real dispatch."""
    if not _allowed(school_id):return
    for ident in recipients:
        if str(ident)==str(student_id):continue
        # Only active, individually authorized school recipients may be queued.
        try:_authorize(cur,report_id,str(ident))
        except HTTPException:continue
        cur.execute("""INSERT INTO verifyd_safety_incident_outbox(
            report_id,recipient_id,event_key) VALUES(%s,%s,'created')
            ON CONFLICT DO NOTHING""",(report_id,str(ident)))

def _dispatch(report_id, event_key):
    with get_db() as conn:
        cur=conn.cursor()
        cur.execute("""SELECT o.recipient_id,o.attempts,r.school_id FROM verifyd_safety_incident_outbox o
           JOIN verifyd_safety_training_reports r ON r.report_id=o.report_id
           WHERE o.report_id=%s AND o.event_key=%s AND o.state='pending'
              AND o.attempts<4 ORDER BY o.created_at LIMIT 100""",(report_id,event_key))
        jobs=[dict(x) for x in cur.fetchall()]
    attempted=0;accepted=0
    for job in jobs:
        if not _allowed(job["school_id"]):continue
        with get_db() as conn:
            try:_authorize(conn.cursor(),report_id,str(job["recipient_id"]))
            except HTTPException:continue
        result=notify_identity_push(str(job["recipient_id"]),
            notification_type="school_safety_test_alert",
            title="VeriFYD Safety — Test School notification",
            body="A test incident needs your attention. Open VeriFYD Safety.",
            url="/Safety",tag="verifyd-school-"+report_id,ttl=600)
        n=int(result.get("sent") or 0)
        with get_db() as conn:
            conn.cursor().execute("""UPDATE verifyd_safety_incident_outbox
               SET attempts=attempts+1,attempted_at=NOW(),accepted=accepted+%s,
                   state=CASE WHEN %s>0 THEN 'attempted' ELSE 'pending' END
               WHERE report_id=%s AND recipient_id=%s AND event_key=%s""",
               (n,n,report_id,str(job["recipient_id"]),event_key))
        attempted+=1;accepted+=n
    return {"attempted_recipients":attempted,"push_provider_accepted":accepted,
            "device_delivery_confirmed":False}

def dispatch_report(report_id):
    return _dispatch(report_id,"created")

class Reply(BaseModel):
    client_id: str=Field(min_length=12,max_length=100)
    body: str=Field(min_length=1,max_length=1500)

@router.get("/alerts/readiness")
def alert_readiness(authorization:Optional[str]=Header(default=None)):
    actor=_actor(authorization);_schema();aid=str(actor["id"])
    with get_db() as conn:
        cur=conn.cursor()
        cur.execute("""SELECT school_id,role FROM verifyd_safety_training_members
            WHERE identity_id=%s ORDER BY school_id""",(aid,))
        memberships=[dict(x) for x in cur.fetchall()]
    return {"ok":True,"test_school_scopes":[
           {"school_id":m["school_id"],"role":m["role"],"test_push_gate_enabled":_allowed(m["school_id"])}
           for m in memberships],
           "web_push_backend_configured":bool(_web_push_enabled() and _web_push_configured()),
           "this_account_web_push_subscription_count":len(_active_subscriptions(aid)),
           "device_delivery_verified":False,
           "automatic_emergency_services_dispatch":False}

@router.get("/alerts/inbox")
def inbox(authorization:Optional[str]=Header(default=None)):
    actor=_actor(authorization);_schema();aid=str(actor["id"])
    with get_db() as conn:
        cur=conn.cursor()
        cur.execute("""SELECT r.report_id,r.school_id,r.category,r.student_status,r.created_at,
          i.status AS incident_status,
          EXISTS(SELECT 1 FROM verifyd_safety_incident_acknowledgments a
             WHERE a.report_id=r.report_id AND a.recipient_id=%s) AS acknowledged,
          EXISTS(SELECT 1 FROM verifyd_safety_incident_outbox o
             WHERE o.report_id=r.report_id AND o.recipient_id=%s
                 AND o.event_key='created' AND o.accepted>0) AS push_provider_accepted
          FROM verifyd_safety_training_reports r
          JOIN verifyd_safety_training_incidents i ON i.incident_id=r.incident_id
          WHERE r.student_id=%s OR EXISTS(
             SELECT 1 FROM verifyd_safety_training_recipients x
             WHERE x.report_id=r.report_id AND x.recipient_id=%s)
          ORDER BY r.created_at DESC LIMIT 80""",(aid,aid,aid,aid))
        rows=[]
        for row in cur.fetchall():
            try:
                _authorize(cur,row["report_id"],aid)
                rows.append(dict(row))
            except HTTPException:continue
    return {"ok":True,"reports":rows,"test_only":True,
            "automatic_911_delivery":False}

@router.post("/alerts/{report_id}/acknowledge")
def acknowledge_alert(report_id:str,authorization:Optional[str]=Header(default=None)):
    actor=_actor(authorization);_schema();aid=str(actor["id"])
    with get_db() as conn:
        cur=conn.cursor();r=_authorize(cur,report_id,aid)
        cur.execute("""INSERT INTO verifyd_safety_incident_acknowledgments
            (report_id,recipient_id) VALUES(%s,%s) ON CONFLICT DO NOTHING""",(report_id,aid))
        _audit(cur,r["school_id"],aid,"school_test_alert_acknowledged",report_id)
    return {"ok":True,"acknowledged":True,"test_only":True,
            "device_notification_delivery_confirmed":False}

@router.get("/reports/{report_id}/conversation")
def conversation(report_id:str,authorization:Optional[str]=Header(default=None)):
    actor=_actor(authorization);_schema();aid=str(actor["id"])
    with get_db() as conn:
        cur=conn.cursor();_authorize(cur,report_id,aid)
        cur.execute("""SELECT m.message_id,m.sender_id,i.handle AS sender_handle,
          m.body,m.created_at FROM verifyd_safety_incident_messages m
          JOIN verifyd_identities i ON i.id=m.sender_id
          WHERE m.report_id=%s ORDER BY m.created_at,m.message_id LIMIT 300""",(report_id,))
        messages=[dict(x) for x in cur.fetchall()]
    return {"ok":True,"messages":messages,"test_only":True}

@router.post("/reports/{report_id}/conversation")
def send_reply(report_id:str,payload:Reply,authorization:Optional[str]=Header(default=None)):
    actor=_actor(authorization);_schema();aid=str(actor["id"])
    with get_db() as conn:
        cur=conn.cursor();r=_authorize(cur,report_id,aid)
        if r["closed_at"] or r["incident_status"]!="active":
            raise HTTPException(409,detail={"error":"incident_closed"})
        mid="ssmsg_"+uuid.uuid4().hex
        cur.execute("""INSERT INTO verifyd_safety_incident_messages
           (message_id,report_id,sender_id,client_id,body) VALUES(%s,%s,%s,%s,%s)
           ON CONFLICT(report_id,sender_id,client_id) DO NOTHING
           RETURNING message_id""",(mid,report_id,aid,payload.client_id,payload.body.strip()))
        saved=cur.fetchone()
        if saved and _allowed(r["school_id"]):
            cur.execute("""SELECT recipient_id FROM verifyd_safety_training_recipients
              WHERE report_id=%s UNION SELECT student_id FROM verifyd_safety_training_reports
              WHERE report_id=%s""",(report_id,report_id))
            for who in cur.fetchall():
                rid=str(who["recipient_id"])
                if rid!=aid:
                    try:_authorize(cur,report_id,rid)
                    except HTTPException:continue
                    cur.execute("""INSERT INTO verifyd_safety_incident_outbox
                       (report_id,recipient_id,event_key) VALUES(%s,%s,%s)
                       ON CONFLICT DO NOTHING""",(report_id,rid,"reply:"+mid))
        if saved:_audit(cur,r["school_id"],aid,"school_incident_reply_saved",report_id)
    delivery={"attempted_recipients":0,"push_provider_accepted":0}
    if saved and _allowed(r["school_id"]):
        # The message is durably saved even if push dispatch is temporarily unavailable.
        try:
            delivery=_dispatch(report_id,"reply:"+mid)
        except Exception:
            delivery={"attempted_recipients":0,"push_provider_accepted":0,
                      "device_delivery_confirmed":False}
    return {"ok":True,"saved":True,"duplicate":not bool(saved),
            "notification":delivery,"test_only":True,"emergency_dispatch":False}

@router.post("/alerts/dispatch-pending")
def dispatch_pending(authorization:Optional[str]=Header(default=None)):
    actor=_actor(authorization);_schema();aid=str(actor["id"])
    # Only a verified school admin can reattempt sends from the current school.
    with get_db() as conn:
        cur=conn.cursor()
        cur.execute("""SELECT DISTINCT r.school_id,r.report_id,o.event_key
          FROM verifyd_safety_incident_outbox o
          JOIN verifyd_safety_training_reports r ON r.report_id=o.report_id
          WHERE o.state='pending' AND o.attempts<4 ORDER BY r.school_id,r.report_id,o.event_key LIMIT 60""")
        jobs=[dict(x) for x in cur.fetchall()]
        eligible=[]
        for job in jobs:
            try:_school_admin(cur,job["school_id"],aid);eligible.append(job)
            except HTTPException:continue
    result=[]
    for job in eligible:
        if _allowed(job["school_id"]):
            result.append(_dispatch(job["report_id"],job["event_key"]))
    return {"ok":True,"retry_jobs":len(result),"test_only":True}
