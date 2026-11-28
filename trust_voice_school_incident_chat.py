"""School Safety incident inbox, durable replies and opted-in test notifications.

School-specific test-alert gate. Never sends 911/E911 traffic. Push acceptance
is not device delivery, and does not authorize a life-safety promise.
"""
from __future__ import annotations
import os
import uuid
import json
from typing import Optional
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field
from database import get_db
from trust_voice_school_safety_training import _actor,_schema as _school_schema,_audit,_membership,_school_response_authorized
from trust_voice_school_safety_registration import _school_admin
from trust_voice_school_safety_consents import POLICY_VERSION
from trust_voice_push import notify_identity_push, _enabled as _web_push_enabled, _configured as _web_push_configured, _active_subscriptions

router=APIRouter(prefix="/trust-voice/school-safety",tags=["School Incident Conversations"])

def _allowed(school_id):
    enabled=os.getenv("VERIFYD_SAFETY_TEST_NOTIFICATIONS_ENABLED","0").lower() in ("1","yes","true","on")
    ids={x.strip() for x in os.getenv("VERIFYD_SAFETY_TEST_NOTIFICATION_SCHOOL_IDS","").split(",") if x.strip()}
    return enabled and school_id in ids

def _live_allowed_in_cursor(cur,school_id):
    enabled=os.getenv("VERIFYD_SAFETY_LIVE_ALERTS_ENABLED","0").lower() in ("1","yes","true","on")
    worker=os.getenv("VERIFYD_SAFETY_INCIDENT_DELIVERY_WORKER_ENABLED","0").lower() in ("1","yes","true","on")
    frontend_ready=os.getenv("VERIFYD_SAFETY_LIVE_ALERT_FRONTEND_READY","0").lower() in ("1","yes","true","on")
    ids={x.strip() for x in os.getenv("VERIFYD_SAFETY_LIVE_ALERT_SCHOOL_IDS","").split(",") if x.strip()}
    if not (enabled and worker and frontend_ready) or school_id not in ids:
        return False
    cur.execute("SELECT active FROM verifyd_safety_live_school_activation WHERE school_id=%s",(school_id,))
    row=cur.fetchone()
    return bool(row and row["active"])

def _live_allowed(school_id):
    enabled=os.getenv("VERIFYD_SAFETY_LIVE_ALERTS_ENABLED","0").lower() in ("1","yes","true","on")
    worker=os.getenv("VERIFYD_SAFETY_INCIDENT_DELIVERY_WORKER_ENABLED","0").lower() in ("1","yes","true","on")
    frontend_ready=os.getenv("VERIFYD_SAFETY_LIVE_ALERT_FRONTEND_READY","0").lower() in ("1","yes","true","on")
    ids={x.strip() for x in os.getenv("VERIFYD_SAFETY_LIVE_ALERT_SCHOOL_IDS","").split(",") if x.strip()}
    if not (enabled and worker and frontend_ready) or school_id not in ids:return False
    with get_db() as conn:return _live_allowed_in_cursor(conn.cursor(),school_id)

def _notifications_allowed(school_id,report_mode="test"):
    # A report is forever bound to the mode active when it was accepted. Turning
    # on a future live rollout cannot reclassify or notify old practice reports.
    if report_mode=="live":return _live_allowed(school_id)
    return _allowed(school_id)

def _record_event(cur,report_id,school_id,event_type,actor_id=None,details=None):
    cur.execute("""INSERT INTO verifyd_safety_incident_lifecycle_events
        (event_id,report_id,school_id,event_type,actor_id,details)
        VALUES(%s,%s,%s,%s,%s,%s::jsonb)""",
        ("ssle_"+uuid.uuid4().hex,report_id,school_id,event_type,actor_id,
         json.dumps(details or {})))

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
        cur.execute("""CREATE TABLE IF NOT EXISTS verifyd_safety_incident_lifecycle_events(
            event_id TEXT PRIMARY KEY,
            report_id TEXT NOT NULL REFERENCES verifyd_safety_training_reports(report_id),
            school_id TEXT NOT NULL,
            event_type TEXT NOT NULL,
            actor_id TEXT REFERENCES verifyd_identities(id),
            details JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )""")
        cur.execute("""CREATE INDEX IF NOT EXISTS safety_incident_lifecycle_lookup
            ON verifyd_safety_incident_lifecycle_events(report_id,created_at,event_type)""")
        cur.execute("""ALTER TABLE verifyd_safety_incident_outbox
            ADD COLUMN IF NOT EXISTS escalation_level INTEGER NOT NULL DEFAULT 0""")
        cur.execute("""CREATE INDEX IF NOT EXISTS safety_alert_inbox_idx
            ON verifyd_safety_incident_outbox(recipient_id,created_at DESC)""")

def _authorize(cur,report_id,actor_id):
    cur.execute("""SELECT r.report_id,r.school_id,r.student_id,r.closed_at,r.report_mode,
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
        # Designated responders may access INITIAL reports only, not the
        # private student–guardian conversation or school administration.
        _school_response_authorized(cur,report["school_id"],actor_id)
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

def _family_chat_authorize(cur, report_id: str, actor_id: str):
    """Limit incident text access to the reporting student and signed guardian(s)."""
    report = _authorize(cur, report_id, actor_id)
    if str(report["student_id"]) == actor_id:
        _membership(cur, report["school_id"], actor_id, {"student"})
        return report
    if _membership(cur, report["school_id"], actor_id) != "guardian":
        raise HTTPException(403, detail={"error": "private_family_conversation_only"})
    # _authorize already verifies report recipient, active guardian-school link,
    # and current guardian acknowledgment. Staff have no access.
    return report


def queue_report(cur,report_id,school_id,student_id,recipients):
    """Called inside report creation transaction; never bypasses a mode gate."""
    cur.execute("SELECT report_mode FROM verifyd_safety_training_reports WHERE report_id=%s",(report_id,))
    row=cur.fetchone() or {}
    report_mode=str(row.get("report_mode") or "test")
    gate=_allowed(school_id) if report_mode=="test" else _live_allowed_in_cursor(cur,school_id)
    if not gate:return 0
    queued=0
    for ident in recipients:
        if str(ident)==str(student_id):continue
        # Only active, individually authorized school recipients may be queued.
        try:_authorize(cur,report_id,str(ident))
        except HTTPException:continue
        cur.execute("""INSERT INTO verifyd_safety_incident_outbox(
            report_id,recipient_id,event_key) VALUES(%s,%s,'created')
            ON CONFLICT DO NOTHING""",(report_id,str(ident)))
        queued+=int(cur.rowcount>0)
    _record_event(cur,report_id,school_id,"authorized_recipients_queued",
        details={"recipient_count":queued})
    return queued

def _dispatch(report_id, event_key):
    with get_db() as conn:
        cur=conn.cursor()
        cur.execute("""SELECT o.recipient_id,o.attempts,r.school_id,r.report_mode FROM verifyd_safety_incident_outbox o
           JOIN verifyd_safety_training_reports r ON r.report_id=o.report_id
           WHERE o.report_id=%s AND o.event_key=%s AND o.state='pending'
              AND o.attempts<4
              AND (o.attempted_at IS NULL OR o.attempted_at < NOW()-INTERVAL '75 seconds')
           ORDER BY o.created_at LIMIT 100""",(report_id,event_key))
        jobs=[dict(x) for x in cur.fetchall()]
    attempted=0;accepted=0
    for job in jobs:
        if not _notifications_allowed(str(job["school_id"]),str(job["report_mode"] or "test")):continue
        with get_db() as conn:
            try:
                if str(event_key).startswith("reply:"):
                    _family_chat_authorize(conn.cursor(),report_id,str(job["recipient_id"]))
                elif str(event_key).startswith("schoolreply:"):
                    from trust_voice_school_staff_responses import _school_chat_authorize
                    _school_chat_authorize(conn.cursor(),report_id,str(job["recipient_id"]))
                else:
                    _authorize(conn.cursor(),report_id,str(job["recipient_id"]))
            except HTTPException:continue
        recipient=str(job["recipient_id"])
        # Keep unsent notification queued until a device has opted into push.
        if not _active_subscriptions(recipient):continue
        with get_db() as conn:
            cur=conn.cursor()
            cur.execute("""UPDATE verifyd_safety_incident_outbox
                 SET attempts=attempts+1,attempted_at=NOW()
                 WHERE report_id=%s AND recipient_id=%s AND event_key=%s
                    AND state='pending' AND attempts<4
                    AND (attempted_at IS NULL OR attempted_at<NOW()-INTERVAL '75 seconds')
                 RETURNING recipient_id""",(report_id,recipient,event_key))
            claimed=bool(cur.fetchone())
        if not claimed:continue
        try:
            private_reply = str(event_key).startswith("reply:")
            school_reply = str(event_key).startswith("schoolreply:")
            escalation = str(event_key).startswith("escalation:")
            live = str(job["report_mode"] or "test")=="live"
            result=notify_identity_push(recipient,
                notification_type=("school_safety_live_family_reply" if private_reply and live
                                   else "school_safety_family_reply_test" if private_reply
                                   else "school_safety_live_escalation" if escalation and live
                                   else "school_safety_test_escalation" if escalation
                                   else "school_safety_live_alert" if live
                                   else "school_safety_test_alert"),
                title=("VeriFYD Safety — private family text" if private_reply
                       else "VeriFYD Safety — school response" if school_reply and live
                       else "VeriFYD Safety — school TEST response" if school_reply
                       else "VeriFYD Safety — escalation" if escalation and live
                       else "VeriFYD Safety — TEST escalation" if escalation
                       else "VeriFYD Safety — school alert" if live
                       else "VeriFYD Safety — Test School notification"),
                body=("A private family text is ready in VeriFYD Safety." if private_reply and live
                      else "A private TEST family text is ready in VeriFYD Safety." if private_reply
                      else "A school alert is still awaiting staff acknowledgment. Open VeriFYD Safety now." if escalation and live
                      else "A TEST alert is still awaiting staff acknowledgment. Open VeriFYD Safety." if escalation
                      else "A school response is ready. Open VeriFYD Safety." if school_reply and live
                      else "A school TEST response is ready. Open your verified Safety app." if school_reply
                      else "A school safety report needs your attention. Open VeriFYD Safety." if live
                      else "An initial test school incident needs your attention."),
                url="/Safety",
                tag="verifyd-school-"+report_id+("-family" if private_reply else "-school" if school_reply else ""),
                ttl=600)
        except Exception:
            result={"sent":0}
        n=int(result.get("sent") or 0)
        with get_db() as conn:
            cur=conn.cursor()
            cur.execute("""UPDATE verifyd_safety_incident_outbox
               SET accepted=accepted+%s,
                   state=CASE WHEN %s>0 THEN 'attempted' ELSE 'pending' END
               WHERE report_id=%s AND recipient_id=%s AND event_key=%s""",
               (n,n,report_id,recipient,event_key))
            _record_event(cur,report_id,str(job["school_id"]),
                "provider_accepted" if n else "provider_rejected",
                details={"recipient_id":recipient,"event_key":event_key,
                         "provider_accepted_count":n})
        attempted+=1;accepted+=n
    return {"attempted_recipients":attempted,"push_provider_accepted":accepted,
            "device_delivery_confirmed":False}

def dispatch_school_pending_once():
    """Bounded scheduler for separately gated TEST or school delivery."""
    enabled=os.getenv("VERIFYD_SAFETY_TEST_NOTIFICATIONS_ENABLED","0").lower() in ("1","yes","true","on")
    live_enabled=os.getenv("VERIFYD_SAFETY_LIVE_ALERTS_ENABLED","0").lower() in ("1","yes","true","on")
    if not (enabled or live_enabled):return {"jobs":0,"provider_accepted":0}
    _schema()
    with get_db() as conn:
        cur=conn.cursor()
        cur.execute("""SELECT DISTINCT r.school_id,r.report_mode,o.report_id,o.event_key
           FROM verifyd_safety_incident_outbox o
           JOIN verifyd_safety_training_reports r ON r.report_id=o.report_id
           WHERE o.state='pending' AND o.attempts<4
             AND (o.attempted_at IS NULL OR o.attempted_at<NOW()-INTERVAL '75 seconds')
           ORDER BY r.school_id,o.report_id,o.event_key LIMIT 30""")
        jobs=[dict(x) for x in cur.fetchall()]
    result={"jobs":0,"provider_accepted":0}
    for job in jobs:
        if not _notifications_allowed(str(job["school_id"]),str(job["report_mode"] or "test")):continue
        delivered=_dispatch(job["report_id"],job["event_key"])
        result["jobs"]+=1
        result["provider_accepted"]+=delivered["push_provider_accepted"]
    return result

def dispatch_escalations_once(limit=30):
    """Queue one configured backup-responder escalation for unattended reports."""
    _schema()
    with get_db() as conn:
        cur=conn.cursor()
        cur.execute("""SELECT r.report_id,r.school_id,r.report_mode,p.delay_seconds,p.backup_responder_ids
            FROM verifyd_safety_training_reports r
            JOIN verifyd_safety_training_incidents i ON i.incident_id=r.incident_id
            JOIN verifyd_safety_incident_escalation_policies p ON p.school_id=r.school_id
            WHERE p.enabled=TRUE AND cardinality(p.backup_responder_ids)>0
              AND r.closed_at IS NULL AND i.status='active'
              AND r.created_at<=NOW()-(p.delay_seconds*INTERVAL '1 second')
              AND EXISTS(SELECT 1 FROM verifyd_safety_incident_outbox o
                 WHERE o.report_id=r.report_id AND o.event_key='created')
              AND NOT EXISTS(SELECT 1 FROM verifyd_safety_incident_outbox o
                 WHERE o.report_id=r.report_id AND o.event_key='escalation:1')
              AND NOT EXISTS(SELECT 1 FROM verifyd_safety_incident_acknowledgments a
                 JOIN verifyd_safety_training_members m ON m.identity_id=a.recipient_id
                   AND m.school_id=r.school_id AND m.role IN ('principal','responder')
                 WHERE a.report_id=r.report_id)
            ORDER BY r.created_at LIMIT %s""",(max(1,min(100,int(limit))),))
        reports=[dict(x) for x in cur.fetchall()]
    queued=0
    for report in reports:
        if not _notifications_allowed(str(report["school_id"]),str(report["report_mode"] or "test")):continue
        with get_db() as conn:
            cur=conn.cursor()
            lock_key="school_safety_escalation:"+str(report["report_id"])
            cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))",(lock_key,))
            cur.execute("SELECT 1 FROM verifyd_safety_incident_outbox WHERE report_id=%s AND event_key='escalation:1' LIMIT 1",(report["report_id"],))
            if cur.fetchone():
                continue
            cur.execute("""SELECT 1 FROM verifyd_safety_incident_acknowledgments a
                JOIN verifyd_safety_training_members m ON m.identity_id=a.recipient_id
                  AND m.school_id=%s AND m.role IN ('principal','responder')
                WHERE a.report_id=%s LIMIT 1""",(report["school_id"],report["report_id"]))
            if cur.fetchone():
                continue
            ids=[str(x) for x in (report["backup_responder_ids"] or [])]
            cur.execute("""SELECT m.identity_id FROM verifyd_safety_training_members m
                JOIN verifyd_identities i ON i.id=m.identity_id AND i.status='active'
                WHERE m.school_id=%s AND m.role='responder' AND m.identity_id=ANY(%s)""",
                (report["school_id"],ids))
            recipients=[str(x["identity_id"]) for x in cur.fetchall()]
            for recipient in recipients:
                try:_school_response_authorized(cur,str(report["school_id"]),recipient)
                except HTTPException:continue
                cur.execute("""INSERT INTO verifyd_safety_incident_outbox
                    (report_id,recipient_id,event_key,escalation_level)
                    VALUES(%s,%s,'escalation:1',1) ON CONFLICT DO NOTHING""",
                    (report["report_id"],recipient))
                queued+=int(cur.rowcount>0)
            if recipients:
                _record_event(cur,str(report["report_id"]),str(report["school_id"]),
                    "escalation_queued",details={"level":1,"recipient_count":len(recipients),
                    "delay_seconds":int(report["delay_seconds"])})
            else:
                _record_event(cur,str(report["report_id"]),str(report["school_id"]),
                    "escalation_skipped_no_active_backup",details={"level":1})
    return {"reports_checked":len(reports),"escalation_jobs_queued":queued}

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
           "live_school_scopes":[
           {"school_id":m["school_id"],"role":m["role"],"live_alerts_enabled":_live_allowed(m["school_id"])}
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
        cur.execute("""SELECT r.report_id,r.school_id,r.report_mode,r.category,r.description,r.student_status,r.created_at,
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
                report = _authorize(cur,row["report_id"],aid)
                item = dict(row)
                if str(report["student_id"]) != aid:
                    role = _membership(cur,report["school_id"],aid)
                    if role in ("principal","responder"):
                        item.pop("student_status",None)
                rows.append(item)
            except HTTPException:continue
    return {"ok":True,"reports":rows,"test_only":all(r.get("report_mode","test")=="test" for r in rows),
            "automatic_emergency_services_dispatch":False}

@router.post("/alerts/{report_id}/acknowledge")
def acknowledge_alert(report_id:str,authorization:Optional[str]=Header(default=None)):
    actor=_actor(authorization);_schema();aid=str(actor["id"])
    with get_db() as conn:
        cur=conn.cursor();r=_authorize(cur,report_id,aid)
        cur.execute("""INSERT INTO verifyd_safety_incident_acknowledgments
            (report_id,recipient_id) VALUES(%s,%s) ON CONFLICT DO NOTHING""",(report_id,aid))
        if cur.rowcount:
            _audit(cur,r["school_id"],aid,"school_test_alert_acknowledged",report_id)
            _record_event(cur,report_id,str(r["school_id"]),"recipient_acknowledged",aid)
    return {"ok":True,"acknowledged":True,"mode":r.get("report_mode","test"),
            "device_notification_delivery_confirmed":False}

@router.post("/alerts/{report_id}/opened")
def mark_alert_opened(report_id:str,authorization:Optional[str]=Header(default=None)):
    actor=_actor(authorization);_schema();aid=str(actor["id"])
    with get_db() as conn:
        cur=conn.cursor();r=_authorize(cur,report_id,aid)
        cur.execute("""INSERT INTO verifyd_safety_incident_lifecycle_events
            (event_id,report_id,school_id,event_type,actor_id,details)
            SELECT %s,%s,%s,'recipient_app_opened',%s,'{}'::jsonb
            WHERE NOT EXISTS(SELECT 1 FROM verifyd_safety_incident_lifecycle_events
                WHERE report_id=%s AND event_type='recipient_app_opened' AND actor_id=%s)""",
            ("ssle_"+uuid.uuid4().hex,report_id,r["school_id"],aid,report_id,aid))
    return {"ok":True,"app_opened_recorded":True,
            "device_notification_delivery_confirmed":False}

@router.get("/reports/{report_id}/delivery-status")
def incident_delivery_status(report_id:str,authorization:Optional[str]=Header(default=None)):
    actor=_actor(authorization);_schema();aid=str(actor["id"])
    with get_db() as conn:
        cur=conn.cursor();r=_authorize(cur,report_id,aid)
        role=_membership(cur,r["school_id"],aid)
        cur.execute("""SELECT event_type,actor_id,details,created_at FROM
            verifyd_safety_incident_lifecycle_events WHERE report_id=%s
            ORDER BY created_at,event_id LIMIT 200""",(report_id,))
        events=[]
        for row in cur.fetchall():
            event=dict(row)
            if role not in ("principal","responder"):
                event.pop("actor_id",None)
                details=event.get("details") or {}
                details.pop("recipient_id",None)
                event["details"]=details
            events.append(event)
    return {"ok":True,"report_id":report_id,"events":events,
        "provider_acceptance_is_not_device_delivery":True,
        "app_opened_means_in_app_open_only":True,
        "emergency_services_contacted":False}

@router.get("/reports/{report_id}/conversation")
def conversation(report_id:str,authorization:Optional[str]=Header(default=None)):
    actor=_actor(authorization);_schema();aid=str(actor["id"])
    with get_db() as conn:
        cur=conn.cursor();_family_chat_authorize(cur,report_id,aid)
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
        cur=conn.cursor();r=_family_chat_authorize(cur,report_id,aid)
        if r["closed_at"] or r["incident_status"]!="active":
            raise HTTPException(409,detail={"error":"incident_closed"})
        mid="ssmsg_"+uuid.uuid4().hex
        cur.execute("""INSERT INTO verifyd_safety_incident_messages
           (message_id,report_id,sender_id,client_id,body) VALUES(%s,%s,%s,%s,%s)
           ON CONFLICT(report_id,sender_id,client_id) DO NOTHING
           RETURNING message_id""",(mid,report_id,aid,payload.client_id,payload.body.strip()))
        saved=cur.fetchone()
        if saved and _notifications_allowed(str(r["school_id"]),str(r.get("report_mode") or "test")):
            cur.execute("""SELECT recipient_id FROM verifyd_safety_training_recipients
              WHERE report_id=%s UNION SELECT student_id FROM verifyd_safety_training_reports
              WHERE report_id=%s""",(report_id,report_id))
            for who in cur.fetchall():
                rid=str(who["recipient_id"])
                if rid!=aid:
                    try:_family_chat_authorize(cur,report_id,rid)
                    except HTTPException:continue
                    cur.execute("""INSERT INTO verifyd_safety_incident_outbox
                       (report_id,recipient_id,event_key) VALUES(%s,%s,%s)
                       ON CONFLICT DO NOTHING""",(report_id,rid,"reply:"+mid))
        if saved:_audit(cur,r["school_id"],aid,"school_incident_reply_saved",report_id)
    delivery={"attempted_recipients":0,"push_provider_accepted":0}
    if saved and _notifications_allowed(str(r["school_id"]),str(r.get("report_mode") or "test")):
        # The message is durably saved even if push dispatch is temporarily unavailable.
        try:
            delivery=_dispatch(report_id,"reply:"+mid)
        except Exception:
            delivery={"attempted_recipients":0,"push_provider_accepted":0,
                      "device_delivery_confirmed":False}
    return {"ok":True,"saved":True,"duplicate":not bool(saved),
            "notification":delivery,"mode":r.get("report_mode","test"),"emergency_dispatch":False}

@router.post("/alerts/dispatch-pending")
def dispatch_pending(authorization:Optional[str]=Header(default=None)):
    actor=_actor(authorization);_schema();aid=str(actor["id"])
    # Only a verified school admin can reattempt sends from the current school.
    with get_db() as conn:
        cur=conn.cursor()
        cur.execute("""SELECT DISTINCT r.school_id,r.report_mode,r.report_id,o.event_key
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
        if _notifications_allowed(str(job["school_id"]),str(job["report_mode"] or "test")):
            result.append(_dispatch(job["report_id"],job["event_key"]))
    escalations=dispatch_escalations_once()
    return {"ok":True,"retry_jobs":len(result),"escalations":escalations,
            "mode":"mode_scoped","emergency_dispatch":False}
