"""Safe school-approved family routing for TEST schools.

Only an independently approved guardian and active current guardian signoff may
be used. Staff manual routing always takes precedence. Never send a report here.
"""
from __future__ import annotations
import os
from fastapi import HTTPException
from trust_voice_school_safety_training import _verified_training_contact, _audit
from trust_voice_school_safety_consents import POLICY_VERSION


def _enabled(school_id: str) -> bool:
    on=os.getenv("VERIFYD_SAFETY_AUTO_ROUTING_ENABLED","0").strip().lower() in {"1","true","yes","on"}
    allow={x.strip() for x in os.getenv("VERIFYD_SAFETY_AUTO_ROUTING_SCHOOL_IDS","").split(",") if x.strip()}
    return bool(on and school_id in allow)


def sync_approved_guardian_route(cur, school_id: str, student_id: str,
                                  actor_id: str | None = None) -> dict:
    """Idempotent, transaction-scoped synchronization; no guardian self-approval.

    Returns status only. Must be called AFTER the school has added approved
    guardian links and the adult's current versioned acknowledgment.
    """
    if not _enabled(school_id):
        return {"enabled":False,"updated":False}
    cur.execute("""SELECT s.local_ref,s.enrollment_status
        FROM verifyd_safety_managed_students s
        JOIN verifyd_safety_training_members m ON m.school_id=s.school_id
           AND m.identity_id=s.student_identity_id AND m.role='student'
        WHERE s.school_id=%s AND s.student_identity_id=%s
        FOR UPDATE OF s""",(school_id,student_id))
    student=cur.fetchone()
    if not student or student["enrollment_status"]!="school_approved":
        return {"enabled":True,"updated":False,"reason":"not_school_approved_student"}
    # Preserve the Principal's intentional TEST-route disable across refreshes.
    # Only a later explicit manual SAVE (not an approval or auto-sync) can
    # restore routing. Existing audit actions supply the durable tombstone.
    cur.execute("""SELECT EXISTS(
        SELECT 1 FROM verifyd_safety_training_audit d
        WHERE d.school_id=%s AND d.object_id=%s
          AND d.action='verified_training_contact_route_disabled'
          AND NOT EXISTS (
            SELECT 1 FROM verifyd_safety_training_audit m
             WHERE m.school_id=d.school_id AND m.object_id=d.object_id
               AND m.action='verified_training_contact_route_saved'
               AND m.created_at>d.created_at
          )
        ) AS principal_disabled""",(school_id,student_id))
    if cur.fetchone()["principal_disabled"]:
        return {"enabled":True,"updated":False,"reason":"principal_intentionally_disabled"}
    cur.execute("""SELECT primary_id,backup_id,route_source
       FROM verifyd_safety_training_contacts
       WHERE school_id=%s AND student_id=%s FOR UPDATE""",(school_id,student_id))
    saved=cur.fetchone()
    if saved and saved["route_source"] != "approved_auto":
        return {"enabled":True,"updated":False,"reason":"principal_manual_override"}

    cur.execute("""SELECT g.guardian_id
       FROM verifyd_safety_training_guardians g
       JOIN verifyd_safety_training_members m
         ON m.school_id=g.school_id AND m.identity_id=g.guardian_id AND m.role='guardian'
       JOIN verifyd_safety_training_signoffs a
         ON a.school_id=g.school_id AND a.actor_id=g.guardian_id
          AND a.subject_id=g.child_id AND a.kind='guardian'
          AND a.policy_version=%s AND a.revoked_at IS NULL
       JOIN verifyd_identities i ON i.id=g.guardian_id AND i.status='active'
       WHERE g.school_id=%s AND g.child_id=%s
       ORDER BY g.guardian_id""",(POLICY_VERSION,school_id,student_id))
    eligible=[]
    for row in cur.fetchall():
        gid=str(row["guardian_id"])
        try:_verified_training_contact(cur,school_id,student_id,gid)
        except HTTPException:continue
        if gid not in eligible:eligible.append(gid)
    if not eligible:
        if saved and saved["route_source"]=="approved_auto":
            cur.execute("""DELETE FROM verifyd_safety_training_contacts
                 WHERE school_id=%s AND student_id=%s AND route_source='approved_auto'""",
                (school_id,student_id))
            if actor_id:_audit(cur,school_id,actor_id,"approved_auto_family_route_invalidated",student_id)
            return {"enabled":True,"updated":True,"status":"removed_ineligible_auto_route"}
        return {"enabled":True,"updated":False,"reason":"no_verified_eligible_guardian"}

    # Only FINAL school-approved slot assignments can specify Primary or Backup.
    cur.execute("""SELECT a.slot,a.guardian_identity_id AS guardian_id FROM
        verifyd_safety_electronic_guardian_ack a
        WHERE a.school_id=%s AND a.local_ref=%s AND a.status='approved'
          AND a.approved_at IS NOT NULL AND a.guardian_identity_id IS NOT NULL
        UNION ALL
        SELECT i.slot,i.claimed_by AS guardian_id
        FROM verifyd_safety_family_invitations i
        WHERE i.school_id=%s AND i.local_ref=%s
          AND i.approval_status='approved' AND i.reviewed_at IS NOT NULL
          AND i.revoked_at IS NULL AND i.claimed_by IS NOT NULL""",
        (school_id,student["local_ref"],school_id,student["local_ref"]))
    roles={1:set(),2:set()}
    for row in cur.fetchall():
        gid=str(row["guardian_id"])
        if gid in eligible and row["slot"] in (1,2):
            roles[row["slot"]].add(gid)
    explicit_primary=next(iter(roles[1])) if len(roles[1])==1 else None
    explicit_backup=next(iter(roles[2])) if len(roles[2])==1 else None
    if saved and saved["route_source"]=="approved_auto" and str(saved["primary_id"]) in eligible:
        primary_id=str(saved["primary_id"])
    elif explicit_primary:
        primary_id=explicit_primary
    elif len(eligible)==1:
        primary_id=eligible[0]
    else:
        # Multiple approved adults but no proven Primary: staff must choose.
        return {"enabled":True,"updated":False,"reason":"ambiguous_primary_requires_school_review"}

    backup_id=(explicit_backup if explicit_backup and explicit_backup!=primary_id else None)
    if saved and saved["route_source"]=="approved_auto":
        if str(saved["primary_id"])==primary_id and (
            str(saved["backup_id"]) if saved["backup_id"] else None)==backup_id:
            return {"enabled":True,"updated":False,"status":"unchanged"}
        cur.execute("""UPDATE verifyd_safety_training_contacts
           SET primary_id=%s,backup_id=%s,updated_at=NOW()
           WHERE school_id=%s AND student_id=%s AND route_source='approved_auto'""",
           (primary_id,backup_id,school_id,student_id))
    else:
        cur.execute("""INSERT INTO verifyd_safety_training_contacts
         (school_id,student_id,primary_id,backup_id,route_source)
         VALUES(%s,%s,%s,%s,'approved_auto') ON CONFLICT DO NOTHING""",
         (school_id,student_id,primary_id,backup_id))
    if actor_id:_audit(cur,school_id,actor_id,"approved_guardian_route_auto_synced",student_id)
    return {"enabled":True,"updated":True,"status":"approved_guardians_auto_routed",
            "primary_assigned":True,"backup_assigned":bool(backup_id),"no_notifications_sent":True}
