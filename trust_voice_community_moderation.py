"""VeriFYD Circle Community Board: authenticated, audited moderator review.

Authorization is based on stable identity IDs from the server-side allowlist.
No shared admin password, client-supplied role, or public moderation actions.
"""
from __future__ import annotations

import os
import uuid
from typing import Literal, Optional

from fastapi import APIRouter, BackgroundTasks, Header, HTTPException, Query
from pydantic import BaseModel, Field

from database import get_db
from trust_voice_messages import _identity_from_bearer
from trust_voice_circle_hub import ensure_schema, now_iso, public_posts_allowed, _insert_community_alerts, notify_community_post

router = APIRouter(tags=['VeriFYD Circle Community Moderation'])
MODERATION_VERSION = '0.3.0'
_moderation_schema_ready = False


def _moderator_ids() -> set[str]:
    return {x.strip().lower() for x in os.environ.get('VERIFYD_CIRCLE_MODERATOR_IDENTITY_IDS', '').split(',') if x.strip()}


def _moderator(authorization: Optional[str]) -> dict:
    user = _identity_from_bearer(authorization)
    # Deny by default until the owner explicitly allowlists one or more IDs.
    if str(user.get('id') or '').strip().lower() not in _moderator_ids():
        raise HTTPException(status_code=403, detail={'error': 'moderator_access_denied'})
    return user


def _ensure_moderation_schema():
    global _moderation_schema_ready
    if _moderation_schema_ready:
        return
    ensure_schema()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute('''CREATE TABLE IF NOT EXISTS verifyd_voice_community_moderation_cases (
            post_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            last_action TEXT NOT NULL,
            last_moderator_id TEXT NOT NULL,
            last_note TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )''')
        cur.execute('''CREATE TABLE IF NOT EXISTS verifyd_voice_community_moderation_audit (
            action_id TEXT PRIMARY KEY,
            post_id TEXT NOT NULL,
            moderator_id TEXT NOT NULL,
            action TEXT NOT NULL,
            reason TEXT NOT NULL,
            prior_hidden_at TEXT,
            prior_deleted_at TEXT,
            created_at TEXT NOT NULL
        )''')
        cur.execute('''CREATE INDEX IF NOT EXISTS idx_tv_community_audit_post
            ON verifyd_voice_community_moderation_audit (post_id,created_at)''')
    _moderation_schema_ready = True


class ModeratorDecision(BaseModel):
    action: Literal['hide', 'restore', 'remove', 'resolve']
    reason: str = Field(min_length=5, max_length=500)


@router.get('/community/moderation/my-identity')
def my_moderation_identity(authorization: Optional[str] = Header(default=None)):
    # A signed-in member can retrieve ONLY their own stable ID for an admin to
    # place in the server-side moderator allowlist; this does not confer access.
    user = _identity_from_bearer(authorization)
    return {'identity_id': str(user['id']), 'handle': user.get('handle_lower') or user.get('handle') or ''}


@router.get('/community/moderation/me')
def moderator_me(authorization: Optional[str] = Header(default=None)):
    actor = _moderator(authorization)
    return {
        'is_moderator': True,
        'identity_id': str(actor['id']),
        'version': MODERATION_VERSION,
        'public_posting_enabled': public_posts_allowed(),
    }


@router.get('/community/moderation/queue')
def moderation_queue(
    status: Literal['pending', 'hidden', 'reviewed', 'all'] = Query(default='pending'),
    limit: int = Query(default=50, ge=1, le=100),
    authorization: Optional[str] = Header(default=None),
):
    _moderator(authorization)
    _ensure_moderation_schema()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute('''SELECT p.post_id,p.title,p.details,p.city,p.location_text,p.event_start,
             p.visibility,p.group_id,p.created_at,p.hidden_at,p.deleted_at,
             u.handle_lower AS author_handle,
             COALESCE(rep.n,0) AS report_count,
             rep.last_report_at,
             COALESCE(rep.reasons,'') AS report_reasons,
             mc.status AS case_status,mc.updated_at AS reviewed_at,
             mc.last_action,mc.last_note
          FROM verifyd_voice_community_posts p
          LEFT JOIN verifyd_identities u ON u.id=p.author_id
          LEFT JOIN (
            SELECT r.post_id,COUNT(*) AS n,MAX(r.created_at) AS last_report_at,
                   STRING_AGG(DISTINCT r.reason, ', ') AS reasons
            FROM verifyd_voice_community_reports r GROUP BY r.post_id
          ) rep ON rep.post_id=p.post_id
          LEFT JOIN verifyd_voice_community_moderation_cases mc ON mc.post_id=p.post_id
          WHERE p.deleted_at IS NULL
          ORDER BY CASE WHEN p.hidden_at IS NOT NULL THEN 0 ELSE 1 END,
                   COALESCE(rep.n,0) DESC,p.created_at DESC
          LIMIT 500''')
        rows = [dict(x) for x in (cur.fetchall() or [])]
    result = []
    for r in rows:
        # A new report after a review reopens the case.
        reviewed = bool(r.get('reviewed_at') and r.get('case_status') == 'reviewed')
        fresh_report = bool(r.get('last_report_at') and (
            not r.get('reviewed_at') or str(r['last_report_at']) > str(r['reviewed_at'])
        ))
        awaiting_public_approval = r.get('case_status') == 'pending_public'
        pending = awaiting_public_approval or bool(r['report_count'] > 0 and (fresh_report or not r.get('reviewed_at')))
        if status == 'pending' and not pending:
            continue
        if status == 'hidden' and not r.get('hidden_at'):
            continue
        if status == 'reviewed' and (pending or not reviewed):
            continue
        r['review_state'] = 'pending_public' if awaiting_public_approval else ('pending' if pending else ('reviewed' if reviewed else 'actioned'))
        r['report_count'] = int(r['report_count'])
        result.append(r)
        if len(result) >= limit:
            break
    return {'items': result, 'count': len(result), 'status': status}


@router.post('/community/moderation/posts/{post_id}/decision')
def moderate_post(post_id: str, payload: ModeratorDecision, tasks: BackgroundTasks,
                  authorization: Optional[str] = Header(default=None)):
    actor = _moderator(authorization)
    _ensure_moderation_schema()
    reason = payload.reason.strip()
    if len(reason) < 5:
        raise HTTPException(422, detail={'error': 'moderation_reason_required'})
    if not post_id.startswith('tvpost_') or len(post_id) > 90:
        raise HTTPException(404, detail={'error': 'event_not_found'})
    stamp = now_iso()
    notified=[]
    from trust_voice_permissions import ensure_trust_circle_schema
    ensure_trust_circle_schema()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute('''SELECT post_id,author_id,visibility,group_id,hidden_at,deleted_at
            FROM verifyd_voice_community_posts WHERE post_id=%s FOR UPDATE''', (post_id,))
        row = cur.fetchone()
        if not row or row.get('deleted_at'):
            raise HTTPException(404, detail={'error': 'event_not_found'})
        cur.execute('SELECT status,last_action FROM verifyd_voice_community_moderation_cases WHERE post_id=%s', (post_id,))
        prior_case=cur.fetchone()
        pending_public = bool(prior_case and prior_case.get('status')=='pending_public')
        if pending_public and payload.action=='resolve':
            raise HTTPException(400,detail={'error':'approval_required', 'message':'Approve (Restore) or remove the public submission.'})
        if payload.action == 'hide':
            cur.execute('UPDATE verifyd_voice_community_posts SET hidden_at=%s WHERE post_id=%s', (stamp, post_id))
        elif payload.action == 'restore':
            # Only an allowlisted moderator can approve. Posting does not
            # broadcast to everyone; notifications go to author's circle.
            cur.execute('UPDATE verifyd_voice_community_posts SET hidden_at=NULL WHERE post_id=%s', (post_id,))
            if pending_public and row['visibility']=='public':
                notified=_insert_community_alerts(cur,post_id,str(row['author_id']),'public')
        elif payload.action == 'remove':
            cur.execute('UPDATE verifyd_voice_community_posts SET deleted_at=%s,hidden_at=%s WHERE post_id=%s', (stamp, stamp, post_id))
        # 'resolve' records a reviewed decision without changing visibility.
        cur.execute('''INSERT INTO verifyd_voice_community_moderation_cases
            (post_id,status,last_action,last_moderator_id,last_note,updated_at)
            VALUES (%s,'reviewed',%s,%s,%s,%s)
            ON CONFLICT(post_id) DO UPDATE SET
            status=EXCLUDED.status,last_action=EXCLUDED.last_action,
            last_moderator_id=EXCLUDED.last_moderator_id,
            last_note=EXCLUDED.last_note,updated_at=EXCLUDED.updated_at''',
            (post_id,payload.action,str(actor['id']),reason,stamp))
        cur.execute('''INSERT INTO verifyd_voice_community_moderation_audit
            (action_id,post_id,moderator_id,action,reason,prior_hidden_at,prior_deleted_at,created_at)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s)''',
            ('tvmod_'+uuid.uuid4().hex,post_id,str(actor['id']),payload.action,
             reason,row.get('hidden_at'),row.get('deleted_at'),stamp))
    for rid in notified:
        tasks.add_task(notify_community_post,rid,post_id)
    return {'ok': True, 'action': payload.action, 'post_id': post_id,
            'reviewed_at': stamp,'trust_circle_alert_count':len(notified)}


@router.get('/community/moderation/audit')
def moderation_audit(limit: int = Query(default=50, ge=1, le=100),
                     authorization: Optional[str] = Header(default=None)):
    _moderator(authorization)
    _ensure_moderation_schema()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute('''SELECT a.action_id,a.post_id,a.action,a.reason,a.created_at,
                 u.handle_lower AS moderator_handle
            FROM verifyd_voice_community_moderation_audit a
            LEFT JOIN verifyd_identities u ON u.id=a.moderator_id
            ORDER BY a.created_at DESC LIMIT %s''', (limit,))
        rows = [dict(x) for x in (cur.fetchall() or [])]
    return {'actions': rows, 'count': len(rows)}
