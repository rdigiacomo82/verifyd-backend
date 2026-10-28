"""VeriFYD Circle: authenticated meeting invitations and community event board.

Additive module. Disabled by default until backend env flags are explicitly enabled.
Public board is NOT a substitute for human moderation; operate initially in a
closed test cohort. Sensitive/private health information must not be posted.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import uuid
from datetime import datetime, timedelta, timezone
from typing import Literal, Optional

from fastapi import APIRouter, BackgroundTasks, Header, HTTPException, Query
from pydantic import BaseModel, Field

from database import get_db
from trust_voice_groups import ensure_group_schema, _require_group_member
from trust_voice_group_video_schedule import ensure_schedule_schema, _public_schedule, _schedule_row
from trust_voice_messages import _identity_from_bearer
from trust_voice_permissions import trust_circle_contains
from trust_voice_signaling import _identity_by_handle, _normalize_handle, _public_identity, _send_to_identity

log = logging.getLogger('verifyd.trust_voice.circle_hub')
router = APIRouter(tags=['VeriFYD Circle Meetings and Community'])
FEATURE_VERSION = '0.2.0'
_schema_ready = False


def enabled(env_key: str) -> bool:
    return os.environ.get(env_key, '').lower().strip() in ('1', 'yes', 'true', 'on')


def public_posts_allowed() -> bool:
    # Do not expose all-user posting without an explicit launch flag AND a
    # server-side moderator allowlist. Configure/test the review queue first.
    moderator_ids = [x.strip() for x in os.environ.get('VERIFYD_CIRCLE_MODERATOR_IDENTITY_IDS','').split(',') if x.strip()]
    return enabled('VERIFYD_TRUST_VOICE_COMMUNITY_PUBLIC_POSTS_ENABLED') and bool(moderator_ids)


def require_feature(env_key: str) -> None:
    if not enabled(env_key):
        raise HTTPException(status_code=503, detail={'error': 'feature_not_enabled', 'message': 'This feature is not yet enabled on the VeriFYD server.'})


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_start(raw: str) -> str:
    try:
        d = datetime.fromisoformat(raw.strip().replace('Z', '+00:00'))
        if d.tzinfo is None:
            raise ValueError('Timezone missing')
        d = d.astimezone(timezone.utc)
    except Exception:
        raise HTTPException(400, detail={'error': 'invalid_start', 'message': 'Choose a valid date and time.'})
    now = datetime.now(timezone.utc)
    if d <= now - timedelta(minutes=2) or d > now + timedelta(days=365):
        raise HTTPException(400, detail={'error': 'start_out_of_range', 'message': 'Choose a future date within one year.'})
    return d.isoformat()


def clean(value: str, max_length: int) -> str:
    s = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', value or '').strip()
    if not s or len(s) > max_length:
        raise HTTPException(400, detail={'error': 'invalid_text', 'message': f'A required text field must be 1 to {max_length} characters.'})
    return s


def ensure_schema():
    global _schema_ready
    if _schema_ready:
        return
    ensure_group_schema()
    ensure_schedule_schema()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute('''CREATE TABLE IF NOT EXISTS verifyd_voice_circle_meetings (
            meeting_id TEXT PRIMARY KEY, schedule_id TEXT NOT NULL UNIQUE,
            host_id TEXT NOT NULL, duration_minutes INTEGER NOT NULL DEFAULT 30,
            created_at TEXT NOT NULL
        )''')
        cur.execute('''CREATE TABLE IF NOT EXISTS verifyd_voice_circle_invitations (
            meeting_id TEXT NOT NULL, invitee_id TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending', responded_at TEXT,
            created_at TEXT NOT NULL, PRIMARY KEY (meeting_id,invitee_id)
        )''')
        cur.execute('''CREATE INDEX IF NOT EXISTS idx_tv_circle_invitee
            ON verifyd_voice_circle_invitations (invitee_id,status)''')
        cur.execute('''CREATE TABLE IF NOT EXISTS verifyd_voice_community_posts (
            post_id TEXT PRIMARY KEY, author_id TEXT NOT NULL,
            title TEXT NOT NULL, details TEXT NOT NULL,
            event_start TEXT NOT NULL, location_text TEXT NOT NULL,
            city TEXT NOT NULL, visibility TEXT NOT NULL,
            group_id TEXT, created_at TEXT NOT NULL,
            deleted_at TEXT, hidden_at TEXT
        )''')
        cur.execute('''CREATE INDEX IF NOT EXISTS idx_tv_community_time
            ON verifyd_voice_community_posts (event_start,created_at)''')
        cur.execute('''CREATE TABLE IF NOT EXISTS verifyd_voice_community_rsvps (
            post_id TEXT NOT NULL, attendee_id TEXT NOT NULL,
            status TEXT NOT NULL, created_at TEXT NOT NULL,
            PRIMARY KEY(post_id,attendee_id)
        )''')
        cur.execute('''CREATE TABLE IF NOT EXISTS verifyd_voice_community_reports (
            post_id TEXT NOT NULL, reporter_id TEXT NOT NULL,
            reason TEXT NOT NULL, created_at TEXT NOT NULL,
            PRIMARY KEY(post_id,reporter_id)
        )''')
    _schema_ready = True


class MeetingCreate(BaseModel):
    title: str = Field(min_length=1, max_length=120)
    scheduled_for: str = Field(min_length=16, max_length=80)
    duration_minutes: int = Field(default=30, ge=15, le=240)
    attendee_handles: list[str] = Field(min_length=1, max_length=24)


class PostCreate(BaseModel):
    title: str = Field(min_length=1, max_length=140)
    details: str = Field(min_length=1, max_length=2500)
    event_start: str = Field(min_length=16, max_length=80)
    location: str = Field(min_length=1, max_length=160)
    city: str = Field(min_length=1, max_length=100)
    visibility: Literal['public', 'trust_circle', 'group']
    group_id: Optional[str] = Field(default=None, max_length=100)
    rules_accepted: bool


class RsvpPayload(BaseModel):
    status: Literal['going', 'interested', 'not_going']


class ReportPayload(BaseModel):
    reason: Literal['harassment', 'hate', 'sexual', 'privacy', 'misinformation', 'spam', 'other']


async def notify_identity(identity_id: str, payload: dict):
    try:
        await _send_to_identity(identity_id, payload)
    except Exception:
        log.exception('Meeting invite WS delivery failed')
    try:
        from trust_voice_push import dispatch_push_for_event
        await asyncio.to_thread(dispatch_push_for_event, identity_id, payload)
    except Exception:
        log.warning('Meeting invite push unavailable', exc_info=True)


@router.get('/calendar/meetings/health')
def meeting_health():
    return {'status': 'ok', 'version': FEATURE_VERSION, 'enabled': enabled('VERIFYD_TRUST_VOICE_CIRCLE_MEETINGS_ENABLED')}


@router.post('/calendar/meetings')
def create_circle_meeting(payload: MeetingCreate, tasks: BackgroundTasks, authorization: Optional[str] = Header(default=None)):
    require_feature('VERIFYD_TRUST_VOICE_CIRCLE_MEETINGS_ENABLED')
    host = _identity_from_bearer(authorization)
    host_id = str(host['id'])
    ensure_schema()
    scheduled_for = validate_start(payload.scheduled_for)
    title = clean(payload.title, 120)
    from trust_voice_blocking import identities_blocked
    recipients, seen = [], {host_id}
    for raw in payload.attendee_handles:
        handle = _normalize_handle(str(raw or ''))
        identity = _identity_by_handle(handle) if handle else None
        if not identity or not identity.get('id') or identity.get('status') not in (None, 'active'):
            raise HTTPException(404, detail={'error': 'invitee_not_found', 'message': f'VeriFYD handle @{handle} was not found.'})
        iid = str(identity['id'])
        if identities_blocked(host_id, iid):
            raise HTTPException(403, detail={'error': 'invitation_not_allowed', 'message': 'This user cannot be invited.'})
        if iid not in seen:
            seen.add(iid)
            recipients.append(identity)
    if not recipients:
        raise HTTPException(400, detail={'error': 'invitee_required', 'message': 'Select at least one other VeriFYD user.'})
    # Create a private meeting group with ONLY the organizer as a member.
    # Invitees obtain access to video/chat only after accepting explicitly.
    group_id = 'tvgroup_' + uuid.uuid4().hex
    meeting_id = 'tvcmtg_' + uuid.uuid4().hex
    schedule_id = 'tvgvsm_' + uuid.uuid4().hex
    now = now_iso()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute('''SELECT COUNT(*) AS n FROM verifyd_voice_circle_meetings
            WHERE host_id=%s AND created_at::timestamptz > NOW() - INTERVAL '24 hours' ''',(host_id,))
        if int(cur.fetchone()['n']) >= 10:
            raise HTTPException(429, detail={'error':'daily_meeting_limit', 'message':'You can create up to 10 meetings per day.'})
        cur.execute('''INSERT INTO verifyd_voice_conversations
            (conversation_id,conversation_type,direct_key,created_at,updated_at)
            VALUES (%s,'group',%s,%s,%s)''', (group_id,f'group::{group_id}',now,now))
        cur.execute('''INSERT INTO verifyd_voice_groups
            (conversation_id,name,owner_identity_id,created_at,updated_at)
            VALUES (%s,%s,%s,%s,%s)''', (group_id,title[:60],host_id,now,now))
        cur.execute('''INSERT INTO verifyd_voice_conversation_members
            (conversation_id,identity_id,joined_at,last_read_at)
            VALUES (%s,%s,%s,NULL)''', (group_id,host_id,now))
        cur.execute('''INSERT INTO verifyd_voice_group_video_schedules
            (schedule_id,conversation_id,title,scheduled_for,scheduled_by_identity_id,status,
             created_at,updated_at,cancelled_at,cancelled_by_identity_id,started_at,started_by_identity_id,session_id)
            VALUES (%s,%s,%s,%s,%s,'scheduled',%s,%s,NULL,NULL,NULL,NULL,NULL)''',
            (schedule_id,group_id,title,scheduled_for,host_id,now,now))
        cur.execute('''INSERT INTO verifyd_voice_circle_meetings
            (meeting_id,schedule_id,host_id,duration_minutes,created_at) VALUES (%s,%s,%s,%s,%s)''',
            (meeting_id,schedule_id,host_id,payload.duration_minutes,now))
        for member in recipients:
            cur.execute('''INSERT INTO verifyd_voice_circle_invitations
                (meeting_id,invitee_id,status,created_at)
                VALUES (%s,%s,'pending',%s)''', (meeting_id,str(member['id']),now))
    # Existing ?scheduled= route opens the authorized Video Center.
    link = f'https://vfvid.com/voice?scheduled={schedule_id}'
    for member in recipients:
        tasks.add_task(notify_identity, str(member['id']), {
            'type':'circle_meeting_invitation', 'meeting_id':meeting_id,
            'title':title, 'scheduled_for':scheduled_for,
            'host':_public_identity(host),
        })
    return {'ok':True,'meeting_id':meeting_id,'schedule_id':schedule_id,
            'group_id':group_id,'title':title,'scheduled_for':scheduled_for,
            'duration_minutes':payload.duration_minutes, 'meeting_url':link,'invited_count':len(recipients),
            'invitation_status':'pending'}


def _meeting_invite(row, viewer_id: str):
    # No invitation link until accepted; never grant group membership by URL.
    accepted = row['status'] == 'accepted'
    return {'meeting_id':row['meeting_id'],'schedule_id':row['schedule_id'],
            'title':row['title'],'scheduled_for':row['scheduled_for'],
            'duration_minutes':int(row['duration_minutes']), 'host_handle':row.get('host_handle') or '', 'status':row['status'],
            'meeting_url':f"https://vfvid.com/voice?scheduled={row['schedule_id']}" if accepted else None}


@router.get('/calendar/meetings/invitations')
def list_circle_invitations(authorization: Optional[str] = Header(default=None)):
    require_feature('VERIFYD_TRUST_VOICE_CIRCLE_MEETINGS_ENABLED')
    viewer = _identity_from_bearer(authorization)
    ensure_schema()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute('''SELECT i.*,s.title,s.scheduled_for,m.schedule_id,m.duration_minutes,h.handle_lower AS host_handle
            FROM verifyd_voice_circle_invitations i
            JOIN verifyd_voice_circle_meetings m ON m.meeting_id=i.meeting_id
            JOIN verifyd_voice_group_video_schedules s ON s.schedule_id=m.schedule_id
            JOIN verifyd_identities h ON h.id=m.host_id
            WHERE i.invitee_id=%s AND s.status='scheduled'
            AND s.scheduled_for::timestamptz >= NOW() - INTERVAL '2 hours'
            ORDER BY s.scheduled_for::timestamptz ASC LIMIT 100''', (str(viewer['id']),))
        rows=[dict(x) for x in (cur.fetchall() or [])]
    return {'invitations':[_meeting_invite(x,str(viewer['id'])) for x in rows]}


@router.post('/calendar/meetings/{meeting_id}/respond')
def respond_circle_meeting(meeting_id: str, payload: RsvpPayload, authorization: Optional[str] = Header(default=None)):
    # RSVP payload shares going/interested/not_going; only going counts as accepted.
    require_feature('VERIFYD_TRUST_VOICE_CIRCLE_MEETINGS_ENABLED')
    viewer = _identity_from_bearer(authorization)
    ensure_schema()
    viewer_id = str(viewer['id'])
    with get_db() as conn:
        cur=conn.cursor()
        cur.execute('''SELECT i.status,m.schedule_id,s.conversation_id,s.status AS meeting_status
            FROM verifyd_voice_circle_invitations i
            JOIN verifyd_voice_circle_meetings m ON m.meeting_id=i.meeting_id
            JOIN verifyd_voice_group_video_schedules s ON s.schedule_id=m.schedule_id
            WHERE i.meeting_id=%s AND i.invitee_id=%s FOR UPDATE OF i''', (meeting_id,viewer_id))
        row=cur.fetchone()
        if not row:
            raise HTTPException(404,detail={'error':'invitation_not_found'})
        if row['meeting_status'] != 'scheduled':
            raise HTTPException(409,detail={'error':'meeting_no_longer_available'})
        action='accepted' if payload.status=='going' else 'declined'
        if action=='accepted':
            cur.execute('''SELECT COUNT(*) AS n FROM verifyd_voice_conversation_members
                WHERE conversation_id=%s''', (row['conversation_id'],))
            if int(cur.fetchone()['n']) >= 25 and row['status'] != 'accepted':
                raise HTTPException(409,detail={'error':'meeting_full'})
            cur.execute('''INSERT INTO verifyd_voice_conversation_members
                (conversation_id,identity_id,joined_at,last_read_at)
                VALUES (%s,%s,%s,NULL) ON CONFLICT(conversation_id,identity_id) DO NOTHING''',
                (row['conversation_id'],viewer_id,now_iso()))
        else:
            cur.execute('''DELETE FROM verifyd_voice_conversation_members
                WHERE conversation_id=%s AND identity_id=%s''', (row['conversation_id'],viewer_id))
        cur.execute('''UPDATE verifyd_voice_circle_invitations SET status=%s,responded_at=%s
            WHERE meeting_id=%s AND invitee_id=%s''', (action,now_iso(),meeting_id,viewer_id))
    return {'ok':True,'status':action,
            'meeting_url':f"https://vfvid.com/voice?scheduled={row['schedule_id']}" if action=='accepted' else None}


@router.get('/community/health')
def community_health():
    return {'status':'ok','version':FEATURE_VERSION,'enabled':enabled('VERIFYD_TRUST_VOICE_COMMUNITY_ENABLED'),
            'public_posts_enabled':public_posts_allowed(),
            'moderation':'admin_review_queue_audit_v2'}


def may_view(row, viewer_id: str) -> bool:
    if row.get('deleted_at') or row.get('hidden_at'):
        return False
    # Hold public listings until a moderator is configured and tested.
    if row.get('visibility') == 'public' and not public_posts_allowed():
        return False
    if row['author_id']==viewer_id: return True
    from trust_voice_blocking import identities_blocked
    if identities_blocked(str(row['author_id']), viewer_id):
        return False
    v=row['visibility']
    if v=='public': return True
    if v=='trust_circle': return trust_circle_contains(str(row['author_id']),viewer_id)
    if v=='group' and row.get('group_id'):
        try:
            _require_group_member(str(row['group_id']),viewer_id)
            return True
        except HTTPException:
            return False
    return False


def public_post(row: dict, viewer_id: str) -> dict:
    return {'post_id':row['post_id'],'title':row['title'],'details':row['details'],
            'event_start':row['event_start'],'location':row['location_text'],
            'city':row['city'],'visibility':row['visibility'],'group_id':row.get('group_id'),
            'created_at':row['created_at'],'author_handle':row.get('author_handle') or '',
            'my_post':str(row['author_id'])==viewer_id,
            'going_count':int(row.get('going_count') or 0),
            'my_rsvp':row.get('my_rsvp')}


@router.get('/community/posts')
def list_posts(city: str = Query(default='',max_length=100),
               q: str=Query(default='',max_length=100),limit:int=Query(default=50,ge=1,le=100),
               authorization: Optional[str] = Header(default=None)):
    require_feature('VERIFYD_TRUST_VOICE_COMMUNITY_ENABLED')
    viewer=_identity_from_bearer(authorization)
    viewer_id=str(viewer['id'])
    ensure_schema()
    # Fetch a bounded set, filter access at the server BEFORE returning records.
    with get_db() as conn:
        cur=conn.cursor()
        cur.execute('''SELECT p.*,i.handle_lower AS author_handle,
            (SELECT COUNT(*) FROM verifyd_voice_community_rsvps r
                WHERE r.post_id=p.post_id AND r.status='going') AS going_count,
            (SELECT r.status FROM verifyd_voice_community_rsvps r
                WHERE r.post_id=p.post_id AND r.attendee_id=%s) AS my_rsvp
            FROM verifyd_voice_community_posts p
            JOIN verifyd_identities i ON i.id=p.author_id
            WHERE p.deleted_at IS NULL AND p.hidden_at IS NULL
              AND p.event_start::timestamptz >= NOW() - INTERVAL '1 day'
              AND (%s='' OR p.city ILIKE %s)
              AND (%s='' OR p.title ILIKE %s OR p.details ILIKE %s)
            ORDER BY p.event_start::timestamptz ASC LIMIT 300''',
            (viewer_id,city,f'%{city}%',q,f'%{q}%',f'%{q}%'))
        rows=[dict(x) for x in (cur.fetchall() or [])]
    found=[public_post(x,viewer_id) for x in rows if may_view(x,viewer_id)]
    return {'posts':found[:limit], 'count':len(found[:limit])}


@router.post('/community/posts')
def create_post(payload: PostCreate, authorization: Optional[str]=Header(default=None)):
    require_feature('VERIFYD_TRUST_VOICE_COMMUNITY_ENABLED')
    user=_identity_from_bearer(authorization)
    uid=str(user['id'])
    ensure_schema()
    if payload.rules_accepted is not True:
        raise HTTPException(400,detail={'error':'rules_acknowledgment_required'})
    if payload.visibility == 'public' and not public_posts_allowed():
        raise HTTPException(403,detail={'error':'public_posting_in_review',
            'message':'Public posting is paused during moderator setup. Choose Trust Circle or a private group.'})
    title=clean(payload.title,140)
    details=clean(payload.details,2500)
    city=clean(payload.city,100)
    location=clean(payload.location,160)
    start=validate_start(payload.event_start)
    if re.search(r'\b(?:https?://|www\.)',location,re.I) or re.search(r'\b(?:https?://|www\.)',city,re.I):
        raise HTTPException(400,detail={'error':'invalid_location'})
    group_id=None
    if payload.visibility=='group':
        group_id=(payload.group_id or '').strip()
        if not group_id:
            raise HTTPException(400,detail={'error':'group_required'})
        _require_group_member(group_id,uid)
    pid='tvpost_'+uuid.uuid4().hex
    with get_db() as conn:
        cur=conn.cursor()
        cur.execute('''SELECT COUNT(*) AS n FROM verifyd_voice_community_posts
            WHERE author_id=%s AND created_at::timestamptz > NOW() - INTERVAL '24 hours'
            AND deleted_at IS NULL''',(uid,))
        if int(cur.fetchone()['n']) >= 5:
            raise HTTPException(429,detail={'error':'daily_post_limit', 'message':'You can post up to 5 events per day.'})
        cur.execute('''INSERT INTO verifyd_voice_community_posts
            (post_id,author_id,title,details,event_start,location_text,city,visibility,group_id,created_at)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)''',
            (pid,uid,title,details,start,location,city,payload.visibility,group_id,now_iso()))
    return {'ok':True,'post_id':pid}


def get_visible_post(pid: str, uid: str) -> dict:
    with get_db() as conn:
        cur=conn.cursor()
        cur.execute('SELECT * FROM verifyd_voice_community_posts WHERE post_id=%s',(pid,))
        item=cur.fetchone()
    if not item or not may_view(item,uid):
        raise HTTPException(404,detail={'error':'event_not_found'})
    return dict(item)


@router.post('/community/posts/{post_id}/rsvp')
def rsvp(post_id:str,payload:RsvpPayload,authorization:Optional[str]=Header(default=None)):
    require_feature('VERIFYD_TRUST_VOICE_COMMUNITY_ENABLED')
    user=_identity_from_bearer(authorization)
    uid=str(user['id'])
    ensure_schema()
    get_visible_post(post_id,uid)
    with get_db() as conn:
        cur=conn.cursor()
        cur.execute('''INSERT INTO verifyd_voice_community_rsvps(post_id,attendee_id,status,created_at)
            VALUES (%s,%s,%s,%s)
            ON CONFLICT(post_id,attendee_id) DO UPDATE SET status=EXCLUDED.status,created_at=EXCLUDED.created_at''',
            (post_id,uid,payload.status,now_iso()))
    return {'ok':True,'status':payload.status}


@router.delete('/community/posts/{post_id}')
def delete_post(post_id:str,authorization:Optional[str]=Header(default=None)):
    require_feature('VERIFYD_TRUST_VOICE_COMMUNITY_ENABLED')
    user=_identity_from_bearer(authorization)
    ensure_schema()
    with get_db() as conn:
        cur=conn.cursor()
        cur.execute('''UPDATE verifyd_voice_community_posts SET deleted_at=%s
            WHERE post_id=%s AND author_id=%s AND deleted_at IS NULL''',
            (now_iso(),post_id,str(user['id'])))
        ok=cur.rowcount>0
    if not ok: raise HTTPException(404,detail={'error':'event_not_found'})
    return {'ok':True}


@router.post('/community/posts/{post_id}/report')
def report_post(post_id:str,payload:ReportPayload,authorization:Optional[str]=Header(default=None)):
    require_feature('VERIFYD_TRUST_VOICE_COMMUNITY_ENABLED')
    user=_identity_from_bearer(authorization)
    uid=str(user['id'])
    ensure_schema()
    item=get_visible_post(post_id,uid)
    if item['author_id']==uid:
        raise HTTPException(400,detail={'error':'cannot_report_own_post'})
    with get_db() as conn:
        cur=conn.cursor()
        cur.execute('''INSERT INTO verifyd_voice_community_reports(post_id,reporter_id,reason,created_at)
            VALUES (%s,%s,%s,%s) ON CONFLICT(post_id,reporter_id) DO NOTHING''',
            (post_id,uid,payload.reason,now_iso()))
        cur.execute('SELECT COUNT(*) AS n FROM verifyd_voice_community_reports WHERE post_id=%s',(post_id,))
        count=int(cur.fetchone()['n'])
        if count>=3:
            cur.execute('''UPDATE verifyd_voice_community_posts SET hidden_at=%s
                WHERE post_id=%s AND hidden_at IS NULL''',(now_iso(),post_id))
    return {'ok':True,'message':'Report received. Thank you.', 'hidden_for_review':count>=3}
