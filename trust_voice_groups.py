# ============================================================
# VeriFYD Trust Voice — Group Messaging Foundation
# VERIFYD_TRUST_VOICE_GROUPS_V1
#
# Durable authenticated group messaging foundation.
# This module intentionally leaves working 1:1 messaging untouched.
# Group video will reuse this membership store and must use an SFU.
# ============================================================

from __future__ import annotations

import uuid
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Header, HTTPException, Query
from pydantic import BaseModel, Field

from database import get_db
from trust_voice_messages import (
    _attachment_metadata_by_message_ids,
    _clean_message_body,
    _identity_from_bearer,
    _now_iso,
    _public_contact,
    _require_conversation_member,
    _validate_before_cursor,
    ensure_message_schema,
)
from trust_voice_permissions import trust_circle_contains
from trust_voice_signaling import (
    _identity_by_handle,
    _normalize_handle,
    notify_message_created,
)

router = APIRouter(prefix='/trust-voice', tags=['Trust Voice Groups'])

FEATURE_VERSION = '0.1.1'
_group_schema_ready = False
MAX_GROUP_MEMBERS = 25
MAX_GROUP_NAME_LENGTH = 60


class GroupCreate(BaseModel):
    name: str = Field(min_length=1, max_length=MAX_GROUP_NAME_LENGTH)
    member_handles: list[str] = Field(min_length=1, max_length=MAX_GROUP_MEMBERS - 1)


class GroupUpdate(BaseModel):
    name: str = Field(min_length=1, max_length=MAX_GROUP_NAME_LENGTH)


class GroupMembersAdd(BaseModel):
    member_handles: list[str] = Field(min_length=1, max_length=MAX_GROUP_MEMBERS - 1)


class GroupTextMessageCreate(BaseModel):
    body: str = Field(min_length=1, max_length=4000)


def _clean_group_name(value: str) -> str:
    name = ' '.join(str(value or '').split()).strip()
    if not name:
        raise HTTPException(status_code=400, detail={'error': 'group_name_required', 'message': 'Enter a group name.'})
    if len(name) > MAX_GROUP_NAME_LENGTH:
        raise HTTPException(status_code=400, detail={'error': 'group_name_too_long'})
    return name


def ensure_group_schema() -> None:
    """
    Initialize the group tables/indexes once per application process.

    Group list/detail/message endpoints are hot paths, especially on mobile.
    Re-running CREATE TABLE / CREATE INDEX DDL on every request adds avoidable
    PostgreSQL catalog work and lock coordination.
    """
    global _group_schema_ready

    if _group_schema_ready:
        return

    ensure_message_schema()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute('''
            CREATE TABLE IF NOT EXISTS verifyd_voice_groups (
                conversation_id   TEXT PRIMARY KEY,
                name              TEXT NOT NULL,
                owner_identity_id TEXT NOT NULL,
                created_at        TEXT NOT NULL,
                updated_at        TEXT NOT NULL
            )
        ''')
        cur.execute('''
            CREATE INDEX IF NOT EXISTS idx_verifyd_voice_groups_owner
            ON verifyd_voice_groups (owner_identity_id, updated_at DESC)
        ''')
        cur.execute('''
            CREATE INDEX IF NOT EXISTS idx_verifyd_voice_groups_updated
            ON verifyd_voice_groups (updated_at DESC)
        ''')

    _group_schema_ready = True


def _group_row(conversation_id: str) -> Optional[dict]:
    ensure_group_schema()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute('''
            SELECT g.*, c.updated_at AS conversation_updated_at
            FROM verifyd_voice_groups g
            INNER JOIN verifyd_voice_conversations c
                ON c.conversation_id = g.conversation_id
            WHERE g.conversation_id = %s
              AND c.conversation_type = 'group'
            LIMIT 1
        ''', (conversation_id,))
        row = cur.fetchone()
    return dict(row) if row else None


def _require_group_member(conversation_id: str, identity_id: str) -> dict:
    _require_conversation_member(conversation_id, identity_id)
    group = _group_row(conversation_id)
    if not group:
        raise HTTPException(status_code=404, detail={'error': 'group_not_found'})
    return group


def _require_group_owner(conversation_id: str, identity_id: str) -> dict:
    group = _require_group_member(conversation_id, identity_id)
    if str(group.get('owner_identity_id') or '') != str(identity_id or ''):
        raise HTTPException(status_code=403, detail={'error': 'group_owner_required', 'message': 'Only the group owner can make this change.'})
    return group


def _group_member_ids(conversation_id: str) -> list[str]:
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute('''
            SELECT identity_id
            FROM verifyd_voice_conversation_members
            WHERE conversation_id = %s
            ORDER BY joined_at ASC
        ''', (conversation_id,))
        rows = cur.fetchall() or []
    return [str(row.get('identity_id') or '') for row in rows if row.get('identity_id')]


def _group_member_rows(conversation_id: str) -> list[dict]:
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute('''
            SELECT vm.identity_id, vm.joined_at, vm.last_read_at, i.*
            FROM verifyd_voice_conversation_members vm
            INNER JOIN verifyd_identities i ON i.id = vm.identity_id
            WHERE vm.conversation_id = %s
              AND i.status = 'active'
            ORDER BY vm.joined_at ASC
        ''', (conversation_id,))
        rows = cur.fetchall() or []
    return [dict(row) for row in rows]


def _public_group_member(row: dict, owner_identity_id: str) -> dict:
    public = _public_contact(row)
    public['role'] = 'owner' if str(row.get('id') or row.get('identity_id') or '') == str(owner_identity_id or '') else 'member'
    public['joined_at'] = row.get('joined_at') or ''
    return public


def _group_message_payload(row: dict, viewer_identity_id: str, identity_by_id: dict[str, dict], attachment: Optional[dict] = None) -> dict:
    sender_identity_id = str(row.get('sender_identity_id') or '')
    sender = identity_by_id.get(sender_identity_id)
    payload = {
        'message_id': row.get('message_id'),
        'conversation_id': row.get('conversation_id'),
        'message_type': row.get('message_type') or 'text',
        'body': row.get('body') or '',
        'sender': _public_contact(sender) if sender else {
            'identity_id': sender_identity_id,
            'handle': '',
            'display_name': '',
            'display_emoji': '',
            'display_label': '',
            'profile_media_url': '',
        },
        'is_mine': sender_identity_id == str(viewer_identity_id or ''),
        'read_by_recipient': None,
        'created_at': row.get('created_at') or '',
        'edited_at': row.get('edited_at'),
    }
    if attachment:
        payload['attachment'] = attachment
    return payload


def _full_group_payload(group: dict, viewer_identity_id: str) -> dict:
    conversation_id = str(group.get('conversation_id') or '')
    rows = _group_member_rows(conversation_id)
    members = [_public_group_member(row, str(group.get('owner_identity_id') or '')) for row in rows]
    return {
        'group_id': conversation_id,
        'conversation_id': conversation_id,
        'conversation_type': 'group',
        'name': group.get('name') or 'Group',
        'owner_identity_id': group.get('owner_identity_id') or '',
        'is_owner': str(group.get('owner_identity_id') or '') == str(viewer_identity_id or ''),
        'member_count': len(members),
        'members': members,
        'created_at': group.get('created_at') or '',
        'updated_at': group.get('conversation_updated_at') or group.get('updated_at') or '',
    }


def _resolve_members(owner_identity: dict, handles: list[str]) -> list[dict]:
    cleaned = []
    seen = set()
    owner_handle = str(owner_identity.get('handle_lower') or '').lower()
    for raw in handles:
        handle = _normalize_handle(str(raw or ''))
        if not handle or handle == owner_handle or handle in seen:
            continue
        seen.add(handle)
        cleaned.append(handle)
    if not cleaned:
        raise HTTPException(status_code=400, detail={'error': 'group_members_required', 'message': 'Choose at least one Trust Circle member.'})

    result = []
    for handle in cleaned:
        target = _identity_by_handle(handle)
        if not target:
            raise HTTPException(status_code=404, detail={'error': 'handle_not_found', 'handle': f'@{handle}'})
        if not trust_circle_contains(str(owner_identity.get('id') or ''), str(target.get('id') or '')):
            raise HTTPException(status_code=403, detail={'error': 'trust_circle_required', 'handle': f'@{handle}', 'message': 'Group members must be in your Trust Circle.'})
        result.append(target)
    return result


@router.get('/groups/health')
def groups_health():
    ensure_group_schema()
    return {
        'status': 'ok',
        'feature': 'trust_voice_groups',
        'version': FEATURE_VERSION,
        'group_messaging': 'durable_text_v1',
        'membership_store': 'postgresql_v1',
        'membership_source': 'trust_circle_owner_add_v1',
        'realtime_notifications': 'message_created_best_effort_v1',
        'schema_init': 'once_per_process_v1',
        'attachments': 'deferred',
        'max_group_members': MAX_GROUP_MEMBERS,
        'group_video_ready': 'membership_foundation_v1',
    }


@router.post('/groups')
def create_group(payload: GroupCreate, authorization: Optional[str] = Header(default=None)):
    identity = _identity_from_bearer(authorization)
    ensure_group_schema()
    name = _clean_group_name(payload.name)
    members = _resolve_members(identity, payload.member_handles)
    if len(members) + 1 > MAX_GROUP_MEMBERS:
        raise HTTPException(status_code=400, detail={'error': 'group_too_large', 'max_group_members': MAX_GROUP_MEMBERS})

    conversation_id = 'tvgroup_' + uuid.uuid4().hex
    now = _now_iso()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute('''
            INSERT INTO verifyd_voice_conversations (
                conversation_id, conversation_type, direct_key, created_at, updated_at
            ) VALUES (%s, 'group', %s, %s, %s)
        ''', (conversation_id, f'group::{conversation_id}', now, now))
        cur.execute('''
            INSERT INTO verifyd_voice_groups (
                conversation_id, name, owner_identity_id, created_at, updated_at
            ) VALUES (%s, %s, %s, %s, %s)
        ''', (conversation_id, name, identity['id'], now, now))
        for member_id in [str(identity['id']), *[str(member['id']) for member in members]]:
            cur.execute('''
                INSERT INTO verifyd_voice_conversation_members (
                    conversation_id, identity_id, joined_at, last_read_at
                ) VALUES (%s, %s, %s, NULL)
            ''', (conversation_id, member_id, now))

    group = _group_row(conversation_id)
    return {'ok': True, 'group': _full_group_payload(group, str(identity['id']))}


@router.get('/groups')
def list_groups(authorization: Optional[str] = Header(default=None)):
    identity = _identity_from_bearer(authorization)
    ensure_group_schema()
    viewer_id = str(identity['id'])

    with get_db() as conn:
        cur = conn.cursor()
        cur.execute('''
            SELECT g.*, c.updated_at AS conversation_updated_at, vm.last_read_at AS viewer_last_read_at
            FROM verifyd_voice_groups g
            INNER JOIN verifyd_voice_conversations c
                ON c.conversation_id = g.conversation_id AND c.conversation_type = 'group'
            INNER JOIN verifyd_voice_conversation_members vm
                ON vm.conversation_id = g.conversation_id AND vm.identity_id = %s
            ORDER BY c.updated_at DESC
        ''', (viewer_id,))
        groups = [dict(row) for row in (cur.fetchall() or [])]
        if not groups:
            return {'count': 0, 'total_unread': 0, 'groups': []}

        ids = [str(row['conversation_id']) for row in groups]
        cur.execute('''
            SELECT conversation_id, COUNT(*) AS member_count
            FROM verifyd_voice_conversation_members
            WHERE conversation_id = ANY(%s)
            GROUP BY conversation_id
        ''', (ids,))
        counts = {str(row['conversation_id']): int(row.get('member_count') or 0) for row in (cur.fetchall() or [])}

        cur.execute('''
            SELECT DISTINCT ON (conversation_id) *
            FROM verifyd_voice_messages
            WHERE conversation_id = ANY(%s) AND deleted_at IS NULL
            ORDER BY conversation_id, created_at DESC
        ''', (ids,))
        latest = {str(row['conversation_id']): dict(row) for row in (cur.fetchall() or [])}
        sender_ids = list({str(row.get('sender_identity_id') or '') for row in latest.values() if row.get('sender_identity_id')})
        identity_by_id = {}
        if sender_ids:
            cur.execute('SELECT * FROM verifyd_identities WHERE id = ANY(%s) AND status = \'active\'', (sender_ids,))
            identity_by_id = {str(row['id']): dict(row) for row in (cur.fetchall() or [])}

        cur.execute('''
            SELECT msg.conversation_id, COUNT(*) AS unread_count
            FROM verifyd_voice_messages msg
            INNER JOIN verifyd_voice_conversation_members vm
                ON vm.conversation_id = msg.conversation_id AND vm.identity_id = %s
            WHERE msg.conversation_id = ANY(%s)
              AND msg.sender_identity_id <> %s
              AND msg.deleted_at IS NULL
              AND (vm.last_read_at IS NULL OR msg.created_at::timestamptz > vm.last_read_at::timestamptz)
            GROUP BY msg.conversation_id
        ''', (viewer_id, ids, viewer_id))
        unread = {str(row['conversation_id']): int(row.get('unread_count') or 0) for row in (cur.fetchall() or [])}

    result = []
    for group in groups:
        cid = str(group['conversation_id'])
        last = latest.get(cid)
        result.append({
            'group_id': cid,
            'conversation_id': cid,
            'conversation_type': 'group',
            'name': group.get('name') or 'Group',
            'owner_identity_id': group.get('owner_identity_id') or '',
            'is_owner': str(group.get('owner_identity_id') or '') == viewer_id,
            'member_count': counts.get(cid, 0),
            'unread_count': unread.get(cid, 0),
            'last_message': _group_message_payload(last, viewer_id, identity_by_id) if last else None,
            'created_at': group.get('created_at') or '',
            'updated_at': group.get('conversation_updated_at') or group.get('updated_at') or '',
        })

    return {'count': len(result), 'total_unread': sum(int(g['unread_count']) for g in result), 'groups': result}


@router.get('/groups/{conversation_id}')
def get_group(conversation_id: str, authorization: Optional[str] = Header(default=None)):
    identity = _identity_from_bearer(authorization)
    group = _require_group_member(conversation_id, str(identity['id']))
    return {'group': _full_group_payload(group, str(identity['id']))}


@router.patch('/groups/{conversation_id}')
def rename_group(conversation_id: str, payload: GroupUpdate, authorization: Optional[str] = Header(default=None)):
    identity = _identity_from_bearer(authorization)
    _require_group_owner(conversation_id, str(identity['id']))
    name = _clean_group_name(payload.name)
    now = _now_iso()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute('UPDATE verifyd_voice_groups SET name = %s, updated_at = %s WHERE conversation_id = %s', (name, now, conversation_id))
        cur.execute('UPDATE verifyd_voice_conversations SET updated_at = %s WHERE conversation_id = %s', (now, conversation_id))
    group = _group_row(conversation_id)
    return {'ok': True, 'group': _full_group_payload(group, str(identity['id']))}


@router.post('/groups/{conversation_id}/members')
def add_group_members(conversation_id: str, payload: GroupMembersAdd, authorization: Optional[str] = Header(default=None)):
    identity = _identity_from_bearer(authorization)
    _require_group_owner(conversation_id, str(identity['id']))
    requested = _resolve_members(identity, payload.member_handles)
    existing = set(_group_member_ids(conversation_id))
    to_add = [member for member in requested if str(member['id']) not in existing]
    if len(existing) + len(to_add) > MAX_GROUP_MEMBERS:
        raise HTTPException(status_code=400, detail={'error': 'group_too_large', 'max_group_members': MAX_GROUP_MEMBERS})
    now = _now_iso()
    with get_db() as conn:
        cur = conn.cursor()
        for member in to_add:
            cur.execute('''
                INSERT INTO verifyd_voice_conversation_members (
                    conversation_id, identity_id, joined_at, last_read_at
                ) VALUES (%s, %s, %s, NULL)
                ON CONFLICT (conversation_id, identity_id) DO NOTHING
            ''', (conversation_id, member['id'], now))
        cur.execute('UPDATE verifyd_voice_groups SET updated_at = %s WHERE conversation_id = %s', (now, conversation_id))
        cur.execute('UPDATE verifyd_voice_conversations SET updated_at = %s WHERE conversation_id = %s', (now, conversation_id))
    group = _group_row(conversation_id)
    return {'ok': True, 'added_count': len(to_add), 'group': _full_group_payload(group, str(identity['id']))}


@router.delete('/groups/{conversation_id}/members/{member_identity_id}')
def remove_group_member(conversation_id: str, member_identity_id: str, authorization: Optional[str] = Header(default=None)):
    identity = _identity_from_bearer(authorization)
    group = _require_group_owner(conversation_id, str(identity['id']))
    member_identity_id = str(member_identity_id or '').strip()
    if member_identity_id == str(group.get('owner_identity_id') or ''):
        raise HTTPException(status_code=400, detail={'error': 'cannot_remove_group_owner'})
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute('DELETE FROM verifyd_voice_conversation_members WHERE conversation_id = %s AND identity_id = %s', (conversation_id, member_identity_id))
        removed = cur.rowcount > 0
        now = _now_iso()
        if removed:
            cur.execute('UPDATE verifyd_voice_groups SET updated_at = %s WHERE conversation_id = %s', (now, conversation_id))
            cur.execute('UPDATE verifyd_voice_conversations SET updated_at = %s WHERE conversation_id = %s', (now, conversation_id))
    if not removed:
        raise HTTPException(status_code=404, detail={'error': 'group_member_not_found'})
    refreshed = _group_row(conversation_id)
    return {'ok': True, 'group': _full_group_payload(refreshed, str(identity['id']))}


@router.post('/groups/{conversation_id}/leave')
def leave_group(conversation_id: str, authorization: Optional[str] = Header(default=None)):
    identity = _identity_from_bearer(authorization)
    identity_id = str(identity['id'])
    group = _require_group_member(conversation_id, identity_id)
    member_ids = _group_member_ids(conversation_id)

    if len(member_ids) <= 1:
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute('DELETE FROM verifyd_voice_messages WHERE conversation_id = %s', (conversation_id,))
            cur.execute('DELETE FROM verifyd_voice_conversation_members WHERE conversation_id = %s', (conversation_id,))
            cur.execute('DELETE FROM verifyd_voice_groups WHERE conversation_id = %s', (conversation_id,))
            cur.execute('DELETE FROM verifyd_voice_conversations WHERE conversation_id = %s', (conversation_id,))
        return {'ok': True, 'left': True, 'group_deleted': True}

    replacement_owner = None
    if str(group.get('owner_identity_id') or '') == identity_id:
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute('''
                SELECT identity_id FROM verifyd_voice_conversation_members
                WHERE conversation_id = %s AND identity_id <> %s
                ORDER BY joined_at ASC LIMIT 1
            ''', (conversation_id, identity_id))
            row = cur.fetchone() or {}
            replacement_owner = str(row.get('identity_id') or '')
        if not replacement_owner:
            raise HTTPException(status_code=409, detail={'error': 'owner_transfer_failed'})

    now = _now_iso()
    with get_db() as conn:
        cur = conn.cursor()
        if replacement_owner:
            cur.execute('UPDATE verifyd_voice_groups SET owner_identity_id = %s, updated_at = %s WHERE conversation_id = %s', (replacement_owner, now, conversation_id))
        cur.execute('DELETE FROM verifyd_voice_conversation_members WHERE conversation_id = %s AND identity_id = %s', (conversation_id, identity_id))
        cur.execute('UPDATE verifyd_voice_conversations SET updated_at = %s WHERE conversation_id = %s', (now, conversation_id))
    return {'ok': True, 'left': True, 'group_deleted': False, 'new_owner_identity_id': replacement_owner}


@router.get('/groups/{conversation_id}/messages')
def list_group_messages(
    conversation_id: str,
    authorization: Optional[str] = Header(default=None),
    limit: int = Query(default=50, ge=1, le=100),
    before: Optional[str] = Query(default=None, max_length=80),
):
    identity = _identity_from_bearer(authorization)
    viewer_id = str(identity['id'])
    _require_group_member(conversation_id, viewer_id)
    before = _validate_before_cursor(before)
    fetch_limit = limit + 1

    with get_db() as conn:
        cur = conn.cursor()
        if before:
            cur.execute('''
                SELECT * FROM verifyd_voice_messages
                WHERE conversation_id = %s AND deleted_at IS NULL
                  AND created_at::timestamptz < %s::timestamptz
                ORDER BY created_at DESC LIMIT %s
            ''', (conversation_id, before, fetch_limit))
        else:
            cur.execute('''
                SELECT * FROM verifyd_voice_messages
                WHERE conversation_id = %s AND deleted_at IS NULL
                ORDER BY created_at DESC LIMIT %s
            ''', (conversation_id, fetch_limit))
        rows = [dict(row) for row in (cur.fetchall() or [])]

    has_more = len(rows) > limit
    page_rows = rows[:limit]
    sender_ids = list({str(row.get('sender_identity_id') or '') for row in page_rows if row.get('sender_identity_id')})
    identity_by_id = {}
    if sender_ids:
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute('SELECT * FROM verifyd_identities WHERE id = ANY(%s) AND status = \'active\'', (sender_ids,))
            identity_by_id = {str(row['id']): dict(row) for row in (cur.fetchall() or [])}

    message_ids = [str(row['message_id']) for row in page_rows if row.get('message_id')]
    attachment_by_message_id = _attachment_metadata_by_message_ids(message_ids)
    messages_desc = []
    for row in page_rows:
        mid = str(row.get('message_id') or '')
        messages_desc.append(_group_message_payload(row, viewer_id, identity_by_id, attachment_by_message_id.get(mid)))

    return {
        'group_id': conversation_id,
        'conversation_id': conversation_id,
        'count': len(messages_desc),
        'messages': list(reversed(messages_desc)),
        'has_more': has_more,
        'next_before': str(page_rows[-1].get('created_at') or '') if has_more and page_rows else None,
        'page_size': limit,
    }


@router.post('/groups/{conversation_id}/messages')
def send_group_message(
    conversation_id: str,
    payload: GroupTextMessageCreate,
    background_tasks: BackgroundTasks,
    authorization: Optional[str] = Header(default=None),
):
    identity = _identity_from_bearer(authorization)
    identity_id = str(identity['id'])
    _require_group_member(conversation_id, identity_id)
    body = _clean_message_body(payload.body)
    message_id = 'tvmsg_' + uuid.uuid4().hex
    now = _now_iso()

    with get_db() as conn:
        cur = conn.cursor()
        cur.execute('''
            INSERT INTO verifyd_voice_messages (
                message_id, conversation_id, sender_identity_id,
                message_type, body, created_at, edited_at, deleted_at
            ) VALUES (%s, %s, %s, 'text', %s, %s, NULL, NULL)
        ''', (message_id, conversation_id, identity_id, body, now))
        cur.execute('UPDATE verifyd_voice_conversations SET updated_at = %s WHERE conversation_id = %s', (now, conversation_id))
        cur.execute('UPDATE verifyd_voice_groups SET updated_at = %s WHERE conversation_id = %s', (now, conversation_id))
        cur.execute('SELECT * FROM verifyd_voice_messages WHERE message_id = %s LIMIT 1', (message_id,))
        row = dict(cur.fetchone())

    for recipient_id in _group_member_ids(conversation_id):
        if recipient_id == identity_id:
            continue
        background_tasks.add_task(
            notify_message_created,
            recipient_identity_id=recipient_id,
            conversation_id=conversation_id,
            message_id=message_id,
            sender=_public_contact(identity),
            created_at=now,
            message_type='text',
        )

    return {
        'ok': True,
        'message': _group_message_payload(row, identity_id, {identity_id: identity}),
    }
