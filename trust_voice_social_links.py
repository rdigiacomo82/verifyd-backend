"""VeriFYD Circle Social Connections v1: optional user-managed profile links.

No social-network login or account verification. Authenticated viewers only;
links stay private unless the owner explicitly changes visibility.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Literal, Optional
from urllib.parse import urlsplit, urlunsplit

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

from database import get_db
from trust_voice_messages import _identity_from_bearer
from trust_voice_signaling import _identity_by_handle, _normalize_handle
from trust_voice_permissions import trust_circle_contains
from trust_voice_blocking import identities_blocked

router = APIRouter(prefix='/social', tags=['VeriFYD Circle Social Connections'])
PLATFORMS = frozenset({'instagram', 'facebook', 'youtube', 'tiktok', 'linkedin', 'x'})
VISIBILITY = frozenset({'private', 'trust_circle', 'everyone'})
HOSTS = {
    'instagram': {'instagram.com', 'www.instagram.com'},
    'facebook': {'facebook.com', 'www.facebook.com', 'm.facebook.com'},
    'youtube': {'youtube.com', 'www.youtube.com'},
    'tiktok': {'tiktok.com', 'www.tiktok.com'},
    'linkedin': {'linkedin.com', 'www.linkedin.com'},
    'x': {'x.com', 'www.x.com', 'twitter.com', 'www.twitter.com'},
}
MAX_LINKS = 4

class SocialLinkInput(BaseModel):
    platform: str = Field(min_length=1, max_length=24)
    url: str = Field(min_length=10, max_length=400)
    visibility: Literal['private', 'trust_circle', 'everyone'] = 'private'
    position: int = Field(ge=0, le=3)

class SocialLinksUpdate(BaseModel):
    links: list[SocialLinkInput] = Field(default_factory=list, max_length=4)

def _normalize_url(platform: str, supplied: str) -> str:
    try:
        parsed = urlsplit(str(supplied or '').strip())
        host = (parsed.hostname or '').lower().rstrip('.')
        if (parsed.scheme.lower() != 'https' or host not in HOSTS[platform]
                or parsed.port is not None or parsed.username or parsed.password
                or parsed.query or parsed.fragment):
            raise ValueError('Unsafe or unsupported URL')
        segments = [v for v in parsed.path.split('/') if v]
        if not segments or len(segments) > 3:
            raise ValueError('A profile URL is required')
        first = segments[0].lower()
        if first in {'login', 'share', 'watch', 'explore', 'accounts', 'redirect', 'search', 'messages', 'settings'}:
            raise ValueError('Use a social profile URL, not a general page')
        if platform == 'tiktok' and not first.startswith('@'):
            raise ValueError('TikTok profile URLs begin with @')
        if platform == 'youtube' and not (first.startswith('@') or first in {'channel', 'user', 'c'}):
            raise ValueError('Use a YouTube channel URL')
        if platform == 'linkedin' and (first not in {'in', 'company', 'school'} or len(segments) < 2):
            raise ValueError('Use a LinkedIn profile or company URL')
        if platform in {'instagram', 'x'} and len(segments) != 1:
            raise ValueError('Use a direct social profile URL')
        return urlunsplit(('https', host, '/' + '/'.join(segments), '', ''))
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail={
            'error': 'invalid_social_profile',
            'message': f'Please enter a valid, direct {platform} profile URL using HTTPS.',
        }) from exc


def _ensure_schema():
    with get_db() as db:
        cursor = db.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS verifyd_circle_social_links (
            owner_id TEXT NOT NULL REFERENCES verifyd_identities(id) ON DELETE CASCADE,
            platform TEXT NOT NULL,
            profile_url TEXT NOT NULL,
            visibility TEXT NOT NULL DEFAULT 'private',
            position INTEGER NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY(owner_id, platform),
            UNIQUE(owner_id, position)
        )''')
        cursor.execute('''CREATE INDEX IF NOT EXISTS idx_circle_social_owner
            ON verifyd_circle_social_links(owner_id, position)''')


def _list_links(owner_id: str) -> list[dict]:
    _ensure_schema()
    with get_db() as db:
        cursor = db.cursor()
        cursor.execute('''SELECT platform, profile_url, visibility, position
            FROM verifyd_circle_social_links WHERE owner_id=%s ORDER BY position''', (owner_id,))
        return [{'platform': r['platform'], 'url': r['profile_url'],
                 'visibility': r['visibility'], 'position': r['position']}
                for r in (cursor.fetchall() or [])]


@router.get('/links/me')
def get_my_social_links(authorization: Optional[str] = Header(default=None)):
    actor = _identity_from_bearer(authorization)
    return {'links': _list_links(str(actor['id']))}


@router.put('/links/me')
def set_my_social_links(payload: SocialLinksUpdate,
                        authorization: Optional[str] = Header(default=None)):
    actor = _identity_from_bearer(authorization)
    owner_id = str(actor['id'])
    if len(payload.links) > MAX_LINKS:
        raise HTTPException(422, detail={'error': 'too_many_social_links'})
    seen, slots, clean = set(), set(), []
    for item in payload.links:
        platform = item.platform.strip().lower()
        if platform not in PLATFORMS or platform in seen or item.position in slots:
            raise HTTPException(422, detail={'error': 'duplicate_or_unsupported_social_link',
                                             'message': 'Choose up to four different social networks.'})
        seen.add(platform)
        slots.add(item.position)
        clean.append((owner_id, platform, _normalize_url(platform, item.url),
                      item.visibility, item.position,
                      datetime.now(timezone.utc).isoformat()))
    _ensure_schema()
    with get_db() as db:
        cursor = db.cursor()
        cursor.execute('DELETE FROM verifyd_circle_social_links WHERE owner_id=%s', (owner_id,))
        for values in clean:
            cursor.execute('''INSERT INTO verifyd_circle_social_links
                (owner_id,platform,profile_url,visibility,position,updated_at)
                VALUES(%s,%s,%s,%s,%s,%s)''', values)
    return {'ok': True, 'links': _list_links(owner_id)}


@router.get('/links/{handle}')
def get_public_social_links(handle: str, authorization: Optional[str] = Header(default=None)):
    viewer = _identity_from_bearer(authorization)
    owner = _identity_by_handle(_normalize_handle(handle))
    if not owner or str(owner.get('status') or 'active') != 'active':
        raise HTTPException(404, detail={'error': 'profile_not_found'})
    owner_id, viewer_id = str(owner['id']), str(viewer['id'])
    is_owner = owner_id == viewer_id
    is_circle_member = False
    links = _list_links(owner_id)
    if not is_owner and identities_blocked(owner_id, viewer_id):
        return {'handle': '@' + str(owner.get('handle_lower') or owner.get('handle') or ''), 'links': []}
    if not is_owner and any(r['visibility'] == 'trust_circle' for r in links):
        is_circle_member = bool(trust_circle_contains(owner_id, viewer_id))
    shown = [r for r in links if (is_owner or r['visibility'] == 'everyone'
                                  or (r['visibility'] == 'trust_circle' and is_circle_member))]
    return {'handle': '@' + str(owner.get('handle_lower') or owner.get('handle') or ''),
            'links': shown}


@router.get('/health')
def social_health():
    return {'status': 'ok', 'version': '1.0.0', 'max_links': MAX_LINKS,
            'account_authentication': 'required',
            'social_account_verification': 'not_claimed'}
