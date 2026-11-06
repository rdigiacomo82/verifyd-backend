"""Test-only session state; NEVER places PSTN or emergency calls.

No live user data, precise coordinates, callback identifiers, or audio recordings.
The exclusive destination is a named simulated endpoint, not a phone number.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from threading import RLock
from uuid import uuid4

LAB_DESTINATION = 'VFVID_LAB_ONLY'
SESSION_LIFETIME_SECONDS = 180
MAX_ACTIVE_SESSIONS = 32


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class LabSession:
    id: str
    destination: str
    location_label: str
    created_at: datetime
    expires_at: datetime
    status: str = 'created'
    sip_probe_status: str = 'not_run'

    def public(self) -> dict:
        return {
            'id': self.id,
            'destination': self.destination,
            'synthetic_location': self.location_label,
            'created_at': self.created_at.isoformat(),
            'expires_at': self.expires_at.isoformat(),
            'status': self.status,
            'sip_probe_status': self.sip_probe_status,
            'real_emergency_call': False,
            'media_connected': False,
            'message': 'LAB ONLY: no 911 routing, no dispatcher, no voice media',
        }


class LabStore:
    def __init__(self) -> None:
        self._sessions: dict[str, LabSession] = {}
        self._lock = RLock()

    def _prune(self) -> None:
        now = utcnow()
        for key, value in list(self._sessions.items()):
            if value.expires_at < now:
                del self._sessions[key]

    def create(self, *, location_label: str = 'SYNTHETIC_LAB_LOCATION') -> LabSession:
        if location_label != 'SYNTHETIC_LAB_LOCATION':
            raise ValueError('Only synthetic location labels are permitted')
        with self._lock:
            self._prune()
            if len(self._sessions) >= MAX_ACTIVE_SESSIONS:
                raise OverflowError('Lab capacity reached')
            now = utcnow()
            item = LabSession(
                id=str(uuid4()), destination=LAB_DESTINATION,
                location_label=location_label,
                created_at=now,
                expires_at=now + timedelta(seconds=SESSION_LIFETIME_SECONDS),
            )
            self._sessions[item.id] = item
            return item

    def get(self, sid: str) -> LabSession | None:
        with self._lock:
            self._prune()
            return self._sessions.get(sid)

    def set_probe(self, sid: str, status: str) -> LabSession:
        if status not in {'sip_options_acknowledged', 'sip_probe_failed', 'sip_probe_unavailable'}:
            raise ValueError('Invalid lab SIP status')
        with self._lock:
            self._prune()
            item = self._sessions.get(sid)
            if item is None or item.status == 'closed':
                raise KeyError(sid)
            item.sip_probe_status = status
            item.status = 'lab_signaling_checked'
            return item

    def close(self, sid: str) -> LabSession:
        with self._lock:
            self._prune()
            item = self._sessions.get(sid)
            if item is None:
                raise KeyError(sid)
            item.status = 'closed'
            return item