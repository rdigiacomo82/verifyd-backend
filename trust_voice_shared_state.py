"""Redis-backed shared state for horizontally scaled Trust Voice signaling.

This module is deliberately opt-in.  With
``VERIFYD_TRUST_VOICE_SHARED_STATE_ENABLED`` unset/false, callers should keep
using the existing single-process path.  When enabled, a missing or unhealthy
Redis connection raises an error instead of silently falling back to local
state (which would split presence and calls across instances).

Redis stores short-lived presence leases, best-effort websocket fan-out, and
ephemeral active-call coordination.  PostgreSQL call history remains the
durable audit record.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from typing import Any, Awaitable, Callable, Optional

log = logging.getLogger("verifyd.trust_voice.shared_state")

PRESENCE_TTL_SECONDS = 45
CALL_STATE_TTL_SECONDS = 6 * 60 * 60
PRESENCE_HEARTBEAT_SECONDS = 15
RING_TTL_SECONDS = 30
EVENT_CHANNEL = "verifyd:trust-voice:events:v1"

EventHandler = Callable[[str, dict], Awaitable[None]]


def enabled() -> bool:
    return (os.getenv("VERIFYD_TRUST_VOICE_SHARED_STATE_ENABLED", "")
            .strip().lower() in {"1", "true", "yes", "on"})


class VoiceSharedState:
    """One Redis connection and event subscription per backend process."""

    def __init__(self, redis_url: str, event_handler: EventHandler,
                 expiry_handler: Optional[EventHandler] = None):
        self.redis_url = redis_url
        self.event_handler = event_handler
        self.expiry_handler = expiry_handler
        self.node_id = uuid.uuid4().hex
        self._redis = None
        self._pubsub = None
        self._listener: Optional[asyncio.Task] = None
        self._heartbeat: Optional[asyncio.Task] = None
        self._reaper: Optional[asyncio.Task] = None
        self._local_identity_counts: dict[str, int] = {}
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        if self._listener and not self._listener.done():
            return
        if not self.redis_url:
            raise RuntimeError("REDIS_URL is required for shared Trust Voice state")
        try:
            from redis.asyncio import Redis
        except ImportError as exc:
            raise RuntimeError("redis package is required for shared Trust Voice state") from exc
        self._redis = Redis.from_url(
            self.redis_url,
            decode_responses=True,
            socket_connect_timeout=3,
            socket_timeout=3,
            health_check_interval=20,
        )
        await self._redis.ping()
        self._pubsub = self._redis.pubsub(ignore_subscribe_messages=True)
        await self._pubsub.subscribe(EVENT_CHANNEL)
        self._listener = asyncio.create_task(self._listen(), name="trust-voice-redis-events")
        self._heartbeat = asyncio.create_task(self._heartbeat_loop(), name="trust-voice-presence-heartbeat")
        self._reaper = asyncio.create_task(self._reaper_loop(), name="trust-voice-call-expiry-reaper")

    async def close(self) -> None:
        for task in (self._listener, self._heartbeat, self._reaper):
            if task and not task.done():
                task.cancel()
        for task in (self._listener, self._heartbeat, self._reaper):
            if task:
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
        if self._pubsub:
            await self._pubsub.aclose()
        if self._redis:
            await self._redis.aclose()
        self._listener = None
        self._heartbeat = None
        self._reaper = None
        self._pubsub = None
        self._redis = None

    async def _listen(self) -> None:
        try:
            async for message in self._pubsub.listen():
                if message.get("type") != "message":
                    continue
                try:
                    event = json.loads(message.get("data") or "{}")
                    if event.get("source") == self.node_id:
                        continue
                    identity_id = str(event.get("identity_id") or "")
                    payload = event.get("payload")
                    if identity_id and isinstance(payload, dict):
                        await self.event_handler(identity_id, payload)
                except Exception:
                    log.warning("Invalid Trust Voice Redis fan-out event", exc_info=True)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("Trust Voice Redis event listener stopped")
            # Do not silently switch to local routing when shared mode is on.

    async def _presence_key(self, identity_id: str) -> str:
        return f"verifyd:trust-voice:presence:{identity_id}"

    async def _refresh_presence(self, identity_id: str) -> None:
        key = await self._presence_key(identity_id)
        expires_at = time.time() + PRESENCE_TTL_SECONDS
        await self._redis.zadd(key, {self.node_id: expires_at})
        await self._redis.expire(key, PRESENCE_TTL_SECONDS * 2)

    async def add_connection(self, identity_id: str) -> None:
        await self.start()
        async with self._lock:
            self._local_identity_counts[identity_id] = self._local_identity_counts.get(identity_id, 0) + 1
            await self._refresh_presence(identity_id)

    async def remove_connection(self, identity_id: str) -> None:
        if not self._redis:
            return
        async with self._lock:
            count = self._local_identity_counts.get(identity_id, 0) - 1
            if count > 0:
                self._local_identity_counts[identity_id] = count
                return
            self._local_identity_counts.pop(identity_id, None)
            key = await self._presence_key(identity_id)
            await self._redis.zrem(key, self.node_id)

    async def _heartbeat_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(PRESENCE_HEARTBEAT_SECONDS)
                async with self._lock:
                    identities = tuple(self._local_identity_counts)
                if identities:
                    pipe = self._redis.pipeline(transaction=False)
                    expires_at = time.time() + PRESENCE_TTL_SECONDS
                    for identity_id in identities:
                        key = await self._presence_key(identity_id)
                        pipe.zadd(key, {self.node_id: expires_at})
                        pipe.expire(key, PRESENCE_TTL_SECONDS * 2)
                    await pipe.execute()
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("Trust Voice shared presence heartbeat failed")

    async def is_online(self, identity_id: str) -> bool:
        await self.start()
        key = await self._presence_key(identity_id)
        now = time.time()
        await self._redis.zremrangebyscore(key, "-inf", now)
        return bool(await self._redis.zcard(key))

    async def publish_identity_event(self, identity_id: str, payload: dict) -> None:
        await self.start()
        event = json.dumps({
            "source": self.node_id,
            "identity_id": str(identity_id),
            "payload": payload,
        }, separators=(",", ":"), ensure_ascii=False)
        await self._redis.publish(EVENT_CHANNEL, event)

    async def health(self) -> bool:
        await self.start()
        return bool(await self._redis.ping())

    @staticmethod
    def _call_key(call_id: str) -> str:
        return f"verifyd:trust-voice:call:{call_id}"

    @staticmethod
    def _identity_call_index(identity_id: str) -> str:
        return f"verifyd:trust-voice:calls:{identity_id}"

    @staticmethod
    def _serializable_call(call: dict) -> dict:
        return {k: v for k, v in call.items() if k != "android_media_socket"}

    async def put_call(self, call: dict) -> None:
        await self.start()
        record = self._serializable_call(call)
        call_id = str(record.get("call_id") or "")
        caller_id = str(record.get("caller_id") or "")
        callee_id = str(record.get("callee_id") or "")
        if not call_id or not caller_id or not callee_id:
            raise ValueError("call_id, caller_id and callee_id are required")
        expires_at = float(record.get("created_at") or time.time()) + CALL_STATE_TTL_SECONDS
        pipe = self._redis.pipeline(transaction=True)
        pipe.set(self._call_key(call_id), json.dumps(record, separators=(",", ":"), ensure_ascii=False), ex=CALL_STATE_TTL_SECONDS)
        pipe.zadd(self._identity_call_index(caller_id), {call_id: expires_at})
        pipe.zadd(self._identity_call_index(callee_id), {call_id: expires_at})
        pipe.zadd("verifyd:trust-voice:ringing-expiry", {
            call_id: float(record.get("created_at") or time.time()) + RING_TTL_SECONDS
        })
        await pipe.execute()

    async def get_call(self, call_id: str) -> Optional[dict]:
        await self.start()
        raw = await self._redis.get(self._call_key(call_id))
        if not raw:
            return None
        try:
            return json.loads(raw)
        except (TypeError, ValueError):
            log.error("Invalid shared Trust Voice call record call_id=%s", call_id)
            return None

    async def list_calls(self, identity_id: str) -> list[dict]:
        await self.start()
        key = self._identity_call_index(identity_id)
        now = time.time()
        await self._redis.zremrangebyscore(key, "-inf", now)
        call_ids = await self._redis.zrangebyscore(key, now, "+inf")
        records = await asyncio.gather(*(self.get_call(str(cid)) for cid in call_ids))
        return [r for r in records if r]

    async def update_call(self, call_id: str, *, expected_state: Optional[str] = None,
                          changes: Optional[dict] = None, remove: bool = False) -> Optional[dict]:
        """Optimistically update a call record; competing transitions retry safely."""
        await self.start()
        try:
            from redis.exceptions import WatchError
        except ImportError as exc:
            raise RuntimeError("redis package is required for shared Trust Voice state") from exc
        key = self._call_key(call_id)
        for _ in range(8):
            async with self._redis.pipeline(transaction=True) as pipe:
                try:
                    await pipe.watch(key)
                    raw = await pipe.get(key)
                    if not raw:
                        await pipe.unwatch()
                        return None
                    record = json.loads(raw)
                    if expected_state and record.get("state") != expected_state:
                        await pipe.unwatch()
                        return None
                    updated = dict(record)
                    if changes:
                        updated.update(changes)
                    caller_id = str(updated.get("caller_id") or "")
                    callee_id = str(updated.get("callee_id") or "")
                    pipe.multi()
                    if remove:
                        pipe.delete(key)
                        if caller_id:
                            pipe.zrem(self._identity_call_index(caller_id), call_id)
                        if callee_id:
                            pipe.zrem(self._identity_call_index(callee_id), call_id)
                        pipe.zrem("verifyd:trust-voice:ringing-expiry", call_id)
                    else:
                        pipe.set(key, json.dumps(updated, separators=(",", ":"), ensure_ascii=False), ex=CALL_STATE_TTL_SECONDS)
                        if updated.get("state") != "ringing":
                            pipe.zrem("verifyd:trust-voice:ringing-expiry", call_id)
                    await pipe.execute()
                    return updated
                except WatchError:
                    continue
        raise RuntimeError("Trust Voice call state remained contended")

    async def _reaper_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(1)
                if not self.expiry_handler:
                    continue
                now = time.time()
                call_ids = await self._redis.zrangebyscore(
                    "verifyd:trust-voice:ringing-expiry", "-inf", now,
                    start=0, num=100,
                )
                for call_id in call_ids:
                    call = await self.update_call(
                        str(call_id), expected_state="ringing",
                        changes={"state": "missed"}, remove=True,
                    )
                    if call:
                        try:
                            await self.expiry_handler(str(call_id), call)
                        except Exception:
                            log.exception("Trust Voice distributed call expiry handler failed")
                    else:
                        await self._redis.zrem("verifyd:trust-voice:ringing-expiry", call_id)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("Trust Voice distributed call expiry reaper stopped")

    async def acquire_invite_cooldown(self, identity_id: str, seconds: int) -> bool:
        await self.start()
        return bool(await self._redis.set(
            f"verifyd:trust-voice:invite-cooldown:{identity_id}",
            str(time.time()), ex=max(1, int(seconds)), nx=True,
        ))


_instance: Optional[VoiceSharedState] = None


async def get_voice_shared_state(event_handler: EventHandler,
                                 expiry_handler: Optional[EventHandler] = None) -> VoiceSharedState:
    global _instance
    if _instance is None:
        _instance = VoiceSharedState(os.getenv("REDIS_URL", ""), event_handler, expiry_handler)
    elif _instance.event_handler is not event_handler:
        _instance.event_handler = event_handler
        _instance.expiry_handler = expiry_handler
    await _instance.start()
    return _instance


async def close_voice_shared_state() -> None:
    global _instance
    if _instance is not None:
        await _instance.close()
        _instance = None
