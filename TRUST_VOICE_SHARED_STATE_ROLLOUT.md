# Trust Voice shared signaling state rollout

The Redis integration is opt-in. With `VERIFYD_TRUST_VOICE_SHARED_STATE_ENABLED` unset or false, signaling continues using the existing single-process state path.

## Staging setup

1. Provision a **separate staging Render Key Value** instance. Do not point staging at the production Redis URL.
2. Add the staging Key Value's private URL as `REDIS_URL` on the staging backend service.
3. Add `VERIFYD_TRUST_VOICE_SHARED_STATE_ENABLED=1` to that staging service only.
4. Deploy the feature branch to staging and check `/trust-voice/signaling/health`. It should report `status: ok`, `shared_state_enabled: true`, `shared_state_available: true`, `presence_store: redis_shared_v1`, `active_call_store: redis_shared_v1`, and `cross_instance_websocket_fanout: redis_pubsub_v1`.
5. Validate calls with two backend instances and synthetic accounts: register/presence, ring and answer across different instances, decline/end, SDP/ICE relay, recording consent, missed-call expiry, reconnect, and Redis interruption/recovery.

Keep production on one instance with shared state disabled until those checks pass. Enable the flag in production only after staging results are reviewed; if Redis becomes unavailable in shared mode, signaling fails closed and the health endpoint reports `degraded` rather than silently reverting to split local state.

Redis stores ephemeral presence leases and call coordination; Postgres remains the durable call-history and session store. WebRTC media remains on the existing transport path.
