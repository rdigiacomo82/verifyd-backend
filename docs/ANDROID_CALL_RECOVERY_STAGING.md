# VeriFYD Circle — Android/locked-call reconnect staging (2026-11-07)

## Status: DRAFT / DO NOT DEPLOY
The current mobile Horizons export already requests a **Screen Wake Lock** while an answered call is visible. This is a client-only browser API and needs no Render configuration.

The installed web frontend still destroys its PeerConnection on *any* signaling WebSocket close:
`TrustVoiceBeta.jsx` > `socket.onclose`: cleanupWebRTC(), setActiveCall(null).
The production backend immediately ends active calls when the participant's last WebSocket unregisters. Neither a native app background microphone nor a call-resume handshake exists yet.

This experimental backend change does **not** fix screen-off microphone capture, closed-app incoming calls, or WebRTC restoration by itself.

## Candidate behind multiple fail-closed gates
- `VERIFYD_CALL_RECONNECT_GRACE_ENABLED` defaults OFF.
- `VERIFYD_CALL_RECONNECT_GRACE_PILOT_IDS` must list exact staging identity IDs; unset means OFF for all.
- `VERIFYD_CALL_RECONNECT_GRACE_SECONDS` defaults to 20 and is bounded 5–30 seconds.
- Only **answered** calls are considered; ringing calls continue existing 30-second/no-answer flow.
- When the last socket goes away, hold the in-memory answered call for the grace interval.
- If the same *authenticated* identity reconnects on a validated session, cancel the pending timeout.
- If it does not reconnect, mark the call ended and notify the peer exactly once.
- No change to user sessions, push notifications, call authorization, voicemail, messages, or LiveKit.
- No databases, migrations, additional service costs, or extra environment variables unless explicitly staged.
- Local in-memory timers do not survive Render process restarts; multi-instance deployments need shared authoritative call state.

## Prerequisites for any rollout
1. Verify the latest installed Horizons wake-lock implementation and call-overlay status on a real Android browser. Confirm **Screen stays awake** while listening and that mic audio reaches the peer.
2. Implement matching **frontend reconnect/resume semantics**; current `socket.onclose` destroys the media and would make server-only grace create a phantom active call.
3. Add server-authoritative resume validation and explicit terminal event reconciliation, handling revoked sessions, blocking, explicit hangups, device switching, multiple sockets and concurrent disconnects.
4. For *actual screen-off/locked background capture*, integrate a signed native Android calling stack using Telecom/ConnectionService and allowed microphone/phone-call foreground service with user-visible notification and correct runtime permissions. A WebView/browser wake lock cannot accomplish that.
5. Run isolated two-device calls, Android locked/unlocked, network drop, call ending during grace, reconnect under active media, and multi-instance/process-restart tests. Deploy only with feature OFF, enable separately on stage identities after review.
6. iOS requires separate native audio engine and PushKit/CallKit integration; the iOS starter repository is not the signed App Store build.

## Emergency scope
This experiment does NOT connect to 911 or guarantee delivery of emergency calls or alerts.
