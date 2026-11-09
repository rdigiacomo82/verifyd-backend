# Native call contract — proposed, inactive (2026-11-07)

Companion: private verifyd-circle-ios `native-integration/`. This document changes no routes,
schema, notification settings or production feature flags. PR #62 remains the Web Push pilot.

## Current gap

The current ringing map is process-local and Answer/Decline is handled over the existing
WebSocket. Do not point native notification actions at invented production REST routes.
Native adapters deliberately require authenticated host implementations before activation.
No Android package or signed iOS target is established by the connected source.

## Required backend implementation before native pilot

Use a distinct default-OFF native incoming-call flag AND exact account allowlist. Native
token registration and sending must also require opt-in. Keep all existing paths unchanged
when the gates are off. Do not reuse Web Push opt-in as consent for native registration.

Device registration API (route names to be implemented and reviewed):
- Authenticate using current revocable trusted-device session, never a Firebase client key.
- Derive identity/session from authentication, ignore client-supplied owner IDs.
- Bind installation, platform, app identifier, APNs environment and token to that session.
- Allow only confirmed signed application IDs and the selected Firebase project.
- Encrypt tokens at rest using a managed secret, use a keyed digest for uniqueness,
  never return raw tokens or log tokens/credentials; bound request sizes and rate limit.
- Reject replay or registration by revoked sessions; enforce a registration generation
  so racing logout and token refresh cannot restore a removed binding.
- Delete/revoke on logout, session revocation, account deletion, opt-out, FCM UNREGISTERED
  and APNs Unregistered; only remove invalid tokens for the matching token generation.
- Use least-privilege workload/service credentials in secret storage, never in GitHub.

Call actions:
- Persist invitations in PostgreSQL with random call ID + UUID, participants, type,
  authoritative expires_at (existing 30s), state/version and accepted device/session.
- Reuse the existing permission/block decisions before creation AND before acceptance.
- Recipient-only accept/decline; session must be current, nonrevoked and device-bound.
- Atomic database compare-and-set: ringing AND unexpired -> answered or declined.
  One winner across multiple workers/devices; retries by the same winning device are
  idempotent. Other devices get an explicit terminal/conflict response.
- Call end, timeout and answer on another device publish cancellation to all other
  devices, and preserve current call history/voicemail events exactly once.
- Reconcile socket signaling with persisted state; do not maintain two conflicting
  authorities. Add an outbox with bounded retries that never send expired invitations.
- Token registration is not proof that a device is reachable. Push receipt is not
  answer acknowledgement; only confirmed state transitions may report answered.

Wire payload for incoming invitation:
```
{"type":"call_incoming","call_id":"opaque_random_id","call_uuid":"UUID",
 "call_type":"audio or video","expires_at_ms":1794024000000}
```
No caller name, message, location, bearer token or media secrets. Treat all payloads as hints.
Clock skew must fail safely; re-query authoritative state before action.

Android: FCM HTTP v1 high-priority **data** message for a real user-visible incoming call,
TTL capped to remaining ring time, unique collapse behavior per call. Notification payload
messages can bypass onMessageReceived in background and must not be used for this adapter.
Terminal hints use `type=call_terminal`, `call_id` and the matching expiry/version. Persist
terminal tombstones so a delayed invitation cannot revive an ended call.

iOS: genuine incoming invitation only via APNs VoIP with correct `.voip` topic,
`apns-push-type=voip`, appropriate priority and expiration no later than ring deadline.
Persist stable call UUID. Never use PushKit for chat, voicemail, location or cancellation-only
wakeups. Ordinary APNs / authenticated signaling reconciles terminal state. CallKit reports
promptly before any network operation, then ends stale/invalid calls.

## Validation gate

Add integration tests against PostgreSQL for simultaneous workers/devices, expiry boundary,
replay, blocking, authorization, cross-owner token writes, invalidation generations,
logout/refresh races, sender retries, process restart and exactly-once history/voicemail.
Run existing messaging, authentication, gaze, accessibility, location, emergency and calling
regressions. Validate provider credentials and delivery in staging; then perform the physical
matrix in the companion README. No production settings or builds may change from this draft.
