# Native iOS call delivery: implementation and release blockers

This branch implements the backend half of incoming calls for the existing
`com.verifyd.videoauth` app (Apple team `3623QELM75`). It incorporates the isolated
payload contract from draft PR #64. It is not proof of APNs delivery or a working
native media session. The feature is OFF by default. The existing Circle starter
uses a different bundle ID and must not be substituted for the recovered app.

## Implemented

- Authenticated opt-in registration bound to a persistent, revocable Trust Voice
  session and an opaque installation UUID. Maximum eight active devices/account.
- Tokens encrypted at rest with Fernet; token hashes enable rotation/deduplication.
  Responses/logs do not expose device or provider tokens. Encryption-key rotation
  requires re-enrollment of existing devices; keep the previous key until a
  planned migration or explicitly require devices to register again.
- Token selection joins the live session table on both session and identity.
  Revoking a session immediately makes its registrations ineligible for dispatch.
  Device replacement and account switching update the owning session. Explicit
  opt-out deletes registrations; it is available even with delivery disabled.
- APNs HTTP/2, P-256/ES256 provider JWT, correct `.voip` topic, independent sandbox
  and production tokens/hosts. Provider JWT is reused for 45 minutes and refreshed
  on signing-file/key-ID/team changes. No retry can extend the original expiry.
- Actual invitation path checks native reachability after caller permissions,
  in addition to the existing live-socket/browser-push paths. A registered native
  recipient can be offline. Push acceptance is not delivery confirmation.
- Invalid/unregistered APNs tokens are disabled with a version check so a late
  failure cannot disable a newly rotated token. Provider credential failures do
  not disable devices. HTTP exception messages containing token URLs are not logged.
- Pending call replay on native-pilot WebSocket reconnect, preserving the same ID
  and deadline. A ringing native recipient's socket loss doesn't cancel the invite;
  caller disconnect still cancels it.
- Authenticated status/answer/decline/end HTTP API. HTTP and WebSocket controls use
  one atomic state transition; answer rechecks current privacy and blocks. An
  expired answer fails even if the expiry task hasn't executed. Racing answers
  have one winner. Caller cancellation during delivery cannot produce a later
  stale `call_ringing` event. History is created before delivery/fast answers.
- Expiry measures the ORIGINAL creation time, not completion of push/DB work.

## API

All routes use `Authorization: Bearer <Trust Voice session token>`. Native routes
require a persistent session, enabled signaling, enabled native calls, and an
explicitly allowed pilot handle. Opt-out needs authentication but not pilot flags.

`POST /trust-voice/native-call-devices`:

```json
{
  "installation_id": "00000000-0000-4000-8000-000000000001",
  "token": "<lowercase hex PushKit token, not ordinary APNs token>",
  "environment": "sandbox",
  "bundle_id": "com.verifyd.videoauth",
  "opted_in": true
}
```

`DELETE /trust-voice/native-call-devices/{installation_id}` removes only the
current identity/session's registration. Call before sign-out, token invalidation,
account switch or opt-out; server-side session revocation also suppresses delivery.

`GET /trust-voice/native-calls/{call_id}` returns authoritative state, call type,
original integer `expires_at` and whether this participant is the callee. A missing
or other user's call returns 404, an expired ringing call 410, a block 403.

`POST /trust-voice/native-calls/{call_id}/answer` (or `/decline`, `/end`) returns
`ok`, event `type`, and `call_id`. An answer ACK means ONLY the server accepted
that state transition. It does not confirm media. Native code must also prepare
its media transport before fulfilling CallKit's answer action. Invalid/competing
state is 409; callers cannot answer/decline their own outgoing invitations.

The existing `/trust-voice/ws?token=...` remains necessary for authenticated
cancellation and SDP/ICE relay. Native HTTP control is not an audio transport.
For one-to-one calls the caller generates the offer after `call_answered`.
Do not open two independent callee media handlers for the same call.

## Deployment settings (no credentials in repository/chat)

Use the service's secret storage, not GitHub or client code:

- `VERIFYD_NATIVE_CALLS_ENABLED=0` initially.
- `VERIFYD_NATIVE_CALLS_PILOT_HANDLES=<explicit comma-separated test handles>`.
- `VERIFYD_NATIVE_CALL_TOKEN_ENCRYPTION_KEY`: a securely generated Fernet key.
- `VERIFYD_APNS_TEAM_ID=3623QELM75`.
- `VERIFYD_APNS_KEY_ID`: the Apple APNs signing key ID.
- `VERIFYD_APNS_PRIVATE_KEY_FILE`: absolute path of a mounted APNs `.p8` secret.
- Existing `VERIFYD_TRUST_VOICE_SIGNALING_ENABLED` and WebRTC/session settings.

Secret-file path must exist in the running service, HTTP/2 must be available,
Apple key must authorize the app's VoIP topic, and the chosen environment must
match the signed app's `aps-environment`. Use sandbox for ordinary development
signing; TestFlight uses production. Generating a new key is not required if a
suitable existing key is available. Do not change store identity/signing keys.

Schema initializes lazily on first enrollment/lookup using additive CREATE TABLE
statements; verify permissions and PostgreSQL behavior on staging first. There
is no production data migration or live setting change in this PR.

## iOS/frontend work still required

The uploaded Xcode app launches `https://vfvid.com/app`. The supplied Horizons
export contains the calling screen at `apps/web/src/pages/TrustVoiceBeta.jsx`
(`/voice`), not in that video-verification screen. Its session keys are
`verifyd_trust_voice_session_token` and `verifyd_trust_voice_session_token_persist`.
The green video-verification email badge is not a Trust Voice session.

The export uses `RTCPeerConnection` with browser media and STUN only. It does not
contain a native WebRTC SDK or TURN provisioning for restricted networks.

Before pilot activation:

1. Implement a native media adapter (including microphone permission, SDP/ICE
   buffering, audio-session ownership and interruption handling). Browser WebRTC
   in a suspended WKWebView is not sufficient for locked-screen native calling.
   Decide the scope of video UI; do not silently answer video as camera-on.
2. Implement `VeriFYDNativeCallTransport` in the recovered Xcode app, using the
   endpoints above and authenticated WebSocket. Preserve bundle/team identity.
3. Add an origin-restricted Capacitor bridge for explicit enrollment from the
   signed-in `/voice` session. Validate session server-side, persist credentials
   in appropriate Keychain accessibility for locked-device cold launch, revoke
   on logout/account change, and keep registration current after token rotation.
   Never scrape a token from the unrelated `/app` email-verification UI.
4. Retain the PushKit/CallKit bridge on cold launch when enrolled. Report VoIP
   pushes promptly. An unauthenticated/revoked/expired invitation must fail closed.
   Confirm server answer plus media readiness before fulfilling CallKit actions.
   Forward server cancellation/expiration to the same active CallKit UUID.
5. Prevent native and hosted `/voice` from independently answering/creating media
   for the same call. Wire the frontend to display native call state when native
   owns a call; keep ordinary browser behavior intact. Accessibility auto-answer
   must follow the same server and CallKit state machine.
6. Configure push, VoIP, background audio and microphone capabilities, then build
   on macOS. This cloud environment has no Swift/Xcode/iOS SDK and cannot verify
   a signed native build. The existing simulator test is only a mock transport.
7. Signed physical-device test via TestFlight (wireless) or Xcode; verify locked
   screen delivery, foreground/background/cold launch, decline/end, duplicate and
   expired pushes, caller cancellation, revocation, two-device answer races,
   permissions, Focus, network switching and two-way audio. Run browser regression
   tests for messages, ordinary calls, accessibility and voicemail.

## Limits before general release

Call/presence state is still the existing process-local beta state. Restarts lose
invitations; multiple workers/instances cannot share/atomically arbitrate calls.
Use a single signaling process for an isolated pilot. Shared durable call state,
terminal-event recovery, per-device media arbitration and a real TURN strategy
are required before claiming production reliability. This PR does not claim to
solve those larger existing architectural limitations.

## Validation

Install `requirements-native-calls.txt`; run the commands in
`.github/workflows/native-call-delivery.yml`. Tests execute real FastAPI routes,
signed session validation, call transitions and WebSocket invitation handling,
plus ES256 verification, mocked HTTP APNs, and browser/session/security regressions.
Database operations are mocked in local tests; PostgreSQL staging validation,
live APNs, native compilation and real-device media tests remain outstanding.

Rollback delivery by setting `VERIFYD_NATIVE_CALLS_ENABLED=0`. Leave encrypted
device data in place until a retention/migration plan; never drop existing tables.

## Automatic readiness check

On startup, the native-call router emits one `VERIFYD_NATIVE_CALL_READINESS`
JSON log through the server logger. It reports only five booleans: required
settings, local APNs key readability/signing, device-token encryption, pilot
allowlist presence and the delivery gate. It never logs environment values,
file paths, JWTs, private keys or exception details. Missing/invalid settings
produce false and do not prevent startup. This check sends no APNs request,
performs no database access, and does not change delivery flags. Local signing
success does not prove that Apple accepts the key for the app's VoIP topic or
that any device receives a call. It is intended for log-based diagnosis when
connected tooling cannot execute a Render Shell command.
