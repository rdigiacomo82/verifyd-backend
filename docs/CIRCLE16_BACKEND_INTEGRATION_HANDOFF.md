# Circle 16 — native calling backend integration (OFF by default)

Source: user Horizons export `2026-11-16T223503.438.zip`, vetted
against current backend `main` and recent Render production deployment.

## What this change includes
1. Android `/trust-voice/native-bridge/android/intents`,
   `/intents/redeem`, `/status/{installation_id}` endpoints.
   Guarded at ROUTER REGISTRATION by
   `VERIFYD_ANDROID_NATIVE_BRIDGE_ENABLED`, default `0`.
   The app gets only a single-use, 60-second challenge; the frontend
   sends its bearer directly to Render, not to Android.
2. `com.verifyd.circle` production redemption is separately guarded by
   `VERIFYD_ANDROID_NATIVE_BRIDGE_PRODUCTION_ENABLED`, default `0`;
   a real Play-recognized, app-signed installation and verified Play
   Integrity app/device/license verdict bound to the installation's
   P-256 proof and FCM token are mandatory.
3. iOS `/trust-voice/native-calls/{call_id}/media` WebSocket uses
   existing Trust Voice audio WebRTC SDP/ICE and native call state. It
   requires both `VERIFYD_NATIVE_CALLS_ENABLED` and
   `VERIFYD_IOS_NATIVE_AUDIO_ENABLED=1`. Existing iOS
   registration/APNs route remains unchanged, and iOS bundle ID
   `com.verifyd.circle` is not added to the production allowlist.
4. No removal or change to regular browser/WebSocket/Web Push voice
   signaling, voicemail, messages, groups, email, or detector endpoints.

## Strict, currently unresolved prerequisites
- Google Play Console Circle production `com.verifyd.circle` app signing
  SHA-256 (not upload/debug key), DAL file on `vfvid.com`, correct Cloud
  project linked to Circle's Play Integrity app, official Play API
  decoding permissions. No fingerprint is fabricated.
- Actual Android Circle signed Gradle target that includes PR #112's
  FCM receiver, permission UI, native ringing, CallStyle, Answer/Decline,
  Play Integrity helper, WebRTC audio and authenticated TWA host.
- Actual iPhone Circle signed Xcode target including private iOS PR #11's
  PushKit/CallKit, native authenticated account session, Keychain,
  APNs entitlement and WebRTC engine; correct server APNs .p8 path,
  team/key IDs, production vs sandbox and bundle topic.
- Real backend session/call state is PROCESS LOCAL in
  `trust_voice_signaling.py`; universal call routing across workers or
  instance restarts requires authoritative shared call state and
  cancellation/receipt handling.
- Historical Render logs Nov 10/12/13 include database pool exhaustion,
  transient database shutdown and a separate deadlock. This source
  does not resolve those pre-existing operational errors.
- No real-device tests are claimed. Sanity source review and ZIP integrity
  are not proof of locked-screen ringing; FCM preflight PASS only verifies
  credentials, not handset delivery.

## Deployment and user impact
This PR is code staging against CURRENT backend `main` only.
Do not enable feature flags, force deploy, publish or merge until the
signed native applications and app association are available.
Render's production service auto-deploys all changes merged into `main`,
including to its background workers and inbound-mail job; avoid an
unnecessary rollout for code that remains inaccessible while disabled.
