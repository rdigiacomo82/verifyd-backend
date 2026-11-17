# Circle 16 — account-bound production Android enrollment (DRAFT, OFF)
Depends on draft PR #108 (pilot bridge); Android app source in draft PR #112.

This branch implements an **independent production application attestation gate** for
`com.verifyd.circle`, while preserving original pilot `com.verifyd.circle.pilot.dev`.
The web bearer travels only browser → Render; never to the Android device.
The Android host authenticates each enrollment with its own Keystore P-256 key
and a single-use, session-bound, short-lived intent (60 seconds). The original
postMessage channel requires the Circle production `vfvid.com` DAL association.

## Production attestation — fail closed
On production redemption, `trust_voice_android_integrity.py` requires:
- A **new** Google Play Integrity standard verdict from Google's decode API.
- `requestHash` bound to the canonical P-256 intent signature message AND the
  SHA256 of the FCM token, so neither credential can be swapped.
- Matching `requestPackageName`, `appIntegrity.packageName`, Play-recognized
  certificate SHA-256 fingerprint, token freshness, device integrity and
  Play license. No unrecognized/debug/sideloaded app may register.
- The existing verified Trust Voice persistent session, active identity,
  original one-time intent, and an unrevoked server-owned device entry.
- The existing max-eight-device rule, token encryption and 30-second call TTL.

The Google Play integrity Cloud project must actually be linked to
`com.verifyd.circle`, not merely the Detector pilot Firebase project.

## Required Render configuration (NOT set by this PR)
- `VERIFYD_ANDROID_NATIVE_BRIDGE_ENABLED=0` unless the full bridge has been merged.
- `VERIFYD_ANDROID_NATIVE_BRIDGE_PRODUCTION_ENABLED=0`.
- `VERIFYD_ANDROID_CALLS_ENABLED=0` until native signed app and media are ready.
- `VERIFYD_ANDROID_CALLS_ALL_OPTED_IN_ENABLED=0` until rollout.
- `VERIFYD_ANDROID_CALLS_ALLOWED_PACKAGES` must include **only** approved signed
  Circle packages when production enrollment is authorized.
- `VERIFYD_ANDROID_CIRCLE_PLAY_SIGNING_SHA256`: colon-separated SHA256 of the
  **App Signing** certificate from the actual Google Play Console Circle app.
  It must NOT be the upload key or debug key.
- `VERIFYD_ANDROID_CIRCLE_PLAY_INTEGRITY_PROJECT_NUMBER`: exact numeric Cloud
  project linked to the Circle Play app.
- Server service account must be granted Play Integrity decode rights for
  the linked project, **separately from** the existing FCM sender permission.
- Existing persistent Fernet key, Firebase service account file and project ID;
  do not rotate or paste secrets into GitHub.

Use a valid Digital Asset Links file at
`https://vfvid.com/.well-known/assetlinks.json` with the real Play App
Signing fingerprint and `delegate_permission/common.use_as_origin` relation.
Current exported file is `[]`, so verified production TWA registration
cannot work from the existing deployed site.

## Frontend replacement contract
Web request `POST /trust-voice/native-bridge/android/intents` supplies only
session-bound public P-256 key, package `com.verifyd.circle` and opt-in;
returns a one-time intent/challenge. Browser sends only intent/challenge through
origin-verified TWA MessagePort. Android generates proof signature, gets FCM
token and obtains Play Integrity attestation with:
`base64url(SHA256(canonical_signature_message + b'\\n' +
SHA256(fcm_token).hexdigest().encode('ascii')))`.
It redeems with `POST .../intents/redeem`; server returns scoped device token
to Android Keystore only. Web verifies owner with
`GET /status/{installation_id}?package_name=com.verifyd.circle`.

User permission and native-call opt-in are independent. An existing opted-in
identity may rebind automatically after session renewal. Account switching
does not transfer an installation across accounts. Sign-out requests unlink
**before** session revocation. Server session revocation disables delivery
even if the device is offline.

## Unresolved production dependencies
Draft branch has **not** been compiled or deployed, nor proven with real
PostgreSQL concurrency, Google's live Decode API, signed Android devices,
WebRTC audio, CallStyle locked-screen display, cancellation or Android OS
permissions. It is source staging, NOT universal calling activation.

iPhone remains separate in private Circle iOS PR #11: PushKit and CallKit
source is staged; signed Xcode target, native session source, APNs signing and
real WebRTC media remain outstanding. Web push fallback stays active.

## iOS native audio handoff staged on this same branch
`trust_voice_native_calls.py` now contains a **separately default-OFF**
`wss://verifyd-backend.onrender.com/trust-voice/native-calls/{call_id}/media`
endpoint, gated by `VERIFYD_IOS_NATIVE_AUDIO_ENABLED=0` and ordinary native
calling enablement. It accepts the signed Circle app's existing authenticated
Trust Voice bearer via an HTTPS-upgraded WebSocket header and verifies its
current server-side installation ID, identity ID, persistent session and
allowed `com.verifyd.circle` bundle. It reuses the single caller/callee
authoritative signaling + native media arbitration from the existing Android
media leg and only relays SDP/ICE.

The actual iOS source counterpart `CircleWebRTCAudioEngine.swift`,
`CircleNativeSessionVault.swift`, CallKit adapter and Keychain APNs enrollment
are staged in private `verifyd-circle-ios` PR #11. The real signed Xcode target
does not exist in the connected checkout, so the engine is NOT compiled or
installed; real microphone/CallKit activation and two-way audio still require
physical-device verification. No APNs/Render native enablement flags were set.
