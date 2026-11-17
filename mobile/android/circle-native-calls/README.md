# VeriFYD Circle production Android native calling — source handoff (NOT A RELEASE)

**Correct app:** `com.verifyd.circle` (separate from `com.verifyd.app` Detector).
**Branch:** staging only. No device build or signed AAB was produced.

## New native source
- `CircleIncomingCallMessagingService`: FCM HTTP v1 data-push receiver, original 30-second expiry and deterministic UUID validation.
- `CircleIncomingCallNotification`: CallStyle/ringtone and user-visible full-screen **ring** surface when Android permits. Heads-up when FSI is denied.
- `CircleAnswerActivity`: visible Answer/Decline controls; displays over lock screen but **does not unlock** the phone and **does not answer automatically**.
- `CircleAudioService` + `CircleWebRtc`: Android audio media leg using existing Trust Voice `/trust-voice/android-calls/{id}/media`; only created after explicit user Answer.
- `CircleCallApi`: Android Keystore P-256 proof for one-time, server-issued intent + separately protected scoped device control token; no user web bearer sent to Android.
- `CircleTwaLauncherActivity` + `CircleTwaBridge`: app-owned TWA CustomTabsSession, validated `use_as_origin` relationship, postMessage channel to the existing Horizons `trustVoiceNativeCalls.js`.
- `CircleNotificationsPermissionActivity`: Android 13+ notification prompt and Android 14+ optional full-screen intent settings.
- `AndroidManifest.merge.xml` and `generate_assetlinks.py`: manual source merge plus production-DAL generator.

## Integrate only into the recovered signed Android Circle project
Known Windows source location: `C:\Users\RDigiacomo2\VeriFYD\Android\VeriFYD_Circle_Android_API36_Starter_20261109`.
The current checked-out backend GitHub repository does NOT contain this signed Android Gradle project, therefore these source files **have not been merged into, built with, or installed from that project**.

1. Preserve the production `applicationId` `com.verifyd.circle`, branding, signing, versioning, and source backups. Never modify `com.verifyd.app`.
2. Copy Java files to the Android Circle app `src/main/java/com/verifyd/circle/calls`; merge permissions and components into its original manifest (replace the old launcher intent-filter; don't create a second launcher).
3. Keep minSdk >=24, compile/target API 36. Android dependencies include `androidx.browser` >=1.6, Android Browser Helper with TrustedWebActivityIntentBuilder, Firebase Messaging + Google services, WebRTC Java, and OkHttp (see existing isolated Detector pilot build.gradle **for dependency examples only**, not its package or upload keys).
4. Register `com.verifyd.circle` in the intended Firebase project and include its matching `google-services.json` in the signed Android build. Do not reuse a Detector app's Firebase config.
5. Obtain the real **Google Play App Signing** certificate SHA-256 for `com.verifyd.circle`; generate a DAL JSON using `generate_assetlinks.py --play-app-signing-sha256 ...`, and publish the resulting verified file at `https://vfvid.com/.well-known/assetlinks.json` via the origin owner. The current Horizon export has `[]`, so production postMessage verification cannot yet complete. **Never publish the pilot debug fingerprint**.
6. Backend `trust_voice_android_bridge.py` in draft PR #108 is **pilot-only** (`com.verifyd.circle.pilot.dev`), default OFF. Do not enable the production package until server-side signed-app proof/attestation, app-bound proof-of-possession, strict request validation and deployment wiring are completed. FCM preflight PASS demonstrates server credentials, not device registration.
7. Native enrollment requires a verified TWA PostMessage channel, one user opt-in permission, current logged-in persistent session, and one-time signed intent. **No manual pairing code, no Render per-user allowlist, no web bearer in Java or app links.** On logout, call `disableNativeCalls` before session revocation.
8. The actual `CircleTwaLauncherActivity` must retain the Chrome session while web `/voice` is active. Android TWA/Chrome behavior, FSI eligibility, microphone permission, locked-screen media and notification presentation were **not built or device-tested here**.

## Release flags (keep OFF before app source is installed)
`VERIFYD_ANDROID_NATIVE_BRIDGE_ENABLED=0`, `VERIFYD_ANDROID_CALLS_ALL_OPTED_IN_ENABLED=0`,
`VERIFYD_ANDROID_CALLS_AUTO_REGISTRATION_ENABLED=0`, and native Android call delivery disabled unless separately staged with real devices.

## Fallback and scope
The existing Horizons browser WebSocket/Web Push ringing remains authoritative when no per-call verified native presentation receipt exists. `nativeOwnsIncomingCalls()` stays false. This native source currently handles **one-to-one audio**; video still follows existing browser flow. The Android app and backend need FCM presentation acknowledgement, accurate call cancel handling, per-call arbitration, production Play Integrity, and a signed release before the all-user feature can be called complete.
