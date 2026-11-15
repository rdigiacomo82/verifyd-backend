# VeriFYD Circle: Locked-screen and closed-app incoming-call failure

**Incident report and corrective implementation specification**  
**Date:** 2026-11-15  
**Scope:** Published Android app `com.verifyd.circle` (reported by owner; public Play detail not independently retrievable), test phone Samsung Galaxy S23 SM-S911U; design must cover supported Android and iOS devices.  
**Status:** Engineering diagnosis; native-client changes and two-device acceptance tests not yet delivered. **Do not describe this as fixed.**

## Observed reports and verified backend evidence

**Reported by user (not an automated measurement):**
- Notification permission has been enabled on Galaxy S23.
- With the app visible and unlocked, calls may work.
- With the app behind a locked screen, occasionally one ring is heard, then silence or voicemail.
- With the app closed/background process unavailable, no wake or ring, often straight to voicemail.
- App was subsequently published on Play; publication did not change those symptoms.

**Verified in current GitHub backend:**
- `trust_voice_signaling.py` creates a 30-second, server-authoritative call invitation and uses existing WebSocket signaling when an online socket exists.
- `trust_voice_android_calls.py` explicitly labels FCM native delivery a *pilot*, off by default unless `VERIFYD_ANDROID_CALLS_ENABLED` and per-handle allowlist are configured, Firebase credentials are valid and a paired device with a supported package/token exists.
- Android FCM sends HIGH-priority **data-only** messages with original 30-second TTL; Firebase acceptance is not proof of device receipt or ringing.
- `GET /trust-voice/android-calls/{call_id}` checks an existing call with paired-device authentication.
- `POST /trust-voice/android-calls/{call_id}/answer` currently rejects with `409 native_media_not_ready`; a **native audio** callee must instead use the WebSocket `/trust-voice/android-calls/{call_id}/media`, enabled by `VERIFYD_ANDROID_NATIVE_AUDIO_ENABLED=1`, then send `{"type":"call_answer"}`. This path requires a live WebRTC-enabled caller and supports audio only.
- `trust_voice_native_calls.py` implements a separate **iOS APNs VoIP pilot**, also gated by flags, credentials and bundle IDs. The default allowed bundle ID in backend code is `com.verifyd.videoauth`, not the separate Circle app; Circle must be explicitly configured if this default is still in effect.
- The checked-in iOS starter `rdigiacomo82/verifyd-circle-ios/verifyd-circle-starter` documents native push/notifications as **not implemented in the starter**. This is not confirmation of the currently installed App Store binary, which may differ.
- Older Render application logs from **2026-11-10** contain many `Android FCM refused incoming invite status=404` messages for `projects/verifyd-detector-pilot/messages:send`; HTTP 404 by itself does not disambiguate invalid project, token, package or API setup. Some older Android pairing/redeem requests returned 503 and 401.
- Checked Render log window 2026-11-15T00:00Z onward: no matching Android native-call request routes and no matching filtered FCM warnings; that **cannot establish** whether the user attempted a call during that interval or which push path was selected.

**External verification limitations:**
- The exact Google Play listing URL could not be read from public web results, so current AAB's manifest, versionCode, Firebase configuration, notification channels and launch activities remain unverified.
- A production Android source tree/APK/AAB was not found in the connected GitHub repositories or latest Horizons export. Horizons contains web pairing UI, not the installed native `FirebaseMessagingService`.
- Do not claim that enabling notification permissions, publishing to Play, changing WebSocket timeouts, or polling the web app alone fixes the defect.

## Technical failure tree (check sequentially)

1. **Identity/session routing** — Is the intended recipient eligible and permitted to receive call? Verify trust/block permissions and `call_permission_decision`, auth/session still valid.
2. **Native device readiness** — Is the installed `com.verifyd.circle` registered through `/android-call-pairings/redeem` and is its FCM token current, bound to the user's persistent session and exactly the configured package?
3. **Firebase provider acceptance** — Does Render send to the correct Firebase project using the correct service account, HTTP v1 endpoint, and an FCM token minted for that project? Record provider HTTP status and redacted error category per call.
4. **Phone receipt** — Does `FirebaseMessagingService.onMessageReceived` execute when the device is locked, dozing, and app not in foreground? Log the call ID and timestamps locally without token or personally identifying information.
5. **Immediate native call presentation** — Does the service immediately display a dedicated high-importance call notification / Android Telecom-compatible incoming-call surface without blocking on HTTP? Is there a foreground-managed call object, ringtone and vibration until answer/decline/timeout?
6. **Android permissions & policies** — Prompt users at first sign-in for notifications, check channel importance/ringtone settings, Android 14+ full-screen intent access, lock-screen visibility, DND, and OEM battery restrictions. Respect refusal; offer a one-tap settings recovery route.
7. **Answer-to-media** — After answer, can native client authenticate and use the supported audio WebSocket call-answer pathway? Do not send HTTP `answer` to a route that rejects it. Confirm caller receives `call_answered`, can exchange SDP/ICE, and call state remains active until hangup.
8. **Hangup / voicemail** — Confirm native UI dismissal occurs only on actual decline, cancel, answer on another device, or the original 30-second expiry. A single notification sound must not be misrepresented as an active 30-second ringtone.

## Android native implementation requirements

**Use one dependable native path, independent of the Horizons WebView lifecycle.**
- Firebase Android app registration must match `com.verifyd.circle`, production signed application and selected Firebase project. The included `google-services.json` and registered backend project/service-account must agree.
- Register `FirebaseMessagingService` in Android manifest and implement `onMessageReceived` for the precise `verifyd.native_call.v1` data payload.
- Validate schema, kind, UUID, expiry, and duplicate/replay call IDs with `native_call_contract.py` semantics. Display incoming-call UI **immediately** and fetch server status asynchronously; never delay presentation for a long network request.
- On Android 13+, request `POST_NOTIFICATIONS` after a clear in-app explanation and allow the user to decline; show the feature's consequence when denied.
- Create an *Incoming calls* channel with high importance, audible ringtone, vibration, visible-on-lockscreen call presentation, and an action to manage notification settings. Channels cannot be silently reconfigured once user settings take control.
- On Android 14+, check `NotificationManager.canUseFullScreenIntent()` and, for eligible calling apps, link to `Settings.ACTION_MANAGE_APP_USE_FULL_SCREEN_INTENT` when permission is missing. Keep an appropriate heads-up/CallStyle fallback.
- Prefer Android Telecom / Jetpack Core-Telecom or a properly implemented self-managed `ConnectionService` for call state, answer/decline, Bluetooth/watch controls and system coexistence. Telecom integration **does not by itself play a ringtone**; implement a persistent call notification and audio behavior correctly.
- For accepted active calls, maintain approved foreground-service behavior, declared phone-call service type where applicable, microphone/audio focus and background WebRTC according to current Android policy.
- Native receiver must manage the incoming call independent of the WebView; sync final answer, decline, missed, cancel and accept events back to the existing Trust Voice server. Deduplicate Web and native rings by call ID.
- Never store registration secrets, full FCM tokens, call contents or notification payloads in diagnostic logs.
- Distinguish user *swiping away* versus *force-stopping* the app in the test results. Android force-stop is a platform condition that may prevent FCM-delivered wakeups until the app is reopened. Do not promise impossible wake behavior.

## iOS native implementation requirements

- Use PushKit VoIP tokens and APNs `voip` pushes with the exact **Circle** production bundle identifier and `.voip` topic.
- Respond promptly with CallKit `CXProvider.reportNewIncomingCall`, even while backgrounded; do not defer until the hosted WebView is visible.
- Properly keep an iOS native `CXProvider` delegate and audio session; on answer, authenticate to the existing server call and establish WebRTC before fulfilling the answer action.
- Persist and renew native VoIP token registration after sign-in, rotation, restore and logout; unlink tokens when revoking a device.
- Maintain existing calling permissions, blocking and one-time 30s deadline.
- Apple may stop delivering VoIP pushes when the application repeatedly fails to report them to CallKit.
- iOS notification authorization, Focus/DND and CallKit UI are separate from generic web push notifications.
- Test locked, background, relaunched, network transitions, and device reboot with a real iPhone, not only Simulator.

## User-facing first-run checklist (native apps)

**After sign-in**, explain why notifications are needed for calls. Request OS permission by explicit user action and display the actual result. Then show independent call readiness checks:

| Check | Android | iOS |
|---|---|---|
| Allow notifications | Runtime request on Android 13+ | Native notification authorization where appropriate |
| Incoming call channel enabled | Check channel and sound/vibration | CallKit configuration and device sound settings |
| Full-screen incoming calls allowed | Android 14+ eligibility and app setting | System CallKit UI |
| Native push linked | Registered FCM token + current paired session | VoIP token registered to correct bundle/env |
| Background battery/access controls | Check OEM restrictions and advise user, do not silently modify | Follow iOS PushKit constraints |
| Test incoming call | Interactive test from trusted second device | Interactive test from trusted second device |

Never assume that the notification prompt is displayed automatically because the app was downloaded; prompt from the installed native app on first sign-in.

## Reproducible acceptance matrix

| State | Galaxy S23 | Other supported Android | iPhone |
|---|---|---|---|
| Foreground, screen on | Caller connects with native/web UI without duplicate sound | Same | Same |
| App in background, screen on | Full incoming-call alert rings until answer/timeout | Same | CallKit |
| Screen locked, app background | Device alerts, screen surfaces incoming-call UI according to OS permissions | Same | CallKit |
| App process evicted but not force-stopped | FCM wakes native receiver and creates call UI | Same | VoIP PushKit wakes native client |
| User force-stopped app | Document platform limitation and recovery | Same | Document iOS termination behavior and constraints |
| Silent mode / DND | Honor OS policies and user settings | Same | Honor Focus |
| Wi-Fi to cellular transition | No duplicate calls, correct final state | Same | Same |
| Incorrect/expired token | Explicit diagnostics; re-register token at next allowed opportunity | Same | Same |
| Answer/decline | Server call state, WebRTC media, history, missed/voicemail correct | Same | Same |

**Release blocker:** Require at least 20 successful real-device locked/background incoming call attempts across multiple Android versions and two or more iPhone models, with recorded outcome categories and no unexplained early voicemail. Never infer phone wake from an FCM/APNs 200 response. Record `server_invite`, `provider_accepted`, `device_received`, `ui_presented`, `user_answered`, and `media_connected` independently.

## Root-cause-oriented next engineering actions

1. Obtain the **exact Play production signed Android source/AAB** installed on the S23 (including manifest, Gradle, Firebase config minus secrets, `FirebaseMessagingService`, notification setup, launch activity and any Telecom integration). Verify package `com.verifyd.circle` and installed versionCode. Do not change the existing Detector app.
2. Validate Firebase project matching, HTTP v1 credentials, token pairing and token rotation using redacted status only. Resolve the historical 404 category, not merely retry it.
3. Native Android receiver must ring while the web process is suspended/closed and use the correct answer/media endpoint.
4. Implement onboarding permissions and in-app **Incoming Call Readiness** report, not just a generic notifications toggle.
5. Implement PushKit/CallKit for iOS Circle, review actual shipping Xcode project and entitlements, then test on real devices.
6. Run the full matrix before deploying changes broadly; keep all existing calls and app identities intact.

### Current status

**Diagnosed, not yet fixed.** Registry Phase 2 is a separate backend change. No native client binaries have been modified or published by this report.

### Primary official references

- FCM high priority & short processing window: https://firebase.google.com/docs/cloud-messaging/android-message-priority
- FCM Android receiving behavior: https://firebase.google.com/docs/cloud-messaging/android/receive-messages
- Android 13+ notification permissions: https://developer.android.com/develop/ui/compose/notifications/notification-permission
- Android 14+ full-screen call notifications: https://developer.android.com/about/versions/14/behavior-changes-14
- Android Telecom / Core-Telecom: https://developer.android.com/develop/connectivity/telecom/voip-app/telecom
- Apple PushKit and CallKit: https://developer.apple.com/documentation/pushkit/responding-to-voip-notifications-from-pushkit
