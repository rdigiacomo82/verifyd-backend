# Detector Android audio-call pilot

This overlay adds a native incoming audio-call candidate to the November 8 audited Detector Android project. It preserves the main `com.verifyd.app` identity, API 36 target, API 21 baseline minimum, branding and signing configuration. The overlay builds the pilot as `com.verifyd.app.pilot.dev`. Its release variant is disabled.

The pilot receives an audio invite through Firebase Cloud Messaging, posts Android's incoming call notification, and opens a visible activity when the user taps Answer. The activity requests the microphone if needed and starts a microphone foreground service. The service prepares a native WebRTC peer, authenticates to the call-scoped backend socket and then accepts the call. It handles browser SDP offers, ICE candidates, hangup and cleanup. No microphone opens on mere push receipt.

The original browser caller uses the existing Trust Voice signaling. This candidate covers an incoming audio call on Android and audio continuity when that Android callee locks the screen. Outgoing native Android calls, video, production Telecom integration and Play publication are outside this candidate.

## Build the debug APK

1. Use a copy of `VeriFYD_Android_Production_v3_Audited_API36_20261108`. Overlay this directory's `app/` onto it. If you previously installed the microphone diagnostic, retain its files and merge its launcher and service entries with `app/src/callpilotDebug/AndroidManifest.xml` if you want both test tools.
2. With Android SDK 36, JDK 21 and Gradle 8.13 available, run `:app:assembleCallpilotDebug` to compile the candidate. On Windows, from the project root and the shell where `GRADLE` is already defined:

   ```bat
   call "%GRADLE%" :app:assembleCallpilotDebug --no-daemon
   ```

3. Before testing live push, add the exact Android app ID `com.verifyd.app.pilot.dev` to the test Firebase project, place its local `google-services.json` in `app/src/callpilotDebug/`, and set `detector_pilot_api_origin` to the HTTPS origin serving this branch. Build again with `-PenableGoogleServices=true`. The Firebase configuration and API origin are intentionally absent from this overlay.

## Server and phone verification

The backend bridge uses the existing Trust Voice call authority and is off by default. Live testing needs a reachable backend running this branch, the existing Trust Voice signaling and WebRTC flags, Firebase configuration and test handles. Scope the device package allowlist to `com.verifyd.app.pilot.dev`; set `VERIFYD_ANDROID_CALLS_ENABLED=1`, `VERIFYD_ANDROID_DETECTOR_CALLS_ENABLED=1` and `VERIFYD_ANDROID_NATIVE_AUDIO_ENABLED=1` in the authorized test configuration. Use the existing pairing-code flow against that backend. The native credential is encrypted on the phone, scoped to Android calls and bound to the account session.

1. With the browser caller signed into the same backend and the Android callee locked, place an audio call. Verify notification, ring, Answer and two-way audio. Verify Decline and expiry separately.
2. During a connected incoming call, lock the Android callee again. Confirm audible speech from both sides for at least 90 seconds with no screen interaction. Repeat over Wi-Fi and cellular. The pilot uses the existing STUN setup; test TURN requirements for restricted networks before shipping.
3. Verify remote hangup and native hangup clear the microphone service. Test microphone denied, expired push, simultaneous answers, notification restrictions and revoked device enrollment.
4. Validate native library alignment for 16 KB devices, add production call lifecycle and Play declarations, and produce a new signed AAB only after device verification. Keep debug APKs out of Play Console.

No keystore, signing secret, Firebase key, account credential, or app bundle is included.
