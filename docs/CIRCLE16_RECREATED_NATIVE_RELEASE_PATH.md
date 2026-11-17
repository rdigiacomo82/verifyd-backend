# Circle 16 — recovering mobile releases without original signed project files

This is a **source-level release recovery path**. It does not certify the app
as ready for users, enable native calling, or contain any signing credentials.

## Android: original Gradle project is not a cryptographic prerequisite

A *new* Android project can ship an update to the SAME Google Play listing
if it preserves the app's Android application ID, versionCode progression,
required user data/permissions and is signed with the Play Console's approved
**upload** key. Where Play App Signing is enabled, Google retains and uses the
existing **app signing key** to distribute installation updates. A lost upload
key can be reset through the Play Console while preserving installed users.
Official: https://developer.android.com/studio/publish/app-signing

The native project is assembled under
`mobile/android/circle-native-calls/` in backend PR #112; package
`com.verifyd.circle`. Do **not** confuse it with the existing
`com.verifyd.app` Detector package. If Circle is not yet registered
on Google Play, this is a new app setup, not an update to the Detector.
The Java/Gradle source, app signing identity, Play Integrity linked Cloud
project, Play app-signing fingerprint and Firebase registration must all agree.

**No Google Play data is read by this repo**; do not guess existing package
registration, upload signing key or fingerprint. Obtain it from the actual
Circle Play Console record once connected.

Re-created source still needs equivalent app features, subscription billing,
manifest, deep links, app icons and data migration policy to preserve users.

## Apple iOS: original .xcodeproj is not always required

Capacitor `npx cap add ios` can generate a new Xcode App target. Existing
App Store Connect app identity can be preserved only if the target uses the
**same registered bundle identifier and Apple Developer team**, appropriate
entitlements/capabilities, and legal build/version progression; Xcode automatic
signing can manage profiles for the existing account. The original .xcodeproj
files are not themselves a cryptographic prerequisite. A new project should
NOT be distributed until persistent data/Keychain/App Groups and native module
compatibility with the installed Circle app have been established.

Official:
- https://developer.apple.com/documentation/xcode/preparing-your-app-for-distribution
- https://developer.apple.com/help/account/provisioning-profiles/create-an-app-store-provisioning-profile/
- https://developer.apple.com/help/app-store-connect/manage-builds/upload-builds/

The private repository `verifyd-circle-ios` PR #11 includes an automatic
native-call integration script for the regenerated Capacitor Xcode project,
the Keychain/Capacitor enrollment bridge, PushKit/CallKit and WebRTC sources.

## Reproducible cloud source builds WITHOUT private signing

- Android: `.github/workflows/circle16-android-source-check.yml`
  on backend PR #112 checks the Gradle source, `com.verifyd.circle`,
  Android 36 and native Java. Only the explicitly opted-in
  `-PcircleUnsignedSourceCheck=true` CI path bypasses a *real* Firebase JSON.
  No usable FCM config, signed AAB or Play upload is produced.
- iOS: `.github/workflows/circle16-ios-source-check.yml`
  on private iOS PR #11 creates a fresh Capacitor Xcode project and attempts
  an unsigned iOS Simulator compile on a macOS runner. This is
  manual-only to avoid unrequested private macOS GitHub Actions costs.
- Both are **staged** and have not been executed. They cannot prove
  real-device FCM/APNs, lock-screen UI, two-way call audio or entitlement.

## Last deployment gates

1. Check original Google Play/Apple developer identifiers and maintain their
   signing authority; determine whether Circle Android is a new Play listing.
2. Securely provision the **Circle-specific** Firebase Android app, linked Play
   Integrity project and official signing fingerprint. Publish the real
   `vfvid.com/.well-known/assetlinks.json` only from the verified signer.
3. Securely provision the iOS distribution team, Push Notifications and VoIP
   background capabilities, correct APNs .p8 key/topic/environment and native
   Keychain owner binding. Do not commit .p8, private keys or passwords.
4. Run unsigned cloud CI; fix compiler and integration errors before signing.
5. On actual installed signed devices, verify PushKit/FCM wake, locked
   screen presentation, user Answer/Decline, microphone and two-way native
   audio, cancellation and session switching; Android may need FSI settings
   and iOS obeys Focus and lock screen policy.
6. Replace in-memory `trust_voice_signaling.py` call state with durable shared
   call arbitration before routing all accounts through native delivery.
7. Only then review draft backend PR #114 and explicitly enable supported
   production options; keep web WebSocket/Web Push fallback intact.

## Current factual state
- No signed native releases, physical-device media tests or native delivery
  acknowledgements have been produced by this plan.
- Existing backend production stays on `main` and native routes remain
  default-OFF.
