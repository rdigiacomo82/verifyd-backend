# VeriFYD Circle native incoming-call readiness — implementation gate

Prepared 2026-11-16. **Staged backend foundation only; NOT a release, not live native calling.** Use alongside the current Horizons web-notification onboarding and the existing Android v4 QA sources. Never replace working production packages with a debug build.

## User experience
1. Sign in normally. Present one educational notification prompt per device/account, then the OS permission request only after tapping Enable Notifications. Respect Not now and Settings toggle.
2. Browser: existing session-bound Web Push registration runs with no manual pairing and remains a fallback. A browser service worker cannot guarantee instant closed-app wake or a native lock-screen call screen.
3. Android native: prompt for POST_NOTIFICATIONS where applicable, obtain an FCM token, and perform account-bound opt-in registration using **a verified native bridge**. The new POST /trust-voice/android-call-devices/auto-register accepts the authenticated existing persistent session, installation UUID, FCM token, allowed package and opted_in=true. Its response returns a scoped control token for encrypted Android KeyStore storage. **Never transport the user's session bearer or native control token through a deep link, JavaScript-injected page, URL/query string, or unverified app scheme.** Verified Android App Links alone do not authorize access to the browser's bearer token. Use an origin-authenticated TWA postMessage bridge with validated origin and app identity, or OAuth authorization code + PKCE with system browser. Do not enable the new server endpoint until this native side is built and evaluated.
4. iOS native: register PushKit VoIP token through existing POST /trust-voice/native-call-devices with an authenticated persistent session. Configure the actual Circle bundle ID and corresponding APNs .voip topic, sandbox/production environment, Push Notifications/VoIP background modes, CallKit provider, and report calls immediately with reportNewIncomingCall. Normal notification permission alone does not constitute VoIP token registration. No signed Circle iOS source was provided for integration here.
5. When a real, still-live call is received, show an Android full-screen incoming-call activity (where OS grants eligibility) or iOS CallKit screen; *do not automatically unlock the phone*. Show the caller's handle only if lock-screen privacy settings permit, otherwise generic caller until unlock. Respect DND/silent/Focus. Android API 34+ uses canUseFullScreenIntent with a user-controlled full-screen settings route. Native push must display a visible call promptly without blocking on Render network.
6. Answer/Decline must be authenticated against the authoritative server state; media is **not implemented** by the current v4 QA notification screen. Make true native Answer-to-WebRTC media handoff before promising lock-screen one-tap answer; Android /android-calls/{id}/answer still responds 409 native_media_not_ready.

## Backend switches — OFF until native client and production Firebase/APNs are verified
- VERIFYD_ANDROID_CALLS_ALL_OPTED_IN_ENABLED=0: when explicitly enabled, eliminates Android per-user allowlist but keeps session/device gates and package restriction.
- VERIFYD_ANDROID_CALLS_AUTO_REGISTRATION_ENABLED=0: when explicitly enabled, permits *authenticated, verified native client* to register a device without the legacy pairing-code UI. No automatic registration without valid persistent session, explicit opt-in and allowed Android package.
- VERIFYD_NATIVE_CALLS_ALL_OPTED_IN_ENABLED=0: when explicitly enabled, eliminates iOS per-user pilot handle list but keeps APNs/session/bundle restrictions.
- Existing VERIFYD_ANDROID_CALLS_ENABLED, VERIFYD_NATIVE_CALLS_ENABLED and required encryption/FCM/APNs credentials remain mandatory; do not turn on absent tested native app.
- Web Push global user opt-in remains independent: VERIFYD_INCOMING_CALL_PUSH_ALL_SUBSCRIBERS_ENABLED=1 (already deployed).

## Contract and QA requirements
- Each FCM/APNs message is a **single actual** call alert, not silent wake spam; 30s ORIGINAL TTL, no extending on retry. No group calls, emergency dispatch, or voicemail changes.
- Android FCM restricted_package_name must match registered production app and Firebase project. Distinguish com.verifyd.circle vs .dev and do not mix Sender IDs.
- iOS APNs topic must equal signed bundle ID plus .voip and use correct Team ID, signing key and environment.
- Preserve verified server-side caller permissions, blocking, revocation, account switch and device limit. Never broadcast device tokens, session tokens or private keys.
- Native device key must be deleted on sign-out/revoke; token refresh must not rebind it to another account.
- Test foreground, background, locked, closed-but-not-force-stopped, canceled before delivery, stale/expired, declined, answered and permission denied; demonstrate actual two-way audio before claiming native answer ready.
- Log sent/accepted/service-worker/native-display/answer/media-ready separately. FCM/APNs HTTP success is transport acceptance, not proof a handset rang.
- Do not submit to Google Play or App Store until release signing, production package, entitlement review, integration tests and final approval.

## Rollback
Setting the new native opt-in flags back to 0 returns native delivery eligibility to the existing pilot allowlist. Web Push and current media/signaling remain unchanged.
