# VeriFYD Circle: locked-screen ringing and native incoming calls
Status: DESIGN + WEB PUSH PILOT, default OFF. 2026-11-07.

## Existing issue confirmed in code
1. The backend only attempts live `incoming_call` signaling when recipient has an open WebSocket.
2. WebSockets can be suspended while phones are locked, even if the session remains valid.
3. The existing push backend supports messages and **missed** calls but not initial ringing.
4. The current PWA service worker already shows background notifications and opens `/voice` on click; this cannot force the screen on or maintain a full-screen ringtone.
5. In a native Capacitor runtime, `trustVoicePush.js` deliberately bypasses Web Push; native APNs/FCM integration is required.
6. Server uses in-memory ringing state; process restarts drop active invitations. Current timeout is 30 seconds.

## Pilot stage (this PR)
- `VERIFYD_INCOMING_CALL_PUSH_PILOT_ENABLED=0` by default, and a separate
  `VERIFYD_INCOMING_CALL_PUSH_PILOT_HANDLES` allowlist. Both are required.
- Existing authorization/blocking checks still apply BEFORE call creation.
- Use only current nonrevoked session-bound Web Push subscriptions as eligible wake targets.
- Deliver privacy-limited incoming-call push (only type, random call ID,
  call type and deep link) in addition to the existing WebSocket event.
- If the recipient has no socket but has a valid registered push subscription,
  let the 30-second invitation be placed via best-effort Web Push.
- Do not claim that an accepted push means the phone received or displayed it.
- When a previously sleeping recipient reopens/reconnects, resend any live
  authenticated, authorized, unexpired ringing invitations over the socket.
- Keep existing foreground call handling and missed-call/voicemail timers unchanged.
- Server sends no answer/decline unless the recipient authenticates and actually answers/declines.
- The web push pilot supports **tap to open the app and answer**; it does not
  provide a guaranteed phone-like wake-up screen, continuous audible ring,
  or actionable OS-level Answer/Decline buttons.

## Client/PWA pilot acceptance tests
A. In the current VeriFYD device Settings > Notifications UI, confirm
   "On for this device", not merely OS-level notification permission.
B. Press "Send Test"; with the app minimized and Android locked, confirm
   the generic push arrives in the notifications shade.
C. On a pilot handle, PC calls Android when Android is locked; confirm a
   **new incoming-call** push appears before the 30-second ring expiry.
D. Tap promptly, unlock, and confirm the WebSocket reconnects and the
   *same* pending call ID rings. Answer, verify two-way media.
E. Leave the push untouched >30 s; confirm exactly one missed call and
   no phantom incoming UI after app reopens.
F. Test blocked callers, revoked sessions, no push subscription,
   DND/Focus, Wi-Fi/mobile changes, 2 phones, and declining the call.
G. Confirm existing audio/video calls, messaging, voicemail, group calls,
   Eye Gaze, calendar and email are unchanged.
H. Validate mobile Home Screen app icon uses valid image assets.
Only enable for a pilot and after confirming backend VAPID push configuration.
Roll back instantly by turning the flag OFF.

## Native Android (required for phone-like ringing behind lock screen)
- Obtain the signed production Android project; no Android repository is currently
  available in the connected workspace.
- Register FCM per app installation; backend stores encrypted/revocable device token,
  platform, user/session linkage, token lifecycle and opt-in.
- On `call_invite`, create durable time-bounded invite with opaque ID,
  idempotency and authoritative `expires_at`, then send FCM **high priority,
  user-visible** call push. Only true calls get calling priority.
- Android receives FCM in `FirebaseMessagingService` and immediately displays
  `Notification.CallStyle.forIncomingCall`/appropriate call notification,
  category CALL, high importance channel, visible ringtone/vibration.
- Full-screen UI is limited by Android 14+ special access / Play review for
  qualified calling apps. If unavailable, show heads-up/lock-screen notification
  rather than bypass the OS.
- Answer/Decline PendingIntents should be immutable, target internal explicit
  activities, validate call ID with backend session, then atomically accept or
  decline the pending call. Do not mark answered before server confirms.
- Integrate appropriate Android Telecom/phone-call foreground service and
  microphone capture for ongoing audio when screen turns off. Respect notification
  permissions, system DND, battery controls and Android restrictions.
- Handle expired/canceled invitation, answer on other device, token revocation,
  network loss and multiple simultaneous invitations.
- Test on actual locked Android devices with current OS/Play policies.

## Native iOS (separate signed app work)
- Obtain actual App Store signed Xcode source; the connected private repository
  currently contains a Capacitor **starter**, not the installed application.
- Register PushKit VoIP token; only real incoming calls use VoIP pushes.
- On VoIP push, promptly call `CXProvider.reportNewIncomingCall`;
  report Answer/Decline actions to server. Apple requires CallKit for PushKit.
- Configure AVAudioSession, background audio and real native media handling for
  calls that continue when the app is backgrounded or locked.
- APNs token authorization/session revocation and device ownership are required.
- For messages, voicemail and ordinary alerts, use normal APNs/user notifications,
  **not** PushKit VoIP notification.
- Test on signed device via TestFlight including force-quit limitations,
  expiration, different Focus settings and permissions.

## Required shared backend final state (before public release)
- Persist queued ringing invites in PostgreSQL with atomic state transitions;
  current stage keeps in-memory state only.
- `POST /trust-voice/calls/{call_id}/accept` and `/decline` authenticate
  recipient and enforce permissions, blocks, expiration, replay defense,
  same-call idempotency, and device arbitration.
- Device push tokens bound to user + revocable trusted-device sessions;
  delete them on logout, account deletion, token invalidation or opt out.
- Caller receives authoritative ringing/answered/declined/no answer outcomes;
  mobile recipient receives cancellation when caller hangs up.
- Use 30-second default unless product team explicitly authorizes a new timeout.
- Instrument sent vs accepted vs displayed (when measurable) vs answered;
  never equate push-API HTTP success with real device delivery.
- Avoid placing contact names or sensitive data in notification body by default.
- No push configuration or signing credentials are stored in GitHub.
- No changes to emergency/E911 logic; emergency delivery is never guaranteed.
