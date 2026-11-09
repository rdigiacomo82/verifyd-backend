# VeriFYD Circle — Silent Distress + Recipient Emergency Location Text (STAGING)

Based on the Horizons export `2026-11-07T181447.124.zip`. This is an additive test-only implementation. **It is not Text-to-911 or E911 interconnection**, and it does not automatically reach police, fire, paramedics, a PSAP, an ECC, or RapidSOS. Never test by sending non-emergency messages to 911.

## Files for Horizons

1. In Horizons, replace `apps/web/src/components/TrustVoiceCircleSafety.jsx` with **all** text from `TrustVoiceCircleSafety.jsx.txt`.
2. Replace `apps/web/src/components/TrustVoiceSettings.jsx` with `TrustVoiceSettings.jsx.txt`.
3. Replace `apps/web/src/pages/TrustVoiceBeta.jsx` with `TrustVoiceBeta.jsx.txt`. This is a large authoritative full replacement based on the newest uploaded export. If Horizons has made newer edits, use the included `.diff` changes against that version instead, preserving newer fixes.
4. Validate imports, JSX, route and device behavior in Horizons before publishing. The implementation is hidden unless Vite `DEV` AND `VITE_VERIFYD_SILENT_DISTRESS_ENABLED=true`. That flag must not be enabled in production.
5. No other components need to be replaced. `TrustVoiceEmergencyWorkflowLab.jsx` and existing 911/Distress features are untouched.

## Backend contract

GitHub branch `experiment/verifyd-silent-distress-text-location-20261107` adds an **opt-in, default-OFF** feature to existing `trust_voice_circle_safety.py`:

- `VERIFYD_CIRCLE_SAFETY_ENABLED=1` and existing Circle Safety pilot rules must already be satisfied.
- `VERIFYD_CIRCLE_SILENT_DISTRESS_ENABLED=1` must be explicitly configured for a private test. Don't enable in production before real-device testing.
- `GET /trust-voice/safety/settings` includes `silent_mode_supported`.
- `POST /trust-voice/safety/distress` accepts `mode: 'silent' | 'standard'`, default `'standard'`. Silent requests are rejected if not enabled, and no standard alert is created for unsupported modes.
- One active incident per identity; a duplicate request returns the existing mode, which client checks.
- Current incident and recipient inbox retain the mode. Recipients can view last reported latitude/longitude, accuracy and timestamp only if still authorized and not muted.
- When a sender enters their configured PIN the existing stop route ends the incident and clears its position.
- Existing recipient push notifications are best effort (not guaranteed on offline/locked devices or without permissions); the sender is not sent a Circle Safety push by this code.

## Expected behavior

**Sender**: Opens Silent Distress on a Circle Safety-enabled test account. The client first confirms server capability. While creating an incident it shows a neutral connecting screen; if creation fails it shows an unambiguous but discreet *not delivered* error. Once the incident is active it shows only **Enter PIN**, masked PIN input and **Enter**. No in-app siren, animation, red banner, vibration or loud local alert is added by this component. This does **not** suppress operating-system alerts, other applications, existing active calls, OS/browser prompts, or sounds produced elsewhere. To improve privacy, grant location permission ahead of time.

**Recipients**: Eligible, currently authorized Trust Circle recipients see a red distress notice and a button that opens the sender's **last reported** location on a map. A manual emergency text helper contains coordinates, accuracy, time, and a disclosure that the recipient is reporting for another person. Recipients may **copy plain text**, separately **copy map link only when requested by dispatch**, or open their phone's native Text-to-911 app (`sms:911`). They must paste and **send themselves**. The text-to-911 conversation takes place in their native messaging app, not Circle. The app cannot verify whether the link is viewed or the SMS is received.

## Cautions

- Native Text-to-911 availability varies by location, carrier, device, and recipient's handset. `sms:911` cannot guarantee SMS delivery, and **Wi-Fi-only desktop/iPad** contexts may not support it.
- A recipient reporting for someone far away may be routed to the **recipient's local 911 center**, not the distressed person's. They must say they are reporting a remote incident, give the actual location first, and request transfer to the appropriate jurisdiction.
- Some 911 centers advise **not to send links, photos or attachments unless they request them**. Default to the readable location text. The map link contains sensitive coordinates and can be forwarded or logged by third-party mapping services; use it only for a real emergency and only if the operator asks.
- Last location may be inaccurate, stale or missing; browsers do not guarantee tracking while backgrounded/closed or device locked.
- Silent distress is **not** a direct emergency connection. Trust Circle notification is not guaranteed and should not replace available phone/911 services.
- Production server and Horizons were NOT deployed or modified by this package. No live 911/933/PSAP calls or SMS were made.

## Staging test checklist

1. Keep production services unchanged. Configure separate staging pilot accounts and keys; retain 911 provider service OFF.
2. Preconfigure sender disengage PIN and recipients' Trust Circle membership, push permission and geolocation permission on a test device.
3. From sender's test device, trigger Silent Distress and verify no *Circle-originated* sound/red banner; verify only PIN view after successful server acknowledgment. Record any OS prompts separately.
4. From each designated recipient account, confirm the distinct alert with map coordinates and last update. Verify no location leak to unrelated/non-eligible or muted accounts. Validate recipient notification on iOS, Android and browsers.
5. Ensure wrong PIN leaves incident active; correct PIN ends it and clears the recipient inbox. Check older servers reject silent mode *before* creating an incident.
6. Use a fake/synthetic incident for texting UI tests. Copy the text and map link into a local note, **not to 911**. Check clipboard error fallback.
7. Test a sender's browser offline, mobile background/locked, GPS denied, and permission revoked. Do not promise background GPS when OS disallows it.
8. Before any real emergency-network integration, obtain an authorized Text-to-911/E911 delivery and acknowledgment path with verified location/callback, then perform provider-authorized testing.