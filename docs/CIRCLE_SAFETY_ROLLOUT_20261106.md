# VeriFYD Circle Safety — deployment and test plan (2026-11-06)

**Scope:** optional location range-exit alerts and PIN-stoppable, 60-minute distress notification to Trust Circle contacts.

## Deployment gate

- Keep draft PR #55 unmerged until the updated `TrustVoiceCircleSafety.jsx` has passed Hostinger Horizons preview and two-account tests.
- Feature flag `VERIFYD_CIRCLE_SAFETY_ENABLED` defaults to `0`. A second **mandatory fail-closed** environment setting, `VERIFYD_CIRCLE_SAFETY_PILOT_HANDLES`, must contain approved handles (comma-separated, without `@`) before any user can access the safety API. For the two verified internal test identities, use `roberttest22,databydesign`. Set these on **Robert's workspace → verifyd-backend web service** only. The existing worker and unrelated services do not need the flag.
- Ensure existing `VERIFYD_TRUST_VOICE_LOCATION_ENABLED` and Trust Voice Web Push are configured before a consented pilot. The monitored pilot member must independently opt into precise sharing and allow range monitoring. Safety is not intended for emergency-service dispatch.
- To disable quickly, set `VERIFYD_CIRCLE_SAFETY_ENABLED=0`; this does not change existing calling or location sharing.
- Do not enable globally until consent, authentication, rate limits, privacy purging, and push delivery have been tested across two devices and account sign-outs.

## Contract

- `GET /trust-voice/safety/settings`; `PUT /settings/range-consent` (only the monitored user can enable permission); `PUT /settings/pin` (4–8 digit PIN with current PIN required for change).
- `GET /trust-voice/safety/eligible` returns *only* mutually-authorized Trust Circle members currently sharing precise location AND explicitly allowing range alerts; never returns coordinates.
- `GET /zones`, `PUT /zones/{target_id}` with `{center_lat,center_lon,radius_m}`, `DELETE /zones/{target_id}`; 100m–20 miles, 20 zones per watcher, requires inside-then-outside crossings and acceptable GPS accuracy.
- `GET /alerts`, `POST /alerts/{alert_id}/dismiss`.
- `GET /distress/current`, `POST /distress`, `PUT /distress/{incident_id}/position`, `POST /distress/{incident_id}/stop`, `GET /distress/inbox`, `POST /distress/{incident_id}/mute`.
- Distress recipients are active, unblocked Trust Circle members. Recipients independently mute an incident. Sender stops with PIN; incident expires after 60 minutes. Attempts to send repeat push occur every five minutes while active and push scheduler runs.
- Additional safety permissions are independent of the existing location-sharing toggle. Changing or revoking ordinary sharing clears range-alert coordinates, invalidates zone state and prevents fresh monitoring.
- Safety web push is best effort, bounded and non-blocking. Push or active location may fail when permission is denied, phone is offline, app is suspended, or the user does not have a valid push subscription. No guaranteed locked-screen flashing, background location service, 911 dispatch, or GPS resolution claim.

## Two-account acceptance checks

1. With feature off, existing audio/video calls, message arrival, voicemail and location sharing remain unchanged, and safety endpoints refuse access.
2. A cannot see B as range-eligible until B enables *both* precise sharing to A and Safety monitoring consent, and both maintain Circle membership.
3. A can save and delete a correctly centered radius. An inside GPS update followed by a reliable outside update yields one alert; GPS uncertainty, stale coordinates, or a second outside update yields no duplicate alert.
4. Revocation of monitoring or location sharing prevents further alerts and hides/clears stored range coordinates.
5. A five-second hold on Call starts sender distress; ordinary tap retains original Call action. Accessible Settings confirmation is also available.
6. A recipient sees red in-app distress and a location link *when coordinates are available*; each recipient can mute independently.
7. Sender enters correct PIN and all active notifications end; incorrect PIN is throttled; no PIN yields no distress initiation.
8. If sender does nothing, distress expires after 60 minutes even when client closes; no pushes are scheduled after expiry.
9. Native iPhone/iPad background-location and notification permissions must be tested separately from Horizons preview.

## Tests

- `python -m unittest discover -s tests -p 'test_circle_safety_v1.py' -v`
- `python -m py_compile trust_voice_circle_safety.py trust_voice_location.py trust_voice_push.py`
- GitHub Actions: `.github/workflows/circle-safety-validation.yml`
