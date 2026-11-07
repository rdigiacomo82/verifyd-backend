# VeriFYD Emergency Messaging Workflow — owned prototype, LAB ONLY

**STATUS:** TRAINING SIMULATION ONLY. No real 911/PSAP/Text-to-911, RapidSOS, EMS, police, fire, SMS, SIP trunk, outbound webhook, dispatch, or trusted contact delivery. Do not represent any simulator acknowledgement as confirmation that emergency services were reached. In a real emergency use the native phone service or supported native Text-to-911.

## Why this design

This is a **VeriFYD-owned workflow engine** modeled on common emergency-event state-machine patterns: initiate a simulated event, exchange two-way accessible text, receive a clearly marked simulated response, and close the event. It uses existing VeriFYD Circle revocable bearer sessions, and normal HTML buttons/text inputs for Eye Gaze/switch/keyboard compatibility. It does not reproduce or claim to use the RapidSOS network; a valid emergency-network agreement is required for actual delivery.

The RapidSOS Workflow API is a partner-controlled system. Official docs describe provider-issued credentials, per-customer workflows and webhooks; those do **not** grant applications independent access to 911. See https://developer.rapidsos.com/tech_partners/default/emergency-workflow . This project is not an official RapidSOS integration.

## Backend

- Source: `verifyd_emergency_workflow_lab.py` at repository root.
- Add `from verifyd_emergency_workflow_lab import router as emergency_workflow_lab_router` to backend `main.py` imports.
- Add `app.include_router(emergency_workflow_lab_router)` near other Trust Voice routers.
- **Default OFF**. Routes return HTTP 404 unless **both** `VERIFYD_EMERGENCY_WORKFLOW_LAB_ENABLED=1` and `VERIFYD_EMERGENCY_WORKFLOW_LAB_IDS=<comma-separated authorized internal identity IDs>` are configured. No service keys, carrier credentials, customer addresses or phone numbers are needed/accepted.
- Uses existing `trust_voice_messages._identity_from_bearer()` so revoked Circle sessions cannot call routes.
- All scenarios and locations are explicitly simulated. Request extra fields (phone, location, recipient) are rejected. The server never sends outbound requests, WebSocket signals, SMS, Circle notifications or calls.
- Incident state is **in-memory**, non-durable, per process, expires after 30 minutes. Intended for local/nonproduction functional testing only; not a reliable incident system.
- Fixed statuses `simulation_open`, `simulation_acknowledged`, `simulation_closed`. Every successful response returns `dispatch_connected:false`, `public_emergency_message_delivered:false`, `psap_acknowledged:false`.
- Throttles new incidents (8 per hour per identity), active incidents (one), messages (24 entries). No personally identifying data should be entered in the training field. **Never** enable on a public production app.

## Frontend — Hostinger Horizons (development only)

Current source baseline: Horizons full export `2026-11-06T230524.482.zip`.

- **NEW:** `apps/web/src/components/TrustVoiceEmergencyWorkflowLab.jsx` from `TrustVoiceEmergencyWorkflowLab.jsx.txt`.
- **COMPLETE replacement:** `apps/web/src/pages/TrustVoiceBeta.jsx` from `TrustVoiceBeta.jsx.txt`, or use `TrustVoiceBeta_minimal.patch` if Horizons should patch rather than replace the protected file.
- This adds one **Emergency Messaging Practice Lab — NOT 911** button to the Call another ID section. It **does not change** existing `911`/`@911` interception, 911 preview/native dialer, Circle Distress recipients, messages, calls, voicemail, groups, or adaptive access code.
- The button and component are visible **only** in `import.meta.env.DEV` on `localhost` / `127.0.0.1` when `VITE_VERIFYD_EMERGENCY_WORKFLOW_LAB_ENABLED=true`. The component requires a Circle session and an explicit `VITE_VERIFYD_EMERGENCY_WORKFLOW_LAB_API=https://YOUR-PRIVATE-STAGING-BACKEND` value. It contains no production API credentials.
- A browser on `vfvid.com` never displays the lab button, regardless of accidentally set Vite flags.
- Browser API host must provide correct CORS access from the local dev site; use a dedicated staging environment and test identities. Do not transplant production secrets or real identity data into a third-party test environment.

## How to validate

1. Run `python -m pytest -q tests/test_emergency_workflow_lab.py`: 23 API test cases plus 6 frontend-static checks.
2. Run `python -m py_compile verifyd_emergency_workflow_lab.py`.
3. Build the Horizons dev app with the new component and guarded import; do not publish without a complete build/regression review.
4. With the backend flag OFF, confirm all lab endpoints return 404 and the production frontend contains no lab button.
5. In a private staging instance, use only a synthetic Circle account; enable only that identity ID. Verify `GET /status`, `POST /incidents`, `GET /incidents/current`, `POST /messages`, `POST /close`, loss of session permission, cross-account isolation, expiration, accessibility and network failure.
6. Confirm all statuses make clear **no dispatch**. Do not send SMS to 911, call 911, or enter real emergency information in tests.

## Next phase — not implemented

- Encrypted durable and audited incident data (if authorized by an emergency services partner)
- Location and callback validation and dynamic location
- Authenticated two-way event updates from an approved network partner or ECC/PSAP
- Real responder delivery and human-in-the-loop acknowledgement (never inferred from HTTP 200)
- Dedicated reliability, retention, regulation, legal and accessibility review
- Carrier/PSAP/ECC interconnection approval and controlled tests before any activation

**This lab does not improve real emergency reachability.** It is an isolated working prototype to verify VeriFYD's own accessible interaction and backend event model.