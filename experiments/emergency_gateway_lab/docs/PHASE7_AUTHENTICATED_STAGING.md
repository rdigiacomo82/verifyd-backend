# VeriFYD Phase 7 — authenticated PRIVATE gateway staging only

Status: **code staged, NOT deployed, NOT production-enabled**. This phase is a guarded bridge between a normal, **server-revocable Trust Voice bearer session** and a **microphone-only LiveKit staging token** for a local echo-room test. It does NOT dial calls, route E911, provision 933, or connect to emergency services.

## Scope and integration contract

- Existing backend auth: `trust_voice_messages._identity_from_bearer(authorization)`; validates the Trust Voice session and resolves the authoritative identity. No new Circle login and no database migration.
- New optional router (not included in `main.py`): `experiments/emergency_gateway_lab/phase7/trust_voice_private_gateway_staging.py`.
- `GET /trust-voice/private-gateway/staging-status`: authenticated + identity allowlisted; reports **configuration state only**, explicitly `network_verified:false`.
- `POST /trust-voice/private-gateway/staging-room-token` body **exactly `{}`**: same gates; issues 180-second, microphone-only LiveKit JWT for fixed lab room `verifyd-private-echo`. No account ID, handle, email, location or telephone number is included in JWT. The response has `Cache-Control: no-store`.
- Not implemented: a call-start endpoint. The already-tested lab operator CLI remains the sole method for targeting fixed Asterisk extension `6000`.
- Default: **404 until explicitly enabled**; exact immutable identity ID allowlist; if credentials absent, re-used from production LiveKit, weak, or URL unsafe, token issuance returns **503**.

## Environment variables — future isolated staging backend only

Do **not** set these in production Render. These values are illustrative; do not use examples as real credentials.

```bash
VERIFYD_PRIVATE_GATEWAY_STAGE_ENABLED=1
VERIFYD_PRIVATE_GATEWAY_STAGE_ALLOW_IDS=<exact-existing-test-identity-uuid>
VERIFYD_PRIVATE_GATEWAY_STAGE_LIVEKIT_URL=ws://127.0.0.1:7880
VERIFYD_PRIVATE_GATEWAY_STAGE_LIVEKIT_KEY=<dedicated-separate-stage-livekit-api-key>
VERIFYD_PRIVATE_GATEWAY_STAGE_LIVEKIT_SECRET=<strong-random-secret-at-least-32-characters>
VERIFYD_PRIVATE_GATEWAY_STAGE_ALLOW_LOCALHOST=1
```

For a future remote staging LiveKit service, URL must use `wss://`; the localhost opt-in is not required. The URL, API key, and API secret must be separate from the production LiveKit group-video service. **Never copy real Circle session secrets or production LiveKit keys to a MacBook demo or public frontend.** An isolated backend environment capable of verifying revocable Circle sessions must be designed and approved before live account testing.

## Deliberate deployment barrier

The experimental router **is not imported by production `main.py`**. To smoke-test this in a disposable integration backend, a developer must explicitly add the router to that backend's application composition:

```python
from experiments.emergency_gateway_lab.phase7.trust_voice_private_gateway_staging import router as private_gateway_staging_router
app.include_router(private_gateway_staging_router)
```

This is an **example only**, not a request to patch or publish `main.py`. Do not wire this into the live Render service until a security review, isolated hosting, CORS, resource/capacity tests, and a separately reviewed release plan have been completed.

## Optional Horizons frontend replacement

`Horizons_OPTIONAL_NEW_FILE/TrustVoicePrivateGatewayLab.jsx.txt` is a **complete replacement** based on the Nov 6, 2026 21:14:34 Horizons export. It preserves manual local token testing. It adds a button to request a token with the signed-in Trust Voice bearer, but the button is **not present unless all flags pass**, AND the browser is localhost:

```ini
VITE_VERIFYD_PRIVATE_GATEWAY_LAB_ENABLED=true
VITE_VERIFYD_PRIVATE_GATEWAY_STAGE_BROKER_ENABLED=true
VITE_VERIFYD_PRIVATE_GATEWAY_STAGE_API_BASE=https://verifyd-backend.onrender.com
```

These flags are for **local Vite DEV ONLY**. Do not enable on hosted Horizons (`vfvid.com`), the regular App Store app, or a live production build. The status/token endpoints are **NOT deployed** to that origin; until a reviewed staging deployment, the optional button will report `HTTP 404`. That is expected. The existing manual local Echo() flow continues to work without any of these new flags.

The browser **never receives a LiveKit API secret or SIP admin token** and cannot set a destination, phone number, location, SIP trunk, or room; it refuses a token response that points anywhere other than `ws://127.0.0.1:7880` and `verifyd-private-echo`.

## Validation

From the experiment root, after copying the new `phase7/` folder and `tests/test_phase7_staging.py`:

```bash
python -m unittest discover -s tests -v
```

Offline tests: **52 passed** (36 previous + 12 staging-auth + 4 frontend guard), including disabled-feature, stale/forged-token rejection via auth-validator test stub, allowlist denial, fixed room, 180-second token, no dial route, request-body allowlist, production LiveKit secret reuse refusal, unsafe URL refusal, configuration failure.

This test uses a **stub for the Trust Voice session validator only**: it is not evidence that the live Circle backend, MacBook network, or real-auth staging integration has been exercised.

## Next release gates

1. Preserve the **successful** MacBook manual echo test, existing Circle production calls, voicemail, group video, Eye Gaze, messages, calendar, location, and Circle Safety.
2. Audit `main.py` and server CORS/session lifecycle with an isolated staging deployment (not Render production), then test allowlisted Circle account logins and revocation.
3. Verify microphone and return audio on Mac/Windows/iPad/Android while monitoring network loss, timeout, no-audio failure, and accessible controls.
4. Only after independent E911 telecom/regulatory review, separately evaluate authorized carrier-connected `933` readback with verified dispatchable address and callback reliability. No public 911 feature exists here.

**Emergency guidance:** VeriFYD private gateway test is NOT an emergency service. Use your device's normal emergency telephone service when needed.