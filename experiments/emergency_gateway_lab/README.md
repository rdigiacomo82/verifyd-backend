# Phase 5 update — ephemeral GitHub Actions headless echo validation

The experimental branch now has an Ubuntu-hosted GitHub Actions test of **real WebRTC + SIP + Asterisk Echo() audio** using a synthetic local test tone. The job runs Docker on a disposable runner, uses new lab-only credentials, and tears itself down. It does not require Docker/WSL on Windows and does not contact public phone numbers, 933, 911 or any carrier. A successful headless test is **not** a human microphone test.

Follow the CI result in the existing draft PR. Source: `.github/workflows/verifyd-private-sip-echo.yml`, `bridge/headless_echo_ci.py`, and `docs/PHASE5_EPHEMERAL_LINUX_CI.md`.

---

# VeriFYD Emergency Gateway — Lab v0.4 (PRIVATE LiveKit-to-Asterisk echo test)

**NOT AN EMERGENCY CALLING SERVICE. NO 911, 933, PSTN, LOCATION ROUTING, or PUBLIC SIP.**

Phase 4 adds an **independent, Linux-only, localhost-bound LiveKit + LiveKit SIP + Asterisk echo-extension lab** under `bridge/`. This is the first phase that can, **once the Docker stack is successfully run and tested on a suitable Linux host**, carry real microphone-to-speaker sound through LiveKit and SIP. It is **not verified live yet** because Docker is unavailable in the development environment. The previous v0.1–v0.3 isolated experiments remain intact.

The only callable SIP destination is the private extension **6000**, implemented as an Asterisk `Echo()` app; a separate browser with a 5-minute microphone-only room JWT provides audio input. `phase4_bridge.py` generates private credentials, obtains a test token and sends one fixed SIP participant request when the operator explicitly confirms it.

**Quick start (Linux host with Docker Engine/Compose, Python 3.11+, Node.js 20.19+):**

```bash
python bridge/phase4_bridge.py init
# Check 5060 UDP, 5070 UDP, 7880 TCP, 7882 UDP, 6379 TCP, and the lab RTP ranges are unused.
docker compose -f bridge/compose.yaml up -d
python bridge/phase4_bridge.py doctor
cd bridge/web
npm install
npm run dev
# Open http://127.0.0.1:5175 in a browser ON THE SAME LINUX MACHINE.
# In a second terminal at the project root:
python bridge/phase4_bridge.py token
# Paste that token into the local browser. Enable microphone.
python bridge/phase4_bridge.py call-echo --confirm-private-echo
# Speak. Asterisk Echo() should return your voice to the browser. Headphones recommended.
# After testing:
docker compose -f bridge/compose.yaml down
```

**Do not put `.generated/credentials.json`, API keys, call tokens, or the lab on production Render or vfvid.com.** The application never dials 933; earlier offline 933 scenarios are unchanged. **Run only on a trusted Linux test machine, not a public VM or exposed server.** Docker's `network_mode: host` has different behavior on Windows/Mac Docker Desktop; Linux is the supported target for this configuration. For a Windows/Mac computer, use a dedicated Linux VM and its browser console, or wait until the environment is validated.

See `docs/PHASE4_PRIVATE_LIVEKIT_ECHO.md` for architecture, constraints and step-by-step troubleshooting.

---

# VeriFYD Emergency Gateway — LAB v0.3 (private SIP/RTP audio simulation)

**NOT an emergency service. DO NOT USE TO REQUEST EMERGENCY HELP.**

A separate, local-only proof of concept for VeriFYD Circle's future emergency voice architecture. This does **not** modify existing `/trust-voice/ws`, WebRTC calls, LiveKit group calls, Render production services, Horizons frontend, emergency services, or 911 routing.

## Implemented

- Independent FastAPI test backend (`gateway.api:app`) with explicit opt-in and a strong lab key.
- Synthetic session creation, state lookup, close, 3-minute expiry, bounded session count.
- Authenticated WebSocket lab signaling (`/lab/ws/{sid}`) that refuses any dialing command.
- SIP **OPTIONS** probe to a local IPv4 loopback-only mock SIP endpoint (`127.0.0.1:5099`). A 200 reply proves a SIP signaling message was received, **NOT** a phone call or audio session.
- Offline simulated **933** readback: synthetic callback and address placeholders with fixed success/failure scenarios; no provider routing or real phone numbers.
- **Local two-way synthetic audio**: a separate SIP `INVITE / 200 OK with SDP / ACK / BYE` dialog and 25 two-way PCMU RTP frames, all on IPv4 loopback. This is not human speech or a connection to LiveKit. New route: `POST /lab/sessions/{id}/private-audio-test`; WebSocket action `private_audio_test`; local CLI below.
- Unit and API tests for fail-closed design, synthetic-location-only payloads, authentication and loopback SIP.

**Not implemented:** live human calling, external SIP INVITE calling, LiveKit SIP bridging, real phone numbers, live 933 dialing, PSTN, E911, dispatchable location, ANI, ALI, PSAP routing, emergency callback, failover, carrier connectivity, any public deployment.

## Why we simulate 933 before placing a real test call

**933 is a provider-specific emergency service test**, not a globally accessible phone number. Telnyx and Bandwidth document it for E911-enabled SIP trunks and provisioned endpoint/callback information. Their automated readback typically verifies the callback number and the registered emergency location; it does **not** contact a dispatcher. Their docs are linked in `docs/933_CARRIER_TEST_CHECKLIST.md`.

The new simulator tests four states (`provisioned`, `missing_callback`, `missing_address`, `unregistered`) with **fictional text values only** and carries the explicit flags `real_933_call_placed: false`, `carrier_connected: false`, `real_emergency_service_validated: false`.

`POST /lab/sessions/{id}/simulate-933` accepts only `{ "scenario": "provisioned" }` or another fixed scenario; unknown fields, destinations, telephone numbers and coordinates are rejected. The local console also has a button **Simulate 933 readback (NO CALL)**.

**Do not change the lab to dial an external 933 endpoint without a separate authorized test SIP trunk, verified test caller ID/emergency address, and a deployment and safety review.** This proof of concept does not validate any public 911 connection.

## Run locally only

Install dependencies in an isolated Python environment:

```bash
python -m pip install 'fastapi>=0.110,<1' 'pydantic>=2,<3' 'uvicorn>=0.27,<1' 'httpx>=0.25,<1'
```

Use a test-only key of at least 32 characters and deliberately enable the lab:

```bash
export VERIFYD_EMERGENCY_LAB_ENABLED=1
export VERIFYD_EMERGENCY_LAB_KEY='REPLACE_WITH_A_RANDOM_TEST_ONLY_KEY_AT_LEAST_32_CHARACTERS'
```

Run **only on loopback** (two terminals). **Do not use public or production secrets:**

```bash
python -m gateway.sip_loopback
python -m uvicorn gateway.api:app --host 127.0.0.1 --port 8099
```

Check test-only health / create a test session:

```bash
curl -s -H "x-verifyd-lab-key: $VERIFYD_EMERGENCY_LAB_KEY" http://127.0.0.1:8099/lab/health
curl -s -X POST -H "x-verifyd-lab-key: $VERIFYD_EMERGENCY_LAB_KEY" -H 'Content-Type: application/json' -d '{}' http://127.0.0.1:8099/lab/sessions
```

Copy a returned session `id` to the next example (`SESSION_ID` is a local demo-only UUID):

```bash
curl -s -X POST -H "x-verifyd-lab-key: $VERIFYD_EMERGENCY_LAB_KEY" "http://127.0.0.1:8099/lab/sessions/$SESSION_ID/sip-probe"
```

For a visual test console, open `http://127.0.0.1:8099/lab/console` in the same computer and paste the lab-only key. Nothing in the console can dial a phone number. The audio test button only sends short synthetic audio tones between two localhost UDP sockets.

To try the **933 simulation** from the console:

1. Choose **Create synthetic session**.
2. Choose **933 simulation scenario**: provisioned / missing callback / missing address / unregistered endpoint.
3. Select **Simulate 933 readback (NO CALL)**.
4. Confirm that the output states `carrier_connected: false`, `real_933_call_placed: false`, and `real_emergency_service_validated: false`.

For Windows PowerShell, replace the two `export` commands above with:

```powershell
$env:VERIFYD_EMERGENCY_LAB_ENABLED = '1'
$env:VERIFYD_EMERGENCY_LAB_KEY = '<a strong randomly generated lab-only key at least 32 characters>'
```

Use **two PowerShell terminals with the same test-only environment variables**, then run the same Python module commands for the SIP mock server and the FastAPI lab server. Keep both servers listening only on `127.0.0.1`.


Expected: `sip_probe_status: "sip_options_acknowledged"`, `real_emergency_call: false`, and `media_connected: false`. If the mock server is not running, status is `sip_probe_unavailable`.

Test private synthetic two-way SIP/RTP audio without running the API (this also works on Windows):

```bash
python -m gateway.private_audio_cli
```

Expected: `mode=PRIVATE_LOCAL_SIP_RTP_SYNTHETIC_AUDIO`, `sip_dialog=invite_200_ack_bye_completed`, 25 RTP packets received per direction, `media_transport_connected=true`, `livekit_connected=false`, `carrier_connected=false`, and `real_933_call_placed=false`. `media_transport_connected` refers **only** to test tones between localhost ports. No user can talk to another person using this route.

The local web console also has **Test two-way synthetic SIP/RTP audio (LOCAL ONLY)**. Create a synthetic session first. The action requires the same strong lab-only key and rejects any field specifying a phone number, destination, address, SIP URI or real coordinates. 

Test the full suite:

```bash
python -m unittest discover -s tests -v
python -m compileall -q gateway tests
```

## Why this architecture

The current Circle frontend already uses the **Render** backend WebSocket `wss://verifyd-backend.onrender.com/trust-voice/ws`, WebRTC for direct calls, and **LiveKit** for group video. We will keep those working paths untouched. The isolated API/WebSocket demonstrates control-plane session, a mock SIP dialog and two-way test-tone RTP between sockets without impersonating a real 911 connection. See `docs/PHASE3_LIVEKIT_BRIDGE_READINESS.md` for the next private LiveKit-to-PBX stage. Media bridging into SIP needs a separate phase and an environment that supports public SIP/RTP UDP traffic; Render's normal HTTPS web service cannot be assumed to expose these UDP ports.

## Proposed next milestone (separate deployment and operator authorization)

- Deploy a private **non-production** LiveKit SIP/FreeSWITCH or Asterisk topology to a UDP-capable lab host and verify **actual human** two-way test audio on a clearly labeled test extension. The current synthetic RTP test is a prerequisite, not a substitute.
- Implement strong **account identity and session authorization** tied to the Circle backend; no debug lab secret in production frontend.
- Implement test-only location, no real location transmitted externally.
- Secure SIP, media, firewall restrictions, metrics, observability, operational controls; keep a hard outbound destination allowlist.
- Seek emergency communications counsel and authorized 911 interconnection before any public emergency dial interface.

No actual emergency call can be placed with this lab. To request emergency assistance, use the device's normal available emergency calling service.