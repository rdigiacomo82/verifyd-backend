# Phase 4: private LiveKit-to-SIP microphone echo lab

## Status and strict scope

- **Implemented in source and offline-validated:** Docker Compose topology for LiveKit + LiveKit SIP + Asterisk + Redis; a separate minimal browser LiveKit microphone room; JWT creation; fixed-extension `CreateSIPParticipant` control; local PBX echo dialplan; standalone unit tests.
- **Not yet live-tested:** Docker boot, LiveKit/SIP interop, browser microphone-to-echo audio, packet-loss measurements, endpoint media quality. **No proof of connected human audio until on-host testing.**
- **Never implemented:** 933 or 911 carrier calls, PSAP access, public SIP trunk, callback number, verified dispatchable location, or emergency transport.
- **Production untouched:** `main.py`, Render, Horizons, existing private WebRTC and group LiveKit setups.

## Diagram

Browser at localhost:5175 (LiveKit microphone-only participant, room verifyd-private-echo)
    ↕ WebSocket 127.0.0.1:7880 / ICE 127.0.0.1:7882
Standalone LiveKit Server at localhost
    ↕ internal signaling via localhost Redis
Standalone LiveKit SIP at localhost (5060; RTP 11000-11099)
    ↕ SIP UDP to localhost:5070; RTP to Asterisk 12000-12099
Standalone Asterisk PBX (context from-livekit, extension 6000 → Answer / Echo / Hangup)

No provider, gateway to public telephone network, LiveKit Cloud connection, or production service is used in this stack. Docker container images come from third-party software maintainers but no telecommunications service account is involved.

## Operator prerequisites

1. A **dedicated, trusted Linux machine** (not a public-facing VM) with Docker Engine, Docker Compose plugin, Python >=3.11, Node >=20.19 and npm. Some images require Linux amd64/arm64. `docker compose version` must work.
2. Check the following ports are free: TCP `6379`, `7880`, `5175`, UDP `5060`, `5070`, `7882`, ranges `11000–11099`, `12000–12099`. The PBX does not have a SIP registration, account, outgoing dialplan or PSTN connection.
3. Keep the host firewall enabled and inbound access from other computers denied. The stack uses **host networking**, which is essential for SIP/RTP on Linux and is not an alternative to a firewall. The LiveKit API, SIP and Asterisk SIP signaling are explicitly loopback-bound. Verify with `ss -lntup` and `docker compose logs` before allowing any other network access.
4. Do not pass production LiveKit/Render credentials. `python bridge/phase4_bridge.py init` generates new random lab credentials under ignored `bridge/.generated/` with restrictive filesystem permissions.

## Setup and validation

From the package root:

```bash
python -m unittest discover -s tests -q
python bridge/phase4_bridge.py init
cd bridge
docker compose -f compose.yaml config
# Verify `network_mode: host` and no unintended ports on this Linux host.
docker compose -f compose.yaml up -d
sleep 6
docker compose -f compose.yaml ps
docker compose -f compose.yaml logs --tail=70 livekit sip asterisk
cd ..
python bridge/phase4_bridge.py doctor
```

The Python `init` command will refuse to overwrite keys if re-run; preserve the generated keys so they match the running services. If Docker Compose reports an incompatible image tag, update only the lab test environment using vetted upstream versions after checking the official SIP/LiveKit changelogs.

Start the local microphone test page in a separate terminal:

```bash
cd bridge/web
npm install
npm run dev
```

Open `http://127.0.0.1:5175` **on the same Linux host**. Generate a five-minute join token in a different local terminal:

```bash
python bridge/phase4_bridge.py token
```

Paste the token and select **Join room and enable microphone**. After microphone permission is granted and the room reports connected, run:

```bash
python bridge/phase4_bridge.py show-fixed-target
python bridge/phase4_bridge.py call-echo --confirm-private-echo
```

If successfully bridged, the local `Echo()` dialplan returns the microphone audio. Use headphones to reduce feedback; turn speaker volume down initially. Quit the browser room afterward to turn off the microphone. The local Python `call-echo` command only submits a fixed internal `6000` target via a loopback-only Twirp endpoint. It cannot accept a phone number, SIP URI or alternate destination argument.

At the end, **stop all containers**:

```bash
docker compose -f bridge/compose.yaml down
```

The `npm install` command downloads open-source JavaScript libraries for the local browser test and does not require a SIP provider account.

## Required pass/fail evidence before the next phase

- Docker shows four services running (`redis`, `livekit`, `sip`, `asterisk`).
- CLI doctor reports Redis and LiveKit reachable on loopback.
- Browser connects, microphone permission succeeds, and room shows a second SIP audio participant after calling `6000`.
- Asterisk logs show an incoming local SIP INVITE and the `from-livekit` **6000** dialplan.
- Audio comes back and stops when the browser leaves; investigate one-way sound if it does not.
- An *unapproved* target (such as 933, 911, or another number) is rejected by both the test CLI and Asterisk's hang-up dialplan.
- All container ports remain inaccessible from outside the test host.

**Do not attempt live 933 tests or claim E911 compliance as a result of this echo test.** Carrier 933 verification still requires a legitimately provisioned emergency-services trunk, usable emergency callback identity, registered address, separate qualified testing and regulatory review.

## Known limitations

- The local environment is **Linux host network only**. No Docker runtime is available in the ChatGPT container, so actual runtime interoperability remains unverified.
- Asterisk and LiveKit container tags should be pinned to known-good digest versions before any broader rollout. The images are lab dependencies, not emergency-network providers.
- SIP service and SDP behavior can vary by upstream release. The operator should inspect `docker compose logs -f sip asterisk` and update only this isolated lab if a configuration field changes.
- Two-way echo is not a verified emergency call, and has no failover, geolocation, emergency callback, PSAP routing or emergency-network resilience.

## Upstream references

- LiveKit self-hosted SIP: https://docs.livekit.io/transport/self-hosting/sip-server/
- LiveKit outbound SIP participants: https://docs.livekit.io/telephony/making-calls/outbound-calls/
- LiveKit tokens/grants: https://docs.livekit.io/frontends/reference/tokens-grants/
- Asterisk PJSIP examples: https://docs.asterisk.org/Configuration/Channel-Drivers/SIP/Configuring-res_pjsip/res_pjsip-Configuration-Examples/