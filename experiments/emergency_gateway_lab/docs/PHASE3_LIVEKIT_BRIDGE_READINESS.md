# Phase 3: two-way audio laboratory and LiveKit bridge readiness

## What v0.3 **actually** demonstrates

The executable `python -m gateway.private_audio_cli` does a real SIP UDP loopback **INVITE / 200 OK (SDP) / ACK / BYE** exchange with a local mock User Agent. The offer and answer negotiate two ephemeral IPv4 loopback RTP sockets with `PCMU/8000` and `a=sendrecv`. It sends 25 **synthetically generated 20 ms audio RTP packets per direction** in about half a second, checks decoding and audio energy, and reports packet counts and loss. No audio device, microphone, speaker, files, external IP, LiveKit connection, SIP trunk, PSAP, location or phone number is involved.

This is a **transport and codec test**, not a human two-way voice test and not evidence of LiveKit / PSTN interoperability.

## What is already present in VeriFYD Circle

- Horizons uses `livekit-client` and `@livekit/components-react` for group video.
- Render and the existing WebSocket stack coordinate user session and calling metadata.
- Keep their existing signaling, room tokens, media recording, WebRTC and group-video code untouched. Never reuse production token-signing secrets in this lab.

## How to reach real private LiveKit-to-SIP two-way audio (next separately tested deployment)

1. Use a **separate, non-production machine** with a UDP-capable network interface. This avoids exposing SIP/RTP from the current HTTP-oriented Render web service.
2. Bring up an **isolated LiveKit server, Redis, and LiveKit SIP bridge** using the *official* self-hosted SIP instructions and a freshly created lab API key/secret. See https://docs.livekit.io/transport/self-hosting/sip-server/ . The documented standard ports include SIP UDP/TCP 5060 and UDP RTP 10000–20000; exposure must be limited by firewall to only approved peers. Public exposure is not appropriate for this lab.
3. Provision a **private Asterisk/FreeSWITCH test PBX**, reachable only on the private lab subnet, with a single synthetic echo extension such as `600`. Do not configure outbound trunks, 911, 933, or general dialing routes.
4. Configure a LiveKit outbound SIP trunk targeting only that private test PBX host and supply an internal test-only caller ID. Create a `CreateSIPParticipant` for the synthetic extension and connect an authenticated **test** LiveKit room participant using WebRTC.
5. Confirm *actual human* two-way audio in the test room, measured jitter/packet loss, SIP INVITE/200/ACK/BYE and correct media teardown. Test disconnects, absent microphones, and device permission issues. **Do not treat a synthetic tone exchange as fulfilling this step.**
6. Add separate, auditable traffic-gating, TURN/firewall, protection against SIP scanning and spoofing, metrics, secure key management and redaction. No public deployment before security and privacy review.

Official LiveKit reference: https://docs.livekit.io/telephony/making-calls/outbound-trunk/

## What we still need for a real, authorized 933 test

- A participating E911 interconnection/SIP trunk **and its written authorization** for 933 testing.
- A provisioned, E911-enabled telephone callback ID and validated emergency address, including clear rules for mobile or moving users.
- Confirmation of transport, destination-number format, interconnect SIP TLS requirements, callback/ANI policies, location method and test allowance.
- A one-call-per-minute limit where required; use only after the provider confirms provisioning and scope.
- Operator-approved firewall and route allowlist with **933 permitted only in a separate authorized test profile; 911 remains disabled by default**.
- A real 933 readback cannot validate all emergency-network routing. Full E911 readiness requires specialist regulatory and operational testing and appropriate emergency-authority coordination.

Official 933 resources:
- https://www.bandwidth.com/support/en/articles/12822950-e911-for-static-voip-integration-testing-guide
- https://support.telnyx.com/en/articles/1130709-how-do-i-test-e911-service

## Production isolation

All `experiments/emergency_gateway_lab/` code is in a separate **draft** GitHub PR. It is not imported into `main.py`, not deployed to Render, and not linked to the Horizons Call button. Its local key is independent of any real user account. No real emergency calling is implemented.