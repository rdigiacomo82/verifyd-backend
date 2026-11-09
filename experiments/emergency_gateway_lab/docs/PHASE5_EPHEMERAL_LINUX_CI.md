# Phase 5 — Ephemeral Linux CI / headless LiveKit-to-SIP echo

**Scope**: Private, automatic GitHub Actions experiment. No user's Windows setup or Render modifications.

This test starts Linux Docker Compose on a *temporary GitHub-hosted Ubuntu runner*, produces new local lab credentials and creates a LiveKit room participant that emits a synthetic 523 Hz tone. An outbound SIP participant targets **127.0.0.1:5070**, the local Asterisk extension **6000** only. Asterisk answers and echoes the audio. A successful verdict requires LiveKit to receive **non-silent audio frames from the returned SIP participant**, not merely report an answered SIP call. The test uses a real WebRTC media track and local SIP bridge, but **no human microphone**.

## Execution

GitHub Actions workflow: `.github/workflows/verifyd-private-sip-echo.yml` (experiment branch only). It runs on test branch pushes changing bridge files or can be started manually using GitHub Actions if access is authorized.

The `Verify true LiveKit-SIP-Asterisk two-way echo` step exits nonzero unless it receives at least 5 non-silent echoed frames. The workflow always tears down all containers and deletes temporary credentials with the runner.

## Limitations and boundaries

- GitHub-hosted runner cannot be used as a persistent service or interactively reached from Windows; a separate private Linux test host remains necessary for **human microphone** testing.
- Passing the headless test would verify a real WebRTC-to-SIP media bridge, not emergency telephony, 933, or a caller ID/address.
- Failing the test can indicate Docker image/config incompatibility, WebRTC ICE, SIP, RTP, or runner limitations; use logs, not assumptions, to diagnose.
- Nothing in this test authenticates a public SIP trunk or permits calls outside localhost. Do not merge this experiment into production until review.

## Required evidence

1. `docker compose` starts Redis, LiveKit, LiveKit SIP and Asterisk.
2. API `CreateSIPParticipant` answers extension 6000; SIP peer joins the room.
3. Python LiveKit SDK subscribes to the remote SIP audio track.
4. RMS/peak signal is recorded in memory and detects non-silent echo frames.
5. CI returns a green `run-private-media-loopback` job, containers are torn down.