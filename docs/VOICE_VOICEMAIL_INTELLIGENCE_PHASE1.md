# VeriFYD Circle — Voicemail Intelligence v1 (guarded prototype)

## Existing services reused
- Private attachment store (R2): `verifyd_voice_attachments`, content integrity, recipient membership.
- `audio_detector.analyze_audio`: **heuristic indicators of synthetic voice**, never voice identity proof.
- `mail_shield_scoring._content_penalty`: existing scam/urgency lexicon, adapted to voicemail transcripts.
- `verifyd_worker.py` and Redis RQ `verifyd` queue: analysis happens outside the request process.
- Existing `POST /trust-voice/blocked-users`: network block action (requires blocking flag).

## Phase 1 API (not activated by code merge)

Feature flag: `VERIFYD_VOICEMAIL_INTELLIGENCE_ENABLED` **defaults to 0**.

- `GET /trust-voice/voicemail/health`: availability; no sensitive content.
- `GET /trust-voice/voicemail/inbox-state`: the signed-in recipient's deleted IDs, statuses, compact verdicts.
- `POST /trust-voice/voicemail/{attachment_id}/review`: requires recipient's session and `{"consent_to_external_transcription":true}`, queues private RQ analysis. Repeated calls are idempotent while queued/completed.
- `GET /trust-voice/voicemail/{attachment_id}/review`: authenticated recipient-only full transcript and advisory.
- `DELETE /trust-voice/voicemail/{attachment_id}`: **removes from the recipient's inbox across devices** and clears transcript/risk data; does **not** physically delete sender's R2 recording.
- `POST /trust-voice/voicemail/{attachment_id}/restore`: restores recipient's inbox entry without restoring a removed transcript.

The existing source is `source_type: verifyd_network`; no phone number is collected from unverifiable claims. There is no public business upload endpoint yet.

Transcription: private source audio from R2 is converted to a temporary WAV, sent to the configured external transcription provider (`gpt-transcribe`, configurable via `VERIFYD_VOICEMAIL_TRANSCRIPTION_MODEL`). Temporary files are deleted. The worker stores the transcript and modest risk signals only for the recipient; raw R2 audio stays in its existing system.

## Red / blue / green semantics

- RED: multiple high-signal social-engineering requests or high-signal plus urgency/payment signs.
- BLUE: suspicious language, possible synthetic voice (low certainty), very short or untranscribable message.
- GREEN: no obvious text-based warnings **were found**. This does **not** verify the caller, prove audio is human, or guarantee safety.

No automatic blocking or deletion based on scores. Always show explanations. Never tell users to trust green as authenticated.

## Activation requirements

1. Review the actual signed-in authorization and R2 retrieval code with a real test account.
2. Validate RQ queue name and presence of `REDIS_URL`, `DATABASE_URL` and `OPENAI_API_KEY` in the same worker environment (without exposing secret values). Confirm `ffmpeg` and audio-detector dependencies.
3. Run CI risk/access tests and real disposable PostgreSQL integration tests for recipient-only reports, blocking, deleting during analysis, account deletion and sender copy preservation.
4. Upload the **full** `TrustVoiceCallVoicemailInbox.jsx` replacement into Horizons (no changes to `TrustVoiceBeta.jsx` required), run Vite build/lint and compare on PC/iOS/Android.
5. Update the privacy policy to clearly disclose optional third-party transcription and retention; obtain consent before audio leaves VeriFYD.
6. Set `VERIFYD_VOICEMAIL_INTELLIGENCE_ENABLED=1` only after verification; confirm health and recipient-only access. Start with test accounts before broad rollout.

## Future business/API product (design only — NOT a live endpoint)

Proposed `POST /v1/business/voicemail-analyses` accepts an authenticated **tenant-scoped** multipart audio upload, optional `caller_e164` as an **unverified claim**, `source_type` (e.g., `pstn`, `sip`, `verifyd_network`, `uploaded_file`), and an `Idempotency-Key`.
Respond `202 {analysis_id,status:"queued"}`, with
`GET /v1/business/voicemail-analyses/{analysis_id}` and signed webhook delivery.

Before any external business endpoint, require:
- Server-stored, hashed tenant API keys with rotation, scopes, revocation, per-tenant rate limits, spend quotas, billing and audit trails.
- Tenant-isolated private audio and report storage; never resolve user attachments from an external API key.
- Encrypted transit, defined audio/transcript retention policy, customer deletion and consent controls, and webhook HMAC signatures.
- Recording/voicemail-processing legal reviews for each jurisdiction and telecom integration.
- Treat caller ID numbers as unverified: caller ID can be spoofed. No identity guarantee or automatic account block across a network based solely on the number.
- Evaluated accuracy and calibration on short noisy voicemail samples across languages, including clear false-positive/negative messaging.
- Separate ingest gateway for PSTN/SIP/Twilio/Asterisk when authorized; do not widen Trust Voice's current verified-network calling model in Phase 1.

**Not included**: automatic voicemail analysis without prior user opt-in, real-time call recording, hard removal of messages from other users' accounts, PSTN calling, customer-facing enterprise billing/API keys.
