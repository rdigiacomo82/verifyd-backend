# VeriFYD Circle E911 Enrollment — backend STAGING, not a telephone service

**Do not advertise or activate 911/E911.** No carrier, emergency services provider,
PSAP route, callback DID, dynamic location, SMS provider or real 933 tests are configured
by this code. Do not dial actual 911 for testing. This is an independent staging proposal
based on production `main`; it does not merge experimental LiveKit/SIP lab code.

## What the isolated router does

- `GET /trust-voice/emergency-services/status`: requires existing revocable Circle bearer token.
  Always returns `can_place_911_call:false`, `provider_connected:false`, with
  either `not_available`, `not_registered` or `verification_pending` enrollment state.
- `POST /trust-voice/emergency-services/enrollment`: only when explicit staging flag AND
  independent Fernet encryption key are set; requires US street address, validated
  US state/ZIP/phone format, explicit data-storage consent and limitations acknowledgment.
  Encrypts full address and mobile at application level in an isolated PostgreSQL table,
  keyed by internal Circle identity ID. Only last four mobile digits are retrievable.
  Response is **pending only**; not verified, not provisioned, not an emergency line.
- `DELETE /trust-voice/emergency-services/enrollment`: lets authorized account remove entry.
- `POST /trust-voice/emergency-services/mobile/request-code`: always 503 until an authorized SMS provider and verified ownership process exist.
- `POST /trust-voice/emergency-services/calls/911`: always 503; this app cannot place 911 calls.
- Schema lazy-creates **only after staging is explicitly enabled and an encryption key exists**.
- No logging of user address, full mobile number or encryption key.

## Configuration — DO NOT ENABLE in live production today

New dependencies: `cryptography>=42,<47`, needed only to handle encrypted stage enrollment.

`VERIFYD_E911_ENROLLMENT_STAGING_ENABLED` default unset/off. Enable **only in an isolated private staging deployment** after legal/privacy review, access controls, data retention and monitoring are approved.

`VERIFYD_E911_PROFILE_FERNET_KEY` must be a separate valid Fernet key, generated with
`from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())`.
Never reuse Circle session secrets, LiveKit credentials or email keys. Store key only in
secret manager/environment, maintain backups and key-rotation procedure. Do not echo secrets
into shell history or commit them to GitHub.

No other setting can turn on 911 dialing; service always reports unavailable.

## Frontend contract

The 2026-11-06 23:05 Horizons export has `TrustVoiceEmergencyEnrollment.jsx` in
**preview only** mode, with **no fetch** to the backend. Therefore even a code merge
would **not** create functioning registration or verification. A subsequent reviewed
frontend change is needed to submit under confirmed consent and display errors/status.
Keep existing 911 popup warning and local 911 shortcut behavior unchanged.

## Release gates before any account enrollment

1. Security review for sensitive PII and data retention/privacy notice. Improve rate limiting
   and audit/observability as appropriate, and establish encryption key custody and rotation.
2. Decide E911 provider architecture for WebRTC/mobile/nomadic caller location and acceptable
   routing modes; obtain contract and interoperable SIP endpoint, real provider DID and PSAP
   callback path. **Verified mobile number is not automatically an emergency callback number.**
3. Implement SMS verification with provider-side status, resend throttling, abuse detection,
   retry limits, and a phone number reassignment strategy.
4. Integrate emergency address provider validation, including unit/floor/location changes.
5. Verify signed Circle session revocation, access controls, location freshness, callback
   handling, media failure, accessibility and realistic device offline behavior.
6. Use authorized provider-specific 933 readback **only after** registered callback,
   dispatchable location and testing arrangements exist; never routine 911 test calling.
7. Professional telecommunications/compliance review of applicable rules, notices and
   whether service must provide 911 to subscribers as a condition of service.

**Keep GitHub PR as DRAFT. Do not merge, deploy to production, or activate the 911 button.**