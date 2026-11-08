# Native incoming-call contract — DRAFT / NOT DEPLOYED

A pure Python payload contract for future Android FCM and iOS APNs VoIP delivery. It is deliberately **not imported into the live FastAPI router or signaling handler**.

The production backend currently accepts authenticated WebSocket `call_invite`, `call_answer`, `call_decline` and `call_end` messages. Its `incoming_call` events use `tvcall_<32-hex>` IDs and a 30-second ringing window. This protocol formats `schema`, `kind`, `call_id`, deterministic `call_uuid`, `call_type` and original `expires_at`. No bearer tokens, location, chat messages, email addresses or unverified identity names are included.

**Do not activate native ringing based on this file.** No session-scoped token registration/revocation, FCM/APNs credential integration, APNs sandbox/production separation, call-state reconfirmation, native audio activation, or signed mobile build is included. There are no environment variable changes or new dependencies.

The user intends to update existing store apps *first*: Android `com.verifyd.app` and Apple's existing Video Authenticator (App Store listing 6760941429). Actual published native sources and signing configurations are not yet available in connected GitHub. Preserve existing Play upload certificate and Apple's actual signed bundle identity.

Validation: `python -m unittest discover -s tests -p test_native_call_contract.py -v`.
