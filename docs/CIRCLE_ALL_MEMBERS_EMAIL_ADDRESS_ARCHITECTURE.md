# VeriFYD Circle — Every member gets a handle-based email address

## Requirement
This is **NOT** a `roberttest22` feature or a list of manually provisioned
mailboxes. One identity handle maps deterministically to one public address:
`@RobertTest22` → `roberttest22@vfvid.com`,
`@AnnSLP` → `annslp@vfvid.com`.

An identity's `handle_lower` is unique in `verifyd_identities`. When a new
account registers, the mapping is automatic, without creating a Hostinger
mailbox, Resend receiving route, or per-user environment variable.

## What this PR implements
- A generic dynamic handle recipient router backed by current active, verified
  identity records.
- No per-account allowlist; one global *off-by-default* rollout gate
  `VERIFYD_CIRCLE_EMAIL_PUBLIC_HANDLE_ENABLED`.
- Exact-address parsing, reserved operational names and optional
  `VERIFYD_CIRCLE_EMAIL_RESERVED_LOCALPARTS` for existing Hostinger role
  mailboxes (not for member lists).
- Per-sender and per-recipient rate limits; unverified external senders always
  flagged; suspect mail to Spam; idempotent Resend event ledger.
- Historic per-message secret reply aliases still accepted.
- Current **tested receiving domain** is `reply.vfvid.com`. When globally
  enabled, every eligible handle can receive `<handle>@reply.vfvid.com`.
  The new Reply-To for each member changes to this short address.
- Authenticated endpoint `GET /trust-voice/circle-email/my-address`
  exposes the current member's short address when the feature is active.
- Public-inbox attachment bytes remain withheld pending malware scan capability.

## Requirements before advertising @vfvid.com addresses
**THIS PR DOES NOT IMPLEMENT @vfvid.com SMTP DELIVERY.**
Current apex `vfvid.com` MX still points to Hostinger:
- `@ MX 5 mx1.hostinger.com`
- `@ MX 10 mx2.hostinger.com`

The verified `reply.vfvid.com` Resend inbound MX must also stay unchanged.
The `send.vfvid.com` SES feedback MX is unrelated and must remain unchanged.

Hostinger confirms one catch-all mailbox per domain can collect mail sent to
nonexistent `@vfvid.com` recipients. It can be piloted as a gateway **without
per-user forwarders**, while existing named Hostinger mailboxes stay there.
But it MUST NOT be presented as finished until we establish:
1. Capture to one gateway mailbox for test `<handle>@vfvid.com` recipients.
2. Obtain the **authoritative original SMTP envelope recipient** from the
   trusted receiving gateway (not sender-provided `To:` headers alone).
3. Enforce exact alias resolution, no role-address collisions and validate
   unknown users are not stored.
4. Authenticate the gateway-to-backend transport (IMAP or signed ingest);
   honor provider authentication outcomes, validate message size/attachments.
5. Test DMARC/SPF forwarding behavior; Hostinger warns forwarded mail can fail.
6. Check mailbox and catch-all quotas, retry queues, deduplication and spam
   defense under expected network growth.

For substantial scale, adopt a provider-managed **domain-wide inbound
gateway with recipient-aware split routing**: deliver existing company mailboxes
to Hostinger, deliver member handles directly to the VeriFYD backend. Changing
root MX may be warranted *only after* this split-delivery integration is
verified with a rollback plan. Do not add a competing Resend MX at `@`; it
may hijack existing Hostinger company mail.

## Rollout controls
1. CI tests for every-account dynamic mapping and existing Circle features.
2. Deploy code with `VERIFYD_CIRCLE_EMAIL_PUBLIC_HANDLE_ENABLED` **unset**.
3. Validate safe gateway and consent/account notices, reserved existing email
   addresses, actual incoming public mail, thread handling, rate limits and
   rollback against staging.
4. After all tests and provider setup, turn on global mail routing. A new
   registered handle automatically receives its public alias, with no manual
   per-user tasks. Provide user-level spam, privacy and block controls.

## Known pre-release limitations
- Current direct mail at `<handle>@reply.vfvid.com` may arrive as a new
  thread (message-ID threading still to be implemented).
- The final `<handle>@vfvid.com` delivery gateway remains unimplemented.
- Global feature is intentionally disabled until infrastructure is ready.
- Receiving unsolicited mail for every registered handle requires proactive
  spam, abuse, retention, throttling and opt-out design before broad launch.
