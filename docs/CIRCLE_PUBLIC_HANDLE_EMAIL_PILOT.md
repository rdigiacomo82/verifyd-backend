# VeriFYD Circle — Short Public Email Addresses (opt-in)

## Goal
Within Circle, users keep emailing `@RobertTest22` via internal handle routing.
Outside Circle, Gmail/Outlook require a full email address:
`roberttest22@reply.vfvid.com`. This phase DOES NOT route
`roberttest22@vfvid.com` or the bare `@roberttest22` from external clients.

## How it works
- The existing `reply.vfvid.com` MX and signed `email.received` webhook are reused;
  no Hostinger DNS modifications.
- Only enabled/explicitly opted-in handles accept unsolicited external mail.
- New outbound emails for opted-in handles set `Reply-To` to the public short
  address. Previous token-based addresses are still recognized unchanged.
- All received content still requires a valid Resend webhook signature and is
  fetched from Resend using the API. Exact single-recipient address + active,
  email-verified identity are required.
- The only accepted public local part is an exact approved handle. No wildcard
  `*`, no support/admin/postmaster addresses, no plus or arbitrary aliases.
- Rate limits: at most 8 external received messages per sender/recipient in any
  24-hour window, and at most 60 external messages total per recipient/day.
  A blocked sender is not told which identities exist (webhook returns success).
- Suspicious content routes to Spam. All external identity is unverified.
- **Public-address attachments are withheld** until a separate malware-scanned
  intake process is validated; existing token reply attachments stay controlled
  by their independent feature flag.
- **Mail sent with a permanent Reply-To may be imported as a new thread rather
  than the original token-address conversation.** Standards-based Message-ID
  threading must be validated separately before promising exact threading.
- No root-domain MX modifications, no app-store or other Circle feature changes.

## Production flags — disabled by default
- `VERIFYD_CIRCLE_EMAIL_PUBLIC_HANDLE_ENABLED=true`
- `VERIFYD_CIRCLE_EMAIL_PUBLIC_HANDLE_ALLOWLIST=roberttest22`

Do NOT enable all accounts by default. Each member should explicitly opt in
before being added to the allowlist. Do not use a wildcard. Leave the existing
`VERIFYD_CIRCLE_EMAIL_INBOUND_ENABLED` and Resend configuration intact.

## Optional authenticated account detail endpoint
`GET /trust-voice/circle-email/my-address` with the existing Bearer token
returns `internal_handle` and a short `public_email_address` only when
enabled for that user. Never include another account's address in this endpoint.

## Release / rollback
1. Verify syntax, existing reply tests and public-mail security tests.
2. Merge gated code without changing existing env flags. Check Render deployment
   succeeds and signed legacy replies still reach the Circle Inbox.
3. Set BOTH flags for one opted-in handle only, and check
   `GET /trust-voice/circle-email/inbound/health` reports public mode enabled.
4. Send a NEW Gmail message to `roberttest22@reply.vfvid.com`, verify
   Resend `email.received` delivered HTTP 200 and a single inbox record.
   Replay webhook twice and verify idempotency.
5. Send a new Circle email outward; verify Gmail sees a short Reply-To, reply
   to it and confirm a received message in Circle.
6. Test unknown handle, wrong domain, malformed plus address, multiple
   recipients, high-volume attempts, suspicious message, attachments withheld.
7. If problems occur, disabling public sending/receiving flags stops new direct
   deliveries but may also block replies to already-sent public Reply-To
   addresses. Prefer reverting only new Reply-To *generation* in code while
   continuing to accept already-issued public address mail.

## Remaining later work
- Self-service per-user opt-in/out and account UI for the short address.
- RFC Message-ID / In-Reply-To threading to reliably associate public-address
  replies with the original outbound thread.
- Stronger spam infrastructure, abuse reporting, quarantine and malware scanning.
- A dedicated root-domain `@vfvid.com` mailbox service (only after reviewing
  existing Hostinger mail/MX).
