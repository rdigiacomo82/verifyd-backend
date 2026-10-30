# VeriFYD Circle: staged Resend apex receiving

## Current production state (do not disable)
- Hostinger remains MX authority for `vfvid.com`.
- Catch-all `*@vfvid.com -> circleinbound@vfvid.com` remains active.
- Render `verifyd-circle-hostinger-inbound` IMAP cron job imports this mailbox every five minutes.
- `reply.vfvid.com` already uses Resend receiving and the authenticated `email.received` webhook.
- `support@vfvid.com` and the other operational aliases remain on Hostinger.

## Staged implementation
- The signed Resend webhook reads canonical `to` recipients from Resend's receiving API, not the visible message header.
- `VERIFYD_CIRCLE_EMAIL_RESEND_APEX_ENABLED` defaults OFF, independently of the existing reply-subdomain route.
- With both this and `VERIFYD_CIRCLE_EMAIL_PUBLIC_HANDLE_ENABLED` ON, exactly one eligible handle at `@vfvid.com` can be saved to the registered active/verified identity.
- Operational local parts (especially `support`) and configurable `VERIFYD_CIRCLE_EMAIL_RESERVED_LOCALPARTS` cannot become member inboxes.
- Existing duplicate-event checks, spam risk classification, per-user rate limits, and sender restrictions still apply.
- `/trust-voice/circle-email/inbound/health` reports the new gate; this is NOT DNS or provider-permission verification.

## Blockers to a production apex MX cutover
1. **Protect operational mail.** Do not move apex MX until `support@vfvid.com` and all other live Hostinger mailboxes/aliases have a fully tested Resend-to-operator delivery path that preserves normal Outlook access. A Resend webhook will not automatically insert mail into a Hostinger mailbox.
2. **DNS authority.** Verify Resend receiving is enabled and domain `vfvid.com` is configured for inbound MX. Its sending/return-path MX (e.g. `send.vfvid.com`) is separate from actual apex inbound MX.
3. **Test signed event & provider GET.** Verify callback at `https://verifyd-backend.onrender.com/trust-voice/circle-email/webhooks/resend-inbound`, signature check, retrieval permission and authoritative recipient.
4. **Test normal and adversarial paths.** Gmail->`roberttest22@vfvid.com` to the correct Circle inbox; `support@vfvid.com` to Outlook; unregistered and reserved aliases do not expose identities; multi-recipient events do not cause cross-account delivery; test `reply.vfvid.com` replies.
5. **Deduplication during handover.** Ensure only one receiver is authoritative and decide how pre-cutover Hostinger mail is backfilled. Do not switch off cron before the new route is verified.
6. **Observe/revert.** Keep existing Hostinger mailbox and cron recoverable. After cutover, review webhook failures, provider event IDs, backend delivery counts and security notices; document an exact DNS rollback.

Until those conditions are met, keep `VERIFYD_CIRCLE_EMAIL_RESEND_APEX_ENABLED` OFF and do not alter the Hostinger catch-all, DNS MX records, or production cron job.
