# VeriFYD Circle: one automatic @vfvid.com address per registered member

## Current status
**Code prepared and tested off-line; production gateway is NOT enabled.**
Do not advertise `handle@vfvid.com` until the Hostinger catch-all + trusted
envelope-recipient test has succeeded. No root MX or DNS record changes.

Every VeriFYD Circle account already has a unique canonical `handle_lower`.
The gateway looks up an active, email-verified identity on each message; no
Hostinger mailbox/alias/forwarder is created for individual members.

Example:
- `@RobertTest22` => `roberttest22@vfvid.com`
- `@AnnSLP` => `annslp@vfvid.com`

Existing `vfvid.com` host-mail MX remains:
`@ MX 5 mx1.hostinger.com`, `@ MX 10 mx2.hostinger.com`.
Resend continues to own `reply.vfvid.com`. Do not touch SES feedback MX at
`send.vfvid.com`.

## Catch-all gateway (single mailbox, no per-member provisioning)

Hostinger confirms one catch-all can send email addressed to **non-existent
Hostinger mailboxes** into a chosen existing Hostinger mailbox. Existing named
business mailboxes continue receiving directly. This is a pilot ingress for
every registered handle, but capacity/quotas may require an upgraded gateway.

1. In Hostinger hPanel, **Emails > Mailboxes > vfvid.com**.
2. Create one dedicated private *gateway mailbox*, for example
   `circle-inbound-gateway@vfvid.com`. This mailbox counts as one Hostinger
   mail account. Use a long unique password; do NOT paste it into ChatGPT.
3. Navigate to **Catch-all email** and choose that new gateway mailbox.
   Confirm via the verification message delivered to that mailbox.
   **Do not change the root-domain MX**.
4. Send a tiny test from Gmail to one new registered member such as
   `roberttest22@vfvid.com`, and one different registered member.
   They should appear in the one gateway mailbox.
5. Open the test email in Hostinger Mail and choose **More (...) > Show
   original**, download an `.eml`. Inspect the *topmost Received* hop.
   It MUST contain the original SMTP recipient as `for <handle@vfvid.com>`
   and name the trusted Hostinger mail server in its `by ...` field.
   If not, **STOP**: visible `To:`, `Delivered-To:`, and
   `X-Original-To:` are NOT independently trusted routing evidence.
   Obtain a trusted SMTP envelope recipient from Hostinger or use a direct
   domain-wide MX gateway instead. Do not weaken recipient validation.

The real `.eml` is a prerequisite to establish Hostinger's actual header
format. If sharing it for review, redact personal content and secrets.

## Backend gateway
New `trust_voice_circle_email_apex_gateway.py` uses TLS-verified Hostinger
IMAP (`imap.hostinger.com:993`) with one read-only INBOX selection. Run it as
a **separate single-instance Render cron job**, NOT inside existing
DINOv2 / RQ background workers or the FastAPI web-service process.

Command:
`python trust_voice_circle_email_apex_gateway.py`

Recommended schedule: once per minute, subject to available Render plan
and mail volume. Dependencies are Python standard library and existing DB
modules in `requirements.txt`. Do not create a paid Render cron until
cost/plan approved.

Its features:
- disabled by default
- exact trusted outermost Received hop checks (not visible To)
- no reliance on sender-controlled mail headers
- recipient must equal one valid canonical `<handle>@vfvid.com`
- account lookup from actual `verifyd_identities` table, active + verified
- sender/owner anti-abuse rates, spam classification
- no public-attachment content downloads until scanning is ready
- dedupe with provider ID keyed by IMAP mailbox+UIDVALIDITY+UID
- transaction-serialized poller, durable UID watermark, quarantine metadata
- bounded 30-message poll batches, 18MiB per mail
- old Resend opaque reply routing continues unchanged

## Required Render cron environment (set in dashboard only)
These must be set on the dedicated cron service. Do not paste secrets here:
`VERIFYD_CIRCLE_EMAIL_PUBLIC_HANDLE_ENABLED=true`
`VERIFYD_CIRCLE_EMAIL_APEX_GATEWAY_ENABLED=true`
`VERIFYD_CIRCLE_EMAIL_HOSTINGER_IMAP_USER=<gateway mailbox @vfvid.com>`
`VERIFYD_CIRCLE_EMAIL_HOSTINGER_IMAP_PASSWORD=<secret>`
`VERIFYD_CIRCLE_EMAIL_APEX_TRUSTED_RECEIVED_BY=<exact server hostname from .eml>`
Plus existing secure DB environment/config via appropriate Render environment
group (not a copy of credentials in source). Keep cron service at ONE replica.
Set `VERIFYD_CIRCLE_EMAIL_RESERVED_LOCALPARTS` for existing Hostinger
business-mail localparts, comma-separated. Do not use this as a member list.

The backend web service also needs `VERIFYD_CIRCLE_EMAIL_PUBLIC_HANDLE_ENABLED=true`
to accept short aliases at `reply.vfvid.com` and show them in account details.

Do NOT set `VERIFYD_CIRCLE_EMAIL_APEX_ADVERTISE_ENABLED=true` on the web
service until at least two real `@vfvid.com` deliveries succeed and inbound
replies pass end-to-end validation. That flag changes *new* external Reply-To
headers from token addresses to `handle@vfvid.com`.

## Pilot acceptance checks
- Test two distinct registered handles from Gmail and Outlook.
- Verify they land in distinct Circle Inboxes, not the same account.
- Verify existing Hostinger business mailbox remains untouched.
- Verify unregistered/role local parts never resolve to a VeriFYD account.
- Test forged `To`, `X-Original-To`, and fake older `Received` headers.
- Test retries and re-run cron; exactly one imported message.
- Test a reply after new outbound message and confirm routing.
- Test spam and high-volume limits before broad enablement.
- Verify unread notifications, attachment withholding, and Message-ID threading.
- Document mail retention, account suspension, abuse reporting, opt-out.

## Important production caveats
- One catch-all mailbox may receive substantial spam and hit limits. This is
  a **bootstrap**, not a validated unbounded high-volume gateway.
- If Hostinger does not supply a secure original-envelope trace, the IMAP
  gateway intentionally quarantines; DO NOT route by untrusted `To:`.
- Long term: use provider-managed SMTP split delivery that preserves existing
  Hostinger business mailboxes and delivers member aliases via a scalable
  ingress. That will need provider design and staged MX migration.
- The activation flag does not alone configure Hostinger. Never state that
  arbitrary `@vfvid.com` mail works before the live test.
- Turning off the web-service feature after advertising public addresses
  may stop future replies; deploy a safe inbound-accept compatibility policy
  before using a production kill switch.

## Rollback
Stop Cron and disable apex advertisement first. Keep
`reply.vfvid.com` routes and standard outbound sending intact. Revisit
quarantine records and the catch-all mailbox for any undelivered messages.
Do not delete the dedicated mailbox before preserving correspondence.
