# VeriFYD Circle — Hostinger catch-all receiving gateway activation

## Facts confirmed from actual email on October 30, 2026

Hostinger accepted a new Gmail/Outlook test addressed to:
`roberttest22@vfvid.com`.

The outermost delivery header contained:
```
Received: from mail-dl1-x1234.google.com
    by de-fra-smtpin6.hostinger.io (mx.hostinger.com) with ESMTPS ...
    for <roberttest22@vfvid.com>; Fri, 30 Oct 2026 15:56:55 +0000 (UTC)
```
This was delivered to Hostinger's domain catch-all mailbox,
`support@vfvid.com`. The recipient is therefore visible in the trusted
Hostinger SMTP hop. Two offline parser tests now cover its exact layout,
Outlook multipart/quoted-printable bodies, and forged-header rejection.

The account `@RobertTest22` was verified active with a verified email identity
by a read-only production identity query.

**This does not mean Circle mail is already imported.**
A single-instance scheduled IMAP job must still be provisioned and tested.

## Render cron job proposal

Render's official docs describe a minimum charge of $1/month per cron job;
active runtime is billed prorated by the second. Obtain account-holder approval
for a new paid service before creating it.

Proposed service name: `verifyd-circle-hostinger-inbound`.
Repository: `https://github.com/rdigiacomo82/verifyd-backend`.
Branch: `main`; runtime: Python 3.x; build: `pip install -r requirements.txt`;
start: `python trust_voice_circle_email_apex_gateway.py`.
Schedule: `*/5 * * * *` (every five minutes, UTC) for a quiet pilot.
Region: Virginia if feasible alongside the current PostgreSQL instance.
Use a single cron service. Do not modify the existing web service or workers.

The cron service needs the following environment variables, entered ONLY in
Render's secure Environment tab (never in GitHub, uploaded exports or chat):

- `DATABASE_URL`: the existing Verifyd Postgres connection, ideally supplied
  via an existing trusted Render environment group
- `VERIFYD_CIRCLE_EMAIL_PUBLIC_HANDLE_ENABLED=true`
- `VERIFYD_CIRCLE_EMAIL_APEX_GATEWAY_ENABLED=true`
- `VERIFYD_CIRCLE_EMAIL_HOSTINGER_IMAP_USER=support@vfvid.com`
- `VERIFYD_CIRCLE_EMAIL_HOSTINGER_IMAP_PASSWORD=<Hostinger app password>`
- `VERIFYD_CIRCLE_EMAIL_APEX_TRUSTED_RECEIVED_BY=de-fra-smtpin6.hostinger.io`
- `VERIFYD_CIRCLE_EMAIL_REPLY_DOMAIN=reply.vfvid.com`
- `VERIFYD_CIRCLE_EMAIL_RESERVED_LOCALPARTS=<comma-separated existing Hostinger
   localparts, including every operational alias that might conflict with a
   Circle handle>`

Hostinger's published IMAP endpoint is `imap.hostinger.com:993` with TLS
certificate validation. Prefer Hostinger's app password mechanism where
supported. It is critical to protect `support@vfvid.com`, which also receives
business correspondence; the gateway does not delete or mark messages as read.

`VERIFYD_CIRCLE_EMAIL_APEX_TRUSTED_RECEIVED_BY` is an exact hostname allowlist.
**Do not configure `*.hostinger.io`**, unverified MTA names, or assume all
Hostinger messages use the same ingress server. For another legitimate
Hostinger server, inspect a separate full original EML and add its exact
verified name. Messages from untrusted hops are quarantined, not misrouted.

**The existing website must NOT enable
`VERIFYD_CIRCLE_EMAIL_APEX_ADVERTISE_ENABLED` yet.**
Keep existing long Resend Reply-To addresses working until multiple real
apex email imports succeed and reply flow is verified.

## Live acceptance test

1. Obtain explicit permission for cron charges. Configure cron in Render.
2. Receive new plain-text test messages from Gmail and Outlook to
   `roberttest22@vfvid.com` and a second verified registered handle.
3. Inspect Render cron logs for `status=ok processed=N quarantined=M`.
4. Open the correct Circle Email Inbox for each user.
5. Confirm no cross-user delivery and no duplicate import on second cron run.
6. Send an email to an unregistered name, a role alias (`support@`), and a
   different named company mailbox; confirm none route into Circle.
7. Confirm a new email received via the existing long Reply-To address still
   arrives as before.
8. Confirm email body and notifications; file attachments for public apex
   mail are intentionally still withheld pending malware screening.
9. Only after receiving and other-account routing are proven, consider
   enabling public short Reply-To and exposing `handle@vfvid.com` in
   the web UI. This requires further frontend and backend validation.

## Limitations and rollback

The current Hostinger gateway implementation is a pilot-scale single mailbox,
not an independently verified high-volume SMTP service.
The Hostinger `support` mailbox's quota and volume constraints may become
inadequate with large network growth.

Do not expose IMAP secrets. On any delivery mismatch, disable the cron job or
set `VERIFYD_CIRCLE_EMAIL_APEX_GATEWAY_ENABLED=false`, without changing
apex MX records or disturbing the established Resend reply system.
