# VeriFYD Circle: recover private inbound email attachments

## Why files were withheld

The existing Resend webhook correctly imports email bodies from replies sent to
`<handle>+<128-bit-token>@reply.vfvid.com` but leaves received attachment
files out of the account when
`VERIFYD_CIRCLE_EMAIL_INBOUND_ATTACHMENTS_ENABLED` is disabled. It puts a
"1 external attachment(s) withheld" notice in the mail body.

This rollout adds member-initiated recovery for those **already received**
private replies, without globally enabling automatic incoming file downloads.

## New authenticated endpoint

`POST /trust-voice/circle-email/messages/{mail_id}/attachments/recover`

Body: `{"acknowledge_unscanned":true}`

Only active account owners with a valid persistent session may call it.
Eligibility requires the incoming message to have been received via an
existing authenticated Resend webhook **and** match a private, non-revoked
stored reply route for the exact external sender. Unsolicited public-handle
mail is excluded. No recipient can be chosen by user-provided headers or
provider email IDs. Another member's message returns 404.

Before any provider network request, the endpoint requires explicit
confirmation that **external files are not malware-scanned**. It then looks up
Resend's received-message provider ID from the trusted event ledger,
lists provider attachment metadata and securely downloads allowable files
into account-scoped private R2 storage. The frontend never sees provider
download URLs. Subsequent requests return existing files without duplicating
objects. The old email body and thread are preserved.

Limits remain:
- Up to 3 files, max 8 MiB each / 16 MiB combined
- Only allowlisted extensions and byte-validated content
- Private R2 objects and account-scoped download authorization
- External files marked `source=external`, `scan_status=not_scanned`
- Download requires `acknowledge_unscanned=true`; browser receives a download,
  not an inline preview or an executable response
- Emails remain available when provider files have expired

## Rollout switches

**Feature off by default.** Only turn on in Render after confirming
receiving API key permissions, signed CDN host and R2 integrity:

`VERIFYD_CIRCLE_EMAIL_INBOUND_ATTACHMENT_RECOVERY_ENABLED=true`

If the live Resend attachment `download_url` hostname differs from the
built-in `inbound-cdn.resend.com`, verify it directly in the authenticated
Resend Receiving API response and set **only the exact approved hostname**:

`VERIFYD_CIRCLE_EMAIL_RECEIVED_ATTACHMENT_CDN_HOST=<verified-exact-cdn-hostname>`

Do not configure an arbitrary hostname, URL, wildcard or IP. The code rejects
HTTP, redirects, unsigned URLs, local/IP names and unexpected hosts.
Use the first non-sensitive URL hostname only; never paste signed URLs or
Resend API keys in support conversations.

The pre-existing
`VERIFYD_CIRCLE_EMAIL_INBOUND_ATTACHMENTS_ENABLED` flag controls automatic
file intake from *new* private replies. Leave it off during this pilot.
Do not enable unsolicited public-handle attachment processing before a
separate malware scanning/abuse policy is ready.

## Acceptance test and validation

1. Deploy new code with new recovery flag off. Run GitHub regression suite.
2. Confirm the Resend **receiving** key works; use the separate full-access
   receiving key if the outbound key is send-only.
3. Ask a test sender to reply with one benign .txt or .pdf to an already
   working opaque `@reply.vfvid.com` Reply-To address.
4. Enable the recovery flag for a controlled test after provider CDN validation.
5. In the **new Horizons email component**, open the existing message and
   select *Recover attachments privately*; accept warning.
6. Verify correct file appears only for the owning account, download requires
   second confirmation, and repeated recovery does not create duplicate files.
7. Confirm other users, unsolicited public mail, oversized/unsupported formats,
   wrong provider keys, expired provider URLs and redirects remain blocked.
8. Confirm messaging, Eye Gaze, video, OTP/session and all other Circle mail
   functions remain unaffected. Disable flag to roll back recovery.

Existing withheld messages may no longer have provider files available if the
provider has expired or purged them; a new test reply can confirm the pipeline.

## Separate issue: @vfvid.com SMTP rejection

This feature **does not** repair Hostinger's
`550 5.1.1 User doesn't exist` response at the apex domain. Hostinger's
incoming SMTP/LMTP service rejects unknown recipients before the backend.
The user's hPanel screenshot shows an existing catch-all assigned to
`support@vfvid.com`, so support should investigate recipient table
propagation, alias delivery, catch-all verification status and SMTP/LMTP
recipient mapping on `fr-int-smtpin28.hostinger.io`.

Do not change apex MX or enable apex public Reply-To advertisement until the
receiving gateway is working end-to-end. Existing `reply.vfvid.com` replies
must remain functional.
