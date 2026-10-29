# VeriFYD Circle mail: inbound retrieval + attachments — staged integration

Status: isolated backend candidate; NOT merged or deployed. Full end-to-end verification required.

## Changes
- Incoming webhook retains authenticated Svix verification, opaque reply routing,
  sender matching, recipient scope, and idempotency. A dedicated Resend read key
  (`VERIFYD_CIRCLE_EMAIL_RECEIVING_API_KEY`) is preferred over `RESEND_API_KEY`.
- The received-email retrieval endpoint uses Full Access Resend API credentials.
  Read failures log **HTTP status only** (no credentials/content).
- Outbound external messages can attach up to three authenticated uploaded files,
  each max 8 MiB and together max 16 MiB (after binary encoding provider accepts
  the base64 payload); file content is re-read from private R2, SHA-256 verified,
  and revalidated before sending. The sender's Sent copy keeps the attachment.
- Internal Sent copies of existing attached mail expose sender-owned file links
  matched on the same thread, sender, and timestamp.
- Incoming replies: attachment metadata is retrieved from Resend, supported files
  are downloaded using a strict HTTPS host/path allowlist (no redirects), size
  bounded, byte-type validated and copied into private R2, with database linkage.
  Unsupported, oversize, and excess files are omitted with an explicit body note.
  Provider failures remain retryable.
- All inbound external attachments are **UNSCANNED**. Browser preview must be
  disabled; download is blocked with HTTP 409 until the user acknowledges the
  warning via `?acknowledge_unscanned=true`. Downloads force octet-stream,
  Content-Disposition attachment and nosniff. They are NOT represented as clean.
  An antivirus/malware scan should be integrated before broad release.

## IMPORTANT: Hosted Horizons frontend still needs to be synchronized
The React Horizons frontend is NOT stored in this backend repository. A current
full export is required for a safe direct code replacement. In Horizons, the
email composer must:
1. Use an accessible `<input type="file" multiple>` and show selected file names,
   sizes, removal controls, and upload errors. Allow max three supported files,
   8 MiB each, 16 MiB combined.
2. Upload each selected file with `POST /trust-voice/circle-email/files`,
   `multipart/form-data` field `file`, and the user's existing Bearer token.
   Do not manually set the Content-Type boundary for FormData.
3. Include returned `file_id` values as `file_ids` in
   `POST /trust-voice/circle-email/send`. Do not remove them for external
   addresses. Keep existing session, message, video, Eye Gaze and auth code.
4. Display `attachments` returned with Inbox, Sent, and individual messages,
   including `source`, `malware_scanned`, `security_notice` and size.
5. Download using authenticated
   `GET /trust-voice/circle-email/files/<file_id>/download`. For external
   attachments show a danger/warning acknowledgment before adding query
   `acknowledge_unscanned=true`; then consume the blob and offer a *download*.
   Do not render file inline, auto-download, or embed links as public URLs.
6. Respect a 409 `attachment_confirmation_required` response; never bypass
   warnings. Preserve disabled buttons, loading and error feedback.

## Configuration
- In Resend -> API Keys, create a **Full Access** key for the backend's inbound
  retrieval API (a Sending Access key can send but not retrieve received mail).
- On the intended Render workspace/service, create encrypted environment secret
  `VERIFYD_CIRCLE_EMAIL_RECEIVING_API_KEY`. Do not put keys in GitHub, Horizons,
  scripts, screenshots, or chat. Keep existing `RESEND_API_KEY` for sending.
- Other existing flags: `VERIFYD_CIRCLE_EMAIL_ENABLED`,
  `VERIFYD_CIRCLE_EMAIL_EXTERNAL_ENABLED`, `VERIFYD_CIRCLE_EMAIL_INBOUND_ENABLED`,
  `VERIFYD_CIRCLE_EMAIL_INBOUND_WEBHOOK_SECRET`, and private R2 credentials.
- `GET /trust-voice/circle-email/inbound/health` reports configuration existence,
  NOT remote key permissions or live delivery.

## Manual verification (staging first)
1. Confirm GitHub Actions syntax/contract checks pass on the repair branch.
2. Build/test with staging DB/R2/Resend keys and a real test account.
3. Verify incoming `Re: Official test email` via replay of the existing signed
   `email.received` event; expect HTTP 200 and one Inbox message only.
   Replay again; expect no duplicate.
4. Send supported small PDF, TXT and PNG to external Gmail; verify attachments
   arrive intact and appear on VeriFYD's Sent copy. Test internal recipient and
   sender Sent copy. Reject >8 MiB, >3 files, total >16 MiB, invalid mime content.
5. Reply from Gmail with PDF/PNG. Verify private R2 storage, metadata in Inbox,
   and HTTP 409 until warning acknowledgment. Test rejected executable, enormous
   file, invalid attachment CDN URL, duplicate event, failed provider retrieval.
6. Verify sender mismatch, wrong opaque alias, revoked route, and unknown event
   do not create an inbox entry. Verify auth and user-owned downloads.
7. Confirm no regressions in chats, calls, video, Eye Gaze, session persistence,
   and app release. Do not merge/publish before these checks.

No production change is made by this branch. Pending `rsend.reply` sending CNAME
does not block receiving a verified MX reply; resolve domain verification separately.
