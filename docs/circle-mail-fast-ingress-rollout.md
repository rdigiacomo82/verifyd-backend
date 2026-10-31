# VeriFYD Circle — fast mail intake and safe attachments

## Existing production (DO NOT remove or replace)

* Hostinger domain MX remains unchanged. The existing catch-all mailbox is **circleinbound@vfvid.com**. **support@vfvid.com** remains its own mailbox.
* Existing Render cron `verifyd-circle-hostinger-inbound` runs every five minutes and is a **backup** even after an always-on listener is activated.
* Resend receives `reply.vfvid.com` transactional replies through the signed Resend webhook. Never change apex MX to Resend as part of this rollout.
* All account routing uses the original Hostinger topmost `Received: ... for <member@vfvid.com>`, registered verified identities, DB transaction lock/UID cursor, and a deduplicating provider event ID.

## Phase A — deployment and provider capability probe (safe)

This commit is inert by default. Merge only after CI passes.
On the existing cron service set:

```
VERIFYD_CIRCLE_EMAIL_IDLE_PROBE_ENABLED=true
```

Observe the next successful cron run for `Hostinger IMAP IDLE advertised=True`. Do **not** create a new always-on service if Hostinger does not advertise IDLE.

For faster/far smaller future builds, set **only the existing cron's build command** to:

```bash
pip install -r requirements-circle-mail.txt
```

Do not change its start command, schedule, database credentials, recipient flags or IMAP credentials. Run a manual *new incoming email* test after each change. The CI's Python 3.14 job imports/tests the mail stack from this reduced dependency file; full `requirements.txt` remains untouched for detection workers.

## Phase B — fast IMAP listener (requires explicit new paid worker)

If provider IDLE capability is confirmed, add an **always-on Render Background Worker** with:
* Name: `verifyd-circle-mail-idle`
* Repo: `rdigiacomo82/verifyd-backend`, main
* Runtime: **Python 3.14+**; same Ohio region as database
* Plan: Starter (512 MB, ordinarily about $7/month; check account billing)
* Build command: `pip install -r requirements-circle-mail.txt`
* Start command: `python trust_voice_circle_email_idle.py`
* Environment values: the same private `DATABASE_URL`, `VERIFYD_CIRCLE_EMAIL_HOSTINGER_IMAP_USER`, `VERIFYD_CIRCLE_EMAIL_HOSTINGER_IMAP_PASSWORD`, `VERIFYD_CIRCLE_EMAIL_PUBLIC_HANDLE_ENABLED`, `VERIFYD_CIRCLE_EMAIL_APEX_GATEWAY_ENABLED`, `VERIFYD_CIRCLE_EMAIL_APEX_TRUSTED_RECEIVED_BY`, `VERIFYD_CIRCLE_EMAIL_APEX_ROTATING_MX_ENABLED`, any configured reserved local parts and `VERIFYD_CIRCLE_EMAIL_REPLY_DOMAIN` from the cron. Set `VERIFYD_CIRCLE_EMAIL_IDLE_ENABLED=true` **only on this worker**. Store secrets only in Render, never in docs, chat or Git.
* Keep the original cron running as fallback. Both call the same ingest function and serialize per mailbox in PostgreSQL.
* Validate that `Hostinger IDLE wakeup processed` appears promptly on a new test email, and no duplicate `provider_email_id` events exist. Then verify messages arrive in the correct member's inbox.

Do not re-use any analysis/document detection worker for this purpose. Don't run it as a sleeping Free web service; a continuous connection needs an always-on worker.

## Phase C — inbound attachment capture (OFF by default)

New **Hostinger** attachment capture is **not** malware scanning. It only validates allowed types and contents, limits MIME complexity and bytes (3 files; 8 MiB/file, 16 MiB total), stores them privately in the existing R2 bucket, and returns authenticated metadata. Downloads are **not** inline and require explicit `acknowledge_unscanned` on the existing endpoint. The UI must clearly display `not_scanned` / security notice.

Enable only after verifying:
1. Private R2 credentials and bucket access are configured **on the mail-ingesting service** (the cron and, if used, the listener).
2. End-to-end tests for harmless PDF/TXT file, rejected executable, oversized file, attachment spoof, duplicate event, authorized owner-only download, unscanned acknowledgement, and R2 upload failure pass.
3. Users understand these files are **not malware scanned**; a dedicated scanner is required before any automatic safe-download or preview policy.

Only after these checks should `VERIFYD_CIRCLE_EMAIL_APEX_ATTACHMENT_CAPTURE_ENABLED=true` be set on the ingesting service(s). Leave it false now. `VERIFYD_CIRCLE_EMAIL_INBOUND_ATTACHMENTS_ENABLED` is an **independent** Resend reply configuration, not an alias for Hostinger.

## Rollback

If a listener cannot stay connected, disable or suspend only its service; the cron continues importing. If attachment storage fails or threat validation cannot be verified, set `VERIFYD_CIRCLE_EMAIL_APEX_ATTACHMENT_CAPTURE_ENABLED=false` on importers; the existing withhold notice is restored. Do not reset UID validity, modify MX, or delete mail to recover.
