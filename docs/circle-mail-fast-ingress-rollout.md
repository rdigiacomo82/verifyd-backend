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

## Phase A2 — bounded IMAP IDLE in the existing cron (no new service)

For users who cannot provision the dedicated worker, a cost-aware interim
fast mode is available **inside the already-running Hostinger cron**. It has
been deployed with the default setting **off**. After successful regression
tests, enable on Render service `verifyd-circle-hostinger-inbound`:

```text
VERIFYD_CIRCLE_EMAIL_APEX_FAST_WINDOW_ENABLED=true
```

The cron keeps `*/5 * * * *` and its normal first batch unchanged. After
the first successful import, it holds a separate **read-only** Hostinger
IMAP IDLE connection and wakes on new-message notifications, with a 30-second
catch-up check to cover missed provider signals. It **always exits** before
the next five-minute UTC schedule boundary (40-second margin; maximum 235
seconds). A failed listener logs only a generic error type and exits; the
next cron run still imports outstanding mail. The existing verified original
recipient, sender limits, attachment validation, PostgreSQL cursor and
private R2 storage remain untouched.

**Important cost tradeoff:** Render charges cron by *active runtime*. This
mode can increase monthly cron spend compared with a ~30-second import
every five minutes. At a rate of $0.00016 per active minute, running for
~235 of every 300 seconds could incur approximately $5.40/month of cron
compute plus small execution overhead, versus the $1/month minimum for
brief runs. Actual charges vary by plan, runtime and billing period;
check the connected service's pricing before long-term use. It creates
no new Render service or separate worker charge.

To roll back immediately set the same flag to `false` and deploy; the
ordinary five-minute cron resumes. When a dedicated IMAP IDLE worker
becomes healthy, disable the cron fast-window flag to avoid duplicate
idle connections and unnecessary active runtime charges.

## Phase B — create the IDLE worker via its isolated Blueprint

The source file is **`render-circle-mail-idle.yaml`**, not the existing
`render.yaml`. The existing root `render.yaml` describes the main analysis
web backend and MUST NOT be used for creating the mail worker.

Hostinger has already advertised `IDLE` in several successful importer runs.
The worker is ready to be provisioned after Blueprint CI passes.

1. In Render Dashboard select **New → Blueprint**.
2. Select GitHub repository `rdigiacomo82/verifyd-backend`, branch `main`.
3. In **Blueprint Path**, enter `render-circle-mail-idle.yaml`.
4. Confirm the proposed changes list contains ONLY one new background worker:
   `verifyd-circle-mail-idle`, in Ohio, Starter plan. If it lists any existing
   web service, other worker, database, disk, or cron as changed: **do not deploy**.
5. The blueprint references existing cron env variables via `fromService`.
   Their secret values never appear in the YAML, chat or Git. Render must allow
   those references in this workspace; if it rejects a reference, fix the
   service reference **in Render**, never paste a secret into chat.
6. Deploy the Blueprint; the new Starter worker adds a separately billed,
   always-on service (check the exact price displayed in your Render account).
7. Verify logs show `Hostinger IDLE ingest status=ok` and later
   `Hostinger IDLE wakeup processed` when a NEW test email is sent. Confirm the
   matching registered member's Inbox receives the email, and one provider
   event ID was recorded, with no duplicate messages.

The worker sets `VERIFYD_CIRCLE_EMAIL_IDLE_EXPECTED_USER=circleinbound@vfvid.com`.
If its copied credentials unexpectedly refer to another mailbox such as
`support@vfvid.com`, it **refuses to start**. It uses a read-only IMAP
connection, and the importer's existing PostgreSQL advisory lock allows the
five-minute cron to continue serving as a fallback.

At first deployment the worker intentionally sets
`VERIFYD_CIRCLE_EMAIL_APEX_ATTACHMENT_CAPTURE_ENABLED=false`. This phase
improves mail delivery latency, NOT inbound attachment availability.

If Render refuses `fromService` links to the existing cron, stop and show
the configuration error (without any secret values). Avoid creating a
nonfunctional paid worker with blank credentials.

## Phase C — inbound attachment capture (OFF by default)

New **Hostinger** attachment capture is **not** malware scanning. It only validates allowed types and contents, limits MIME complexity and bytes (3 files; 8 MiB/file, 16 MiB total), stores them privately in the existing R2 bucket, and returns authenticated metadata. Downloads are **not** inline and require explicit `acknowledge_unscanned` on the existing endpoint. The UI must clearly display `not_scanned` / security notice.

Enable only after verifying:
1. Private R2 credentials and bucket access are configured **on the mail-ingesting service** (the cron and, if used, the listener).
2. End-to-end tests for harmless PDF/TXT file, rejected executable, oversized file, attachment spoof, duplicate event, authorized owner-only download, unscanned acknowledgement, and R2 upload failure pass.
3. Users understand these files are **not malware scanned**; a dedicated scanner is required before any automatic safe-download or preview policy.

Only after these checks should `VERIFYD_CIRCLE_EMAIL_APEX_ATTACHMENT_CAPTURE_ENABLED=true` be set on the ingesting service(s). Leave it false now. `VERIFYD_CIRCLE_EMAIL_INBOUND_ATTACHMENTS_ENABLED` is an **independent** Resend reply configuration, not an alias for Hostinger.

## Rollback

If a listener cannot stay connected, disable or suspend only its service; the cron continues importing. If attachment storage fails or threat validation cannot be verified, set `VERIFYD_CIRCLE_EMAIL_APEX_ATTACHMENT_CAPTURE_ENABLED=false` on importers; the existing withhold notice is restored. Do not reset UID validity, modify MX, or delete mail to recover.
