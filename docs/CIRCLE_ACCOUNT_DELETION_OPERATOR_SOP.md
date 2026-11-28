# VeriFYD Circle permanent account deletion — operator SOP

**Purpose:** Complete, not merely acknowledge, a signed-in user's deletion request for Apple App Review guideline 5.1.1(v). This SOP is a release blocker until assigned to an authorized operator with a real schedule and access to all data stores.

## User self-service initiation
Settings → Account & Privacy → Delete Account → type DELETE →
POST /trust-voice/account-deletion with signed-in Trust Voice bearer.
The API returns 202 with a receipt, target completion within 30 days,
and explicit deletion_complete=false. Duplicate submissions keep the
same pending request. **Never display “Account deleted” on 202.**

## Operator process (real work, not a status toggle)
1. Verify that POST generated a durable queue record and notification.
   Confirm the operator mailbox (default support@vfvid.com) is monitored,
   and that VERIFYD_ACCOUNT_DELETION_OPERATOR_IDS is explicitly allowlisted.
   GET /trust-voice/account-deletion/operator/queue is operator-only.
2. Validate the request against the authenticated Trust Voice identity,
   review school-managed/student/guardian assignments, and establish
   whether any child-safety, education, legal, financial or abuse-investigation
   retention is actually required. Do not erase records that are legally
   required to be retained; restrict them and record the legal basis.
3. Build a **customer-specific deletion inventory** from the actual schemas.
   Review PostgreSQL: verifyd_identities, verifyd_voice_sessions, Trust Circle
   membership, conversations/message memberships, direct and group messages,
   profile/avatar media, voicemail, events/calendar, push/device registrations,
   per-user analytics/preferences, rank/game data, documents and any newly
   introduced modules. Inspect FK dependencies before changing rows.
   Review hosting and object stores: R2 private attachments, voicemail
   recordings, profile images, linked PocketBase user data, service accounts,
   cached tokens and backup retention. Do not falsely claim a database row
   deletion removes remote object bytes.
4. Revoke all Trust Voice sessions and push tokens, disable future notification
   delivery, and revoke linked Apple Sign In credentials if that sign-in
   feature is used. Remove or anonymize personal data across every store,
   preserving only minimum records explicitly required by law; secure any
   retained data and provide a privacy statement about it.
5. Check any Apple/PayPal subscription separately. Account deletion does
   not itself cancel Apple subscriptions; show the Apple subscription
   management link and remind the person to cancel if desired.
6. Conduct a second-person check against the inventory to confirm actual
   erasure and retention, including R2 and downstream processors. Record
   request ID, what was deleted, any lawful retention and when, without
   copying personal content into audit logs.
7. Send a final completion notice to the customer, preferably to the verified
   address associated with the original account request; track delivery.
8. ONLY AFTER steps 1-7, use the allowlisted
   POST /trust-voice/account-deletion/operator/confirm-completion with all
   four truthful attestations and an audit note. This endpoint **does not
   delete any user data**; it only marks previously completed work.

## App Review validation
Use a synthetic adult disposable test account, not an actual student/
guardian/medical user. Demonstrate full in-app initiation, 202 receipt,
operator completion, and final notice on a physical iPad/iPhone. Add
recording and exact taps to App Review Information. Keep a dedicated
operator/on-call contact and test the queue daily.

Do not merge or enable until the operator SOP, 30-day operational service
promise, backups/retention policy, and destructive-action privileges
have been independently reviewed. Do not run test deletion against real data.
