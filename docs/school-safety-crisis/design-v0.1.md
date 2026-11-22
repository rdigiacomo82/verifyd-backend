# VeriFYD VeriFYD School Safety — School Crisis Coordination v0.1
**Naming note: “SafeSignal” is already in use for emergency-alert software, so use “VeriFYD School Safety” as a descriptive working title until counsel reviews naming.**

**Status: design only; production feature OFF. Do not use for real emergency reporting.**

## Non-negotiable safeguards
1. Existing Trust Voice PWA incoming-call push, personal distress and 911 preview remain untouched. School crisis is an isolated, feature-flagged tenant-aware system.
2. Require enrollment in a **verified school** and scoped role-based authorization. Student selects two distinct approved Trust Circle contacts. Server adds the school-approved principal and response-team designees. No unauthenticated school-name selection.
3. Every report is persisted, durable, idempotent and acknowledged BEFORE a client shows "Received." On outage or offline show **NOT SENT**. Store per-report timestamps, actor, verification, incident category, updates, location accuracy and report delivery state.
4. Never drop the 11th or 100th report. Aggregate staff notification presentation for 10+ similar reports into one incident counter; preserve each record and escalate new critical facts (especially **I need help**). Staff notification acknowledgements, backup escalation, repeated delivery attempts and audit records mandatory.
5. Separate **student status** (need help / safe / situation improved) from **official incident resolution** (school safety team only, with signed audit). Student disengagement only closes their own report.
6. Only school-verified guardians see their linked child; no global roster access. Parent location view is emergency-only, authorized, audit-logged and constrained by school policy; indoor Chromebook location is **approximate**. Use self-reported school building/floor/room, display confidence and age of location.
7. A browser can suppress its own generated audio, not system-wide audio. Require actual managed Chromebook/iPad tests and school procedures; no automatic microphone/camera activation.
8. District FERPA, NY Education Law 2-d / Part 121, COPPA/child privacy, data-processing agreement, custody controls, encryption, retention/deletion, breach response, school safety plan review, accessibility and drills required.
9. Starlink is a future district-managed WAN failover option, not an app feature or solution for broken classroom Wi-Fi/power.
10. Live emergency routing, 911/Text-to-911, school staff and guardian notifications must remain DISABLED until authorized and tested end-to-end.

## School roles
- **Student:** School identity; primary/backup contact setup; 5s hold with accessible two-step alternative; submit category + optional room/location; message only after activation; mark own status; disengage own report.
- **Principal / designated school responders:** One-school roster, incident dashboard, official update/resolve, escalation, response acknowledgements; mandatory MFA.
- **Verified guardians:** Only linked child’s school-authorized incident status and last permitted location; no other students.
- **District safety:** Assigned schools only; responder roster, audit and escalation oversight.
- **Operations:** Aggregated health, not student PII by default.

## Minimal schema
`schools`, `school_memberships`, `guardian_links`, `school_response_assignments`, `student_trusted_recipients`, `crisis_incidents`, `crisis_reports`, `crisis_updates`, `crisis_delivery_attempts`, `crisis_access_audit`, `crisis_aggregation_groups`.

## Backend API (future; authenticated and school-scoped)
- `GET /trust-voice/crisis/enrollment` — server-verified roster and eligible contacts only.
- `PUT /trust-voice/crisis/student-recipients` — exactly two distinct eligible adults.
- `POST /trust-voice/crisis/reports` — client idempotency key; server derives school and recipients. Durable response includes delivery state; never return success before write.
- `POST /trust-voice/crisis/reports/{report_id}/updates` — allowed student updates after activation.
- `POST /trust-voice/crisis/reports/{report_id}/end` — personal report only, verified PIN/session.
- `GET /trust-voice/crisis/schools/{school_id}/incidents` — verified school response staff only.
- `POST /trust-voice/crisis/incidents/{incident_id}/official-updates` and `/resolve` — audited school responders.
- `GET /trust-voice/crisis/family/my-child/{child_id}` — verified guardian relationship + incident authorization each request.
- `GET /trust-voice/crisis/health` — sanitized, no PII or credentials.

## Alert aggregator design
- Assign incidents by school + type + related time/location, using a tentative correlation window; allow authorized staff to split/merge.
- Persist each report first, then asynchronously enqueue notifications. Sender gets backend report receipt separately from guardian/staff acknowledgement.
- Publish one staff incident notification on first report; at 10+ coalesce normal growth to badge/count and periodic digest. Do NOT suppress incoming records, urgent escalation, separate incidents or designated backup routing.
- Per-identity duplicate prevention + automation/bot abuse flags, with case review and safety exceptions. No blanket school emergency shutoff.
- Escalation if no staff acknowledgement; audit attempted delivery and failures.
- Training mode has isolated tenant/channel, bold drill marking, no live recipients.
- On an offline device, local queue is never represented as dispatched, and stale location is never represented as current.

## Release gates
Design review → signed school/district requirements and privacy approval → feature-OFF backend scaffold/test database → authorization & 10/11/100/1000 report tests → secure push delivery & school acknowledgement drills → managed Chromebook/iPad accessibility/offline/notification tests → school pilot sign-off → controlled opt-in deployment.

**Proposed flag:** `VERIFYD_SCHOOL SAFETY_ENABLED=0` by default.

## Frontend
VeriFYD School Safety Phase A Horizons preview (separate delivery): route `/crisis` + `/Crisis`, PWA identity `/crisis`, sample-only student/school/family screens and five-second activation. It intentionally has **no network calls** and does not send alerts. Existing `/voice` manifests, root call worker, and personal distress remain unchanged.
