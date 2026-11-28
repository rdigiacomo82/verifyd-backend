# VeriFYD Circle Apple StoreKit subscriptions — review-gated

**This is a staging PR, not a deployed subscription service.** No production entitlements have been modified. Keep the feature flags OFF until staging, signed-sandbox testing, operational notifications, and transaction reconciliation are verified.

## Identity
- iOS bundle ID: com.verifyd.circle
- App Store Connect app ID: 6826285833
- Auto-renewing monthly subscription product ID: com.verifyd.circle.monthly
- Apple price configured in App Store Connect; UI reads localized StoreKit price.

## HTTP contract (hosted https://verifyd-backend.onrender.com)
- Authenticated GET /trust-voice/apple-iap/config: {product_id, bundle_id, verification_ready}. The hosted paywall refuses to charge if verification_ready=false.
- Authenticated POST /trust-voice/apple-iap/verify: {"transaction_jws":"Apple compact signed JWT"}; returns entitlement_active only after Apple-signed, server-verified purchase is persisted to the authenticated user's identity. One Apple original transaction cannot be attached to two Circle accounts.
- Authenticated GET /trust-voice/apple-iap/status: current server-derived entitlement.
- Public POST /trust-voice/apple-iap/notifications accepts only cryptographically verified App Store Server Notifications V2 with a signed signedPayload. Embedded transaction and renewal are independently verified.

## Configuration — NONE set automatically by this PR
- VERIFYD_CIRCLE_APPLE_IAP_ENABLED=1: master enable
- VERIFYD_CIRCLE_APPLE_IAP_NOTIFICATIONS_ENABLED=1: require signed V2 endpoint readiness
- VERIFYD_CIRCLE_APPLE_ROOT_CERT_PATHS=/etc/secrets/apple-root-1.der,/etc/secrets/apple-root-2.der: local DER Apple PKI root certificates, secured via Render secret files. Obtain only from Apple PKI; do not commit or paste certs/secrets.
- VERIFYD_CIRCLE_APPLE_IAP_SANDBOX_ACCOUNT_IDS=<id1>,<id2>: trusted backend identity IDs, allowlisted disposable sandbox test identities. Sandbox receipts cannot activate an arbitrary production identity.
- app-store-server-library==3.1.2 is pinned to Apple's patched official verifier (fixes GHSA-8f6j-263m-g72x OCSP certificate status freshness flaw).

## Required operational work before enabling
1. Set up staging backend and separate PostgreSQL (no live customer receipts); provide Apple roots as Render secret files; verify HTTPS, CORS and authentication.
2. Confirm Apple Paid Apps Agreement In Effect, subscription localization and review metadata. Existing product status is Developer Rejected; update and add product to next App Review submission.
3. Configure App Store Server Notifications V2 with the exact HTTPS endpoint. Test sandbox V2 delivery, refunds, renewals, expiration and family sharing removal.
4. Add scheduled App Store Server API reconciliation (not yet implemented). Apple can retry notifications but one lost event can otherwise leave stale entitlement; do not assert production readiness solely from webhook processing.
5. Test duplicate transaction replay, ownership conflicts, privilege escalation, Sandbox cannot unlock non-allowlisted users, bad JWS, and refunds. Review history and billing-retry/grace-period policy.
6. Validate all paid features against the server-derived entitlement. This PR does not change any existing Trust Voice calling permissions; do not turn on blanket paid gating or charge existing subscribers without explicit rollout.
7. Verify signed iPhone and iPad StoreKit purchase/restore, sign-out, locked-screen calling, background operation and account-deletion flow before submitting build 6+.

## Privacy and safety
Do not email, log, forward, or persist transaction JWS or Trust Voice session tokens in debug logs. Customer account deletion must separately address Apple subscriptions; closing a Circle account does not cancel Apple billing. Refund and revoke events must remove entitlement when verified.
