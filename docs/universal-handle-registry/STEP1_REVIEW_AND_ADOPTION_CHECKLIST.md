# Universal Handle Registry — Step 1 Review and Adoption Checklist

**Status:** Draft policy package, prepared 15 November 2026. Not operative. No user account, database, production routing, user-facing acceptance requirement, or publication setting is changed by this PR.

## Existing implementation verified in GitHub

- Current identity implementation: trust_voice_identity.py: email-OTP initial claim; immutable generated identity ID; handle_lower uniqueness; 3–24 character ASCII pattern; protected operational names; one existing identity per verified email.
- Existing routing: trust_voice_signaling.py, trust_voice_circle_email.py and trust_voice_circle_email_public.py use VeriFYD handles; current public email is separately gated.
- Existing social accounts: trust_voice_social_links.py accepts selected user-supplied profile URLs with audience controls but explicitly does **not** claim verified external account ownership.
- Existing Trust Circle privacy: trust_voice_trust_circle.py and trust_voice_permissions.py control relationships and access.

Do not change these behaviors merely because this policy draft is merged or published.

## Decisions needed before policy adoption

| # | Decision | Proposed default | Approval |
|---|---|---|---|
| 1 | Exact legal registry operator and support address | [Confirm corporate entity, jurisdiction, and contact] | Pending |
| 2 | Scope of Phase 1 uniqueness | Existing VeriFYD namespace only | Pending |
| 3 | Existing member allocations | Grandfather, no compulsory renaming or re-registration | Pending |
| 4 | Does registration confer property title? | No; continued service-specific right, subject to fair policy | Pending legal |
| 5 | Public discoverability | Off by default; no private contact endpoints in public records | Pending privacy |
| 6 | Children's public directory | Disabled pending child privacy and parental consent review | Pending legal |
| 7 | Transfers, resale, and renaming | Not supported in initial registry release | Pending |
| 8 | Handle recycling after deletion | No automatic reassignment until a documented schedule is adopted | Pending privacy |
| 9 | Dispute response targets | 5 business days acknowledge / 15 business days initial review / 30 days appeal (draft only) | Pending |
| 10 | Corporate verification proofs | Separate legal authority validation; no social URL == verification | Pending |
| 11 | Registration fees | Free basic personal handles proposed; no new fee in this step | Pending |
| 12 | Public policy/ToS change protocol | Legal review, versioning, notice, required consent, rollback | Pending |

## Definition of Step 1 done

- Founding operator and the requested legal rights are unambiguous.
- Charter and ownership policy language approved by appropriate legal, security/privacy, accessibility and product reviewers.
- Public-facing and private operating procedures separated (public charter / handle policy versus internal anti-abuse playbook).
- Complaint and appeal contact and a working owner are assigned.
- Children's public discoverability and recovery rules approved.
- Existing members receive a clear legacy-rights/migration notice only if necessary.
- Version IDs and effective dates approved.
- No operational changes until subsequent Step 2 backend implementation and tests are signed off.

## Step 2 implementation request (NOT PART OF THIS PR)

Create isolated, feature-gated registry and audit tables keyed to the existing verifyd_identities.id; maintain compatibility with all existing calling, messaging, email, groups, therapy, Rank and trust circle records. Implement the minimum record types:

1. Registration status and handle history (no silent reassignment).
2. Consent-scoped public discovery and service capability routing.
3. Verified external link *claims* with proof method and expiry.
4. Recovery and reservation decisions, with dual-approval for sensitive changes.
5. Dispute intake, appeals and tamper-evident administrative events.
6. Privacy tests: no PII, phone, email, social link, location or child data leakage.

No federation, WebFinger public endpoint, or third-party handles are enabled simply by drafting the policy.

## References

- IETF RFC 7565: https://www.rfc-editor.org/rfc/rfc7565
- IETF RFC 7033: https://www.rfc-editor.org/rfc/rfc7033
- ICANN UDRP scope: https://www.icann.org/resources/pages/udrp-2023-01-20-en
- FTC COPPA FAQ: https://www.ftc.gov/business-guidance/resources/complying-coppa-frequently-asked-questions
