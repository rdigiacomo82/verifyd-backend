# VeriFYD Universal Handle Registry Charter

**Version:** 0.1 — DRAFT FOR FOUNDER, PRODUCT, SECURITY AND LEGAL REVIEW  
**Prepared:** 15 November 2026  
**Proposed operator:** VeriFYD [CONFIRM LEGAL OPERATING ENTITY AND JURISDICTION]  
**Status:** NON-OPERATIVE. This document does not amend any current user's rights, terms, privacy settings, registry status, or existing contract. It takes effect only if expressly approved, published, and introduced through a lawful notice/consent process.

## 1. Founding purpose

The VeriFYD Universal Handle Registry Project exists to make one durable, human-readable communication identity usable across consented services while protecting the holder's autonomy, safety, privacy, accessibility, and ability to change communication providers.

The initial registry is **operated by VeriFYD and authoritative only within VeriFYD's namespace and participating partner systems**. It has no jurisdiction over usernames, phone numbers, domain names, or email accounts controlled by other providers. The long-term objective is an openly specified, independently governable interoperability framework that competing providers can implement without surrendering their users' rights to VeriFYD.

## 2. Initial scope and exclusions

**In scope:** Registration and resolution of VeriFYD handles; continuity of existing identity IDs and handles; documented handle status, privacy controls and authorized communication capabilities; ownership and organization-verification claims; recovery, disputes, abuse prevention, auditability, and future domain-qualified federation.

**Not in scope in Phase 1:** Global exclusivity for unqualified @names; assignment of E.164 telephone numbers; automatic PSTN/911 connection; proof of real-world identity from email verification alone; verified ownership of third-party social profiles without completed proof; automatic domain-name registration; a transferable property title in a string; a universal login identity; cross-network interoperability without partner agreements.

## 3. Definitions and namespaces

- **Identity ID:** A stable internal VeriFYD account identifier, independent of the displayed handle. The operator must avoid reusing it for another account.
- **Handle:** A human-readable label such as @sample, unique under VeriFYD's current registration rules, normalized case-insensitively.
- **Qualified account identifier:** Proposed standards-compatible identifier such as acct:sample@vfvid.com. This identifies a provider-hosted account; it is not itself an email address or routing authorization.
- **Registrant:** The person or legal organization registered as the controller of a handle in VeriFYD.
- **Verified attribute:** A claim substantiated through an explicitly described proof process; email, personal identity, organization, domain control and social-account control are different claims.
- **Resolver:** A server that responds only with a permitted subset of contact or discovery capabilities. Resolution never overrides trust, block or privacy settings.
- **Partner registrar:** A future independent provider explicitly authorized by contract to perform a narrow set of registration functions; no partner accreditation exists at the time of this draft.

## 4. Charter principles

1. **Continuity first.** Preserve all valid existing VeriFYD accounts, internal identity IDs, registered handles, messages, conversations, email routes, call permissions and Trust Circle relationships. Any migration must be reversible or backed by a tested recovery plan.
2. **One current handle, one account in the VeriFYD namespace.** Handle uniqueness and normalization are server-enforced; display formatting does not grant separate identities.
3. **Clear distinction between registration and verification.** A registered handle is not evidence of legal identity, a professional license, official status, a trademark, or ownership of the same name on another network.
4. **Privacy by default.** Registry records are not a public address book by default. Discovery may return only information expressly authorized for the audience and purpose; private email, telephone, social data, location, child-related information, and trust relationships are protected.
5. **Owner control.** Holders may control reachability and revoke linked-service permissions. Removing discovery must not silently delete the holder's account or inbound messages.
6. **Pseudonymity with accountability.** A verified real name is optional unless a specific regulated, organizational, or contractual service genuinely requires it. Impersonation and fraud are prohibited regardless of pseudonymity.
7. **Security before popularity.** Strong authentication and recovery, rate-limited lookup and registration, anti-enumeration safeguards, tamper-evident administrative audit, and periodic adversarial testing precede public federation.
8. **Fair process.** Reserved-name decisions, conflicts, suspensions, reassignments, and complaints require recorded reasons, proportional interim actions, notice where safe and lawful, and an accessible appeal channel.
9. **Accessibility as a protocol requirement.** Registration, verification, calling, messaging, recovery, and disputes must accommodate screen readers, keyboard-only access, switch access, and Eye Gaze-capable clients. Do not penalize an accessibility method in identity trust decisions.
10. **Open interoperability.** Prefer existing internet standards and published conformance tests; avoid granting a single commercial app absolute control over an otherwise open protocol.

## 5. Governance and accountability

**Phase 1 operator:** VeriFYD [LEGAL ENTITY TO CONFIRM], operating its own namespace, authentication, registry and contact services. The operator is responsible for secure handling of member data, responding to complaints, maintaining auditable policies, and making an effective recovery process available.

**Decision rights:** Product owns proposed changes; security/privacy reviews risks; legal reviews binding rights, protected identifiers and cross-border applicability; a designated registry steward approves operating changes. An administrator with a financial, personal or organizational interest in a disputed handle must recuse from the dispute.

**Separation of roles:** Registry ownership decisions, identity verification, moderation, billing, and communications routing are separate activities with separate authorization and audit trails. Subscription payments never establish personal identity or override an ownership dispute.

**Future governance:** After a working pilot and external participation, publish technical specifications and seek an independent multi-stakeholder advisory or standards body, with published decision procedures, conflicts policy and an appeals route. No present claim is made that such a body exists.

## 6. Existing registrants

All legitimately registered, active VeriFYD handles retain their current registry allocation when Phase 1 launches. No forced rename, mass re-registration, reassignment, or new payment condition should occur solely because the registry architecture is introduced. Legacy conflicts or names that overlap a newly reserved word need individualized, documented review rather than silent deletion.

The current account ID remains the internal primary key. Alias history, if later added, must never replace identity IDs in messages, calls, Trust Circle membership, Homeworks, rewards, or other existing records.

## 7. Technical architecture principles

The current VeriFYD backend remains the source of truth for authenticated identities and internal routing. A future isolated registry service may add ownership events, verified linked services, consent-based capability resolution, recovery proofs, disputes and delegated business permissions without replacing existing communication endpoints.

Use transport security, least-privilege credentials, backend authorization, server-side transactional uniqueness, rate limiting, structured audits, and integration tests. Do not expose direct SIP, telephone, mailbox, social or geographic targets in public discovery records when a consent-controlled intermediary can route communications instead.

Interoperable identity discovery may use RFC 7565 (acct URI) and RFC 7033 (WebFinger); their existence does not itself supply calling or messaging transport. Other identity frameworks, including W3C DIDs, may be evaluated later if their privacy and portability properties are justified.

## 8. Data protection and vulnerable users

Do not make real email addresses, phone numbers, precise locations, private social profiles, private trust contacts, identity-verification evidence, legal names, or authentication metadata available to unauthenticated resolvers by default.

Use purpose limitation, minimum necessary retention, verifiable authority for disclosure, documented retention/deletion schedules, incident response, and access logs. Do not use a global handle as a forced cross-service tracking identifier; selectively disclose stable identifiers only where needed.

**Children:** No public discoverable under-13 handle directory or open contact routing should be launched until age treatment, verified parental consent where required, safeguarding, and applicable child-data law are reviewed and implemented. A parent's ownership of an account is not proof of consent to every outward-facing service.

## 9. Abuse, disputes and fair procedure

The separate Handle Ownership Policy defines registration, reserved names, protection, impersonation reporting, competing claims, response timelines, temporary holds, recovery, suspension, and appeals. The operator must provide a way to challenge mistakes and document significant administrative action. Security holds may be applied without advance notice where necessary to prevent immediate harm, but they must be promptly reviewed.

ICANN's domain-name dispute framework offers procedural lessons only; it does not govern VeriFYD usernames or transfer rights.

## 10. Federation and portability commitments

Phase 1 uses a VeriFYD-controlled namespace. Any published portability claim must identify exactly what moves: profile data, a qualified identity, consented linked service, message history, cryptographic credential, or bare handle. Never advertise portability that participating providers cannot technically or contractually honor.

Future partner APIs require explicit trust agreements, key management, authorization scopes, service-level definitions, misuse controls, offboarding procedures and conformance tests. If a user disables partner access, future routing must honor revocation.

## 11. Publication, revisions and transparency

Material changes require a versioned changelog, privacy/security review, reasonable notice to affected users, and where legally or contractually required new consent. Registry policy should be publicly accessible in plain language and accessible formats. A proposed annual transparency report should include aggregate disputes, suspensions, appeals, reversals and security incidents, without revealing complainants' or registrants' private information.

## 12. Preconditions for operational adoption

This draft becomes an operational commitment only after all of the following are recorded:

- Confirm legal operator, governing law, accountable owner and support/dispute contact.
- Complete privacy, consumer-protection, trademark, accessibility and child-safety counsel reviews.
- Approve the versioned Handle Ownership Policy, terms integration, and retention/appeal rules.
- Implement tests for legacy preservation, unauthorized reassignment, uniqueness, recovery, rate limiting, blocked access and audit logging.
- Confirm account deletion/export and any required notice/acceptance.
- Stage behind feature flags and conduct a closed pilot with documented rollback and incident-response owners.
- Publish an effective date and a migration communication that says exactly what, if anything, changes for existing members.

## 13. Reference documents

- IETF RFC 7565 — The acct URI Scheme: https://www.rfc-editor.org/rfc/rfc7565
- IETF RFC 7033 — WebFinger: https://www.rfc-editor.org/rfc/rfc7033
- W3C DID Core privacy considerations: https://www.w3.org/TR/did-core/
- ICANN Uniform Domain Name Dispute Resolution Policy (informational analogy, not binding on handles): https://www.icann.org/resources/pages/udrp-2023-01-20-en
- FTC COPPA compliance FAQ: https://www.ftc.gov/business-guidance/resources/complying-coppa-frequently-asked-questions

**Approval record:** [Founder] [Legal] [Security/Privacy] [Effective date] — ALL PENDING.
