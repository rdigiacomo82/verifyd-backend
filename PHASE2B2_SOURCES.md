# Phase 2B.2 - Curated YARA Source Registry

This phase records candidate third-party sources before any rule is promoted into an enabled production pack.

## Candidate-approved

- Tenable YARA Rules - BSD-3-Clause
- Intezer YARA Rules - MIT

## Hold

- Elastic protections-artifacts - Elastic License 2.0; separate review required
- Yara-Rules/rules - GPL-2.0; do not bundle into the commercial product without explicit distribution review

## Rule promotion requirements

An external rule is not production-ready merely because its repository is approved as a candidate source.

Each rule must pass:

1. license/source attribution review
2. YARA-X compile validation
3. duplicate/conflict review
4. severity/category metadata normalization
5. benign corpus / false-positive review
6. malware/security test fixture review where appropriate
7. final enablement into a named VeriFYD pack

Candidate ingestion never automatically enables a pack.
