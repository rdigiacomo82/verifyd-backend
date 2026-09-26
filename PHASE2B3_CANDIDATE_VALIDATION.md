# Phase 2B.3 - Candidate Rule Ingestion and Compile Validation

## Isolation

Candidate third-party rules are staged under `yara_candidates/`, not `rules/`.
This prevents the production rule loader from seeing unreviewed external rules.

## Workflow

1. Fetch only sources allowed by `rules/sources.json`.
2. Record repository commit and remote URL.
3. Discover `.yar` and `.yara` files.
4. Compile each file independently with YARA-X and includes disabled.
5. Record SHA-256, compile outcome, warnings, and errors.
6. Produce a JSON report for promotion review.
7. Do not enable or package candidate rules.

## Commands

    python tools\yara_candidate_fetch.py
    python tools\yara_candidate_validate.py

Candidate compile failures are expected. They identify rules that need adaptation,
dependencies, unsupported constructs, or rejection before production promotion.
