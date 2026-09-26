# Phase 2B.4 - Candidate Rule Triage

This phase performs pre-promotion quality checks on third-party candidate rules.

Checks:

1. discover individual rule declarations
2. detect duplicate rule names across candidate sources
3. measure metadata completeness
4. compile all candidate files together under separate namespaces
5. run a small benign smoke-test corpus
6. keep all candidate rules outside the production `rules/` tree

Commands:

    python tools\yara_candidate_triage.py
    python tools\yara_candidate_benign_smoke.py

A clean smoke test is useful but is not sufficient for production promotion.
A larger benign corpus and rule-by-rule policy review are still required.
