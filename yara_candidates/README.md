# VeriFYD YARA Candidate Staging

This directory is intentionally separate from `rules/`.

External YARA sources are cloned into `yara_candidates/sources/` for review and
compile testing only. The production Lens loader reads `rules/`, not this tree.

Candidate rule files must never be copied into an enabled production pack until
they pass source/license review, YARA-X compile validation, metadata normalization,
duplicate/conflict checks, and false-positive testing.

Commands:

    python tools\yara_candidate_fetch.py
    python tools\yara_candidate_validate.py

Generated source clones and reports are gitignored.
