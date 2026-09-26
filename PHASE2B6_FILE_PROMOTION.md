# Phase 2B.6 - File-Level Promotion Review

YARA promotion is performed at the file level, not by copying isolated rules.

Why:

- YARA files can contain private helper rules.
- Rules can reference other rules in the same file.
- Global rules can change behavior across an entire namespace.
- Automated family/vaccine rules need a different review standard from manually curated rules.

This phase therefore ranks candidate files and flags:

- private helper rules
- global rules
- compile warnings
- automated "Vaccine" rules
- unusually large rule files

Recommendations:

- SHORTLIST: strongest candidates for final regression review
- REVIEW: technically valid but needs additional manual review
- HOLD: do not promote yet

Command:

    python tools\yara_file_promotion_rank.py

No candidate files are copied into production by this phase.
