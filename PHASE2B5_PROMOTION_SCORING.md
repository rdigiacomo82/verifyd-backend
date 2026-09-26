# Phase 2B.5 - Rule Promotion Scoring

This phase ranks candidate rules for manual promotion review.

The score is intentionally conservative and considers:

- YARA-X compilation success
- compile warnings
- author metadata
- description metadata
- references
- dates
- candidate-source approval

Tiers:

- A: 90-100
- B: 80-89
- C: 70-79
- HOLD: below 70 or blocked

A high score does not automatically promote a rule. It only identifies the best
candidates for the first production subset.

Command:

    python tools\yara_promotion_rank.py

The next phase should manually inspect the top-ranked rules, normalize their
metadata/severity/category fields, and copy only a small approved subset into a
disabled production pack for final regression testing.
