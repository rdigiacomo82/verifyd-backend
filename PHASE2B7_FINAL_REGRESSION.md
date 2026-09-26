# Phase 2B.7 - First Curated Production Candidate Subset

Selected initial files:

Tenable:
- cerber3.yar
- venom.yar

Intezer:
- AgeLocker.yar
- Doki_Attack.yar

These are intentionally staged in source-specific packs so license attribution and
upstream provenance remain clear.

The packs are created with:

    "enabled": false
    "trusted": false

This means the production VeriFYD loader will ignore them.

Commands:

    python tools\yara_phase2b7_stage.py
    python tools\yara_pack_audit.py
    python tools\yara_phase2b7_regression.py

Only after the audit and final regression both pass should these packs be considered
for explicit production enablement.
