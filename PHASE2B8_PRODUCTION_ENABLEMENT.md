# Phase 2B.8 - Controlled Production Enablement

This phase enables the first curated third-party YARA packs:

- `tenable_curated`
- `intezer_curated`

They remain `trusted: false` because they are externally sourced.

The regression compiles the full production rules tree, including `verifyd_core`,
and verifies:

1. required namespaces are enabled
2. the entire enabled ruleset compiles
3. benign fixtures produce zero YARA matches
4. harmless synthetic signature fixtures positively exercise selected external rules

Commands:

    python tools\yara_phase2b8_enable.py
    python tools\yara_pack_audit.py
    python tools\yara_phase2b8_production_regression.py

Rollback command:

    python tools\yara_phase2b8_disable.py

Do not rebuild or redistribute the Lens installer unless the production regression passes.
