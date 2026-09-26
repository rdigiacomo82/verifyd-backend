# VeriFYD Phase 2B — Curated YARA Rule Packs

## Purpose

Phase 2B expands the shared YARA-X engine using controlled rule packs while preserving
the working Phase 2A production path.

## Safety principles

1. VeriFYD-owned rules remain isolated in `rules/verifyd_core`.
2. Third-party packs stay disabled until source and license review are complete.
3. Every pack has an ID, namespace, version, source, license, enabled state, and trust state.
4. Includes remain disabled in production unless explicitly vetted later.
5. Pack inventory/audit is independent of runtime loading so validation cannot break Lens.
6. A pack with manifest errors must not be promoted to production.
7. Curated rules are evidence signals; final VeriFYD Security/Authenticity/Trust verdicts remain policy-driven.

## Planned production packs

- `verifyd_core` — VeriFYD-owned rules
- `malware_generic` — generic malware indicators
- `suspicious_scripts` — PowerShell, script, LOLBin, downloader indicators
- `documents` — PDF/Office active content and exploit indicators
- `archives` — archive/container indicators

## Audit command

From the repository root:

    python tools\yara_pack_audit.py

The command exits 0 only when all manifests pass validation.

## Promotion workflow

Third-party rules should move through:

`candidate -> license review -> syntax compile -> false-positive tests -> metadata normalization -> enabled pack -> production build`

Do not place unreviewed external rules directly into an enabled production pack.
