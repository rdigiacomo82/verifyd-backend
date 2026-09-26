# VeriFYD Lens YARA-X compatibility adapter
# VERIFYD_LENS_SHARED_YARA_PHASE2A2
#
# Lens keeps its existing scan_yara_security() contract while delegating rule
# loading, scanning, normalization, deduplication, timeout handling and scoring
# to the shared verifyd_engine.yara service.

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict

if not getattr(sys, "frozen", False):
    _repo_root = Path(__file__).resolve().parents[2]
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))

try:
    from verifyd_engine.yara import YaraXEngine
except Exception:
    YaraXEngine = None  # type: ignore

ENGINE_ID = "verifyd_yara_x_shared_v1"
_DEFAULT_SCAN_TIMEOUT_SECONDS = 10
_ENGINE = None
_ENGINE_ERROR = ""


def _rules_root() -> Path:
    override = os.environ.get("VERIFYD_LENS_RULES_DIR", "").strip()
    if override:
        return Path(override).expanduser().resolve()

    frozen_root = getattr(sys, "_MEIPASS", None)
    if frozen_root:
        candidate = Path(frozen_root) / "rules"
        if candidate.exists():
            return candidate

    return Path(__file__).resolve().parents[2] / "rules"


def _empty(status: str, finding: str = "") -> Dict[str, Any]:
    return {
        "engine": ENGINE_ID,
        "engine_label": "YARA-X shared VeriFYD engine",
        "status": status,
        "score_delta": 0,
        "hard_block": False,
        "matches": [],
        "match_count": 0,
        "rule_count": 0,
        "rule_files": [],
        "finding": finding,
        "elapsed_ms": 0,
        "details": {},
    }


def initialize(force: bool = False) -> Dict[str, Any]:
    global _ENGINE, _ENGINE_ERROR

    if _ENGINE is not None and not force:
        health = _ENGINE.health()
        return {
            "engine": ENGINE_ID,
            "engine_label": "YARA-X shared VeriFYD engine",
            "status": "READY" if health.get("available") else "UNAVAILABLE",
            "health": health,
            "error": _ENGINE_ERROR,
        }

    if YaraXEngine is None:
        _ENGINE_ERROR = "Shared verifyd_engine.yara package is unavailable."
        return _empty("UNAVAILABLE", _ENGINE_ERROR)

    try:
        _ENGINE = YaraXEngine(
            _rules_root(),
            timeout_seconds=int(
                os.environ.get(
                    "VERIFYD_LENS_YARA_TIMEOUT_SECONDS",
                    str(_DEFAULT_SCAN_TIMEOUT_SECONDS),
                )
            ),
        )
        ok = bool(_ENGINE.load())
        health = _ENGINE.health()
        _ENGINE_ERROR = "" if ok else str(
            health.get("last_load_error") or "YARA-X ruleset load failed."
        )
        return {
            "engine": ENGINE_ID,
            "engine_label": "YARA-X shared VeriFYD engine",
            "status": "READY" if ok else "UNAVAILABLE",
            "health": health,
            "error": _ENGINE_ERROR,
        }
    except Exception as exc:
        _ENGINE = None
        _ENGINE_ERROR = (
            f"Shared YARA-X initialization failed: "
            f"{type(exc).__name__}: {str(exc)[:240]}"
        )
        return _empty("UNAVAILABLE", _ENGINE_ERROR)


def _legacy_match(match: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "rule": match.get("rule") or "unknown",
        "namespace": match.get("namespace") or "",
        "severity": str(match.get("severity") or "informational").upper(),
        "category": match.get("category") or "signature",
        "description": match.get("description") or match.get("rule") or "YARA-X finding",
        "metadata": match.get("raw_metadata") or {},
        "confidence": match.get("confidence"),
        "source": match.get("source"),
        "family": match.get("family"),
        "tags": list(match.get("tags") or []),
        "risk_points": int(match.get("risk_points") or 0),
        "evidence": list(match.get("evidence") or []),
    }


def scan_file(
    path: str | os.PathLike[str],
    timeout_seconds: int = _DEFAULT_SCAN_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    global _ENGINE

    init = initialize()
    if _ENGINE is None or init.get("status") != "READY":
        return _empty(
            "UNAVAILABLE",
            str(
                init.get("error")
                or init.get("finding")
                or "Shared YARA-X engine is unavailable."
            ),
        )

    try:
        _ENGINE.timeout_seconds = int(
            timeout_seconds or _DEFAULT_SCAN_TIMEOUT_SECONDS
        )
        p = Path(path)
        result = _ENGINE.scan_file(
            p,
            context={
                "filename": p.name,
                "source": "lens",
                "size": p.stat().st_size if p.exists() else 0,
            },
        )
        payload = result.to_dict()
        matches = [
            _legacy_match(dict(m)) for m in (payload.get("matches") or [])
        ]
        status = str(payload.get("status") or "SCAN_ERROR")
        points = int(payload.get("risk_points") or 0)
        highest = str(payload.get("highest_severity") or "informational").lower()
        action = str(payload.get("recommended_action") or "none").lower()

        if status == "SCAN_COMPLETED_WITH_MATCHES":
            legacy_status = "MATCH"
            finding = "YARA-X matched security rules: " + ", ".join(
                m["rule"] for m in matches
            )
        elif status == "SCAN_COMPLETED_NO_MATCHES":
            legacy_status = "NO_MATCH"
            finding = "YARA-X found no matching VeriFYD security rules."
        elif status == "SCAN_TIMEOUT":
            legacy_status = "TIMEOUT"
            finding = (
                "YARA-X scan timed out; Lens continued with its remaining "
                "security checks."
            )
        elif status in {
            "RULESET_LOAD_FAILED",
            "ENGINE_UNAVAILABLE",
            "DISABLED",
        }:
            legacy_status = "UNAVAILABLE"
            finding = (
                "YARA-X was unavailable; Lens continued with its remaining "
                "security checks."
            )
        else:
            legacy_status = "ERROR"
            finding = (
                "YARA-X scan failed open; Lens continued with its remaining "
                "security checks."
            )

        ruleset = payload.get("ruleset") or {}
        errors = list(payload.get("errors") or [])
        warnings = list(payload.get("warnings") or [])
        hard_block = bool(action == "block" or highest in {"critical", "high"})

        return {
            "engine": ENGINE_ID,
            "engine_label": "YARA-X shared VeriFYD engine",
            "status": legacy_status,
            "score_delta": -points,
            "hard_block": hard_block,
            "matches": matches,
            "match_count": int(payload.get("match_count") or len(matches)),
            "rule_count": int(ruleset.get("rules_loaded") or 0),
            "rule_files": [],
            "finding": finding,
            "elapsed_ms": int(payload.get("scan_duration_ms") or 0),
            "details": {
                "shared_status": status,
                "highest_severity": payload.get("highest_severity"),
                "risk_points": points,
                "recommended_action": payload.get("recommended_action"),
                "engine_version": payload.get("engine_version"),
                "ruleset": ruleset,
                "warnings": warnings,
                "errors": errors,
            },
        }
    except Exception as exc:
        result = _empty(
            "ERROR",
            f"Shared YARA-X scan failed open: {type(exc).__name__}",
        )
        result["details"] = {"error": str(exc)[:240]}
        return result


if __name__ == "__main__":
    import json

    print(json.dumps(initialize(force=True), indent=2))
    for arg in sys.argv[1:]:
        print(json.dumps(scan_file(arg), indent=2))
