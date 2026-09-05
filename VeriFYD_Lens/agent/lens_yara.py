# VeriFYD Lens YARA-X Engine
# VERIFYD_LENS_YARA_ENGINE_V1
# Standalone signature engine. This module is intentionally additive and
# independent from lens_agent.py until Phase 2A.2 integration.

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import os
import time

ENGINE_ID = "verifyd_yara_x_v1"
_DEFAULT_SCAN_TIMEOUT_SECONDS = 10

_RULES = None
_RULE_COUNT = 0
_RULE_FILES: List[str] = []
_INIT_ERROR = ""
_INITIALIZED = False


def _agent_dir() -> Path:
    return Path(__file__).resolve().parent


def _default_rules_root() -> Path:
    # lens_yara.py lives in VeriFYD_Lens/agent, so parent is VeriFYD_Lens.
    return _agent_dir().parent / "rules"


def _rules_root() -> Path:
    raw = os.environ.get("VERIFYD_LENS_RULES_DIR", "").strip()
    return Path(raw).expanduser().resolve() if raw else _default_rules_root()


def _engine_label() -> str:
    try:
        import yara_x  # type: ignore
        version = getattr(yara_x, "__version__", "") or "unknown"
        return f"YARA-X {version}"
    except Exception:
        return "YARA-X unavailable"


def _empty_result(status: str, finding: str = "") -> Dict[str, Any]:
    return {
        "engine": ENGINE_ID,
        "engine_label": _engine_label(),
        "status": status,
        "score_delta": 0,
        "hard_block": False,
        "matches": [],
        "match_count": 0,
        "rule_count": int(_RULE_COUNT or 0),
        "rule_files": list(_RULE_FILES),
        "finding": finding,
        "elapsed_ms": 0,
        "details": {},
    }


def _load_rule_files(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return sorted(
        [p for p in root.rglob("*.yar") if p.is_file()] +
        [p for p in root.rglob("*.yara") if p.is_file()]
    )


def initialize(force: bool = False) -> Dict[str, Any]:
    """Compile and cache YARA-X rules. Safe to call repeatedly."""
    global _RULES, _RULE_COUNT, _RULE_FILES, _INIT_ERROR, _INITIALIZED

    if _INITIALIZED and not force:
        if _INIT_ERROR:
            return _empty_result("UNAVAILABLE", _INIT_ERROR)
        return {
            "engine": ENGINE_ID,
            "engine_label": _engine_label(),
            "status": "READY" if _RULES is not None else "NO_RULES",
            "rule_count": int(_RULE_COUNT or 0),
            "rule_files": list(_RULE_FILES),
            "error": "",
        }

    _RULES = None
    _RULE_COUNT = 0
    _RULE_FILES = []
    _INIT_ERROR = ""
    _INITIALIZED = True

    try:
        import yara_x  # type: ignore
    except Exception as exc:
        _INIT_ERROR = f"YARA-X import failed: {type(exc).__name__}"
        return _empty_result("UNAVAILABLE", _INIT_ERROR)

    root = _rules_root()
    files = _load_rule_files(root)
    _RULE_FILES = [str(p) for p in files]

    if not files:
        return {
            "engine": ENGINE_ID,
            "engine_label": _engine_label(),
            "status": "NO_RULES",
            "rule_count": 0,
            "rule_files": [],
            "error": "",
        }

    source_parts: List[str] = []
    compile_errors: List[str] = []
    for path in files:
        try:
            source_parts.append(path.read_text(encoding="utf-8", errors="replace"))
        except Exception as exc:
            compile_errors.append(f"{path}: read failed: {type(exc).__name__}")

    if not source_parts:
        _INIT_ERROR = "; ".join(compile_errors) or "No readable YARA rule files."
        return _empty_result("UNAVAILABLE", _INIT_ERROR)

    try:
        # Concatenate verified local rule files into a single ruleset. The initial
        # Phase 2A pack uses globally unique rule names to avoid namespace issues.
        _RULES = yara_x.compile("\n\n".join(source_parts))
        _RULE_COUNT = sum(1 for text in source_parts for line in text.splitlines() if line.strip().startswith("rule "))
        return {
            "engine": ENGINE_ID,
            "engine_label": _engine_label(),
            "status": "READY",
            "rule_count": int(_RULE_COUNT or 0),
            "rule_files": list(_RULE_FILES),
            "error": "",
            "compile_warnings": compile_errors,
        }
    except Exception as exc:
        _INIT_ERROR = f"YARA-X compile failed: {type(exc).__name__}: {str(exc)[:240]}"
        return _empty_result("UNAVAILABLE", _INIT_ERROR)


def _normalize_match(match: Any) -> Dict[str, Any]:
    identifier = str(getattr(match, "identifier", "") or getattr(match, "rule", "") or "unknown")
    metadata = getattr(match, "metadata", None)
    meta: Dict[str, Any] = {}
    if isinstance(metadata, dict):
        meta = dict(metadata)
    else:
        # yara-x metadata is version-dependent; keep parsing defensive.
        try:
            for item in metadata or []:
                key = getattr(item, "identifier", "") or getattr(item, "key", "")
                val = getattr(item, "value", None)
                if key:
                    meta[str(key)] = val
        except Exception:
            meta = {}

    severity = str(meta.get("severity") or "INFO").upper()
    category = str(meta.get("category") or "signature")
    description = str(meta.get("description") or identifier)

    return {
        "rule": identifier,
        "severity": severity,
        "category": category,
        "description": description,
        "metadata": meta,
    }


def scan_file(path: str | os.PathLike[str], timeout_seconds: int = _DEFAULT_SCAN_TIMEOUT_SECONDS) -> Dict[str, Any]:
    """Scan one file with cached YARA-X rules. Fails open on any engine error."""
    start = time.perf_counter()
    init = initialize()

    if init.get("status") in {"UNAVAILABLE", "NO_RULES"}:
        result = _empty_result(str(init.get("status") or "UNAVAILABLE"), str(init.get("finding") or init.get("error") or "YARA-X is not ready."))
        result["elapsed_ms"] = int((time.perf_counter() - start) * 1000)
        return result

    target = Path(path)
    if not target.exists() or not target.is_file():
        result = _empty_result("ERROR", "YARA-X could not scan because the file does not exist.")
        result["elapsed_ms"] = int((time.perf_counter() - start) * 1000)
        result["details"] = {"path": str(target)}
        return result

    try:
        # The yara-x API supports scanning bytes consistently across versions.
        # Keep the read bounded by the same 250 MB Lens MVP limit used elsewhere.
        max_bytes = int(os.environ.get("VERIFYD_LENS_YARA_MAX_BYTES", str(250 * 1024 * 1024)))
        size = target.stat().st_size
        if size > max_bytes:
            result = _empty_result("SKIPPED", "YARA-X skipped this file because it exceeds the configured scan limit.")
            result["elapsed_ms"] = int((time.perf_counter() - start) * 1000)
            result["details"] = {"size_bytes": size, "max_bytes": max_bytes}
            return result

        data = target.read_bytes()
        scan_result = _RULES.scan(data)  # type: ignore[union-attr]
        raw_matches = list(getattr(scan_result, "matching_rules", []) or [])
        matches = [_normalize_match(m) for m in raw_matches]

        status = "MATCH" if matches else "NO_MATCH"
        severities = {m.get("severity", "INFO") for m in matches}
        hard_block = bool(severities.intersection({"HIGH", "CRITICAL"}))
        score_delta = -75 if "CRITICAL" in severities else -55 if "HIGH" in severities else -25 if "MEDIUM" in severities else -10 if matches else 0

        finding = (
            "YARA-X matched security rules: " + ", ".join(m["rule"] for m in matches)
            if matches
            else "YARA-X found no matching built-in or custom security rules."
        )

        return {
            "engine": ENGINE_ID,
            "engine_label": _engine_label(),
            "status": status,
            "score_delta": score_delta,
            "hard_block": hard_block,
            "matches": matches,
            "match_count": len(matches),
            "rule_count": int(_RULE_COUNT or 0),
            "rule_files": list(_RULE_FILES),
            "finding": finding,
            "elapsed_ms": int((time.perf_counter() - start) * 1000),
            "details": {"size_bytes": size},
        }
    except Exception as exc:
        return {
            "engine": ENGINE_ID,
            "engine_label": _engine_label(),
            "status": "ERROR",
            "score_delta": 0,
            "hard_block": False,
            "matches": [],
            "match_count": 0,
            "rule_count": int(_RULE_COUNT or 0),
            "rule_files": list(_RULE_FILES),
            "finding": f"YARA-X scan failed open: {type(exc).__name__}",
            "elapsed_ms": int((time.perf_counter() - start) * 1000),
            "details": {"error": str(exc)[:240]},
        }


if __name__ == "__main__":
    import json
    import sys

    print(json.dumps(initialize(force=True), indent=2))
    for arg in sys.argv[1:]:
        print(json.dumps(scan_file(arg), indent=2))
