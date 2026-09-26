from __future__ import annotations

import importlib
import importlib.metadata
import threading
import time
from pathlib import Path
from typing import Any

from .loader import CompiledBundle, compile_rule_packs
from .models import YaraScanResult, YaraStatus
from .normalizer import normalize_rule_match
from .policy import YaraRiskPolicy


class YaraXEngine:
    """Thread-safe YARA-X service used by VeriFYD clients.

    Compiled rule objects are immutable snapshots. A reload builds a complete
    replacement first and only swaps it into service after compilation succeeds.
    Active scans keep their old snapshot and are unaffected by reloads.
    """

    def __init__(
        self,
        rules_root: str | Path,
        *,
        timeout_seconds: int = 10,
        max_matches_per_pattern: int = 100,
        evidence_limit_per_rule: int = 20,
        policy: YaraRiskPolicy | None = None,
        yara_module: Any | None = None,
        enabled: bool = True,
        ruleset_version: str | None = None,
    ) -> None:
        self.rules_root = Path(rules_root)
        self.timeout_seconds = timeout_seconds
        self.max_matches_per_pattern = max_matches_per_pattern
        self.evidence_limit_per_rule = evidence_limit_per_rule
        self.policy = policy or YaraRiskPolicy()
        self.enabled = enabled
        self.ruleset_version = ruleset_version
        self._lock = threading.RLock()
        self._bundle: CompiledBundle | None = None
        self._last_load_error: str | None = None
        self._yara_x = yara_module

    def _backend(self) -> Any:
        if self._yara_x is not None:
            return self._yara_x
        try:
            self._yara_x = importlib.import_module("yara_x")
            return self._yara_x
        except ImportError as exc:
            raise RuntimeError("YARA-X Python package is not installed (pip install yara-x)") from exc

    @property
    def ready(self) -> bool:
        with self._lock:
            return self.enabled and self._bundle is not None

    def load(self) -> bool:
        if not self.enabled:
            return False
        try:
            backend = self._backend()
            candidate = compile_rule_packs(
                backend,
                self.rules_root,
                version=self.ruleset_version,
            )
        except Exception as exc:
            with self._lock:
                self._last_load_error = f"{type(exc).__name__}: {exc}"
            return False

        with self._lock:
            self._bundle = candidate
            self._last_load_error = None
        return True

    def reload(self) -> bool:
        # Atomic by construction: compile_rule_packs returns only after the
        # complete replacement is valid. A failed reload never clears _bundle.
        return self.load()

    def health(self) -> dict[str, Any]:
        with self._lock:
            bundle = self._bundle
            last_error = self._last_load_error
        backend_version = None
        try:
            backend_version = getattr(self._backend(), "__version__", None) or importlib.metadata.version("yara-x")
        except Exception:
            pass
        return {
            "available": self.enabled and bundle is not None,
            "enabled": self.enabled,
            "engine": "YARA-X",
            "engine_version": backend_version,
            "ruleset": None if bundle is None else {
                "version": bundle.info.version,
                "sha256": bundle.info.sha256,
                "rules_loaded": bundle.info.rules_loaded,
                "rule_files_loaded": bundle.info.rule_files_loaded,
                "namespaces": list(bundle.info.namespaces),
                "loaded_at": bundle.info.loaded_at,
                "compile_warnings": len(bundle.info.compile_warnings),
            },
            "last_load_error": last_error,
        }

    def scan_file(self, path: str | Path, *, context: dict[str, Any] | None = None) -> YaraScanResult:
        if not self.enabled:
            return YaraScanResult(status=YaraStatus.DISABLED)

        file_path = Path(path)
        if not file_path.exists() or not file_path.is_file():
            return YaraScanResult(
                status=YaraStatus.SCAN_ERROR,
                errors=(f"File not found or not a regular file: {file_path}",),
            )

        with self._lock:
            bundle = self._bundle
            load_error = self._last_load_error

        if bundle is None:
            # Lazy startup is useful in local agents, but a bad ruleset must
            # remain a degraded state instead of making YARA evidence appear clean.
            if not self.load():
                try:
                    self._backend()
                    status = YaraStatus.RULESET_LOAD_FAILED
                except RuntimeError:
                    status = YaraStatus.ENGINE_UNAVAILABLE
                return YaraScanResult(status=status, errors=((self._last_load_error or load_error or "YARA-X unavailable"),))
            with self._lock:
                bundle = self._bundle

        assert bundle is not None
        try:
            backend = self._backend()
            scanner = backend.Scanner(bundle.rules)
            scanner.set_timeout(self.timeout_seconds)
            if hasattr(scanner, "max_matches_per_pattern"):
                scanner.max_matches_per_pattern(self.max_matches_per_pattern)
            if hasattr(scanner, "fast_scan"):
                scanner.fast_scan(False)

            ctx = dict(context or {})
            # Optional external global. Rule packs may use this in Phase 2B.
            # set_global raises if the compiled rules did not define it, so the
            # call is intentionally best-effort and never affects scanning.
            if ctx and hasattr(scanner, "set_global"):
                file_info = {
                    "name": str(ctx.get("filename") or file_path.name),
                    "size": int(ctx.get("size") or file_path.stat().st_size),
                    "mime_type": str(ctx.get("mime_type") or ""),
                    "source": str(ctx.get("source") or "unknown"),
                }
                try:
                    scanner.set_global("file_info", file_info)
                except Exception:
                    pass

            started = time.perf_counter()
            scan_results = scanner.scan_file(str(file_path))
            elapsed_ms = int(round((time.perf_counter() - started) * 1000))
            normalized = tuple(
                normalize_rule_match(rule, evidence_limit=self.evidence_limit_per_rule)
                for rule in scan_results.matching_rules
            )
            normalized, points, highest, action = self.policy.apply(normalized)
            status = YaraStatus.COMPLETED_WITH_MATCHES if normalized else YaraStatus.COMPLETED_NO_MATCHES
            return YaraScanResult(
                status=status,
                engine_version=getattr(backend, "__version__", None) or importlib.metadata.version("yara-x"),
                ruleset=bundle.info,
                scan_duration_ms=elapsed_ms,
                match_count=len(normalized),
                highest_severity=highest,
                risk_points=points,
                recommended_action=action,
                matches=normalized,
            )
        except Exception as exc:
            backend = None
            try:
                backend = self._backend()
            except Exception:
                pass
            timeout_type = getattr(backend, "TimeoutError", ()) if backend is not None else ()
            scan_error_type = getattr(backend, "ScanError", ()) if backend is not None else ()
            if timeout_type and isinstance(exc, timeout_type):
                return YaraScanResult(
                    status=YaraStatus.TIMEOUT,
                    engine_version=getattr(backend, "__version__", None) or importlib.metadata.version("yara-x"),
                    ruleset=bundle.info,
                    errors=(f"YARA-X scan exceeded {self.timeout_seconds}s timeout",),
                )
            if scan_error_type and isinstance(exc, scan_error_type):
                status = YaraStatus.SCAN_ERROR
            else:
                status = YaraStatus.SCAN_ERROR
            return YaraScanResult(
                status=status,
                engine_version=(getattr(backend, "__version__", None) or importlib.metadata.version("yara-x")) if backend else None,
                ruleset=bundle.info,
                errors=(f"{type(exc).__name__}: {exc}",),
            )
