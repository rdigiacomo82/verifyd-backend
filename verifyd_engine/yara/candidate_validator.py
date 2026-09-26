from __future__ import annotations

import hashlib
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CandidateFileResult:
    source_id: str
    path: str
    sha256: str
    compiled: bool
    warnings: tuple[str, ...] = ()
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_value(repo_dir: Path, *args: str) -> str:
    try:
        cp = subprocess.run(
            ["git", "-C", str(repo_dir), *args],
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        )
        return (cp.stdout or "").strip()
    except Exception:
        return ""


def source_metadata(source_id: str, repo_dir: Path) -> dict[str, Any]:
    return {
        "source_id": source_id,
        "directory": str(repo_dir),
        "commit": _git_value(repo_dir, "rev-parse", "HEAD"),
        "remote": _git_value(repo_dir, "config", "--get", "remote.origin.url"),
    }


def candidate_rule_files(repo_dir: Path) -> tuple[Path, ...]:
    return tuple(sorted(set(repo_dir.rglob("*.yar")) | set(repo_dir.rglob("*.yara"))))


def compile_candidate_file(yara_x: Any, source_id: str, path: Path) -> CandidateFileResult:
    compiler = yara_x.Compiler()
    if hasattr(compiler, "max_warnings"):
        compiler.max_warnings(100)
    compiler.enable_includes(False)

    try:
        compiler.add_source(path.read_text(encoding="utf-8", errors="replace"), origin=str(path))
        warnings_raw = tuple(compiler.warnings()) if hasattr(compiler, "warnings") else ()
        compiler.build()
        warnings = tuple(str(w) for w in warnings_raw)
        return CandidateFileResult(
            source_id=source_id,
            path=str(path),
            sha256=_sha256(path),
            compiled=True,
            warnings=warnings,
        )
    except Exception as exc:
        return CandidateFileResult(
            source_id=source_id,
            path=str(path),
            sha256=_sha256(path),
            compiled=False,
            error=f"{type(exc).__name__}: {exc}",
        )


def validate_sources(yara_x: Any, sources: dict[str, Path]) -> dict[str, Any]:
    results: list[CandidateFileResult] = []
    metadata: list[dict[str, Any]] = []

    for source_id, repo_dir in sources.items():
        metadata.append(source_metadata(source_id, repo_dir))
        for path in candidate_rule_files(repo_dir):
            results.append(compile_candidate_file(yara_x, source_id, path))

    compiled = sum(1 for r in results if r.compiled)
    failed = len(results) - compiled

    return {
        "ok": failed == 0,
        "candidate_only": True,
        "production_enabled": False,
        "source_count": len(sources),
        "file_count": len(results),
        "compiled": compiled,
        "failed": failed,
        "sources": metadata,
        "results": [r.to_dict() for r in results],
    }
