from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_RULE_RE = re.compile(
    r"(?ms)^\s*(?:global\s+|private\s+)*rule\s+([A-Za-z_][A-Za-z0-9_]*)"
    r"(?:\s*:[^{]+)?\s*\{(.*?)^\s*\}"
)
_META_RE = re.compile(r"(?ms)^\s*meta\s*:\s*(.*?)(?=^\s*(?:strings|condition)\s*:)")
_META_KV_RE = re.compile(r'(?m)^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?:"([^"]*)"|([^\r\n]+))')


@dataclass(frozen=True)
class RuleRecord:
    source_id: str
    file: str
    rule: str
    sha256: str
    meta_keys: tuple[str, ...]
    has_author: bool
    has_description: bool
    has_reference: bool
    has_date: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _meta_from_body(body: str) -> dict[str, str]:
    m = _META_RE.search(body)
    if not m:
        return {}
    out: dict[str, str] = {}
    for km in _META_KV_RE.finditer(m.group(1)):
        key = km.group(1)
        value = km.group(2) if km.group(2) is not None else (km.group(3) or "").strip()
        out[key.lower()] = value
    return out


def inventory_source(source_id: str, repo_dir: Path) -> list[RuleRecord]:
    records: list[RuleRecord] = []
    files = sorted(set(repo_dir.rglob("*.yar")) | set(repo_dir.rglob("*.yara")))
    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace")
        file_sha = _sha256(path)
        for match in _RULE_RE.finditer(text):
            name = match.group(1)
            meta = _meta_from_body(match.group(2))
            keys = tuple(sorted(meta.keys()))
            records.append(
                RuleRecord(
                    source_id=source_id,
                    file=str(path),
                    rule=name,
                    sha256=file_sha,
                    meta_keys=keys,
                    has_author="author" in meta,
                    has_description=any(k in meta for k in ("description", "desc")),
                    has_reference=any(k in meta for k in ("reference", "references", "ref")),
                    has_date=any(k in meta for k in ("date", "created", "creation_date", "modified")),
                )
            )
    return records


def triage_sources(sources: dict[str, Path]) -> dict[str, Any]:
    records: list[RuleRecord] = []
    for source_id, repo_dir in sources.items():
        records.extend(inventory_source(source_id, repo_dir))

    by_name: dict[str, list[RuleRecord]] = {}
    for record in records:
        by_name.setdefault(record.rule, []).append(record)

    duplicates = {
        name: [r.to_dict() for r in rs]
        for name, rs in by_name.items()
        if len(rs) > 1
    }

    missing_author = [r.to_dict() for r in records if not r.has_author]
    missing_description = [r.to_dict() for r in records if not r.has_description]
    missing_reference = [r.to_dict() for r in records if not r.has_reference]
    missing_date = [r.to_dict() for r in records if not r.has_date]

    return {
        "candidate_only": True,
        "production_enabled": False,
        "source_count": len(sources),
        "rule_count": len(records),
        "duplicate_rule_name_count": len(duplicates),
        "duplicates": duplicates,
        "metadata_quality": {
            "missing_author": len(missing_author),
            "missing_description": len(missing_description),
            "missing_reference": len(missing_reference),
            "missing_date": len(missing_date),
        },
        "rules": [r.to_dict() for r in records],
    }


def combined_compile(yara_x: Any, sources: dict[str, Path]) -> dict[str, Any]:
    compiler = yara_x.Compiler()
    if hasattr(compiler, "max_warnings"):
        compiler.max_warnings(100)
    compiler.enable_includes(False)

    added_files = 0
    for source_id, repo_dir in sources.items():
        compiler.new_namespace(source_id)
        files = sorted(set(repo_dir.rglob("*.yar")) | set(repo_dir.rglob("*.yara")))
        for path in files:
            compiler.add_source(
                path.read_text(encoding="utf-8", errors="replace"),
                origin=str(path),
            )
            added_files += 1

    warnings = tuple(str(w) for w in (compiler.warnings() if hasattr(compiler, "warnings") else ()))
    try:
        rules = compiler.build()
        return {
            "ok": True,
            "files": added_files,
            "namespaces": sorted(sources),
            "warnings": warnings,
            "rules_object": rules,
            "error": None,
        }
    except Exception as exc:
        return {
            "ok": False,
            "files": added_files,
            "namespaces": sorted(sources),
            "warnings": warnings,
            "rules_object": None,
            "error": f"{type(exc).__name__}: {exc}",
        }
