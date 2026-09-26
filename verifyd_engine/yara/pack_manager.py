from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_RULE_DECL_RE = re.compile(
    r"(?m)^\s*(?:global\s+|private\s+)*rule\s+[A-Za-z_][A-Za-z0-9_]*"
)

REQUIRED_FIELDS = (
    "schema_version",
    "id",
    "namespace",
    "enabled",
    "trusted",
    "allow_includes",
    "version",
    "source",
    "license",
    "description",
)


@dataclass(frozen=True)
class PackAudit:
    id: str
    namespace: str
    enabled: bool
    trusted: bool
    allow_includes: bool
    version: str
    source_name: str
    license_id: str
    directory: str
    rule_files: int
    rules: int
    sha256: str
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sha256_files(directory: Path, files: tuple[Path, ...]) -> str:
    h = hashlib.sha256()
    for path in files:
        rel = str(path.relative_to(directory)).replace("\\", "/")
        h.update(rel.encode("utf-8"))
        h.update(path.read_bytes())
    return h.hexdigest()


def _rule_files(directory: Path) -> tuple[Path, ...]:
    return tuple(sorted(set(directory.rglob("*.yar")) | set(directory.rglob("*.yara"))))


def load_manifest(directory: Path) -> dict[str, Any]:
    path = directory / "pack.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def validate_manifest(directory: Path, manifest: dict[str, Any]) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    for field in REQUIRED_FIELDS:
        if field not in manifest:
            errors.append(f"missing required field: {field}")

    if manifest.get("schema_version") not in {1}:
        errors.append("schema_version must be 1")

    pack_id = str(manifest.get("id") or "")
    namespace = str(manifest.get("namespace") or "")

    if not pack_id:
        errors.append("id must be non-empty")
    if not namespace:
        errors.append("namespace must be non-empty")
    if pack_id and not re.fullmatch(r"[a-z0-9][a-z0-9_.-]*", pack_id):
        errors.append("id must use lowercase letters, digits, dot, underscore, or hyphen")
    if namespace and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", namespace):
        errors.append("namespace must be a valid YARA namespace identifier")

    if not isinstance(manifest.get("enabled"), bool):
        errors.append("enabled must be boolean")
    if not isinstance(manifest.get("trusted"), bool):
        errors.append("trusted must be boolean")
    if not isinstance(manifest.get("allow_includes"), bool):
        errors.append("allow_includes must be boolean")

    source = manifest.get("source")
    if not isinstance(source, dict) or not str(source.get("name") or "").strip():
        errors.append("source.name must be provided")

    license_info = manifest.get("license")
    if not isinstance(license_info, dict) or not str(license_info.get("id") or "").strip():
        errors.append("license.id must be provided")

    if manifest.get("allow_includes") is True:
        warnings.append("includes requested; unified production compilation currently disables includes")

    files = _rule_files(directory)
    if manifest.get("enabled") is True and not files:
        warnings.append("enabled pack contains no .yar/.yara files")

    return errors, warnings


def audit_pack(directory: Path) -> PackAudit:
    manifest = load_manifest(directory)
    errors, warnings = validate_manifest(directory, manifest)
    files = _rule_files(directory)

    rules = 0
    for path in files:
        source = path.read_text(encoding="utf-8", errors="replace")
        rules += len(_RULE_DECL_RE.findall(source))

    source_info = manifest.get("source") if isinstance(manifest.get("source"), dict) else {}
    license_info = manifest.get("license") if isinstance(manifest.get("license"), dict) else {}

    return PackAudit(
        id=str(manifest.get("id") or directory.name),
        namespace=str(manifest.get("namespace") or directory.name),
        enabled=bool(manifest.get("enabled", True)),
        trusted=bool(manifest.get("trusted", False)),
        allow_includes=bool(manifest.get("allow_includes", False)),
        version=str(manifest.get("version") or "unknown"),
        source_name=str(source_info.get("name") or "unknown"),
        license_id=str(license_info.get("id") or "unknown"),
        directory=str(directory),
        rule_files=len(files),
        rules=rules,
        sha256=_sha256_files(directory, files) if files else hashlib.sha256(b"").hexdigest(),
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


def inventory_rule_packs(rules_root: Path) -> dict[str, Any]:
    packs: list[PackAudit] = []
    global_errors: list[str] = []

    if not rules_root.exists():
        return {"ok": False, "errors": [f"rules root not found: {rules_root}"], "packs": []}

    for directory in sorted(p for p in rules_root.iterdir() if p.is_dir() and p.name != "disabled"):
        if not (directory / "pack.json").exists():
            continue
        packs.append(audit_pack(directory))

    ids: dict[str, str] = {}
    namespaces: dict[str, str] = {}

    for pack in packs:
        if pack.id in ids:
            global_errors.append(f"duplicate pack id: {pack.id} ({ids[pack.id]} and {pack.directory})")
        else:
            ids[pack.id] = pack.directory

        if pack.namespace in namespaces:
            global_errors.append(
                f"duplicate namespace: {pack.namespace} ({namespaces[pack.namespace]} and {pack.directory})"
            )
        else:
            namespaces[pack.namespace] = pack.directory

    pack_errors = sum(len(p.errors) for p in packs)

    return {
        "ok": not global_errors and pack_errors == 0,
        "pack_count": len(packs),
        "enabled_pack_count": sum(1 for p in packs if p.enabled),
        "rule_file_count": sum(p.rule_files for p in packs),
        "rule_count": sum(p.rules for p in packs),
        "errors": global_errors,
        "packs": [p.to_dict() for p in packs],
    }
