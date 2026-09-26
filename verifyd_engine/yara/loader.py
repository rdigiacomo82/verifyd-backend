from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import RulesetInfo


_RULE_DECL_RE = re.compile(r"(?m)^\s*(?:global\s+|private\s+)*rule\s+[A-Za-z_][A-Za-z0-9_]*")


@dataclass(frozen=True)
class RulePack:
    namespace: str
    directory: Path
    files: tuple[Path, ...]
    trusted: bool = False
    allow_includes: bool = False


@dataclass(frozen=True)
class CompiledBundle:
    rules: Any
    info: RulesetInfo


def _pack_config(directory: Path) -> dict[str, Any]:
    path = directory / "pack.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def discover_rule_packs(rules_root: Path) -> tuple[RulePack, ...]:
    if not rules_root.exists():
        return ()
    packs: list[RulePack] = []
    for directory in sorted(p for p in rules_root.iterdir() if p.is_dir() and p.name != "disabled"):
        cfg = _pack_config(directory)
        if cfg.get("enabled", True) is False:
            continue
        files = tuple(sorted(set(directory.rglob("*.yar")) | set(directory.rglob("*.yara"))))
        if not files:
            continue
        packs.append(
            RulePack(
                namespace=str(cfg.get("namespace") or directory.name),
                directory=directory,
                files=files,
                trusted=bool(cfg.get("trusted", False)),
                allow_includes=bool(cfg.get("allow_includes", False)),
            )
        )
    return tuple(packs)


def _ruleset_digest(packs: tuple[RulePack, ...]) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    rule_count = 0
    file_count = 0
    for pack in packs:
        digest.update(pack.namespace.encode("utf-8"))
        for path in pack.files:
            source = path.read_text(encoding="utf-8")
            digest.update(str(path.relative_to(pack.directory)).encode("utf-8"))
            digest.update(source.encode("utf-8"))
            file_count += 1
            rule_count += len(_RULE_DECL_RE.findall(source))
    return digest.hexdigest(), rule_count, file_count


def compile_rule_packs(yara_x: Any, rules_root: Path, *, version: str | None = None) -> CompiledBundle:
    packs = discover_rule_packs(rules_root)
    if not packs:
        raise RuntimeError(f"No enabled YARA rule packs found under {rules_root}")

    compiler = yara_x.Compiler()
    if hasattr(compiler, "max_warnings"):
        compiler.max_warnings(100)

    # Phase 2A policy: includes remain disabled for the unified compilation.
    # Phase 2B can pre-resolve/vet include-dependent third-party packs before
    # production promotion. This avoids implicit filesystem dependencies.
    compiler.enable_includes(False)

    for pack in packs:
        compiler.new_namespace(pack.namespace)
        for path in pack.files:
            compiler.add_source(path.read_text(encoding="utf-8"), origin=str(path))

    warnings = tuple(compiler.warnings()) if hasattr(compiler, "warnings") else ()
    rules = compiler.build()
    sha256, rule_count, file_count = _ruleset_digest(packs)
    loaded_at = datetime.now(timezone.utc).isoformat()
    version = version or datetime.now(timezone.utc).strftime("%Y.%m.%d.%H%M")
    info = RulesetInfo(
        version=version,
        sha256=sha256,
        rules_loaded=rule_count,
        rule_files_loaded=file_count,
        namespaces=tuple(pack.namespace for pack in packs),
        compile_warnings=warnings,
        loaded_at=loaded_at,
    )
    return CompiledBundle(rules=rules, info=info)
