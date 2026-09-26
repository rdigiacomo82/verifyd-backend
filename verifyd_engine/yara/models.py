from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class YaraStatus(str, Enum):
    COMPLETED_NO_MATCHES = "SCAN_COMPLETED_NO_MATCHES"
    COMPLETED_WITH_MATCHES = "SCAN_COMPLETED_WITH_MATCHES"
    TIMEOUT = "SCAN_TIMEOUT"
    RULESET_LOAD_FAILED = "RULESET_LOAD_FAILED"
    SCAN_ERROR = "SCAN_ERROR"
    NOT_SCANNED = "NOT_SCANNED"
    DISABLED = "DISABLED"
    ENGINE_UNAVAILABLE = "ENGINE_UNAVAILABLE"


@dataclass(frozen=True)
class MatchEvidence:
    pattern: str
    offset: int
    length: int
    xor_key: int | None = None


@dataclass(frozen=True)
class RuleMatch:
    rule: str
    namespace: str
    severity: str
    category: str
    confidence: int
    description: str
    source: str
    family: str | None = None
    tags: tuple[str, ...] = ()
    dedupe_key: str | None = None
    references: tuple[str, ...] = ()
    evidence: tuple[MatchEvidence, ...] = ()
    raw_metadata: dict[str, Any] = field(default_factory=dict)
    risk_points: int = 0


@dataclass(frozen=True)
class RulesetInfo:
    version: str
    sha256: str
    rules_loaded: int
    rule_files_loaded: int
    namespaces: tuple[str, ...]
    compile_warnings: tuple[dict[str, Any], ...] = ()
    loaded_at: str | None = None


@dataclass(frozen=True)
class YaraScanResult:
    module: str = "yara_x"
    status: YaraStatus = YaraStatus.NOT_SCANNED
    engine_version: str | None = None
    ruleset: RulesetInfo | None = None
    scan_duration_ms: int | None = None
    match_count: int = 0
    highest_severity: str = "none"
    risk_points: int = 0
    recommended_action: str = "none"
    matches: tuple[RuleMatch, ...] = ()
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        return payload
