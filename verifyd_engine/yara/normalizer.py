from __future__ import annotations

import re
from typing import Any, Iterable

from .models import MatchEvidence, RuleMatch


_SEVERITY_ALIASES = {
    "info": "informational",
    "informational": "informational",
    "low": "low",
    "medium": "medium",
    "med": "medium",
    "moderate": "medium",
    "high": "high",
    "critical": "critical",
    "crit": "critical",
}


def _metadata_dict(metadata: Iterable[tuple[str, Any]] | dict[str, Any] | None) -> dict[str, Any]:
    if metadata is None:
        return {}
    if isinstance(metadata, dict):
        return dict(metadata)
    return {str(k): v for k, v in metadata}


def _first(meta: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in meta and meta[key] not in (None, ""):
            return meta[key]
    return default


def normalize_severity(value: Any) -> str:
    if isinstance(value, (int, float)):
        if value >= 9:
            return "critical"
        if value >= 7:
            return "high"
        if value >= 4:
            return "medium"
        if value > 0:
            return "low"
        return "informational"
    text = str(value or "medium").strip().lower()
    return _SEVERITY_ALIASES.get(text, "medium")


def normalize_confidence(value: Any, *, default: int = 70) -> int:
    try:
        v = float(value)
        if 0 <= v <= 1:
            v *= 100
        return max(0, min(100, int(round(v))))
    except (TypeError, ValueError):
        return default


def normalize_tags(rule: Any, meta: dict[str, Any]) -> tuple[str, ...]:
    # YARA-X's documented Python Rule API exposes identifier, namespace,
    # patterns and metadata. Some builds may expose tags too, so use them if
    # present; otherwise VeriFYD rules mirror tags in the "tags" metadata.
    raw = getattr(rule, "tags", None)
    if raw:
        return tuple(str(x) for x in raw)
    raw_meta = _first(meta, "tags", "tag", default="")
    if isinstance(raw_meta, str):
        return tuple(x for x in re.split(r"[,;\s]+", raw_meta.strip()) if x)
    if isinstance(raw_meta, (list, tuple, set)):
        return tuple(str(x) for x in raw_meta)
    return ()


def normalize_references(meta: dict[str, Any]) -> tuple[str, ...]:
    raw = _first(meta, "references", "reference", "ref", default="")
    if isinstance(raw, str):
        return tuple(x.strip() for x in re.split(r"[,;]", raw) if x.strip())
    if isinstance(raw, (list, tuple, set)):
        return tuple(str(x) for x in raw)
    return ()


def normalize_rule_match(rule: Any, *, evidence_limit: int = 20) -> RuleMatch:
    meta = _metadata_dict(getattr(rule, "metadata", None))
    severity = normalize_severity(_first(meta, "severity", "level", "threat_level", "score", default="medium"))
    category = str(_first(meta, "category", "type", "threat_type", default="unclassified"))
    confidence = normalize_confidence(_first(meta, "confidence", "confidence_score", default=70))
    description = str(_first(meta, "description", "desc", default=f"YARA-X rule {rule.identifier} matched"))
    source = str(_first(meta, "source", "author", default=getattr(rule, "namespace", "unknown")))
    family = _first(meta, "family", "malware_family", default=None)
    dedupe_key = _first(meta, "dedupe_key", "technique", "behavior", default=None)

    evidence: list[MatchEvidence] = []
    for pattern in getattr(rule, "patterns", ()):
        for match in getattr(pattern, "matches", ()):
            if len(evidence) >= evidence_limit:
                break
            evidence.append(
                MatchEvidence(
                    pattern=str(getattr(pattern, "identifier", "$unknown")),
                    offset=int(getattr(match, "offset", 0)),
                    length=int(getattr(match, "length", 0)),
                    xor_key=getattr(match, "xor_key", None),
                )
            )
        if len(evidence) >= evidence_limit:
            break

    return RuleMatch(
        rule=str(rule.identifier),
        namespace=str(getattr(rule, "namespace", "default")),
        severity=severity,
        category=category,
        confidence=confidence,
        description=description,
        source=source,
        family=str(family) if family is not None else None,
        tags=normalize_tags(rule, meta),
        dedupe_key=str(dedupe_key) if dedupe_key is not None else None,
        references=normalize_references(meta),
        evidence=tuple(evidence),
        raw_metadata=meta,
    )
