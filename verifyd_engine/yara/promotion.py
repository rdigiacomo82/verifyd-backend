from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PromotionCandidate:
    source_id: str
    file: str
    rule: str
    score: int
    tier: str
    reasons: tuple[str, ...]
    blockers: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _tier(score: int, blockers: tuple[str, ...]) -> str:
    if blockers:
        return "HOLD"
    if score >= 90:
        return "A"
    if score >= 80:
        return "B"
    if score >= 70:
        return "C"
    return "HOLD"


def score_rule(
    rule: dict[str, Any],
    compile_result: dict[str, Any] | None,
) -> PromotionCandidate:
    score = 50
    reasons: list[str] = []
    blockers: list[str] = []

    if compile_result and compile_result.get("compiled") is True:
        score += 20
        reasons.append("YARA-X file compilation passed")
    else:
        blockers.append("candidate file did not pass YARA-X compilation")

    warnings = (compile_result or {}).get("warnings") or []
    if warnings:
        score -= min(15, 5 * len(warnings))
        reasons.append(f"compile warnings: {len(warnings)}")
    else:
        score += 5
        reasons.append("no compile warnings")

    if rule.get("has_author"):
        score += 5
        reasons.append("author metadata present")
    else:
        score -= 3
        reasons.append("author metadata missing")

    if rule.get("has_description"):
        score += 8
        reasons.append("description metadata present")
    else:
        score -= 5
        reasons.append("description metadata missing")

    if rule.get("has_reference"):
        score += 7
        reasons.append("reference metadata present")
    else:
        score -= 4
        reasons.append("reference metadata missing")

    if rule.get("has_date"):
        score += 5
        reasons.append("date metadata present")
    else:
        score -= 2
        reasons.append("date metadata missing")

    # Candidate source registry already limits this phase to approved sources.
    score += 5
    reasons.append("source approved for candidate ingestion")

    score = max(0, min(100, score))
    blockers_tuple = tuple(blockers)
    return PromotionCandidate(
        source_id=str(rule.get("source_id") or ""),
        file=str(rule.get("file") or ""),
        rule=str(rule.get("rule") or ""),
        score=score,
        tier=_tier(score, blockers_tuple),
        reasons=tuple(reasons),
        blockers=blockers_tuple,
    )


def rank_candidates(
    triage_report: dict[str, Any],
    compile_report: dict[str, Any],
) -> list[PromotionCandidate]:
    compile_by_path = {
        str(item.get("path")): item
        for item in (compile_report.get("results") or [])
    }

    ranked: list[PromotionCandidate] = []
    for rule in (triage_report.get("triage", {}).get("rules") or []):
        ranked.append(score_rule(rule, compile_by_path.get(str(rule.get("file")))))

    ranked.sort(key=lambda x: (-x.score, x.source_id, x.rule))
    return ranked
