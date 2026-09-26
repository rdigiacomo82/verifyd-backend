from __future__ import annotations

from dataclasses import replace
from typing import Iterable

from .models import RuleMatch


_SEVERITY_RANK = {
    "none": 0,
    "informational": 1,
    "low": 2,
    "medium": 3,
    "high": 4,
    "critical": 5,
}


class YaraRiskPolicy:
    """Converts YARA evidence into a bounded VeriFYD security contribution.

    Rule authors never directly control the final score. Rule metadata is
    normalized, confidence-weighted and deduplicated here.
    """

    DEFAULT_POINTS = {
        "informational": 0,
        "low": 3,
        "medium": 8,
        "high": 18,
        "critical": 30,
    }

    def __init__(
        self,
        *,
        severity_points: dict[str, int] | None = None,
        max_total_points: int = 45,
        corroboration_points: int = 3,
        max_corroboration_per_cluster: int = 6,
    ) -> None:
        self.severity_points = dict(self.DEFAULT_POINTS)
        if severity_points:
            self.severity_points.update(severity_points)
        self.max_total_points = max_total_points
        self.corroboration_points = corroboration_points
        self.max_corroboration_per_cluster = max_corroboration_per_cluster

    @staticmethod
    def confidence_weight(confidence: int) -> float:
        if confidence >= 90:
            return 1.0
        if confidence >= 75:
            return 0.8
        if confidence >= 50:
            return 0.6
        return 0.0

    def score_match(self, match: RuleMatch) -> int:
        base = self.severity_points.get(match.severity, self.severity_points["medium"])
        return int(round(base * self.confidence_weight(match.confidence)))

    @staticmethod
    def cluster_key(match: RuleMatch) -> str:
        # Explicit rule metadata wins. Then malware family/category groups
        # related rules so public packs do not linearly multiply risk.
        if match.dedupe_key:
            return f"dedupe:{match.dedupe_key.lower()}"
        if match.family:
            return f"family:{match.family.lower()}"
        if match.category and match.category != "unclassified":
            return f"category:{match.category.lower()}"
        return f"rule:{match.namespace.lower()}:{match.rule.lower()}"

    def apply(self, matches: Iterable[RuleMatch]) -> tuple[tuple[RuleMatch, ...], int, str, str]:
        enriched = tuple(replace(m, risk_points=self.score_match(m)) for m in matches)
        if not enriched:
            return enriched, 0, "none", "none"

        clusters: dict[str, list[RuleMatch]] = {}
        highest = "none"
        for match in enriched:
            clusters.setdefault(self.cluster_key(match), []).append(match)
            if _SEVERITY_RANK.get(match.severity, 0) > _SEVERITY_RANK.get(highest, 0):
                highest = match.severity

        total = 0
        for group in clusters.values():
            group_points = sorted((m.risk_points for m in group), reverse=True)
            if not group_points:
                continue
            primary = group_points[0]
            corroboration_count = sum(1 for p in group_points[1:] if p > 0)
            corroboration = min(
                self.max_corroboration_per_cluster,
                corroboration_count * self.corroboration_points,
            )
            total += primary + corroboration

        total = min(self.max_total_points, total)
        if highest in {"critical", "high"} or total >= 18:
            action = "review"
        elif total > 0:
            action = "monitor"
        else:
            action = "none"
        return enriched, total, highest, action
