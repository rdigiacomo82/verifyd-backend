from __future__ import annotations

from copy import deepcopy
from typing import Any

from verifyd_engine.yara.models import YaraScanResult, YaraStatus


def merge_yara_into_security(
    security_result: dict[str, Any],
    yara_result: YaraScanResult,
    *,
    minimum_score: int = 0,
) -> dict[str, Any]:
    """Merge YARA evidence without allowing module failure to masquerade as clean.

    Existing security deductions are preserved. YARA risk_points are a bounded
    contribution computed by YaraRiskPolicy.
    """
    result = deepcopy(security_result)
    modules = result.setdefault("modules", {})
    modules["yara_x"] = yara_result.to_dict()

    base_score = int(result.get("score", 100))
    if yara_result.status == YaraStatus.COMPLETED_WITH_MATCHES:
        result["score"] = max(minimum_score, base_score - yara_result.risk_points)
    else:
        result.setdefault("score", base_score)

    degraded = result.setdefault("degraded_modules", [])
    if yara_result.status in {
        YaraStatus.TIMEOUT,
        YaraStatus.RULESET_LOAD_FAILED,
        YaraStatus.SCAN_ERROR,
        YaraStatus.ENGINE_UNAVAILABLE,
    }:
        if "yara_x" not in degraded:
            degraded.append("yara_x")

    if yara_result.highest_severity in {"high", "critical"}:
        result["verdict"] = "REVIEW"
    return result
