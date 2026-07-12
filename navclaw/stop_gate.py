"""Sanitized lower-layer gate for exposing the VLM STOP request skill."""

from __future__ import annotations

import math
import os
import re
from typing import Any, Dict, Optional, Tuple


_ALIAS_GROUPS = {
    "couch": {"couch", "sofa"},
    "tv_monitor": {"tv", "television", "tv_monitor", "tv monitor", "monitor"},
    "plant": {"plant", "potted_plant", "potted plant"},
}


def _canonical_label(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if raw.startswith("target:"):
        raw = raw.split(":", 1)[1]
    text = re.sub(r"[^a-z0-9]+", "_", raw).strip("_")
    if text.startswith("target_"):
        text = text[len("target_") :]
    for canonical, aliases in _ALIAS_GROUPS.items():
        normalized_aliases = {
            re.sub(r"[^a-z0-9]+", "_", item).strip("_") for item in aliases
        }
        if text in normalized_aliases:
            return canonical
    return text


def _float_env(name: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        value = default
    if not math.isfinite(value):
        value = default
    return max(minimum, min(maximum, value))


def _int_env(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


def stop_thresholds() -> Dict[str, Any]:
    return {
        "min_target_score": _float_env(
            "NAVCLAW_STOP_MIN_TARGET_SCORE", 0.65, 0.0, 1.0
        ),
        "max_distance_m": _float_env(
            "NAVCLAW_STOP_MAX_DISTANCE_M", 1.0, 0.05, 10.0
        ),
        "min_observations": _int_env(
            "NAVCLAW_STOP_MIN_OBSERVATIONS", 2, 1, 1000
        ),
        "half_fov_deg": _float_env(
            "NAVCLAW_STOP_HALF_FOV_DEG", 39.5, 1.0, 180.0
        ),
    }


def _finite(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _observation_count(obj: Dict[str, Any]) -> int:
    value = obj.get("target_observation_count", obj.get("observation_count", 0))
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _evidence(obj: Dict[str, Any]) -> Dict[str, Any]:
    evidence: Dict[str, Any] = {
        "label_hint": str(obj.get("label_hint") or "unknown"),
        "observation_count": _observation_count(obj),
    }
    score = _finite(obj.get("target_score"))
    distance = _finite(obj.get("distance", obj.get("distance_m")))
    theta = _finite(obj.get("relative_theta_deg"))
    if score is not None:
        evidence["target_score"] = round(score, 4)
    if distance is not None:
        evidence["distance_m"] = round(distance, 4)
    if theta is not None:
        evidence["relative_theta_deg"] = round(theta, 2)
    return evidence


def _rank(obj: Dict[str, Any]) -> Tuple[float, int, float]:
    score = _finite(obj.get("target_score"))
    distance = _finite(obj.get("distance", obj.get("distance_m")))
    return (
        score if score is not None else -1.0,
        _observation_count(obj),
        -(distance if distance is not None else 1e9),
    )


def evaluate_stop_gate(candidate_data: Dict[str, Any]) -> Dict[str, Any]:
    """Decide whether STOP may be exposed for the current observation.

    This gate does not terminate the episode. It authorizes the VLM to request
    STOP only when lower-layer target evidence satisfies all configured checks.
    The VLM must still visually confirm the target in the current RGB.
    """

    thresholds = stop_thresholds()
    target = _canonical_label(candidate_data.get("target"))
    detector_map = candidate_data.get("detector_semantic_map") or {}
    objects = detector_map.get("objects") or []
    objects = [obj for obj in objects if isinstance(obj, dict)]

    result: Dict[str, Any] = {
        "eligible": False,
        "reason": "target_unknown",
        "target": target or "unknown",
        "thresholds": thresholds,
        "evidence": None,
        "evaluated_object_count": len(objects),
    }
    if not target or target == "unknown":
        return result
    if not objects:
        result["reason"] = "no_detector_objects"
        return result

    matching = [
        obj for obj in objects if _canonical_label(obj.get("label_hint")) == target
    ]
    if not matching:
        result["reason"] = "no_matching_target"
        return result

    best = max(matching, key=_rank)
    result["evidence"] = _evidence(best)
    score = _finite(best.get("target_score"))
    distance = _finite(best.get("distance", best.get("distance_m")))
    theta = _finite(best.get("relative_theta_deg"))
    count = _observation_count(best)

    if score is None or score < thresholds["min_target_score"]:
        result["reason"] = "target_score_below_threshold"
        return result
    if theta is None or abs(theta) > thresholds["half_fov_deg"]:
        result["reason"] = "target_outside_current_view"
        return result
    if distance is None or distance > thresholds["max_distance_m"]:
        result["reason"] = "target_too_far"
        return result
    if count < thresholds["min_observations"]:
        result["reason"] = "insufficient_observations"
        return result

    result["eligible"] = True
    result["reason"] = "verified_current_target"
    return result
