"""Strict observation and action contracts for NavClaw."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Sequence


class ContractError(ValueError):
    """Raised when an observation or model action violates the contract."""


def _finite_number(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number or number in (float("inf"), float("-inf")):
        return default
    return number


def is_projected(candidate: Dict[str, Any]) -> bool:
    if candidate.get("projected") is True:
        return True
    projection = candidate.get("projection")
    return isinstance(projection, (list, tuple)) and len(projection) >= 2


def selectable_candidates(candidates: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return the only candidates that the upper agent may select.

    Coordinates are intentionally omitted from the returned records.
    """

    selected: List[Dict[str, Any]] = []
    seen = set()
    for candidate in candidates or []:
        candidate_id = str(candidate.get("id") or "").strip()
        if not candidate_id or candidate_id in seen:
            continue
        if candidate.get("reachable", True) is not True:
            continue
        if candidate.get("valid", True) is False:
            continue
        if not is_projected(candidate):
            continue
        seen.add(candidate_id)
        selected.append(
            {
                "id": candidate_id,
                "source": str(candidate.get("source") or "unknown"),
                "direction": str(candidate.get("direction") or "unknown"),
                "distance_m": round(_finite_number(candidate.get("distance"), -1.0), 3),
                "path_length_m": round(_finite_number(candidate.get("path_length"), -1.0), 3),
                "path_ratio": round(_finite_number(candidate.get("path_ratio"), -1.0), 3),
                "clearance_m": round(_finite_number(candidate.get("clearance"), -1.0), 3),
                "score": round(_finite_number(candidate.get("score"), 0.0), 4),
                "visited_recently": bool(candidate.get("visited_recently", False)),
                "projection_type": str(candidate.get("projection_type") or "unknown"),
                "projected": True,
            }
        )
    return selected


def valid_candidate_ids(candidates: Sequence[Dict[str, Any]]) -> List[str]:
    return [str(candidate["id"]) for candidate in selectable_candidates(candidates)]


def compact_observation(data: Dict[str, Any]) -> Dict[str, Any]:
    """Remove coordinates, dense grids, and private controller state."""

    metric_map = data.get("metric_map_summary") or {}
    detector_map = data.get("detector_semantic_map") or {}
    objects = []
    for obj in (detector_map.get("objects") or [])[:12]:
        objects.append(
            {
                "label_hint": str(obj.get("label_hint") or "unknown"),
                "relative_theta_deg": round(_finite_number(obj.get("relative_theta_deg")), 1),
                "distance_m": round(_finite_number(obj.get("distance"), -1.0), 2),
                "target_score": round(_finite_number(obj.get("target_score")), 3),
                "observation_count": int(obj.get("target_observation_count") or 0),
            }
        )
    task_type = str(data.get("task_type") or "objectnav")
    instruction = str(data.get("instruction") or data.get("task_goal") or "")
    target = str(data.get("target") or "unknown")
    if task_type.lower().startswith("vln"):
        objects = []
    compact = {
        "target": target,
        "task_type": task_type,
        "instruction": instruction,
        "task": instruction or target,
        "episode": data.get("episode"),
        "step": data.get("step"),
        "mode": str(data.get("mode") or "unknown"),
        "robot": {
            "yaw": round(_finite_number((data.get("robot") or {}).get("yaw")), 4),
        },
        "scan_context": {
            "active": bool((data.get("scan_context") or {}).get("active", False)),
            "scan_count": int((data.get("scan_context") or {}).get("scan_count") or 0),
            "cumulative_angle_deg": round(
                _finite_number((data.get("scan_context") or {}).get("cumulative_angle_deg")), 1
            ),
            "last_action": (data.get("scan_context") or {}).get("last_action"),
        },
        "map_summary": {
            "available": bool(metric_map.get("available", False)),
            "explored_ratio": round(_finite_number(metric_map.get("explored_ratio")), 5),
            "cell_counts": metric_map.get("cell_counts") or {},
            "area_m2": metric_map.get("area_m2") or {},
            "frontiers": metric_map.get("frontiers") or {},
        },
        "semantic_objects": objects,
        "candidates": selectable_candidates(data.get("candidates") or []),
    }
    return compact


def extract_json_object(raw: str) -> Dict[str, Any]:
    """Parse one JSON object without inventing missing fields."""

    text = (raw or "").strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    parsed = json.loads(text.strip())
    if not isinstance(parsed, dict):
        raise ContractError("model_output_must_be_json_object")
    return parsed


def normalize_agent_action(
    model_output: Dict[str, Any],
    allowed_candidate_ids: Sequence[str],
    allow_stop: bool = False,
    force_select_waypoint: bool = False,
) -> Dict[str, Any]:
    """Validate one AerialClaw-style skill call and map it to the C++ contract."""

    if model_output.get("fallback") is True:
        raise ContractError("model_must_not_request_fallback")

    decision = str(model_output.get("decision") or "").strip().lower()
    if decision != "act":
        if allow_stop and decision in {"done", "stuck"}:
            skill = "stop"
            parameters: Dict[str, Any] = {}
        else:
            raise ContractError("decision_must_be_act")
    else:
        action = model_output.get("action")
        if not isinstance(action, dict):
            raise ContractError("action_must_be_object")
        skill = str(action.get("skill") or "").strip().lower()
        parameters = action.get("parameters") or {}
        if not isinstance(parameters, dict):
            raise ContractError("action_parameters_must_be_object")

    allowed = set(str(value) for value in allowed_candidate_ids)
    selected = None
    if skill == "select_waypoint":
        selected = str(parameters.get("candidate_id") or "").strip()
        if not selected:
            raise ContractError("select_waypoint_requires_candidate_id")
        if selected not in allowed:
            raise ContractError("selected_candidate_not_in_current_observation")
        cpp_decision = "SELECT_WAYPOINT"
    elif skill == "look_left_60":
        cpp_decision = "LOOK_LEFT_60"
    elif skill == "look_right_60":
        cpp_decision = "LOOK_RIGHT_60"
    elif skill == "stop" and allow_stop:
        cpp_decision = "STOP"
    else:
        raise ContractError("skill_not_in_current_registry")
    if force_select_waypoint and cpp_decision != "SELECT_WAYPOINT":
        raise ContractError("current_mode_requires_select_waypoint")

    confidence = max(0.0, min(1.0, _finite_number(model_output.get("confidence"), 0.0)))
    thinking = str(model_output.get("thinking") or "").strip()
    reflection = model_output.get("reflection")
    progress = str(model_output.get("goal_progress") or "").strip()
    return {
        "decision": cpp_decision,
        "selected": selected,
        "selected_id": selected,
        "reason": thinking,
        "confidence": confidence,
        "fallback": False,
        "fallback_reason": "",
        "front_scene_type": "navclaw_agent_decision",
        "candidate_analysis": [],
        "navclaw": {
            "skill": skill,
            "reflection": reflection,
            "goal_progress": progress,
            "strict_candidate_constraint": True,
            "rule_fallback_enabled": False,
        },
    }
