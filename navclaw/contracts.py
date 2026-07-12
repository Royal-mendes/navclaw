"""Strict observation, action, and execution-feedback contracts for NavClaw."""

from __future__ import annotations

import json
import math
from typing import Any, Dict, Iterable, List, Optional, Sequence


class ContractError(ValueError):
    """Raised when an observation, model action, or feedback violates the contract."""


def _finite_number(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return number


def _strict_finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError("{0}_must_be_number".format(field))
    number = float(value)
    if not math.isfinite(number):
        raise ContractError("{0}_must_be_finite".format(field))
    return number


def projection_xy(candidate: Dict[str, Any]) -> Optional[List[float]]:
    """Return a finite image projection or None.

    A bare ``projected=true`` flag is insufficient because the bridge must be
    able to draw the exact candidate that is exposed to the model.
    """

    projection = candidate.get("projection")
    if not isinstance(projection, (list, tuple)) or len(projection) < 2:
        return None
    try:
        x = float(projection[0])
        y = float(projection[1])
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    return [x, y]


def is_projected(candidate: Dict[str, Any]) -> bool:
    return projection_xy(candidate) is not None


def filter_selectable_candidates(
    candidates: Iterable[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Return original candidate records that satisfy the action contract."""

    selected: List[Dict[str, Any]] = []
    seen = set()
    for candidate in candidates or []:
        if not isinstance(candidate, dict):
            continue
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
        selected.append(candidate)
    return selected


def selectable_candidates(candidates: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return compact candidates that the upper agent may select.

    Coordinates and pixel locations are intentionally omitted from the model
    observation. The bridge retains them privately for annotation and execution
    feedback only.
    """

    selected: List[Dict[str, Any]] = []
    for candidate in filter_selectable_candidates(candidates):
        candidate_id = str(candidate.get("id") or "").strip()
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
    return {
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


def _validate_optional_text(model_output: Dict[str, Any], field: str) -> None:
    if field in model_output and not isinstance(model_output[field], str):
        raise ContractError("{0}_must_be_string".format(field))


def normalize_agent_action(
    model_output: Dict[str, Any],
    allowed_candidate_ids: Sequence[str],
    allow_stop: bool = False,
    force_select_waypoint: bool = False,
) -> Dict[str, Any]:
    """Validate one strict skill call and map it to the C++ contract."""

    allowed_top_level = {
        "thinking",
        "decision",
        "action",
        "reflection",
        "goal_progress",
        "confidence",
    }
    unknown_top_level = set(model_output) - allowed_top_level
    if unknown_top_level:
        raise ContractError(
            "model_output_has_unknown_fields:{0}".format(
                ",".join(sorted(str(value) for value in unknown_top_level))
            )
        )
    if "decision" not in model_output or "action" not in model_output:
        raise ContractError("model_output_requires_decision_and_action")
    _validate_optional_text(model_output, "thinking")
    _validate_optional_text(model_output, "goal_progress")
    if "reflection" in model_output and model_output["reflection"] is not None and not isinstance(
        model_output["reflection"], str
    ):
        raise ContractError("reflection_must_be_string_or_null")

    decision = model_output.get("decision")
    if not isinstance(decision, str) or decision.strip().lower() != "act":
        raise ContractError("decision_must_be_act")

    action = model_output.get("action")
    if not isinstance(action, dict):
        raise ContractError("action_must_be_object")
    unknown_action_fields = set(action) - {"skill", "parameters"}
    if unknown_action_fields:
        raise ContractError(
            "action_has_unknown_fields:{0}".format(
                ",".join(sorted(str(value) for value in unknown_action_fields))
            )
        )
    if set(action) != {"skill", "parameters"}:
        raise ContractError("action_requires_skill_and_parameters")

    raw_skill = action.get("skill")
    if not isinstance(raw_skill, str):
        raise ContractError("skill_must_be_string")
    skill = raw_skill.strip().lower()
    parameters = action.get("parameters")
    if not isinstance(parameters, dict):
        raise ContractError("action_parameters_must_be_object")

    allowed = set(str(value) for value in allowed_candidate_ids)
    selected = None
    if skill == "select_waypoint":
        if set(parameters) != {"candidate_id"}:
            raise ContractError("select_waypoint_parameters_must_only_contain_candidate_id")
        candidate_id = parameters.get("candidate_id")
        if not isinstance(candidate_id, str):
            raise ContractError("candidate_id_must_be_string")
        selected = candidate_id.strip()
        if not selected:
            raise ContractError("select_waypoint_requires_candidate_id")
        if selected not in allowed:
            raise ContractError("selected_candidate_not_in_current_observation")
        cpp_decision = "SELECT_WAYPOINT"
    elif skill in {"look_left_60", "look_right_60"}:
        if parameters:
            raise ContractError("look_parameters_must_be_empty")
        cpp_decision = "LOOK_LEFT_60" if skill == "look_left_60" else "LOOK_RIGHT_60"
    elif skill == "stop" and allow_stop:
        if parameters:
            raise ContractError("stop_parameters_must_be_empty")
        cpp_decision = "STOP"
    else:
        raise ContractError("skill_not_in_current_registry")
    if force_select_waypoint and cpp_decision != "SELECT_WAYPOINT":
        raise ContractError("current_mode_requires_select_waypoint")

    confidence = 0.0
    if "confidence" in model_output:
        confidence = _strict_finite_number(model_output["confidence"], "confidence")
        if confidence < 0.0 or confidence > 1.0:
            raise ContractError("confidence_must_be_between_zero_and_one")
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


def normalize_execution_feedback(feedback: Dict[str, Any]) -> Dict[str, Any]:
    """Validate bridge-observed execution feedback before it enters memory."""

    if not isinstance(feedback, dict):
        raise ContractError("execution_feedback_must_be_object")
    required = {
        "feedback_id",
        "request_id",
        "next_request_id",
        "selected_id",
        "execution_outcome",
        "source",
    }
    optional_numeric = {
        "final_distance_m",
        "start_distance_m",
        "travel_distance_m",
        "elapsed_s",
        "reached_threshold_m",
    }
    optional_any = {"issued_step", "observed_step"}
    allowed = required | optional_numeric | optional_any
    unknown = set(feedback) - allowed
    if unknown:
        raise ContractError(
            "execution_feedback_has_unknown_fields:{0}".format(
                ",".join(sorted(str(value) for value in unknown))
            )
        )
    missing = required - set(feedback)
    if missing:
        raise ContractError(
            "execution_feedback_missing_fields:{0}".format(
                ",".join(sorted(str(value) for value in missing))
            )
        )

    normalized: Dict[str, Any] = {}
    for field in required - {"execution_outcome"}:
        value = feedback.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ContractError("{0}_must_be_nonempty_string".format(field))
        normalized[field] = value.strip()
    outcome = feedback.get("execution_outcome")
    if outcome not in {"reached", "stalled_or_aborted", "unknown"}:
        raise ContractError("execution_outcome_not_supported")
    normalized["execution_outcome"] = outcome

    for field in optional_numeric:
        if field in feedback and feedback[field] is not None:
            normalized[field] = round(_strict_finite_number(feedback[field], field), 4)
    for field in optional_any:
        if field in feedback:
            normalized[field] = feedback[field]
    return normalized
