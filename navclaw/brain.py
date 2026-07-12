"""Persistent one-action-per-observation NavClaw brain."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .contracts import (
    ContractError,
    extract_json_object,
    normalize_agent_action,
    normalize_execution_feedback,
)
from .llm_client import LLMClient, LLMError
from .memory import SessionMemory
from .skills import SkillRegistry


class DecisionError(RuntimeError):
    pass


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


class NavClawBrain:
    def __init__(self, project_root: Path, run_dir: Path):
        self.project_root = Path(project_root)
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.memory = SessionMemory(
            self.run_dir / "sessions",
            max_prompt_items=_int_env("NAVCLAW_MEMORY_PROMPT_ITEMS", 8),
        )
        self.skill_registry = SkillRegistry()
        self.llm = LLMClient(self.run_dir)
        self.max_decision_attempts = max(1, _int_env("NAVCLAW_DECISION_ATTEMPTS", 3))
        self.decision_log = self.run_dir / "decisions.jsonl"
        self._log_lock = threading.Lock()
        self._profiles = self._load_profiles()

    def _load_profiles(self) -> Dict[str, str]:
        profiles = {}
        for name in ("SOUL.md", "BODY.md", "MEMORY.md"):
            path = self.project_root / "profiles" / name
            profiles[name] = path.read_text(encoding="utf-8").strip() if path.exists() else ""
        return profiles

    def _append_decision_log(self, record: Dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False, sort_keys=True)
        with self._log_lock:
            with self.decision_log.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def _messages(
        self,
        observation: Dict[str, Any],
        skills: List[Dict[str, Any]],
        history: List[Dict[str, Any]],
        image_data_uri: Optional[str],
    ) -> List[Dict[str, Any]]:
        system_prompt = """You are NavClaw, the upper-level agent for a ground navigation robot.
You operate in a strict observe-think-act loop. Return exactly one action for the current observation.
You may use only a skill listed in CURRENT_SKILLS. Candidate IDs are ephemeral: a waypoint may be selected only when its exact ID is listed in CURRENT_CANDIDATES for this observation.
Never output coordinates, safe_goal values, old candidate IDs, MapGPT Place IDs, PX4/AirSim actions, or an unlisted skill.
Do not claim task completion. The environment owns success detection. Do not use deterministic or geometric fallback reasoning.
RECENT_EXECUTION_MEMORY contains observed lower-layer outcomes when available. Treat reached and stalled_or_aborted as environment evidence, not as model speculation.
Return exactly one object in json format with this schema:
{"thinking":"brief first-person reasoning","decision":"act","action":{"skill":"one listed skill","parameters":{}},"reflection":"lesson from prior execution or null","goal_progress":"brief status","confidence":0.0}
For select_waypoint, parameters must be {"candidate_id":"one exact current ID"}. LOOK and STOP skills take empty parameters. Do not add any other fields.
"""
        if self._profiles.get("SOUL.md"):
            system_prompt += "\nIDENTITY:\n" + self._profiles["SOUL.md"]
        if self._profiles.get("BODY.md"):
            system_prompt += "\nCAPABILITY_BOUNDARY:\n" + self._profiles["BODY.md"]
        user_text = "Return exactly one object in json format for this observation.\n" + json.dumps(
            {
                "CURRENT_OBSERVATION": observation,
                "CURRENT_CANDIDATES": [candidate.get("id") for candidate in observation.get("candidates", [])],
                "CURRENT_SKILLS": skills,
                "RECENT_EXECUTION_MEMORY": history,
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        content: List[Dict[str, Any]] = [{"type": "text", "text": user_text}]
        if image_data_uri:
            content.extend(
                [
                    {
                        "type": "text",
                        "text": (
                            "Current annotated RGB observation. Every drawn waypoint label is also "
                            "present in CURRENT_CANDIDATES, and no other waypoint is selectable."
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": image_data_uri}},
                ]
            )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]

    def decide(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        request_id = str(payload.get("request_id") or "unknown")
        session_id = str(payload.get("session_id") or "unknown")
        observation = payload.get("observation")
        if not isinstance(observation, dict):
            raise DecisionError("observation_must_be_object")

        feedback_recorded = False
        normalized_feedback = None
        if payload.get("execution_feedback") is not None:
            try:
                normalized_feedback = normalize_execution_feedback(payload["execution_feedback"])
            except ContractError as exc:
                raise DecisionError(str(exc)) from exc
            if normalized_feedback.get("next_request_id") != request_id:
                raise DecisionError("execution_feedback_next_request_id_mismatch")
            feedback_recorded = self.memory.append_feedback(session_id, normalized_feedback)

        candidates = observation.get("candidates") or []
        candidate_ids = [str(candidate.get("id")) for candidate in candidates if candidate.get("id")]
        allow_stop = bool(payload.get("allow_stop", False))
        force_select_waypoint = bool(payload.get("force_select_waypoint", False))
        if force_select_waypoint and not candidate_ids:
            raise DecisionError("force_select_waypoint_without_candidates")
        skill_defs = self.skill_registry.for_observation(
            candidate_ids,
            allow_stop=allow_stop and not force_select_waypoint,
            allow_look=not force_select_waypoint,
        )
        skills = self.skill_registry.prompt_catalog(skill_defs)
        history = self.memory.recent(session_id)
        messages = self._messages(
            observation,
            skills,
            history,
            payload.get("image_data_uri"),
        )

        failures: List[Dict[str, Any]] = []
        request_body_hash: Optional[str] = None
        total_api_attempts = 0
        for decision_attempt in range(1, self.max_decision_attempts + 1):
            raw = ""
            try:
                raw, body_hash, api_attempts = self.llm.chat(messages, request_id=request_id)
                total_api_attempts += api_attempts
                if request_body_hash is None:
                    request_body_hash = body_hash
                elif request_body_hash != body_hash:
                    raise DecisionError("retry_request_body_changed")
                parsed = extract_json_object(raw)
                result = normalize_agent_action(
                    parsed,
                    candidate_ids,
                    allow_stop=allow_stop,
                    force_select_waypoint=force_select_waypoint,
                )
                result["navclaw"].update(
                    {
                        "request_id": request_id,
                        "session_id": session_id,
                        "decision_attempt": decision_attempt,
                        "api_attempt_count": total_api_attempts,
                        "request_body_sha256": body_hash,
                        "current_candidate_ids": candidate_ids,
                        "force_select_waypoint": force_select_waypoint,
                        "execution_feedback_received": normalized_feedback is not None,
                        "execution_feedback_recorded": feedback_recorded,
                    }
                )
                record = {
                    "event_type": "decision",
                    "status": "ok",
                    "request_id": request_id,
                    "session_id": session_id,
                    "target": observation.get("target"),
                    "step": observation.get("step"),
                    "valid_candidate_ids": candidate_ids,
                    "force_select_waypoint": force_select_waypoint,
                    "decision_attempt": decision_attempt,
                    "api_attempt_count": total_api_attempts,
                    "request_body_sha256": body_hash,
                    "raw_model_output": raw,
                    "result": result,
                    "failures_before_success": failures,
                    "execution_feedback": normalized_feedback,
                    "fallback": False,
                }
                self._append_decision_log(record)
                self.memory.append(session_id, record)
                return result
            except (ContractError, json.JSONDecodeError, DecisionError) as exc:
                failures.append(
                    {
                        "decision_attempt": decision_attempt,
                        "kind": "invalid_model_action",
                        "error": str(exc),
                        "raw_model_output": raw[:4000],
                    }
                )
            except LLMError as exc:
                failures.append(
                    {
                        "decision_attempt": decision_attempt,
                        "kind": "llm_error",
                        "error": str(exc),
                    }
                )

        record = {
            "event_type": "decision",
            "status": "error",
            "request_id": request_id,
            "session_id": session_id,
            "target": observation.get("target"),
            "step": observation.get("step"),
            "valid_candidate_ids": candidate_ids,
            "force_select_waypoint": force_select_waypoint,
            "request_body_sha256": request_body_hash,
            "failures": failures,
            "execution_feedback": normalized_feedback,
            "fallback": False,
            "error": "no_valid_model_action_after_attempts",
            "created_at": time.time(),
        }
        self._append_decision_log(record)
        self.memory.append(session_id, record)
        raise DecisionError("no_valid_model_action_after_attempts")
