"""Auditable session memory with execution-feedback correlation."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, List


class SessionMemory:
    def __init__(self, root: Path, max_prompt_items: int = 8):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_prompt_items = max(0, int(max_prompt_items))
        self._records: Dict[str, List[Dict[str, Any]]] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _safe_session_id(session_id: str) -> str:
        cleaned = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in session_id)
        return cleaned[:120] or "unknown"

    def _path(self, session_id: str) -> Path:
        return self.root / (self._safe_session_id(session_id) + ".jsonl")

    def _load_locked(self, session_id: str) -> List[Dict[str, Any]]:
        records = self._records.get(session_id)
        if records is not None:
            return records
        records = []
        path = self._path(session_id)
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    records.append(value)
        self._records[session_id] = records
        return records

    def recent(self, session_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            records = list(self._load_locked(session_id))
            feedback_by_request: Dict[str, Dict[str, Any]] = {}
            decisions = []
            for record in records:
                if record.get("event_type") == "execution_feedback":
                    request_id = str(record.get("request_id") or "")
                    if request_id:
                        feedback_by_request[request_id] = record
                elif record.get("event_type") == "decision" or "result" in record:
                    decisions.append(record)

            prompt_records = []
            for record in decisions[-self.max_prompt_items :]:
                request_id = str(record.get("request_id") or "")
                feedback = feedback_by_request.get(request_id)
                feedback_summary = None
                if feedback:
                    feedback_summary = {
                        "execution_outcome": feedback.get("execution_outcome"),
                        "final_distance_m": feedback.get("final_distance_m"),
                        "start_distance_m": feedback.get("start_distance_m"),
                        "travel_distance_m": feedback.get("travel_distance_m"),
                        "elapsed_s": feedback.get("elapsed_s"),
                        "source": feedback.get("source"),
                    }
                prompt_records.append(
                    {
                        "request_id": request_id,
                        "step": record.get("step"),
                        "target": record.get("target"),
                        "valid_candidate_ids": record.get("valid_candidate_ids") or [],
                        "decision": (record.get("result") or {}).get("decision"),
                        "selected": (record.get("result") or {}).get("selected"),
                        "reason": (record.get("result") or {}).get("reason"),
                        "decision_status": "issued" if record.get("status") == "ok" else "failed",
                        "execution_outcome": (
                            feedback.get("execution_outcome")
                            if feedback
                            else "unknown_not_yet_observed"
                        ),
                        "execution_feedback": feedback_summary,
                    }
                )
            return prompt_records

    def _append_locked(self, session_id: str, item: Dict[str, Any]) -> None:
        self._load_locked(session_id).append(item)
        line = json.dumps(item, ensure_ascii=False, sort_keys=True)
        with self._path(session_id).open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def append(self, session_id: str, record: Dict[str, Any]) -> None:
        item = dict(record)
        item.setdefault("event_type", "decision")
        item.setdefault("created_at", time.time())
        with self._lock:
            self._append_locked(session_id, item)

    def append_feedback(self, session_id: str, feedback: Dict[str, Any]) -> bool:
        """Append feedback once. Returns False for an idempotent duplicate."""

        item = dict(feedback)
        item["event_type"] = "execution_feedback"
        item.setdefault("created_at", time.time())
        feedback_id = str(item.get("feedback_id") or "")
        with self._lock:
            records = self._load_locked(session_id)
            if any(
                record.get("event_type") == "execution_feedback"
                and str(record.get("feedback_id") or "") == feedback_id
                for record in records
            ):
                return False
            self._append_locked(session_id, item)
            return True

    def session_count(self) -> int:
        with self._lock:
            return len(self._records)
