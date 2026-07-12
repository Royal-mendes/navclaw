"""Auditable session memory with no policy mutation or skill generation."""

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

    def recent(self, session_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            records = list(self._records.get(session_id, []))
            if not records:
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
            prompt_records = []
            for record in records[-self.max_prompt_items :]:
                prompt_records.append(
                    {
                        "step": record.get("step"),
                        "target": record.get("target"),
                        "valid_candidate_ids": record.get("valid_candidate_ids") or [],
                        "decision": (record.get("result") or {}).get("decision"),
                        "selected": (record.get("result") or {}).get("selected"),
                        "reason": (record.get("result") or {}).get("reason"),
                        "decision_status": "issued" if record.get("status") == "ok" else "failed",
                        "execution_outcome": "unknown_not_reported_by_current_bridge",
                    }
                )
            return prompt_records

    def append(self, session_id: str, record: Dict[str, Any]) -> None:
        item = dict(record)
        item.setdefault("created_at", time.time())
        line = json.dumps(item, ensure_ascii=False, sort_keys=True)
        with self._lock:
            self._records.setdefault(session_id, []).append(item)
            path = self._path(session_id)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def session_count(self) -> int:
        with self._lock:
            return len(self._records)
