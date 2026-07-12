#!/usr/bin/env python3
"""Audit a live NavClaw episode for strict action and logging invariants."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from navclaw.contracts import selectable_candidates  # noqa: E402


def json_lines(path: Path) -> List[Dict[str, Any]]:
    records = []
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nav-run-dir", type=Path, required=True)
    parser.add_argument("--mapgpt-run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result_files = sorted(args.mapgpt_run_dir.glob("ep*/pass1/vlm_waypoint_logs/*_vlm_result.json"))
    result_records = []
    invalid = []
    fallback_count = 0
    for result_path in result_files:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        candidate_path = result_path.with_name(
            result_path.name.replace("_vlm_result.json", "_candidates.json")
        )
        valid_ids: List[str] = []
        if candidate_path.exists():
            candidates = json.loads(candidate_path.read_text(encoding="utf-8"))
            valid_ids = [item["id"] for item in selectable_candidates(candidates.get("candidates") or [])]
        decision = result.get("decision")
        selected = result.get("selected")
        valid = (
            decision in {"LOOK_LEFT_60", "LOOK_RIGHT_60"} and selected in (None, "")
        ) or (decision == "SELECT_WAYPOINT" and selected in valid_ids)
        strict_metadata = (
            result.get("fallback") is False
            and (result.get("navclaw") or {}).get("rule_fallback_enabled") is False
            and (result.get("navclaw") or {}).get("strict_candidate_constraint") is True
        )
        if result.get("fallback") is True:
            fallback_count += 1
        if not valid or not strict_metadata:
            invalid.append(str(result_path))
        result_records.append(
            {
                "path": str(result_path),
                "candidate_path": str(candidate_path),
                "valid_candidate_ids": valid_ids,
                "decision": decision,
                "selected": selected,
                "valid": valid,
                "strict_metadata": strict_metadata,
            }
        )

    brain_decisions = json_lines(args.nav_run_dir / "brain" / "decisions.jsonl")
    api_attempts = json_lines(args.nav_run_dir / "brain" / "api_attempts.jsonl")
    hashes_by_request = defaultdict(set)
    for attempt in api_attempts:
        hashes_by_request[str(attempt.get("request_id"))].add(attempt.get("request_body_sha256"))
    retry_hash_consistent = all(len(values - {None}) <= 1 for values in hashes_by_request.values())
    brain_pids = []
    pid_file = args.nav_run_dir / "brain" / "brain.pid"
    if pid_file.exists():
        brain_pids.append(pid_file.read_text(encoding="utf-8").strip())

    metrics_files = sorted(args.mapgpt_run_dir.glob("ep*/pass1/metrics.json"))
    metrics = []
    for path in metrics_files:
        try:
            metrics.append({"path": str(path), "value": json.loads(path.read_text(encoding="utf-8"))})
        except json.JSONDecodeError:
            metrics.append({"path": str(path), "error": "invalid_json"})

    required_patterns = {
        "habitat_log": bool(list(args.mapgpt_run_dir.glob("ep*/pass1/habitat.log"))),
        "roslaunch_log": bool(list(args.mapgpt_run_dir.glob("ep*/pass1/roslaunch.log"))),
        "metrics_json": bool(metrics_files),
        "batch_status": (args.mapgpt_run_dir / "batch_status.jsonl").exists(),
        "episode_done": bool(list(args.mapgpt_run_dir.glob("ep*/episode_done.marker"))),
        "brain_decisions": (args.nav_run_dir / "brain" / "decisions.jsonl").exists(),
        "api_attempts": (args.nav_run_dir / "brain" / "api_attempts.jsonl").exists(),
    }
    summary = {
        "result_count": len(result_records),
        "brain_decision_count": len([r for r in brain_decisions if r.get("status") == "ok"]),
        "api_attempt_count": len(api_attempts),
        "fallback_count": fallback_count,
        "invalid_result_count": len(invalid),
        "invalid_result_paths": invalid,
        "retry_request_hash_consistent": retry_hash_consistent,
        "brain_pid_values": brain_pids,
        "required_logs": required_patterns,
        "metrics": metrics,
        "results": result_records,
    }
    summary["pass"] = (
        summary["result_count"] > 0
        and summary["result_count"] == summary["brain_decision_count"]
        and fallback_count == 0
        and not invalid
        and retry_hash_consistent
        and all(required_patterns.values())
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
