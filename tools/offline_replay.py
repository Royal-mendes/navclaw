#!/usr/bin/env python3
"""Replay existing AgentNav candidate logs through the live NavClaw brain."""

from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from navclaw.contracts import selectable_candidates  # noqa: E402


def find_annotated_image(candidate_path: Path) -> Optional[Path]:
    stem = candidate_path.name
    suffix = "_candidates.json"
    if stem.endswith(suffix):
        stem = stem[: -len(suffix)]
    pass_root = candidate_path.parent.parent
    candidates = [
        pass_root / "vlm_waypoints" / (stem + "_annotated.png"),
        pass_root / "vlm_waypoints" / (stem + "_navclaw_annotated.png"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = sorted(pass_root.glob("vlm_waypoints/{0}*annotated*.png".format(stem)))
    return matches[0] if matches else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-json", action="append", default=[])
    parser.add_argument("--glob", action="append", default=[])
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--server-url", default="http://127.0.0.1:8765/decide")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--bridge", type=Path, default=ROOT / "bridge" / "selector_client.py")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths: List[Path] = [Path(value) for value in args.candidate_json]
    for pattern in args.glob:
        paths.extend(Path(value) for value in sorted(glob.glob(pattern)))
    unique = []
    seen = set()
    for path in paths:
        resolved = str(path.resolve())
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    paths = unique[: max(0, args.limit)]
    if not paths:
        raise SystemExit("no candidate JSON files selected")

    args.run_dir.mkdir(parents=True, exist_ok=True)
    records: List[Dict] = []
    for index, candidate_path in enumerate(paths, start=1):
        data = json.loads(candidate_path.read_text(encoding="utf-8"))
        valid = selectable_candidates(data.get("candidates") or [])
        valid_ids = [candidate["id"] for candidate in valid]
        image_path = find_annotated_image(candidate_path)
        if image_path is None:
            records.append(
                {
                    "candidate_json": str(candidate_path),
                    "status": "error",
                    "error": "annotated_image_not_found",
                    "valid_candidate_ids": valid_ids,
                }
            )
            continue
        output_path = args.run_dir / ("replay_{0:02d}_vlm_result.json".format(index))
        command = [
            sys.executable,
            str(args.bridge),
            "--candidate-json",
            str(candidate_path),
            "--image",
            str(image_path),
            "--image-already-annotated",
            "--output-json",
            str(output_path),
            "--target",
            str(data.get("target") or "unknown"),
            "--episode",
            str(data.get("episode", "unknown")),
            "--step",
            str(data.get("step", index)),
            "--mode",
            str(data.get("mode") or "offline_replay"),
            "--session-id",
            "offline_replay_{0}".format(index),
            "--server-url",
            args.server_url,
        ]
        started = time.time()
        completed = subprocess.run(command, text=True, capture_output=True)
        record: Dict = {
            "candidate_json": str(candidate_path),
            "image": str(image_path),
            "output_json": str(output_path),
            "returncode": completed.returncode,
            "elapsed_s": round(time.time() - started, 3),
            "valid_candidate_ids": valid_ids,
            "stderr": completed.stderr[-2000:],
        }
        if completed.returncode == 0 and output_path.exists():
            result = json.loads(output_path.read_text(encoding="utf-8"))
            decision = result.get("decision")
            selected = result.get("selected")
            constraint_pass = (
                decision in {"LOOK_LEFT_60", "LOOK_RIGHT_60"}
                or (decision == "SELECT_WAYPOINT" and selected in valid_ids)
            )
            record.update(
                {
                    "status": "ok" if constraint_pass and result.get("fallback") is False else "error",
                    "decision": decision,
                    "selected": selected,
                    "fallback": result.get("fallback"),
                    "constraint_pass": constraint_pass,
                    "result": result,
                }
            )
        else:
            record.update({"status": "error", "constraint_pass": False})
        records.append(record)

    summary = {
        "created_at": time.time(),
        "replay_count": len(records),
        "success_count": sum(1 for record in records if record.get("status") == "ok"),
        "constraint_pass": bool(records) and all(record.get("constraint_pass") for record in records),
        "fallback_count": sum(1 for record in records if record.get("fallback") is True),
        "records": records,
    }
    summary_path = args.run_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0 if summary["constraint_pass"] and summary["fallback_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
