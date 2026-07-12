#!/usr/bin/env python3
"""Idempotently patch the external ApexNav lower planner for verified VLM STOP."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Dict, Tuple


HEADER_REL = Path(
    "src/planner/exploration_manager/include/exploration_manager/exploration_manager.h"
)
MANAGER_REL = Path("src/planner/exploration_manager/src/exploration_manager.cpp")
FSM_REL = Path("src/planner/exploration_manager/src/exploration_fsm.cpp")


def _replace_once(text: str, old: str, new: str, label: str) -> Tuple[str, bool]:
    if new in text:
        return text, False
    count = text.count(old)
    if count != 1:
        raise RuntimeError("patch_anchor_count:{0}:{1}".format(label, count))
    return text.replace(old, new, 1), True


def _patch_header(text: str) -> Tuple[str, bool]:
    changed = False
    replacements = [
        (
            "string decision;                ///< SELECT_WAYPOINT, LOOK_LEFT_60, or LOOK_RIGHT_60",
            "string decision;                ///< SELECT_WAYPOINT, LOOK_LEFT_60, LOOK_RIGHT_60, or STOP",
            "header_decision_comment",
        ),
        (
            "  bool consumePendingVLMForcedAction(int& action_code);\n",
            "  bool consumePendingVLMForcedAction(int& action_code);\n"
            "  bool consumePendingVLMStopRequest();\n",
            "header_stop_method",
        ),
        (
            "  int pending_vlm_forced_action_steps_ = 0;\n",
            "  int pending_vlm_forced_action_steps_ = 0;\n"
            "  bool pending_vlm_stop_request_ = false;\n",
            "header_stop_member",
        ),
    ]
    for old, new, label in replacements:
        text, did_change = _replace_once(text, old, new, label)
        changed = changed or did_change
    return text, changed


def _patch_policy_stop_branch(text: str) -> Tuple[str, bool]:
    marker = "void ExplorationManager::findVLMGuidedFrontierPolicy"
    end_marker = "void ExplorationManager::findTSPTourPolicy"
    if "pending_vlm_stop_request_ = true;" in text:
        return text, False
    start = text.find(marker)
    end = text.find(end_marker, start + 1)
    if start < 0 or end < 0:
        raise RuntimeError("patch_section_missing:findVLMGuidedFrontierPolicy")
    section = text[start:end]
    pattern = re.compile(
        r'(?m)^(?P<indent>[ \t]*)else if \(decision\.decision == "SELECT_WAYPOINT"\) \{'
    )
    matches = list(pattern.finditer(section))
    if len(matches) != 1:
        raise RuntimeError(
            "patch_anchor_count:manager_stop_branch:{0}".format(len(matches))
        )
    match = matches[0]
    indent = match.group("indent")
    replacement = (
        '{0}else if (decision.decision == "STOP") {{\n'
        '{0}  clearActiveVLMWaypoint("vlm_verified_stop");\n'
        '{0}  resetVLMScanContext("vlm_verified_stop");\n'
        '{0}  pending_vlm_stop_request_ = true;\n'
        '{0}  next_best_path.clear();\n'
        '{0}  ROS_ERROR("[VLM STOP] Verified stop request accepted: %s",\n'
        '{0}      decision.reason.c_str());\n'
        '{0}}}\n'
        '{0}else if (decision.decision == "SELECT_WAYPOINT") {{'
    ).format(indent)
    section = section[: match.start()] + replacement + section[match.end() :]
    return text[:start] + section + text[end:], True


def _patch_manager(text: str) -> Tuple[str, bool]:
    changed = False
    old_parse = (
        '  return decision.fallback || decision.decision == "SELECT_WAYPOINT" ||\n'
        '         decision.decision == "LOOK_LEFT_60" || decision.decision == "LOOK_RIGHT_60";'
    )
    new_parse = (
        '  return decision.fallback || decision.decision == "SELECT_WAYPOINT" ||\n'
        '         decision.decision == "LOOK_LEFT_60" || decision.decision == "LOOK_RIGHT_60" ||\n'
        '         decision.decision == "STOP";'
    )
    text, did_change = _replace_once(
        text, old_parse, new_parse, "manager_parse_stop"
    )
    changed = changed or did_change

    old_accept = (
        "  if (isVLMForcedLookDecision(decision.decision))\n"
        "    return true;\n\n"
        '  if (decision.decision != "SELECT_WAYPOINT") {'
    )
    new_accept = (
        "  if (isVLMForcedLookDecision(decision.decision))\n"
        "    return true;\n"
        '  if (decision.decision == "STOP")\n'
        "    return true;\n\n"
        '  if (decision.decision != "SELECT_WAYPOINT") {'
    )
    text, did_change = _replace_once(
        text, old_accept, new_accept, "manager_accept_stop"
    )
    changed = changed or did_change

    method = (
        "bool ExplorationManager::consumePendingVLMStopRequest()\n"
        "{\n"
        "  if (!pending_vlm_stop_request_)\n"
        "    return false;\n"
        "  pending_vlm_stop_request_ = false;\n"
        "  return true;\n"
        "}\n\n"
    )
    method_anchor = "void ExplorationManager::findVLMGuidedFrontierPolicy"
    if method not in text:
        count = text.count(method_anchor)
        if count != 1:
            raise RuntimeError(
                "patch_anchor_count:manager_stop_consumer:{0}".format(count)
            )
        text = text.replace(method_anchor, method + method_anchor, 1)
        changed = True

    text, did_change = _patch_policy_stop_branch(text)
    changed = changed or did_change
    return text, changed


def _patch_fsm(text: str) -> Tuple[str, bool]:
    old = (
        "  expl_res = expl_manager_->planNextBestPoint(fd_->start_pt_, "
        "fd_->start_yaw_(0));\n"
    )
    new = (
        old
        + "\n"
        + "  if (expl_manager_->consumePendingVLMStopRequest()) {\n"
        + '    ROS_ERROR("[VLM STOP] Verified target stop requested; finishing episode.");\n'
        + "    final_res = FINAL_RESULT::REACH_OBJECT;\n"
        + "    return final_res;\n"
        + "  }\n"
    )
    return _replace_once(text, old, new, "fsm_consume_stop")


def _atomic_write(path: Path, content: str) -> None:
    temporary = path.with_name(path.name + ".tmp.{0}".format(os.getpid()))
    temporary.write_text(content, encoding="utf-8")
    os.replace(str(temporary), str(path))


def patch_workspace(root: Path) -> Dict[str, object]:
    root = Path(root).resolve()
    specs = [
        (HEADER_REL, _patch_header),
        (MANAGER_REL, _patch_manager),
        (FSM_REL, _patch_fsm),
    ]
    changed_files = []
    unchanged_files = []
    for relative, patcher in specs:
        path = root / relative
        if not path.exists():
            raise RuntimeError("required_lower_file_missing:{0}".format(path))
        original = path.read_text(encoding="utf-8")
        patched, changed = patcher(original)
        if changed:
            backup = path.with_name(path.name + ".navclaw_verified_stop.bak")
            if not backup.exists():
                shutil.copy2(str(path), str(backup))
            _atomic_write(path, patched)
            changed_files.append(str(relative))
        else:
            unchanged_files.append(str(relative))

    summary: Dict[str, object] = {
        "ok": True,
        "patch": "verified_vlm_stop_request_v1",
        "root": str(root),
        "changed_files": changed_files,
        "already_patched_files": unchanged_files,
        "created_at": time.time(),
    }
    marker = root / ".navclaw_verified_stop_patch.json"
    _atomic_write(
        marker,
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    summary = patch_workspace(args.root)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
