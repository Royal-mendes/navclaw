#!/usr/bin/env python3
"""CLI-compatible thin client between AgentNav C++ and the NavClaw brain."""

from __future__ import print_function

import argparse
import base64
import hashlib
import json
import math
import mimetypes
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from navclaw.contracts import (  # noqa: E402
    ContractError,
    compact_observation,
    filter_selectable_candidates,
    projection_xy,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-json", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--target", default="unknown")
    parser.add_argument("--episode", default="unknown")
    parser.add_argument("--step", default="unknown")
    parser.add_argument("--mode", default="vlm_guided_geometric")
    parser.add_argument("--depth-image", default="")
    parser.add_argument("--task-type", default="objectnav")
    parser.add_argument("--instruction", default="")
    parser.add_argument("--session-id", default="")
    parser.add_argument(
        "--server-url",
        default=os.getenv("NAVCLAW_SERVER_URL", "http://127.0.0.1:8765/decide"),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.getenv("NAVCLAW_BRIDGE_TIMEOUT", "900")),
    )
    parser.add_argument("--image-already-annotated", action="store_true")
    parser.add_argument("--allow-stop", action="store_true")
    return parser.parse_args()


def write_json_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.{0}".format(os.getpid()))
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(str(temporary), str(path))


def request_id_from_output(output_path):
    name = Path(output_path).name
    suffix = "_vlm_result.json"
    return name[: -len(suffix)] if name.endswith(suffix) else Path(name).stem


def safe_session_id(session_id):
    cleaned = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in session_id)
    return cleaned[:120] or "unknown"


def resolve_session_id(args):
    return (
        args.session_id
        or os.getenv("NAVCLAW_SESSION_ID", "")
        or "episode_{0}".format(args.episode)
    )


def _finite_xy(value):
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        x = float(value[0])
        y = float(value[1])
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    return [x, y]


def robot_xy(candidate_data):
    robot = candidate_data.get("robot") or {}
    return _finite_xy([robot.get("x"), robot.get("y")])


def _projection_in_bounds(candidate, width, height):
    projection = projection_xy(candidate)
    if projection is None:
        return False
    x, y = projection
    return 0.0 <= x < float(width) and 0.0 <= y < float(height)


def renderable_candidates(candidate_data, image_path):
    """Return the exact original records that can be drawn and selected."""

    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("pillow_required_for_candidate_validation") from exc

    base_candidates = filter_selectable_candidates(candidate_data.get("candidates") or [])
    views = candidate_data.get("panorama_views") or []
    if len(views) >= 2:
        dimensions = {}
        for index, view in enumerate(views):
            view_id = str(view.get("view_id") or "P{0}".format(index + 1))
            raw_path = Path(str(view.get("raw_image") or view.get("image_path") or ""))
            if not raw_path.exists():
                continue
            with Image.open(str(raw_path)) as image:
                dimensions[view_id] = image.size
        selected = []
        for candidate in base_candidates:
            view_id = str(candidate.get("view_id") or "")
            size = dimensions.get(view_id)
            if size and _projection_in_bounds(candidate, size[0], size[1]):
                selected.append(candidate)
        return selected

    with Image.open(str(image_path)) as image:
        width, height = image.size
    return [
        candidate
        for candidate in base_candidates
        if _projection_in_bounds(candidate, width, height)
    ]


def annotate_current_image(image_path, output_path, candidates):
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise RuntimeError("pillow_required_for_live_candidate_annotation") from exc

    image = Image.open(str(image_path)).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for candidate in candidates:
        projection = projection_xy(candidate)
        if projection is None:
            continue
        x = int(round(projection[0]))
        y = int(round(projection[1]))
        if x < 0 or y < 0 or x >= image.width or y >= image.height:
            continue
        label = str(candidate.get("id") or "")
        radius = 14
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=(0, 255, 0), width=4)
        text_box = draw.textbbox((x + radius + 3, y - radius), label, font=font)
        draw.rectangle(text_box, fill=(0, 0, 0))
        draw.text((x + radius + 3, y - radius), label, fill=(0, 255, 0), font=font)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(str(output_path), format="PNG")
    return output_path


def compose_panorama(candidate_data, output_path):
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise RuntimeError("pillow_required_for_live_panorama_annotation") from exc

    views = candidate_data.get("panorama_views") or []
    all_candidates = candidate_data.get("candidates") or []
    panels = []
    font = ImageFont.load_default()
    for index, view in enumerate(views):
        raw_path = Path(str(view.get("raw_image") or view.get("image_path") or ""))
        if not raw_path.exists():
            continue
        view_id = str(view.get("view_id") or "P{0}".format(index + 1))
        panel_candidates = [
            candidate
            for candidate in all_candidates
            if str(candidate.get("view_id") or "") == view_id
        ]
        image = Image.open(str(raw_path)).convert("RGB")
        draw = ImageDraw.Draw(image)
        for candidate in panel_candidates:
            projection = projection_xy(candidate)
            if projection is None:
                continue
            x = int(round(projection[0]))
            y = int(round(projection[1]))
            if x < 0 or y < 0 or x >= image.width or y >= image.height:
                continue
            label = str(candidate.get("id") or "")
            draw.ellipse((x - 14, y - 14, x + 14, y + 14), outline=(0, 255, 0), width=4)
            draw.rectangle((x + 17, y - 14, x + 17 + max(18, len(label) * 7), y), fill=(0, 0, 0))
            draw.text((x + 18, y - 13), label, fill=(0, 255, 0), font=font)
        header_h = 24
        framed = Image.new("RGB", (image.width, image.height + header_h), (20, 20, 20))
        framed.paste(image, (0, header_h))
        header_draw = ImageDraw.Draw(framed)
        header_draw.text(
            (6, 6),
            "{0} yaw={1} candidates={2}".format(
                view_id,
                view.get("relative_yaw_deg", view.get("view_angle_deg", "?")),
                len(panel_candidates),
            ),
            fill=(255, 255, 255),
            font=font,
        )
        panels.append(framed)
    if not panels:
        raise RuntimeError("panorama_has_no_readable_views")
    columns = min(3, len(panels))
    rows = (len(panels) + columns - 1) // columns
    cell_w = max(panel.width for panel in panels)
    cell_h = max(panel.height for panel in panels)
    canvas = Image.new("RGB", (columns * cell_w, rows * cell_h), (0, 0, 0))
    for index, panel in enumerate(panels):
        x = (index % columns) * cell_w
        y = (index // columns) * cell_h
        canvas.paste(panel, (x, y))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(str(output_path), format="PNG")
    return output_path


def encode_data_uri(path):
    path = Path(path)
    mime = mimetypes.guess_type(str(path))[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return "data:{0};base64,{1}".format(mime, encoded)


def bridge_state_path(output_path, session_id):
    configured = os.getenv("NAVCLAW_BRIDGE_STATE_DIR", "").strip()
    root = Path(configured) if configured else Path(output_path).parent / ".navclaw_state"
    return root / (safe_session_id(session_id) + ".json")


def load_bridge_state(path):
    path = Path(path)
    if not path.exists():
        return {"version": 1, "pending_waypoint": None}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"version": 1, "pending_waypoint": None}
    return value if isinstance(value, dict) else {"version": 1, "pending_waypoint": None}


def reached_threshold():
    try:
        value = float(os.getenv("NAVCLAW_WAYPOINT_REACHED_THRESHOLD", "0.25"))
    except ValueError:
        value = 0.25
    return max(0.05, min(1.0, value))


def build_execution_feedback(state, candidate_data, current_request_id):
    pending = state.get("pending_waypoint") if isinstance(state, dict) else None
    if not isinstance(pending, dict):
        return None
    previous_request_id = str(pending.get("request_id") or "")
    selected_id = str(pending.get("selected_id") or "")
    if not previous_request_id or not selected_id or previous_request_id == current_request_id:
        return None

    current_xy = robot_xy(candidate_data)
    start_xy = _finite_xy(pending.get("start_robot_xy"))
    safe_goal = _finite_xy(pending.get("safe_goal"))
    threshold = float(pending.get("reached_threshold_m") or reached_threshold())
    final_distance = None
    start_distance = None
    travel_distance = None
    outcome = "unknown"
    if current_xy is not None and safe_goal is not None:
        final_distance = math.hypot(current_xy[0] - safe_goal[0], current_xy[1] - safe_goal[1])
        outcome = "reached" if final_distance <= threshold else "stalled_or_aborted"
    if start_xy is not None and safe_goal is not None:
        start_distance = math.hypot(start_xy[0] - safe_goal[0], start_xy[1] - safe_goal[1])
    if start_xy is not None and current_xy is not None:
        travel_distance = math.hypot(current_xy[0] - start_xy[0], current_xy[1] - start_xy[1])

    observed_at = time.time()
    issued_at = pending.get("issued_at")
    elapsed_s = None
    if isinstance(issued_at, (int, float)) and math.isfinite(float(issued_at)):
        elapsed_s = max(0.0, observed_at - float(issued_at))
    feedback_id = hashlib.sha256(
        (previous_request_id + "\0" + current_request_id + "\0" + selected_id).encode("utf-8")
    ).hexdigest()
    feedback = {
        "feedback_id": feedback_id,
        "request_id": previous_request_id,
        "next_request_id": current_request_id,
        "selected_id": selected_id,
        "execution_outcome": outcome,
        "source": "bridge_next_decision_observation",
        "reached_threshold_m": threshold,
        "issued_step": pending.get("step"),
        "observed_step": candidate_data.get("step"),
    }
    if final_distance is not None:
        feedback["final_distance_m"] = final_distance
    if start_distance is not None:
        feedback["start_distance_m"] = start_distance
    if travel_distance is not None:
        feedback["travel_distance_m"] = travel_distance
    if elapsed_s is not None:
        feedback["elapsed_s"] = elapsed_s
    return feedback


def build_payload(args, candidate_data, image_path, execution_feedback=None):
    candidate_data = dict(candidate_data)
    candidate_data["target"] = args.target or candidate_data.get("target") or "unknown"
    candidate_data["episode"] = args.episode
    candidate_data["step"] = args.step
    candidate_data["mode"] = args.mode or candidate_data.get("mode")
    candidate_data["task_type"] = args.task_type or candidate_data.get("task_type") or "objectnav"
    if args.instruction:
        candidate_data["instruction"] = args.instruction
    observation = compact_observation(candidate_data)
    payload = {
        "request_id": request_id_from_output(args.output_json),
        "session_id": resolve_session_id(args),
        "observation": observation,
        "image_data_uri": encode_data_uri(image_path),
        "allow_stop": bool(args.allow_stop),
        "force_select_waypoint": bool(candidate_data.get("force_select_waypoint"))
        or str(candidate_data.get("mode") or args.mode) == "vlm_scan_full_circle_choice",
        "strict_no_fallback": True,
    }
    if execution_feedback is not None:
        payload["execution_feedback"] = execution_feedback
    return payload


def post_decision(url, payload, timeout):
    body = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "User-Agent": "NavClawBridge/0.2"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_body = response.read().decode("utf-8")
            status = int(getattr(response, "status", 200))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:2000]
        raise RuntimeError("navclaw_server_http_{0}:{1}".format(exc.code, detail))
    except urllib.error.URLError as exc:
        raise RuntimeError("navclaw_server_unreachable:{0}".format(exc))
    parsed = json.loads(response_body)
    if status != 200 or not parsed.get("ok"):
        raise RuntimeError("navclaw_server_rejected:{0}".format(response_body[:2000]))
    result = parsed.get("result")
    if not isinstance(result, dict):
        raise RuntimeError("navclaw_server_missing_result")
    return result, hashlib.sha256(body).hexdigest()


def validate_cpp_result(result, candidate_ids, allow_stop=False, force_select_waypoint=False):
    if result.get("fallback") is not False:
        raise ContractError("navclaw_result_must_set_fallback_false")
    decision = str(result.get("decision") or "")
    selected = result.get("selected")
    if decision == "SELECT_WAYPOINT":
        if str(selected or "") not in set(candidate_ids):
            raise ContractError("bridge_selected_candidate_not_current")
    elif decision in {"LOOK_LEFT_60", "LOOK_RIGHT_60"}:
        if selected not in (None, ""):
            raise ContractError("look_action_must_not_select_candidate")
    elif decision == "STOP" and allow_stop:
        if selected not in (None, ""):
            raise ContractError("stop_action_must_not_select_candidate")
    else:
        raise ContractError("bridge_received_unknown_decision")
    if force_select_waypoint and decision != "SELECT_WAYPOINT":
        raise ContractError("bridge_full_scan_requires_waypoint")


def pending_waypoint_from_result(result, candidate_data, request_id, threshold):
    if result.get("decision") != "SELECT_WAYPOINT":
        return None
    selected_id = str(result.get("selected") or "")
    selected = next(
        (
            candidate
            for candidate in candidate_data.get("candidates") or []
            if str(candidate.get("id") or "") == selected_id
        ),
        None,
    )
    if not isinstance(selected, dict):
        return None
    return {
        "request_id": request_id,
        "selected_id": selected_id,
        "safe_goal": _finite_xy(selected.get("safe_goal")),
        "start_robot_xy": robot_xy(candidate_data),
        "target": candidate_data.get("target"),
        "episode": candidate_data.get("episode"),
        "step": candidate_data.get("step"),
        "issued_at": time.time(),
        "reached_threshold_m": threshold,
    }


def run(args):
    candidate_path = Path(args.candidate_json)
    image_path = Path(args.image)
    output_path = Path(args.output_json)
    if output_path.exists():
        output_path.unlink()
    candidate_data = json.loads(candidate_path.read_text(encoding="utf-8"))

    exact_candidates = renderable_candidates(candidate_data, image_path)
    candidate_data = dict(candidate_data)
    candidate_data["candidates"] = exact_candidates
    valid_ids = [str(item.get("id")) for item in exact_candidates]

    if args.image_already_annotated:
        decision_image = image_path
    else:
        waypoint_dir = output_path.parent.parent / "vlm_waypoints"
        annotation_path = waypoint_dir / (
            request_id_from_output(output_path) + "_navclaw_annotated.png"
        )
        if len(candidate_data.get("panorama_views") or []) >= 2:
            decision_image = compose_panorama(candidate_data, annotation_path)
        else:
            decision_image = annotate_current_image(
                image_path,
                annotation_path,
                exact_candidates,
            )

    request_id = request_id_from_output(output_path)
    session_id = resolve_session_id(args)
    state_path = bridge_state_path(output_path, session_id)
    state = load_bridge_state(state_path)
    execution_feedback = build_execution_feedback(state, candidate_data, request_id)
    payload = build_payload(args, candidate_data, decision_image, execution_feedback)
    result, bridge_request_hash = post_decision(args.server_url, payload, args.timeout)
    validate_cpp_result(
        result,
        valid_ids,
        allow_stop=args.allow_stop,
        force_select_waypoint=bool(payload.get("force_select_waypoint")),
    )

    threshold = reached_threshold()
    new_pending = pending_waypoint_from_result(
        result, candidate_data, request_id, threshold
    )
    write_json_atomic(
        state_path,
        {
            "version": 1,
            "session_id": session_id,
            "pending_waypoint": new_pending,
            "last_execution_feedback": execution_feedback,
            "updated_at": time.time(),
        },
    )

    result.update(
        {
            "target": args.target,
            "episode": args.episode,
            "step": args.step,
            "mode": args.mode,
            "candidate_json": str(candidate_path),
            "annotated_image": str(decision_image),
            "visible_candidate_count": len(valid_ids),
            "current_candidate_ids": valid_ids,
            "execution_feedback": execution_feedback,
            "bridge_request_sha256": bridge_request_hash,
            "created_at": time.time(),
        }
    )
    write_json_atomic(output_path, result)
    bridge_log = output_path.with_name(
        output_path.name.replace("_vlm_result.json", "_navclaw_bridge.json")
    )
    write_json_atomic(
        bridge_log,
        {
            "status": "ok",
            "request_id": request_id,
            "session_id": session_id,
            "candidate_json": str(candidate_path),
            "decision_image": str(decision_image),
            "valid_candidate_ids": valid_ids,
            "execution_feedback": execution_feedback,
            "pending_waypoint": new_pending,
            "result": result,
            "fallback": False,
        },
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


def main():
    args = parse_args()
    output_path = Path(args.output_json)
    try:
        return run(args)
    except Exception as exc:
        if output_path.exists():
            output_path.unlink()
        error_log = output_path.with_name(output_path.stem + "_navclaw_error.json")
        write_json_atomic(
            error_log,
            {
                "status": "error",
                "error": repr(exc),
                "candidate_json": args.candidate_json,
                "image": args.image,
                "fallback": False,
                "created_at": time.time(),
            },
        )
        print("NavClaw selector failed: {0}".format(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
