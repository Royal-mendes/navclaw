#!/usr/bin/env python3
"""CLI-compatible thin client between AgentNav C++ and the NavClaw brain."""

from __future__ import print_function

import argparse
import base64
import hashlib
import json
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
    normalize_agent_action,
    selectable_candidates,
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


def annotate_current_image(image_path, output_path, candidates):
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise RuntimeError("pillow_required_for_live_candidate_annotation") from exc

    image = Image.open(str(image_path)).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for candidate in candidates:
        projection = candidate.get("projection")
        if not isinstance(projection, (list, tuple)) or len(projection) < 2:
            continue
        try:
            x = int(round(float(projection[0])))
            y = int(round(float(projection[1])))
        except (TypeError, ValueError):
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
            projection = candidate.get("projection")
            if not isinstance(projection, (list, tuple)) or len(projection) < 2:
                continue
            try:
                x = int(round(float(projection[0])))
                y = int(round(float(projection[1])))
            except (TypeError, ValueError):
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


def build_payload(args, candidate_data, image_path):
    candidate_data = dict(candidate_data)
    candidate_data["target"] = args.target or candidate_data.get("target") or "unknown"
    candidate_data["episode"] = args.episode
    candidate_data["step"] = args.step
    candidate_data["mode"] = args.mode or candidate_data.get("mode")
    candidate_data["task_type"] = args.task_type or candidate_data.get("task_type") or "objectnav"
    if args.instruction:
        candidate_data["instruction"] = args.instruction
    observation = compact_observation(candidate_data)
    request_id = request_id_from_output(args.output_json)
    session_id = (
        args.session_id
        or os.getenv("NAVCLAW_SESSION_ID", "")
        or "episode_{0}".format(args.episode)
    )
    return {
        "request_id": request_id,
        "session_id": session_id,
        "observation": observation,
        "image_data_uri": encode_data_uri(image_path),
        "allow_stop": bool(args.allow_stop),
        "force_select_waypoint": bool(candidate_data.get("force_select_waypoint"))
        or str(candidate_data.get("mode") or args.mode) == "vlm_scan_full_circle_choice",
        "strict_no_fallback": True,
    }


def post_decision(url, payload, timeout):
    body = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "User-Agent": "NavClawBridge/0.1"},
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
        pass
    else:
        raise ContractError("bridge_received_unknown_decision")
    if force_select_waypoint and decision != "SELECT_WAYPOINT":
        raise ContractError("bridge_full_scan_requires_waypoint")


def run(args):
    candidate_path = Path(args.candidate_json)
    image_path = Path(args.image)
    output_path = Path(args.output_json)
    if output_path.exists():
        output_path.unlink()
    candidate_data = json.loads(candidate_path.read_text(encoding="utf-8"))
    valid = selectable_candidates(candidate_data.get("candidates") or [])
    valid_ids = [item["id"] for item in valid]

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
                candidate_data.get("candidates") or [],
            )

    payload = build_payload(args, candidate_data, decision_image)
    result, bridge_request_hash = post_decision(args.server_url, payload, args.timeout)
    validate_cpp_result(
        result,
        valid_ids,
        allow_stop=args.allow_stop,
        force_select_waypoint=bool(payload.get("force_select_waypoint")),
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
            "request_id": request_id_from_output(output_path),
            "candidate_json": str(candidate_path),
            "decision_image": str(decision_image),
            "valid_candidate_ids": valid_ids,
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
