#!/usr/bin/env python3
"""Strict original-frontier-only entrypoint for the NavClaw selector bridge.

The lower planner may still emit diagnostic or recovery candidates, but this
entrypoint removes every candidate whose source is not exactly ``frontier``
before annotation, prompt construction, validation, and execution-feedback
tracking. The filtered JSON is retained beside the selector output for audit.
"""

from __future__ import print_function

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


BRIDGE_DIR = Path(__file__).resolve().parent
SELECTOR_CLIENT = BRIDGE_DIR / "selector_client.py"
ORIGINAL_FRONTIER_SOURCE = "frontier"
POLICY_NAME = "original_frontier_only"


def write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.{0}".format(os.getpid()))
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(str(temporary), str(path))


def original_frontier_candidates(candidates: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep only unmodified planner frontiers.

    Exact source matching is intentional. Derived candidates such as
    ``frontier_cluster_average``, ``frontier_cluster_sample``,
    ``frontier_cluster_endpoint``, and ``frontier_slid`` are excluded together
    with local-view, connector, target, recovery, and emergency proxies.
    """

    return [
        candidate
        for candidate in candidates or []
        if isinstance(candidate, dict)
        and str(candidate.get("source") or "").strip() == ORIGINAL_FRONTIER_SOURCE
    ]


def discarded_source_counts(candidates: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for candidate in candidates or []:
        if not isinstance(candidate, dict):
            source = "invalid_record"
        else:
            source = str(candidate.get("source") or "missing_source").strip() or "missing_source"
        if source == ORIGINAL_FRONTIER_SOURCE:
            continue
        counts[source] = counts.get(source, 0) + 1
    return dict(sorted(counts.items()))


def filtered_json_path(candidate_path: Path, output_path: Path) -> Path:
    candidate_path = Path(candidate_path)
    output_path = Path(output_path)
    directory = output_path.parent / "frontier_only_inputs"
    return directory / (candidate_path.stem + "_original_frontier_only.json")


def replace_cli_option(argv: Sequence[str], option: str, value: str) -> List[str]:
    rewritten = list(argv)
    try:
        index = rewritten.index(option)
    except ValueError as exc:
        raise RuntimeError("missing_required_cli_option:{0}".format(option)) from exc
    if index + 1 >= len(rewritten):
        raise RuntimeError("missing_cli_option_value:{0}".format(option))
    rewritten[index + 1] = value
    return rewritten


def patch_audit_artifact(
    path: Path,
    source_candidate_json: Path,
    filtered_candidate_json: Path,
    original_count: int,
    kept_count: int,
    discarded_sources: Dict[str, int],
) -> None:
    if not path.exists():
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if not isinstance(payload, dict):
        return
    payload["candidate_policy"] = POLICY_NAME
    payload["source_candidate_json"] = str(source_candidate_json)
    payload["filtered_candidate_json"] = str(filtered_candidate_json)
    payload["original_candidate_count"] = original_count
    payload["kept_original_frontier_count"] = kept_count
    payload["discarded_candidate_count"] = max(0, original_count - kept_count)
    payload["discarded_candidate_sources"] = discarded_sources
    # Keep candidate_json user-facing as the original lower-layer artifact.
    payload["candidate_json"] = str(source_candidate_json)
    write_json_atomic(path, payload)


def parse_wrapper_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--candidate-json", required=True)
    parser.add_argument("--output-json", required=True)
    args, _ = parser.parse_known_args(list(argv))
    return args


def main(argv: Sequence[str] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--image-already-annotated" in argv:
        raise RuntimeError(
            "original_frontier_only_requires_raw_image_to_prevent_stale_non_frontier_labels"
        )
    args = parse_wrapper_args(argv)
    source_candidate_path = Path(args.candidate_json)
    output_path = Path(args.output_json)
    source_data = json.loads(source_candidate_path.read_text(encoding="utf-8"))
    if not isinstance(source_data, dict):
        raise RuntimeError("candidate_json_must_be_object")

    original_candidates = source_data.get("candidates") or []
    kept_candidates = original_frontier_candidates(original_candidates)
    discarded_sources = discarded_source_counts(original_candidates)

    filtered_data = dict(source_data)
    filtered_data["candidates"] = kept_candidates
    filtered_data["candidate_policy"] = {
        "name": POLICY_NAME,
        "allowed_source": ORIGINAL_FRONTIER_SOURCE,
        "original_candidate_count": len(original_candidates),
        "kept_candidate_count": len(kept_candidates),
        "discarded_candidate_sources": discarded_sources,
    }

    filtered_path = filtered_json_path(source_candidate_path, output_path)
    write_json_atomic(filtered_path, filtered_data)

    rewritten_argv = replace_cli_option(argv, "--candidate-json", str(filtered_path))
    completed = subprocess.run([sys.executable, str(SELECTOR_CLIENT)] + rewritten_argv)

    bridge_log = output_path.with_name(
        output_path.name.replace("_vlm_result.json", "_navclaw_bridge.json")
    )
    error_log = output_path.with_name(output_path.stem + "_navclaw_error.json")
    for artifact in (output_path, bridge_log, error_log):
        patch_audit_artifact(
            artifact,
            source_candidate_path,
            filtered_path,
            len(original_candidates),
            len(kept_candidates),
            discarded_sources,
        )
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
