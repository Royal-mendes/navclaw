#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
IMAGE="${IMAGE:-agent-apexnav:noetic-gpu}"
HOST_DATA_DIR="${HOST_DATA_DIR:-/home/ubuntu/papers_repro/ApexNav/data}"
DOCKER_TTY_ARGS=()
if [[ -t 0 && -t 1 ]]; then DOCKER_TTY_ARGS=(-it); fi
mkdir -p "${REPO_ROOT}/logs" "${REPO_ROOT}/debug" "${REPO_ROOT}/videos" "${REPO_ROOT}/data"
docker run --rm "${DOCKER_TTY_ARGS[@]}" \
  --gpus all \
  --network host \
  --ipc host \
  --privileged \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e LD_PRELOAD="/opt/micromamba/envs/apexnav/lib/libstdc++.so.6${LD_PRELOAD:+:${LD_PRELOAD}}" \
  -e PYTHONPATH="/workspace/Agent-apexnav/third_party/python:${PYTHONPATH:-}" \
  -e HF_HOME=/workspace/Agent-apexnav/third_party/hf_cache \
  -e TRANSFORMERS_CACHE=/workspace/Agent-apexnav/third_party/hf_cache \
  -e APEXNAV_DATA_DIR=/workspace/Agent-apexnav/data \
  -e APEXNAV_VLM_DEBUG_DIR=/workspace/Agent-apexnav/debug \
  -e APEXNAV_VLM_SELECTOR_SCRIPT=/workspace/Agent-apexnav/vlm_waypoint_selector.py \
  -e APEXNAV_SKIP_LOCAL_VLM_SERVERS="${APEXNAV_SKIP_LOCAL_VLM_SERVERS:-0}" \
  -e APEXNAV_SKIP_BLIP2="${APEXNAV_SKIP_BLIP2:-1}" \
  -e APEXNAV_SKIP_ITM_SERVER="${APEXNAV_SKIP_ITM_SERVER:-0}" \
  -e APEXNAV_SKIP_DETECTOR_SERVERS="${APEXNAV_SKIP_DETECTOR_SERVERS:-0}" \
  -e APEXNAV_VLM_API_KEY \
  -e OPENAI_API_KEY \
  -e VLM_API_KEY \
  -e APEXNAV_VLM_API_BASE \
  -e OPENAI_BASE_URL \
  -e VLM_API_BASE \
  -e APEXNAV_VLM_MODEL \
  -e VLM_MODEL \
  -v "${REPO_ROOT}:/workspace/Agent-apexnav:rw" \
  -v "${HOST_DATA_DIR}:/workspace/Agent-apexnav/data:ro" \
  "${IMAGE}" "$@"
