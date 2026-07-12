#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAPGPT_ROOT="${MAPGPT_ROOT:-/home/ubuntu/papers_repro/mapgpt_onmyagentnav}"
HOST_DATA_DIR="${HOST_DATA_DIR:-/home/ubuntu/papers_repro/ApexNav/data}"
IMAGE="${IMAGE:-agent-apexnav:noetic-gpu}"
EPISODE="${1:-21}"
DATASET="${2:-hm3dv2}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ID="${NAVCLAW_RUN_ID:-navclaw_smoke_ep${EPISODE}_${TIMESTAMP}}"
CONTAINER="${NAVCLAW_CONTAINER:-navclaw-ep${EPISODE}-${TIMESTAMP}}"
NAV_RUN_DIR="${ROOT}/runs/${RUN_ID}"
MAPGPT_RUN_DIR="${MAPGPT_ROOT}/logs/${RUN_ID}"
PORT="${NAVCLAW_PORT:-8765}"
SELECTOR_PATH="/workspace/navclaw/bridge/frontier_only_selector.py"
CANDIDATE_POLICY="original_frontier_only"

mkdir -p "${NAV_RUN_DIR}" "${MAPGPT_RUN_DIR}"
printf '%s\n' "${CONTAINER}" > "${NAV_RUN_DIR}/container_name.txt"
printf '%s\n' "${RUN_ID}" > "${NAV_RUN_DIR}/run_id.txt"
printf '%s\n' "${MAPGPT_RUN_DIR}" > "${NAV_RUN_DIR}/mapgpt_log_dir.txt"
printf '%s\n' "${CANDIDATE_POLICY}" > "${NAV_RUN_DIR}/candidate_policy.txt"

export NAVCLAW_RUN_ID="${RUN_ID}"
export NAVCLAW_RUN_DIR="${NAV_RUN_DIR}/brain"
export NAVCLAW_PORT="${PORT}"
"${ROOT}/scripts/start_brain.sh"
curl -fsS "http://127.0.0.1:${PORT}/health" > "${NAV_RUN_DIR}/brain_health_before.json"

if docker inspect "${CONTAINER}" >/dev/null 2>&1; then
  echo "Container already exists: ${CONTAINER}" >&2
  exit 1
fi

docker run -d \
  --name "${CONTAINER}" \
  --gpus all \
  --network host \
  --ipc host \
  --privileged \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e LD_PRELOAD=/opt/micromamba/envs/apexnav/lib/libstdc++.so.6 \
  -e PYTHONPATH=/workspace/Agent-apexnav/third_party/python \
  -e HF_HOME=/workspace/Agent-apexnav/third_party/hf_cache \
  -e TRANSFORMERS_CACHE=/workspace/Agent-apexnav/third_party/hf_cache \
  -e APEXNAV_DATA_DIR=/workspace/Agent-apexnav/data \
  -e APEXNAV_VLM_DEBUG_DIR=/workspace/Agent-apexnav/debug \
  -e APEXNAV_VLM_SELECTOR_SCRIPT="${SELECTOR_PATH}" \
  -e APEXNAV_VLM_FRONTIER_ONLY_CANDIDATES=1 \
  -e NAVCLAW_SERVER_URL="http://127.0.0.1:${PORT}/decide" \
  -e NAVCLAW_SESSION_ID="${RUN_ID}:ep${EPISODE}" \
  -e NAVCLAW_BRIDGE_TIMEOUT="${NAVCLAW_BRIDGE_TIMEOUT:-900}" \
  -e REPO_ROOT=/workspace/Agent-apexnav \
  -e APEXNAV_RUN_ID="${RUN_ID}" \
  -e APEXNAV_BATCH_LOG_DIR="/workspace/Agent-apexnav/logs/${RUN_ID}" \
  -v "${MAPGPT_ROOT}:/workspace/Agent-apexnav:rw" \
  -v "${HOST_DATA_DIR}:/workspace/Agent-apexnav/data:ro" \
  -v "${ROOT}:/workspace/navclaw:ro" \
  "${IMAGE}" \
  bash -lc "
    set -euo pipefail
    cd /workspace/Agent-apexnav
    export REPO_ROOT=/workspace/Agent-apexnav
    export APEXNAV_RUN_ID='${RUN_ID}'
    export APEXNAV_BATCH_LOG_DIR='/workspace/Agent-apexnav/logs/${RUN_ID}'
    export APEXNAV_VLM_SELECTOR_SCRIPT='${SELECTOR_PATH}'
    export APEXNAV_VLM_FRONTIER_ONLY_CANDIDATES=1
    export NAVCLAW_SERVER_URL='http://127.0.0.1:${PORT}/decide'
    export NAVCLAW_SESSION_ID='${RUN_ID}:ep${EPISODE}'
    export NAVCLAW_BRIDGE_TIMEOUT='${NAVCLAW_BRIDGE_TIMEOUT:-900}'
    export APEXNAV_VLM_MODEL='${APEXNAV_VLM_MODEL:-gpt-5.4}'
    export APEXNAV_VLM_REQUIRE_MODEL_DECISION=1
    export APEXNAV_VLM_REQUIRE_SUCCESS=1
    export APEXNAV_VLM_RETRY_FOREVER=1
    export APEXNAV_VLM_RETRY_NON_RETRYABLE=1
    export APEXNAV_VLM_DECISION_MAX_ATTEMPTS=0
    export APEXNAV_VLM_TIMEOUT=600
    export APEXNAV_VLM_HARD_TIMEOUT=780
    export APEXNAV_MAX_EPISODE_STEPS='${APEXNAV_MAX_EPISODE_STEPS:-300}'
    export APEXNAV_SAVE_ALL_VLM_DECISION_IMAGES=1
    export APEXNAV_SKIP_LOCAL_VLM_SERVERS=0
    export APEXNAV_SKIP_DETECTOR_SERVERS=0
    export APEXNAV_SKIP_BLIP2=1
    export APEXNAV_SKIP_ITM_SERVER=1
    export APEXNAV_SKIP_EP_ON_SCAN_NO_VISIBLE_CANDIDATES=1
    export APEXNAV_SKIP_EP_ON_ROS_WAIT_SECONDS=180
    exec timeout --signal=TERM --kill-after=30s 45m \
      scripts/run_mapgpt_upper_episode_list_hm3dv2.sh '${EPISODE}' '${DATASET}'
  "

docker inspect --format '{{.State.Status}} pid={{.State.Pid}}' "${CONTAINER}" \
  > "${NAV_RUN_DIR}/container_launch_state.txt"
echo "NavClaw episode launched run_id=${RUN_ID} container=${CONTAINER}"
echo "candidate_policy=${CANDIDATE_POLICY}"
echo "brain_logs=${NAV_RUN_DIR}/brain"
echo "navigation_logs=${MAPGPT_RUN_DIR}"
