#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${1:?usage: $0 NAVCLAW_RUN_DIR}"
CONTAINER_FILE="${RUN_DIR}/container_name.txt"
if [[ -f "${CONTAINER_FILE}" ]]; then
  CONTAINER="$(<"${CONTAINER_FILE}")"
  if docker inspect "${CONTAINER}" >/dev/null 2>&1; then
    docker stop -t 30 "${CONTAINER}" >/dev/null 2>&1 || docker kill "${CONTAINER}" >/dev/null 2>&1 || true
  fi
fi
"${ROOT}/scripts/stop_brain.sh" "${RUN_DIR}/brain"
echo "NavClaw episode resources stopped run_dir=${RUN_DIR}"
