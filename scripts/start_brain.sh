#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${NAVCLAW_PORT:-8765}"
RUN_ID="${NAVCLAW_RUN_ID:-navclaw_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${NAVCLAW_RUN_DIR:-${ROOT}/runs/${RUN_ID}}"
PID_FILE="${RUN_DIR}/brain.pid"
ENV_FILE="${NAVCLAW_ENV_FILE:-/home/ubuntu/papers_repro/mapgpt_onmyagentnav/.env.vlm.local}"

mkdir -p "${RUN_DIR}"
if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

export NAVCLAW_RUN_ID="${RUN_ID}"
export NAVCLAW_RUN_DIR="${RUN_DIR}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ -f "${PID_FILE}" ]] && kill -0 "$(<"${PID_FILE}")" 2>/dev/null; then
  echo "NavClaw brain already running pid=$(<"${PID_FILE}") run_dir=${RUN_DIR}"
  exit 0
fi

nohup setsid python3 -m navclaw.server \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --project-root "${ROOT}" \
  --run-dir "${RUN_DIR}" \
  >"${RUN_DIR}/service.stdout.log" 2>&1 < /dev/null &
PID=$!
echo "${PID}" > "${PID_FILE}"

for _ in $(seq 1 30); do
  if curl -fsS "http://127.0.0.1:${PORT}/health" > "${RUN_DIR}/health.json"; then
    echo "NavClaw brain started pid=${PID} run_dir=${RUN_DIR}"
    exit 0
  fi
  if ! kill -0 "${PID}" 2>/dev/null; then
    echo "NavClaw brain exited during startup; see ${RUN_DIR}/service.stdout.log" >&2
    exit 1
  fi
  sleep 1
done

echo "NavClaw brain health check timed out" >&2
exit 1
