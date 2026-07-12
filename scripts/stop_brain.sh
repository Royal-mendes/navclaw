#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${1:-${NAVCLAW_RUN_DIR:-}}"
if [[ -z "${RUN_DIR}" ]]; then
  echo "usage: $0 RUN_DIR" >&2
  exit 2
fi
PID_FILE="${RUN_DIR}/brain.pid"
if [[ ! -f "${PID_FILE}" ]]; then
  echo "No brain pid file at ${PID_FILE}"
  exit 0
fi
PID="$(<"${PID_FILE}")"
if kill -0 "${PID}" 2>/dev/null; then
  kill "${PID}"
  for _ in $(seq 1 30); do
    if ! kill -0 "${PID}" 2>/dev/null; then
      break
    fi
    sleep 1
  done
fi
if kill -0 "${PID}" 2>/dev/null; then
  kill -KILL "${PID}"
fi
rm -f "${PID_FILE}"
echo "NavClaw brain stopped run_dir=${RUN_DIR}"
