#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"${SCRIPT_DIR}/run_noetic_container.sh" bash -lc 'cd /workspace/Agent-apexnav && source /opt/ros/noetic/setup.bash && python3 -m py_compile habitat_evaluation.py vlm_waypoint_selector.py tools/make_ep60_full_process_video.py && catkin config -DPYTHON_EXECUTABLE=/usr/bin/python3 >/dev/null && catkin build exploration_manager --no-status -j1 -p1 --make-args -j1'
