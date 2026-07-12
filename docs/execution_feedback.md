# Execution feedback contract

The bridge maintains private per-session state for a selected waypoint. The private state may contain the lower-layer `safe_goal` and the robot pose at issue time, but neither value is included in the compact model observation or execution memory.

At the next selector invocation, the bridge compares the current robot pose with the pending goal and emits one sanitized outcome before the next model request:

- `reached`: final distance is within `NAVCLAW_WAYPOINT_REACHED_THRESHOLD` (default 0.25 m).
- `stalled_or_aborted`: a new selector decision was requested while the previous goal was still outside the threshold.
- `unknown`: the candidate JSON did not provide enough finite pose data to determine the result.

The Brain validates and records feedback by the previous request ID, then includes it in `RECENT_EXECUTION_MEMORY`. Feedback IDs are idempotent.

`stalled_or_aborted` intentionally combines several lower-layer causes. A future explicit planner callback can refine this into separate stall, timeout, cancellation, or collision outcomes without changing the Brain-memory contract.
