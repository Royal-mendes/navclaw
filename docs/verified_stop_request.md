# Verified VLM STOP request

NavClaw keeps movement strictly original-frontier-only while treating STOP as a separate action.

## Gate

`bridge/frontier_only_selector.py` calls `navclaw.stop_gate.evaluate_stop_gate` on the lower-layer candidate JSON. STOP is added to `CURRENT_SKILLS` only when one target-matching detector record satisfies all configured checks:

- target category matches the task target;
- `target_score >= NAVCLAW_STOP_MIN_TARGET_SCORE` (default `0.65`);
- `distance <= NAVCLAW_STOP_MAX_DISTANCE_M` (default `1.0 m`);
- `abs(relative_theta_deg) <= NAVCLAW_STOP_HALF_FOV_DEG` (default `39.5 deg`);
- observation count is at least `NAVCLAW_STOP_MIN_OBSERVATIONS` (default `2`).

The gate is only lower-layer authorization. The VLM must still see the task target clearly in the current RGB before requesting STOP. It must not stop from memory, map hints, a distant object, or an ambiguous look-alike.

## Lower-layer execution

`scripts/apply_verified_stop_patch.py` idempotently patches the external AgentNav/ApexNav workspace used by `scripts/run_one_episode.sh`:

1. the C++ result parser accepts `STOP`;
2. the VLM frontier policy records a pending verified stop request;
3. the exploration FSM consumes the request and returns `FINAL_RESULT::REACH_OBJECT`;
4. the existing FINISH state publishes Habitat `ACTION::STOP`.

The launcher applies the patch before starting the container and rebuilds the `exploration_manager` package by default. Set `NAVCLAW_REBUILD_LOWER=0` only when that external workspace has already been rebuilt after the patch.

## Audit

The filtered candidate input and selector artifacts record:

- `stop_gate`;
- `stop_request_enabled`;
- the exact thresholds and sanitized evidence used by the gate;
- all discarded non-frontier candidate sources.
