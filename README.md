# NavClaw

NavClaw keeps the deterministic AgentNav navigation stack below the candidate JSON boundary and replaces the MapGPT prompt/decision layer with a persistent, candidate-constrained agent service.

## Candidate policy

The current runtime is strictly original-frontier-only:

- only candidates whose lower-layer `source` is exactly `frontier` are exposed to the VLM;
- `frontier_cluster_average`, `frontier_cluster_sample`, `frontier_cluster_endpoint`, `frontier_slid`, local-view, connector, target, recovery, and emergency candidates are removed before annotation and prompt construction;
- the filtered candidate JSON and discarded-source counts are retained for audit.

## Action boundary

- movement is restricted to one currently projected, reachable original-frontier candidate ID;
- active observation uses `LOOK_LEFT_60` or `LOOK_RIGHT_60`;
- `STOP` is an independent verified termination request, not a waypoint type;
- STOP is exposed only when lower-layer target category, confidence, distance, view-angle, and repeated-observation checks pass, and the VLM must still confirm the target in the current RGB;
- the patched lower planner converts an accepted STOP request into the normal Habitat FINISH/STOP path;
- the agent never receives or emits map coordinates or `safe_goal` values;
- API and parsing failures never create a geometric or nearest-candidate fallback;
- observations, raw replies, validation results, actions, stop-gate evidence, and execution feedback are logged.

## Layout

- `navclaw/`: Brain, strict contracts, skills, session memory, and verified STOP gate.
- `bridge/selector_client.py`: core C++/NavClaw bridge.
- `bridge/frontier_only_selector.py`: required runtime entrypoint enforcing exact original-frontier-only movement and verified STOP exposure.
- `scripts/apply_verified_stop_patch.py`: idempotently patches the external lower planner to accept verified STOP and route it through the normal FSM finish path.
- `tools/offline_replay.py`: replay candidate JSON and image logs.
- `profiles/`: identity and capability boundary documents.
- `scripts/`: service and bounded episode launch helpers.

## Failure contract

Transient API failures retry the exact same serialized model request. Invalid model output is retried without changing the observation or substituting an action. When attempts are exhausted, the request fails explicitly and no selector result file is created.
