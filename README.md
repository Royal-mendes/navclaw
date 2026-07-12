# NavClaw

NavClaw keeps the deterministic AgentNav navigation stack below the candidate JSON boundary and replaces the MapGPT prompt/decision layer with a persistent, candidate-constrained agent service.

## Candidate policy

The current runtime is strictly original-frontier-only:

- only candidates whose lower-layer `source` is exactly `frontier` are exposed to the VLM;
- `frontier_cluster_average`, `frontier_cluster_sample`, `frontier_cluster_endpoint`, `frontier_slid`, local-view, connector, target, recovery, and emergency candidates are removed before annotation and prompt construction;
- the filtered candidate JSON and discarded-source counts are retained for audit.

## Action boundary

- the agent may select one currently projected, reachable candidate ID;
- the agent may request `LOOK_LEFT_60` or `LOOK_RIGHT_60`;
- it never receives or emits map coordinates or `safe_goal` values;
- API and parsing failures never create a geometric or nearest-candidate fallback;
- observations, raw replies, validation results, actions, and execution feedback are logged.

## Layout

- `navclaw/`: Brain, strict contracts, skills, and session memory.
- `bridge/selector_client.py`: core C++/NavClaw bridge.
- `bridge/frontier_only_selector.py`: required runtime entrypoint enforcing exact original-frontier-only selection.
- `tools/offline_replay.py`: replay candidate JSON and image logs.
- `profiles/`: identity and capability boundary documents.
- `scripts/`: service and bounded episode launch helpers.

## Failure contract

Transient API failures retry the exact same serialized model request. Invalid model output is retried without changing the observation or substituting an action. When attempts are exhausted, the request fails explicitly and no selector result file is created.
