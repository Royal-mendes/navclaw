# NavClaw

NavClaw keeps the deterministic AgentNav navigation stack below the candidate
JSON boundary and replaces the MapGPT prompt/decision layer with a persistent,
candidate-constrained agent service inspired by AerialClaw's Brain / Skill /
Memory split.

The current implementation deliberately has a narrow contract:

- the VLM receives only original planner frontiers whose source is exactly
  `frontier`;
- derived frontier cluster points, slid frontiers, local-view points, connector
  points, target proxies, recovery points, and emergency points are removed
  before image annotation and prompt construction;
- the agent may select one currently projected, reachable candidate ID;
- the agent may request `LOOK_LEFT_60` or `LOOK_RIGHT_60`;
- it never receives or emits map coordinates or `safe_goal` values;
- API and parsing failures never create a geometric or nearest-candidate
  fallback;
- every observation, raw model reply, validation result, selected action, and
  discarded candidate source count is logged.

The original repository remains untouched. A read-only lower-layer snapshot is
stored under `snapshots/`, while the pinned AerialClaw reference is stored under
`vendor/`.

## Layout

- `navclaw/`: persistent Brain, strict action contract, skills, and session memory.
- `bridge/selector_client.py`: core CLI-compatible bridge between C++ and NavClaw.
- `bridge/frontier_only_selector.py`: strict runtime entrypoint that exposes only
  exact `source="frontier"` candidates to the core bridge.
- `tools/offline_replay.py`: replay existing candidate JSON and image logs.
- `profiles/`: ground-navigation identity and capability boundary documents.
- `scripts/`: service and bounded single-episode launch helpers.

## Failure contract

Transient API failures retry the exact same serialized model request. Invalid
model output is retried without changing the observation or substituting an
action. When the configured attempts are exhausted, the request fails
explicitly and no selector result file is created.
