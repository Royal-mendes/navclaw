# NavClaw Capability Boundary

- Platform: indoor ground robot in Habitat through ROS and AgentNav.
- Perception: current annotated RGB plus compact map, scan, and candidate metadata.
- Executable motion: only a current original-frontier candidate ID accepted by the controller.
- Active observation: LOOK_LEFT_60 or LOOK_RIGHT_60.
- Verified termination: STOP may appear only after lower-layer target category,
  confidence, distance, current-view angle, and repeated-observation checks pass.
  When available, STOP is a termination request that still requires clear visual
  confirmation of the task target in the current RGB; it is not a speculative
  declaration of success.
- Forbidden: arbitrary coordinates, PX4/AirSim flight skills, old candidate IDs,
  private safe_goal values, rule-selected nearest candidates, geometric fallback,
  and STOP based only on memory, map hints, distant targets, or ambiguous objects.
- Failure behavior: retry the identical model request for transient API errors;
  otherwise fail explicitly and let the strict runtime stop the episode.
- Issuing an action is not evidence that execution succeeded. Until the lower
  layer reports explicit feedback, session memory records its outcome as unknown.
