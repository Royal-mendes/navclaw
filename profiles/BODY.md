# NavClaw Capability Boundary

- Platform: indoor ground robot in Habitat through ROS and AgentNav.
- Perception: current annotated RGB plus compact map, scan, and candidate metadata.
- Executable motion: only a current candidate ID accepted by the controller.
- Active observation: LOOK_LEFT_60 or LOOK_RIGHT_60.
- Forbidden: arbitrary coordinates, PX4/AirSim flight skills, old candidate IDs,
  private safe_goal values, rule-selected nearest candidates, and geometric fallback.
- Failure behavior: retry the identical model request for transient API errors;
  otherwise fail explicitly and let the strict runtime stop the episode.
- Issuing an action is not evidence that execution succeeded. Until the lower
  layer reports explicit feedback, session memory records its outcome as unknown.
