# Design scope

This change does not modify the external C++ planner workspace. It uses the existing candidate JSON robot pose and private `safe_goal` fields inside the bridge, then sends only sanitized outcome metrics to the persistent Brain.
