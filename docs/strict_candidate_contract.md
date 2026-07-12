# Strict candidate contract

The bridge derives one exact candidate set and reuses it for image annotation, the compact JSON observation, and result validation. Candidates are excluded when they are unreachable, invalid, missing a finite projection, outside image bounds, or attached to an unreadable panorama view.

The model action object is also validated strictly. Unknown top-level fields, unknown action fields, extra coordinate parameters, non-string candidate IDs, and non-empty LOOK/STOP parameters are rejected rather than ignored.
