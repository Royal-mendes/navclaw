# Strict original-frontier-only policy

The runtime selector entrypoint is `bridge/frontier_only_selector.py`.

Before the core bridge annotates an image or constructs a model observation, the wrapper keeps only candidate records whose `source` field is exactly `frontier`.

The following derived candidates are intentionally excluded:

- `frontier_cluster_average`
- `frontier_cluster_sample`
- `frontier_cluster_endpoint`
- `frontier_slid`
- local-view candidates
- connector proxies
- target-object proxies
- recovery proxies
- emergency escape proxies

The filtered candidate JSON is retained under `frontier_only_inputs/` for audit, and the result logs record counts grouped by discarded source.
