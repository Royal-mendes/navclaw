import importlib.util
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "bridge" / "frontier_only_selector.py"
SPEC = importlib.util.spec_from_file_location("frontier_only_selector", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class FrontierOnlySelectorTests(unittest.TestCase):
    def candidates(self):
        sources = [
            "frontier",
            "frontier_cluster_average",
            "frontier_cluster_sample",
            "frontier_cluster_endpoint",
            "frontier_slid",
            "local_view",
            "connector_proxy",
            "target_object_proxy",
            "recovery_proxy",
            "emergency_escape_proxy",
        ]
        return [
            {
                "id": "C{0}".format(index),
                "source": source,
                "projection": [10 + index, 20],
                "reachable": True,
                "valid": True,
                "safe_goal": [float(index), 0.0],
                "view_id": "P1",
            }
            for index, source in enumerate(sources, start=1)
        ]

    def test_keeps_only_exact_original_frontier_source(self):
        selected = MODULE.original_frontier_candidates(self.candidates())
        self.assertEqual([candidate["source"] for candidate in selected], ["frontier"])
        self.assertEqual(selected[0]["id"], "C1")

    def test_preserves_private_execution_fields_for_kept_frontier(self):
        selected = MODULE.original_frontier_candidates(self.candidates())
        self.assertEqual(selected[0]["safe_goal"], [1.0, 0.0])
        self.assertEqual(selected[0]["view_id"], "P1")

    def test_reports_every_discarded_source(self):
        counts = MODULE.discarded_source_counts(self.candidates())
        self.assertNotIn("frontier", counts)
        self.assertEqual(counts["frontier_slid"], 1)
        self.assertEqual(counts["local_view"], 1)
        self.assertEqual(counts["target_object_proxy"], 1)
        self.assertEqual(sum(counts.values()), 9)

    def test_filtered_path_is_separate_audit_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = MODULE.filtered_json_path(
                root / "step_candidates.json",
                root / "results" / "step_vlm_result.json",
            )
            self.assertEqual(path.parent.name, "frontier_only_inputs")
            self.assertTrue(path.name.endswith("_original_frontier_only.json"))

    def test_rewrites_only_candidate_json_cli_value(self):
        argv = [
            "--candidate-json",
            "before.json",
            "--image",
            "frame.png",
            "--output-json",
            "result.json",
        ]
        rewritten = MODULE.replace_cli_option(argv, "--candidate-json", "after.json")
        self.assertEqual(rewritten[1], "after.json")
        self.assertEqual(rewritten[3:], argv[3:])


if __name__ == "__main__":
    unittest.main()
