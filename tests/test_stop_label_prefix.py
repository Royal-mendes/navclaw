import unittest

from navclaw.stop_gate import evaluate_stop_gate


class StopLabelPrefixTests(unittest.TestCase):
    def test_lower_detector_target_prefix_matches_task_target(self):
        result = evaluate_stop_gate(
            {
                "target": "bed",
                "detector_semantic_map": {
                    "objects": [
                        {
                            "label_hint": "target:bed",
                            "target_score": 0.82,
                            "distance": 0.72,
                            "relative_theta_deg": 4.0,
                            "target_observation_count": 3,
                        }
                    ]
                },
            }
        )
        self.assertTrue(result["eligible"])
        self.assertEqual(result["reason"], "verified_current_target")


if __name__ == "__main__":
    unittest.main()
