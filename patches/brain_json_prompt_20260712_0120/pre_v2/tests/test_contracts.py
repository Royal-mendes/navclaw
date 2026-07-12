import json
import unittest

from navclaw.brain import NavClawBrain
from navclaw.contracts import (
    ContractError,
    compact_observation,
    extract_json_object,
    normalize_agent_action,
    selectable_candidates,
)


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.candidates = [
            {"id": "F1", "reachable": True, "valid": True, "projection": [10, 20], "safe_goal": [1, 2]},
            {"id": "F2", "reachable": True, "valid": True, "projection": None, "safe_goal": [3, 4]},
            {"id": "F3", "reachable": False, "valid": True, "projection": [30, 40]},
        ]

    def test_selectable_candidates_exclude_unprojected_and_unreachable(self):
        selected = selectable_candidates(self.candidates)
        self.assertEqual([item["id"] for item in selected], ["F1"])
        self.assertNotIn("safe_goal", selected[0])

    def test_compact_observation_removes_coordinates_and_dense_grid(self):
        compact = compact_observation(
            {
                "target": "bed",
                "robot": {"x": 1.0, "y": 2.0, "yaw": 0.5},
                "metric_map_summary": {"occupancy_grid": {"free": [[1, 2]]}, "explored_ratio": 0.1},
                "candidates": self.candidates,
            }
        )
        encoded = str(compact)
        self.assertNotIn("safe_goal", encoded)
        self.assertNotIn("occupancy_grid", encoded)
        self.assertNotIn("'x'", encoded)

    def test_valid_current_candidate_is_accepted(self):
        result = normalize_agent_action(
            {
                "decision": "act",
                "action": {"skill": "select_waypoint", "parameters": {"candidate_id": "F1"}},
                "thinking": "current visible route",
            },
            ["F1"],
        )
        self.assertEqual(result["decision"], "SELECT_WAYPOINT")
        self.assertEqual(result["selected"], "F1")
        self.assertFalse(result["fallback"])

    def test_invalid_candidate_is_rejected_without_fallback(self):
        with self.assertRaises(ContractError):
            normalize_agent_action(
                {
                    "decision": "act",
                    "action": {"skill": "select_waypoint", "parameters": {"candidate_id": "F2"}},
                },
                ["F1"],
            )

    def test_look_action_is_allowed(self):
        result = normalize_agent_action(
            {"decision": "act", "action": {"skill": "look_left_60", "parameters": {}}},
            ["F1"],
        )
        self.assertEqual(result["decision"], "LOOK_LEFT_60")
        self.assertIsNone(result["selected"])
        self.assertFalse(result["fallback"])

    def test_full_scan_rejects_look(self):
        with self.assertRaises(ContractError):
            normalize_agent_action(
                {"decision": "act", "action": {"skill": "look_right_60", "parameters": {}}},
                ["F1"],
                force_select_waypoint=True,
            )

    def test_lowercase_candidate_is_not_fuzzy_matched(self):
        with self.assertRaises(ContractError):
            normalize_agent_action(
                {
                    "decision": "act",
                    "action": {"skill": "select_waypoint", "parameters": {"candidate_id": "f1"}},
                },
                ["F1"],
            )

    def test_place_id_and_coordinates_are_rejected(self):
        for action in (
            {"skill": "select_waypoint", "parameters": {"candidate_id": "Place 1"}},
            {"skill": "fly_to", "parameters": {"x": 1.0, "y": 2.0}},
        ):
            with self.assertRaises(ContractError):
                normalize_agent_action({"decision": "act", "action": action}, ["F1"])

    def test_fallback_flag_is_rejected(self):
        with self.assertRaises(ContractError):
            normalize_agent_action(
                {
                    "decision": "act",
                    "action": {"skill": "select_waypoint", "parameters": {"candidate_id": "F1"}},
                    "fallback": True,
                },
                ["F1"],
            )

    def test_malformed_json_is_rejected(self):
        with self.assertRaises(json.JSONDecodeError):
            extract_json_object("{not-json}")

    def test_vln_observation_drops_objectnav_semantics(self):
        compact = compact_observation(
            {
                "task_type": "vlnce",
                "instruction": "walk past the sofa",
                "target": "unknown",
                "detector_semantic_map": {"objects": [{"label_hint": "bed"}]},
                "candidates": self.candidates,
            }
        )
        self.assertEqual(compact["task"], "walk past the sofa")
        self.assertEqual(compact["semantic_objects"], [])

    def test_json_mode_prompt_contains_lowercase_json_keyword(self):
        brain = object.__new__(NavClawBrain)
        brain._profiles = {}
        messages = brain._messages(
            {"candidates": []},
            [],
            [],
            None,
        )
        self.assertIn("json", messages[0]["content"])


if __name__ == "__main__":
    unittest.main()
