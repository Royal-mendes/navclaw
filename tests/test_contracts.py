import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from navclaw.brain import NavClawBrain
from navclaw.contracts import (
    ContractError,
    compact_observation,
    extract_json_object,
    normalize_agent_action,
    normalize_execution_feedback,
    selectable_candidates,
)
from navclaw.memory import SessionMemory


BRIDGE_PATH = Path(__file__).resolve().parents[1] / "bridge" / "selector_client.py"
spec = importlib.util.spec_from_file_location("selector_client", BRIDGE_PATH)
bridge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bridge)


class FakeLLM:
    model = "fake-model"
    configured = True

    def __init__(self):
        self.messages = None

    def chat(self, messages, request_id):
        self.messages = messages
        return (
            json.dumps(
                {
                    "thinking": "avoid repeating the stalled route",
                    "decision": "act",
                    "action": {"skill": "look_left_60", "parameters": {}},
                    "reflection": "the previous waypoint did not complete",
                    "goal_progress": "seeking a new route",
                    "confidence": 0.7,
                }
            ),
            "stable-hash",
            1,
        )


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.candidates = [
            {"id": "F1", "reachable": True, "valid": True, "projection": [10, 20], "safe_goal": [1, 2]},
            {"id": "F2", "reachable": True, "valid": True, "projection": None, "safe_goal": [3, 4]},
            {"id": "F3", "reachable": False, "valid": True, "projection": [30, 40]},
            {"id": "F4", "reachable": True, "valid": True, "projected": True, "projection": [float("nan"), 4]},
        ]

    def valid_action(self, skill="select_waypoint", parameters=None):
        if parameters is None:
            parameters = {"candidate_id": "F1"} if skill == "select_waypoint" else {}
        return {
            "thinking": "current visible route",
            "decision": "act",
            "action": {"skill": skill, "parameters": parameters},
            "reflection": None,
            "goal_progress": "exploring",
            "confidence": 0.8,
        }

    def test_selectable_candidates_exclude_unprojected_unreachable_and_nonfinite(self):
        selected = selectable_candidates(self.candidates)
        self.assertEqual([item["id"] for item in selected], ["F1"])
        self.assertNotIn("safe_goal", selected[0])
        self.assertNotIn("projection", selected[0])

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
        result = normalize_agent_action(self.valid_action(), ["F1"])
        self.assertEqual(result["decision"], "SELECT_WAYPOINT")
        self.assertEqual(result["selected"], "F1")
        self.assertFalse(result["fallback"])

    def test_invalid_candidate_is_rejected_without_fallback(self):
        with self.assertRaises(ContractError):
            normalize_agent_action(
                self.valid_action(parameters={"candidate_id": "F2"}),
                ["F1"],
            )

    def test_look_action_is_allowed(self):
        result = normalize_agent_action(self.valid_action("look_left_60"), ["F1"])
        self.assertEqual(result["decision"], "LOOK_LEFT_60")
        self.assertIsNone(result["selected"])

    def test_full_scan_rejects_look(self):
        with self.assertRaises(ContractError):
            normalize_agent_action(
                self.valid_action("look_right_60"),
                ["F1"],
                force_select_waypoint=True,
            )

    def test_strict_schema_rejects_extra_coordinate_parameter(self):
        with self.assertRaisesRegex(ContractError, "only_contain_candidate_id"):
            normalize_agent_action(
                self.valid_action(parameters={"candidate_id": "F1", "x": 1.0, "y": 2.0}),
                ["F1"],
            )

    def test_strict_schema_rejects_extra_top_level_field(self):
        action = self.valid_action()
        action["fallback"] = False
        with self.assertRaisesRegex(ContractError, "unknown_fields"):
            normalize_agent_action(action, ["F1"])

    def test_candidate_id_is_not_coerced(self):
        with self.assertRaisesRegex(ContractError, "candidate_id_must_be_string"):
            normalize_agent_action(
                self.valid_action(parameters={"candidate_id": 1}),
                ["1"],
            )

    def test_place_id_and_unlisted_skill_are_rejected(self):
        for action in (
            self.valid_action(parameters={"candidate_id": "Place 1"}),
            self.valid_action("fly_to", {"x": 1.0, "y": 2.0}),
        ):
            with self.assertRaises(ContractError):
                normalize_agent_action(action, ["F1"])

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

    def test_execution_feedback_contract(self):
        normalized = normalize_execution_feedback(
            {
                "feedback_id": "fb1",
                "request_id": "req1",
                "next_request_id": "req2",
                "selected_id": "F1",
                "execution_outcome": "reached",
                "source": "bridge_next_decision_observation",
                "final_distance_m": 0.12,
            }
        )
        self.assertEqual(normalized["execution_outcome"], "reached")
        with self.assertRaises(ContractError):
            normalize_execution_feedback({**normalized, "safe_goal": [1, 2]})

    def test_renderable_candidates_are_exactly_drawable(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "rgb.png"
            Image.new("RGB", (100, 80)).save(image_path)
            data = {
                "candidates": [
                    {"id": "F1", "reachable": True, "valid": True, "projection": [10, 20]},
                    {"id": "F2", "reachable": True, "valid": True, "projection": [120, 20]},
                    {"id": "F3", "reachable": False, "valid": True, "projection": [30, 20]},
                    {"id": "F4", "reachable": True, "valid": True, "projection": None},
                ]
            }
            selected = bridge.renderable_candidates(data, image_path)
            self.assertEqual([item["id"] for item in selected], ["F1"])
            output = Path(tmp) / "annotated.png"
            bridge.annotate_current_image(image_path, output, selected)
            self.assertTrue(output.exists())

    def test_execution_feedback_marks_reached(self):
        state = {
            "pending_waypoint": {
                "request_id": "req1",
                "selected_id": "F1",
                "safe_goal": [1.0, 0.0],
                "start_robot_xy": [0.0, 0.0],
                "step": 1,
                "issued_at": 100.0,
                "reached_threshold_m": 0.25,
            }
        }
        data = {"robot": {"x": 0.9, "y": 0.0}, "step": 2}
        feedback = bridge.build_execution_feedback(state, data, "req2")
        self.assertEqual(feedback["execution_outcome"], "reached")
        self.assertAlmostEqual(feedback["final_distance_m"], 0.1)
        self.assertNotIn("safe_goal", feedback)

    def test_execution_feedback_marks_stalled_or_aborted(self):
        state = {
            "pending_waypoint": {
                "request_id": "req1",
                "selected_id": "F1",
                "safe_goal": [2.0, 0.0],
                "start_robot_xy": [0.0, 0.0],
                "step": 1,
                "reached_threshold_m": 0.25,
            }
        }
        data = {"robot": {"x": 0.2, "y": 0.0}, "step": 2}
        feedback = bridge.build_execution_feedback(state, data, "req2")
        self.assertEqual(feedback["execution_outcome"], "stalled_or_aborted")
        self.assertAlmostEqual(feedback["travel_distance_m"], 0.2)

    def test_feedback_updates_prior_memory_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory = SessionMemory(Path(tmp), max_prompt_items=8)
            memory.append(
                "session1",
                {
                    "status": "ok",
                    "request_id": "req1",
                    "step": 1,
                    "target": "bed",
                    "valid_candidate_ids": ["F1"],
                    "result": {
                        "decision": "SELECT_WAYPOINT",
                        "selected": "F1",
                        "reason": "go forward",
                    },
                },
            )
            feedback = {
                "feedback_id": "fb1",
                "request_id": "req1",
                "next_request_id": "req2",
                "selected_id": "F1",
                "execution_outcome": "reached",
                "source": "bridge_next_decision_observation",
                "final_distance_m": 0.1,
            }
            self.assertTrue(memory.append_feedback("session1", feedback))
            self.assertFalse(memory.append_feedback("session1", feedback))
            recent = memory.recent("session1")
            self.assertEqual(recent[0]["execution_outcome"], "reached")

    def test_feedback_is_in_prompt_before_next_decision(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            brain = NavClawBrain(root, root / "run")
            brain.max_decision_attempts = 1
            fake_llm = FakeLLM()
            brain.llm = fake_llm
            brain.memory.append(
                "session1",
                {
                    "status": "ok",
                    "request_id": "req1",
                    "step": 1,
                    "target": "bed",
                    "valid_candidate_ids": ["F1"],
                    "result": {
                        "decision": "SELECT_WAYPOINT",
                        "selected": "F1",
                        "reason": "try the doorway",
                    },
                },
            )
            result = brain.decide(
                {
                    "request_id": "req2",
                    "session_id": "session1",
                    "observation": {"target": "bed", "step": 2, "candidates": []},
                    "execution_feedback": {
                        "feedback_id": "fb1",
                        "request_id": "req1",
                        "next_request_id": "req2",
                        "selected_id": "F1",
                        "execution_outcome": "stalled_or_aborted",
                        "source": "bridge_next_decision_observation",
                        "final_distance_m": 1.2,
                    },
                }
            )
            self.assertEqual(result["decision"], "LOOK_LEFT_60")
            user_message = next(
                message for message in fake_llm.messages if message["role"] == "user"
            )
            prompt_text = "\n".join(
                part["text"]
                for part in user_message["content"]
                if part.get("type") == "text"
            )
            self.assertIn("stalled_or_aborted", prompt_text)

    def test_json_mode_user_input_contains_lowercase_json_keyword(self):
        brain = object.__new__(NavClawBrain)
        brain._profiles = {}
        messages = brain._messages({"candidates": []}, [], [], None)
        user_message = next(message for message in messages if message["role"] == "user")
        user_text = "\n".join(
            part["text"]
            for part in user_message["content"]
            if part.get("type") == "text"
        )
        self.assertIn("json", user_text)


if __name__ == "__main__":
    unittest.main()
