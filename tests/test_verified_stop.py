import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from navclaw.stop_gate import evaluate_stop_gate


ROOT = Path(__file__).resolve().parents[1]
WRAPPER_PATH = ROOT / "bridge" / "frontier_only_selector.py"
PATCHER_PATH = ROOT / "scripts" / "apply_verified_stop_patch.py"

wrapper_spec = importlib.util.spec_from_file_location(
    "frontier_only_selector_verified_stop", WRAPPER_PATH
)
wrapper = importlib.util.module_from_spec(wrapper_spec)
assert wrapper_spec.loader is not None
wrapper_spec.loader.exec_module(wrapper)

patcher_spec = importlib.util.spec_from_file_location(
    "apply_verified_stop_patch", PATCHER_PATH
)
patcher = importlib.util.module_from_spec(patcher_spec)
assert patcher_spec.loader is not None
patcher_spec.loader.exec_module(patcher)


class StopGateTests(unittest.TestCase):
    def candidate_data(self, **overrides):
        obj = {
            "label_hint": "bed",
            "target_score": 0.8,
            "distance": 0.7,
            "relative_theta_deg": 5.0,
            "target_observation_count": 3,
        }
        obj.update(overrides)
        return {
            "target": "bed",
            "detector_semantic_map": {"objects": [obj]},
        }

    def test_exposes_stop_only_for_verified_current_target(self):
        result = evaluate_stop_gate(self.candidate_data())
        self.assertTrue(result["eligible"])
        self.assertEqual(result["reason"], "verified_current_target")

    def test_rejects_wrong_category(self):
        result = evaluate_stop_gate(self.candidate_data(label_hint="chair"))
        self.assertFalse(result["eligible"])
        self.assertEqual(result["reason"], "no_matching_target")

    def test_rejects_low_score(self):
        result = evaluate_stop_gate(self.candidate_data(target_score=0.4))
        self.assertEqual(result["reason"], "target_score_below_threshold")

    def test_rejects_target_outside_current_view(self):
        result = evaluate_stop_gate(self.candidate_data(relative_theta_deg=80.0))
        self.assertEqual(result["reason"], "target_outside_current_view")

    def test_rejects_distant_target(self):
        result = evaluate_stop_gate(self.candidate_data(distance=1.8))
        self.assertEqual(result["reason"], "target_too_far")

    def test_rejects_single_observation(self):
        result = evaluate_stop_gate(
            self.candidate_data(target_observation_count=1)
        )
        self.assertEqual(result["reason"], "insufficient_observations")

    def test_category_aliases_match(self):
        data = self.candidate_data(label_hint="sofa")
        data["target"] = "couch"
        self.assertTrue(evaluate_stop_gate(data)["eligible"])

    def test_thresholds_are_configurable(self):
        with mock.patch.dict(
            os.environ, {"NAVCLAW_STOP_MAX_DISTANCE_M": "2.0"}
        ):
            self.assertTrue(
                evaluate_stop_gate(self.candidate_data(distance=1.8))["eligible"]
            )

    def test_wrapper_adds_and_removes_stop_flag(self):
        argv = ["--candidate-json", "a.json", "--allow-stop"]
        enabled = wrapper.set_cli_flag(argv, "--allow-stop", True)
        disabled = wrapper.set_cli_flag(argv, "--allow-stop", False)
        self.assertEqual(enabled.count("--allow-stop"), 1)
        self.assertNotIn("--allow-stop", disabled)


class LowerStopPatcherTests(unittest.TestCase):
    def write_fixture(self, root):
        header = root / patcher.HEADER_REL
        manager = root / patcher.MANAGER_REL
        fsm = root / patcher.FSM_REL
        header.parent.mkdir(parents=True, exist_ok=True)
        manager.parent.mkdir(parents=True, exist_ok=True)
        fsm.parent.mkdir(parents=True, exist_ok=True)
        header.write_text(
            "string decision;                ///< SELECT_WAYPOINT, LOOK_LEFT_60, or LOOK_RIGHT_60\n"
            "  bool consumePendingVLMForcedAction(int& action_code);\n"
            "  int pending_vlm_forced_action_steps_ = 0;\n",
            encoding="utf-8",
        )
        manager.write_text(
            "bool parse() {\n"
            '  return decision.fallback || decision.decision == "SELECT_WAYPOINT" ||\n'
            '         decision.decision == "LOOK_LEFT_60" || decision.decision == "LOOK_RIGHT_60";\n'
            "}\n"
            "bool select() {\n"
            "  if (isVLMForcedLookDecision(decision.decision))\n"
            "    return true;\n\n"
            '  if (decision.decision != "SELECT_WAYPOINT") {\n'
            "    return false;\n"
            "  }\n"
            "}\n"
            "void ExplorationManager::findVLMGuidedFrontierPolicy() {\n"
            "  if (!query_ok) {\n"
            "  }\n"
            '  else if (decision.decision == "SELECT_WAYPOINT") {\n'
            "  }\n"
            "}\n"
            "void ExplorationManager::findTSPTourPolicy() {}\n",
            encoding="utf-8",
        )
        fsm.write_text(
            "  expl_res = expl_manager_->planNextBestPoint(fd_->start_pt_, fd_->start_yaw_(0));\n",
            encoding="utf-8",
        )

    def test_patches_lower_layer_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_fixture(root)
            first = patcher.patch_workspace(root)
            self.assertEqual(len(first["changed_files"]), 3)
            header = (root / patcher.HEADER_REL).read_text(encoding="utf-8")
            manager = (root / patcher.MANAGER_REL).read_text(encoding="utf-8")
            fsm = (root / patcher.FSM_REL).read_text(encoding="utf-8")
            self.assertIn("consumePendingVLMStopRequest", header)
            self.assertIn('decision.decision == "STOP"', manager)
            self.assertIn("pending_vlm_stop_request_ = true", manager)
            self.assertIn("FINAL_RESULT::REACH_OBJECT", fsm)
            second = patcher.patch_workspace(root)
            self.assertEqual(len(second["changed_files"]), 0)
            self.assertEqual(len(second["already_patched_files"]), 3)


if __name__ == "__main__":
    unittest.main()
