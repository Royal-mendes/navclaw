"""
Habitat ObjectNav Evaluation Script for HM3D/MP3D Datasets

This script evaluates object navigation performance using the Habitat simulator
with support for HM3D-v1, HM3D-v2, and MP3D datasets. It communicates with ROS for
real-time planning and decision making, incorporates vision-language models
for object detection and image-text matching, and generates comprehensive
evaluation metrics.

Usage:
    # Run with HM3D-v1 dataset
    python habitat_evaluation.py --dataset hm3dv1

    # Run with HM3D-v2 dataset (default)
    python habitat_evaluation.py --dataset hm3dv2

    # Run with MP3D dataset
    python habitat_evaluation.py --dataset mp3d

    # Test specific episode
    python habitat_evaluation.py --dataset hm3dv2 test_epi_num=10

Author: Zager-Zhang
"""

# Standard library imports
import argparse
import gzip
import json
import os
import signal
import time
from copy import deepcopy

# Third-party library imports
from hydra import initialize, compose
import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from omegaconf import DictConfig, OmegaConf
from prettytable import PrettyTable
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import (
    Float32MultiArray,
    Float64,
    Float64MultiArray,
    Int32,
    Int32MultiArray,
    String,
)
import tqdm

# Habitat-related imports
import habitat
from habitat.config.default import patch_config
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.utils.visualizations.utils import (
    images_to_video,
    observations_to_image,
    overlay_frame,
)
try:
    from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
except Exception:
    ShortestPathFollower = None
from habitat.utils.geometry_utils import quaternion_from_coeff, quaternion_rotate_vector

# ROS message imports
from plan_env.msg import MultipleMasksWithConfidence

# Local project imports
from basic_utils.failure_check.count_files import count_files_in_directory
from basic_utils.failure_check.failure_check import check_failure, is_on_same_floor
from basic_utils.object_point_cloud_utils.object_point_cloud import (
    get_object_point_cloud,
)
from basic_utils.record_episode.read_record import read_record
from basic_utils.record_episode.write_record import write_record
from habitat2ros import habitat_publisher
from llm.answer_reader.answer_reader import read_answer
from params import HABITAT_STATE, ROS_STATE, ACTION, RESULT_TYPES
from vlm.Labels import MP3D_ID_TO_NAME
from vlm.utils.get_itm_message import get_itm_message_cosine
from vlm.utils.get_object_utils import get_object


def env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def stable_scene_basename(episode):
    scene_id = str(getattr(episode, "scene_id", "") or "")
    base = os.path.basename(scene_id)
    if base.endswith(".glb"):
        base = os.path.splitext(base)[0]
    if base.endswith(".basis"):
        base = os.path.splitext(base)[0]
    if not base and scene_id:
        base = scene_id.rstrip("/").split("/")[-1].split("-")[-1]
    return base


def stable_sorted_episode_at(env, index):
    episodes = getattr(env, "episodes", None)
    if episodes is None:
        dataset = getattr(env, "_dataset", None)
        episodes = getattr(dataset, "episodes", None)
    episodes = list(episodes or [])
    if index < 0 or index >= len(episodes):
        raise IndexError(f"stable_episode_index_out_of_range:{index}/{len(episodes)}")
    indexed = list(enumerate(episodes))
    indexed.sort(key=lambda item: (stable_scene_basename(item[1]), item[0]))
    return indexed[index][1]


def publish_int32(publisher, data):
    msg = Int32()
    msg.data = data
    publisher.publish(msg)


def publish_float64(publisher, data):
    msg = Float64()
    msg.data = data
    publisher.publish(msg)


def publish_int32_array(publisher, data_list):
    msg = Int32MultiArray()
    msg.data = data_list
    publisher.publish(msg)


def publish_float32_array(publisher, data_list):
    msg = Float32MultiArray()
    msg.data = data_list
    publisher.publish(msg)


def publish_float64_array(publisher, data_list):
    msg = Float64MultiArray()
    msg.data = data_list
    publisher.publish(msg)


def habitat_point_to_ros_xy(point, env=None):
    if env is not None:
        try:
            origin = np.array(env.current_episode.start_position, dtype=np.float64)
            rotation_world_start = quaternion_from_coeff(env.current_episode.start_rotation)
            episode_pos = quaternion_rotate_vector(
                rotation_world_start.inverse(), np.asarray(point, dtype=np.float64) - origin
            )
            return (-float(episode_pos[2]), -float(episode_pos[0]))
        except Exception as exc:
            print(f"FrontierOracle: world-to-episode conversion failed: {exc}")
    return (-float(point[2]), -float(point[0]))


def safe_agent_world_position(env):
    try:
        return np.array(env.sim.get_agent_state().position, dtype=np.float64)
    except Exception:
        return None


def episode_goal_navigation_positions(env, include_object_centers=True):
    positions = []
    for goal in getattr(env.current_episode, "goals", []):
        for view_point in getattr(goal, "view_points", []) or []:
            agent_state = getattr(view_point, "agent_state", None)
            point = getattr(agent_state, "position", None)
            if point is None:
                point = getattr(view_point, "position", None)
            if point is not None:
                point = np.asarray(point, dtype=np.float64)
                if point.shape == (3,) and np.all(np.isfinite(point)):
                    positions.append(point)
        if include_object_centers and hasattr(goal, "position"):
            point = np.asarray(goal.position, dtype=np.float64)
            if point.shape == (3,) and np.all(np.isfinite(point)):
                positions.append(point)
    return positions


def build_critic_gt_reference(env):
    """Build private GT path metadata for critic logging only."""
    agent_pos = safe_agent_world_position(env)
    if agent_pos is None:
        return [], (np.nan, np.nan), False

    best_goal = None
    best_dist = np.inf
    for goal_pos in episode_goal_navigation_positions(env):
        try:
            dist = env.sim.geodesic_distance(agent_pos, goal_pos, env.current_episode)
        except Exception:
            dist = np.linalg.norm(agent_pos[[0, 2]] - goal_pos[[0, 2]])
        if np.isfinite(dist) and dist < best_dist:
            best_dist = dist
            best_goal = goal_pos

    if best_goal is None:
        return [], (np.nan, np.nan), False

    gt_path_points = []
    try:
        gt_path_points = [
            np.array(point, dtype=np.float64)
            for point in env.sim.get_straight_shortest_path_points(agent_pos, best_goal)
        ]
    except Exception as exc:
        print(f"GTTrainingCritic: shortest path unavailable: {exc}")

    return gt_path_points, habitat_point_to_ros_xy(best_goal, env), True


def endpoint_to_gt_path(env, gt_path_points):
    agent_pos = safe_agent_world_position(env)
    if agent_pos is None or not gt_path_points:
        return np.nan, False
    distances = [
        np.linalg.norm(agent_pos[[0, 2]] - np.asarray(point, dtype=np.float64)[[0, 2]])
        for point in gt_path_points
    ]
    if not distances:
        return np.nan, False
    return float(np.min(distances)), True


def publish_critic_gt(
    publisher,
    env,
    episode_id,
    step_id,
    distance_to_goal,
    gt_path_points,
    target_ros_xy,
    have_target_position,
):
    endpoint_dist, have_gt_path = endpoint_to_gt_path(env, gt_path_points)
    distance = float(distance_to_goal) if np.isfinite(distance_to_goal) else -1.0
    target_x, target_y = target_ros_xy if have_target_position else (np.nan, np.nan)
    publish_float64_array(
        publisher,
        [
            float(episode_id),
            float(step_id),
            distance,
            float(target_x),
            float(target_y),
            1.0 if have_target_position else 0.0,
            endpoint_dist if have_gt_path else -1.0,
            1.0 if have_gt_path else 0.0,
        ],
    )


_last_frontier_oracle_request_id = ""


def ros_xy_to_habitat_point(env, x, y):
    # /habitat/odom is published from Habitat GPS, whose coordinates are in the
    # episode-local frame. Convert ROS xy back to episode coordinates, then rotate
    # and translate into Habitat world coordinates for geodesic queries.
    agent_pos = safe_agent_world_position(env)
    try:
        origin = np.array(env.current_episode.start_position, dtype=np.float64)
        rotation_world_start = quaternion_from_coeff(env.current_episode.start_rotation)
        height = 0.0
        if agent_pos is not None:
            agent_episode_pos = quaternion_rotate_vector(
                rotation_world_start.inverse(), agent_pos - origin
            )
            height = float(agent_episode_pos[1])
        episode_pos = np.array([-float(y), height, -float(x)], dtype=np.float64)
        return origin + quaternion_rotate_vector(rotation_world_start, episode_pos)
    except Exception as exc:
        print(f"FrontierOracle: episode-to-world conversion failed: {exc}")
        height = float(agent_pos[1]) if agent_pos is not None else 0.0
        return np.array([-float(y), height, -float(x)], dtype=np.float64)


def snap_to_navmesh(env, point, max_snap_distance=2.0):
    try:
        snapped = np.array(env.sim.pathfinder.snap_point(point), dtype=np.float64)
    except Exception as exc:
        print(f"FrontierOracle: navmesh snap failed: {exc}")
        return point, False, np.inf

    if not np.all(np.isfinite(snapped)):
        return point, False, np.inf

    snap_distance = float(np.linalg.norm(snapped - point))
    if snap_distance > max_snap_distance:
        return point, False, snap_distance

    return snapped, True, snap_distance


def safe_geodesic_distance(env, start, ends):
    if isinstance(ends, np.ndarray) and ends.shape == (3,):
        ends = [ends]
    # For arbitrary frontier candidates, prefer the direct start/end geodesic
    # query. Passing the episode can make Habitat resolve distance to the
    # episode goal instead of the supplied candidate endpoint.
    try:
        dist = env.sim.geodesic_distance(start, ends)
    except Exception:
        try:
            dist = env.sim.geodesic_distance(start, ends, env.current_episode)
        except Exception:
            return np.inf
    return float(dist) if np.isfinite(dist) and dist >= 0 else np.inf


def select_best_goal_by_geodesic(env, start, goal_positions):
    best_goal = None
    best_dist = np.inf
    for goal_pos in goal_positions:
        dist = safe_geodesic_distance(env, start, goal_pos)
        if np.isfinite(dist) and dist < best_dist:
            best_dist = dist
            best_goal = goal_pos
    return best_goal, best_dist


def safe_shortest_path_points(env, start, goal):
    try:
        points = env.sim.get_straight_shortest_path_points(start, goal)
    except Exception as exc:
        print(f"FrontierOracle: shortest path unavailable: {exc}")
        points = []

    path_points = [
        np.array(point, dtype=np.float64)
        for point in points
        if np.all(np.isfinite(np.asarray(point, dtype=np.float64)))
    ]
    if len(path_points) < 2:
        try:
            import habitat_sim

            shortest_path = habitat_sim.ShortestPath()
            shortest_path.requested_start = np.asarray(start, dtype=np.float32)
            shortest_path.requested_end = np.asarray(goal, dtype=np.float32)
            if env.sim.pathfinder.find_path(shortest_path):
                path_points = [
                    np.array(point, dtype=np.float64)
                    for point in shortest_path.points
                    if np.all(np.isfinite(np.asarray(point, dtype=np.float64)))
                ]
        except Exception as exc:
            print(f"FrontierOracle: pathfinder shortest path unavailable: {exc}")

    if len(path_points) < 2:
        return []
    if np.linalg.norm(path_points[0][[0, 2]] - np.asarray(start)[[0, 2]]) > 0.25:
        path_points.insert(0, np.array(start, dtype=np.float64))
    return path_points


def point_to_polyline_progress_xz(point, path_points):
    if len(path_points) < 2:
        return np.inf, np.nan, 0.0

    query = np.asarray(point, dtype=np.float64)[[0, 2]]
    best_dist = np.inf
    best_s = 0.0
    path_s = 0.0
    path_len = 0.0

    for start, end in zip(path_points[:-1], path_points[1:]):
        a = np.asarray(start, dtype=np.float64)[[0, 2]]
        b = np.asarray(end, dtype=np.float64)[[0, 2]]
        seg = b - a
        seg_len = float(np.linalg.norm(seg))
        if seg_len < 1e-6:
            continue
        path_len += seg_len
        t = float(np.clip(np.dot(query - a, seg) / (seg_len * seg_len), 0.0, 1.0))
        proj = a + t * seg
        dist = float(np.linalg.norm(query - proj))
        if dist < best_dist:
            best_dist = dist
            best_s = path_s + t * seg_len
        path_s += seg_len

    if not np.isfinite(best_dist):
        best_dist = float(np.linalg.norm(query - np.asarray(path_points[0])[[0, 2]]))
        best_s = 0.0
    return best_dist, best_s, path_len


def point_at_path_s_xz(path_points, target_s):
    if not path_points:
        return None
    if len(path_points) == 1 or target_s <= 0:
        return np.array(path_points[0], dtype=np.float64)

    walked = 0.0
    for start, end in zip(path_points[:-1], path_points[1:]):
        start = np.asarray(start, dtype=np.float64)
        end = np.asarray(end, dtype=np.float64)
        seg_len = float(np.linalg.norm(end[[0, 2]] - start[[0, 2]]))
        if seg_len < 1e-6:
            continue
        if walked + seg_len >= target_s:
            t = (target_s - walked) / seg_len
            return start + t * (end - start)
        walked += seg_len
    return np.array(path_points[-1], dtype=np.float64)


def path_downward_metrics(path_points, start=None, end=None):
    points = [
        np.asarray(point, dtype=np.float64)
        for point in (path_points or [])
        if np.all(np.isfinite(np.asarray(point, dtype=np.float64)))
    ]
    if start is not None and np.all(np.isfinite(np.asarray(start, dtype=np.float64))):
        start_point = np.asarray(start, dtype=np.float64)
        if not points or np.linalg.norm(points[0] - start_point) > 1e-3:
            points.insert(0, start_point)
    if end is not None and np.all(np.isfinite(np.asarray(end, dtype=np.float64))):
        end_point = np.asarray(end, dtype=np.float64)
        if not points or np.linalg.norm(points[-1] - end_point) > 1e-3:
            points.append(end_point)
    if not points:
        return 0.0, 0.0, np.nan

    start_height = float(points[0][1])
    min_height = float(min(point[1] for point in points))
    end_height = float(points[-1][1])
    downward_drop = max(0.0, start_height - min_height)
    endpoint_delta = end_height - start_height
    return downward_drop, endpoint_delta, min_height


def densify_path_points_xz(path_points, max_step=0.35):
    dense = []
    points = [
        np.asarray(point, dtype=np.float64)
        for point in (path_points or [])
        if np.all(np.isfinite(np.asarray(point, dtype=np.float64)))
    ]
    if not points:
        return dense
    dense.append(points[0])
    for start, end in zip(points[:-1], points[1:]):
        delta = end - start
        seg_len = float(np.linalg.norm(delta[[0, 2]]))
        if seg_len < 1e-6:
            continue
        steps = max(1, int(np.ceil(seg_len / max(0.05, max_step))))
        for step in range(1, steps + 1):
            dense.append(start + (step / steps) * delta)
    return dense


def path_stairwell_metrics(env, path_points, max_drop_threshold=0.35):
    """Detect stair/landing exposure by sampling lower navmesh near the path."""
    points = densify_path_points_xz(path_points, max_step=0.30)
    if not points:
        return 0.0, False

    radii = (0.35, 0.55, 0.80, 1.05)
    angles = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    max_local_drop = 0.0
    strong_drop_hits = 0
    for point in points:
        base = np.asarray(point, dtype=np.float64)
        for radius in radii:
            for angle in angles:
                query = np.array(
                    [
                        base[0] + radius * np.cos(angle),
                        base[1],
                        base[2] + radius * np.sin(angle),
                    ],
                    dtype=np.float64,
                )
                try:
                    snapped = np.asarray(env.sim.pathfinder.snap_point(query), dtype=np.float64)
                except Exception:
                    continue
                if not np.all(np.isfinite(snapped)):
                    continue
                xz_snap_distance = float(np.linalg.norm(snapped[[0, 2]] - query[[0, 2]]))
                if xz_snap_distance > 0.75:
                    continue
                local_drop = float(base[1] - snapped[1])
                if local_drop <= 0.0:
                    continue
                max_local_drop = max(max_local_drop, local_drop)
                if local_drop > max_drop_threshold:
                    strong_drop_hits += 1
                    if strong_drop_hits >= 2:
                        return max_local_drop, True
    return max_local_drop, max_local_drop > max_drop_threshold


def maybe_handle_frontier_oracle_request(env):
    """Respond to C++ frontier-oracle candidate scoring requests with private GT geodesics."""
    global _last_frontier_oracle_request_id
    try:
        request_id = rospy.get_param("/apexnav/frontier_oracle/request_id", "")
    except Exception:
        return
    if not request_id or request_id == _last_frontier_oracle_request_id:
        return

    try:
        candidate_ids = rospy.get_param("/apexnav/frontier_oracle/candidate_ids", [])
        candidate_xy = rospy.get_param("/apexnav/frontier_oracle/candidate_xy", [])
        current_xy = rospy.get_param("/apexnav/frontier_oracle/current_xy", [])
    except Exception as exc:
        print(f"FrontierOracle: failed to read request params: {exc}")
        return

    if not isinstance(candidate_ids, list) or len(current_xy) < 2:
        return
    if len(candidate_xy) < 2 * len(candidate_ids):
        return

    # Score frontier requests from the planner's reported state.  The Habitat
    # agent pose can lag the planner pose while ROS is executing a waypoint,
    # which makes GT-path lookahead point back to the episode start.
    current_pos = ros_xy_to_habitat_point(env, current_xy[0], current_xy[1])
    if current_pos is None or not np.all(np.isfinite(current_pos)):
        current_pos = safe_agent_world_position(env)
    current_pos, current_snapped, current_snap_distance = snap_to_navmesh(env, current_pos)
    if not current_snapped:
        print(
            "FrontierOracle: current position could not be snapped to navmesh "
            f"(snap_distance={current_snap_distance}) for request {request_id}"
        )

    goal_positions = episode_goal_navigation_positions(env)
    if not goal_positions:
        return

    lookahead_m = float(
        rospy.get_param("/apexnav/frontier_oracle/gt_path_lookahead_m", 1.5)
    )
    max_downward_drop_m = float(
        rospy.get_param("/apexnav/frontier_oracle/max_downward_drop_m", 0.35)
    )
    best_goal, _best_goal_dist = select_best_goal_by_geodesic(
        env, current_pos, goal_positions
    )
    gt_path_points = safe_shortest_path_points(env, current_pos, best_goal) if best_goal is not None else []
    path_available = len(gt_path_points) >= 2
    current_gt_path_s = 0.0
    gt_path_length = 0.0
    lookahead_xy = [0.0, 0.0]
    gt_path_downward_drop = 0.0
    gt_path_endpoint_height_delta = 0.0
    if path_available:
        gt_path_downward_drop, gt_path_endpoint_height_delta, _gt_path_min_height = (
            path_downward_metrics(gt_path_points, start=current_pos, end=best_goal)
        )
        _current_path_dist, current_gt_path_s, gt_path_length = point_to_polyline_progress_xz(
            current_pos, gt_path_points
        )
        lookahead_s = min(current_gt_path_s + max(0.1, lookahead_m), gt_path_length)
        lookahead_point = point_at_path_s_xz(gt_path_points, lookahead_s)
        if lookahead_point is not None:
            lookahead_xy = list(habitat_point_to_ros_xy(lookahead_point, env))

    target_shortest_path_downstairs = gt_path_downward_drop > max_downward_drop_m
    # A shortest path that drops to a lower floor is useful diagnostic metadata,
    # but it is not a valid teaching path for this project.  In that case, do
    # not use its polyline as the "GT path" for waypoint progress or scan
    # lookahead; the planner should search for same-floor visible candidates.
    learning_path_available = path_available and not target_shortest_path_downstairs

    response_scores = []
    for i, _candidate_id in enumerate(candidate_ids):
        x = candidate_xy[2 * i]
        y = candidate_xy[2 * i + 1]
        candidate_pos = ros_xy_to_habitat_point(env, x, y)
        candidate_pos, snapped, snap_distance = snap_to_navmesh(env, candidate_pos)
        if not snapped:
            print(
                "FrontierOracle: candidate could not be snapped to navmesh "
                f"id={_candidate_id} snap_distance={snap_distance} request={request_id}"
            )
        current_to_candidate = safe_geodesic_distance(env, current_pos, candidate_pos)
        candidate_best_goal, candidate_to_target = select_best_goal_by_geodesic(
            env, candidate_pos, goal_positions
        )
        available = np.isfinite(current_to_candidate) and np.isfinite(candidate_to_target)
        total = current_to_candidate + candidate_to_target if available else np.inf
        endpoint_to_path = np.inf
        candidate_gt_path_s = np.nan
        gt_path_progress = np.nan
        progress_per_cost = np.nan
        gt_path_available = False
        if learning_path_available:
            endpoint_to_path, candidate_gt_path_s, _path_len = point_to_polyline_progress_xz(
                candidate_pos, gt_path_points
            )
            gt_path_progress = candidate_gt_path_s - current_gt_path_s
            if np.isfinite(gt_path_progress) and np.isfinite(current_to_candidate):
                progress_per_cost = gt_path_progress / max(current_to_candidate, 1e-3)
            gt_path_available = (
                np.isfinite(endpoint_to_path)
                and np.isfinite(candidate_gt_path_s)
                and np.isfinite(gt_path_progress)
            )
        candidate_path_points = safe_shortest_path_points(env, current_pos, candidate_pos)
        candidate_downward_drop, candidate_endpoint_height_delta, _candidate_min_height = (
            path_downward_metrics(candidate_path_points, start=current_pos, end=candidate_pos)
        )
        candidate_stairwell_drop, candidate_stairwell_path = path_stairwell_metrics(
            env, candidate_path_points, max_downward_drop_m
        )
        remaining_path_points = (
            safe_shortest_path_points(env, candidate_pos, candidate_best_goal)
            if candidate_best_goal is not None
            else []
        )
        remaining_downward_drop, remaining_endpoint_height_delta, _remaining_min_height = (
            path_downward_metrics(
                remaining_path_points, start=candidate_pos, end=candidate_best_goal
            )
            if candidate_best_goal is not None
            else (0.0, 0.0, np.nan)
        )
        remaining_stairwell_drop, remaining_stairwell_path = path_stairwell_metrics(
            env, remaining_path_points, max_downward_drop_m
        )
        downstairs_path = candidate_downward_drop > max_downward_drop_m
        remaining_downstairs_path = remaining_downward_drop > max_downward_drop_m
        response_scores.extend(
            [
                float(current_to_candidate if np.isfinite(current_to_candidate) else 1e9),
                float(candidate_to_target if np.isfinite(candidate_to_target) else 1e9),
                float(total if np.isfinite(total) else 1e9),
                1.0 if available else 0.0,
                float(endpoint_to_path if np.isfinite(endpoint_to_path) else 1e9),
                float(current_gt_path_s if np.isfinite(current_gt_path_s) else 0.0),
                float(candidate_gt_path_s if np.isfinite(candidate_gt_path_s) else 0.0),
                float(gt_path_progress if np.isfinite(gt_path_progress) else 0.0),
                float(progress_per_cost if np.isfinite(progress_per_cost) else 0.0),
                1.0 if gt_path_available else 0.0,
                float(candidate_downward_drop),
                float(candidate_endpoint_height_delta),
                1.0 if downstairs_path else 0.0,
                float(remaining_downward_drop),
                float(remaining_endpoint_height_delta),
                1.0 if remaining_downstairs_path else 0.0,
                float(candidate_stairwell_drop),
                1.0 if candidate_stairwell_path else 0.0,
                float(remaining_stairwell_drop),
                1.0 if remaining_stairwell_path else 0.0,
            ]
        )

    rospy.set_param("/apexnav/frontier_oracle/response_stride", 20)
    rospy.set_param(
        "/apexnav/frontier_oracle/response_gt_path_available",
        bool(learning_path_available),
    )
    rospy.set_param("/apexnav/frontier_oracle/response_current_gt_path_s", float(current_gt_path_s))
    rospy.set_param("/apexnav/frontier_oracle/response_gt_path_length", float(gt_path_length))
    rospy.set_param(
        "/apexnav/frontier_oracle/response_target_path_downward_drop",
        float(gt_path_downward_drop),
    )
    rospy.set_param(
        "/apexnav/frontier_oracle/response_target_path_endpoint_height_delta",
        float(gt_path_endpoint_height_delta),
    )
    rospy.set_param(
        "/apexnav/frontier_oracle/response_downstairs_required",
        bool(target_shortest_path_downstairs),
    )
    rospy.set_param("/apexnav/frontier_oracle/response_lookahead_xy", lookahead_xy)
    rospy.set_param(
        "/apexnav/frontier_oracle/response_scan_direction_source",
        (
            "gt_path_lookahead"
            if learning_path_available
            else (
                "target_shortest_path_downstairs"
                if target_shortest_path_downstairs
                else "unavailable"
            )
        ),
    )
    rospy.set_param("/apexnav/frontier_oracle/response_scan_direction", "LOOK_LEFT_60")
    rospy.set_param("/apexnav/frontier_oracle/response_lookahead_bearing_deg", 0.0)
    rospy.set_param("/apexnav/frontier_oracle/response_scores", response_scores)
    rospy.set_param("/apexnav/frontier_oracle/response_id", request_id)
    _last_frontier_oracle_request_id = request_id
    print(
        "FrontierOracle: scored "
        f"{len(candidate_ids)} candidate(s) for request {request_id}, "
        f"gt_path_available={learning_path_available}, gt_path_length={gt_path_length:.3f}, "
        f"target_path_downward_drop={gt_path_downward_drop:.3f}, "
        f"target_shortest_path_downstairs={target_shortest_path_downstairs}"
    )


def signal_handler(sig, frame):
    """Handle Ctrl+C signal for graceful shutdown"""
    print("Ctrl+C detected! Shutting down...")
    rospy.signal_shutdown("Manual shutdown")
    os._exit(0)


def transform_rgb_bgr(image):
    """Convert RGB image to BGR format"""
    return image[:, :, [2, 1, 0]]


def publish_observations(event):
    """Timer callback to publish habitat observations and trigger messages"""
    publish_current_ros_inputs(send_trigger=True)


def publish_current_ros_inputs(send_trigger=True):
    """Republish the latest Habitat observation bundle for ROS synchronization."""
    global msg_observations, fusion_threshold
    global ros_pub, trigger_pub, confidence_threshold_pub
    if msg_observations is None:
        return
    tmp = deepcopy(msg_observations)
    ros_pub.habitat_publish_ros_topic(tmp)
    publish_float64(confidence_threshold_pub, fusion_threshold)
    if send_trigger:
        trigger = PoseStamped()
        trigger_pub.publish(trigger)


def build_target_confirmation_inputs(
    cfg,
    observations,
    label,
    detector_cfg,
    llm_answer,
    skip_detector_servers,
    camera_pitch,
):
    """Run target detector once and build ROS inputs for target confirmation."""
    raw_rgb = observations["rgb"].copy()

    if skip_detector_servers:
        score_list, object_masks_list, label_list = [], [], []
    else:
        try:
            observations["rgb"], score_list, object_masks_list, label_list = get_object(
                label, raw_rgb.copy(), detector_cfg, llm_answer
            )
        except Exception as exc:
            print(f"Detector service unavailable, continuing with empty detections: {exc}")
            score_list, object_masks_list, label_list = [], [], []

    observations["camera_pitch"] = camera_pitch
    msg = deepcopy(observations)
    msg["rgb_raw"] = raw_rgb
    del observations["camera_pitch"]

    obj_point_cloud_list = get_object_point_cloud(cfg, observations, object_masks_list)

    cld_msg = MultipleMasksWithConfidence()
    cld_msg.point_clouds = obj_point_cloud_list
    cld_msg.confidence_scores = score_list
    cld_msg.label_indices = label_list
    return raw_rgb, msg, cld_msg


def ros_action_callback(msg):
    global global_action
    global_action = msg.data


def ros_state_callback(msg):
    global ros_state
    ros_state = msg.data


def ros_final_state_callback(msg):
    global final_state
    final_state = msg.data


def ros_expl_result_callback(msg):
    global expl_result
    expl_result = msg.data



def _resolve_data_path(relative_path):
    data_root = os.environ.get("APEXNAV_DATA_DIR", "").strip()
    if not data_root:
        return os.path.join("data", relative_path)
    return os.path.join(data_root, relative_path)


def _resolve_prefixed_data_path(path_value):
    data_root = os.environ.get("APEXNAV_DATA_DIR", "").strip()
    if not data_root or not isinstance(path_value, str):
        return path_value
    if path_value == "data":
        return data_root
    if path_value.startswith("data/"):
        return os.path.join(data_root, path_value[len("data/") :])
    return path_value


def maybe_register_scene_memory_episode(env, target_label: str, episode_index: int) -> None:
    """Persist Habitat episode->scene mapping for scene-specific memory."""
    raw = os.environ.get("APEXNAV_SCENE_MEMORY_ENABLED", "")
    if raw.strip().lower() not in ("1", "true", "yes", "on"):
        return
    try:
        import scene_memory_bridge

        scene_id = str(getattr(env.current_episode, "scene_id", "") or "")
        episode_id = str(getattr(env.current_episode, "episode_id", episode_index))
        aliases = [
            episode_index,
            f"ep{episode_index}",
            episode_id,
            f"ep{episode_id}",
            os.environ.get("APEXNAV_CURRENT_EPISODE_ID", ""),
            os.environ.get("APEXNAV_EPISODE_ID", ""),
        ]
        scene_memory_bridge.record_episode_scene(scene_id, episode_id, target_label, aliases=aliases)
        os.environ["APEXNAV_CURRENT_SCENE_ID"] = scene_memory_bridge.current_scene_id(episode_id)
        os.environ["APEXNAV_CURRENT_EPISODE_ID"] = episode_id
        print(
            "[SceneMemory] registered episode scene: "
            f"episode_id={episode_id} episode_index={episode_index} "
            f"scene={os.environ['APEXNAV_CURRENT_SCENE_ID']} target={target_label}",
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[SceneMemory] episode scene registration skipped: {exc}", flush=True)


def _apply_env_data_root(cfg: DictConfig) -> None:
    if not os.environ.get("APEXNAV_DATA_DIR", "").strip():
        return
    with habitat.config.read_write(cfg):
        data_path = OmegaConf.select(cfg, "habitat.dataset.data_path")
        if data_path:
            OmegaConf.update(
                cfg,
                "habitat.dataset.data_path",
                _resolve_prefixed_data_path(data_path),
                merge=False,
            )
        for key in ["habitat.simulator.scene_dataset", "habitat.dataset.scenes_dir"]:
            value = OmegaConf.select(cfg, key)
            if value:
                OmegaConf.update(cfg, key, _resolve_prefixed_data_path(value), merge=False)


def _parse_dataset_arg():
    """Parse CLI to choose dataset and capture remaining Hydra overrides."""
    parser = argparse.ArgumentParser(
        description="Habitat ObjectNav Evaluation", add_help=True
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["hm3dv1", "hm3dv2", "mp3d"],
        default="hm3dv2",
        help="Choose dataset: hm3dv1, hm3dv2 or mp3d (default: hm3dv2)",
    )
    # Keep unknown so users can still pass Hydra-style overrides (e.g., key=value)
    args, unknown = parser.parse_known_args()
    return args.dataset, unknown


def main(cfg: DictConfig) -> None:
    global msg_observations, global_action, ros_state, fusion_threshold
    global ros_pub, trigger_pub, obj_point_cloud_pub, confidence_threshold_pub
    global final_state, expl_result

    _apply_env_data_root(cfg)

    # Load MP3D validation data for object category mapping
    mp3d_category_path = _resolve_data_path("datasets/objectnav/mp3d/v1/val/val.json.gz")
    with gzip.open(mp3d_category_path, "rt", encoding="utf-8") as f:
        val_data = json.load(f)
    category_to_coco = val_data.get("category_to_mp3d_category_id", {})
    id_to_name = {
        category_to_coco[cat]: MP3D_ID_TO_NAME[idx]
        for idx, cat in enumerate(category_to_coco)
    }

    start_time = time.time()

    final_state = 0
    expl_result = 0
    result_list = [0] * len(RESULT_TYPES)

    cfg = patch_config(cfg)

    # Extract configuration parameters
    video_output_path = cfg.video_output_path.format(split=cfg.habitat.dataset.split)
    need_video = cfg.need_video
    record_file_path = os.path.join(video_output_path, cfg.record_file_name)
    continue_path = os.path.join(video_output_path, cfg.continue_file_name)
    max_episode_steps = cfg.habitat.environment.max_episode_steps
    max_episode_steps_override = os.environ.get("APEXNAV_MAX_EPISODE_STEPS", "").strip()
    if max_episode_steps_override:
        try:
            override_value = int(max_episode_steps_override)
            if override_value > 0:
                if override_value < max_episode_steps:
                    print(
                        "APEXNAV_MAX_EPISODE_STEPS override: "
                        f"{max_episode_steps} -> {override_value}"
                    )
                max_episode_steps = min(max_episode_steps, override_value)
        except ValueError:
            print(
                "Ignoring invalid APEXNAV_MAX_EPISODE_STEPS="
                f"{max_episode_steps_override!r}"
            )
    success_distance = cfg.habitat.task.measurements.success.success_distance

    detector_cfg = cfg.detector
    skip_all_local_vlm_servers = env_flag("APEXNAV_SKIP_LOCAL_VLM_SERVERS", False)
    skip_blip2_itm_server = (
        skip_all_local_vlm_servers
        or env_flag("APEXNAV_SKIP_BLIP2", True)
        or env_flag("APEXNAV_SKIP_ITM_SERVER", False)
    )
    skip_detector_servers = (
        skip_all_local_vlm_servers
        or env_flag("APEXNAV_SKIP_DETECTOR_SERVERS", False)
    )
    print(
        "Local VLM service flags: "
        f"skip_all={skip_all_local_vlm_servers}, "
        f"skip_blip2_itm={skip_blip2_itm_server}, "
        f"skip_detectors={skip_detector_servers}"
    )
    direct_gt_oracle = env_flag("APEXNAV_DIRECT_GT_ORACLE", False)
    if direct_gt_oracle:
        print("Direct GT oracle policy enabled: Habitat shortest-path follower controls actions.")

    llm_cfg = cfg.llm
    llm_client = llm_cfg.llm_client
    llm_answer_path = llm_cfg.llm_answer_path
    llm_response_path = llm_cfg.llm_response_path

    # Single test parameters
    env_num_once = cfg.test_epi_num  # Which episode to test for single run
    flag_once = env_num_once != -1  # Whether to run single test

    # Create directories if they don't exist
    os.makedirs(os.path.dirname(llm_answer_path), exist_ok=True)
    os.makedirs(video_output_path, exist_ok=True)

    # Add top_down_map and collisions visualization
    with habitat.config.read_write(cfg):
        cfg.habitat.task.measurements.update(
            {
                "top_down_map": TopDownMapMeasurementConfig(
                    map_padding=3,
                    map_resolution=256,
                    draw_source=True,
                    draw_border=True,
                    draw_shortest_path=True,
                    draw_view_points=True,
                    draw_goal_positions=True,
                    draw_goal_aabbs=False,
                    fog_of_war=FogOfWarConfig(
                        draw=True,
                        visibility_dist=5.0,
                        fov=79,
                    ),
                ),
                "collisions": CollisionsMeasurementConfig(),
            }
        )

    env = habitat.Env(cfg)
    print("Environment creation successful")
    number_of_episodes = env.number_of_episodes

    # Read previous records and set initial values
    (
        num_total,
        num_success,
        spl_all,
        soft_spl_all,
        distance_to_goal_all,
        distance_to_goal_reward_all,
        last_time,
    ) = read_record(continue_path, flag_once)

    if num_total >= number_of_episodes:
        raise ValueError("Already finished all episodes.")

    pbar = tqdm.tqdm(total=env.number_of_episodes)

    env_count = num_total if not flag_once else env_num_once
    if flag_once and env_flag("APEXNAV_SORT_EPISODE_ORDER_BY_SCENE_BASENAME", False):
        env.current_episode = stable_sorted_episode_at(env, env_num_once)
        print(
            "Stable episode-order override enabled: "
            f"test_epi_num={env_num_once}, "
            f"scene={getattr(env.current_episode, 'scene_id', '')}, "
            f"episode_id={getattr(env.current_episode, 'episode_id', '')}, "
            f"object_category={getattr(env.current_episode, 'object_category', '')}"
        )
        env_count = 0
    while env_count:
        pbar.update()
        env.current_episode = next(env.episode_iterator)
        env_count -= 1

    # Initialize ROS publishers, subscribers, and timers
    obj_point_cloud_pub = rospy.Publisher(
        "habitat/object_point_cloud", PointCloud2, queue_size=10
    )
    ros_pub = habitat_publisher.ROSPublisher()
    rospy.Subscriber("/habitat/plan_action", Int32, ros_action_callback, queue_size=10)
    rospy.Subscriber("/ros/state", Int32, ros_state_callback, queue_size=10)
    rospy.Subscriber("/ros/expl_state", Int32, ros_final_state_callback, queue_size=10)
    rospy.Subscriber("/ros/expl_result", Int32, ros_expl_result_callback, queue_size=10)
    state_pub = rospy.Publisher("/habitat/state", Int32, queue_size=10)
    trigger_pub = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=10)
    itm_score_pub = rospy.Publisher("/blip2/cosine_score", Float64, queue_size=10)
    confidence_threshold_pub = rospy.Publisher(
        "/detector/confidence_threshold", Float64, queue_size=10
    )
    target_label_pub = rospy.Publisher(
        "/detector/label", String, queue_size=1, latch=True
    )
    cld_with_score_pub = rospy.Publisher(
        "/detector/clouds_with_scores", MultipleMasksWithConfidence, queue_size=10
    )
    progress_pub = rospy.Publisher("/habitat/progress", Int32MultiArray, queue_size=10)
    record_pub = rospy.Publisher("/habitat/record", Float32MultiArray, queue_size=10)
    critic_gt_pub = rospy.Publisher("/habitat/critic_gt", Float64MultiArray, queue_size=10)

    for epi in range(number_of_episodes - num_total):
        # Publish progress information
        publish_int32_array(progress_pub, [num_total, number_of_episodes])

        if flag_once:
            while env_count:
                env.current_episode = next(env.episode_iterator)
                env_count -= 1

        # Initialize episode variables
        pass_object = 0.0
        near_object = 0.0
        global_action = None
        cld_with_score_msg = MultipleMasksWithConfidence()
        count_steps = 0

        camera_pitch = 0.0
        observations = env.reset()
        label = env.current_episode.object_category

        # Convert object category to coco name format
        if label in category_to_coco:
            coco_id = category_to_coco[label]
            label = id_to_name.get(coco_id, label)

        maybe_register_scene_memory_episode(env, label, int(num_total))
        target_label_pub.publish(String(data=label))

        # Get LLM answer and fusion threshold for the target object
        llm_answer, room, fusion_threshold = read_answer(
            llm_answer_path, llm_response_path, label, llm_client
        )

        raw_rgb, msg_observations, cld_with_score_msg = build_target_confirmation_inputs(
            cfg,
            observations,
            label,
            detector_cfg,
            llm_answer,
            skip_detector_servers,
            camera_pitch,
        )
        for _ in range(2):
            ros_pub.habitat_publish_ros_topic(deepcopy(msg_observations))
            publish_float64(confidence_threshold_pub, fusion_threshold)
            rospy.sleep(0.05)
            cld_with_score_pub.publish(cld_with_score_msg)
            rospy.sleep(0.05)

        # Initialize video frame collection
        vis_frames = []
        info = env.get_metrics()
        critic_gt_path_points, critic_target_ros_xy, critic_have_target_position = (
            build_critic_gt_reference(env)
        )
        direct_gt_follower = None
        direct_gt_goal = None
        if direct_gt_oracle:
            agent_pos = safe_agent_world_position(env)
            goal_positions = episode_goal_navigation_positions(env)
            if agent_pos is not None and goal_positions:
                direct_gt_goal, direct_gt_distance = select_best_goal_by_geodesic(
                    env, agent_pos, goal_positions
                )
                if direct_gt_goal is not None and ShortestPathFollower is not None:
                    direct_gt_follower = ShortestPathFollower(
                        env.sim, success_distance, False
                    )
                    print(
                        "DirectGTOracle: selected nearest goal with initial "
                        f"geodesic={direct_gt_distance:.3f}"
                    )
            if direct_gt_follower is None:
                print(
                    "DirectGTOracle: shortest-path follower unavailable; "
                    "falling back to ROS planner actions."
                )
        publish_critic_gt(
            critic_gt_pub,
            env,
            num_total,
            count_steps,
            info.get("distance_to_goal", np.nan),
            critic_gt_path_points,
            critic_target_ros_xy,
            critic_have_target_position,
        )
        if need_video:
            frame = observations_to_image(observations, info)
            info.pop("top_down_map")
            frame = overlay_frame(frame, info)
            vis_frames = [frame]

        # Start publishing basic information and trigger messages
        pub_timer = rospy.Timer(rospy.Duration(0.25), publish_observations)

        print("Agent is waiting in the environment!!!")

        # Wait for ROS system to be ready
        rate = rospy.Rate(10)
        ros_state = ROS_STATE.INIT
        while ros_state == ROS_STATE.INIT or ros_state == ROS_STATE.WAIT_TRIGGER:
            maybe_handle_frontier_oracle_request(env)
            if ros_state == ROS_STATE.INIT:
                print("Waiting for ROS to get odometry...")
            elif ros_state == ROS_STATE.WAIT_TRIGGER:
                print("Waiting for ROS trigger...")
            rate.sleep()

        # Stop timer publishing when starting action execution
        pub_timer.shutdown()

        print("Agent is ready to go!!!!")

        rate = rospy.Rate(10)
        wait_action_republish_period = float(
            os.environ.get("APEXNAV_WAIT_ACTION_REPUBLISH_PERIOD", "0.25")
        )
        wait_action_diagnostic_seconds = float(
            os.environ.get("APEXNAV_WAIT_ACTION_DIAGNOSTIC_SECONDS", "5.0")
        )
        wait_action_start_time = None
        last_wait_action_republish_time = 0.0
        last_wait_action_diagnostic_time = 0.0
        while not rospy.is_shutdown() and not env.episode_over:
            # Skip episode if target is not on the same floor
            is_feasible = 0
            for goal in env.current_episode.goals:
                height = goal.position[1]
                is_feasible += is_on_same_floor(
                    height=height, episode=env.current_episode
                )
            if not is_feasible:
                break

            # Parse action from decision system
            action = None
            if direct_gt_oracle and direct_gt_follower is not None and direct_gt_goal is not None:
                metrics = env.get_metrics()
                if count_steps == max_episode_steps - 1:
                    action = HabitatSimActions.stop
                elif metrics.get("distance_to_goal", np.inf) <= success_distance:
                    action = HabitatSimActions.stop
                else:
                    action = direct_gt_follower.get_next_action(direct_gt_goal)
                    if action is None:
                        action = HabitatSimActions.stop
                global_action = None
            elif global_action is not None:
                if count_steps == max_episode_steps - 1:
                    global_action = ACTION.STOP

                if global_action == ACTION.MOVE_FORWARD:
                    action = HabitatSimActions.move_forward
                elif global_action == ACTION.TURN_LEFT:
                    action = HabitatSimActions.turn_left
                elif global_action == ACTION.TURN_RIGHT:
                    action = HabitatSimActions.turn_right
                elif global_action == ACTION.TURN_DOWN:
                    action = HabitatSimActions.look_down
                    camera_pitch = camera_pitch - np.pi / 6.0
                elif global_action == ACTION.TURN_UP:
                    action = HabitatSimActions.look_up
                    camera_pitch = camera_pitch + np.pi / 6.0
                elif global_action == ACTION.STOP:
                    action = HabitatSimActions.stop

                global_action = None

            if action is None:
                maybe_handle_frontier_oracle_request(env)
                now = time.monotonic()
                if wait_action_start_time is None:
                    wait_action_start_time = now
                if now - last_wait_action_republish_time >= wait_action_republish_period:
                    publish_current_ros_inputs(send_trigger=True)
                    last_wait_action_republish_time = now
                if now - last_wait_action_diagnostic_time >= wait_action_diagnostic_seconds:
                    print(
                        "Waiting for ROS plan action for "
                        f"{now - wait_action_start_time:.1f}s; "
                        f"ros_state={ros_state}; republishing current observations."
                    )
                    last_wait_action_diagnostic_time = now
                rate.sleep()
                continue

            wait_action_start_time = None
            count_steps += 1
            print(f"\n--------------Step: {count_steps}--------------")
            print(f"Finding [{label}]; Action: {action};")

            # Notify ROS system that action execution is starting
            publish_int32(state_pub, HABITAT_STATE.ACTION_EXEC)

            observations = env.step(action)

            # Calculate ITM cosine similarity score
            if skip_blip2_itm_server:
                cosine = 0.0
            else:
                try:
                    cosine = get_itm_message_cosine(observations["rgb"], label, room)
                except Exception as exc:
                    print(f"ITM service unavailable, using 0.0 similarity: {exc}")
                    cosine = 0.0
            print(f"Target related room: {room}")
            print(f"ITM cosine similarity: {cosine:.3f}")

            publish_float64(itm_score_pub, cosine)

            # Publish detector-free RGB for VLM and detector point clouds for target confirmation.
            raw_rgb, msg_observations, cld_with_score_msg = build_target_confirmation_inputs(
                cfg,
                observations,
                label,
                detector_cfg,
                llm_answer,
                skip_detector_servers,
                camera_pitch,
            )
            ros_pub.habitat_publish_ros_topic(msg_observations)
            cld_with_score_pub.publish(cld_with_score_msg)

            # Generate video frame
            info = env.get_metrics()
            if need_video:
                frame = observations_to_image(observations, info)
                info.pop("top_down_map")
                frame = overlay_frame(frame, info)
                vis_frames.append(frame)

            # Track if agent has passed close to the target
            distance_to_goal = info["distance_to_goal"]
            publish_critic_gt(
                critic_gt_pub,
                env,
                num_total,
                count_steps,
                distance_to_goal,
                critic_gt_path_points,
                critic_target_ros_xy,
                critic_have_target_position,
            )
            if distance_to_goal <= success_distance and pass_object == 0:
                pass_object = 1

            # Notify ROS system that action execution is complete
            publish_int32(state_pub, HABITAT_STATE.ACTION_FINISH)
            rate.sleep()

        # Notify ROS system that current episode evaluation is complete
        final_info_for_critic = env.get_metrics()
        publish_critic_gt(
            critic_gt_pub,
            env,
            num_total,
            count_steps,
            final_info_for_critic.get("distance_to_goal", np.nan),
            critic_gt_path_points,
            critic_target_ros_xy,
            critic_have_target_position,
        )
        publish_int32(state_pub, HABITAT_STATE.EPISODE_FINISH)

        # Collect evaluation metrics
        info = env.get_metrics()
        spl = info["spl"]
        soft_spl = info["soft_spl"]
        distance_to_goal = info["distance_to_goal"]
        distance_to_goal_reward = info["distance_to_goal_reward"]
        success = info["success"]

        # Check if agent got close to the target object
        if distance_to_goal <= success_distance:
            near_object = 1

        # Determine episode result
        if success == 1:
            num_success += 1
            result_text = "success"
        else:
            result_text = check_failure(
                env.current_episode,
                final_state,
                expl_result,
                count_steps,
                max_episode_steps,
                pass_object,
                near_object,
            )

        # Update cumulative statistics
        num_total += 1
        spl_all += spl
        soft_spl_all += soft_spl
        distance_to_goal_all += distance_to_goal
        distance_to_goal_reward_all += distance_to_goal_reward

        # Generate video file
        scene_id = env.current_episode.scene_id
        episode_id = env.current_episode.episode_id
        video_name = f"{os.path.basename(scene_id)}_{episode_id}"
        time_spend = time.time() - start_time + last_time

        img2video_output_path = os.path.join(video_output_path, result_text)

        if flag_once:
            img2video_output_path = "videos"
            video_name = "video_once"

        if need_video:
            images_to_video(
                vis_frames, img2video_output_path, video_name, fps=6, quality=9
            )
        vis_frames.clear()

        # Display average performance metrics
        table1 = PrettyTable(["Metric", "Average"])
        table1.add_row(["Average Success", f"{num_success/num_total * 100:.2f}%"])
        table1.add_row(["Average SPL", f"{spl_all/num_total * 100:.2f}%"])
        table1.add_row(["Average Soft SPL", f"{soft_spl_all/num_total * 100:.2f}%"])
        table1.add_row(
            ["Average Distance to Goal", f"{distance_to_goal_all/num_total:.4f}"]
        )
        print(table1)
        print(f"Episode {num_total} data written to {record_file_path}")
        print(f"Result: {result_text}")

        # Display total performance metrics
        table2 = PrettyTable(["Metric", "Total"])
        table2.add_row(["Total Success", f"{num_success}"])
        table2.add_row(["Total SPL", f"{spl_all:.2f}"])
        table2.add_row(["Total Soft SPL", f"{soft_spl_all:.2f}"])
        table2.add_row(["Total Distance to Goal", f"{distance_to_goal_all:.4f}"])

        if flag_once:
            break

        # Write results to record file
        write_record(
            scene_id,
            episode_id,
            table1,
            result_text,
            label,
            num_total,
            time_spend,
            record_file_path,
        )

        # Write results to continue file
        write_record(
            scene_id,
            episode_id,
            table2,
            result_text,
            label,
            num_total,
            time_spend,
            continue_path,
        )

        # Count files in each result category folder
        for i in range(len(RESULT_TYPES)):
            folder = RESULT_TYPES[i]  # Get current category (folder name)
            folder_path = os.path.join(video_output_path, folder)  # Build folder path
            file_count = count_files_in_directory(folder_path)  # Count files in folder
            result_list[i] = file_count

        # Publish comprehensive record data
        record_data = [
            num_success / num_total * 100,
            spl_all / num_total * 100,
            soft_spl_all / num_total * 100,
            distance_to_goal_all / num_total,
        ]
        record_data.extend(result_list)
        publish_float32_array(record_pub, record_data)

        pbar.update()
        env.current_episode = next(env.episode_iterator)
        rospy.sleep(0.1)  # wait a moment

    env.close()
    pbar.close()


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    rospy.init_node("habitat_eval_node", anonymous=True)

    try:
        dataset, overrides = _parse_dataset_arg()
        cfg_name = f"habitat_eval_{dataset}"
        # Compose the chosen config and pass through extra Hydra overrides
        with initialize(version_base=None, config_path="config"):
            cfg = compose(config_name=cfg_name, overrides=overrides)
        main(cfg)
    except Exception as e:
        print(f"Unexpected error occurred: {e}")
        rospy.signal_shutdown("Shutdown due to error")
        os._exit(1)
