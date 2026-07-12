#ifndef _EXPLORATION_MANAGER_H_
#define _EXPLORATION_MANAGER_H_

// Third-party libraries
#include <Eigen/Eigen>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <opencv2/core.hpp>
#include <sensor_msgs/Image.h>
#include <std_msgs/Float64MultiArray.h>
#include <std_msgs/Int32.h>
#include <std_msgs/Int32MultiArray.h>
#include <std_msgs/String.h>

// Standard C++ libraries
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

// ROS core
#include <ros/ros.h>

// Plan environment
#include <plan_env/frontier_map2d.h>
#include <plan_env/object_map2d.h>
#include <plan_env/sdf_map2d.h>
#include <plan_env/value_map2d.h>

// Path searching
#include <path_searching/astar2d.h>

using Eigen::Vector2d;
using Eigen::Vector3d;
using std::shared_ptr;
using std::string;
using std::unique_ptr;
using std::vector;

namespace apexnav_planner {
class SDFMap2D;
class FrontierMap2D;
class Gcopter;
class KinoAstar;
struct ExplorationParam;
struct ExplorationData;

struct SemanticFrontier {
  Vector2d position;      ///< 2D position of the frontier
  double semantic_value;  ///< Semantic value at the frontier location
  double path_length;     ///< Path length to reach this frontier
  vector<Vector2d> path;  ///< Complete path to the frontier

  bool operator<(const SemanticFrontier& other) const
  {
    if (fabs(semantic_value - other.semantic_value) < 1e-4) {
      // If semantic values are equal, sort by path length (ascending)
      return path_length < other.path_length;
    }
    // Otherwise, sort by semantic value (descending)
    return semantic_value > other.semantic_value;
  }
};

struct VLMWaypointCandidate {
  string id;                 ///< Candidate id exposed to the VLM, e.g. F1, LF, F, RF
  string label;              ///< Human-readable label; local-view candidates use LF/F/RF
  string source;             ///< Candidate source/type for debug logs
  bool filtered = false;     ///< True only for debug records of removed candidates
  string filtered_reason;    ///< Reason for debug-filtered candidates
  Vector2d map_position;     ///< Original frontier/map coordinate
  Vector2d safe_goal;        ///< Reachable goal coordinate returned by path search
  double theta_deg = 0.0;    ///< Relative yaw angle from robot/camera forward, left positive
  double euclidean_distance; ///< Straight-line distance from robot
  double path_length;        ///< Planned path length from robot
  double path_ratio = 1.0;   ///< path_length / euclidean_distance
  double clearance;          ///< ESDF clearance estimate at safe_goal
  double score = 0.0;        ///< Candidate ranking score for debug analysis
  double free_distance = -1.0; ///< Local-view robust depth/free-distance estimate
  int sampled_ray_count = 0; ///< Number of depth rays sampled for this candidate source/sector
  bool valid = true;         ///< False for debug rejected candidates
  string reject_reason;      ///< Rejection reason for debug records
  string direction;          ///< Relative direction from current robot heading
  bool reachable;            ///< True only if A* found a valid path
  int frontier_size;         ///< Frontier cluster size when available
  bool recently_selected;    ///< True if close to a recently selected VLM goal
  bool has_projection;       ///< True if candidate projects into the current camera image
  int image_u;               ///< Candidate image projection x pixel
  int image_v;               ///< Candidate image projection y pixel
  double projected_depth;    ///< Actual forward camera-frame distance of safe_goal
  string projection_type = "none"; ///< exact, ground_ray_proxy, or none
  bool projection_proxy = false; ///< True if RGB marker is a visible proxy on the same ground ray
  double projection_proxy_depth = -1.0; ///< Forward distance of the visible proxy marker
  vector<Vector2d> path;     ///< Executable path to safe_goal
};

struct LocalReachableCandidate {
  string label;              ///< LF, F, or RF
  string id;                 ///< ID drawn on RGB and exposed to VLM
  double x = 0.0;
  double y = 0.0;
  double theta_deg = 0.0;
  double distance = 0.0;
  double clearance = -1.0;
  double path_length = 0.0;
  double path_ratio = 1.0;
  double score = 0.0;
  string source = "local_view";
  bool valid = false;
  string reject_reason;
};

struct VLMDecisionResult {
  string decision;                ///< SELECT_WAYPOINT, LOOK_LEFT_60, or LOOK_RIGHT_60
  string selected_id;             ///< Candidate ID when decision == SELECT_WAYPOINT
  string front_scene_type;        ///< VLM inferred current scene type
  string reason;                  ///< VLM final explanation
  string fallback_reason;         ///< Error/fallback reason when available
  string raw_response;            ///< Raw JSON payload from the selector
  string candidate_analysis_json; ///< Raw candidate_analysis JSON array for logs/context
  double confidence = 0.0;        ///< VLM confidence
  bool fallback = false;          ///< True when Python selector already fell back
};

struct VLMInitialPanoramaView {
  string view_id;
  int view_index = 0;
  int init_action_count = 0;
  double relative_yaw_deg = 0.0;
  double yaw = 0.0;
  Vector2d robot = Vector2d::Zero();
  string raw_image_path;
  string depth_image_path;
  vector<VLMWaypointCandidate> candidates;
};

struct GTTrainingCriticPendingDecision {
  bool active = false;
  bool finalized = false;
  string request_id;
  int episode_id = -1;
  int step_id = -1;
  string selected_action;
  string selected_id;
  string candidate_type;
  string direction_group;
  string vlm_reason;
  double confidence = 0.0;
  string annotated_rgb_path;
  string copied_annotated_rgb_path;
  Vector2d anchor_position = Vector2d::Zero();
  double anchor_yaw = 0.0;
  bool have_distance_before = false;
  double distance_to_target_before = -1.0;
  string distance_source_before;
  int scan_count = 0;
  double cumulative_scan_angle_deg = 0.0;
  string scan_history_json;
};

struct GTTrainingCriticEpisodeStats {
  int episode_id = -1;
  int total_vlm_decisions = 0;
  int waypoint_decisions = 0;
  int scan_decisions = 0;
  int good_count = 0;
  int bad_count = 0;
  int uncertain_count = 0;
  int scan_unjudged_count = 0;
  double good_target_progress_sum = 0.0;
  double bad_target_progress_sum = 0.0;
  double bad_endpoint_to_gt_path_sum = 0.0;
  int bad_endpoint_to_gt_path_count = 0;
};

struct FrontierOracleCandidateScore {
  string id;
  double current_to_candidate = std::numeric_limits<double>::infinity();
  double candidate_to_target = std::numeric_limits<double>::infinity();
  double total_geodesic = std::numeric_limits<double>::infinity();
  double target_progress = 0.0;
  bool geodesic_available = false;
  double endpoint_to_gt_path = std::numeric_limits<double>::infinity();
  double current_gt_path_s = std::numeric_limits<double>::quiet_NaN();
  double candidate_gt_path_s = std::numeric_limits<double>::quiet_NaN();
  double gt_path_progress = std::numeric_limits<double>::quiet_NaN();
  double progress_per_cost = std::numeric_limits<double>::quiet_NaN();
  bool gt_path_available = false;
  double path_downward_drop = 0.0;
  double endpoint_height_delta = 0.0;
  double remaining_path_downward_drop = 0.0;
  double remaining_endpoint_height_delta = 0.0;
  bool remaining_downstairs_path = false;
  double path_stairwell_drop = 0.0;
  bool stairwell_path = false;
  double remaining_path_stairwell_drop = 0.0;
  bool remaining_stairwell_path = false;
  bool downstairs_path = false;
  string source = "unavailable";
};

enum EXPL_RESULT {
  EXPLORATION,               ///< Normal exploration mode
  SEARCH_BEST_OBJECT,        ///< Found high-confidence object
  SEARCH_OVER_DEPTH_OBJECT,  ///< Searching over-depth object
  SEARCH_SUSPICIOUS_OBJECT,  ///< Investigating suspicious object
  NO_PASSABLE_FRONTIER,      ///< No reachable frontiers available
  NO_COVERABLE_FRONTIER,     ///< No coverable frontiers found
  SEARCH_EXTREME,            ///< Extreme search mode activated
  REACH_VLM_TARGET_OBJECT    ///< Reached a model-selected target proxy with live evidence
};

class ExplorationManager {
public:
  ExplorationManager() = default;
  ~ExplorationManager();  // Explicit destructor declaration for shared_ptr with forward declaration

  void initialize(ros::NodeHandle& nh);

  int planNextBestPoint(const Vector3d& pos, const double& yaw);
  bool planTrajectory(const Eigen::VectorXd& start, const Eigen::VectorXd& end, const Vector3d& ctrl);
  void clearActiveVLMWaypoint(const string& reason);
  bool hasActiveVLMWaypoint() const;
  bool consumePendingVLMForcedAction(int& action_code);
  bool shouldStopOracleRolloutAtTarget(const Vector2d& cur_pos) const;
  void recordInitialVLMPanoramaView(const Vector2d& cur_pos, double cur_yaw,
      const vector<Vector2d>& frontiers, int init_action_count);
  void getSortedSemanticFrontiers(const Vector2d& cur_pos, const vector<Vector2d>& frontiers,
      vector<SemanticFrontier>& sem_frontiers);
  void calcSemanticFrontierInfo(const vector<SemanticFrontier>& sem_frontiers, double& std_dev,
      double& max_to_mean, double& mean, bool if_print = false);

  shared_ptr<ExplorationData> ed_;            ///< Exploration data container
  shared_ptr<ExplorationParam> ep_;           ///< Exploration parameters
  unique_ptr<Astar2D> path_finder_;           ///< A* path finding algorithm
  shared_ptr<FrontierMap2D> frontier_map2d_;  ///< 2D frontier map
  shared_ptr<ObjectMap2D> object_map2d_;      ///< 2D object map
  shared_ptr<SDFMap2D> sdf_map_;              ///< Signed distance field map
  shared_ptr<Gcopter> gcopter_;               ///< Trajectory optimizer (for real-world)
  shared_ptr<KinoAstar> kinoastar_;           ///< Kinodynamic A* planner (for real-world)

  typedef shared_ptr<ExplorationManager> Ptr;

private:
  // Exploration Policy
  void chooseExplorationPolicy(Vector2d cur_pos, double cur_yaw, vector<Vector2d> frontiers,
      Vector2d& next_best_pos, vector<Vector2d>& next_best_path);
  void findClosestFrontierPolicy(Vector2d cur_pos, vector<Vector2d> frontiers,
      Vector2d& next_best_pos, vector<Vector2d>& next_best_path);
  void findHighestSemanticsFrontierPolicy(Vector2d cur_pos, vector<Vector2d> frontiers,
      Vector2d& next_best_pos, vector<Vector2d>& next_best_path);
  void hybridExplorePolicy(Vector2d cur_pos, double cur_yaw, vector<Vector2d> frontiers,
      Vector2d& next_best_pos, vector<Vector2d>& next_best_path);
  void findTSPTourPolicy(Vector2d cur_pos, vector<Vector2d> frontiers, Vector2d& next_best_pos,
      vector<Vector2d>& next_best_path);
  void findVLMGuidedFrontierPolicy(Vector2d cur_pos, double cur_yaw, vector<Vector2d> frontiers,
      Vector2d& next_best_pos, vector<Vector2d>& next_best_path);

  // VLM waypoint utilities
  bool consumeReachedVLMTargetObjectProxyStop();
  void rgbCallback(const sensor_msgs::ImageConstPtr& msg);
  void depthCallback(const sensor_msgs::ImageConstPtr& msg);
  void targetLabelCallback(const std_msgs::StringConstPtr& msg);
  void progressCallback(const std_msgs::Int32MultiArrayConstPtr& msg);
  vector<VLMWaypointCandidate> buildVLMWaypointCandidates(
      const Vector2d& cur_pos, double cur_yaw, const vector<Vector2d>& frontiers);
  vector<VLMWaypointCandidate> buildLocalViewWaypointCandidates(
      const Vector2d& cur_pos, double cur_yaw, int image_width, int image_height);
  vector<VLMWaypointCandidate> buildRecoveryWaypointCandidates(
      const Vector2d& cur_pos, double cur_yaw, int image_width, int image_height,
      int max_candidates);
  bool buildEmergencyEscapeWaypointCandidate(const Vector2d& cur_pos, double cur_yaw,
      int image_width, int image_height, VLMWaypointCandidate& candidate);
  double estimateLocalViewRayFreeDistance(
      double theta_deg, int& sampled_pixel_count, int& image_u) const;
  bool selectWaypointWithVLM(const string& request_id, const Vector2d& cur_pos, double cur_yaw,
      const vector<VLMWaypointCandidate>& candidates, VLMDecisionResult& decision,
      VLMWaypointCandidate& selected);
  bool selectWaypointWithInitialPanoramaVLM(const string& request_id, const Vector2d& cur_pos,
      double cur_yaw, vector<VLMWaypointCandidate>& candidates, VLMDecisionResult& decision,
      VLMWaypointCandidate& selected);
  bool selectWaypointWithScanPanoramaVLM(const string& request_id, const Vector2d& cur_pos,
      double cur_yaw, vector<VLMWaypointCandidate>& candidates, VLMDecisionResult& decision,
      VLMWaypointCandidate& selected);
  bool selectWaypointWithInitialPanoramaOracle(const string& request_id, const Vector2d& cur_pos,
      double cur_yaw, vector<VLMWaypointCandidate>& candidates, VLMDecisionResult& decision,
      VLMWaypointCandidate& selected);
  bool saveLatestRGBImage(const string& image_path, string& error_msg);
  bool saveLatestDepthImage(const string& image_path, string& error_msg);
  bool projectCandidateToImage(const Vector2d& cur_pos, double cur_yaw, const Vector2d& goal,
      int image_width, int image_height, int& u, int& v, double& forward_depth) const;
  bool projectCandidateToImageWithGroundProxy(const Vector2d& cur_pos, double cur_yaw,
      const Vector2d& goal, int image_width, int image_height, int& u, int& v,
      double& forward_depth, bool& used_proxy, double& proxy_forward_depth) const;
  bool projectCandidateToImageWithEdgeProxy(const Vector2d& cur_pos, double cur_yaw,
      const Vector2d& goal, int image_width, int image_height, int& u, int& v,
      double& forward_depth, double& proxy_forward_depth) const;
  string directionLabel(const Vector2d& cur_pos, double cur_yaw, const Vector2d& goal) const;
  int estimateFrontierSize(const Vector2d& frontier) const;
  bool isRecentlySelectedGoal(const Vector2d& goal) const;
  bool isRejectedByVLMScan(const Vector2d& goal) const;
  void rememberSelectedGoal(const Vector2d& goal);
  void rememberVLMScanRejectedCandidates(
      const vector<VLMWaypointCandidate>& candidates, const string& selected_id);
  void setActiveVLMWaypoint(
      const string& selected_id, const Vector2d& goal, const vector<Vector2d>& path,
      bool target_object_proxy = false);
  bool continueActiveVLMWaypoint(const Vector2d& cur_pos);
  string makeVLMRequestId();
  string shellQuote(const string& value) const;
  string jsonEscape(const string& value) const;
  void writeVLMDetectorSemanticMapSnapshot(
      std::ostream& out, const Vector2d& cur_pos, double cur_yaw) const;
  void writeVLMMetricMapSummary(std::ostream& out) const;
  bool parseVLMDecision(const string& result_json, VLMDecisionResult& decision) const;
  string extractJsonArrayField(const string& result_json, const string& key) const;
  void writeActualCandidateReferenceLabel(const string& request_id, const Vector2d& cur_pos,
      double cur_yaw, const vector<VLMWaypointCandidate>& candidates,
      const VLMDecisionResult& online_decision);
  void resetVLMScanContext(const string& reason);
  void resetInitialVLMPanorama(const string& reason);
  void recordVLMScanPanoramaView(const string& request_id, const Vector2d& cur_pos,
      double cur_yaw, const vector<VLMWaypointCandidate>& candidates);
  void updateVLMScanContext(const VLMDecisionResult& decision, int visible_candidate_count);
  void setPendingVLMForcedTurn(const string& decision);
  bool isVLMForcedLookDecision(const string& decision) const;
  void writeVLMSelectionLog(const string& request_id,
      const vector<VLMWaypointCandidate>& candidates, const VLMDecisionResult& decision,
      const Vector2d& final_goal, bool fallback, const string& fallback_reason,
      bool scan_context_reset) const;
  bool isCriticOnlyDebugMode() const;
  void criticGTCallback(const std_msgs::Float64MultiArrayConstPtr& msg);
  void habitatStateCallback(const std_msgs::Int32ConstPtr& msg);
  bool getCriticDistanceToTarget(const Vector2d& pos, double& distance, string& source,
      vector<string>& reason_codes) const;
  void startCriticDecision(const string& request_id, const VLMDecisionResult& decision,
      const VLMWaypointCandidate* selected, const Vector2d& anchor_pos, double anchor_yaw);
  void maybeFinalizePendingCriticDecision(const Vector2d& cur_pos, const string& outcome_reason);
  void finalizePendingCriticDecision(const Vector2d& cur_pos, const string& outcome_reason);
  void writeCriticDecisionRecord(const GTTrainingCriticPendingDecision& pending,
      const Vector2d& final_pos, const string& outcome_reason, const string& verdict,
      const vector<string>& reason_codes, double distance_after, bool have_distance_after,
      const string& distance_source_after, double target_progress, bool have_target_progress,
      double endpoint_to_gt_path, bool have_endpoint_to_gt_path, const string& after_rgb_path);
  void resetCriticEpisodeStats(int episode_id);
  void updateCriticEpisodeStats(const string& action, const string& verdict,
      bool have_target_progress, double target_progress, bool have_endpoint_to_gt_path,
      double endpoint_to_gt_path);
  void writeCriticEpisodeSummary(const string& reason);
  bool isOracleFrontierRolloutMode() const;
  bool isOracleFrontierLabelOnlyMode() const;
  bool selectWaypointWithFrontierOracle(const string& request_id, const Vector2d& cur_pos,
      double cur_yaw, const vector<VLMWaypointCandidate>& candidates,
      VLMDecisionResult& decision, VLMWaypointCandidate& selected);
  vector<FrontierOracleCandidateScore> requestFrontierOracleScores(
      const string& request_id, const Vector2d& cur_pos,
      const vector<VLMWaypointCandidate>& candidates);
  void writeFrontierOracleLogAndImage(const string& request_id, const Vector2d& cur_pos,
      double cur_yaw, const vector<VLMWaypointCandidate>& candidates,
      const vector<FrontierOracleCandidateScore>& scores, const string& selected_id,
      const string& reason);
  void writeFrontierOraclePanoramaLogAndImages(const string& request_id,
      const Vector2d& cur_pos, double cur_yaw, const vector<VLMWaypointCandidate>& candidates,
      const vector<FrontierOracleCandidateScore>& scores, const string& selected_id,
      const string& reason);

  // Path Search Utils
  bool searchObjectPath(const Vector3d& start,
      const pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& object_cloud,
      Eigen::Vector2d& refined_pos, std::vector<Eigen::Vector2d>& refined_path);
  bool searchObjectPathExtreme(const Vector3d& start,
      const pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& object_cloud,
      Eigen::Vector2d& refined_pos, std::vector<Eigen::Vector2d>& refined_path);
  bool searchFrontierPath(const Vector2d& start, const Vector2d& end, Eigen::Vector2d& refined_pos,
      std::vector<Eigen::Vector2d>& refined_path);
  void shortenPath(vector<Vector2d>& path);

  // Helper functions for object path searching
  Vector2d findNearestObjectPoint(
      const Vector3d& start, const pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& object_cloud);
  bool trySearchObjectPathWithDistance(const Vector2d& start2d, const Vector2d& object_pose,
      double distance, double max_search_time, Eigen::Vector2d& refined_pos,
      std::vector<Eigen::Vector2d>& refined_path, const std::string& debug_msg);

  // TSP Optimization Methods
  void computeATSPTour(
      const Vector2d& cur_pos, const vector<Vector2d>& frontiers, vector<int>& indices);
  void computeATSPCostMatrix(
      const Vector2d& cur_pos, const vector<Vector2d>& frontiers, Eigen::MatrixXd& cost_matrix);
  double computePathCost(const Vector2d& pos1, const Vector2d& pos2);
  vector<Vector2i> allNeighbors(const Eigen::Vector2i& idx, int grid_radius);

  ros::ServiceClient tsp_client_;         ///< ROS service client for TSP solver
  unique_ptr<RayCaster2D> ray_caster2d_;  ///< Ray casting for collision checking
  ros::Subscriber rgb_sub_;               ///< Latest Habitat RGB frame for VLM annotation
  ros::Subscriber depth_sub_;             ///< Latest Habitat depth frame for debug overlay
  ros::Subscriber target_label_sub_;      ///< Target object label for VLM prompting
  ros::Subscriber progress_sub_;          ///< Episode progress for debug filenames
  ros::Subscriber critic_gt_sub_;         ///< Private Habitat GT metrics for critic-only debug mode
  ros::Subscriber habitat_state_sub_;     ///< Habitat episode boundary for critic summaries
  cv::Mat latest_rgb_image_;
  cv::Mat latest_depth_viz_image_;
  cv::Mat latest_depth_meters_;
  bool have_latest_rgb_ = false;
  bool have_latest_depth_ = false;
  bool have_latest_depth_meters_ = false;
  string target_label_ = "unknown";
  int episode_index_ = -1;
  int vlm_request_counter_ = 0;
  vector<Vector2d> vlm_recent_goals_;
  vector<Vector2d> vlm_recent_open_view_poses_;
  bool active_vlm_waypoint_ = false;
  Vector2d active_vlm_goal_ = Vector2d::Zero();
  vector<Vector2d> active_vlm_path_;
  string active_vlm_selected_id_;
  int active_vlm_follow_steps_ = 0;
  double active_vlm_best_distance_ = std::numeric_limits<double>::infinity();
  int active_vlm_stall_steps_ = 0;
  bool active_vlm_target_object_proxy_ = false;
  bool vlm_selected_target_object_proxy_in_last_plan_ = false;
  bool vlm_target_object_proxy_ready_to_stop_ = false;
  bool vlm_reached_target_object_proxy_stop_in_last_plan_ = false;
  bool vlm_scan_context_active_ = false;
  int vlm_scan_count_ = 0;
  double vlm_cumulative_scan_angle_deg_ = 0.0;
  string vlm_scan_last_action_;
  string vlm_scan_last_reason_;
  vector<string> vlm_inspected_views_json_;
  vector<Vector2d> vlm_scan_rejected_goals_;
  vector<VLMWaypointCandidate> vlm_last_low_clearance_filtered_candidates_;
  vector<VLMWaypointCandidate> vlm_last_high_path_ratio_filtered_candidates_;
  vector<VLMWaypointCandidate> vlm_last_scan_rejected_filtered_candidates_;
  vector<VLMWaypointCandidate> vlm_last_local_view_rejected_candidates_;
  vector<VLMInitialPanoramaView> vlm_initial_panorama_views_;
  vector<VLMInitialPanoramaView> vlm_scan_panorama_views_;
  bool vlm_initial_panorama_consumed_ = false;
  bool vlm_initial_panorama_have_start_yaw_ = false;
  double vlm_initial_panorama_start_yaw_ = 0.0;
  int vlm_initial_panorama_view_counter_ = 0;
  int vlm_last_candidate_count_before_clearance_filter_ = 0;
  int vlm_last_candidate_count_after_clearance_filter_ = 0;
  int vlm_last_candidate_count_before_path_ratio_filter_ = 0;
  int vlm_last_candidate_count_after_path_ratio_filter_ = 0;
  int vlm_last_candidate_count_before_scan_reject_filter_ = 0;
  int vlm_last_candidate_count_after_scan_reject_filter_ = 0;
  int pending_vlm_forced_action_ = -1;
  int pending_vlm_forced_action_steps_ = 0;
  GTTrainingCriticPendingDecision critic_pending_;
  GTTrainingCriticEpisodeStats critic_stats_;
  bool critic_have_geodesic_distance_ = false;
  double critic_geodesic_distance_ = -1.0;
  bool critic_have_target_position_ = false;
  Vector2d critic_target_position_ = Vector2d::Zero();
  bool critic_have_endpoint_to_gt_path_ = false;
  double critic_endpoint_to_gt_path_ = -1.0;
  string oracle_scan_direction_hint_ = "LOOK_LEFT_60";
  string oracle_scan_direction_source_ = "unavailable";
  bool oracle_have_gt_path_hint_ = false;
  double oracle_lookahead_bearing_deg_ = 0.0;
  double oracle_current_gt_path_s_ = std::numeric_limits<double>::quiet_NaN();
  double oracle_gt_path_length_ = std::numeric_limits<double>::quiet_NaN();
  Vector2d oracle_lookahead_point_ = Vector2d::Zero();
  bool oracle_downstairs_required_ = false;
  bool oracle_frontier_blocked_downstairs_ = false;
  double oracle_target_path_downward_drop_ = 0.0;
  double oracle_target_path_endpoint_height_delta_ = 0.0;
  int critic_gt_episode_id_ = -1;
  int critic_gt_step_id_ = -1;
  Vector2d critic_last_planning_pos_ = Vector2d::Zero();
};

inline bool ExplorationManager::searchFrontierPath(const Vector2d& start, const Vector2d& end,
    Eigen::Vector2d& refined_pos, std::vector<Eigen::Vector2d>& refined_path)
{
  path_finder_->reset();
  if (path_finder_->astarSearch(start, end, 0.25, 2.0, Astar2D::SAFETY_MODE::OPTIMISTIC) ==
      Astar2D::REACH_END) {
    refined_pos = end;
    refined_path = path_finder_->getPath();
    return true;
  }
  return false;
}

inline bool ExplorationManager::searchObjectPathExtreme(const Vector3d& start,
    const pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& object_cloud,
    Eigen::Vector2d& refined_pos, std::vector<Eigen::Vector2d>& refined_path)
{
  Vector2d object_pose = findNearestObjectPoint(start, object_cloud);
  if (object_pose.x() < -999.0)
    return false;  // Error finding nearest point

  Vector2d start2d = Vector2d(start(0), start(1));
  path_finder_->reset();
  if (path_finder_->astarSearch(start2d, object_pose, 0.25, 0.2, Astar2D::SAFETY_MODE::EXTREME) ==
      Astar2D::REACH_END) {
    refined_pos = object_pose;
    refined_path = path_finder_->getPath();
    return true;
  }
  return false;
}

inline void ExplorationManager::shortenPath(vector<Vector2d>& path)
{
  if (path.empty()) {
    ROS_ERROR("Empty path to shorten");
    return;
  }

  // Shorten the path by keeping only critical intermediate points
  const double dist_thresh = 3.0;  // Minimum distance threshold for waypoint retention
  vector<Vector2d> short_tour = { path.front() };

  for (int i = 1; i < (int)path.size() - 1; ++i) {
    if ((path[i] - short_tour.back()).norm() > dist_thresh)
      short_tour.push_back(path[i]);
    else {
      // Add waypoints only when necessary to avoid collision
      ray_caster2d_->input(short_tour.back(), path[i + 1]);
      Eigen::Vector2i idx;
      while (ray_caster2d_->nextId(idx) && ros::ok()) {
        if (sdf_map_->getInflateOccupancy(idx) == 1 ||
            sdf_map_->getOccupancy(idx) == SDFMap2D::UNKNOWN) {
          short_tour.push_back(path[i]);
          break;
        }
      }
    }
  }

  // Always include the final destination
  if ((path.back() - short_tour.back()).norm() > 1e-3)
    short_tour.push_back(path.back());

  // Ensure minimum path complexity (at least three points)
  if (short_tour.size() == 2)
    short_tour.insert(short_tour.begin() + 1, 0.5 * (short_tour[0] + short_tour[1]));

  path = short_tour;
}

inline vector<Eigen::Vector2i> ExplorationManager::allNeighbors(
    const Eigen::Vector2i& idx, int grid_radius)
{
  vector<Eigen::Vector2i> neighbors;

  for (int x = -grid_radius; x <= grid_radius; ++x) {
    for (int y = -grid_radius; y <= grid_radius; ++y) {
      if (x == 0 && y == 0)
        continue;  // Skip center point
      Eigen::Vector2i offset(x, y);
      neighbors.push_back(idx + offset);
    }
  }
  return neighbors;
}

}  // namespace apexnav_planner

#endif
