// Copyright (c) 2020, Samsung Research America
// Copyright (c) 2023, Open Navigation LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License. Reserved.

#include <string>
#include <memory>
#include <vector>
#include <algorithm>
#include <limits>
#include <iostream>

#include "smac_planner_hybrid.hpp"

// #define BENCHMARK_TESTING

#ifndef M_PIf32
#define M_PIf32 3.14159265358979323846f
#endif


using namespace std::chrono;  // NOLINT
using std::placeholders::_1;

SmacPlannerHybrid::SmacPlannerHybrid()
: _a_star(nullptr),
  _collision_checker(nullptr, 1),
  _smoother(nullptr),
  _costmap(nullptr),
  _costmap_master(nullptr),
  _costmap_downsampler(nullptr)
{
 std::cout<<"SmacPlannerHybrid"<<std::endl;
}

SmacPlannerHybrid::~SmacPlannerHybrid()
{
 std::cout << "Destroying plugin SmacPlannerHybrid" << std::endl;
}

void SmacPlannerHybrid::configure(
 const SmootherParams& params_s, const SmacPlannerHybridParams& params, std::shared_ptr<Costmap2D_Master> costmap_master)
{

  _costmap = costmap_master->getCostmap();
  _costmap_master = costmap_master;

  std::cout << "Configuring SmacPlannerHybrid" << std::endl;

  bool smooth_path = false;


  // General planner params

  // Параметры даунсемплинга карты стоимости
     _downsample_costmap = params.downsample_costmap;        // ".downsample_costmap" - включение даунсемплинга карты стоимости
     _downsampling_factor = params.downsampling_factor;      // ".downsampling_factor" - фактор даунсемплинга

     // Параметры квантования углов
     _angle_bin_size = 2.0f * M_PIf32 / params.angle_quantization_bins;
     _angle_quantizations = static_cast<unsigned int>(params.angle_quantization_bins); // ".angle_quantization_bins" - количество бинов для квантования углов

     // Основные параметры планировщика
     _tolerance = params.tolerance;                          // ".tolerance" - допуск достижения цели
     _allow_unknown = params.allow_unknown;                  // ".allow_unknown" - разрешение неизвестных областей
     _max_iterations = params.max_iterations;                // ".max_iterations" - максимальное количество итераций
     _max_on_approach_iterations = params.max_on_approach_iterations; // ".max_on_approach_iterations" - итерации при приближении
     _terminal_checking_interval = params.terminal_checking_interval; // ".terminal_checking_interval" - интервал проверки завершения

     // Параметры поиска и эвристик
     _search_info.allow_primitive_interpolation = params.allow_primitive_interpolation; // ".allow_primitive_interpolation" - интерполяция примитивов
     _search_info.cache_obstacle_heuristic = params.cache_obstacle_heuristic; // ".cache_obstacle_heuristic" - кэширование эвристики препятствий
     _search_info.reverse_penalty = params.reverse_penalty;  // ".reverse_penalty" - штраф за движение назад
     _search_info.change_penalty = params.change_penalty;    // ".change_penalty" - штраф за смену направления
     _search_info.non_straight_penalty = params.non_straight_penalty; // ".non_straight_penalty" - штраф за непрямое движение
     _search_info.cost_penalty = params.cost_penalty;        // ".cost_penalty" - штраф за стоимость
     _search_info.retrospective_penalty = params.retrospective_penalty; // ".retrospective_penalty" - ретроспективный штраф
     _search_info.analytic_expansion_ratio = params.analytic_expansion_ratio; // ".analytic_expansion_ratio" - коэффициент аналитического расширения
     _search_info.analytic_expansion_max_cost = params.analytic_expansion_max_cost; // ".analytic_expansion_max_cost" - максимальная стоимость аналитического расширения
     _search_info.analytic_expansion_max_cost_override = params.analytic_expansion_max_cost_override; // ".analytic_expansion_max_cost_override" - переопределение максимальной стоимости
     _search_info.use_quadratic_cost_penalty = params.use_quadratic_cost_penalty; // ".use_quadratic_cost_penalty" - использование квадратичного штрафа стоимости
     _search_info.downsample_obstacle_heuristic = params.downsample_obstacle_heuristic; // ".downsample_obstacle_heuristic" - даунсемплинг эвристики препятствий

     // Остальные параметры...
     _minimum_turning_radius_global_coords = params.minimum_turning_radius; // ".minimum_turning_radius" - минимальный радиус поворота
     _search_info.analytic_expansion_max_length = params.analytic_expansion_max_length / _costmap->getResolution(); // ".analytic_expansion_max_length" - максимальная длина аналитического расширения
     _max_planning_time = params.max_planning_time;          // ".max_planning_time" - максимальное время планирования
     _lookup_table_size = params.lookup_table_size;          // ".lookup_table_size" - размер таблицы поиска
     _debug_visualizations = params.debug_visualizations;    // ".debug_visualizations" - визуализация отладки
     _motion_model_for_search = params.motion_model_for_search; // ".motion_model_for_search" - модель движения для поиска
     _motion_model = fromString(_motion_model_for_search);
     _goal_heading_mode = fromStringToGH(params.goal_heading_mode); // ".goal_heading_mode" - режим ориентации цели
     _coarse_search_resolution = params.coarse_search_resolution; // ".coarse_search_resolution" - разрешение грубого поиска


  if (_goal_heading_mode == GoalHeadingMode::UNKNOWN) {

    std::string goal_heading_type = toStringFromGH(_goal_heading_mode);
    std::string error_msg = "Unable to get GoalHeader type. Given '" + goal_heading_type + "' "
      "Valid options are DEFAULT, BIDIRECTIONAL, ALL_DIRECTION. ";
    throw PlannerException(error_msg);
  }

  _motion_model = fromString(_motion_model_for_search);

  if (_motion_model == MotionModel::UNKNOWN) {
    std::cout << "WARNING: Unable to get MotionModel search type. Given '"
              << _motion_model_for_search
              << "', valid options are MOORE, VON_NEUMANN, DUBIN, REEDS_SHEPP, STATE_LATTICE."
              << std::endl;
  }

  if (_max_on_approach_iterations <= 0) {
    std::cout << "WARNING: On approach iteration selected as <= 0, "
              << "disabling tolerance and on approach iterations." << std::endl;
    _max_on_approach_iterations = std::numeric_limits<int>::max();
  }

  if (_max_iterations <= 0) {
    std::cout << "WARNING: maximum iteration selected as <= 0, "
              << "disabling maximum iterations." << std::endl;
    _max_iterations = std::numeric_limits<int>::max();
  }

  if (_coarse_search_resolution <= 0) {
    std::cout << "WARNING: coarse iteration resolution selected as <= 0, "
              << "disabling coarse iteration resolution search for goal heading"
              << std::endl;
    _coarse_search_resolution = 1;
  }

  if (_angle_quantizations % _coarse_search_resolution != 0) {
    std::string error_msg = "coarse iteration should be an increment"
      " of the number of angular bins configured";
    throw PlannerException(error_msg);
  }

  if (_minimum_turning_radius_global_coords < _costmap->getResolution() * _downsampling_factor) {

    std::cout << "WARNING: Min turning radius cannot be less than the search grid cell resolution!" << std::endl;

    _minimum_turning_radius_global_coords = _costmap->getResolution() * _downsampling_factor;
  }
  // convert to grid coordinates
  if (!_downsample_costmap) {
    _downsampling_factor = 1;
  }
  _search_info.minimum_turning_radius =
    _minimum_turning_radius_global_coords / (_costmap->getResolution() * _downsampling_factor);
  _lookup_table_dim =
    static_cast<float>(_lookup_table_size) /
    static_cast<float>(_costmap->getResolution() * _downsampling_factor);

  // Make sure its a whole number
  _lookup_table_dim = static_cast<float>(static_cast<int>(_lookup_table_dim));

  // Make sure its an odd number
  if (static_cast<int>(_lookup_table_dim) % 2 == 0) {

    std::cout << "INFO: Even sized heuristic lookup table size set " << _lookup_table_dim
              << ", increasing size by 1 to make odd" << std::endl;

    _lookup_table_dim += 1.0f;
  }

  // Initialize collision checker
  _collision_checker = CollisionChecker(_costmap_master, _angle_quantizations);
  _collision_checker.setFootprint(
    _costmap_master->getRobotFootprint(),
    _costmap_master->getUseRadius(),
    findCircumscribedCost(_costmap_master));

  // Initialize A* template
  _a_star = std::make_unique<AStarAlgorithm<NodeHybrid>>(_motion_model, _search_info);
  _a_star->initialize(
    _allow_unknown,
    _max_iterations,
    _max_on_approach_iterations,
    _terminal_checking_interval,
    _max_planning_time,
    _lookup_table_dim,
    _angle_quantizations);

  // Initialize path smoother
  if (smooth_path) {
    _smoother = std::make_unique<Smoother>(params_s);
    _smoother->initialize(_minimum_turning_radius_global_coords);
  }

  // Initialize costmap downsampler
  _costmap_downsampler = std::make_unique<CostmapDownsampler>();
  _costmap_downsampler->on_configure( _costmap, _downsampling_factor);



  if (_debug_visualizations) {
  }

  std::cout << "Configured plugin SmacPlannerHybrid with maximum iterations " << _max_iterations
            << ", max on approach iterations " << _max_on_approach_iterations
            << ", and " << (_allow_unknown ? "allowing unknown traversal" : "not allowing unknown traversal")
            << ". Tolerance " << std::fixed << std::setprecision(2) << _tolerance
            << ". Using motion model: " << toString(_motion_model) << "." << std::endl;
}




Path SmacPlannerHybrid::createPlan(
  const Pose & start,
  const Pose & goal,
  std::function<bool()> cancel_checker)
{
  std::lock_guard<std::mutex> lock_reinit(_mutex);
  steady_clock::time_point a = steady_clock::now();

  std::unique_lock<Costmap2D::mutex_t> lock(*(_costmap->getMutex()));

  // Downsample costmap, if required
  Costmap2D * costmap = _costmap;
  if (_downsample_costmap && _downsampling_factor > 1) {
    costmap = _costmap_downsampler->downsample(_downsampling_factor);
    _collision_checker.setCostmap(costmap);
  }

  // Set collision checker and costmap information
  _collision_checker.setFootprint(
    _costmap_master->getRobotFootprint(),
    _costmap_master->getUseRadius(),
    findCircumscribedCost(_costmap_master));
  _a_star->setCollisionChecker(&_collision_checker);

  // Set starting point, in A* bin search coordinates
  //float mx_start, my_start, mx_goal, my_goal;
  if (!costmap->PosCheck(
    start.position.x(),
    start.position.y()
    ))
  {
    throw StartOutsideMapBounds(
            "Start Coordinates of(" + std::to_string(start.position.x()) + ", " +
            std::to_string(start.position.y()) + ") was outside bounds");
  }

  float start_orientation_bin = std::round(start.orientation / _angle_bin_size);
  while (start_orientation_bin < 0.0f) {
    start_orientation_bin += static_cast<float>(_angle_quantizations);
  }
  // This is needed to handle precision issues
  if (start_orientation_bin >= static_cast<float>(_angle_quantizations)) {
    start_orientation_bin -= static_cast<float>(_angle_quantizations);
  }
  unsigned int start_orientation_bin_int =
    static_cast<unsigned int>(start_orientation_bin);
  _a_star->setStart(start.position.x(),  start.position.y(), start_orientation_bin_int);

  // Set goal point, in A* bin search coordinates
  if (!costmap->PosCheck(
    goal.position.x(),
    goal.position.y()
    ))
  {
    throw GoalOutsideMapBounds(
            "Goal Coordinates of(" + std::to_string(goal.position.x()) + ", " +
            std::to_string(goal.position.y()) + ") was outside bounds");
  }
  float goal_orientation_bin = std::round(goal.orientation / _angle_bin_size);
  while (goal_orientation_bin < 0.0f) {
    goal_orientation_bin += static_cast<float>(_angle_quantizations);
  }
  // This is needed to handle precision issues
  if (goal_orientation_bin >= static_cast<float>(_angle_quantizations)) {
    goal_orientation_bin -= static_cast<float>(_angle_quantizations);
  }
  unsigned int goal_orientation_bin_int =
    static_cast<unsigned int>(goal_orientation_bin);
  _a_star->setGoal(goal.position.x(), goal.position.y(), static_cast<unsigned int>(goal_orientation_bin_int),
    _goal_heading_mode, _coarse_search_resolution);

  // Setup message
  Path plan;

  Pose pose;

  pose.orientation = 0.0f;

  // Corner case of start and goal being on the same cell
  if (std::floor(start.position.x()) == std::floor(goal.position.x()) &&
    std::floor( start.position.y()) == std::floor(goal.position.y()) &&
    start_orientation_bin_int == goal_orientation_bin_int)
  {
    pose = start;
    pose.orientation = goal.orientation;
    plan.poses.push_back(pose);


    return plan;
  }

  // Compute plan
  NodeHybrid::CoordinateVector path;
  int num_iterations = 0;
  std::string error;
  std::unique_ptr<std::vector<std::tuple<float, float, float>>> expansions = nullptr;
  if (_debug_visualizations) {
    expansions = std::make_unique<std::vector<std::tuple<float, float, float>>>();
  }
  // Note: All exceptions thrown are handled by the planner server and returned to the action
  if (!_a_star->createPath(
      path, num_iterations,
      _tolerance / static_cast<float>(costmap->getResolution()), cancel_checker, expansions.get()))
  {
    if (_debug_visualizations) {
      PoseArray msg;
      Pose msg_pose;
      for (auto & e : *expansions) {
        msg_pose.position.x() = std::get<0>(e);
        msg_pose.position.y() = std::get<1>(e);
        msg_pose.orientation = std::get<2>(e);
        msg.poses.push_back(msg_pose);
      }

    }

    // Note: If the start is blocked only one iteration will occur before failure
    if (num_iterations == 1) {
      throw StartOccupied("Start occupied");
    }

    if (num_iterations < _a_star->getMaxIterations()) {
      throw NoValidPathCouldBeFound("no valid path found");
    } else {
      throw PlannerTimedOut("exceeded maximum iterations");
    }
  }

  // Convert to world coordinates
  plan.poses.reserve(path.size());
  for (int i = path.size() - 1; i >= 0; --i) {
    pose.position = getCoords(path[i].x, path[i].y);
    pose.orientation = path[i].theta;
    plan.poses.push_back(pose);
  }



  if (_debug_visualizations) {
    // Publish expansions for debug

    PoseArray msg;
    Pose msg_pose;

    for (auto & e : *expansions) {
      msg_pose.position.x() = std::get<0>(e);
      msg_pose.position.y() = std::get<1>(e);
      msg_pose.orientation = std::get<2>(e);
      msg.poses.push_back(msg_pose);
    }


//    if (_planned_footprints_publisher->get_subscription_count() > 0) {
//      // Clear all markers first
//      auto marker_array = std::make_unique<visualization_msgs::msg::MarkerArray>();
//      visualization_msgs::msg::Marker clear_all_marker;
//      clear_all_marker.action = visualization_msgs::msg::Marker::DELETEALL;
//      marker_array->markers.push_back(clear_all_marker);
//      _planned_footprints_publisher->publish(std::move(marker_array));

//      // Publish smoothed footprints for debug
//      marker_array = std::make_unique<visualization_msgs::msg::MarkerArray>();
//      for (size_t i = 0; i < plan.poses.size(); i++) {
//        const std::vector<Eigen::Vector3f> edge =
//          transformFootprintToEdges(plan.poses[i].pose, _costmap_ros->getRobotFootprint());
//        marker_array->markers.push_back(createMarker(edge, i, _global_frame, now));
//      }
//      _planned_footprints_publisher->publish(std::move(marker_array));
//    }
  }

  // Find how much time we have left to do smoothing
  steady_clock::time_point b = steady_clock::now();
  duration<double> time_span = duration_cast<duration<double>>(b - a);
  double time_remaining = _max_planning_time - static_cast<double>(time_span.count());

#ifdef BENCHMARK_TESTING
  std::cout << "It took " << time_span.count() * 1000 <<
    " milliseconds with " << num_iterations << " iterations." << std::endl;
#endif

  // Smooth plan
  if (_smoother && num_iterations > 1) {
    _smoother->smooth(plan, costmap, time_remaining);
  }

#ifdef BENCHMARK_TESTING
  steady_clock::time_point c = steady_clock::now();
  duration<double> time_span2 = duration_cast<duration<double>>(c - b);
  std::cout << "It took " << time_span2.count() * 1000 <<
    " milliseconds to smooth path." << std::endl;
#endif

  if (_debug_visualizations) {
//    if (_smoothed_footprints_publisher->get_subscription_count() > 0) {
//      // Clear all markers first
//      auto marker_array = std::make_unique<visualization_msgs::msg::MarkerArray>();
//      visualization_msgs::msg::Marker clear_all_marker;
//      clear_all_marker.action = visualization_msgs::msg::Marker::DELETEALL;
//      marker_array->markers.push_back(clear_all_marker);
//      _smoothed_footprints_publisher->publish(std::move(marker_array));

//      // Publish smoothed footprints for debug
//      marker_array = std::make_unique<visualization_msgs::msg::MarkerArray>();
//      auto now = _clock->now();
//      for (size_t i = 0; i < plan.poses.size(); i++) {
//        const std::vector<Eigen::Vector3f> edge =
//          transformFootprintToEdges(plan.poses[i].pose, _costmap_ros->getRobotFootprint());
//        marker_array->markers.push_back(createMarker(edge, i, _global_frame, now));
//      }
//      _smoothed_footprints_publisher->publish(std::move(marker_array));
//    }
  }

  return plan;
}


bool SmacPlannerHybrid::dynamicParametersCallback(const SmootherParams& params_s, const SmacPlannerHybridDynamicParams& params)
{
  std::lock_guard<std::mutex> lock_reinit(_mutex);

  bool reinit_collision_checker = false;
  bool reinit_a_star = false;
  bool reinit_downsampler = false;
  bool reinit_smoother = false;

  // Process double parameters
  if (_max_planning_time != params.max_planning_time) {
    reinit_a_star = true;
    _max_planning_time = params.max_planning_time;
  }

  if (_tolerance != static_cast<float>(params.tolerance)) {
    _tolerance = static_cast<float>(params.tolerance);
  }

  if (_lookup_table_size != params.lookup_table_size) {
    reinit_a_star = true;
    _lookup_table_size = params.lookup_table_size;
  }

  if (_minimum_turning_radius_global_coords != static_cast<float>(params.minimum_turning_radius)) {
    reinit_a_star = true;
    if (_smoother) {
      reinit_smoother = true;
    }

    if (params.minimum_turning_radius < _costmap->getResolution() * _downsampling_factor) {
      std::cout << "ERROR: Min turning radius cannot be less than the search grid cell resolution!" << std::endl;
      return false;
    }

    _minimum_turning_radius_global_coords = static_cast<float>(params.minimum_turning_radius);
  }

  if (_search_info.reverse_penalty != static_cast<float>(params.reverse_penalty)) {
    reinit_a_star = true;
    _search_info.reverse_penalty = static_cast<float>(params.reverse_penalty);
  }

  if (_search_info.change_penalty != static_cast<float>(params.change_penalty)) {
    reinit_a_star = true;
    _search_info.change_penalty = static_cast<float>(params.change_penalty);
  }

  if (_search_info.non_straight_penalty != static_cast<float>(params.non_straight_penalty)) {
    reinit_a_star = true;
    _search_info.non_straight_penalty = static_cast<float>(params.non_straight_penalty);
  }

  if (_search_info.cost_penalty != static_cast<float>(params.cost_penalty)) {
    reinit_a_star = true;
    _search_info.cost_penalty = static_cast<float>(params.cost_penalty);
  }

  if (_search_info.analytic_expansion_ratio != static_cast<float>(params.analytic_expansion_ratio)) {
    reinit_a_star = true;
    _search_info.analytic_expansion_ratio = static_cast<float>(params.analytic_expansion_ratio);
  }

  float new_analytic_expansion_max_length = static_cast<float>(params.analytic_expansion_max_length) / _costmap->getResolution();
  if (_search_info.analytic_expansion_max_length != new_analytic_expansion_max_length) {
    reinit_a_star = true;
    _search_info.analytic_expansion_max_length = new_analytic_expansion_max_length;
  }

  if (_search_info.analytic_expansion_max_cost != static_cast<float>(params.analytic_expansion_max_cost)) {
    reinit_a_star = true;
    _search_info.analytic_expansion_max_cost = static_cast<float>(params.analytic_expansion_max_cost);
  }

  // Special case: costmap resolution change
  if (params.resolution > 0.0f && params.resolution != _costmap->getResolution()) {
    std::cout << "INFO: Costmap resolution changed. Reinitializing SmacPlannerHybrid." << std::endl;
    reinit_collision_checker = true;
    reinit_a_star = true;
    reinit_downsampler = true;
    reinit_smoother = true;
  }

  // Process boolean parameters
  if (_downsample_costmap != params.downsample_costmap) {
    reinit_downsampler = true;
    _downsample_costmap = params.downsample_costmap;
  }

  if (_allow_unknown != params.allow_unknown) {
    reinit_a_star = true;
    _allow_unknown = params.allow_unknown;
  }

  if (_search_info.cache_obstacle_heuristic != params.cache_obstacle_heuristic) {
    reinit_a_star = true;
    _search_info.cache_obstacle_heuristic = params.cache_obstacle_heuristic;
  }

  if (_search_info.allow_primitive_interpolation != params.allow_primitive_interpolation) {
    reinit_a_star = true;
    _search_info.allow_primitive_interpolation = params.allow_primitive_interpolation;
  }

  if (params.smooth_path && !_smoother) {
    reinit_smoother = true;
  } else if (!params.smooth_path && _smoother) {
    _smoother.reset();
  }

  if (_search_info.analytic_expansion_max_cost_override != params.analytic_expansion_max_cost_override) {
    reinit_a_star = true;
    _search_info.analytic_expansion_max_cost_override = params.analytic_expansion_max_cost_override;
  }

  // Process integer parameters
  if (_downsampling_factor != params.downsampling_factor) {
    reinit_a_star = true;
    reinit_downsampler = true;
    _downsampling_factor = params.downsampling_factor;
  }

  if (_max_iterations != params.max_iterations) {
    reinit_a_star = true;
    _max_iterations = params.max_iterations;
    if (_max_iterations <= 0) {
      std::cout << "INFO: maximum iteration selected as <= 0, disabling maximum iterations." << std::endl;
      _max_iterations = std::numeric_limits<int>::max();
    }
  }

  if (_max_on_approach_iterations != params.max_on_approach_iterations) {
    reinit_a_star = true;
    _max_on_approach_iterations = params.max_on_approach_iterations;
    if (_max_on_approach_iterations <= 0) {
      std::cout << "INFO: On approach iteration selected as <= 0, disabling tolerance and on approach iterations." << std::endl;
      _max_on_approach_iterations = std::numeric_limits<int>::max();
    }
  }

  if (_terminal_checking_interval != params.terminal_checking_interval) {
    reinit_a_star = true;
    _terminal_checking_interval = params.terminal_checking_interval;
  }

  if (_angle_quantizations != static_cast<unsigned int>(params.angle_quantization_bins)) {
    reinit_collision_checker = true;
    reinit_a_star = true;
    int angle_quantizations = params.angle_quantization_bins;
    _angle_bin_size = 2.0f * M_PIf32 / angle_quantizations;
    _angle_quantizations = static_cast<unsigned int>(angle_quantizations);

    if (_angle_quantizations % _coarse_search_resolution != 0) {
      std::cout << "WARNING: coarse iteration should be an increment of the number of angular bins configured. Disabling course research!" << std::endl;
      _coarse_search_resolution = 1;
    }
  }

  if (_coarse_search_resolution != params.coarse_search_resolution) {
    _coarse_search_resolution = params.coarse_search_resolution;
    if (_coarse_search_resolution <= 0) {
      std::cout << "WARNING: coarse iteration resolution selected as <= 0. Disabling course research!" << std::endl;
      _coarse_search_resolution = 1;
    }
    if (_angle_quantizations % _coarse_search_resolution != 0) {
      std::cout << "WARNING: coarse iteration should be an increment of the number of angular bins configured. Disabling course research!" << std::endl;
      _coarse_search_resolution = 1;
    }
  }

  // Process string parameters
  if (_motion_model_for_search != params.motion_model_for_search) {
    reinit_a_star = true;
    _motion_model_for_search = params.motion_model_for_search;
    _motion_model = fromString(params.motion_model_for_search);
    if (_motion_model == MotionModel::UNKNOWN) {
      std::cout << "WARNING: Unable to get MotionModel search type. Given '"
                << _motion_model_for_search
                << "', valid options are MOORE, VON_NEUMANN, DUBIN, REEDS_SHEPP." << std::endl;
    }
  }

  GoalHeadingMode new_goal_heading_mode = fromStringToGH(params.goal_heading_mode);
  if (_goal_heading_mode != new_goal_heading_mode) {
    std::cout << "INFO: GoalHeadingMode type set to '" << params.goal_heading_mode << "'." << std::endl;

    if (new_goal_heading_mode == GoalHeadingMode::UNKNOWN) {
      std::cout << "WARNING: Unable to get GoalHeader type. Given '"
                << params.goal_heading_mode
                << "', Valid options are DEFAULT, BIDIRECTIONAL, ALL_DIRECTION." << std::endl;
    } else {
      _goal_heading_mode = new_goal_heading_mode;
    }
  }

  // Re-init if needed with mutex lock (to avoid re-init while creating a plan)
  if (reinit_a_star || reinit_downsampler || reinit_collision_checker || reinit_smoother) {
    // convert to grid coordinates
    if (!_downsample_costmap) {
      _downsampling_factor = 1;
    }
    _search_info.minimum_turning_radius =
      _minimum_turning_radius_global_coords / (_costmap->getResolution() * _downsampling_factor);
    _lookup_table_dim =
      static_cast<float>(_lookup_table_size) /
      static_cast<float>(_costmap->getResolution() * _downsampling_factor);

    // Make sure its a whole number
    _lookup_table_dim = static_cast<float>(static_cast<int>(_lookup_table_dim));

    // Make sure its an odd number
    if (static_cast<int>(_lookup_table_dim) % 2 == 0) {
      std::cout << "INFO: Even sized heuristic lookup table size set " << _lookup_table_dim
                << ", increasing size by 1 to make odd" << std::endl;
      _lookup_table_dim += 1.0f;
    }

    // Re-Initialize A* template
    if (reinit_a_star) {
      _a_star = std::make_unique<AStarAlgorithm<NodeHybrid>>(_motion_model, _search_info);
      _a_star->initialize(
        _allow_unknown,
        _max_iterations,
        _max_on_approach_iterations,
        _terminal_checking_interval,
        _max_planning_time,
        _lookup_table_dim,
        _angle_quantizations);
    }

    // Re-Initialize costmap downsampler
    if (reinit_downsampler) {
      if (_downsample_costmap && _downsampling_factor > 1) {
        _costmap_downsampler = std::make_unique<CostmapDownsampler>();
        _costmap_downsampler->on_configure( _costmap, _downsampling_factor);
      }
    }

    // Re-Initialize collision checker
    if (reinit_collision_checker) {
      _collision_checker = CollisionChecker(_costmap_master, _angle_quantizations);
      _collision_checker.setFootprint(
        _costmap_master->getRobotFootprint(),
        _costmap_master->getUseRadius(),
        findCircumscribedCost(_costmap_master));
    }

    // Re-Initialize smoother
    if (reinit_smoother) {
      _smoother = std::make_unique<Smoother>(params_s);
      _smoother->initialize(_minimum_turning_radius_global_coords);
    }
  }

  return true;
}

std::string SmacPlannerHybrid::toStringFromGH(const GoalHeadingMode &goal_heading_mode)
{
    switch (goal_heading_mode) {
        case GoalHeadingMode::DEFAULT:
          return "DEFAULT";
        case GoalHeadingMode::BIDIRECTIONAL:
          return "BIDIRECTIONAL";
        case GoalHeadingMode::ALL_DIRECTION:
          return "ALL_DIRECTION";
        case GoalHeadingMode::UNKNOWN:
        default:
          return "UNKNOWN";
      }
}
