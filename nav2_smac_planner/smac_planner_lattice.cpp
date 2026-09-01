// Copyright (c) 2021, Samsung Research America
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

#include "smac_planner_lattice.hpp"

// #define BENCHMARK_TESTING



using namespace std::chrono;  // NOLINT

SmacPlannerLattice::SmacPlannerLattice()
: _a_star(nullptr),
  _collision_checker(nullptr, 1),
  _smoother(nullptr),
  _costmap(nullptr)
{
 std::cout<<"SmacPlannerLattice"<<std::endl;
}

SmacPlannerLattice::~SmacPlannerLattice()
{
 std::cout << "Destroying plugin SmacPlannerLattice" << std::endl;
}

void SmacPlannerLattice::configure(

  const SmootherParams& params_s, const SmacPlannerLatticeParams& params, std::shared_ptr<Costmap2D_Master> costmap_master)
{

    _costmap = costmap_master->getCostmap();
    _costmap_master = costmap_master;

  std::cout << "Configuring SmacPlannerLattice" << std::endl;

  // General planner params

  // Инициализация переменных
  bool smooth_path = false;                        // Флаг сглаживания пути

  // Основные параметры планировщика
  _tolerance = params.tolerance;                          // ".tolerance" - допуск достижения цели в метрах
  _allow_unknown = params.allow_unknown;                  // ".allow_unknown" - разрешение неизвестных областей
  _max_iterations = params.max_iterations;                // ".max_iterations" - максимальное количество итераций поиска
  _max_on_approach_iterations = params.max_on_approach_iterations; // ".max_on_approach_iterations" - итерации при приближении к цели
  _terminal_checking_interval = params.terminal_checking_interval; // ".terminal_checking_interval" - интервал проверки завершения

  // Параметры решетчатой модели (lattice)
  _search_info.lattice_filepath = params.lattice_filepath; // ".lattice_filepath" - путь к файлу с примитивами решетки

  // Параметры эвристик и штрафов
   _search_info.cache_obstacle_heuristic = params.cache_obstacle_heuristic; // ".cache_obstacle_heuristic" - кэширование эвристики препятствий
  _search_info.reverse_penalty = params.reverse_penalty;  // ".reverse_penalty" - штраф за движение задним ходом
  _search_info.change_penalty = params.change_penalty;    // ".change_penalty" - штраф за смену направления
  _search_info.non_straight_penalty = params.non_straight_penalty; // ".non_straight_penalty" - штраф за непрямолинейное движение
  _search_info.cost_penalty = params.cost_penalty;        // ".cost_penalty" - штраф за стоимость клетки
  _search_info.retrospective_penalty = params.retrospective_penalty; // ".retrospective_penalty" - ретроспективный штраф
  _search_info.rotation_penalty = params.rotation_penalty; // ".rotation_penalty" - штраф за вращение
  _search_info.analytic_expansion_ratio = params.analytic_expansion_ratio; // ".analytic_expansion_ratio" - коэффициент аналитического расширения
  _search_info.analytic_expansion_max_cost = params.analytic_expansion_max_cost; // ".analytic_expansion_max_cost" - максимальная стоимость аналитического расширения
  _search_info.analytic_expansion_max_cost_override = params.analytic_expansion_max_cost_override; // ".analytic_expansion_max_cost_override" - переопределение максимальной стоимости

  // Параметры аналитического расширения
  _search_info.analytic_expansion_max_length = params.analytic_expansion_max_length / _costmap->getResolution(); // ".analytic_expansion_max_length" - максимальная длина аналитического расширения в метрах (конвертация в клетки карты)

  // Параметры производительности и времени
  _max_planning_time = params.max_planning_time;          // ".max_planning_time" - максимальное время планирования
  _lookup_table_size = params.lookup_table_size;          // ".lookup_table_size" - размер таблицы поиска
  _search_info.allow_reverse_expansion = params.allow_reverse_expansion; // ".allow_reverse_expansion" - разрешение обратного расширения

  // Параметры отладки
  _debug_visualizations = params.debug_visualizations;    // ".debug_visualizations" - визуализация отладки

  // Параметры цели
  _goal_heading_mode = fromStringToGH(params.goal_heading_mode); // ".goal_heading_mode" - режим ориентации цели (конвертация строки в enum)

  // Параметры поиска
  _coarse_search_resolution = params.coarse_search_resolution; // ".coarse_search_resolution" - разрешение грубого поиска


  if (_goal_heading_mode == GoalHeadingMode::UNKNOWN) {
    std::string goal_heading_type = toStringFromGH(_goal_heading_mode);
    std::string error_msg = "Unable to get GoalHeader type. Given '" + goal_heading_type + "' "
      "Valid options are DEFAULT, BIDIRECTIONAL, ALL_DIRECTION. ";
    throw PlannerException(error_msg);
  }

  _metadata = LatticeMotionTable::getLatticeMetadata(_search_info.lattice_filepath);
  _search_info.minimum_turning_radius =
    _metadata.min_turning_radius / (_costmap->getResolution());
  _motion_model = MotionModel::STATE_LATTICE;

  if (_max_on_approach_iterations <= 0) {
    std::cout << "INFO: On approach iteration selected as <= 0, "
              << "disabling tolerance and on approach iterations." << std::endl;
    _max_on_approach_iterations = std::numeric_limits<int>::max();
  }

  if (_max_iterations <= 0) {
    std::cout << "INFO: maximum iteration selected as <= 0, "
              << "disabling maximum iterations." << std::endl;
    _max_iterations = std::numeric_limits<int>::max();
  }

  if (_coarse_search_resolution <= 0) {
    std::cout << "WARNING: coarse iteration resolution selected as <= 0, "
              << "disabling coarse iteration resolution search for goal heading" << std::endl;
    _coarse_search_resolution = 1;
  }

  if (_metadata.number_of_headings % _coarse_search_resolution != 0) {
    std::string error_msg = "coarse iteration should be an increment of"
      " the number of angular bins configured";
    throw PlannerException(error_msg);
  }

  float lookup_table_dim =
    static_cast<float>(_lookup_table_size) /
    static_cast<float>(_costmap->getResolution());

  // Make sure its a whole number
  lookup_table_dim = static_cast<float>(static_cast<int>(lookup_table_dim));

  // Make sure its an odd number
  if (static_cast<int>(lookup_table_dim) % 2 == 0) {

      std::cout << "INFO: Even sized heuristic lookup table size set " << lookup_table_dim
                << ", increasing size by 1 to make odd" << std::endl;

    lookup_table_dim += 1.0f;
  }

  // Initialize collision checker using 72 evenly sized bins instead of the lattice
  // heading angles. This is done so that we have precomputed angles every 5 degrees.
  // If we used the sparse lattice headings (usually 16), then when we attempt to collision
  // check for intermediary points of the primitives, we're forced to round to one of the 16
  // increments causing "wobbly" checks that could cause larger robots to virtually show collisions
  // in valid configurations. This approximation helps to bound orientation error for all checks
  // in exchange for slight inaccuracies in the collision headings in terminal search states.
  _collision_checker = CollisionChecker(_costmap_master, 72u);
  _collision_checker.setFootprint(
    costmap_master->getRobotFootprint(),
    costmap_master->getUseRadius(),
    findCircumscribedCost(costmap_master));

  // Initialize A* template
  _a_star = std::make_unique<AStarAlgorithm<NodeLattice>>(_motion_model, _search_info);
  _a_star->initialize(
    _allow_unknown,
    _max_iterations,
    _max_on_approach_iterations,
    _terminal_checking_interval,
    _max_planning_time,
    lookup_table_dim,
    _metadata.number_of_headings);

  // Initialize path smoother
  if (smooth_path) {
    _smoother = std::make_unique<Smoother>(params_s);
    _smoother->initialize(_metadata.min_turning_radius);
  }

  if (_debug_visualizations) {

  }

  std::cout << "Configured plugin SmacPlannerLattice with maximum iterations " << _max_iterations
            << ", max on approach iterations " << _max_on_approach_iterations
            << ", and " << (_allow_unknown ? "allowing unknown traversal" : "not allowing unknown traversal")
            << ". Tolerance " << std::fixed << std::setprecision(2) << _tolerance
            << ". Using motion model: " << toString(_motion_model)
            << ". State lattice file: " << _search_info.lattice_filepath << "." << std::endl;

}


Path SmacPlannerLattice::createPlan(
  const Pose & start,
  const Pose& goal,
  std::function<bool()> cancel_checker)
{
  std::lock_guard<std::mutex> lock_reinit(_mutex);
  steady_clock::time_point a = steady_clock::now();

  std::unique_lock<Costmap2D::mutex_t> lock(*(_costmap->getMutex()));

  // Set collision checker and costmap information
  _collision_checker.setFootprint(
    _costmap_master->getRobotFootprint(),
    _costmap_master->getUseRadius(),
    findCircumscribedCost(_costmap_master));
  _a_star->setCollisionChecker(&_collision_checker);

  // Set starting point, in A* bin search coordinates
  //float mx_start, my_start, mx_goal, my_goal;
  if (!_costmap->PosCheck(start.position.x(), start.position.y()))
  {
    throw StartOutsideMapBounds(
            "Start Coordinates of(" + std::to_string(start.position.x()) + ", " +
            std::to_string(start.position.y()) + ") was outside bounds");
  }
  unsigned int start_bin =
    NodeLattice::motion_table.getClosestAngularBin(start.orientation);
  _a_star->setStart(start.position.x(), start.position.y(), start_bin);

  // Set goal point, in A* bin search coordinates
  if (!_costmap->PosCheck(goal.position.x(), goal.position.y()))
  {
    throw GoalOutsideMapBounds(
            "Goal Coordinates of(" + std::to_string(goal.position.x()) + ", " +
            std::to_string(goal.position.y()) + ") was outside bounds");
  }
  unsigned int goal_bin =
    NodeLattice::motion_table.getClosestAngularBin(goal.orientation);
  _a_star->setGoal(
    goal.position.x(), goal.position.y(), goal_bin,
      _goal_heading_mode, _coarse_search_resolution);

  // Setup message
  Path plan;

  Pose pose;

  pose.orientation = 0.0f;

  // Corner case of start and goal being on the same cell
  if (std::floor(start.position.x()) == std::floor(goal.position.x()) &&
    std::floor(start.position.y()) == std::floor(goal.position.y()) &&
    start_bin == goal_bin)
  {
    pose = start;
    pose.orientation = goal.orientation;
    plan.poses.push_back(pose);


    return plan;
  }

  // Compute plan
  NodeLattice::CoordinateVector path;
  int num_iterations = 0;
  std::string error;
  std::unique_ptr<std::vector<std::tuple<float, float, float>>> expansions = nullptr;
  if (_debug_visualizations) {

    last_expansions_.clear();

    expansions = std::make_unique<std::vector<std::tuple<float, float, float>>>();
  }

  // Note: All exceptions thrown are handled by the planner server and returned to the action
  if (!_a_star->createPath(
      path, num_iterations,
      _tolerance / static_cast<float>(_costmap->getResolution()), cancel_checker, expansions.get()))
  {
      if (_debug_visualizations) {

      if (expansions != nullptr) {
         last_expansions_ = *expansions;
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
  Pose last_pose = pose;
  for (int i = path.size() - 1; i >= 0; --i) {
    pose.position = getCoords(path[i].x, path[i].y);
    pose.orientation = path[i].theta;
    if (fabs(pose.position.x() - last_pose.position.x()) < 1e-4f &&
      fabs(pose.position.y() - last_pose.position.y()) < 1e-4f &&
      fabs(pose.orientation - last_pose.orientation) < 1e-4f)
    {
        std::cout << "DEBUG: Removed a path from the path due to replication. "
                  << "Make sure your minimum control set does not contain duplicate values!" << std::endl;
      continue;
    }
    last_pose = pose;
    plan.poses.push_back(pose);
  }

  if (_debug_visualizations) {

      if (expansions != nullptr) {
         last_expansions_ = *expansions;
       }

    // Publish expansions for debug
//    PoseArray msg;
//    Pose msg_pose;

//    for (auto & e : *expansions) {
//      msg_pose.position.x() = std::get<0>(e);
//      msg_pose.position.y() = std::get<1>(e);
//      msg_pose.orientation = std::get<2>(e);
//      msg.poses.push_back(msg_pose);
//    }

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
    _smoother->smooth(plan, _costmap, time_remaining);
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


bool SmacPlannerLattice::dynamicParametersCallback(const SmootherParams& params_s, const SmacPlannerLatticeDynamicParams& params)
{
  std::lock_guard<std::mutex> lock_reinit(_mutex);

  bool reinit_a_star = false;
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

  if (_search_info.rotation_penalty != static_cast<float>(params.rotation_penalty)) {
    reinit_a_star = true;
    _search_info.rotation_penalty = static_cast<float>(params.rotation_penalty);
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

  // Process boolean parameters
  if (_allow_unknown != params.allow_unknown) {
    reinit_a_star = true;
    _allow_unknown = params.allow_unknown;
  }

  if (_search_info.cache_obstacle_heuristic != params.cache_obstacle_heuristic) {
    reinit_a_star = true;
    _search_info.cache_obstacle_heuristic = params.cache_obstacle_heuristic;
  }

  if (_search_info.allow_reverse_expansion != params.allow_reverse_expansion) {
    reinit_a_star = true;
    _search_info.allow_reverse_expansion = params.allow_reverse_expansion;
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

  if (_coarse_search_resolution != params.coarse_search_resolution) {
    _coarse_search_resolution = params.coarse_search_resolution;
    if (_coarse_search_resolution <= 0) {
      std::cout << "WARNING: coarse iteration resolution selected as <= 0. Disabling course research!" << std::endl;
      _coarse_search_resolution = 1;
    }
    if (_metadata.number_of_headings % _coarse_search_resolution != 0) {
      std::cout << "WARNING: coarse iteration should be an increment of the number of angular bins configured. Disabling course research!" << std::endl;
      _coarse_search_resolution = 1;
    }
  }

  // Process string parameters
  if (_search_info.lattice_filepath != params.lattice_filepath) {
    reinit_a_star = true;
    if (_smoother) {
      reinit_smoother = true;
    }
    _search_info.lattice_filepath = params.lattice_filepath;
    _metadata = LatticeMotionTable::getLatticeMetadata(_search_info.lattice_filepath);
    _search_info.minimum_turning_radius = _metadata.min_turning_radius / _costmap->getResolution();

    if (_metadata.number_of_headings % _coarse_search_resolution != 0) {
      std::cout << "WARNING: coarse iteration should be an increment of the number of angular bins configured. Disabling course research!" << std::endl;
      _coarse_search_resolution = 1;
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
  if (reinit_a_star || reinit_smoother) {
    // convert to grid coordinates
    _search_info.minimum_turning_radius = _metadata.min_turning_radius / _costmap->getResolution();
    float lookup_table_dim = static_cast<float>(_lookup_table_size) / static_cast<float>(_costmap->getResolution());

    // Make sure its a whole number
    lookup_table_dim = static_cast<float>(static_cast<int>(lookup_table_dim));

    // Make sure its an odd number
    if (static_cast<int>(lookup_table_dim) % 2 == 0) {
      std::cout << "INFO: Even sized heuristic lookup table size set " << lookup_table_dim
                << ", increasing size by 1 to make odd" << std::endl;
      lookup_table_dim += 1.0f;
    }

    // Re-Initialize smoother
    if (reinit_smoother) {
      _smoother = std::make_unique<Smoother>(params_s);
      _smoother->initialize(_metadata.min_turning_radius);
    }

    // Re-Initialize A* template
    if (reinit_a_star) {
      _a_star = std::make_unique<AStarAlgorithm<NodeLattice>>(_motion_model, _search_info);
      _a_star->initialize(
        _allow_unknown,
        _max_iterations,
        _max_on_approach_iterations,
        _terminal_checking_interval,
        _max_planning_time,
        lookup_table_dim,
        _metadata.number_of_headings);
    }
  }

  return true;
}

std::string SmacPlannerLattice::toStringFromGH(const GoalHeadingMode &goal_heading_mode)
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

