// Copyright (c) 2020, Samsung Research America
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
#include <limits>
#include <algorithm>

#include "smac_planner_2d.hpp"
#include "geometry_utils.hpp"

// #define BENCHMARK_TESTING


using namespace std::chrono;  // NOLINT
using std::placeholders::_1;

SmacPlanner2D::SmacPlanner2D()
: _a_star(nullptr),
  _collision_checker(nullptr, 1),
  _smoother(nullptr),
  _costmap(nullptr),
  _costmap_downsampler(nullptr)
{
 std::cout <<"SmacPlanner2D"<< std::endl;
}

SmacPlanner2D::~SmacPlanner2D()
{
  std::cout << "Destroying plugin SmacPlanner2D" << std::endl;
}

void SmacPlanner2D::analyzeCostmap() const
{
    if (!_costmap) {
        std::cout << "[ANALYSIS] Costmap is null!" << std::endl;
        return;
    }

    unsigned int size_x = _costmap->getSizeInCellsX();
    unsigned int size_y = _costmap->getSizeInCellsY();
    unsigned int total_cells = size_x * size_y;

    std::map<unsigned char, unsigned int> cost_distribution;
    unsigned int unknown_count = 0;
    unsigned int lethal_count = 0;
    unsigned int free_count = 0;
    unsigned int inflated_count = 0;

    // Константы из cost_values.hpp (предположительные)
    const unsigned char FREE_SPACE = 0;
    const unsigned char LETHAL_OBSTACLE = 254; // или 255
    const unsigned char INSCRIBED_INFLATED_OBSTACLE = 253;
    const unsigned char NO_INFORMATION = 255;

    for (unsigned int y = 0; y < size_y; ++y) {
        for (unsigned int x = 0; x < size_x; ++x) {
            unsigned char cost = _costmap->getCost(x, y);
            cost_distribution[cost]++;

            if (cost == NO_INFORMATION) unknown_count++;
            else if (cost == LETHAL_OBSTACLE) lethal_count++;
            else if (cost == FREE_SPACE) free_count++;
            else if (cost >= INSCRIBED_INFLATED_OBSTACLE) inflated_count++;
        }
    }

    std::cout << "\n=== COSTMAP ANALYSIS ===" << std::endl;
    std::cout << "Size: " << size_x << "x" << size_y << " (" << total_cells << " cells)" << std::endl;
    std::cout << "Physical: " << _costmap->getSizeInMetersX()
              << "x" << _costmap->getSizeInMetersY() << " meters" << std::endl;
    std::cout << "Resolution: " << _costmap->getResolution() << " m/cell" << std::endl;
    std::cout << "Origin: (" << _costmap->getOriginX()
              << ", " << _costmap->getOriginY() << ")" << std::endl;

    std::cout << "\nCost distribution:" << std::endl;
    std::cout << "  Free (0): " << free_count << " ("
              << (free_count * 100.0 / total_cells) << "%)" << std::endl;
    std::cout << "  Lethal (254): " << lethal_count << " ("
              << (lethal_count * 100.0 / total_cells) << "%)" << std::endl;
    std::cout << "  Inflated (253): " << inflated_count << " ("
              << (inflated_count * 100.0 / total_cells) << "%)" << std::endl;
    std::cout << "  Unknown (255): " << unknown_count << " ("
              << (unknown_count * 100.0 / total_cells) << "%)" << std::endl;

    std::cout << "\nOther cost values:" << std::endl;
    for (const auto& [cost, count] : cost_distribution) {
        if (cost != FREE_SPACE && cost != LETHAL_OBSTACLE &&
            cost != INSCRIBED_INFLATED_OBSTACLE && cost != NO_INFORMATION) {
            std::cout << "  Cost " << (int)cost << ": " << count << " cells" << std::endl;
        }
    }

    // Проверка границ
    std::cout << "\nBorder check:" << std::endl;
    unsigned int border_obstacles = 0;
    // Верхняя и нижняя границы
    for (unsigned int x = 0; x < size_x; ++x) {
        if (_costmap->getCost(x, 0) == LETHAL_OBSTACLE) border_obstacles++;
        if (_costmap->getCost(x, size_y-1) == LETHAL_OBSTACLE) border_obstacles++;
    }
    // Левая и правая границы (учитывая углы)
    for (unsigned int y = 1; y < size_y-1; ++y) {
        if (_costmap->getCost(0, y) == LETHAL_OBSTACLE) border_obstacles++;
        if (_costmap->getCost(size_x-1, y) == LETHAL_OBSTACLE) border_obstacles++;
    }
    std::cout << "  Border obstacles: " << border_obstacles << std::endl;
}

void SmacPlanner2D::configure(const SmootherParams& params_s,
 const SmacPlanner2DParams& params, std::shared_ptr<Costmap2D_Master> costmap_master)
{
  _costmap = costmap_master->getCostmap();

 std::cout << "Configuring SmacPlanner2D" << std::endl;

  // General planner params

 // Основные параметры планирования
 _tolerance = params.tolerance;                          // ".tolerance" - допуск достижения цели в метрах
 _search_info.cost_penalty = params.cost_penalty;        // ".cost_penalty" - множитель штрафа за стоимость клетки
 _max_planning_time = params.max_planning_time;          // ".max_planning_time" - максимальное время планирования в секундах

 // Параметры оптимизации и поиска
 _downsample_costmap = params.downsample_costmap;        // ".downsample_costmap" - включение даунсемплинга карты стоимости
 _downsampling_factor = params.downsampling_factor;      // ".downsampling_factor" - фактор уменьшения разрешения карты стоимости
 _allow_unknown = params.allow_unknown;                  // ".allow_unknown" - разрешение планирования через неизвестные области
 _max_iterations = params.max_iterations;                // ".max_iterations" - максимальное количество итераций поиска
 _max_on_approach_iterations = params.max_on_approach_iterations; // ".max_on_approach_iterations" - макс. итераций при приближении к цели
 _terminal_checking_interval = params.terminal_checking_interval; // ".terminal_checking_interval" - интервал проверки условий завершения

 // Параметры поведения
 _use_final_approach_orientation = params.use_final_approach_orientation; // ".use_final_approach_orientation" - использование финальной ориентации подхода
 _motion_model = params.motion_model;                    // ".motion_model" - модель движения (2D для плоского планирования)


  if (_max_on_approach_iterations <= 0) {
    std::cout << "On approach iteration selected as <= 0, "
              << "disabling tolerance and on approach iterations." << std::endl;
    _max_on_approach_iterations = std::numeric_limits<int>::max();
  }

  if (_max_iterations <= 0) {
    std::cout << "maximum iteration selected as <= 0, "
              << "disabling maximum iterations." << std::endl;
    _max_iterations = std::numeric_limits<int>::max();
  }

  // Initialize collision checker
  _collision_checker = CollisionChecker(costmap_master, 1 /*for 2D, most be 1*/);
  _collision_checker.setFootprint(
  costmap_master->getRobotFootprint(),
    true /*for 2D, most use radius*/,
    0.0f /*for 2D cost at inscribed isn't relevant*/);

  auto footprint = costmap_master->getRobotFootprint();
  std::cout << "[DEBUG] Footprint sent to collision checker:" << std::endl;
  for (const auto& p : footprint) {
      std::cout << "  (" << p.x() << ", " << p.y() << ")" << std::endl;
  }

  // Initialize A* template
  _a_star = std::make_unique<AStarAlgorithm<Node2D>>(_motion_model, _search_info);
  _a_star->initialize(
    _allow_unknown,
    _max_iterations,
    _max_on_approach_iterations,
    _terminal_checking_interval,
    _max_planning_time,
    0.0f /*unused for 2D*/,
    1.0f /*unused for 2D*/);

  _smoother = std::make_unique<Smoother>(params_s);
  _smoother->initialize(1e-50 /*No valid minimum turning radius for 2D*/);

  // Initialize costmap downsampler
  _costmap_downsampler = std::make_unique<CostmapDownsampler>();
  _costmap_downsampler->on_configure(_costmap, _downsampling_factor);

  std::cout << "Configured plugin SmacPlanner2D with tolerance " << std::fixed << std::setprecision(2) << _tolerance
            << ", maximum iterations " << _max_iterations
            << ", max on approach iterations " << _max_on_approach_iterations
            << ", and " << (_allow_unknown ? "allowing unknown traversal" : "not allowing unknown traversal")
            << "." << std::endl;
}

  Path SmacPlanner2D::createPlan(
  const Pose& start,
  const Pose & goal,
  std::function<bool()> cancel_checker)
{
  std::lock_guard<std::mutex> lock_reinit(_mutex);
  steady_clock::time_point a = steady_clock::now();

  std::unique_lock<Costmap2D::mutex_t> lock(*(_costmap->getMutex()));

  // Downsample costmap, if required
    Costmap2D * costmap = _costmap;
    if (_downsample_costmap && _downsampling_factor > 1) {
      std::cout << "[DEBUG] Downsampling costmap with factor: " << _downsampling_factor << std::endl;
      costmap = _costmap_downsampler->downsample(_downsampling_factor);
      std::cout << "[DEBUG] Downsampled costmap size: " << costmap->getSizeInCellsX()
                << "x" << costmap->getSizeInCellsY() << std::endl;
      _collision_checker.setCostmap(costmap);
    }

  // Set collision checker and costmap information
  _a_star->setCollisionChecker(&_collision_checker);

  // Set starting point
  //float mx_start, my_start, mx_goal, my_goal;
  if (!costmap->PosCheck(start.position.x(), start.position.y()))
  {
    throw StartOutsideMapBounds(
            "Start Coordinates of(" + std::to_string(start.position.x()) + ", " +
            std::to_string(start.position.y()) + ") was outside bounds");
  }

  std::cout << "[DEBUG] Start world: (" << start.position.x() << ", " << start.position.y() << ")"
             << " -> map: (" << start.position.x() << ", " << start.position.y() << ")"
             << " cell: (" << std::floor(start.position.x()) << ", " << std::floor(start.position.y()) << ")" << std::endl;

  _a_star->setStart(start.position.x(), start.position.y(), 0);

  // Set goal point
  if (!costmap->PosCheck(goal.position.x(),goal.position.y()))
  {
    throw GoalOutsideMapBounds(
            "Goal Coordinates of(" + std::to_string(goal.position.x()) + ", " +
            std::to_string(goal.position.y()) + ") was outside bounds");
  }

  std::cout << "[DEBUG] Goal world: (" << goal.position.x() << ", " << goal.position.y() << ")"
             << " -> map: (" << goal.position.x() << ", " << goal.position.y() << ")"
             << " cell: (" << std::floor(goal.position.x()) << ", " << std::floor(goal.position.y()) << ")" << std::endl;

   // Проверка: не в одной ли ячейке?
   if (std::floor(start.position.x()) == std::floor(goal.position.x()) && std::floor(start.position.y()) == std::floor(goal.position.y())) {
     std::cout << "[WARNING] Start and goal are in the same cell! A* will not run." << std::endl;
     std::cout << "[WARNING] This is likely due to coordinate conversion error." << std::endl;
   }


  _a_star->setGoal(goal.position.x(), goal.position.y(), 0);

  // Setup message
  Path plan;
  Pose pose;

  pose.orientation = 0.0f;

  // Corner case of start and goal being on the same cell
  if (std::floor(start.position.x()) == std::floor(goal.position.x()) && std::floor(start.position.y()) == std::floor(goal.position.y())) {
    pose = start;
    // if we have a different start and goal orientation, set the unique path pose to the goal
    // orientation, unless use_final_approach_orientation=true where we need it to be the start
    // orientation to avoid movement from the local planner
    if (!orientationsEqual(start.orientation, goal.orientation, 1e-6f)  && !_use_final_approach_orientation) {
          pose.orientation = goal.orientation;
     }
    plan.poses.push_back(pose);

    return plan;
  }

  // Compute plan
  Node2D::CoordinateVector path;
  int num_iterations = 0;

  analyzeCostmap();

  std::cout << "[DEBUG] Calling A* createPath..." << std::endl;

  // Note: All exceptions thrown are handled by the planner server and returned to the action
  if (!_a_star->createPath(
      path, num_iterations,
      _tolerance / static_cast<float>(costmap->getResolution()), cancel_checker))
  {
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

  std::cout << "[DEBUG] A* completed with " << num_iterations << " iterations" << std::endl;
    std::cout << "[DEBUG] Raw path has " << path.size() << " nodes" << std::endl;

    // Вывести первые 10 точек пути
    for (size_t i = 0; i < std::min(path.size(), size_t(10)); ++i) {
      std::cout << "[DEBUG] Path node " << i << ": (" << path[i].x << ", " << path[i].y << ")" << std::endl;
    }

  // Convert to world coordinates
  plan.poses.reserve(path.size());
  for (int i = path.size() - 1; i >= 0; --i) {
    pose.position = getCoords(path[i].x, path[i].y);
    plan.poses.push_back(pose);
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
  _smoother->smooth(plan, costmap, time_remaining);

  // If use_final_approach_orientation=true, interpolate the last pose orientation from the
  // previous pose to set the orientation to the 'final approach' orientation of the robot so
  // it does not rotate.
  // And deal with corner case of plan of length 1
  // If use_final_approach_orientation=false (default), override last pose orientation to match goal
  size_t plan_size = plan.poses.size();
  if (_use_final_approach_orientation) {
    if (plan_size == 1) {
      plan.poses.back().orientation = start.orientation;
    } else if (plan_size > 1) {
      float dx, dy, theta;
      auto last_pose = plan.poses.back().position;
      auto approach_pose = plan.poses[plan_size - 2].position;
      dx = last_pose.x() - approach_pose.x();
      dy = last_pose.y() - approach_pose.y();
      theta = atan2(dy, dx);
      plan.poses.back().orientation = theta;
    }
  } else if (plan_size > 0) {
    plan.poses.back().orientation = goal.orientation;
  }

  return plan;
}


  bool SmacPlanner2D::dynamicParametersCallback(const SmacPlanner2dDynamicParams& params)
  {
    std::lock_guard<std::mutex> lock_reinit(_mutex);

    bool reinit_a_star = false;
    bool reinit_downsampler = false;

    // Обработка параметров с проверкой изменений
    if (_tolerance != params.tolerance) {
      _tolerance = params.tolerance;
    }

    if (_search_info.cost_penalty != params.cost_travel_multiplier) {
      reinit_a_star = true;
      _search_info.cost_penalty = params.cost_travel_multiplier;
    }

    if (_max_planning_time != params.max_planning_time) {
      reinit_a_star = true;
      _max_planning_time = params.max_planning_time;
    }

    if (_downsample_costmap != params.downsample_costmap) {
      reinit_downsampler = true;
      _downsample_costmap = params.downsample_costmap;
    }

    if (_allow_unknown != params.allow_unknown) {
      reinit_a_star = true;
      _allow_unknown = params.allow_unknown;
    }

    if (_use_final_approach_orientation != params.use_final_approach_orientation) {
      _use_final_approach_orientation = params.use_final_approach_orientation;
    }

    if (_downsampling_factor != params.downsampling_factor) {
      reinit_downsampler = true;
      _downsampling_factor = params.downsampling_factor;
    }

    if (_max_iterations != params.max_iterations) {
      reinit_a_star = true;
      _max_iterations = params.max_iterations;
      if (_max_iterations <= 0) {
        std::cout << "maximum iteration selected as <= 0, disabling maximum iterations." << std::endl;
        _max_iterations = std::numeric_limits<int>::max();
      }
    }

    if (_max_on_approach_iterations != params.max_on_approach_iterations) {
      reinit_a_star = true;
      _max_on_approach_iterations = params.max_on_approach_iterations;
      if (_max_on_approach_iterations <= 0) {
        std::cout << "On approach iteration selected as <= 0, disabling tolerance and on approach iterations." << std::endl;
        _max_on_approach_iterations = std::numeric_limits<int>::max();
      }
    }

    if (_terminal_checking_interval != params.terminal_checking_interval) {
      reinit_a_star = true;
      _terminal_checking_interval = params.terminal_checking_interval;
    }

    // Re-init if needed with mutex lock
    if (reinit_a_star || reinit_downsampler) {
      // Re-Initialize A* template
      if (reinit_a_star) {
        _a_star = std::make_unique<AStarAlgorithm<Node2D>>(_motion_model, _search_info);
        _a_star->initialize(
          _allow_unknown,
          _max_iterations,
          _max_on_approach_iterations,
          _terminal_checking_interval,
          _max_planning_time,
          0.0f /*unused for 2D*/,
          1.0f /*unused for 2D*/);
      }

      // Re-Initialize costmap downsampler
      if (reinit_downsampler) {
        if (_downsample_costmap && _downsampling_factor > 1) {
          _costmap_downsampler = std::make_unique<CostmapDownsampler>();
          _costmap_downsampler->on_configure(_costmap, _downsampling_factor);
        }
      }
    }

    return true;
  }


