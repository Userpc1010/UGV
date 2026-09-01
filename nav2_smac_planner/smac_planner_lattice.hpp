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

#ifndef NAV2_SMAC_PLANNER__SMAC_PLANNER_LATTICE_HPP_
#define NAV2_SMAC_PLANNER__SMAC_PLANNER_LATTICE_HPP_

#include <memory>
#include <vector>
#include <string>

#include "a_star.hpp"
#include "smoother.hpp"
#include "utils.hpp"
#include "types.hpp"
#include "nav2_costmap_2d/costmap_2d.hpp"
#include "nav2_costmap_2d/costmap_2d_ros.hpp"



class SmacPlannerLattice
{
public:
  /**
   * @brief constructor
   */
  SmacPlannerLattice();

  /**
   * @brief destructor
   */
  ~SmacPlannerLattice();

  /**
   * @brief Configuring plugin
   * @param parent Lifecycle node pointer
   * @param name Name of plugin map
   * @param tf Shared ptr of TF2 buffer
   * @param costmap_ros Costmap2DROS object
   */
  void configure(
   const SmootherParams& params_s, const SmacPlannerLatticeParams& params, std::shared_ptr<Costmap2D_Master> costmap_master);

  /**
   * @brief Creating a plan from start and goal poses
   * @param start Start pose
   * @param goal Goal pose
   * @param cancel_checker Function to check if the action has been canceled
   * @return nav2_msgs::Path of the generated path
   */
  Path createPlan(
    const Pose& start,
    const Pose & goal,
    std::function<bool()> cancel_checker);

  // Добавляем метод для получения экспансий
    const std::vector<std::tuple<float, float, float>>& getLastExpansions() const {
      return last_expansions_;
    }

    // Добавляем метод для очистки экспансий
    void clearExpansions() {
      last_expansions_.clear();
    }

  /**
   * @brief Callback executed when a parameter change is detected
   * @param parameters list of changed parameters
   */

  bool dynamicParametersCallback(const SmootherParams& params_s, const SmacPlannerLatticeDynamicParams& params);

protected:

  std::unique_ptr<AStarAlgorithm<NodeLattice>> _a_star;
  CollisionChecker _collision_checker;
  std::unique_ptr<Smoother> _smoother;
  Costmap2D * _costmap;
  std::shared_ptr<Costmap2D_Master> _costmap_master;
  std::vector<std::tuple<float, float, float>> last_expansions_;
  MotionModel _motion_model;
  LatticeMetadata _metadata;
  SearchInfo _search_info;
  bool _allow_unknown;
  int _max_iterations;
  int _max_on_approach_iterations;
  int _terminal_checking_interval;
  float _tolerance;
  double _max_planning_time;
  float _lookup_table_size;
  bool _debug_visualizations;
  GoalHeadingMode _goal_heading_mode;
  int _coarse_search_resolution;
  std::mutex _mutex;

private:

  std::string toStringFromGH(const GoalHeadingMode& goal_heading_mode);

  // Dynamic parameters handler

};



#endif  // NAV2_SMAC_PLANNER__SMAC_PLANNER_LATTICE_HPP_
