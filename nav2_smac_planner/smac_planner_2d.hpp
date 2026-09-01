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

#ifndef NAV2_SMAC_PLANNER__SMAC_PLANNER_2D_HPP_
#define NAV2_SMAC_PLANNER__SMAC_PLANNER_2D_HPP_

#include <memory>
#include <vector>
#include <string>
#include <mutex>

#include "a_star.hpp"
#include "smoother.hpp"
#include "utils.hpp"
#include "types.hpp"
#include "costmap_downsampler.hpp"
#include "nav2_costmap_2d/costmap_2d.hpp"
#include "nav2_costmap_2d/costmap_2d_ros.hpp"


class SmacPlanner2D
{
public:
  /**
   * @brief constructor
   */
  SmacPlanner2D();

  /**
   * @brief destructor
   */
  ~SmacPlanner2D();

  /**
   * @brief Configuring plugin
   * @param parent Lifecycle node pointer
   * @param name Name of plugin map
   * @param tf Shared ptr of TF2 buffer
   * @param costmap_ros Costmap2DROS object
   */
  void configure(const SmootherParams& params_s, const SmacPlanner2DParams& params,
    std::shared_ptr<Costmap2D_Master> costmap_master);

  /**
   * @brief Creating a plan from start and goal poses
   * @param start Start pose
   * @param goal Goal pose
   * @param cancel_checker Function to check if the action has been canceled
   * @return nav2_msgs::Path of the generated path
   */
  Path createPlan(
  const Pose & start,
  const Pose & goal,
  std::function<bool()> cancel_checker);

  void analyzeCostmap() const;

protected:
  /**
   * @brief Callback executed when a parameter change is detected
   * @param event ParameterEvent message
   */
 bool dynamicParametersCallback(const SmacPlanner2dDynamicParams& params);

  std::unique_ptr<AStarAlgorithm<Node2D>> _a_star;
  CollisionChecker _collision_checker;
  std::unique_ptr<Smoother> _smoother;
  Costmap2D * _costmap;
  std::unique_ptr<CostmapDownsampler> _costmap_downsampler;
  float _tolerance;
  int _downsampling_factor;
  bool _downsample_costmap;
  double _max_planning_time;
  bool _allow_unknown;
  int _max_iterations;
  int _max_on_approach_iterations;
  int _terminal_checking_interval;
  bool _use_final_approach_orientation;
  SearchInfo _search_info;
  std::string _motion_model_for_search;
  MotionModel _motion_model;
  std::mutex _mutex;

};



#endif  // NAV2_SMAC_PLANNER__SMAC_PLANNER_2D_HPP_



