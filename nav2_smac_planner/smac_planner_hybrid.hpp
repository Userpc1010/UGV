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

#ifndef NAV2_SMAC_PLANNER__SMAC_PLANNER_HYBRID_HPP_
#define NAV2_SMAC_PLANNER__SMAC_PLANNER_HYBRID_HPP_

#include <memory>
#include <vector>
#include <string>

#include "a_star.hpp"
#include "smoother.hpp"
#include "utils.hpp"
#include "types.hpp"
#include "costmap_downsampler.hpp"
#include "nav2_costmap_2d/costmap_2d.hpp"
#include "nav2_costmap_2d/costmap_2d_ros.hpp"



class SmacPlannerHybrid
{
public:
  /**
   * @brief constructor
   */
  SmacPlannerHybrid();

  /**
   * @brief destructor
   */
  ~SmacPlannerHybrid();

  /**
   * @brief Configuring plugin
   * @param parent Lifecycle node pointer
   * @param name Name of plugin map
   * @param tf Shared ptr of TF2 buffer
   * @param costmap_ros Costmap2DROS object
   */
  void configure(
    const SmootherParams& params_s, const SmacPlannerHybridParams& params, std::shared_ptr<Costmap2D_Master> costmap_master);


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

  /**
   * @brief Callback executed when a parameter change is detected
   * @param parameters list of changed parameters
   */

  bool dynamicParametersCallback(const SmootherParams& params_s, const SmacPlannerHybridDynamicParams& params);


protected:

  std::unique_ptr<AStarAlgorithm<NodeHybrid>> _a_star;
  CollisionChecker _collision_checker;
  std::unique_ptr<Smoother> _smoother;
  Costmap2D * _costmap;
  std::shared_ptr<Costmap2D_Master> _costmap_master;
  std::unique_ptr<CostmapDownsampler> _costmap_downsampler;
  float _lookup_table_dim;
  float _tolerance;
  bool _downsample_costmap;
  int _downsampling_factor;
  float _angle_bin_size;
  unsigned int _angle_quantizations;
  bool _allow_unknown;
  int _max_iterations;
  int _max_on_approach_iterations;
  int _terminal_checking_interval;
  SearchInfo _search_info;
  double _max_planning_time;
  float _lookup_table_size;
  float _minimum_turning_radius_global_coords;
  bool _debug_visualizations;
  std::string _motion_model_for_search;
  MotionModel _motion_model;
  GoalHeadingMode _goal_heading_mode;
  int _coarse_search_resolution;
  std::mutex _mutex;

private:

  std::string toStringFromGH(const GoalHeadingMode& goal_heading_mode);


};



#endif  // NAV2_SMAC_PLANNER__SMAC_PLANNER_HYBRID_HPP_
