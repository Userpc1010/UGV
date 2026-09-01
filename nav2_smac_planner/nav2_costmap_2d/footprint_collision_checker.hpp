// Copyright (c) 2019 Intel Corporation
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
// limitations under the License.
//
// Modified by: Shivang Patel (shivaang14@gmail.com)

#ifndef NAV2_COSTMAP_2D__FOOTPRINT_COLLISION_CHECKER_HPP_
#define NAV2_COSTMAP_2D__FOOTPRINT_COLLISION_CHECKER_HPP_

#include <string>
#include <vector>
#include <memory>
#include <algorithm>

#include "nav2_costmap_2d/costmap_2d.hpp"

typedef struct{
    uint16_t x;
    uint16_t y;
} Footprint;

/**
 * @class FootprintCollisionChecker
 * @brief Checker for collision with a footprint on a costmap
 */
template<typename CostmapT>
class FootprintCollisionChecker
{
public:
  /**
   * @brief A constructor.
   */
  FootprintCollisionChecker();
  /**
   * @brief A constructor.
   */
  explicit FootprintCollisionChecker(CostmapT costmap);
  /**
   * @brief Find the footprint cost in oriented footprint
   */
  float footprintCost(const uint16_t& x, const uint16_t& y, const Footprint * footprint, const uint16_t& size);
  /**
   * @brief Find the footprint cost a a post with an unoriented footprint
   */
  float footprintCostAtPose(float x, float y, float theta, const Footprint * footprint,  const uint16_t& size);
  /**
   * @brief Get the cost for a line segment
   */
  float lineCost(int x0, int x1, int y0, int y1) const;
  /**
   * @brief Check the map coordinates
   */
  bool PosCheck (float px, float py);
  /**
   * @brief Get the cost of a point
   */
  float pointCost(int x, int y) const;
  /**
  * @brief Set the current costmap object to use for collision detection
  */
  void setCostmap(CostmapT costmap);
  /**
  * @brief Get the current costmap object
  */
  CostmapT getCostmap()
  {
    return costmap_;
  }

protected:
  CostmapT costmap_;
};



#endif  // NAV2_COSTMAP_2D__FOOTPRINT_COLLISION_CHECKER_HPP_
