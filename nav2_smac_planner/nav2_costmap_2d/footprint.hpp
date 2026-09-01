/*********************************************************************
 *
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2008, 2013, Willow Garage, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 * Author: Eitan Marder-Eppstein
 *         David V. Lu!!
 *********************************************************************/
#ifndef NAV2_COSTMAP_2D__FOOTPRINT_HPP_
#define NAV2_COSTMAP_2D__FOOTPRINT_HPP_

#include <string>
#include <vector>
#include <utility>

#include "datatypes_smac_planner.h"


/**
 * @brief Calculate the extreme distances for the footprint
 *
 * @param footprint The footprint to examine
 * @param min_dist Output parameter of the minimum distance
 * @param max_dist Output parameter of the maximum distance
 */
std::pair<float, float> calculateMinAndMaxDistances(
  const std::vector<Eigen::Vector2f> & footprint);

/**
 * @brief Convert Point32 to Point
 */
Eigen::Vector2f toPoint(const Eigen::Vector2d & pt);

/**
 * @brief Convert Point to Point32
 */
Eigen::Vector2d toPoint64(const Eigen::Vector2f & pt);

/**
 * @brief Convert vector of Points to Polygon msg
 */
Polygon toPolygon(const std::vector<Eigen::Vector2f> & pts);

/**
 * @brief Convert Polygon msg to vector of Points.
 */
std::vector<Eigen::Vector2f> toPointVector(
  const Polygon & polygon);

/**
 * @brief  Given a pose and base footprint, build the oriented footprint of the robot (list of Points)
 * @param  x The x position of the robot
 * @param  y The y position of the robot
 * @param  theta The orientation of the robot
 * @param  footprint_spec Basic shape of the footprint
 * @param  oriented_footprint Will be filled with the points in the oriented footprint of the robot
*/
void transformFootprint(
  float x, float y, float theta,
  const std::vector<Eigen::Vector2f> & footprint_spec,
  std::vector<Eigen::Vector2f> & oriented_footprint);

/**
 * @brief  Given a pose and base footprint, build the oriented footprint of the robot (PolygonStamped)
 * @param  x The x position of the robot
 * @param  y The y position of the robot
 * @param  theta The orientation of the robot
 * @param  footprint_spec Basic shape of the footprint
 * @param  oriented_footprint Will be filled with the points in the oriented footprint of the robot
*/
void transformFootprint(
  float x, float y, float theta,
  const std::vector<Eigen::Vector2f> & footprint_spec,
  PolygonStamped & oriented_footprint);

/**
 * @brief Adds the specified amount of padding to the footprint (in place)
 */
void padFootprint(std::vector<Eigen::Vector2f> & footprint, float padding);

/**
 * @brief Create a circular footprint from a given radius
 */
std::vector<Eigen::Vector2f> makeFootprintFromRadius(float radius);

/**
 * @brief Make the footprint from the given string.
 *
 * Format should be bracketed array of arrays of floats, like so: [[1.0, 2.2], [3.3, 4.2], ...]
 *
 */
bool makeFootprintFromString(
  const std::string & footprint_string,
  std::vector<Eigen::Vector2f> & footprint);


#endif  // NAV2_COSTMAP_2D__FOOTPRINT_HPP_
