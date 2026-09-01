// Copyright (c) 2022, Samsung Research America
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

#ifndef NAV2_UTIL__SMOOTHER_UTILS_HPP_
#define NAV2_UTIL__SMOOTHER_UTILS_HPP_

#include <cmath>
#include <vector>
#include <string>
#include <memory>
#include <utility>

#include "geometry_utils.hpp"
#include "angles/angles.h"

#ifndef M_PIf32
#define M_PIf32 3.14159265358979323846f

#endif

#ifndef M_PIf32_2
#define M_PIf32_2 1.57079632679489661923f

#endif

/**
 * @class nav2_util::PathSegment
 * @brief A segment of a path in start/end indices
 */
struct PathSegment
{
  unsigned int start;
  unsigned int end;
};



/**
 * @brief Finds the starting and end indices of path segments where
 * the robot is traveling in the same direction (e.g. forward vs reverse)
 * @param path Path in which to look for cusps
 * @param is_holonomic Whether the motion model is holonomic (default is false)
 Only set as true when the input path is known to be generated from a holonomic planner like NavFn.
 * @return Set of index pairs for each segment of the path in a given direction
 */
inline std::vector<PathSegment> findDirectionalPathSegments(
  const Path & path, bool is_holonomic = false)
{
  std::vector<PathSegment> segments;
  PathSegment curr_segment;
  curr_segment.start = 0;

  // If holonomic, no directional changes and
  // may have abrupt angular changes from naive grid search
  if (is_holonomic) {
    curr_segment.end = path.poses.size() - 1;
    segments.push_back(curr_segment);
    return segments;
  }

  // Iterating through the path to determine the position of the cusp
  for (unsigned int idx = 1; idx < path.poses.size() - 1; ++idx) {
    // We have two vectors for the dot product OA and AB. Determining the vectors.
    float oa_x = path.poses[idx].position.x() -
      path.poses[idx - 1].position.x();
    float oa_y = path.poses[idx].position.y() -
      path.poses[idx - 1].position.y();
    float ab_x = path.poses[idx + 1].position.x() -
      path.poses[idx].position.x();
    float ab_y = path.poses[idx + 1].position.y() -
      path.poses[idx].position.y();

    // Checking for the existence of cusp, in the path, using the dot product.
    float dot_product = (oa_x * ab_x) + (oa_y * ab_y);
    if (dot_product < 0.0f) {
      curr_segment.end = idx;
      segments.push_back(curr_segment);
      curr_segment.start = idx;
    }

    // Checking for the existence of a differential rotation in place.
    float cur_theta = path.poses[idx].orientation;
    float next_theta = path.poses[idx + 1].orientation;
    float dtheta = angles::shortest_angular_distance(cur_theta, next_theta);
    if (fabs(ab_x) < 1e-4f && fabs(ab_y) < 1e-4f && fabs(dtheta) > 1e-4f) {
      curr_segment.end = idx;
      segments.push_back(curr_segment);
      curr_segment.start = idx;
    }
  }

  curr_segment.end = path.poses.size() - 1;
  segments.push_back(curr_segment);
  return segments;
}

/**
 * @brief For a given path, update the path point orientations based on smoothing
 * @param path Path to approximate the path orientation in
 * @param reversing_segment Return if this is a reversing segment
 * @param is_holonomic Whether the motion model is holonomic (default is false)
 Only set as true when the input path is known to be generated from a holonomic planner like NavFn.
 */
inline void updateApproximatePathOrientations(
  Path & path,
  bool & reversing_segment,
  bool is_holonomic = false)
{
  float dx, dy, theta, pt_yaw;
  reversing_segment = false;

  // Find if this path segment is in reverse
  dx = path.poses[2].position.x() - path.poses[1].position.x();
  dy = path.poses[2].position.y() - path.poses[1].position.y();
  theta = atan2(dy, dx);
  pt_yaw = path.poses[1].orientation;
  if (!is_holonomic && fabs(angles::shortest_angular_distance(pt_yaw, theta)) > M_PIf32_2) {
    reversing_segment = true;
  }

  // Find the angle relative the path position vectors
  for (unsigned int i = 0; i != path.poses.size() - 1; i++) {
    dx = path.poses[i + 1].position.x() - path.poses[i].position.x();
    dy = path.poses[i + 1].position.y() - path.poses[i].position.y();
    theta = atan2(dy, dx);

    // If points are overlapping, pass
    if (fabs(dx) < 1e-4f && fabs(dy) < 1e-4f) {
      continue;
    }

    // Flip the angle if this path segment is in reverse
    if (reversing_segment) {
      theta += M_PIf32;  // orientationAroundZAxis will normalize
    }

    path.poses[i].orientation = theta;
  }
}



#endif  // NAV2_UTIL__SMOOTHER_UTILS_HPP_
