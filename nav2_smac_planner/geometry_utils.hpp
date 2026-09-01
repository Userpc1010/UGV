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

#ifndef NAV2_UTIL__GEOMETRY_UTILS_HPP_
#define NAV2_UTIL__GEOMETRY_UTILS_HPP_

#include <cmath>
#include <vector>
#include <datatypes_smac_planner.h>

/**
 * @brief Get the euclidean distance between 2 geometry_msgs::Points
 * @param pos1 First point
 * @param pos1 Second point
 * @param is_3d True if a true L2 distance is desired (default false)
 * @return double L2 distance
 */
inline float euclidean_distance(
  const Eigen::Vector2f & pos1,
  const Eigen::Vector2f & pos2)
{
  float dx = pos1.x() - pos2.x();
  float dy = pos1.y() - pos2.y();

  return std::hypot(dx, dy);
}

/**
 * @brief Get the L2 distance between 2 geometry_msgs::Poses
 * @param pos1 First pose
 * @param pos1 Second pose
 * @param is_3d True if a true L2 distance is desired (default false)
 * @return double euclidean distance
 */
inline float euclidean_distance(
  const Pose& pos1,
  const Pose& pos2)
{
  float dx = pos1.position.x() - pos2.position.x();
  float dy = pos1.position.y() - pos2.position.y();

  return std::hypot(dx, dy);
}

/**
 * Find element in iterator with the minimum calculated value
 */
template<typename Iter, typename Getter>
inline Iter min_by(Iter begin, Iter end, Getter getCompareVal)
{
  if (begin == end) {
    return end;
  }
  auto lowest = getCompareVal(*begin);
  Iter lowest_it = begin;
  for (Iter it = ++begin; it != end; ++it) {
    auto comp = getCompareVal(*it);
    if (comp <= lowest) {
      lowest = comp;
      lowest_it = it;
    }
  }
  return lowest_it;
}

/**
 * Find first element in iterator that is greater integrated distance than comparevalue
 */
template<typename Iter, typename Getter>
inline Iter first_after_integrated_distance(Iter begin, Iter end, Getter getCompareVal)
{
  if (begin == end) {
    return end;
  }
  Getter dist = 0.0f;
  for (Iter it = begin; it != end - 1; it++) {
    dist += euclidean_distance(*it, *(it + 1));
    if (dist > getCompareVal) {
      return it + 1;
    }
  }
  return end;
}

/**
 * @brief Calculate the length of the provided path, starting at the provided index
 * @param path Path containing the poses that are planned
 * @param start_index Optional argument specifying the starting index for
 * the calculation of path length. Provide this if you want to calculate length of a
 * subset of the path.
 * @return double Path length
 */
inline float calculate_path_length(const Path & path, size_t start_index = 0)
{
  if (start_index + 1 >= path.poses.size()) {
    return 0.0f;
  }
  float path_length = 0.0f;
  for (size_t idx = start_index; idx < path.poses.size() - 1; ++idx) {
    path_length += euclidean_distance(path.poses[idx], path.poses[idx + 1]);
  }
  return path_length;
}

/**
 * @brief Find the index of the first goal in `PENDING` status matching the
 * given target pose.
 * @param waypoint_statuses List of waypoint statuses to search through.
 * @param goal Target pose to match against waypoint goals.
 * @return Index of the first matching goal in PENDING status, -1 if not found.
 */
//inline int find_next_matching_goal_in_waypoint_statuses(
//  const std::vector<nav2_msgs::msg::WaypointStatus> & waypoint_statuses,
//  const geometry_msgs::msg::PoseStamped & goal)
//{
//  auto itr = std::find_if(waypoint_statuses.begin(), waypoint_statuses.end(),
//      [&goal](const nav2_msgs::msg::WaypointStatus & status){
//        return status.waypoint_pose == goal &&
//               status.waypoint_status == nav2_msgs::msg::WaypointStatus::PENDING;
//    });

//  if (itr == waypoint_statuses.end()) {
//    return -1;
//  }

//  return itr - waypoint_statuses.begin();
//}

/**
 * @brief Checks if point is inside the polygon
 * @param px X-coordinate of the given point to check
 * @param py Y-coordinate of the given point to check
 * @param polygon Polygon to check if the point is inside
 * @return True if given point is inside polygon, otherwise false
 */
template<class PointT>
inline bool isPointInsidePolygon(
  const float px, const float py, const std::vector<PointT> & polygon)
{
  // Adaptation of Shimrat, Moshe. "Algorithm 112: position of point relative to polygon."
  // Communications of the ACM 5.8 (1962): 434.
  // Implementation of ray crossings algorithm for point in polygon task solving.
  // Y coordinate is fixed. Moving the ray on X+ axis starting from given point.
  // Odd number of intersections with polygon boundaries means the point is inside polygon.
  const int points_num = polygon.size();
  int i, j;  // Polygon vertex iterators
  bool res = false;  // Final result, initialized with already inverted value

  // Starting from the edge where the last point of polygon is connected to the first
  i = points_num - 1;
  for (j = 0; j < points_num; j++) {
    // Checking the edge only if given point is between edge boundaries by Y coordinates.
    // One of the condition should contain equality in order to exclude the edges
    // parallel to X+ ray.
    if ((py <= polygon[i].y) == (py > polygon[j].y)) {
      // Calculating the intersection coordinate of X+ ray
      const float x_inter = polygon[i].x +
        (py - polygon[i].y) *
        (polygon[j].x - polygon[i].x) /
        (polygon[j].y - polygon[i].y);
      // If intersection with checked edge is greater than point x coordinate,
      // inverting the result
      if (x_inter > px) {
        res = !res;
      }
    }
    i = j;
  }
  return res;
}

/**
 * @brief Computes the shortest (perpendicular) distance from a point to a path segment.
 *
 * Given a point and a line segment defined by two poses, calculates the minimum distance
 * from the point to the segment in the XY plane. If the segment is too short,
 * returns the distance from the point to the segment's start.
 *
 * See: https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
 *
 * @param point The point to measure from (geometry_msgs::msg::Point).
 * @param start The starting pose of the segment (geometry_msgs::msg::Pose).
 * @param end The ending pose of the segment (geometry_msgs::msg::Pose).
 * @return The shortest distance from the point to the segment in meters.
 */
inline float distance_to_path_segment(
  const Eigen::Vector2f & point,
  const Pose & start,
  const Pose & end)
{
  const auto & p = point;
  const auto & a = start.position;
  const auto & b = end.position;

  const float dx_seg = b.x() - a.x();
  const float dy_seg = b.y() - a.y();

  const float seg_len_sq = (dx_seg * dx_seg) + (dy_seg * dy_seg);

  if (seg_len_sq <= 1e-7f) {
    return euclidean_distance(point, a);
  }

  const float dot = ((p.x() - a.x()) * dx_seg) + ((p.y() - a.y()) * dy_seg);
  const float t = std::clamp(dot / seg_len_sq, 0.0f, 1.0f);

  const float proj_x = a.x() + t * dx_seg;
  const float proj_y = a.y() + t * dy_seg;

  const float dx_proj = p.x() - proj_x;
  const float dy_proj = p.y() - proj_y;
  return std::hypot(dx_proj, dy_proj);
}

/**
 * @brief Computes the 2D cross product between the vector from start to end and the vector from start to point.
 * The sign of this calculation's result can be used to determine which side are you on of the track.
 *
 * See: https://en.wikipedia.org/wiki/Cross_product
 *
 * @param point The point to check relative to the segment.
 * @param start The starting pose of the segment.
 * @param end The ending pose of the segment.
 * @return The signed 2D cross product value.
 */
inline float cross_product_2d(
  const Eigen::Vector2f & point,
  const Pose & start,
  const Pose & end)
{
  const auto & p = point;
  const auto & a = start.position;
  const auto & b = end.position;

  const float path_vec_x = b.x() - a.x();
  const float path_vec_y = b.y() - a.y();

  const float robot_vec_x = p.x() - a.x();
  const float robot_vec_y = p.y() - a.y();

  return (path_vec_x * robot_vec_y) - (path_vec_y * robot_vec_x);
}



#endif  // NAV2_UTIL__GEOMETRY_UTILS_HPP_
