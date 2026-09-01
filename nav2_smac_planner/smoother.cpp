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

#include <ompl/base/ScopedState.h>
#include <ompl/base/spaces/DubinsStateSpace.h>

#include <chrono>
#include <memory>
#include <vector>

#include "angles/angles.h"

#include "smoother.hpp"
#include "smoother_utils.hpp"

#ifndef M_PIf32
#define M_PIf32 3.14159265358979323846f
#endif

using namespace std::chrono;  // NOLINT

Smoother::Smoother(const SmootherParams& params)
{
    // Параметры сходимости и оптимизации
    tolerance_ = params.tolerance;             // допуск сходимости алгоритма сглаживания
    max_its_ = params.max_iterations;          // максимальное количество итераций оптимизации

    // Весовые коэффициенты
    data_w_ = params.w_data;                   // вес сохранения исходных данных (привязка к оригинальному пути)
    smooth_w_ = params.w_smooth;               // вес сглаживания траектории

    // Параметры поведения
    is_holonomic_ = params.is_holonomic;       // флаг голономности робота (возможность движения в любом направлении)
    do_refinement_ = params.do_refinement;     // включение дополнительного улучшения траектории
    refinement_num_ = params.refinement_num;   // количество дополнительных проходов улучшения (рефинментов)
}

void Smoother::initialize(const float & min_turning_radius)
{
  min_turning_rad_ = min_turning_radius;
  state_space_ = std::make_unique<ompl::base::DubinsStateSpace>(min_turning_rad_);
}

bool Smoother::smooth(
  Path & path,
  const Costmap2D * costmap,
  const double & max_time)
{
  // by-pass path orientations approximation when skipping smac smoother
  if (max_its_ == 0) {
    return false;
  }

  steady_clock::time_point start = steady_clock::now();
  double time_remaining = max_time;
  bool success = true, reversing_segment;
  Path curr_path_segment;
  std::vector<PathSegment> path_segments = findDirectionalPathSegments(path,
      is_holonomic_);

  for (unsigned int i = 0; i != path_segments.size(); i++) {
    if (path_segments[i].end - path_segments[i].start > 10) {
      // Populate path segment
      curr_path_segment.poses.clear();
      std::copy(
        path.poses.begin() + path_segments[i].start,
        path.poses.begin() + path_segments[i].end + 1,
        std::back_inserter(curr_path_segment.poses));

      // Make sure we're still able to smooth with time remaining
      steady_clock::time_point now = steady_clock::now();
      time_remaining = max_time - duration_cast<duration<double>>(now - start).count();
      refinement_ctr_ = 0;

      // Smooth path segment naively
      const Pose start_pose = curr_path_segment.poses.front();
      const Pose goal_pose = curr_path_segment.poses.back();
      bool local_success =
        smoothImpl(curr_path_segment, reversing_segment, costmap, time_remaining);
      success = success && local_success;

      // Enforce boundary conditions
      if (!is_holonomic_ && local_success) {
        enforceStartBoundaryConditions(start_pose, curr_path_segment, costmap, reversing_segment);
        enforceEndBoundaryConditions(goal_pose, curr_path_segment, costmap, reversing_segment);
      }

      // Assemble the path changes to the main path
      std::copy(
        curr_path_segment.poses.begin(),
        curr_path_segment.poses.end(),
        path.poses.begin() + path_segments[i].start);
    }
  }

  return success;
}

bool Smoother::smoothImpl(
  Path & path,
  bool & reversing_segment,
  const Costmap2D * costmap,
  const double & max_time)
{
    std::chrono::steady_clock::time_point a = std::chrono::steady_clock::now();
    std::chrono::duration<double> max_dur = std::chrono::duration<double>(max_time);

  int its = 0;
  float change = tolerance_;
  const unsigned int & path_size = path.poses.size();
  float x_i, y_i, y_m1, y_ip1, y_i_org;
  unsigned int mx, my;

  Path new_path = path;
  Path last_path = path;

  while (change >= tolerance_) {
    its += 1;
    change = 0.0f;

    if (its >= max_its_) {
      std::cout << "SmacPlannerSmoother: Number of iterations has exceeded limit of "
                << max_its_ << "." << std::endl;
      path = last_path;
      updateApproximatePathOrientations(path, reversing_segment, is_holonomic_);
      return false;
    }

    // Make sure still have time left to process
    std::chrono::steady_clock::time_point b = std::chrono::steady_clock::now();
    std::chrono::duration<double> timespan = b - a;

    if (timespan > max_dur) {
      std::cout << "SmacPlannerSmoother: Smoothing time exceeded allowed duration of "
                << max_time << "." << std::endl;
      path = last_path;
      updateApproximatePathOrientations(path, reversing_segment, is_holonomic_);
      return false;
    }

    for (unsigned int i = 1; i != path_size - 1; i++) {
      for (unsigned int j = 0; j != 2; j++) {
        x_i = getFieldByDim(path.poses[i], j);
        y_i = getFieldByDim(new_path.poses[i], j);
        y_m1 = getFieldByDim(new_path.poses[i - 1], j);
        y_ip1 = getFieldByDim(new_path.poses[i + 1], j);
        y_i_org = y_i;

        // Smooth based on local 3 point neighborhood and original data locations
        y_i += data_w_ * (x_i - y_i) + smooth_w_ * (y_ip1 + y_m1 - (2.0f * y_i));
        setFieldByDim(new_path.poses[i], j, y_i);
        change += abs(y_i - y_i_org);
      }

      // validate update is admissible, only checks cost if a valid costmap pointer is provided
      float cost = 0.0f;
      mx = getFieldByDim(new_path.poses[i], 0);
      my = getFieldByDim(new_path.poses[i], 1);
      if (costmap) { costmap->PosCheck(mx, my);
        cost = static_cast<float>(costmap->getCost(mx, my));
      }

      if (cost > MAX_NON_OBSTACLE_COST && cost != UNKNOWN_COST) {
        std::cout << "SmacPlannerSmoother: Smoothing process resulted in an infeasible collision. "
                  << "Returning the last path before the infeasibility was introduced." << std::endl;
        path = last_path;
        updateApproximatePathOrientations(path, reversing_segment, is_holonomic_);
        return false;
      }
    }

    last_path = new_path;
  }

  // Let's do additional refinement, it shouldn't take more than a couple milliseconds
  // but really puts the path quality over the top.
  if (do_refinement_ && refinement_ctr_ < refinement_num_) {
    refinement_ctr_++;
    smoothImpl(new_path, reversing_segment, costmap, max_time);
  }

  updateApproximatePathOrientations(new_path, reversing_segment, is_holonomic_);
  path = new_path;
  return true;
}

float Smoother::getFieldByDim(
  const Pose & msg, const unsigned int & dim)
{
  if (dim == 0) {
    return msg.position.x();
  } else {
    return msg.position.y();
  }
}

void Smoother::setFieldByDim(
  Pose & msg, const unsigned int dim,
  const float & value)
{
  if (dim == 0) {
    msg.position.x() = value;
  } else{
    msg.position.y() = value;
  }
}

unsigned int Smoother::findShortestBoundaryExpansionIdx(
  const BoundaryExpansions & boundary_expansions)
{
  // Check which is valid with the minimum integrated length such that
  // shorter end-points away that are infeasible to achieve without
  // a loop-de-loop are punished
  float min_length = 1e9f;
  int shortest_boundary_expansion_idx = 1e9;
  for (unsigned int idx = 0; idx != boundary_expansions.size(); idx++) {
    if (boundary_expansions[idx].expansion_path_length<min_length &&
      !boundary_expansions[idx].in_collision &&
      boundary_expansions[idx].path_end_idx>0.0f &&
      boundary_expansions[idx].expansion_path_length > 0.0f)
    {
      min_length = boundary_expansions[idx].expansion_path_length;
      shortest_boundary_expansion_idx = idx;
    }
  }

  return shortest_boundary_expansion_idx;
}

void Smoother::findBoundaryExpansion(
  const Pose & start,
  const Pose & end,
  BoundaryExpansion & expansion,
  const Costmap2D * costmap)
{
  static ompl::base::ScopedState<> from(state_space_), to(state_space_), s(state_space_);

  from[0] = start.position.x();
   from[1] = start.position.y();
   from[2] = start.orientation;
   to[0] = end.position.x();
   to[1] = end.position.y();
   to[2] = end.orientation;

  float d = state_space_->distance(from(), to());
  // If this path is too long compared to the original, then this is probably
  // a loop-de-loop, treat as invalid as to not deviate too far from the original path.
  // 2.0 selected from prinicipled choice of boundary test points
  // r, 2 * r, r * PI, and 2 * PI * r. If there is a loop, it will be
  // approximately 2 * PI * r, which is 2 * PI > r, PI > 2 * r, and 2 > r * PI.
  // For all but the last backup test point, a loop would be approximately
  // 2x greater than any of the selections.
  if (d > 2.0f * expansion.original_path_length) {
    return;
  }

  std::vector<float> reals;
  float theta(0.0f), x(0.0f), y(0.0f);
  float x_m = start.position.x();
  float y_m = start.position.y();

  // Get intermediary poses
  for (float i = 0; i <= expansion.path_end_idx; i++) {
    state_space_->interpolate(from(), to(), i / expansion.path_end_idx, s());
    reals = s.reals();
    // Make sure in range [0, 2PI)
    theta = (reals[2] < 0.0f) ? (reals[2] + 2.0f * M_PIf32) : reals[2];
    theta = (theta > 2.0f * M_PIf32) ? (theta - 2.0f * M_PIf32) : theta;
    x = reals[0];
    y = reals[1];

    // Check for collision
    costmap->PosCheck(x, y);
    if (static_cast<float>(costmap->getCost(x, y)) >= INSCRIBED_COST) {
      expansion.in_collision = true;
    }

    // Integrate path length
    expansion.expansion_path_length += hypot(x - x_m, y - y_m);
    x_m = x;
    y_m = y;

    // Store point
    expansion.pts.emplace_back(x, y, theta);
  }
}

template<typename IteratorT>
BoundaryExpansions Smoother::generateBoundaryExpansionPoints(IteratorT start, IteratorT end)
{
  std::vector<float> distances = {
    min_turning_rad_,  // Radius
    2.0f * min_turning_rad_,  // Diameter
    M_PIf32 * min_turning_rad_,  // 50% Circumference
    2.0f * M_PIf32 * min_turning_rad_  // Circumference
  };

  BoundaryExpansions boundary_expansions;
  boundary_expansions.resize(distances.size());
  float curr_dist = 0.0f;
  float x_last = start->position.x();
  float y_last = start->position.y();
  Eigen::Vector2f pt;
  unsigned int curr_dist_idx = 0;

  for (IteratorT iter = start; iter != end; iter++) {
    pt = iter->position;
    curr_dist += hypot(pt.x() - x_last, pt.y() - y_last);
    x_last = pt.x();
    y_last = pt.y();

    if (curr_dist >= distances[curr_dist_idx]) {
      boundary_expansions[curr_dist_idx].path_end_idx = iter - start;
      boundary_expansions[curr_dist_idx].original_path_length = curr_dist;
      curr_dist_idx++;
    }

    if (curr_dist_idx == boundary_expansions.size()) {
      break;
    }
  }

  return boundary_expansions;
}

void Smoother::enforceStartBoundaryConditions(
  const Pose & start_pose,
  Path & path,
  const Costmap2D * costmap,
  const bool & reversing_segment)
{
  // Find range of points for testing
  BoundaryExpansions boundary_expansions =
    generateBoundaryExpansionPoints<PathIterator>(path.poses.begin(), path.poses.end());

  // Generate the motion model and metadata from start -> test points
  for (unsigned int i = 0; i != boundary_expansions.size(); i++) {
    BoundaryExpansion & expansion = boundary_expansions[i];
    if (expansion.path_end_idx == 0.0f) {
      continue;
    }

    if (!reversing_segment) {
      findBoundaryExpansion(
        start_pose, path.poses[expansion.path_end_idx], expansion,
        costmap);
    } else {
      findBoundaryExpansion(
        path.poses[expansion.path_end_idx], start_pose, expansion,
        costmap);
    }
  }

  // Find the shortest kinematically feasible boundary expansion
  unsigned int best_expansion_idx = findShortestBoundaryExpansionIdx(boundary_expansions);
  if (best_expansion_idx > boundary_expansions.size()) {
    return;
  }

  // Override values to match curve
  BoundaryExpansion & best_expansion = boundary_expansions[best_expansion_idx];
  if (reversing_segment) {
    std::reverse(best_expansion.pts.begin(), best_expansion.pts.end());
  }
  for (unsigned int i = 0; i != best_expansion.pts.size(); i++) {
    path.poses[i].position.x() = best_expansion.pts[i].x;
    path.poses[i].position.y() = best_expansion.pts[i].y;
    path.poses[i].orientation = best_expansion.pts[i].theta;
  }
}

void Smoother::enforceEndBoundaryConditions(
  const Pose & end_pose,
  Path & path,
  const Costmap2D * costmap,
  const bool & reversing_segment)
{
  // Find range of points for testing
  BoundaryExpansions boundary_expansions =
    generateBoundaryExpansionPoints<ReversePathIterator>(path.poses.rbegin(), path.poses.rend());

  // Generate the motion model and metadata from start -> test points
  unsigned int expansion_starting_idx;
  for (unsigned int i = 0; i != boundary_expansions.size(); i++) {
    BoundaryExpansion & expansion = boundary_expansions[i];
    if (expansion.path_end_idx == 0.0f) {
      continue;
    }
    expansion_starting_idx = path.poses.size() - expansion.path_end_idx - 1;
    if (!reversing_segment) {
      findBoundaryExpansion(path.poses[expansion_starting_idx], end_pose, expansion, costmap);
    } else {
      findBoundaryExpansion(end_pose, path.poses[expansion_starting_idx], expansion, costmap);
    }
  }

  // Find the shortest kinematically feasible boundary expansion
  unsigned int best_expansion_idx = findShortestBoundaryExpansionIdx(boundary_expansions);
  if (best_expansion_idx > boundary_expansions.size()) {
    return;
  }

  // Override values to match curve
  BoundaryExpansion & best_expansion = boundary_expansions[best_expansion_idx];
  if (reversing_segment) {
    std::reverse(best_expansion.pts.begin(), best_expansion.pts.end());
  }
  expansion_starting_idx = path.poses.size() - best_expansion.path_end_idx - 1;
  for (unsigned int i = 0; i != best_expansion.pts.size(); i++) {
    path.poses[expansion_starting_idx + i].position.x() = best_expansion.pts[i].x;
    path.poses[expansion_starting_idx + i].position.y() = best_expansion.pts[i].y;
    path.poses[expansion_starting_idx + i].orientation = best_expansion.pts[i].theta;
  }
}


