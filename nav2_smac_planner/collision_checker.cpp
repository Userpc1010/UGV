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

#include "collision_checker.hpp"
#include "iostream"

#include <Eigen/Core>
#include <Eigen/Geometry>

#ifndef M_PIf32
#define M_PIf32 3.14159265358979323846f
#endif


CollisionChecker::CollisionChecker( std::shared_ptr<Costmap2D_Master> costmap_master, unsigned int num_quantizations)
: FootprintCollisionChecker(costmap_master ? costmap_master->getCostmap() : nullptr)
{

  if (costmap_master) {
      costmap_master_ = costmap_master;
   }

  // Convert number of regular bins into angles
  float bin_size = 2 * M_PIf32 / static_cast<float>(num_quantizations);
  angles_.reserve(num_quantizations);
  for (unsigned int i = 0; i != num_quantizations; i++) {
    angles_.push_back(bin_size * i);
  }

  std::cout<<"SmacPlannerCollisionChecker"<<std::endl;
}

// GridCollisionChecker::GridCollisionChecker(
//   nav2_costmap_2d::Costmap2D * costmap,
//   std::vector<float> & angles)
// : FootprintCollisionChecker(costmap),
//   angles_(angles)
// {
// }

//void CollisionChecker::setFootprint(
//  const Footprint & footprint,
//  const bool & radius,
//  const float & possible_collision_cost)
//{
//  possible_collision_cost_ = static_cast<float>(possible_collision_cost);
//  if (possible_collision_cost_ <= 0.0f) {
//    std::cout << "Inflation layer either not found or inflation is not set sufficiently for "
//                 "optimized non-circular collision checking capabilities. It is HIGHLY recommended to set"
//                 " the inflation radius to be at MINIMUM half of the robot's largest cross-section. See "
//                 "github.com/ros-planning/navigation2/tree/main/nav2_smac_planner#potential-fields"
//                 " for full instructions. This will substantially impact run-time performance." << std::endl;
//  }

//  footprint_is_radius_ = radius;

//  // Use radius, no caching required
//  if (radius) {
//    return;
//  }

//  // No change, no updates required
//  if (footprint == unoriented_footprint_) {
//    return;
//  }

//  oriented_footprints_.clear();
//  oriented_footprints_.reserve(angles_.size());
//  float sin_th, cos_th;

//  Eigen::Vector2f new_pt;
//  const unsigned int footprint_size = footprint.size();

//  // Precompute the orientation bins for checking to use
//  for (unsigned int i = 0; i != angles_.size(); i++) {
//    sin_th = sin(angles_[i]);
//    cos_th = cos(angles_[i]);
//    Footprint oriented_footprint;
//    oriented_footprint.reserve(footprint_size);

//    for (unsigned int j = 0; j < footprint_size; j++) {
//      new_pt.x() = footprint[j].x() * cos_th - footprint[j].y() * sin_th;
//      new_pt.y() = footprint[j].x() * sin_th + footprint[j].y() * cos_th;
//      oriented_footprint.push_back(new_pt);
//    }

//    oriented_footprints_.push_back(oriented_footprint);
//  }

//  unoriented_footprint_ = footprint;
//}

void CollisionChecker::setFootprint(
  const std::vector<Eigen::Vector2f> & footprint,
  const bool & radius,
  const float & possible_collision_cost)
{
    possible_collision_cost_ = static_cast<float>(possible_collision_cost);
    if (possible_collision_cost_ <= 0.0f) {
      std::cout << "Inflation layer either not found or inflation is not set sufficiently for "
                   "optimized non-circular collision checking capabilities. It is HIGHLY recommended to set"
                   " the inflation radius to be at MINIMUM half of the robot's largest cross-section. See "
                   "github.com/ros-planning/navigation2/tree/main/nav2_smac_planner#potential-fields"
                   " for full instructions. This will substantially impact run-time performance." << std::endl;
    }

  footprint_is_radius_ = radius;

  if (radius) return;


  if (oriented_footprints_ != nullptr) {
    delete[] oriented_footprints_;
  }
  if (unoriented_footprint_ != nullptr) {
    delete[] unoriented_footprint_;
  }

  footprint_size = footprint.size();
  unsigned int num_angles = angles_.size();


  oriented_footprints_ = new Footprint[num_angles * footprint_size];
  unoriented_footprint_ = new Footprint[footprint_size];


  for (unsigned int j = 0; j < footprint_size; j++) {
    unoriented_footprint_[j].x = static_cast<uint16_t>(footprint[j].x());
    unoriented_footprint_[j].y = static_cast<uint16_t>(footprint[j].y());
  }

  // 4. Предвычисление поворотов
  for (unsigned int i = 0; i < num_angles; i++) {
    float sin_th = sin(angles_[i]);
    float cos_th = cos(angles_[i]);

    // Вычисляем смещение для текущего угла в плоском массиве
    unsigned int angle_offset = i * footprint_size;

    for (unsigned int j = 0; j < footprint_size; j++) {
      float fx = footprint[j].x();
      float fy = footprint[j].y();

      // Пишем по индексу с учетом смещения угла
      oriented_footprints_[angle_offset + j].x = static_cast<uint16_t>(fx * cos_th - fy * sin_th);
      oriented_footprints_[angle_offset + j].y = static_cast<uint16_t>(fx * sin_th + fy * cos_th);
    }
  }
}



bool CollisionChecker::inCollision(
  const float & x,
  const float & y,
  const float & angle_bin,
  const bool & traverse_unknown)
{
  // Check to make sure cell is inside the map
  if (outsideRange(costmap_->getSizeInCellsX(), x) ||
    outsideRange(costmap_->getSizeInCellsY(), y))
  {

    return true;
  }

  // Assumes setFootprint already set
  center_cost_ = static_cast<float>(costmap_->getCost( static_cast<unsigned int>(x + 0.5f), static_cast<unsigned int>(y + 0.5f)));

  if (!footprint_is_radius_) {
    // if footprint, then we check for the footprint's points, but first see
    // if the robot is even potentially in an inscribed collision
    if (center_cost_ < possible_collision_cost_ && possible_collision_cost_ > 0.0f) {

      return false;
    }

    // If its inscribed, in collision, or unknown in the middle,
    // no need to even check the footprint, its invalid
    if (center_cost_ == UNKNOWN_COST && !traverse_unknown) {

      return true;
    }

    if (center_cost_ == INSCRIBED_COST || center_cost_ == OCCUPIED_COST) {

      return true;
    }

    // if possible inscribed, need to check actual footprint pose.
    // Use precomputed oriented footprints are done on initialization,
    // offset by translation value to collision check
//    float wx, wy;
//    costmap_->mapToWorld(x, y, wx, wy);

    // Вычисляем индекс начала данных для нужного угла
    unsigned int angle_index = static_cast<unsigned int>(angle_bin);
    unsigned int offset = angle_index * footprint_size;

    // Передаем адрес ПЕРВОЙ точки этого угла в функцию
    float footprint_cost = footprintCost(x, y, &oriented_footprints_[offset], footprint_size);


    if (footprint_cost == UNKNOWN_COST && traverse_unknown) {

      return false;
    }

    // if occupied or unknown and not to traverse unknown space
    return footprint_cost >= OCCUPIED_COST;
  } else {
    // if radius, then we can check the center of the cost assuming inflation is used
    if (center_cost_ == UNKNOWN_COST && traverse_unknown) {
    }

    // if occupied or unknown and not to traverse unknown space
    return center_cost_ >= INSCRIBED_COST;
  }
}

bool CollisionChecker::inCollision(
  const unsigned int & i,
  const bool & traverse_unknown)
{
  center_cost_ = costmap_->getCost(i);
  if (center_cost_ == UNKNOWN_COST && traverse_unknown) {
    return false;
  }

  // if occupied or unknown and not to traverse unknown space
  return center_cost_ >= INSCRIBED_COST;
}

float CollisionChecker::getCost()
{
  // Assumes inCollision called prior
  return static_cast<float>(center_cost_);
}

bool CollisionChecker::outsideRange(const unsigned int & max, const float & value)
{
  return value < 0.0f || value > max;
}


