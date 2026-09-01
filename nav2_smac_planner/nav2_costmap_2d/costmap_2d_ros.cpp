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

#include "nav2_costmap_2d/costmap_2d_ros.hpp"

#include <memory>
#include <chrono>
#include <string>
#include <vector>
#include <utility>
#include <iostream>
#include <QDebug>

#include "nav2_costmap_2d/cost_values.hpp"


Costmap2D_Master::Costmap2D_Master()
{
}

Costmap2D_Master::~Costmap2D_Master()
{
}

bool Costmap2D_Master::on_configure(const CostmapConfig& config)
{
    std::cout << "Configuring Costmap..." << std::endl;
    try {
      initParameters(config);
    } catch (const std::exception & e) {
      std::cout << "ERROR: Failed to configure costmap! " << e.what() << std::endl;
      return 0;
    }

  // Create the costmap itself
  master_costmap_ = std::make_unique<Costmap2D>((unsigned int)(map_width_meters_),(unsigned int)(map_height_meters_), resolution_, origin_x_, origin_y_);

  return 1;
}

void Costmap2D_Master::initParameters(const CostmapConfig& config)
{
  std::cout << "[DEBUG] Initializing parameters from configuration" << std::endl;

  // Присваиваем все параметры из структуры напрямую
  map_vis_z_ = config.map_vis_z;
  footprint_ = config.footprint;
  footprint_padding_ = config.footprint_padding;
  global_frame_ = config.global_frame;
  map_height_meters_ = config.height;
  map_width_meters_ = config.width;
  origin_x_ = config.origin_x;
  origin_y_ = config.origin_y;
  resolution_ = config.resolution;
  robot_base_frame_ = config.robot_base_frame;
  robot_radius_ = config.robot_radius;
  rolling_window_ = config.rolling_window;
  track_unknown_space_ = config.track_unknown_space;
  transform_tolerance_ = config.transform_tolerance;
  initial_transform_timeout_ = config.initial_transform_timeout;
  cost_scaling_factor_ = config.cost_scaling_factor;
  InflationRadius_ = config.inflation_radius;

  subscribe_to_stamped_footprint_ = config.subscribe_to_stamped_footprint;

  // Обработка footprint
  use_radius_ = true;

  if (footprint_ != "" && footprint_ != "[]") {
    std::cout << "[INFO] Processing custom footprint: " << footprint_ << std::endl;
    std::vector<Eigen::Vector2f> new_footprint;

    if (makeFootprintFromString(footprint_, new_footprint)) {
      use_radius_ = false;

      std::cout << "[INFO] Using custom polygonal footprint" << std::endl;

      std::cout << "[DEBUG] Parsed footprint points:" << std::endl;
      for (auto& p : new_footprint) {
          std::cout << "  (" << p.x() << ", " << p.y() << ")" << std::endl;
      }
      setRobotFootprint(new_footprint);

    } else {
      std::cout << "[ERROR] The footprint parameter is invalid: \"" << footprint_ << "\", using radius (" << robot_radius_ << ") instead" << std::endl;
    }

  } else {
    std::cout << "[INFO] Using circular footprint with radius: " << robot_radius_ << std::endl;

    std::vector<Eigen::Vector2f> new_footprint = makeFootprintFromRadius(robot_radius_);

    std::cout << "[DEBUG] Parsed footprint points:" << std::endl;
    for (auto& p : new_footprint) {
        std::cout << "  (" << p.x() << ", " << p.y() << ")" << std::endl;
    }
    setRobotFootprint(new_footprint);
  }


  // Проверка размеров карты
  if (map_width_meters_ <= 0) {
    std::cout << "[ERROR] Map width cannot be negative or zero. Please provide a positive value." << std::endl;
    throw std::invalid_argument("Invalid map width");
  }

  if (map_height_meters_ <= 0) {
    std::cout << "[ERROR] Map height cannot be negative or zero. Please provide a positive value." << std::endl;
    throw std::invalid_argument("Invalid map height");
  }

  // Логирование итоговых параметров
  std::cout << "[INFO] Parameters initialized successfully:" << std::endl;
  std::cout << "  - Map size: " << map_width_meters_ << "x" << map_height_meters_ << " cells " << std::endl;
  std::cout << "  - Resolution: " << resolution_ << " m/cell" << std::endl;
  std::cout << "  - Robot radius: " << robot_radius_ << " cell " << std::endl;
  std::cout << "  - Global frame: " << global_frame_ << std::endl;
}


void
Costmap2D_Master::setRobotFootprint(const std::vector<Eigen::Vector2f> & points)
{
  unpadded_footprint_ = points;
  padded_footprint_ = points;
  padFootprint(padded_footprint_, footprint_padding_);
  setFootprint(padded_footprint_);
}

void
Costmap2D_Master::setRobotFootprintPolygon(
  const Polygon & footprint)
{
  setRobotFootprint(toPointVector(footprint));
}

void Costmap2D_Master::setFootprint(const std::vector<Eigen::Vector2f> & footprint_spec)
{
  std::pair<float, float> inside_outside = calculateMinAndMaxDistances(footprint_spec);
  // use atomic store here since footprint is used by various planners/controllers
  // and not otherwise locked

  inscribed_radius_ = std::get<0>(inside_outside);
  circumscribed_radius_ = std::get<1>(inside_outside);
}

unsigned char Costmap2D_Master::inflation_layer_computeCost(float distance) const
{
  unsigned char cost = 0;
  if (distance == 0) {
      cost = LETHAL_OBSTACLE;
  } else if (distance * resolution_ <= inscribed_radius_) {
   cost = INSCRIBED_INFLATED_OBSTACLE;
  } else {
  // make sure cost falls off by Euclidean distance

  qDebug()<<"cost_scaling_factor_"<<cost_scaling_factor_<<" resolution_ "<<resolution_<<" inscribed_radius_ "<<inscribed_radius_<<" distance "<<distance;
  float factor =
  exp(-1.0f * cost_scaling_factor_ * (distance * resolution_ - inscribed_radius_));
  cost = static_cast<unsigned char>((INSCRIBED_INFLATED_OBSTACLE - 1) * factor);
  }
  return cost;
}


void Costmap2D_Master::updateMap(const uint8_t * CostMap)
{
  master_costmap_ ->setMap(CostMap);
}
