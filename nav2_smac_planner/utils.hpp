// Copyright (c) 2021, Samsung Research America
// Copyright (c) 2023, Open Navigation LLC
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

#ifndef NAV2_SMAC_PLANNER__UTILS_HPP_
#define NAV2_SMAC_PLANNER__UTILS_HPP_

#include <vector>
#include <memory>
#include <cmath>
#include <string>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <QDebug>

#include "nav2_costmap_2d/costmap_2d.hpp"
#include "nav2_costmap_2d/costmap_2d_ros.hpp"
#include "nlohmann/json.hpp"
#include "types.hpp"

#ifndef M_PIf32
#define M_PIf32 3.14159265358979323846f
#endif

/**
* @brief Create an Eigen Vector2D of world poses from continuous map coords
* @param mx float of map X coordinate
* @param my float of map Y coordinate
* @param costmap Costmap pointer
* @return Eigen::Vector2d eigen vector of the generated path
*/
//inline Eigen::Vector3f getWorldCoords(
//  const float & mx, const float & my, const Costmap2D * costmap)
//{
//  Eigen::Vector3f msg;
//  msg.x() = static_cast<float>(costmap->getOriginX()) + mx * costmap->getResolution();
//  msg.y() = static_cast<float>(costmap->getOriginY()) + my * costmap->getResolution();
//  return msg;
//}
inline Eigen::Vector2f getCoords(
  const float & mx, const float & my)
{
  Eigen::Vector2f msg;
  msg.x() = mx;
  msg.y() = my;
  return msg;
}

/**
* @brief Find the min cost of the inflation decay function for which the robot MAY be
* in collision in any orientation
* @param costmap Costmap2D to get minimum inscribed cost (e.g. 128 in inflation layer documentation)
* @return float circumscribed cost, any higher than this and need to do full footprint collision checking
* since some element of the robot could be in collision
*/
inline float findCircumscribedCost(std::shared_ptr<Costmap2D_Master> costmap)
{
  float result = -1.0f;

  // check if the costmap has an inflation layer

  if (costmap->getInflationRadius() > 0.1f) {
    float circum_radius = costmap->getCircumscribedRadius();
    float inflation_radius = costmap->getInflationRadius();

    if (inflation_radius < circum_radius) {
      std::cout << "ERROR: The inflation radius (" << inflation_radius
                << ") is smaller than the circumscribed radius (" << circum_radius << "). "
                << "If this is an SE2-collision checking plugin, it cannot use costmap potential "
                << "field to speed up collision checking by only checking the full footprint "
                << "when robot is within possibly-inscribed radius of an obstacle. This may "
                << "significantly slow down planning times!" << std::endl;
      result = 0.0f;
      return result;
    }
    result = costmap->inflation_layer_computeCost(circum_radius);
    } else {
    std::cout << "WARNING: No inflation layer found in costmap configuration. "
              << "If this is an SE2-collision checking plugin, it cannot use costmap potential "
              << "field to speed up collision checking by only checking the full footprint "
              << "when robot is within possibly-inscribed radius of an obstacle. This may "
              << "significantly slow down planning times!" << std::endl;
    }

  return result;
}



/**
 * @brief convert json to lattice metadata
 * @param[in] json json object
 * @param[out] lattice meta data
 */
inline void fromJsonToMetaData(const nlohmann::json & json, LatticeMetadata & lattice_metadata)
{
  json.at("turning_radius").get_to(lattice_metadata.min_turning_radius);
  json.at("grid_resolution").get_to(lattice_metadata.grid_resolution);
  json.at("num_of_headings").get_to(lattice_metadata.number_of_headings);
  json.at("heading_angles").get_to(lattice_metadata.heading_angles);
  json.at("number_of_trajectories").get_to(lattice_metadata.number_of_trajectories);
  json.at("motion_model").get_to(lattice_metadata.motion_model);
}

/**
 * @brief convert json to pose
 * @param[in] json json object
 * @param[out] pose
 */
inline void fromJsonToPose(const nlohmann::json & json, MotionPose & pose)
{
  pose._x = json[0];
  pose._y = json[1];
  pose._theta = json[2];
}

/**
 * @brief convert json to motion primitive
 * @param[in] json json object
 * @param[out] motion primitive
 */
inline void fromJsonToMotionPrimitive(
  const nlohmann::json & json, MotionPrimitive & motion_primitive)
{
  json.at("trajectory_id").get_to(motion_primitive.trajectory_id);
  json.at("start_angle_index").get_to(motion_primitive.start_angle);
  json.at("end_angle_index").get_to(motion_primitive.end_angle);
  json.at("trajectory_radius").get_to(motion_primitive.turning_radius);
  json.at("trajectory_length").get_to(motion_primitive.trajectory_length);
  json.at("arc_length").get_to(motion_primitive.arc_length);
  json.at("straight_length").get_to(motion_primitive.straight_length);
  json.at("left_turn").get_to(motion_primitive.left_turn);

  for (unsigned int i = 0; i < json["poses"].size(); i++) {
    MotionPose pose;
    fromJsonToPose(json["poses"][i], pose);
    motion_primitive.poses.push_back(pose);
  }
}


//Сравнение углов рысканья

inline bool orientationsEqual(float start_yaw, float goal_yaw, float epsilon = 1e-6f) {
    // Сравниваем разницу углов по модулю
    float diff = std::abs(start_yaw - goal_yaw);

    // Учитываем циклическую природу углов (2π)
    diff = std::min(diff, 2 * M_PIf32 - diff);

    return diff < epsilon;
}

///**
// * @brief transform footprint into edges
// * @param[in] rot orientation quaternion
// * @param[in] pos robot position
// * @param[in] footprint robot footprint
// * @param[out] robot footprint edges
// */
//inline std::vector<Eigen::Vector2f> transformFootprintToEdges( const Eigen::Quaternionf & rot, const Eigen::Vector2f & pos, const std::vector<Eigen::Vector2f> & footprint)
//{
//  // Создаем матрицу поворота 3x3 из кватерниона
//  Eigen::Matrix3f rotation_2d = rot.toRotationMatrix().block<3, 3>(0, 0);

//  std::vector<Eigen::Vector2f> out_footprint;
//  out_footprint.resize(2 * footprint.size());

//  for (unsigned int i = 0; i < footprint.size(); i++) {
//    // Применяем поворот и трансляцию
//    out_footprint[2 * i] = pos + rotation_2d * footprint[i];

//    // Создаем ребра
//    if (i == 0) {
//      out_footprint.back() = out_footprint[2 * i];
//    } else {
//      out_footprint[2 * i - 1] = out_footprint[2 * i];
//    }
//  }
//  return out_footprint;
//}

/**
 * @brief transform footprint into edges
 * @param[in] yaw_angle ориентация (угол рыскания) в радианах
 * @param[in] pos позиция робота
 * @param[in] footprint полигон робота в локальной системе координат
 * @param[out] рёбра полигона робота в глобальной системе координат
 */
inline std::vector<Eigen::Vector2f> transformFootprintToEdges( float yaw_angle, const Eigen::Vector2f & pos, const std::vector<Eigen::Vector2f> & footprint)
{
  // Создаем матрицу поворота 2x2 из угла рыскания
  const float cos_yaw = std::cos(yaw_angle);
  const float sin_yaw = std::sin(yaw_angle);

  Eigen::Matrix2f rotation_2d;
  rotation_2d << cos_yaw, -sin_yaw,
                 sin_yaw,  cos_yaw;

  std::vector<Eigen::Vector2f> out_footprint;
  out_footprint.resize(2 * footprint.size());

  for (unsigned int i = 0; i < footprint.size(); i++) {
    // Применяем поворот и трансляцию
    out_footprint[2 * i] = pos + rotation_2d * footprint[i];

    // Создаем ребра
    if (i == 0) {
      out_footprint.back() = out_footprint[2 * i];
    } else {
      out_footprint[2 * i - 1] = out_footprint[2 * i];
    }
  }
  return out_footprint;
}
/**
 * @brief initializes marker to visualize shape of linestring
 * @param edge       edge to mark of footprint
 * @param i          marker ID
 * @param frame_id   frame of the marker
 * @param timestamp  timestamp of the marker
 * @return marker populated
 */
//inline visualization_msgs::msg::Marker createMarker(  const std::vector<geometry_msgs::msg::Point> edge, unsigned int i, const std::string & frame_id, const rclcpp::Time & timestamp)
//{
//  visualization_msgs::msg::Marker marker;
//  marker.header.frame_id = frame_id;
//  marker.header.stamp = timestamp;
//  marker.frame_locked = false;
//  marker.ns = "planned_footprint";
//  marker.action = visualization_msgs::msg::Marker::ADD;
//  marker.type = visualization_msgs::msg::Marker::LINE_LIST;
//  marker.lifetime = rclcpp::Duration(0, 0);

//  marker.id = i;
//  for (auto & point : edge) {
//    marker.points.push_back(point);
//  }

//  marker.pose.orientation.x = 0.0;
//  marker.pose.orientation.y = 0.0;
//  marker.pose.orientation.z = 0.0;
//  marker.pose.orientation.w = 1.0;
//  marker.scale.x = 0.05;
//  marker.scale.y = 0.05;
//  marker.scale.z = 0.05;
//  marker.color.r = 0.0f;
//  marker.color.g = 0.0f;
//  marker.color.b = 1.0f;
//  marker.color.a = 1.3f;
//  return marker;
//}




#endif  // NAV2_SMAC_PLANNER__UTILS_HPP_
