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
#ifndef NAV2_COSTMAP_2D__COSTMAP_2D_ROS_HPP_
#define NAV2_COSTMAP_2D__COSTMAP_2D_ROS_HPP_

#include <atomic>
#include <memory>
#include <string>
#include <vector>

#include "nav2_costmap_2d/costmap_2d.hpp"
#include "nav2_costmap_2d/footprint.hpp"
#include "nav2_costmap_2d/footprint_collision_checker.hpp"
#include "datatypes_smac_planner.h"
#include "types.hpp"



/** @brief A ROS wrapper for a 2D Costmap. Handles subscribing to
 * topics that provide observations about obstacles in either the form
 * of PointCloud or LaserScan messages. */
class Costmap2D_Master
{
public:
  /**
   * @brief  Constructor for the wrapper
   * @param options Additional options to control creation of the node.
   */
  Costmap2D_Master();

  /**
   * @brief A destructor
   */
  ~Costmap2D_Master();

  /**
   * @brief Configure node
   */
  bool on_configure(const CostmapConfig& config);

  /**
   * @brief Update the map with the layered costmap / plugins
   */
  void updateMap(const uint8_t * CostMap);

  /** @brief Returns the delay in transform (tf) data that is tolerable in seconds */
  float getTransformTolerance() const
  {
    return transform_tolerance_;
  }

  /**
   * @brief Return a pointer to the "master" costmap which receives updates from all the layers.
   *
   * Same as calling getLayeredCostmap()->getCostmap().
   */
  Costmap2D * getCostmap()
  {
    return master_costmap_.get();
  }

  /**
   * @brief  Returns the global frame of the costmap
   * @return The global frame of the costmap
   */
  std::string getGlobalFrameID()
  {
    return global_frame_;
  }

  /**
   * @brief  Returns the local frame of the costmap
   * @return The local frame of the costmap
   */
  std::string getBaseFrameID()
  {
    return robot_base_frame_;
  }

  /** @brief Returns the current padded footprint as a geometry_msgs::msg::Polygon. */
  Polygon getRobotFootprintPolygon()
  {
    return toPolygon(padded_footprint_);
  }

  /** @brief Return the current footprint of the robot as a vector of points.
   *
   * This version of the footprint is padded by the footprint_padding_
   * distance, set in the rosparam "footprint_padding".
   *
   * The footprint initially comes from the rosparam "footprint" but
   * can be overwritten by dynamic reconfigure or by messages received
   * on the "footprint" topic. */
  std::vector<Eigen::Vector2f> getRobotFootprint()
  {
    return padded_footprint_;
  }

  /** @brief Return the current unpadded footprint of the robot as a vector of points.
   *
   * This is the raw version of the footprint without padding.
   *
   * The footprint initially comes from the rosparam "footprint" but
   * can be overwritten by dynamic reconfigure or by messages received
   * on the "footprint" topic. */
  std::vector<Eigen::Vector2f> getUnpaddedRobotFootprint()
  {
    return unpadded_footprint_;
  }

  /** @brief Set the footprint of the robot to be the given set of
   * points, padded by footprint_padding.
   *
   * Should be a convex polygon, though this is not enforced.
   *
   * First expands the given polygon by footprint_padding_ and then
   * sets padded_footprint_ and calls
   * layered_costmap_->setFootprint().  Also saves the unpadded
   * footprint, which is available from
   * getUnpaddedRobotFootprint(). */
  void setRobotFootprint(const std::vector<Eigen::Vector2f> & points);

  /** @brief Set the footprint of the robot to be the given polygon,
   * padded by footprint_padding.
   *
   * Should be a convex polygon, though this is not enforced.
   *
   * First expands the given polygon by footprint_padding_ and then
   * sets padded_footprint_ and calls
   * layered_costmap_->setFootprint().  Also saves the unpadded
   * footprint, which is available from
   * getUnpaddedRobotFootprint(). */
  void setRobotFootprintPolygon(const Polygon & footprint);

  /** @brief Updates the stored footprint, updates the circumscribed
    * and inscribed radii, and calls onFootprintChanged() in all
    * layers. */
  void setFootprint(const std::vector<Eigen::Vector2f> & footprint_spec);

  /**
   * @brief  Get the costmap's use_radius_ parameter, corresponding to
   * whether the footprint for the robot is a circle with radius robot_radius_
   * or an arbitrarily defined footprint in footprint_.
   * @return  use_radius_
   */
  bool getUseRadius() {return use_radius_;}

  /**
   * @brief  Get the costmap's robot_radius_ parameter, corresponding to
   * raidus of the robot footprint when it is defined as as circle
   * (i.e. when use_radius_ == true).
   * @return  robot_radius_
   */
  float getRobotRadius() {return robot_radius_;}


  /** @brief  Given a distance, compute a cost.
   * @param  distance The distance from an obstacle in cells
   * @return A cost value for the distance */
  unsigned char inflation_layer_computeCost(float distance) const;


  float getInscribedRadius() const
  {
   return inscribed_radius_;
  }

  float getInflationRadius() const
  {
    return InflationRadius_;
  }

  float getCircumscribedRadius () const
  {
   return circumscribed_radius_;
  }


protected:
  /**
   * @brief Function on timer for costmap update
   */
  bool map_update_thread_shutdown_{false};
  std::atomic<bool> stop_updates_{false};
  std::atomic<bool> initialized_{false};
  std::atomic<bool> stopped_{true};
  std::mutex _dynamic_parameter_mutex;
  /**
   * @brief Get parameters for node
   */
  void initParameters(const CostmapConfig& config);
  std::string footprint_;
  float footprint_padding_{0};
  std::string global_frame_;                ///< The global frame for the costmap
  int map_height_meters_{0};
  int map_width_meters_{0};
  float origin_x_{0};
  float origin_y_{0};
  float resolution_{0};
  std::string robot_base_frame_;            ///< The frame_id of the robot base
  float robot_radius_;
  bool rolling_window_{false};          ///< Whether to use a rolling window version of the costmap
  bool track_unknown_space_{false};
  float transform_tolerance_{0};           ///< The timeout before transform errors
  float initial_transform_timeout_{0};   ///< The timeout before activation of the node errors
  float map_vis_z_{0};                 ///< The height of map, allows to avoid flickering at -0.008
  /// If true, the footprint subscriber expects a PolygonStamped msg
  bool subscribe_to_stamped_footprint_{false};

  // Derived parameters
  bool use_radius_{false};
  std::vector<Eigen::Vector2f> unpadded_footprint_;
  std::vector<Eigen::Vector2f> padded_footprint_;

  //Пример для прямоугольного робота 1.0×0.5 м:
  //InscribedRadius (0.25 м) - гарантированное столкновение
  //CircumscribedRadius (0.56 м) - гарантированное отсутствие столкновений
  //InflationRadius (0.55+ м) - буферная зона для безопасного планирования

  float cost_scaling_factor_; //это параметр, который определяет, насколько быстро уменьшается стоимость ячеек по мере удаления от препятствий в зоне инфляции
  float InflationRadius_;    //это радиус, на который "раздуваются" препятствия на карте стоимости для создания буферной зоны вокруг них
  float circumscribed_radius_; //это расстояние от центра робота до самой дальней точки его контура
  float inscribed_radius_;   //это максимальный радиус круга, который можно полностью вписать внутрь контура робота

  std::unique_ptr<Costmap2D> master_costmap_;
};


#endif  // NAV2_COSTMAP_2D__COSTMAP_2D_ROS_HPP_



