/*
 * Copyright (c) 2013, Willow Garage, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the Willow Garage, Inc. nor the names of its
 *       contributors may be used to endorse or promote products derived from
 *       this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */
#include "nav2_costmap_2d/footprint.hpp"

#include <algorithm>
#include <limits>
#include <string>
#include <vector>
#include <iostream>

#ifndef M_PIf32
#define M_PIf32 3.14159265358979323846f
#endif


#include "nav2_costmap_2d/costmap_math.hpp"

std::pair<float, float> calculateMinAndMaxDistances( const std::vector<Eigen::Vector2f> & footprint)
{
  float min_dist = std::numeric_limits<float>::max();
  float max_dist = 0.0f;

  if (footprint.size() <= 2) {
    return std::pair<float, float>(min_dist, max_dist);
  }

  for (unsigned int i = 0; i < footprint.size() - 1; ++i) {
    // check the distance from the robot center point to the first vertex
    float vertex_dist = distance(0.0f, 0.0f, footprint[i].x(), footprint[i].y());
    float edge_dist = distanceToLine(
      0.0f, 0.0f, footprint[i].x(), footprint[i].y(),
      footprint[i + 1].x(), footprint[i + 1].y());
    min_dist = std::min(min_dist, std::min(vertex_dist, edge_dist));
    max_dist = std::max(max_dist, std::max(vertex_dist, edge_dist));
  }

  // we also need to do the last vertex and the first vertex
  float vertex_dist = distance(0.0f, 0.0f, footprint.back().x(), footprint.back().y());
  float edge_dist = distanceToLine(
    0.0f, 0.0f, footprint.back().x(), footprint.back().y(),
    footprint.front().x(), footprint.front().y());
  min_dist = std::min(min_dist, std::min(vertex_dist, edge_dist));
  max_dist = std::max(max_dist, std::max(vertex_dist, edge_dist));

  return std::pair<float, float>(min_dist, max_dist);
}

Eigen::Vector2d toPoint64(const Eigen::Vector2f & pt)
{
  Eigen::Vector2d point64;
  point64.x() = pt.x();
  point64.y() = pt.y();
  return point64;
}

Eigen::Vector2f toPoint(const Eigen::Vector2d & pt)
{
  Eigen::Vector2f point;
  point.x() = pt.x();
  point.y() = pt.y();
  return point;
}

Polygon toPolygon(const std::vector<Eigen::Vector2f> & pts)
{
  Polygon polygon;
  polygon.points.reserve(pts.size());
  for (const auto & pt : pts) {
    polygon.points.push_back(pt);
  }
  return polygon;
}

std::vector<Eigen::Vector2f> toPointVector(const Polygon & polygon)
{
  std::vector<Eigen::Vector2f> pts;
  pts.reserve(polygon.points.size());
  for (const auto & point : polygon.points) {
    pts.push_back(point);
  }
  return pts;
}

void transformFootprint(
  float x, float y, float theta,
  const std::vector<Eigen::Vector2f> & footprint_spec,
  std::vector<Eigen::Vector2f> & oriented_footprint)
{
  // build the oriented footprint at a given location
  oriented_footprint.resize(footprint_spec.size());
  float cos_th = cos(theta);
  float sin_th = sin(theta);
  for (unsigned int i = 0; i < footprint_spec.size(); ++i) {
    float new_x = x + (footprint_spec[i].x() * cos_th - footprint_spec[i].y() * sin_th);
    float new_y = y + (footprint_spec[i].x() * sin_th + footprint_spec[i].y() * cos_th);
    Eigen::Vector2f & new_pt = oriented_footprint[i];
    new_pt.x() = new_x;
    new_pt.y() = new_y;
  }
}

void transformFootprint(
  float x, float y, float theta,
  const std::vector<Eigen::Vector2f> & footprint_spec,
  PolygonStamped & oriented_footprint)
{
  // build the oriented footprint at a given location
  oriented_footprint.polygon.points.clear();
  float cos_th = cos(theta);
  float sin_th = sin(theta);
  for (unsigned int i = 0; i < footprint_spec.size(); ++i) {
    Eigen::Vector2f new_pt;
    new_pt.x() = x + (footprint_spec[i].x() * cos_th - footprint_spec[i].y() * sin_th);
    new_pt.y() = y + (footprint_spec[i].x() * sin_th + footprint_spec[i].y() * cos_th);
    oriented_footprint.polygon.points.push_back(new_pt);
  }
}

void padFootprint(std::vector<Eigen::Vector2f> & footprint, float padding)
{
  // pad footprint in place
  for (unsigned int i = 0; i < footprint.size(); i++) {
    Eigen::Vector2f & pt = footprint[i];
    pt.x() += sign0(pt.x()) * padding;
    pt.y() += sign0(pt.y()) * padding;
  }
}


std::vector<Eigen::Vector2f> makeFootprintFromRadius(float radius)
{
  std::vector<Eigen::Vector2f> points;

  // Loop over 16 angles around a circle making a point each time
  int N = 16;
  Eigen::Vector2f pt;
  for (int i = 0; i < N; ++i) {
    float angle = i * 2 * M_PIf32 / N;
    pt.x() = cos(angle) * radius;
    pt.y() = sin(angle) * radius;

    points.push_back(pt);
  }

  return points;
}


std::vector<std::vector<float>> parseVVF(const std::string & input, std::string & error_return)
{
  std::vector<std::vector<float>> result;

  std::stringstream input_ss(input);
  int depth = 0;
  std::vector<float> current_vector;
  while (!!input_ss && !input_ss.eof()) {
    switch (input_ss.peek()) {
      case EOF:
        break;
      case '[':
        depth++;
        if (depth > 2) {
          error_return = "Array depth greater than 2";
          return result;
        }
        input_ss.get();
        current_vector.clear();
        break;
      case ']':
        depth--;
        if (depth < 0) {
          error_return = "More close ] than open [";
          return result;
        }
        input_ss.get();
        if (depth == 1) {
          result.push_back(current_vector);
        }
        break;
      case ',':
      case ' ':
      case '\t':
        input_ss.get();
        break;
      default:  // All other characters should be part of the numbers.
        if (depth != 2) {
          std::stringstream err_ss;
          err_ss << "Numbers at depth other than 2. Char was '" << char(input_ss.peek()) << "'.";
          error_return = err_ss.str();
          return result;
        }
        float value;
        input_ss >> value;
        if (!!input_ss) {
          current_vector.push_back(value);
        }
        break;
    }
  }

  if (depth != 0) {
    error_return = "Unterminated vector string.";
  } else {
    error_return = "";
  }

  return result;
}

bool makeFootprintFromString(
  const std::string & footprint_string,
  std::vector<Eigen::Vector2f> & footprint)
{
  std::string error;
  std::vector<std::vector<float>> vvf = parseVVF(footprint_string, error);

  if (error != "") {
    std::cout << "nav2_costmap_2d: Error parsing footprint parameter: '" << error << "'" << std::endl;
    std::cout << "nav2_costmap_2d: Footprint string was '" << footprint_string << "'." << std::endl;
    return false;
  }

  // convert vvf into points.
  if (vvf.size() < 3) {
    std::cout << "nav2_costmap_2d: You must specify at least three points for the robot footprint,"
                 " reverting to previous footprint." << std::endl;
    return false;
  }
  footprint.reserve(vvf.size());
  for (unsigned int i = 0; i < vvf.size(); i++) {
    if (vvf[i].size() == 2) {
      Eigen::Vector2f point;
      point.x() = vvf[i][0];
      point.y() = vvf[i][1];
      footprint.push_back(point);
    } else {
      std::cout << "nav2_costmap_2d: Points in the footprint specification must be pairs of numbers."
                   " Found a point with " << vvf[i].size() << " numbers." << std::endl;
      return false;
    }
  }

  return true;
}


