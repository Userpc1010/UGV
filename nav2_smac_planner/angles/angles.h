/*********************************************************************
* Software License Agreement (BSD License)
*
*  Copyright (c) 2008, Willow Garage, Inc.
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
*   * Neither the name of the Willow Garage nor the names of its
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
*********************************************************************/

#ifndef GEOMETRY_ANGLES_UTILS_H
#define GEOMETRY_ANGLES_UTILS_H

#include <algorithm>
#include <cmath>

#ifndef M_PIf32
#define M_PIf32 3.14159265358979323846f
#endif


namespace angles
{

  static inline float from_degrees(float degrees)
  {
    return degrees * M_PIf32 / 180.0f;
  }

  static inline float to_degrees(float radians)
  {
    return radians * 180.0f / M_PIf32;
  }


  static inline float normalize_angle_positive(float angle)
  {
    return fmod(fmod(angle, 2.0f*M_PIf32) + 2.0f*M_PIf32, 2.0f*M_PIf32);
  }


  static inline float normalize_angle(float angle)
  {
    float a = normalize_angle_positive(angle);
    if (a > M_PIf32)
      a -= 2.0f *M_PIf32;
    return a;
  }


  static inline float shortest_angular_distance(float from, float to)
  {
    return normalize_angle(to-from);
  }

  static inline float two_pi_complement(float angle)
  {
    //check input conditions
    if (angle > 2*M_PIf32 || angle < -2.0f*M_PIf32)
      angle = fmod(angle, 2.0f*M_PIf32);
    if(angle < 0)
      return (2*M_PIf32+angle);
    else if (angle > 0)
      return (-2*M_PIf32+angle);

    return(2*M_PIf32);
  }

  static bool find_min_max_delta(float from, float left_limit, float right_limit, float &result_min_delta, float &result_max_delta)
  {
    float delta[4];

    delta[0] = shortest_angular_distance(from,left_limit);
    delta[1] = shortest_angular_distance(from,right_limit);

    delta[2] = two_pi_complement(delta[0]);
    delta[3] = two_pi_complement(delta[1]);

    if(delta[0] == 0)
    {
      result_min_delta = delta[0];
      result_max_delta = std::max<float>(delta[1],delta[3]);
      return true;
    }

    if(delta[1] == 0)
    {
        result_max_delta = delta[1];
        result_min_delta = std::min<float>(delta[0],delta[2]);
        return true;
    }


    float delta_min = delta[0];
    float delta_min_2pi = delta[2];
    if(delta[2] < delta_min)
    {
      delta_min = delta[2];
      delta_min_2pi = delta[0];
    }

    float delta_max = delta[1];
    float delta_max_2pi = delta[3];
    if(delta[3] > delta_max)
    {
      delta_max = delta[3];
      delta_max_2pi = delta[1];
    }


    //    printf("%f %f %f %f\n",delta_min,delta_min_2pi,delta_max,delta_max_2pi);
    if((delta_min <= delta_max_2pi) || (delta_max >= delta_min_2pi))
    {
      result_min_delta = delta_max_2pi;
      result_max_delta = delta_min_2pi;
      if(left_limit == -M_PIf32 && right_limit == M_PIf32)
        return true;
      else
        return false;
    }
    result_min_delta = delta_min;
    result_max_delta = delta_max;
    return true;
  }


  static inline bool shortest_angular_distance_with_limits(float from, float to, float left_limit, float right_limit, float &shortest_angle)
  {

    float min_delta = -2*M_PIf32;
    float max_delta = 2*M_PIf32;
    float min_delta_to = -2*M_PIf32;
    float max_delta_to = 2*M_PIf32;
    bool flag    = find_min_max_delta(from,left_limit,right_limit,min_delta,max_delta);
    float delta = shortest_angular_distance(from,to);
    float delta_mod_2pi  = two_pi_complement(delta);


    if(flag)//from position is within the limits
    {
      if(delta >= min_delta && delta <= max_delta)
      {
        shortest_angle = delta;
        return true;
      }
      else if(delta_mod_2pi >= min_delta && delta_mod_2pi <= max_delta)
      {
        shortest_angle = delta_mod_2pi;
        return true;
      }
      else //to position is outside the limits
      {
        find_min_max_delta(to,left_limit,right_limit,min_delta_to,max_delta_to);
          if(fabs(min_delta_to) < fabs(max_delta_to))
            shortest_angle = std::max<float>(delta,delta_mod_2pi);
          else if(fabs(min_delta_to) > fabs(max_delta_to))
            shortest_angle =  std::min<float>(delta,delta_mod_2pi);
          else
          {
            if (fabs(delta) < fabs(delta_mod_2pi))
              shortest_angle = delta;
            else
              shortest_angle = delta_mod_2pi;
          }
          return false;
      }
    }
    else // from position is outside the limits
    {
        find_min_max_delta(to,left_limit,right_limit,min_delta_to,max_delta_to);

          if(fabs(min_delta) < fabs(max_delta))
            shortest_angle = std::min<float>(delta,delta_mod_2pi);
          else if (fabs(min_delta) > fabs(max_delta))
            shortest_angle =  std::max<float>(delta,delta_mod_2pi);
          else
          {
            if (fabs(delta) < fabs(delta_mod_2pi))
              shortest_angle = delta;
            else
              shortest_angle = delta_mod_2pi;
          }
      return false;
    }

    shortest_angle = delta;
    return false;
  }
}

#endif
