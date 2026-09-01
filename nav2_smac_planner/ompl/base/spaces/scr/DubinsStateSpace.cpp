/*********************************************************************
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2010, Rice University
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
 *   * Neither the name of the Rice University nor the names of its
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

/* Author: Mark Moll */

#include "ompl/base/spaces/DubinsStateSpace.h"
#include "ompl/base/SpaceInformation.h"
#include "ompl/util/Exception.h"
#include <boost/math/constants/constants.hpp>

using namespace ompl::base;

namespace
{
    constexpr float twopi = 2.f * boost::math::constants::pi<float>();
    constexpr float onepi = boost::math::constants::pi<float>();
    constexpr float halfpi = boost::math::constants::half_pi<float>();
    const float DUBINS_EPS = 1e-4f;
    const float DUBINS_ZERO = 10 * std::numeric_limits<float>::epsilon();

    enum DubinsClass
    {
        A11 = 0,
        A12 = 1,
        A13 = 2,
        A14 = 3,
        A21 = 4,
        A22 = 5,
        A23 = 6,
        A24 = 7,
        A31 = 8,
        A32 = 9,
        A33 = 10,
        A34 = 11,
        A41 = 12,
        A42 = 13,
        A43 = 14,
        A44 = 15
    };

    inline float mod2pi(float x)
    {
        if (x < 0 && x > DUBINS_ZERO)
            return 0;
        float xm = x - twopi * floor(x / twopi);
        if (twopi - xm < 0.5f * DUBINS_EPS)
            xm = 0.0f;
        return xm;
    }

    inline float t_lsr(float d, float alpha, float beta)
    {
        float ca = cos(alpha), sa = sin(alpha), cb = cos(beta), sb = sin(beta);
        const float tmp = -2.0f + d * d + 2.0f * (ca * cb + sa * sb + d * (sa + sb));
        const float p = sqrtf(std::max(tmp, 0.0f));
        const float theta = atan2f(-ca - cb, d + sa + sb) - atan2f(-2.0f, p);
        return mod2pi(-alpha + theta);  // t
    }

    inline float p_lsr(float d, float alpha, float beta)
    {
        float ca = cos(alpha), sa = sin(alpha), cb = cos(beta), sb = sin(beta);
        const float tmp = -2.0f + d * d + 2.0f * (ca * cb + sa * sb + d * (sa + sb));
        return sqrtf(std::max(tmp, 0.0f));  // p
    }

    inline float q_lsr(float d, float alpha, float beta)
    {
        float ca = cos(alpha), sa = sin(alpha), cb = cos(beta), sb = sin(beta);
        const float tmp = -2.0f + d * d + 2.0f * (ca * cb + sa * sb + d * (sa + sb));
        const float p = sqrtf(std::max(tmp, 0.0f));
        const float theta = atan2f(-ca - cb, d + sa + sb) - atan2f(-2.0f, p);
        return mod2pi(-beta + theta);  // q
    }

    inline float t_rsl(float d, float alpha, float beta)
    {
        float ca = cos(alpha), sa = sin(alpha), cb = cos(beta), sb = sin(beta);
        const float tmp = d * d - 2.0f + 2.0f * (ca * cb + sa * sb - d * (sa + sb));
        const float p = sqrtf(std::max(tmp, 0.0f));
        const float theta = atan2f(ca + cb, d - sa - sb) - atan2f(2.0f, p);
        return mod2pi(alpha - theta);  // t
    }

    inline float p_rsl(float d, float alpha, float beta)
    {
        float ca = cos(alpha), sa = sin(alpha), cb = cos(beta), sb = sin(beta);
        const float tmp = d * d - 2.0f + 2.0f * (ca * cb + sa * sb - d * (sa + sb));
        return sqrtf(std::max(tmp, 0.0f));  // p
    }

    inline float q_rsl(float d, float alpha, float beta)
    {
        float ca = cos(alpha), sa = sin(alpha), cb = cos(beta), sb = sin(beta);
        const float tmp = d * d - 2.0f + 2.0f * (ca * cb + sa * sb - d * (sa + sb));
        const float p = sqrtf(std::max(tmp, 0.0f));
        const float theta = atan2f(ca + cb, d - sa - sb) - atan2f(2.0f, p);
        return mod2pi(beta - theta);  // q
    }

    inline float t_rsr(float d, float alpha, float beta)
    {
        float ca = cos(alpha), sa = sin(alpha), cb = cos(beta), sb = sin(beta);
        const float theta = atan2f(ca - cb, d - sa + sb);
        return mod2pi(alpha - theta);  // t
    }

    inline float p_rsr(float d, float alpha, float beta)
    {
        float ca = cos(alpha), sa = sin(alpha), cb = cos(beta), sb = sin(beta);
        const float tmp = 2.0f + d * d - 2.0f * (ca * cb + sa * sb - d * (sb - sa));
        return sqrtf(std::max(tmp, 0.0f));  // p
    }

    inline float q_rsr(float d, float alpha, float beta)
    {
        float ca = cos(alpha), sa = sin(alpha), cb = cos(beta), sb = sin(beta);
        const float theta = atan2f(ca - cb, d - sa + sb);
        return mod2pi(-beta + theta);  // q
    }

    inline float t_lsl(float d, float alpha, float beta)
    {
        float ca = cos(alpha), sa = sin(alpha), cb = cos(beta), sb = sin(beta);
        const float theta = atan2f(cb - ca, d + sa - sb);
        return mod2pi(-alpha + theta);  // t
    }

    inline float p_lsl(float d, float alpha, float beta)
    {
        float ca = cos(alpha), sa = sin(alpha), cb = cos(beta), sb = sin(beta);
        const float tmp = 2.0f + d * d - 2.0f * (ca * cb + sa * sb - d * (sa - sb));
        return sqrtf(std::max(tmp, 0.0f));  // p
    }

    inline float q_lsl(float d, float alpha, float beta)
    {
        float ca = cos(alpha), sa = sin(alpha), cb = cos(beta), sb = sin(beta);
        const float theta = atan2f(cb - ca, d + sa - sb);
        return mod2pi(beta - theta);  // q
    }

    inline float s_12(float d, float alpha, float beta)
    {
        return p_rsr(d, alpha, beta) - p_rsl(d, alpha, beta) - 2.0f * (q_rsl(d, alpha, beta) - onepi);
    }

    inline float s_13(float d, float alpha, float beta)
    {  // t_rsr - pi
        return t_rsr(d, alpha, beta) - onepi;
    }

    inline float s_14_1(float d, float alpha, float beta)
    {
        return t_rsr(d, alpha, beta) - onepi;
    }

    inline float s_21(float d, float alpha, float beta)
    {
        return p_lsl(d, alpha, beta) - p_rsl(d, alpha, beta) - 2.0f * (t_rsl(d, alpha, beta) - onepi);
    }

    inline float s_22_1(float d, float alpha, float beta)
    {
        return p_lsl(d, alpha, beta) - p_rsl(d, alpha, beta) - 2.0f * (t_rsl(d, alpha, beta) - onepi);
    }

    inline float s_22_2(float d, float alpha, float beta)
    {
        return p_rsr(d, alpha, beta) - p_rsl(d, alpha, beta) - 2.0f * (q_rsl(d, alpha, beta) - onepi);
    }

    inline float s_24(float d, float alpha, float beta)
    {
        return q_rsr(d, alpha, beta) - onepi;
    }

    inline float s_31(float d, float alpha, float beta)
    {
        return q_lsl(d, alpha, beta) - onepi;
    }

    inline float s_33_1(float d, float alpha, float beta)
    {
        return p_rsr(d, alpha, beta) - p_lsr(d, alpha, beta) - 2.0f * (t_lsr(d, alpha, beta) - onepi);
    }

    inline float s_33_2(float d, float alpha, float beta)
    {
        return p_lsl(d, alpha, beta) - p_lsr(d, alpha, beta) - 2.0f * (q_lsr(d, alpha, beta) - onepi);
    }

    inline float s_34(float d, float alpha, float beta)
    {
        return p_rsr(d, alpha, beta) - p_lsr(d, alpha, beta) - 2.0f * (t_lsr(d, alpha, beta) - onepi);
    }

    inline float s_41_1(float d, float alpha, float beta)
    {
        return t_lsl(d, alpha, beta) - onepi;
    }

    inline float s_41_2(float d, float alpha, float beta)
    {
        return q_lsl(d, alpha, beta) - onepi;
    }

    inline float s_42(float d, float alpha, float beta)
    {
        return t_lsl(d, alpha, beta) - onepi;
    }

    inline float s_43(float d, float alpha, float beta)
    {
        return p_lsl(d, alpha, beta) - p_lsr(d, alpha, beta) - 2.0f * (q_lsr(d, alpha, beta) - onepi);
    }

    DubinsStateSpace::PathType dubinsLSL(float d, float alpha, float beta)
    {
        float ca = cos(alpha), sa = sin(alpha), cb = cos(beta), sb = sin(beta);
        float tmp = 2.f + d * d - 2.f * (ca * cb + sa * sb - d * (sa - sb));
        if (tmp >= DUBINS_ZERO)
        {
            float theta = atan2(cb - ca, d + sa - sb);
            float t = mod2pi(-alpha + theta);
            float p = sqrt(std::max(tmp, 0.0f));
            float q = mod2pi(beta - theta);
            assert(fabs(p * cos(alpha + t) - sa + sb - d) < (1 + p) * DUBINS_EPS);
            assert(fabs(p * sin(alpha + t) + ca - cb) < (1 + p) * DUBINS_EPS);
            assert(mod2pi(alpha + t + q - beta + 0.5f * DUBINS_EPS) < DUBINS_EPS);
            return DubinsStateSpace::PathType(DubinsStateSpace::dubinsPathType()[0], t, p, q);
        }
        return {};
    }

    DubinsStateSpace::PathType dubinsRSR(float d, float alpha, float beta)
    {
        float ca = cos(alpha), sa = sin(alpha), cb = cos(beta), sb = sin(beta);
        float tmp = 2.f + d * d - 2.f * (ca * cb + sa * sb - d * (sb - sa));
        if (tmp >= DUBINS_ZERO)
        {
            float theta = atan2(ca - cb, d - sa + sb);
            float t = mod2pi(alpha - theta);
            float p = sqrt(std::max(tmp, 0.0f));
            float q = mod2pi(-beta + theta);
            assert(fabs(p * cos(alpha - t) + sa - sb - d) < (1 + p) * DUBINS_EPS);
            assert(fabs(p * sin(alpha - t) - ca + cb) < (1 + p) * DUBINS_EPS);
            assert(mod2pi(alpha - t - q - beta + 0.5f * DUBINS_EPS) < DUBINS_EPS);
            return DubinsStateSpace::PathType(DubinsStateSpace::dubinsPathType()[1], t, p, q);
        }
        return {};
    }

    DubinsStateSpace::PathType dubinsRSL(float d, float alpha, float beta)
    {
        float ca = cos(alpha), sa = sin(alpha), cb = cos(beta), sb = sin(beta);
        float tmp = d * d - 2.f + 2.f * (ca * cb + sa * sb - d * (sa + sb));
        if (tmp >= DUBINS_ZERO)
        {
            float p = sqrt(std::max(tmp, 0.0f));
            float theta = atan2(ca + cb, d - sa - sb) - atan2(2.f, p);
            float t = mod2pi(alpha - theta);
            float q = mod2pi(beta - theta);
            assert(fabs(p * cos(alpha - t) - 2.f * sin(alpha - t) + sa + sb - d) < 2 * DUBINS_EPS);
            assert(fabs(p * sin(alpha - t) + 2.f * cos(alpha - t) - ca - cb) < 2 * DUBINS_EPS);
            assert(mod2pi(alpha - t + q - beta + 0.5f * DUBINS_EPS) < DUBINS_EPS);
            return DubinsStateSpace::PathType(DubinsStateSpace::dubinsPathType()[2], t, p, q);
        }
        return {};
    }

    DubinsStateSpace::PathType dubinsLSR(float d, float alpha, float beta)
    {
        float ca = cos(alpha), sa = sin(alpha), cb = cos(beta), sb = sin(beta);
        float tmp = -2.f + d * d + 2.f * (ca * cb + sa * sb + d * (sa + sb));
        if (tmp >= DUBINS_ZERO)
        {
            float p = sqrt(std::max(tmp, 0.0f));
            float theta = atan2(-ca - cb, d + sa + sb) - atan2(-2.f, p);
            float t = mod2pi(-alpha + theta);
            float q = mod2pi(-beta + theta);
            assert(fabs(p * cos(alpha + t) + 2.f * sin(alpha + t) - sa - sb - d) < 2 * DUBINS_EPS);
            assert(fabs(p * sin(alpha + t) - 2.f * cos(alpha + t) + ca + cb) < 2 * DUBINS_EPS);
            assert(mod2pi(alpha + t - q - beta + 0.5f * DUBINS_EPS) < DUBINS_EPS);
            return DubinsStateSpace::PathType(DubinsStateSpace::dubinsPathType()[3], t, p, q);
        }
        return {};
    }

    DubinsStateSpace::PathType dubinsRLR(float d, float alpha, float beta)
    {
        float ca = cos(alpha), sa = sin(alpha), cb = cos(beta), sb = sin(beta);
        float tmp = 0.125f * (6.f - d * d + 2.f * (ca * cb + sa * sb + d * (sa - sb)));
        if (fabs(tmp) < 1.f)
        {
            float p = twopi - acos(tmp);
            float theta = atan2(ca - cb, d - sa + sb);
            float t = mod2pi(alpha - theta + 0.5f * p);
            float q = mod2pi(alpha - beta - t + p);
            assert(fabs(2.f * sin(alpha - t + p) - 2.f * sin(alpha - t) - d + sa - sb) < 2 * DUBINS_EPS);
            assert(fabs(-2.f * cos(alpha - t + p) + 2.f * cos(alpha - t) - ca + cb) < 2 * DUBINS_EPS);
            assert(mod2pi(alpha - t + p - q - beta + 0.5f * DUBINS_EPS) < DUBINS_EPS);
            return DubinsStateSpace::PathType(DubinsStateSpace::dubinsPathType()[4], t, p, q);
        }
        return {};
    }

    DubinsStateSpace::PathType dubinsLRL(float d, float alpha, float beta)
    {
        float ca = cos(alpha), sa = sin(alpha), cb = cos(beta), sb = sin(beta);
        float tmp = 0.125f * (6.f - d * d + 2.f * (ca * cb + sa * sb - d * (sa - sb)));
        if (fabs(tmp) < 1.f)
        {
            float p = twopi - acos(tmp);
            float theta = atan2(-ca + cb, d + sa - sb);
            float t = mod2pi(-alpha + theta + 0.5f * p);
            float q = mod2pi(beta - alpha - t + p);
            assert(fabs(-2.f * sin(alpha + t - p) + 2.f * sin(alpha + t) - d - sa + sb) < 2 * DUBINS_EPS);
            assert(fabs(2.f * cos(alpha + t - p) - 2.f * cos(alpha + t) + ca - cb) < 2 * DUBINS_EPS);
            assert(mod2pi(alpha + t - p + q - beta + 0.5f * DUBINS_EPS) < DUBINS_EPS);
            return DubinsStateSpace::PathType(DubinsStateSpace::dubinsPathType()[5], t, p, q);
        }
        return {};
    }

    bool isLongPath(float d, float alpha, float beta)
    {
        return (std::abs(std::sin(alpha)) + std::abs(std::sin(beta)) +
                std::sqrt(4 - std::pow(std::cos(alpha) + std::cos(beta), 2)) - d) < 0;
    }

    DubinsStateSpace::PathType dubinsExhaustive(const float d, const float alpha, const float beta)
    {
        if (d < DUBINS_EPS && fabs(alpha - beta) < DUBINS_EPS)
            return {DubinsStateSpace::dubinsPathType()[0], 0, d, 0};

        DubinsStateSpace::PathType path(dubinsLSL(d, alpha, beta)), tmp(dubinsRSR(d, alpha, beta));
        float len, minLength = path.length();

        if ((len = tmp.length()) < minLength)
        {
            minLength = len;
            path = tmp;
        }
        tmp = dubinsRSL(d, alpha, beta);
        if ((len = tmp.length()) < minLength)
        {
            minLength = len;
            path = tmp;
        }
        tmp = dubinsLSR(d, alpha, beta);
        if ((len = tmp.length()) < minLength)
        {
            minLength = len;
            path = tmp;
        }
        tmp = dubinsRLR(d, alpha, beta);
        if ((len = tmp.length()) < minLength)
        {
            minLength = len;
            path = tmp;
        }
        tmp = dubinsLRL(d, alpha, beta);
        if ((len = tmp.length()) < minLength)
            path = tmp;
        return path;
    }

    DubinsClass getDubinsClass(const float alpha, const float beta)
    {
        int row(0), column(0);
        if (0 <= alpha && alpha <= halfpi)
        {
            row = 1;
        }
        else if (halfpi < alpha && alpha <= onepi)
        {
            row = 2;
        }
        else if (onepi < alpha && alpha <= 3 * halfpi)
        {
            row = 3;
        }
        else if (3 * halfpi < alpha && alpha <= twopi)
        {
            row = 4;
        }

        if (0 <= beta && beta <= halfpi)
        {
            column = 1;
        }
        else if (halfpi < beta && beta <= onepi)
        {
            column = 2;
        }
        else if (onepi < beta && beta <= 3 * halfpi)
        {
            column = 3;
        }
        else if (3 * halfpi < beta && beta <= 2.0f * onepi)
        {
            column = 4;
        }

        assert(row >= 1 && row <= 4 &&
               "alpha is not in the range of [0,2pi] in classifyPath(double alpha, double beta).");
        assert(column >= 1 && column <= 4 &&
               "beta is not in the range of [0,2pi] in classifyPath(double alpha, double beta).");
        assert((column - 1) + 4 * (row - 1) >= 0 && (column - 1) + 4 * (row - 1) <= 15 &&
               "class is not in range [0,15].");
        return (DubinsClass)((column - 1) + 4 * (row - 1));
    }

    DubinsStateSpace::PathType dubinsClassification(const float d, const float alpha, const float beta)
    {
        if (d < DUBINS_EPS && fabs(alpha - beta) < DUBINS_EPS)
            return {DubinsStateSpace::dubinsPathType()[0], 0, d, 0};
        // Dubins set classification scheme
        // Shkel, Andrei M., and Vladimir Lumelsky. "Classification of the Dubins set."
        //   Robotics and Autonomous Systems 34.4 (2001): 179-202.
        // Lim, Jaeyoung, et al. "Circling Back: Dubins set Classification Revisited."
        //   Workshop on Energy Efficient Aerial Robotic Systems, International Conference on Robotics and Automation
        //   2023.
        DubinsStateSpace::PathType path;
        auto dubins_class = getDubinsClass(alpha, beta);
        switch (dubins_class)
        {
            case DubinsClass::A11:
            {
                path = dubinsRSL(d, alpha, beta);
                break;
            }
            case DubinsClass::A12:
            {
                if (s_13(d, alpha, beta) < 0.0f)
                {
                    path = (s_12(d, alpha, beta) < 0.0f) ? dubinsRSR(d, alpha, beta) : dubinsRSL(d, alpha, beta);
                }
                else
                {
                    path = dubinsLSR(d, alpha, beta);
                    DubinsStateSpace::PathType tmp = dubinsRSL(d, alpha, beta);
                    if (path.length() > tmp.length())
                    {
                        path = tmp;
                    }
                }
                break;
            }
            case DubinsClass::A13:
            {
                if (s_13(d, alpha, beta) < 0.0f)
                {
                    path = dubinsRSR(d, alpha, beta);
                }
                else
                {
                    path = dubinsLSR(d, alpha, beta);
                }
                break;
            }
            case DubinsClass::A14:
            {
                if (s_14_1(d, alpha, beta) > 0.0f)
                {
                    path = dubinsLSR(d, alpha, beta);
                }
                else if (s_24(d, alpha, beta) > 0.0f)
                {
                    path = dubinsRSL(d, alpha, beta);
                }
                else
                {
                    path = dubinsRSR(d, alpha, beta);
                }
                break;
            }
            case DubinsClass::A21:
            {
                if (s_31(d, alpha, beta) < 0.0f)
                {
                    if (s_21(d, alpha, beta) < 0.0f)
                    {
                        path = dubinsLSL(d, alpha, beta);
                    }
                    else
                    {
                        path = dubinsRSL(d, alpha, beta);
                    }
                }
                else
                {
                    path = dubinsLSR(d, alpha, beta);
                    DubinsStateSpace::PathType tmp = dubinsRSL(d, alpha, beta);
                    if (path.length() > tmp.length())
                    {
                        path = tmp;
                    }
                }
                break;
            }
            case DubinsClass::A22:
            {
                if (alpha > beta)
                {
                    path = (s_22_1(d, alpha, beta) < 0.0f) ? dubinsLSL(d, alpha, beta) : dubinsRSL(d, alpha, beta);
                }
                else
                {
                    path = (s_22_2(d, alpha, beta) < 0.0f) ? dubinsRSR(d, alpha, beta) : dubinsRSL(d, alpha, beta);
                }
                break;
            }
            case DubinsClass::A23:
            {
                path = dubinsRSR(d, alpha, beta);
                break;
            }
            case DubinsClass::A24:
            {
                if (s_24(d, alpha, beta) < 0.0f)
                {
                    path = dubinsRSR(d, alpha, beta);
                }
                else
                {
                    path = dubinsRSL(d, alpha, beta);
                }
                break;
            }
            case DubinsClass::A31:
            {
                if (s_31(d, alpha, beta) < 0.0f)
                {
                    path = dubinsLSL(d, alpha, beta);
                }
                else
                {
                    path = dubinsLSR(d, alpha, beta);
                }
                break;
            }
            case DubinsClass::A32:
            {
                path = dubinsLSL(d, alpha, beta);
                break;
            }
            case DubinsClass::A33:
            {
                if (alpha < beta)
                {
                    if (s_33_1(d, alpha, beta) < 0.0f)
                    {
                        path = dubinsRSR(d, alpha, beta);
                    }
                    else
                    {
                        path = dubinsLSR(d, alpha, beta);
                    }
                }
                else
                {
                    if (s_33_2(d, alpha, beta) < 0.0f)
                    {
                        path = dubinsLSL(d, alpha, beta);
                    }
                    else
                    {
                        path = dubinsLSR(d, alpha, beta);
                    }
                }
                break;
            }
            case DubinsClass::A34:
            {
                if (s_24(d, alpha, beta) < 0.0f)
                {
                    if (s_34(d, alpha, beta) < 0.0f)
                    {
                        path = dubinsRSR(d, alpha, beta);
                    }
                    else
                    {
                        path = dubinsLSR(d, alpha, beta);
                    }
                }
                else
                {
                    path = dubinsLSR(d, alpha, beta);
                    DubinsStateSpace::PathType tmp = dubinsRSL(d, alpha, beta);
                    if (path.length() > tmp.length())
                    {
                        path = tmp;
                    }
                }
                break;
            }
            case DubinsClass::A41:
            {
                if (s_41_1(d, alpha, beta) > 0.0f)
                {
                    path = dubinsRSL(d, alpha, beta);
                }
                else if (s_41_2(d, alpha, beta) > 0.0f)
                {
                    path = dubinsLSR(d, alpha, beta);
                }
                else
                {
                    path = dubinsLSL(d, alpha, beta);
                }
                break;
            }
            case DubinsClass::A42:
            {
                if (s_42(d, alpha, beta) < 0.0f)
                {
                    path = dubinsLSL(d, alpha, beta);
                }
                else
                {
                    path = dubinsRSL(d, alpha, beta);
                }
                break;
            }
            case DubinsClass::A43:
            {
                if (s_42(d, alpha, beta) < 0.0f)
                {
                    if (s_43(d, alpha, beta) < 0.0f)
                    {
                        path = dubinsLSL(d, alpha, beta);
                    }
                    else
                    {
                        path = dubinsLSR(d, alpha, beta);
                    }
                }
                else
                {
                    path = dubinsLSR(d, alpha, beta);
                    DubinsStateSpace::PathType tmp = dubinsRSL(d, alpha, beta);
                    if (path.length() > tmp.length())
                    {
                        path = tmp;
                    }
                }
                break;
            }
            case DubinsClass::A44:
            {
                path = dubinsLSR(d, alpha, beta);
                break;
            }
        }
        return path;
    }

}  // namespace

namespace ompl::base
{
    std::ostream &operator<<(std::ostream &os, const DubinsStateSpace::PathType &path)
    {
        os << "DubinsPath[ type=";
        for (unsigned i = 0; i < 3; ++i)
            if (path.type_->at(i) == DubinsStateSpace::DUBINS_LEFT)
                os << "L";
            else if (path.type_->at(i) == DubinsStateSpace::DUBINS_STRAIGHT)
                os << "S";
            else
                os << "R";
        os << ", length=" << path.length_[0] << '+' << path.length_[1] << '+' << path.length_[2] << '=' << path.length()
           << ", reverse=" << path.reverse_ << " ]";
        return os;
    }
}  // namespace ompl::base

DubinsStateSpace::PathType getPath(float d, float alpha, float beta)
{
    if (d < DUBINS_EPS && fabs(alpha - beta) < DUBINS_EPS)
        return {DubinsStateSpace::dubinsPathType()[0], 0, d, 0};
    alpha = mod2pi(alpha);
    beta = mod2pi(beta);
    return isLongPath(d, alpha, beta) ? ::dubinsClassification(d, alpha, beta) : ::dubinsExhaustive(d, alpha, beta);
}

const std::vector<std::vector<DubinsStateSpace::DubinsPathSegmentType> >& DubinsStateSpace::dubinsPathType() {
  static std::vector<std::vector<DubinsStateSpace::DubinsPathSegmentType> >* pathType
    = new std::vector<std::vector<DubinsStateSpace::DubinsPathSegmentType> >(
      {{
        {{DUBINS_LEFT, DUBINS_STRAIGHT, DUBINS_LEFT}},
        {{DUBINS_RIGHT, DUBINS_STRAIGHT, DUBINS_RIGHT}},
        {{DUBINS_RIGHT, DUBINS_STRAIGHT, DUBINS_LEFT}},
        {{DUBINS_LEFT, DUBINS_STRAIGHT, DUBINS_RIGHT}},
        {{DUBINS_RIGHT, DUBINS_LEFT, DUBINS_RIGHT}},
        {{DUBINS_LEFT, DUBINS_RIGHT, DUBINS_LEFT}}
      }}
    );
    return *pathType;
}

float DubinsStateSpace::distance(const State *state1, const State *state2) const
{
    return isSymmetric_ ? symmetricDistance(state1, state2, rho_) : distance(state1, state2, rho_);
}
float DubinsStateSpace::distance(const State *state1, const State *state2, float radius)
{
    return radius * getPath(state1, state2, radius).length();
}
float DubinsStateSpace::symmetricDistance(const State *state1, const State *state2, float radius)
{
    return radius * std::min(getPath(state1, state2, radius).length(), getPath(state2, state1, radius).length());
}

void DubinsStateSpace::interpolate(const State *from, const State *to, const float t, State *state) const
{
    bool firstTime = true;
    PathType path;
    interpolate(from, to, t, firstTime, path, state);
}

void DubinsStateSpace::interpolate(const State *from, const State *to, const float t, bool &firstTime,
                                   PathType &path, State *state) const
{
    if (firstTime)
    {
        if (t >= 1.f)
        {
            if (to != state)
                copyState(state, to);
            return;
        }
        if (t <= 0.0f)
        {
            if (from != state)
                copyState(state, from);
            return;
        }

        path = getPath(from, to);
        if (isSymmetric_)
        {
            PathType path2(getPath(to, from));
            if (path2.length() < path.length())
            {
                path2.reverse_ = true;
                path = path2;
            }
        }
        firstTime = false;
    }
    interpolate(from, path, t, state, rho_);
}

void DubinsStateSpace::interpolate(const State *from, const PathType &path, float t, State *state,
                                   float radius) const
{
    auto *s = allocState()->as<StateType>();
    float seg = t * path.length(), phi, v;

    s->setXY(0.0f, 0.0f);
    s->setYaw(from->as<StateType>()->getYaw());
    if (!path.reverse_)
    {
        for (unsigned int i = 0; i < 3 && seg > 0; ++i)
        {
            v = std::min(seg, path.length_[i]);
            phi = s->getYaw();
            seg -= v;
            switch (path.type_->at(i))
            {
                case DUBINS_LEFT:
                    s->setXY(s->getX() + sin(phi + v) - sin(phi), s->getY() - cos(phi + v) + cos(phi));
                    s->setYaw(phi + v);
                    break;
                case DUBINS_RIGHT:
                    s->setXY(s->getX() - sin(phi - v) + sin(phi), s->getY() + cos(phi - v) - cos(phi));
                    s->setYaw(phi - v);
                    break;
                case DUBINS_STRAIGHT:
                    s->setXY(s->getX() + v * cos(phi), s->getY() + v * sin(phi));
                    break;
            }
        }
    }
    else
    {
        for (unsigned int i = 0; i < 3 && seg > 0; ++i)
        {
            v = std::min(seg, path.length_[2 - i]);
            phi = s->getYaw();
            seg -= v;
            switch (path.type_->at(2 - i))
            {
                case DUBINS_LEFT:
                    s->setXY(s->getX() + sin(phi - v) - sin(phi), s->getY() - cos(phi - v) + cos(phi));
                    s->setYaw(phi - v);
                    break;
                case DUBINS_RIGHT:
                    s->setXY(s->getX() - sin(phi + v) + sin(phi), s->getY() + cos(phi + v) - cos(phi));
                    s->setYaw(phi + v);
                    break;
                case DUBINS_STRAIGHT:
                    s->setXY(s->getX() - v * cos(phi), s->getY() - v * sin(phi));
                    break;
            }
        }
    }
    state->as<StateType>()->setX(s->getX() * radius + from->as<StateType>()->getX());
    state->as<StateType>()->setY(s->getY() * radius + from->as<StateType>()->getY());
    getSubspace(1)->enforceBounds(s->as<SO2StateSpace::StateType>(1));
    state->as<StateType>()->setYaw(s->getYaw());
    freeState(s);
}

unsigned int DubinsStateSpace::validSegmentCount(const State *state1, const State *state2) const
{
    return StateSpace::validSegmentCount(state1, state2);
}

DubinsStateSpace::PathType DubinsStateSpace::getPath(const State *state1, const State *state2) const
{
    return getPath(state1, state2, rho_);
}

DubinsStateSpace::PathType DubinsStateSpace::getPath(const State *state1, const State *state2, float radius)
{
    const auto *s1 = static_cast<const DubinsStateSpace::StateType *>(state1);
    const auto *s2 = static_cast<const DubinsStateSpace::StateType *>(state2);
    float x1 = s1->getX(), y1 = s1->getY(), th1 = s1->getYaw();
    float x2 = s2->getX(), y2 = s2->getY(), th2 = s2->getYaw();
    float dx = x2 - x1, dy = y2 - y1, d = sqrt(dx * dx + dy * dy) / radius, th = atan2(dy, dx);
    float alpha = mod2pi(th1 - th), beta = mod2pi(th2 - th);
    return ::getPath(d, alpha, beta);
}
