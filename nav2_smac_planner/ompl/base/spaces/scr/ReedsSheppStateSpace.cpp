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

#include "ompl/base/spaces/ReedsSheppStateSpace.h"
#include "ompl/base/SpaceInformation.h"
#include "ompl/util/Exception.h"
#include <boost/math/constants/constants.hpp>

using namespace ompl::base;

namespace
{
    // The comments, variable names, etc. use the nomenclature from the Reeds & Shepp paper.

    const float pi = boost::math::constants::pi<float>();
    const float twopi = 2.f * pi;
#ifndef NDEBUG
    const float RS_EPS = 1e-4f;
#endif
    const float ZERO = 10 * std::numeric_limits<float>::epsilon();

    inline float mod2pi(float x)
    {
        float v = fmod(x, twopi);
        if (v < -pi)
            v += twopi;
        else if (v > pi)
            v -= twopi;
        return v;
    }
    inline void polar(float x, float y, float &r, float &theta)
    {
        r = sqrt(x * x + y * y);
        theta = atan2(y, x);
    }
    inline void tauOmega(float u, float v, float xi, float eta, float phi, float &tau, float &omega)
    {
        float delta = mod2pi(u - v), A = sin(u) - sin(delta), B = cos(u) - cos(delta) - 1.0f;
        float t1 = atan2(eta * A - xi * B, xi * A + eta * B), t2 = 2.f * (cos(delta) - cos(v) - cos(u)) + 3;
        tau = (t2 < 0) ? mod2pi(t1 + pi) : mod2pi(t1);
        omega = mod2pi(tau - u + v - phi);
    }

    // formula 8.1 in Reeds-Shepp paper
    inline bool LpSpLp(float x, float y, float phi, float &t, float &u, float &v)
    {
        polar(x - sin(phi), y - 1.0f + cos(phi), u, t);
        if (t >= -ZERO)
        {
            v = mod2pi(phi - t);
            if (v >= -ZERO)
            {
                assert(fabs(u * cos(t) + sin(phi) - x) < RS_EPS);
                assert(fabs(u * sin(t) - cos(phi) + 1 - y) < RS_EPS);
                assert(fabs(mod2pi(t + v - phi)) < RS_EPS);
                return true;
            }
        }
        return false;
    }
    // formula 8.2
    inline bool LpSpRp(float x, float y, float phi, float &t, float &u, float &v)
    {
        float t1, u1;
        polar(x + sin(phi), y - 1.0f - cos(phi), u1, t1);
        u1 = u1 * u1;
        if (u1 >= 4.0f)
        {
            float theta;
            u = sqrt(u1 - 4.0f);
            theta = atan2(2.f, u);
            t = mod2pi(t1 + theta);
            v = mod2pi(t - phi);
            assert(fabs(2 * sin(t) + u * cos(t) - sin(phi) - x) < RS_EPS);
            assert(fabs(-2 * cos(t) + u * sin(t) + cos(phi) + 1 - y) < RS_EPS);
            assert(fabs(mod2pi(t - v - phi)) < RS_EPS);
            return t >= -ZERO && v >= -ZERO;
        }
        return false;
    }
    void CSC(float x, float y, float phi, ReedsSheppStateSpace::PathType &path)
    {
        float t, u, v, Lmin = path.length(), L;
        if (LpSpLp(x, y, phi, t, u, v) && Lmin > (L = fabs(t) + fabs(u) + fabs(v)))
        {
            path = ReedsSheppStateSpace::PathType(ReedsSheppStateSpace::reedsSheppPathType[14], t, u, v);
            Lmin = L;
        }
        if (LpSpLp(-x, y, -phi, t, u, v) && Lmin > (L = fabs(t) + fabs(u) + fabs(v)))  // timeflip
        {
            path = ReedsSheppStateSpace::PathType(ReedsSheppStateSpace::reedsSheppPathType[14], -t, -u, -v);
            Lmin = L;
        }
        if (LpSpLp(x, -y, -phi, t, u, v) && Lmin > (L = fabs(t) + fabs(u) + fabs(v)))  // reflect
        {
            path = ReedsSheppStateSpace::PathType(ReedsSheppStateSpace::reedsSheppPathType[15], t, u, v);
            Lmin = L;
        }
        if (LpSpLp(-x, -y, phi, t, u, v) && Lmin > (L = fabs(t) + fabs(u) + fabs(v)))  // timeflip + reflect
        {
            path = ReedsSheppStateSpace::PathType(ReedsSheppStateSpace::reedsSheppPathType[15], -t, -u, -v);
            Lmin = L;
        }
        if (LpSpRp(x, y, phi, t, u, v) && Lmin > (L = fabs(t) + fabs(u) + fabs(v)))
        {
            path = ReedsSheppStateSpace::PathType(ReedsSheppStateSpace::reedsSheppPathType[12], t, u, v);
            Lmin = L;
        }
        if (LpSpRp(-x, y, -phi, t, u, v) && Lmin > (L = fabs(t) + fabs(u) + fabs(v)))  // timeflip
        {
            path = ReedsSheppStateSpace::PathType(ReedsSheppStateSpace::reedsSheppPathType[12], -t, -u, -v);
            Lmin = L;
        }
        if (LpSpRp(x, -y, -phi, t, u, v) && Lmin > (L = fabs(t) + fabs(u) + fabs(v)))  // reflect
        {
            path = ReedsSheppStateSpace::PathType(ReedsSheppStateSpace::reedsSheppPathType[13], t, u, v);
            Lmin = L;
        }
        if (LpSpRp(-x, -y, phi, t, u, v) && Lmin > (L = fabs(t) + fabs(u) + fabs(v)))  // timeflip + reflect
            path = ReedsSheppStateSpace::PathType(ReedsSheppStateSpace::reedsSheppPathType[13], -t, -u, -v);
    }
    // formula 8.3 / 8.4  *** TYPO IN PAPER ***
    inline bool LpRmL(float x, float y, float phi, float &t, float &u, float &v)
    {
        float xi = x - sin(phi), eta = y - 1.0f + cos(phi), u1, theta;
        polar(xi, eta, u1, theta);
        if (u1 <= 4.0f)
        {
            u = -2.f * asin(0.25f * u1);
            t = mod2pi(theta + 0.5f * u + pi);
            v = mod2pi(phi - t + u);
            assert(fabs(2 * (sin(t) - sin(t - u)) + sin(phi) - x) < RS_EPS);
            assert(fabs(2 * (-cos(t) + cos(t - u)) - cos(phi) + 1 - y) < RS_EPS);
            assert(fabs(mod2pi(t - u + v - phi)) < RS_EPS);
            return t >= -ZERO && u <= ZERO;
        }
        return false;
    }
    void CCC(float x, float y, float phi, ReedsSheppStateSpace::PathType &path)
    {
        float t, u, v, Lmin = path.length(), L;
        if (LpRmL(x, y, phi, t, u, v) && Lmin > (L = fabs(t) + fabs(u) + fabs(v)))
        {
            path = ReedsSheppStateSpace::PathType(ReedsSheppStateSpace::reedsSheppPathType[0], t, u, v);
            Lmin = L;
        }
        if (LpRmL(-x, y, -phi, t, u, v) && Lmin > (L = fabs(t) + fabs(u) + fabs(v)))  // timeflip
        {
            path = ReedsSheppStateSpace::PathType(ReedsSheppStateSpace::reedsSheppPathType[0], -t, -u, -v);
            Lmin = L;
        }
        if (LpRmL(x, -y, -phi, t, u, v) && Lmin > (L = fabs(t) + fabs(u) + fabs(v)))  // reflect
        {
            path = ReedsSheppStateSpace::PathType(ReedsSheppStateSpace::reedsSheppPathType[1], t, u, v);
            Lmin = L;
        }
        if (LpRmL(-x, -y, phi, t, u, v) && Lmin > (L = fabs(t) + fabs(u) + fabs(v)))  // timeflip + reflect
        {
            path = ReedsSheppStateSpace::PathType(ReedsSheppStateSpace::reedsSheppPathType[1], -t, -u, -v);
            Lmin = L;
        }

        // backwards
        float xb = x * cos(phi) + y * sin(phi), yb = x * sin(phi) - y * cos(phi);
        if (LpRmL(xb, yb, phi, t, u, v) && Lmin > (L = fabs(t) + fabs(u) + fabs(v)))
        {
            path = ReedsSheppStateSpace::PathType(ReedsSheppStateSpace::reedsSheppPathType[0], v, u, t);
            Lmin = L;
        }
        if (LpRmL(-xb, yb, -phi, t, u, v) && Lmin > (L = fabs(t) + fabs(u) + fabs(v)))  // timeflip
        {
            path = ReedsSheppStateSpace::PathType(ReedsSheppStateSpace::reedsSheppPathType[0], -v, -u, -t);
            Lmin = L;
        }
        if (LpRmL(xb, -yb, -phi, t, u, v) && Lmin > (L = fabs(t) + fabs(u) + fabs(v)))  // reflect
        {
            path = ReedsSheppStateSpace::PathType(ReedsSheppStateSpace::reedsSheppPathType[1], v, u, t);
            Lmin = L;
        }
        if (LpRmL(-xb, -yb, phi, t, u, v) && Lmin > (L = fabs(t) + fabs(u) + fabs(v)))  // timeflip + reflect
            path = ReedsSheppStateSpace::PathType(ReedsSheppStateSpace::reedsSheppPathType[1], -v, -u, -t);
    }
    // formula 8.7
    inline bool LpRupLumRm(float x, float y, float phi, float &t, float &u, float &v)
    {
        float xi = x + sin(phi), eta = y - 1.0f - cos(phi), rho = 0.25f * (2.f + sqrt(xi * xi + eta * eta));
        if (rho <= 1.0f)
        {
            u = acos(rho);
            tauOmega(u, -u, xi, eta, phi, t, v);
            assert(fabs(2 * (sin(t) - sin(t - u) + sin(t - 2 * u)) - sin(phi) - x) < RS_EPS);
            assert(fabs(2 * (-cos(t) + cos(t - u) - cos(t - 2 * u)) + cos(phi) + 1 - y) < RS_EPS);
            assert(fabs(mod2pi(t - 2 * u - v - phi)) < RS_EPS);
            return t >= -ZERO && v <= ZERO;
        }
        return false;
    }
    // formula 8.8
    inline bool LpRumLumRp(float x, float y, float phi, float &t, float &u, float &v)
    {
        float xi = x + sin(phi), eta = y - 1.0f - cos(phi), rho = (20. - xi * xi - eta * eta) / 16.;
        if (rho >= 0 && rho <= 1)
        {
            u = -acos(rho);
            if (u >= -0.5f * pi)
            {
                tauOmega(u, u, xi, eta, phi, t, v);
                assert(fabs(4 * sin(t) - 2 * sin(t - u) - sin(phi) - x) < RS_EPS);
                assert(fabs(-4 * cos(t) + 2 * cos(t - u) + cos(phi) + 1 - y) < RS_EPS);
                assert(fabs(mod2pi(t - v - phi)) < RS_EPS);
                return t >= -ZERO && v >= -ZERO;
            }
        }
        return false;
    }
    void CCCC(float x, float y, float phi, ReedsSheppStateSpace::PathType &path)
    {
        float t, u, v, Lmin = path.length(), L;
        if (LpRupLumRm(x, y, phi, t, u, v) && Lmin > (L = fabs(t) + 2.f * fabs(u) + fabs(v)))
        {
            path = ReedsSheppStateSpace::PathType(ReedsSheppStateSpace::reedsSheppPathType[2], t, u, -u, v);
            Lmin = L;
        }
        if (LpRupLumRm(-x, y, -phi, t, u, v) && Lmin > (L = fabs(t) + 2.f * fabs(u) + fabs(v)))  // timeflip
        {
            path = ReedsSheppStateSpace::PathType(ReedsSheppStateSpace::reedsSheppPathType[2], -t, -u, u, -v);
            Lmin = L;
        }
        if (LpRupLumRm(x, -y, -phi, t, u, v) && Lmin > (L = fabs(t) + 2.f * fabs(u) + fabs(v)))  // reflect
        {
            path = ReedsSheppStateSpace::PathType(ReedsSheppStateSpace::reedsSheppPathType[3], t, u, -u, v);
            Lmin = L;
        }
        if (LpRupLumRm(-x, -y, phi, t, u, v) && Lmin > (L = fabs(t) + 2.f * fabs(u) + fabs(v)))  // timeflip + reflect
        {
            path = ReedsSheppStateSpace::PathType(ReedsSheppStateSpace::reedsSheppPathType[3], -t, -u, u, -v);
            Lmin = L;
        }

        if (LpRumLumRp(x, y, phi, t, u, v) && Lmin > (L = fabs(t) + 2.f * fabs(u) + fabs(v)))
        {
            path = ReedsSheppStateSpace::PathType(ReedsSheppStateSpace::reedsSheppPathType[2], t, u, u, v);
            Lmin = L;
        }
        if (LpRumLumRp(-x, y, -phi, t, u, v) && Lmin > (L = fabs(t) + 2.f * fabs(u) + fabs(v)))  // timeflip
        {
            path = ReedsSheppStateSpace::PathType(ReedsSheppStateSpace::reedsSheppPathType[2], -t, -u, -u, -v);
            Lmin = L;
        }
        if (LpRumLumRp(x, -y, -phi, t, u, v) && Lmin > (L = fabs(t) + 2.f * fabs(u) + fabs(v)))  // reflect
        {
            path = ReedsSheppStateSpace::PathType(ReedsSheppStateSpace::reedsSheppPathType[3], t, u, u, v);
            Lmin = L;
        }
        if (LpRumLumRp(-x, -y, phi, t, u, v) && Lmin > (L = fabs(t) + 2.f * fabs(u) + fabs(v)))  // timeflip + reflect
            path = ReedsSheppStateSpace::PathType(ReedsSheppStateSpace::reedsSheppPathType[3], -t, -u, -u, -v);
    }
    // formula 8.9
    inline bool LpRmSmLm(float x, float y, float phi, float &t, float &u, float &v)
    {
        float xi = x - sin(phi), eta = y - 1.0f + cos(phi), rho, theta;
        polar(xi, eta, rho, theta);
        if (rho >= 2.f)
        {
            float r = sqrt(rho * rho - 4.0f);
            u = 2.f - r;
            t = mod2pi(theta + atan2(r, -2.f));
            v = mod2pi(phi - 0.5f * pi - t);
            assert(fabs(2 * (sin(t) - cos(t)) - u * sin(t) + sin(phi) - x) < RS_EPS);
            assert(fabs(-2 * (sin(t) + cos(t)) + u * cos(t) - cos(phi) + 1 - y) < RS_EPS);
            assert(fabs(mod2pi(t + pi / 2 + v - phi)) < RS_EPS);
            return t >= -ZERO && u <= ZERO && v <= ZERO;
        }
        return false;
    }
    // formula 8.10
    inline bool LpRmSmRm(float x, float y, float phi, float &t, float &u, float &v)
    {
        float xi = x + sin(phi), eta = y - 1.0f - cos(phi), rho, theta;
        polar(-eta, xi, rho, theta);
        if (rho >= 2.f)
        {
            t = theta;
            u = 2.f - rho;
            v = mod2pi(t + 0.5f * pi - phi);
            assert(fabs(2 * sin(t) - cos(t - v) - u * sin(t) - x) < RS_EPS);
            assert(fabs(-2 * cos(t) - sin(t - v) + u * cos(t) + 1 - y) < RS_EPS);
            assert(fabs(mod2pi(t + pi / 2 - v - phi)) < RS_EPS);
            return t >= -ZERO && u <= ZERO && v <= ZERO;
        }
        return false;
    }
    void CCSC(float x, float y, float phi, ReedsSheppStateSpace::PathType &path)
    {
        float t, u, v, Lmin = path.length() - 0.5f * pi, L;
        if (LpRmSmLm(x, y, phi, t, u, v) && Lmin > (L = fabs(t) + fabs(u) + fabs(v)))
        {
            path = ReedsSheppStateSpace::PathType(ReedsSheppStateSpace::reedsSheppPathType[4], t, -0.5f * pi, u, v);
            Lmin = L;
        }
        if (LpRmSmLm(-x, y, -phi, t, u, v) && Lmin > (L = fabs(t) + fabs(u) + fabs(v)))  // timeflip
        {
            path =
                ReedsSheppStateSpace::PathType(ReedsSheppStateSpace::reedsSheppPathType[4], -t, 0.5f * pi, -u, -v);
            Lmin = L;
        }
        if (LpRmSmLm(x, -y, -phi, t, u, v) && Lmin > (L = fabs(t) + fabs(u) + fabs(v)))  // reflect
        {
            path = ReedsSheppStateSpace::PathType(ReedsSheppStateSpace::reedsSheppPathType[5], t, -0.5f * pi, u, v);
            Lmin = L;
        }
        if (LpRmSmLm(-x, -y, phi, t, u, v) && Lmin > (L = fabs(t) + fabs(u) + fabs(v)))  // timeflip + reflect
        {
            path =
                ReedsSheppStateSpace::PathType(ReedsSheppStateSpace::reedsSheppPathType[5], -t, 0.5f * pi, -u, -v);
            Lmin = L;
        }

        if (LpRmSmRm(x, y, phi, t, u, v) && Lmin > (L = fabs(t) + fabs(u) + fabs(v)))
        {
            path = ReedsSheppStateSpace::PathType(ReedsSheppStateSpace::reedsSheppPathType[8], t, -0.5f * pi, u, v);
            Lmin = L;
        }
        if (LpRmSmRm(-x, y, -phi, t, u, v) && Lmin > (L = fabs(t) + fabs(u) + fabs(v)))  // timeflip
        {
            path =
                ReedsSheppStateSpace::PathType(ReedsSheppStateSpace::reedsSheppPathType[8], -t, 0.5f * pi, -u, -v);
            Lmin = L;
        }
        if (LpRmSmRm(x, -y, -phi, t, u, v) && Lmin > (L = fabs(t) + fabs(u) + fabs(v)))  // reflect
        {
            path = ReedsSheppStateSpace::PathType(ReedsSheppStateSpace::reedsSheppPathType[9], t, -0.5f * pi, u, v);
            Lmin = L;
        }
        if (LpRmSmRm(-x, -y, phi, t, u, v) && Lmin > (L = fabs(t) + fabs(u) + fabs(v)))  // timeflip + reflect
        {
            path =
                ReedsSheppStateSpace::PathType(ReedsSheppStateSpace::reedsSheppPathType[9], -t, 0.5f * pi, -u, -v);
            Lmin = L;
        }

        // backwards
        float xb = x * cos(phi) + y * sin(phi), yb = x * sin(phi) - y * cos(phi);
        if (LpRmSmLm(xb, yb, phi, t, u, v) && Lmin > (L = fabs(t) + fabs(u) + fabs(v)))
        {
            path = ReedsSheppStateSpace::PathType(ReedsSheppStateSpace::reedsSheppPathType[6], v, u, -0.5f * pi, t);
            Lmin = L;
        }
        if (LpRmSmLm(-xb, yb, -phi, t, u, v) && Lmin > (L = fabs(t) + fabs(u) + fabs(v)))  // timeflip
        {
            path =
                ReedsSheppStateSpace::PathType(ReedsSheppStateSpace::reedsSheppPathType[6], -v, -u, 0.5f * pi, -t);
            Lmin = L;
        }
        if (LpRmSmLm(xb, -yb, -phi, t, u, v) && Lmin > (L = fabs(t) + fabs(u) + fabs(v)))  // reflect
        {
            path = ReedsSheppStateSpace::PathType(ReedsSheppStateSpace::reedsSheppPathType[7], v, u, -0.5f * pi, t);
            Lmin = L;
        }
        if (LpRmSmLm(-xb, -yb, phi, t, u, v) && Lmin > (L = fabs(t) + fabs(u) + fabs(v)))  // timeflip + reflect
        {
            path =
                ReedsSheppStateSpace::PathType(ReedsSheppStateSpace::reedsSheppPathType[7], -v, -u, 0.5f * pi, -t);
            Lmin = L;
        }

        if (LpRmSmRm(xb, yb, phi, t, u, v) && Lmin > (L = fabs(t) + fabs(u) + fabs(v)))
        {
            path =
                ReedsSheppStateSpace::PathType(ReedsSheppStateSpace::reedsSheppPathType[10], v, u, -0.5f * pi, t);
            Lmin = L;
        }
        if (LpRmSmRm(-xb, yb, -phi, t, u, v) && Lmin > (L = fabs(t) + fabs(u) + fabs(v)))  // timeflip
        {
            path =
                ReedsSheppStateSpace::PathType(ReedsSheppStateSpace::reedsSheppPathType[10], -v, -u, 0.5f * pi, -t);
            Lmin = L;
        }
        if (LpRmSmRm(xb, -yb, -phi, t, u, v) && Lmin > (L = fabs(t) + fabs(u) + fabs(v)))  // reflect
        {
            path =
                ReedsSheppStateSpace::PathType(ReedsSheppStateSpace::reedsSheppPathType[11], v, u, -0.5f * pi, t);
            Lmin = L;
        }
        if (LpRmSmRm(-xb, -yb, phi, t, u, v) && Lmin > (L = fabs(t) + fabs(u) + fabs(v)))  // timeflip + reflect
            path =
                ReedsSheppStateSpace::PathType(ReedsSheppStateSpace::reedsSheppPathType[11], -v, -u, 0.5f * pi, -t);
    }
    // formula 8.11 *** TYPO IN PAPER ***
    inline bool LpRmSLmRp(float x, float y, float phi, float &t, float &u, float &v)
    {
        float xi = x + sin(phi), eta = y - 1.0f - cos(phi), rho, theta;
        polar(xi, eta, rho, theta);
        if (rho >= 2.f)
        {
            u = 4.0f - sqrt(rho * rho - 4.0f);
            if (u <= ZERO)
            {
                t = mod2pi(atan2((4 - u) * xi - 2 * eta, -2 * xi + (u - 4) * eta));
                v = mod2pi(t - phi);
                assert(fabs(4 * sin(t) - 2 * cos(t) - u * sin(t) - sin(phi) - x) < RS_EPS);
                assert(fabs(-4 * cos(t) - 2 * sin(t) + u * cos(t) + cos(phi) + 1 - y) < RS_EPS);
                assert(fabs(mod2pi(t - v - phi)) < RS_EPS);
                return t >= -ZERO && v >= -ZERO;
            }
        }
        return false;
    }
    void CCSCC(float x, float y, float phi, ReedsSheppStateSpace::PathType &path)
    {
        float t, u, v, Lmin = path.length() - pi, L;
        if (LpRmSLmRp(x, y, phi, t, u, v) && Lmin > (L = fabs(t) + fabs(u) + fabs(v)))
        {
            path = ReedsSheppStateSpace::PathType(ReedsSheppStateSpace::reedsSheppPathType[16], t, -0.5f * pi, u,
                                                        -0.5f * pi, v);
            Lmin = L;
        }
        if (LpRmSLmRp(-x, y, -phi, t, u, v) && Lmin > (L = fabs(t) + fabs(u) + fabs(v)))  // timeflip
        {
            path = ReedsSheppStateSpace::PathType(ReedsSheppStateSpace::reedsSheppPathType[16], -t, 0.5f * pi, -u,
                                                        0.5f * pi, -v);
            Lmin = L;
        }
        if (LpRmSLmRp(x, -y, -phi, t, u, v) && Lmin > (L = fabs(t) + fabs(u) + fabs(v)))  // reflect
        {
            path = ReedsSheppStateSpace::PathType(ReedsSheppStateSpace::reedsSheppPathType[17], t, -0.5f * pi, u,
                                                        -0.5f * pi, v);
            Lmin = L;
        }
        if (LpRmSLmRp(-x, -y, phi, t, u, v) && Lmin > (L = fabs(t) + fabs(u) + fabs(v)))  // timeflip + reflect
            path = ReedsSheppStateSpace::PathType(ReedsSheppStateSpace::reedsSheppPathType[17], -t, 0.5f * pi, -u,
                                                        0.5f * pi, -v);
    }

    ReedsSheppStateSpace::PathType getPath(float x, float y, float phi)
    {
        ReedsSheppStateSpace::PathType path;
        CSC(x, y, phi, path);
        CCC(x, y, phi, path);
        CCCC(x, y, phi, path);
        CCSC(x, y, phi, path);
        CCSCC(x, y, phi, path);
        return path;
    }
}

const ompl::base::ReedsSheppStateSpace::ReedsSheppPathSegmentType
    ompl::base::ReedsSheppStateSpace::reedsSheppPathType[18][5] = {
        {RS_LEFT, RS_RIGHT, RS_LEFT, RS_NOP, RS_NOP},         // 0
        {RS_RIGHT, RS_LEFT, RS_RIGHT, RS_NOP, RS_NOP},        // 1
        {RS_LEFT, RS_RIGHT, RS_LEFT, RS_RIGHT, RS_NOP},       // 2
        {RS_RIGHT, RS_LEFT, RS_RIGHT, RS_LEFT, RS_NOP},       // 3
        {RS_LEFT, RS_RIGHT, RS_STRAIGHT, RS_LEFT, RS_NOP},    // 4
        {RS_RIGHT, RS_LEFT, RS_STRAIGHT, RS_RIGHT, RS_NOP},   // 5
        {RS_LEFT, RS_STRAIGHT, RS_RIGHT, RS_LEFT, RS_NOP},    // 6
        {RS_RIGHT, RS_STRAIGHT, RS_LEFT, RS_RIGHT, RS_NOP},   // 7
        {RS_LEFT, RS_RIGHT, RS_STRAIGHT, RS_RIGHT, RS_NOP},   // 8
        {RS_RIGHT, RS_LEFT, RS_STRAIGHT, RS_LEFT, RS_NOP},    // 9
        {RS_RIGHT, RS_STRAIGHT, RS_RIGHT, RS_LEFT, RS_NOP},   // 10
        {RS_LEFT, RS_STRAIGHT, RS_LEFT, RS_RIGHT, RS_NOP},    // 11
        {RS_LEFT, RS_STRAIGHT, RS_RIGHT, RS_NOP, RS_NOP},     // 12
        {RS_RIGHT, RS_STRAIGHT, RS_LEFT, RS_NOP, RS_NOP},     // 13
        {RS_LEFT, RS_STRAIGHT, RS_LEFT, RS_NOP, RS_NOP},      // 14
        {RS_RIGHT, RS_STRAIGHT, RS_RIGHT, RS_NOP, RS_NOP},    // 15
        {RS_LEFT, RS_RIGHT, RS_STRAIGHT, RS_LEFT, RS_RIGHT},  // 16
        {RS_RIGHT, RS_LEFT, RS_STRAIGHT, RS_RIGHT, RS_LEFT}   // 17
};

ompl::base::ReedsSheppStateSpace::PathType::PathType(const ReedsSheppPathSegmentType *type, float t,
                                                                 float u, float v, float w, float x)
  : type_(type)
{
    length_[0] = t;
    length_[1] = u;
    length_[2] = v;
    length_[3] = w;
    length_[4] = x;
    totalLength_ = fabs(t) + fabs(u) + fabs(v) + fabs(w) + fabs(x);
}

float ompl::base::ReedsSheppStateSpace::distance(const State *state1, const State *state2) const
{
    return rho_ * getPath(state1, state2).length();
}

void ompl::base::ReedsSheppStateSpace::interpolate(const State *from, const State *to, const float t,
                                                   State *state) const
{
    bool firstTime = true;
    PathType path;
    interpolate(from, to, t, firstTime, path, state);
}

void ompl::base::ReedsSheppStateSpace::interpolate(const State *from, const State *to, const float t, bool &firstTime,
    PathType &path, State *state) const
{
    if (firstTime)
    {
        if (t >= 1.0f)
        {
            if (to != state)
                copyState(state, to);
            return;
        }
        if (t <= 0.f)
        {
            if (from != state)
                copyState(state, from);
            return;
        }
        path = getPath(from, to);
        firstTime = false;
    }
    interpolate(from, path, t, state);
}

void ompl::base::ReedsSheppStateSpace::interpolate(const State *from, const PathType &path, float t,
                                                   State *state) const
{
    auto *s = allocState()->as<StateType>();
    float seg = t * path.length(), phi, v;

    s->setXY(0.f, 0.f);
    s->setYaw(from->as<StateType>()->getYaw());
    for (unsigned int i = 0; i < 5 && seg > 0; ++i)
    {
        if (path.length_[i] < 0)
        {
            v = std::max(-seg, path.length_[i]);
            seg += v;
        }
        else
        {
            v = std::min(seg, path.length_[i]);
            seg -= v;
        }
        phi = s->getYaw();
        switch (path.type_[i])
        {
            case RS_LEFT:
                s->setXY(s->getX() + sin(phi + v) - sin(phi), s->getY() - cos(phi + v) + cos(phi));
                s->setYaw(phi + v);
                break;
            case RS_RIGHT:
                s->setXY(s->getX() - sin(phi - v) + sin(phi), s->getY() + cos(phi - v) - cos(phi));
                s->setYaw(phi - v);
                break;
            case RS_STRAIGHT:
                s->setXY(s->getX() + v * cos(phi), s->getY() + v * sin(phi));
                break;
            case RS_NOP:
                break;
        }
    }
    state->as<StateType>()->setX(s->getX() * rho_ + from->as<StateType>()->getX());
    state->as<StateType>()->setY(s->getY() * rho_ + from->as<StateType>()->getY());
    getSubspace(1)->enforceBounds(s->as<SO2StateSpace::StateType>(1));
    state->as<StateType>()->setYaw(s->getYaw());
    freeState(s);
}

ompl::base::ReedsSheppStateSpace::PathType ompl::base::ReedsSheppStateSpace::getPath(const State *state1,
                                                                                              const State *state2) const
{
    const auto *s1 = static_cast<const StateType *>(state1);
    const auto *s2 = static_cast<const StateType *>(state2);
    float x1 = s1->getX(), y1 = s1->getY(), th1 = s1->getYaw();
    float x2 = s2->getX(), y2 = s2->getY(), th2 = s2->getYaw();
    float dx = x2 - x1, dy = y2 - y1, c = cos(th1), s = sin(th1);
    float x = c * dx + s * dy, y = -s * dx + c * dy, phi = th2 - th1;
    return ::getPath(x / rho_, y / rho_, phi);
}
