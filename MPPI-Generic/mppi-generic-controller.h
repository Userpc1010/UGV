#ifndef MPPIGENERICCONTROLLER_H
#define MPPIGENERICCONTROLLER_H

#include <cuda_runtime.h>
#include <cuda.h>
#include <memory.h>
#include <memory>
#include <iostream>
#include <assert.h>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <stdio.h>
#include <math.h>
#include <cmath>

#include <array>
#include <cfloat>
#include <type_traits>
#include <vector>
#include <map>
#include <string>

#include <cufft.h>
#include <curand.h>
#include <device_launch_parameters.h>  // For block idx and thread idx, etc

#include <cstdarg>
#include <cstdio>
#include <cassert>

#include <chrono>
#include <type_traits>
#include <unordered_set>
#include <random>
#include <algorithm>
#include <stdexcept>
#include <utility>
















namespace mppi
{
class RiskMeasure
{
public:
  enum FUNC_TYPE : int
  {
    MAX = 0,
    MIN,
    VAR,
    CVAR,
    MEAN,
    MEDIAN,
    NUM_FUNCS
  };

  FUNC_TYPE func_ = CVAR;
  float alpha_ = 0.8;

  __host__ __device__ float shaping_func(float* __restrict__ costs, const int num_costs)
  {
    return shaping_func(costs, num_costs, func_, alpha_);
  }

  static __host__ __device__ float shaping_func(float* __restrict__ costs, const int num_costs,
                                                const FUNC_TYPE type = MEAN, const float risk_tolerance = 0.5f)
  {
    float cost = 0.0f;
    if (num_costs == 1)
    {
      return costs[0];
    }
    switch (type)
    {
      case CVAR:
        cost = cvar(costs, num_costs, risk_tolerance);
        break;
      case MAX:
        cost = max_measure(costs, num_costs);
        break;
      case MEDIAN:
        cost = var(costs, num_costs, 0.5f);
        break;
      case MIN:
        cost = min_measure(costs, num_costs);
        break;
      case VAR:
        cost = var(costs, num_costs, risk_tolerance);
        break;
      default: // go to mean case
      case MEAN:
        cost = mean_measure(costs, num_costs);
        break;
    }
    return cost;
  }

  static __host__ __device__ float max_measure(const float* __restrict__ costs, const int num_costs)
  {
    float max_cost = costs[0];
    for (int i = 1; i < num_costs; i++)
    {
      if (costs[i] > max_cost)
      {
        max_cost = costs[i];
      }
    }
    return max_cost;
  }

  static __host__ __device__ float min_measure(const float* __restrict__ costs, const int num_costs)
  {
    float min_cost = costs[0];
    for (int i = 1; i < num_costs; i++)
    {
      if (costs[i] < min_cost)
      {
        min_cost = costs[i];
      }
    }
    return min_cost;
  }

  static __host__ __device__ float mean_measure(const float* __restrict__ costs, const int num_costs)
  {
    float cost = 0.0f;
    for (int i = 0; i < num_costs; i++)
    {
      cost += costs[i];
    }
    return cost / num_costs;
  }

  static __host__ __device__ float h_index(const int num_costs, const float alpha)
  {
    return alpha * (num_costs - 1);
  }

  static __host__ __device__ float var(float* __restrict__ costs, const int num_costs, float alpha);

  static __host__ __device__ float cvar(float* __restrict__ costs, const int num_costs, float alpha);
};
}  // namespace mppi







namespace angle_utils
{
/**
 *
 * @param angle
 * @return
 */
__host__ __device__ static inline double normalizeAngle(double angle)
{
  const double result = fmod(angle + M_PI, 2.0 * M_PI);
  if (result <= 0.0)
    return result + M_PI;
  return result - M_PI;
}
// float version might not be exact
// different systems will have slightly different values.
__host__ __device__ static inline float normalizeAngle(float angle)
{
  const float result = fmodf(angle + static_cast<float>(M_PI), static_cast<float>(2.0 * M_PI));
  if (result <= 0.0f)
    return result + static_cast<float>(M_PI);
  return result - static_cast<float>(M_PI);
}

/**
 *
 * @param from
 * @param to
 * @return
 */
__host__ __device__ static inline double shortestAngularDistance(double from, double to)
{
  return normalizeAngle(to - from);
}
__host__ __device__ static inline float shortestAngularDistance(float from, float to)
{
  return normalizeAngle(to - from);
}

/**
 * Does a linear interpolation of the euler angle while respecting -pi to pi wrapping
 * solution from https://www.ri.cmu.edu/pub_files/pub4/kuffner_james_2004_1/kuffner_james_2004_1.pdf algorithm 6
 * @param angle_1
 * @param angle_2
 * @param alpha
 * @return
 */
__host__ __device__ static inline double interpolateEulerAngleLinear(double angle_1, double angle_2, double alpha)
{
  double angle_diff = shortestAngularDistance(angle_1, angle_2);
  return normalizeAngle(angle_1 + alpha * angle_diff);
}
__host__ __device__ static inline float interpolateEulerAngleLinear(float angle_1, float angle_2, float alpha)
{
  float angle_diff = shortestAngularDistance(angle_1, angle_2);
  return normalizeAngle(angle_1 + alpha * angle_diff);
}
}  // namespace angle_utils
















#ifndef __UNROLL
#define __xstr__(s) __str__(s)
#define __str__(s) #s
#ifdef __CUDACC__
#define __UNROLL(a) _Pragma("unroll")
#elif defined(__GNUC__)  // GCC is the compiler and uses different unroll syntax
#define __UNROLL(a) _Pragma(__xstr__(GCC unroll a))
#endif
#endif

// Matching float4 syntax
template <class T = float>
struct __align__(4 * sizeof(T)) type4
{
  T x;
  T y;
  T z;
  T w;
  // Allow writing to struct using array index
  __host__ __device__ T& operator[](int i)
  {
    assert(i >= 0);
    assert(i < 4);
    return (i > 1) ? ((i == 2) ? z : w) : ((i == 0) ? x : y);
  }
  // Allow reading from struct using array index
  __host__ __device__ const T& operator[](int i) const
  {
    assert(i >= 0);
    assert(i < 4);
    return (i > 1) ? ((i == 2) ? z : w) : ((i == 0) ? x : y);
  }
};

template <class T = float>
struct __align__(2 * sizeof(T)) type2
{
  T x;
  T y;

  // Allow writing to struct using array index
  __host__ __device__ T& operator[](int i)
  {
    assert(i >= 0);
    assert(i < 2);
    return (i == 0) ? x : y;
  }
  // Allow reading from struct using array index
  __host__ __device__ const T& operator[](int i) const
  {
    assert(i >= 0);
    assert(i < 2);
    return (i == 0) ? x : y;
  }
};

// Scalar-Vector Multiplication
__host__ __device__ inline float2 operator*(const float2& a, const float& b)
{
  return make_float2(a.x * b, a.y * b);
}
__host__ __device__ inline float3 operator*(const float3& a, const float& b)
{
  return make_float3(a.x * b, a.y * b, a.z * b);
}
__host__ __device__ inline float4 operator*(const float4& a, const float& b)
{
  return make_float4(a.x * b, a.y * b, a.z * b, a.w * b);
}

__host__ __device__ inline float2 operator*(const float& b, const float2& a)
{
  return make_float2(a.x * b, a.y * b);
}
__host__ __device__ inline float3 operator*(const float& b, const float3& a)
{
  return make_float3(a.x * b, a.y * b, a.z * b);
}
__host__ __device__ inline float4 operator*(const float& b, const float4& a)
{
  return make_float4(a.x * b, a.y * b, a.z * b, a.w * b);
}

// Scalar-Vector Addition
__host__ __device__ inline float2 operator+(const float2& a, const float& b)
{
  return make_float2(a.x + b, a.y + b);
}
__host__ __device__ inline float3 operator+(const float3& a, const float& b)
{
  return make_float3(a.x + b, a.y + b, a.z + b);
}
__host__ __device__ inline float4 operator+(const float4& a, const float& b)
{
  return make_float4(a.x + b, a.y + b, a.z + b, a.w + b);
}

// Scalar-Vector Subtraction
__host__ __device__ inline float2 operator-(const float2& a, const float& b)
{
  return make_float2(a.x - b, a.y - b);
}
__host__ __device__ inline float3 operator-(const float3& a, const float& b)
{
  return make_float3(a.x - b, a.y - b, a.z - b);
}
__host__ __device__ inline float4 operator-(const float4& a, const float& b)
{
  return make_float4(a.x - b, a.y - b, a.z - b, a.w - b);
}

// Scalar-Vector Dvision
__host__ __device__ inline float2 operator/(const float2& a, const float& b)
{
  return make_float2(a.x / b, a.y / b);
}
__host__ __device__ inline float3 operator/(const float3& a, const float& b)
{
  return make_float3(a.x / b, a.y / b, a.z / b);
}
__host__ __device__ inline float4 operator/(const float4& a, const float& b)
{
  return make_float4(a.x / b, a.y / b, a.z / b, a.w / b);
}

// Vector-Vector Addition
__host__ __device__ inline float2 operator+(const float2& a, const float2& b)
{
  return make_float2(a.x + b.x, a.y + b.y);
}
__host__ __device__ inline float3 operator+(const float3& a, const float3& b)
{
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__host__ __device__ inline float4 operator+(const float4& a, const float4& b)
{
  return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

// Vector-Vector Subtraction
__host__ __device__ inline float2 operator-(const float2& a, const float2& b)
{
  return make_float2(a.x - b.x, a.y - b.y);
}

__host__ __device__ inline float3 operator-(const float3& a, const float3& b)
{
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ inline float4 operator-(const float4& a, const float4& b)
{
  return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

// Vector-Vector Multiplication
__host__ __device__ inline float2 operator*(const float2& a, const float2& b)
{
  return make_float2(a.x * b.x, a.y * b.y);
}

__host__ __device__ inline float3 operator*(const float3& a, const float3& b)
{
  return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__host__ __device__ inline float4 operator*(const float4& a, const float4& b)
{
  return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}

// Vector-Vector Division
__host__ __device__ inline float2 operator/(const float2& a, const float2& b)
{
  return make_float2(a.x / b.x, a.y / b.y);
}

__host__ __device__ inline float3 operator/(const float3& a, const float3& b)
{
  return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}

__host__ __device__ inline float4 operator/(const float4& a, const float4& b)
{
  return make_float4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}

// Scalar-Vector multiply and set
__host__ __device__ inline float2& operator*=(float2& a, const float& b)
{
  a.x *= b;
  a.y *= b;
  return a;
}
__host__ __device__ inline float3& operator*=(float3& a, const float& b)
{
  a.x *= b;
  a.y *= b;
  a.z *= b;
  return a;
}
__host__ __device__ inline float4& operator*=(float4& a, const float& b)
{
  a.x *= b;
  a.y *= b;
  a.z *= b;
  a.w *= b;
  return a;
}

// Vector-Vector add and set
__host__ __device__ inline float2& operator+=(float2& a, const float2& b)
{
  a.x += b.x;
  a.y += b.y;
  return a;
}

__host__ __device__ inline float3& operator+=(float3& a, const float3& b)
{
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  return a;
}
__host__ __device__ inline float4& operator+=(float4& a, const float4& b)
{
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  a.w += b.w;
  return a;
}

__host__ __device__ inline float dot(const float2& a, const float2& b)
{
  return a.x * b.x + a.y * b.y;
}

__host__ __device__ inline float cross(const float2& a, const float2& b)
{
  return a.x * b.y - a.y * b.x;
}

__host__ __device__ inline float norm(const float2& a)
{
  return sqrtf(dot(a, a));
}

__host__ __device__ inline float dot(const float3& a, const float3& b)
{
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ inline float norm(const float3& a)
{
  return sqrtf(dot(a, a));
}

__host__ __device__ inline float dot(const float4& a, const float4& b)
{
  return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

__host__ __device__ inline float norm(const float4& a)
{
  return sqrtf(dot(a, a));
}

__host__ __device__ inline float2 operator-(const float2& a)
{
  return make_float2(-a.x, -a.y);
}

__host__ __device__ inline float3 operator-(const float3& a)
{
  return make_float3(-a.x, -a.y, -a.z);
}

__host__ __device__ inline float4 operator-(const float4& a)
{
  return make_float4(-a.x, -a.y, -a.z, -a.w);
}

__host__ __device__ inline bool operator==(const float2& lhs, const float2& rhs)
{
  return (lhs.x == rhs.x) && (lhs.y == rhs.y);
}

__host__ __device__ inline bool operator==(const float3& lhs, const float3& rhs)
{
  return (lhs.x == rhs.x) && (lhs.y == rhs.y) && (lhs.z == rhs.z);
}

__host__ __device__ inline bool operator==(const float4& lhs, const float4& rhs)
{
  return (lhs.x == rhs.x) && (lhs.y == rhs.y) && (lhs.z == rhs.z) && (lhs.w == rhs.w);
}

// Create different data types given the same input. Needed for handling border values in cudaTextures with Border
// adress mode
template <class DATA_T>
inline __host__ __device__ DATA_T createPartialCudaTuple(const float& r, const float& g, const float& b,
                                                         const float& a);

template <>
inline __host__ __device__ float createPartialCudaTuple<float>(const float& r, const float& g, const float& b,
                                                               const float& a)
{
  return r;
}

template <>
inline __host__ __device__ float2 createPartialCudaTuple<float2>(const float& r, const float& g, const float& b,
                                                                 const float& a)
{
  return make_float2(r, g);
}

template <>
inline __host__ __device__ float3 createPartialCudaTuple<float3>(const float& r, const float& g, const float& b,
                                                                 const float& a)
{
  return make_float3(r, g, b);
}

template <>
inline __host__ __device__ float4 createPartialCudaTuple<float4>(const float& r, const float& g, const float& b,
                                                                 const float& a)
{
  return make_float4(r, g, b, a);
}

template <>
inline __host__ __device__ int createPartialCudaTuple<int>(const float& r, const float& g, const float& b,
                                                           const float& a)
{
  return static_cast<int>(r);
}

template <>
inline __host__ __device__ int2 createPartialCudaTuple<int2>(const float& r, const float& g, const float& b,
                                                             const float& a)
{
  return make_int2(static_cast<int>(r), static_cast<int>(g));
}

template <>
inline __host__ __device__ int3 createPartialCudaTuple<int3>(const float& r, const float& g, const float& b,
                                                             const float& a)
{
  return make_int3(static_cast<int>(r), static_cast<int>(g), static_cast<int>(b));
}

template <>
inline __host__ __device__ int4 createPartialCudaTuple<int4>(const float& r, const float& g, const float& b,
                                                             const float& a)
{
  return make_int4(static_cast<int>(r), static_cast<int>(g), static_cast<int>(b), static_cast<int>(a));
}














namespace mppi
{
namespace p1  // parallelize to 1 index and step
{
enum class Parallel1Dir : int
{
  THREAD_X = 0,
  THREAD_Y,
  THREAD_Z,
  THREAD_XY,
  THREAD_YX,
  THREAD_XZ,
  THREAD_ZX,
  THREAD_YZ,
  THREAD_ZY,
  THREAD_XYZ,
  GLOBAL_X,
  GLOBAL_Y,
  GLOBAL_Z,
  NONE,
};

template <Parallel1Dir P_DIR>
inline __host__ __device__ void getParallel1DIndex(int& p_index, int& p_step);

template <>
inline __host__ __device__ void getParallel1DIndex<Parallel1Dir::THREAD_X>(int& p_index, int& p_step)
{
#ifdef __CUDA_ARCH__
  p_index = threadIdx.x;
  p_step = blockDim.x;
#else
  p_index = 0;
  p_step = 1;
#endif
}

template <>
inline __host__ __device__ void getParallel1DIndex<Parallel1Dir::THREAD_Y>(int& p_index, int& p_step)
{
#ifdef __CUDA_ARCH__
  p_index = threadIdx.y;
  p_step = blockDim.y;
#else
  p_index = 0;
  p_step = 1;
#endif
}

template <>
inline __host__ __device__ void getParallel1DIndex<Parallel1Dir::THREAD_Z>(int& p_index, int& p_step)
{
#ifdef __CUDA_ARCH__
  p_index = threadIdx.z;
  p_step = blockDim.z;
#else
  p_index = 0;
  p_step = 1;
#endif
}

template <>
inline __host__ __device__ void getParallel1DIndex<Parallel1Dir::THREAD_XY>(int& p_index, int& p_step)
{
#ifdef __CUDA_ARCH__
  p_index = threadIdx.x + blockDim.x * threadIdx.y;
  p_step = blockDim.x * blockDim.y;
#else
  p_index = 0;
  p_step = 1;
#endif
}

template <>
inline __host__ __device__ void getParallel1DIndex<Parallel1Dir::THREAD_XZ>(int& p_index, int& p_step)
{
#ifdef __CUDA_ARCH__
  p_index = threadIdx.x + blockDim.x * threadIdx.z;
  p_step = blockDim.x * blockDim.z;
#else
  p_index = 0;
  p_step = 1;
#endif
}

template <>
inline __host__ __device__ void getParallel1DIndex<Parallel1Dir::THREAD_YX>(int& p_index, int& p_step)
{
#ifdef __CUDA_ARCH__
  p_index = threadIdx.y + blockDim.y * threadIdx.x;
  p_step = blockDim.y * blockDim.x;
#else
  p_index = 0;
  p_step = 1;
#endif
}

template <>
inline __host__ __device__ void getParallel1DIndex<Parallel1Dir::THREAD_YZ>(int& p_index, int& p_step)
{
#ifdef __CUDA_ARCH__
  p_index = threadIdx.y + blockDim.y * threadIdx.z;
  p_step = blockDim.y * blockDim.z;
#else
  p_index = 0;
  p_step = 1;
#endif
}

template <>
inline __host__ __device__ void getParallel1DIndex<Parallel1Dir::THREAD_ZX>(int& p_index, int& p_step)
{
#ifdef __CUDA_ARCH__
  p_index = threadIdx.z + blockDim.z * threadIdx.x;
  p_step = blockDim.z * blockDim.x;
#else
  p_index = 0;
  p_step = 1;
#endif
}

template <>
inline __host__ __device__ void getParallel1DIndex<Parallel1Dir::THREAD_ZY>(int& p_index, int& p_step)
{
#ifdef __CUDA_ARCH__
  p_index = threadIdx.z + blockDim.z * threadIdx.y;
  p_step = blockDim.z * blockDim.y;
#else
  p_index = 0;
  p_step = 1;
#endif
}

template <>
inline __host__ __device__ void getParallel1DIndex<Parallel1Dir::THREAD_XYZ>(int& p_index, int& p_step)
{
#ifdef __CUDA_ARCH__
  p_index = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
  p_step = blockDim.x * blockDim.y * blockDim.z;
#else
  p_index = 0;
  p_step = 1;
#endif
}

template <>
inline __host__ __device__ void getParallel1DIndex<Parallel1Dir::GLOBAL_X>(int& p_index, int& p_step)
{
#ifdef __CUDA_ARCH__
  p_index = threadIdx.x + blockDim.x * blockIdx.x;
  p_step = gridDim.x * blockDim.x;
#else
  p_index = 0;
  p_step = 1;
#endif
}

template <>
inline __host__ __device__ void getParallel1DIndex<Parallel1Dir::GLOBAL_Y>(int& p_index, int& p_step)
{
#ifdef __CUDA_ARCH__
  p_index = threadIdx.y + blockDim.y * blockIdx.y;
  p_step = gridDim.y * blockDim.y;
#else
  p_index = 0;
  p_step = 1;
#endif
}

template <>
inline __host__ __device__ void getParallel1DIndex<Parallel1Dir::GLOBAL_Z>(int& p_index, int& p_step)
{
#ifdef __CUDA_ARCH__
  p_index = threadIdx.z + blockDim.z * blockIdx.z;
  p_step = gridDim.z * blockDim.z;
#else
  p_index = 0;
  p_step = 1;
#endif
}

template <>
inline __host__ __device__ void getParallel1DIndex<Parallel1Dir::NONE>(int& p_index, int& p_step)
{
  p_index = 0;
  p_step = 1;
}

template <Parallel1Dir P_DIR = Parallel1Dir::THREAD_Y, class T = float>
inline __device__ void loadArrayParallel(T* __restrict__ a1, const int off1, const T* __restrict__ a2, const int off2,
                                         const int N)
{
  int p_index, p_step;
  getParallel1DIndex<P_DIR>(p_index, p_step);
  if (N % 4 == 0 && sizeof(type4<T>) <= 16 && off1 % 4 == 0 && off2 % 4 == 0)
  {
    for (int i = p_index; i < N / 4; i += p_step)
    {
      reinterpret_cast<type4<T>*>(&a1[off1])[i] = reinterpret_cast<const type4<T>*>(&a2[off2])[i];
    }
  }
  else if (N % 2 == 0 && sizeof(type2<T>) <= 16 && off1 % 2 == 0 && off2 % 2 == 0)
  {
    for (int i = p_index; i < N / 2; i += p_step)
    {
      reinterpret_cast<type2<T>*>(&a1[off1])[i] = reinterpret_cast<const type2<T>*>(&a2[off2])[i];
    }
  }
  else
  {
    for (int i = p_index; i < N; i += p_step)
    {
      a1[off1 + i] = a2[off2 + i];
    }
  }
}

template <int N, Parallel1Dir P_DIR = Parallel1Dir::THREAD_Y, class T = float>
inline __device__ void loadArrayParallel(T* __restrict__ a1, const int off1, const T* __restrict__ a2, const int off2)
{
  loadArrayParallel<P_DIR, T>(a1, off1, a2, off2, N);
}
}  // namespace p1

namespace p2  // parallelize using 2 indices and steps
{
enum class Parallel2Dir : int
{
  THREAD_XY = 0,
  THREAD_XZ,
  THREAD_YZ,
  THREAD_YX,
  THREAD_ZX,
  THREAD_ZY,
  NONE
};

template <Parallel2Dir P_DIR>
inline __host__ __device__ void getParallel2DIndex(int& p1_index, int& p2_index, int& p1_step, int& p2_step)
{
#ifndef __CUDA_ARCH__
  p1_index = 0;
  p2_index = 0;
  p1_step = 1;
  p2_step = 1;
#endif
}

template <>
inline __device__ void getParallel2DIndex<Parallel2Dir::THREAD_XY>(int& p1_index, int& p2_index, int& p1_step,
                                                                   int& p2_step)
{
#ifdef __CUDA_ARCH__
  p1_index = threadIdx.x;
  p1_step = blockDim.x;
  p2_index = threadIdx.y;
  p2_step = blockDim.y;
#else
  p1_index = 0;
  p2_index = 0;
  p1_step = 1;
  p2_step = 1;
#endif
}

template <>
inline __device__ void getParallel2DIndex<Parallel2Dir::THREAD_YZ>(int& p1_index, int& p2_index, int& p1_step,
                                                                   int& p2_step)
{
#ifdef __CUDA_ARCH__
  p1_index = threadIdx.y;
  p1_step = blockDim.y;
  p2_index = threadIdx.z;
  p2_step = blockDim.z;
#else
  p1_index = 0;
  p2_index = 0;
  p1_step = 1;
  p2_step = 1;
#endif
}

template <>
inline __device__ void getParallel2DIndex<Parallel2Dir::THREAD_XZ>(int& p1_index, int& p2_index, int& p1_step,
                                                                   int& p2_step)
{
#ifdef __CUDA_ARCH__
  p1_index = threadIdx.x;
  p1_step = blockDim.x;
  p2_index = threadIdx.z;
  p2_step = blockDim.z;
#else
  p1_index = 0;
  p2_index = 0;
  p1_step = 1;
  p2_step = 1;
#endif
}

template <>
inline __device__ void getParallel2DIndex<Parallel2Dir::THREAD_YX>(int& p1_index, int& p2_index, int& p1_step,
                                                                   int& p2_step)
{
#ifdef __CUDA_ARCH__
  p1_index = threadIdx.y;
  p1_step = blockDim.y;
  p2_index = threadIdx.x;
  p2_step = blockDim.x;
#else
  p1_index = 0;
  p2_index = 0;
  p1_step = 1;
  p2_step = 1;
#endif
}

template <>
inline __device__ void getParallel2DIndex<Parallel2Dir::THREAD_ZY>(int& p1_index, int& p2_index, int& p1_step,
                                                                   int& p2_step)
{
#ifdef __CUDA_ARCH__
  p1_index = threadIdx.z;
  p1_step = blockDim.z;
  p2_index = threadIdx.y;
  p2_step = blockDim.y;
#else
  p1_index = 0;
  p2_index = 0;
  p1_step = 1;
  p2_step = 1;
#endif
}

template <>
inline __device__ void getParallel2DIndex<Parallel2Dir::THREAD_ZX>(int& p1_index, int& p2_index, int& p1_step,
                                                                   int& p2_step)
{
#ifdef __CUDA_ARCH__
  p1_index = threadIdx.z;
  p1_step = blockDim.z;
  p2_index = threadIdx.x;
  p2_step = blockDim.x;
#else
  p1_index = 0;
  p2_index = 0;
  p1_step = 1;
  p2_step = 1;
#endif
}

template <>
inline __device__ void getParallel2DIndex<Parallel2Dir::NONE>(int& p1_index, int& p2_index, int& p1_step, int& p2_step)
{
  p1_index = 0;
  p1_step = 1;
  p2_index = 0;
  p2_step = 1;
}
}  // namespace p2
}  // namespace mppi













namespace mppi
{
namespace matrix_multiplication
{
/**
 * Utility Functions
 **/
inline __host__ __device__ int2 const unravelColumnMajor(const int index, const int num_rows)
{
  int col = index / num_rows;
  int row = index % num_rows;
  return make_int2(row, col);
}

inline __host__ __device__ int2 const unravelRowMajor(const int index, const int num_cols)
{
  int row = index / num_cols;
  int col = index % num_cols;
  return make_int2(row, col);
}
inline __host__ __device__ constexpr int columnMajorIndex(const int row, const int col, const int num_rows)
{
  return col * num_rows + row;
}

inline __host__ __device__ constexpr int rowMajorIndex(const int row, const int col, const int num_cols)
{
  return row * num_cols + col;
}

/**
 * Utility Classes
 **/
enum class MAT_OP : int
{
  NONE = 0,
  TRANSPOSE
};

template <int M, int N, class T = float>
class devMatrix
{
public:
  T* data = nullptr;
  static constexpr int rows = M;
  static constexpr int cols = N;
  devMatrix(T* n_data)
  {
    data = n_data;
  };

  T operator()(const int i, const int j) const
  {
    return data[columnMajorIndex(i, j, rows)];
  }
};

/**
 * @brief GEneral Matrix Multiplication
 * Conducts the operation
 * C = alpha * op(A) * op(B) + beta * C
 * on matrices of type T
 * TODO: Add transpose options like cuBLAS GEMM
 * Inputs:
 * op(A) - T-type column-major matrix of size M * K, stored in shared/global mem
 * op(B) - T-type column-major matrix of size K * N, stored in shared/global mem
 * alpha - T-type to multiply A * B
 * beta - T-type multipling C
 * A_OP - whether or not you should use A or A transpose
 * B_OP - whether or not you should use B or B transpose
 * Outputs:
 * C - float column-major matrix of size M * N, stored in shared/global mem
 *
 */
template <int M, int K, int N, p1::Parallel1Dir P_DIR = p1::Parallel1Dir::THREAD_Y, class T = float>
inline __device__ __host__ void gemm1(const T* A, const T* B, T* C, const T alpha = 1, const T beta = 0,
                                      const MAT_OP A_OP = MAT_OP::NONE, const MAT_OP B_OP = MAT_OP::NONE)
{
  int parallel_index;
  int parallel_step;
  int p, k;
  p1::getParallel1DIndex<P_DIR>(parallel_index, parallel_step);
  int2 mn;
  const bool all_stride = (A_OP == MAT_OP::NONE) && (B_OP == MAT_OP::TRANSPOSE);
  for (p = parallel_index; p < M * N; p += parallel_step)
  {
    T accumulator = 0;
    mn = unravelColumnMajor(p, M);
    if (K % 4 == 0 && sizeof(type4<T>) <= 16 && !all_stride)
    {  // Fetch 4 B values using single load memory operator of up to 128 bits since B is contiguous wrt k
      __UNROLL(10)
      for (k = 0; k < K; k += 4)
      {
        if (A_OP == MAT_OP::NONE && B_OP == MAT_OP::NONE)
        {
          const type4<T> b_tmp = reinterpret_cast<const type4<T>*>(&B[columnMajorIndex(k, mn.y, K)])[0];
          accumulator += A[columnMajorIndex(mn.x, k + 0, M)] * b_tmp[0];
          accumulator += A[columnMajorIndex(mn.x, k + 1, M)] * b_tmp[1];
          accumulator += A[columnMajorIndex(mn.x, k + 2, M)] * b_tmp[2];
          accumulator += A[columnMajorIndex(mn.x, k + 3, M)] * b_tmp[3];
        }
        else if (A_OP == MAT_OP::TRANSPOSE && B_OP == MAT_OP::NONE)
        {
          const type4<T> b_tmp = reinterpret_cast<const type4<T>*>(&B[columnMajorIndex(k, mn.y, K)])[0];
          const type4<T> a_tmp = reinterpret_cast<const type4<T>*>(&A[rowMajorIndex(mn.x, k, K)])[0];
          accumulator += a_tmp[0] * b_tmp[0];
          accumulator += a_tmp[1] * b_tmp[1];
          accumulator += a_tmp[2] * b_tmp[2];
          accumulator += a_tmp[3] * b_tmp[3];
        }
        else if (A_OP == MAT_OP::TRANSPOSE && B_OP == MAT_OP::TRANSPOSE)
        {
          // const type4<T> b_tmp = reinterpret_cast<const type4<T>*>(&B[columnMajorIndex(k, mn.y, K)])[0];
          const type4<T> a_tmp = reinterpret_cast<const type4<T>*>(&A[rowMajorIndex(mn.x, k, K)])[0];
          accumulator += a_tmp[0] * B[rowMajorIndex(k + 0, mn.y, N)];
          accumulator += a_tmp[1] * B[rowMajorIndex(k + 1, mn.y, N)];
          accumulator += a_tmp[2] * B[rowMajorIndex(k + 2, mn.y, N)];
          accumulator += a_tmp[3] * B[rowMajorIndex(k + 3, mn.y, N)];
        }
      }
    }
    else if (K % 2 == 0 && sizeof(type2<T>) <= 16 && !all_stride)
    {  // Fetch 2 B values using single load memory operator of up to 128 bits since B is contiguous wrt k
      __UNROLL(10)
      for (k = 0; k < K; k += 2)
      {
        if (A_OP == MAT_OP::NONE && B_OP == MAT_OP::NONE)
        {
          const type2<T> b_tmp = reinterpret_cast<const type2<T>*>(&B[columnMajorIndex(k, mn.y, K)])[0];
          accumulator += A[columnMajorIndex(mn.x, k + 0, M)] * b_tmp[0];
          accumulator += A[columnMajorIndex(mn.x, k + 1, M)] * b_tmp[1];
        }
        else if (A_OP == MAT_OP::TRANSPOSE && B_OP == MAT_OP::NONE)
        {
          const type2<T> b_tmp = reinterpret_cast<const type2<T>*>(&B[columnMajorIndex(k, mn.y, K)])[0];
          const type2<T> a_tmp = reinterpret_cast<const type2<T>*>(&A[rowMajorIndex(mn.x, k, K)])[0];
          accumulator += a_tmp[0] * b_tmp[0];
          accumulator += a_tmp[1] * b_tmp[1];
        }
        else if (A_OP == MAT_OP::TRANSPOSE && B_OP == MAT_OP::TRANSPOSE)
        {
          const type2<T> a_tmp = reinterpret_cast<const type2<T>*>(&A[rowMajorIndex(mn.x, k, K)])[0];
          accumulator += a_tmp[0] * B[rowMajorIndex(k + 0, mn.y, N)];
          accumulator += a_tmp[1] * B[rowMajorIndex(k + 1, mn.y, N)];
        }
      }
    }
    else
    {  // Either K is odd or sizeof(T) is large enough that
      T a;
      T b;
      __UNROLL(10)
      for (k = 0; k < K; k++)
      {
        if (A_OP == MAT_OP::NONE && B_OP == MAT_OP::NONE)
        {
          a = A[columnMajorIndex(mn.x, k, M)];
          b = B[columnMajorIndex(k, mn.y, K)];
        }
        else if (A_OP == MAT_OP::TRANSPOSE && B_OP == MAT_OP::NONE)
        {
          a = A[rowMajorIndex(mn.x, k, K)];
          b = B[columnMajorIndex(k, mn.y, K)];
        }
        else if (A_OP == MAT_OP::TRANSPOSE && B_OP == MAT_OP::TRANSPOSE)
        {
          a = A[rowMajorIndex(mn.x, k, K)];
          b = B[rowMajorIndex(k, mn.y, N)];
        }
        else
        {
          a = A[columnMajorIndex(mn.x, k, M)];
          b = B[rowMajorIndex(k, mn.y, N)];
        }

        accumulator += a * b;
      }
    }
    if (beta == 0)
    {  // Special case to remove extraneous memory accesses
      C[p] = alpha * accumulator;
    }
    else
    {
      C[p] = alpha * accumulator + beta * C[p];
    }
  }
}

/**
 * @brief GEneral Matrix Multiplication
 * Conducts the operation
 * C = alpha * A * B + beta * C
 * using two parallelization directions
 * TODO: Add transpose options like cuBLAS GEMM
 * Inputs:
 * A - float column-major matrix of size M * K, stored in shared/global mem
 * B - float column-major matrix of size K * N, stored in shared/global mem
 * alpha - float to multiply A * B
 * beta - float multipling C
 * Outputs:
 * C - float column-major matrix of size M * N, stored in shared/global mem
 *
 */
template <int M, int K, int N, p2::Parallel2Dir P_DIR = p2::Parallel2Dir::THREAD_XY>
inline __device__ void gemm2(const float* A, const float* B, float* C, const float alpha = 1.0f,
                             const float beta = 0.0f)
{
  int m_ind_start;
  int m_ind_size;
  int n_ind_start;
  int n_ind_size;
  p2::getParallel2DIndex<P_DIR>(m_ind_start, n_ind_start, m_ind_size, n_ind_size);
  for (int m = m_ind_start; m < M; m += m_ind_size)
  {
    for (int n = n_ind_start; n < N; n += n_ind_size)
    {
      float accumulator = 0;
      __UNROLL(10)
      for (int k = 0; k < K; k++)
      {
        accumulator += A[columnMajorIndex(m, k, M)] * B[columnMajorIndex(k, n, K)];
      }
      C[columnMajorIndex(m, n, M)] = alpha * accumulator + beta * C[columnMajorIndex(m, n, M)];
    }
  }
}

/**
 * @brief Perform Guass Jordan Elimination in place on columan-major MxN matrix A.
 * Useful for solving Cx = b where A = [C | b] as well as inverting matrices.
 *
 * @tparam M - number of rows
 * @tparam N - number of cols
 * @tparam P_DIR - Parallelization axes for on the GPU
 * @tparam T - type of data in A
 * @param A - column-major matrix of type T with M rows and N cols
 *
 * @return reduced row echelon form of A is returned in A.
 */
template <int M, int N, p1::Parallel1Dir P_DIR = p1::Parallel1Dir::THREAD_Y, class T = float>
inline __host__ __device__ void GaussJordanElimination(T* A)
{
  int p_index, step;
  p1::getParallel1DIndex<P_DIR>(p_index, step);
  int row, col, offset = 0;
  T accumulator;
  for (int i = 0; i < M; i++)
  {
    // Check if row-swap is needed
    row = i;
    while (A[columnMajorIndex(row, i + offset, M)] == 0)
    {
      row++;
      if (row == M)
      {  // column is all zeros, need to move on to the next column
        offset++;
        if (i + offset >= N)
        {  // Ran out of columns to check.
          return;
        }
        row = i;
      }
    }
    // swap rows if needed (row != i)
    for (col = i + p_index; row != i && col < N; col += step)
    {
      accumulator = A[columnMajorIndex(row, col, M)];
      A[columnMajorIndex(row, col, M)] = A[columnMajorIndex(i, col, M)];
      A[columnMajorIndex(i, col, M)] = accumulator;
    }
    // normalize the current row
    accumulator = 1.0f / A[columnMajorIndex(i, i + offset, M)];
    for (col = i + offset + p_index; col < N; col += step)
    {
      A[columnMajorIndex(i, col, M)] *= accumulator;
    }
#ifdef __CUDA_ARCH__
    __syncthreads();
#endif
    // Now eliminate pivot from both rows above and below
    for (row = p_index; row < M; row += step)
    {
      if (row == i)
      {
        continue;
      }
      accumulator = -A[columnMajorIndex(row, i + offset, M)];
      for (col = i + offset; col < N; col++)
      {
        A[columnMajorIndex(row, col, M)] += accumulator * A[columnMajorIndex(i, col, M)];
      }
    }
#ifdef __CUDA_ARCH__
    __syncthreads();
#endif
  }
}

template <p1::Parallel1Dir P_DIR = p1::Parallel1Dir::NONE, int M = 1, int K = 1, int N = 1, class T = float>
void matMult1(const devMatrix<M, K, T>& A, const devMatrix<K, N, T>& B, devMatrix<M, N, T>& C)
{
  gemm1<M, K, N, P_DIR, T>(A.data, B.data, C.data);
}

}  // namespace matrix_multiplication
}  // namespace mppi












namespace mppi
{
namespace util
{
enum class LOG_LEVEL : int
{
  DEBUG = 0,
  INFO,
  WARNING,
  ERROR,
  NONE
};

static LOG_LEVEL GLOBAL_LOG_LEVEL = LOG_LEVEL::WARNING;

const char BLACK[] = "\033[0;30m";
const char RED[] = "\033[0;31m";
const char GREEN[] = "\033[0;32m";
const char YELLOW[] = "\033[0;33m";
const char BLUE[] = "\033[0;34m";
const char MAGENTA[] = "\033[0;35m";
const char CYAN[] = "\033[0;36m";
const char WHITE[] = "\033[0;37m";
const char RESET[] = "\033[0m";

class MPPILogger
{
public:
  MPPILogger() = default;
  MPPILogger(const MPPILogger& other) = default;
  MPPILogger(MPPILogger&& other) = default;
  virtual ~MPPILogger() = default;
  MPPILogger& operator=(const MPPILogger& other) = default;
  MPPILogger& operator=(MPPILogger&& other) = default;

  explicit MPPILogger(LOG_LEVEL level)
  {
    setLogLevel(level);
  }

  /**
   * @brief Set the Log Level
   *
   * @param level
   */
  void setLogLevel(const LOG_LEVEL& level)
  {
    log_level_ = level;
  }

  /**
   * @brief Set the Output Stream
   *
   * @param output file stream to write (stdout, stderr, nullptr, etc.)
   */
  void setOutputStream(std::FILE* const output)
  {
    output_stream_ = output;
  }

  /**
   * @brief Get the Log Level object
   *
   * @return LOG_LEVEL
   */
  LOG_LEVEL getLogLevel() const
  {
    return log_level_;
  }

  /**
   * @brief Get the Output Stream object
   *
   * @return std::FILE*
   */
  std::FILE* getOutputStream() const
  {
    return output_stream_;
  }

  /**
   * @brief       Log debug messages to the output stream in green if the log level is set for DEBUG
   * @param fmt   Format string (if additional arguments are passed) or message to display
   */
  virtual void debug(const char* fmt, ...)
  {
    if (log_level_ <= LOG_LEVEL::DEBUG)
    {
      std::va_list argptr;
      va_start(argptr, fmt);
      surround_fprintf(output_stream_, GREEN, RESET, fmt, argptr);
      va_end(argptr);
    }
  }

  /**
   * @brief       Log info messages to the output stream in cyan if the log level is set for INFO
   * @param fmt   Format string (if additional arguments are passed) or message to display
   */
  virtual void info(const char* fmt, ...)
  {
    if (log_level_ <= LOG_LEVEL::INFO)
    {
      std::va_list argptr;
      va_start(argptr, fmt);
      surround_fprintf(output_stream_, CYAN, RESET, fmt, argptr);
      va_end(argptr);
    }
  }

  /**
   * @brief       Log debug messages to the output stream in yellow if the log level is set for WARNING
   * @param fmt   Format string (if additional arguments are passed) or message to display
   */
  virtual void warning(const char* fmt, ...)
  {
    if (log_level_ <= LOG_LEVEL::WARNING)
    {
      std::va_list argptr;
      va_start(argptr, fmt);
      surround_fprintf(output_stream_, YELLOW, RESET, fmt, argptr);
      va_end(argptr);
    }
  }

  /**
   * @brief       Log debug messages to the output stream in red if the log level is set for ERROR
   * @param fmt   Format string (if additional arguments are passed) or message to display
   */
  virtual void error(const char* fmt, ...)
  {
    if (log_level_ <= LOG_LEVEL::ERROR)
    {
      std::va_list argptr;
      va_start(argptr, fmt);
      surround_fprintf(output_stream_, RED, RESET, fmt, argptr);
      va_end(argptr);
    }
  }

protected:
  LOG_LEVEL log_level_ = GLOBAL_LOG_LEVEL;
  std::FILE* output_stream_ = stdout;

  /**
   * @brief Prints a colored output to a provided fstream. It does this by first creating the formatted string
   * as a std::vector<char> so that it can be used as an input to fprintf with a different format string
   *
   * @param fstream   file stream to write output to
   * @param color     color code to use on provided string
   * @param fmt       format string
   * @param ...       extra variables for format string
   */
  virtual void surround_fprintf(std::FILE* fstream, const char* prefix, const char* suffix, const char* fmt,
                                std::va_list args)
  {
    // introducing a second copy of the args as calling vsnprintf leaves args in an indeterminate state
    std::va_list args_cpy;
    va_copy(args_cpy, args);
    // figure out size of formatted string, also uses up args
    std::vector<char> buf(1 + std::vsnprintf(nullptr, 0, fmt, args));
    // Fill buffer with formatted string using copy of the args
    std::vsnprintf(buf.data(), buf.size(), fmt, args_cpy);
    va_end(args_cpy);
    // print formatted string but colored
    std::fprintf(fstream, "%s%s%s", prefix, buf.data(), suffix);
  }
};

using MPPILoggerPtr = std::shared_ptr<MPPILogger>;
}  // namespace util
}  // namespace mppi



































#ifndef DEPRECATED
#if __cplusplus >= 201402L
#define DEPRECATED [[deprecated]]
#elif defined(__GNUC__) || defined(__clang__) || defined(__CUDACC__)
#define DEPRECATED __attribute__((deprecated))
#elif defined(_MSC_VER)
#define DEPRECATED __declspec(deprecated)
#else
#pragma message("WARNING: You need to implement DEPRECATED for this compiler")
#define DEPRECATED
#endif
#endif
// #ifndef __DEPRECATED__
// #if defined(_WIN32)
// # define __DEPRECATED__(msg) __declspec(deprecated(msg))
// #elif (defined(__GNUC__) && (__GNUC__ < 4 || (__GNUC__ == 4 && __GNUC_MINOR__ < 5 && !defined(__clang__))))
// # define __DEPRECATED__(msg) __attribute__((deprecated))
// #else
// # define __DEPRECATED__(msg) __attribute__((deprecated(msg)))
// #endif
// #endif

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
  if (code != cudaSuccess)
  {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort)
      exit(code);
  }
}

inline void __cudaCheckError(const char* file, const int line)
{
  cudaError err = cudaGetLastError();
  if (cudaSuccess != err)
  {
    fprintf(stderr, "cudaCheckError() failed at %s:%i : %s\n", file, line, cudaGetErrorString(err));
    exit(-1);
  }

  // More careful checking. However, this will affect performance.
  // Comment away if needed.
  err = cudaDeviceSynchronize();
  if (cudaSuccess != err)
  {
    fprintf(stderr, "cudaCheckError() with sync failed at %s:%i : %s\n", file, line, cudaGetErrorString(err));
    exit(-1);
  }
}

inline const char* cufftGetErrorString(cufftResult& code)
{
  // Codes from https://docs.nvidia.com/cuda/cufft/index.html#cufftresult
  switch (code)
  {
    case CUFFT_SUCCESS:
      return "Success";
    case CUFFT_INVALID_PLAN:
      return "cuFFT was passed an invalid plan handle";
    case CUFFT_ALLOC_FAILED:
      return "cuFFT failed to allocate GPU or CPU memory";
    case CUFFT_INVALID_VALUE:
      return "User specified an invalid pointer or parameter";
    case CUFFT_INTERNAL_ERROR:
      return "Driver or internal cuFFT library error";
    case CUFFT_EXEC_FAILED:
      return "Failed to execute an FFT on the GPU";
    case CUFFT_SETUP_FAILED:
      return "The cuFFT library failed to initialize";
    case CUFFT_INVALID_SIZE:
      return "User specified an invalid transform size";
    case CUFFT_UNALIGNED_DATA:
      return "No longer used but unaligned data";
    case CUFFT_INVALID_DEVICE:
      return "Execution of a plan was on different GPU than plan creation";
    case CUFFT_NO_WORKSPACE:
      return "No workspace has been provided prior to plan execution";
    case CUFFT_NOT_IMPLEMENTED:
      return "Function does not implement functionality for parameters given.";
#if CUDART_VERSION < 13000
    case CUFFT_INCOMPLETE_PARAMETER_LIST:
      return "Missing parameters in call";
    case CUFFT_PARSE_ERROR:
      return "Internal plan database error";
    case CUFFT_LICENSE_ERROR:
      return "License Error. Used in previous versions";
#else
    case CUFFT_MISSING_DEPENDENCY:
      return "cuFFT is unable to find a dependency";
    case CUFFT_NVRTC_FAILURE:
      return "An NVRTC failure was encountered during a cuFFT operation";
    case CUFFT_NVJITLINK_FAILURE:
      return "An nvJitLink failure was encountered during a cuFFT operation";
    case CUFFT_NVSHMEM_FAILURE:
      return "An NVSHMEM failure was encountered during a cuFFT operation";
#endif
    case CUFFT_NOT_SUPPORTED:
      return "Operation is not supported for parameters given.";
    default:
      return "cuFFT ERROR";
  }
}

inline const char* curandGetErrorString(curandStatus_t code)
{
  // Codes from https://docs.nvidia.com/cuda/curand/group__HOST.html#group__HOST_1gb94a31d5c165858c96b6c18b70644437
  switch (code)
  {
    case CURAND_STATUS_SUCCESS:
      return "No errors.";
    case CURAND_STATUS_VERSION_MISMATCH:
      return "Header file and linked library version do not match.";
    case CURAND_STATUS_NOT_INITIALIZED:
      return "Generator not initialized.";
    case CURAND_STATUS_ALLOCATION_FAILED:
      return "Memory allocation failed.";
    case CURAND_STATUS_TYPE_ERROR:
      return "Generator is wrong type.";
    case CURAND_STATUS_OUT_OF_RANGE:
      return "Argument out of range.";
    case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
      return "Length requested is not a multple of dimension.";
    case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
      return "GPU does not have double precision required by MRG32k3a.";
    case CURAND_STATUS_LAUNCH_FAILURE:
      return "Kernel launch failure.";
    case CURAND_STATUS_PREEXISTING_FAILURE:
      return "Preexisting failure on library entry.";
    case CURAND_STATUS_INITIALIZATION_FAILED:
      return "Initialization of CUDA failed.";
    case CURAND_STATUS_ARCH_MISMATCH:
      return "Architecture mismatch, GPU does not support requested feature.";
    case CURAND_STATUS_INTERNAL_ERROR:
      return "Internal library error.";
    default:
      return "Cureand Error";
  }
}

inline void cufftAssert(cufftResult code, const char* file, int line, bool abort = true)
{
  if (code != CUFFT_SUCCESS)
  {
    fprintf(stderr, "CUFFTassert: %s %s %d\n", cufftGetErrorString(code), file, line);
    if (abort)
    {
      exit(code);
    }
  }
}

inline void curandAssert(curandStatus_t code, const char* file, int line, bool abort = true)
{
  if (code != CURAND_STATUS_SUCCESS)
  {
    fprintf(stderr, "Curandassert: %s %s %d\n", curandGetErrorString(code), file, line);
    if (abort)
    {
      exit(code);
    }
  }
}

#define CudaCheckError() __cudaCheckError(__FILE__, __LINE__)
#define HANDLE_ERROR(ans)                                                                                              \
  {                                                                                                                    \
    gpuAssert((ans), __FILE__, __LINE__);                                                                              \
  }

#define HANDLE_CUFFT_ERROR(ans)                                                                                        \
  {                                                                                                                    \
    cufftAssert((ans), __FILE__, __LINE__);                                                                            \
  }

#define HANDLE_CURAND_ERROR(ans)                                                                                       \
  {                                                                                                                    \
    curandAssert((ans), __FILE__, __LINE__);                                                                           \
  }












#ifndef SQ
#define SQ(a) ((a) * (a))
#endif  // SQ

// For aligning parameters within structs such as a float array to 16 bytes
// Ex: float name[size] MPPI_ALIGN(16) = {0.0f};
#if defined(__CUDACC__)  // NVCC
#define MPPI_ALIGN(n) __align__(n)
#elif defined(__GNUC__)  // GCC
#define MPPI_ALIGN(n) __attribute__((aligned(n)))
#elif defined(_MSC_VER)  // MSVC
#define MPPI_ALIGN(n) __declspec(align(n))
#else
#error "Please provide a definition for MPPI_ALIGN macro for your host compiler!"
#endif

namespace mppi
{
namespace math
{
namespace matrix = ::mppi::matrix_multiplication;

const float GRAVITY = 9.81f;
// Based off of https://gormanalysis.com/blog/random-numbers-in-cpp
inline std::vector<int> sample_without_replacement(const int k, const int N,
                                                   std::default_random_engine g = std::default_random_engine())
{
  if (k > N)
  {
    throw std::logic_error("Can't sample more than n times without replacement");
  }
  // Create an unordered set to store the samples
  std::unordered_set<int> samples;

  // For loop runs k times
  for (int r = N - k; r < N; r++)
  {
    if (r == 0)
    {
      samples.insert(N - 1);
      continue;
    }
    int v = std::uniform_int_distribution<>(1, r)(g);  // sample between 1 and r
    if (!samples.insert(v - 1).second)
    {  // if v exists in the set
      samples.insert(r - 1);
    }
  }
  // Copy set into a vector
  std::vector<int> final_sequence(samples.begin(), samples.end());
  // Shuffle the vector to get the final sequence of sampling
  std::shuffle(final_sequence.begin(), final_sequence.end(), g);
  return final_sequence;
}

inline __host__ __device__ float expr(const float r, const float x)
{
  float mid_term = 1.0f + (r - 1.0f) * x;
  return (mid_term > 0) * powf(mid_term, 1.0f / (r - 1.0f));
}

/**
 * Linear interpolation
 * Given two coordinates (x_min, y_min) and (x_max, y_max)
 * And the x location of a third (x), return the y location
 * along the line between the two points
 */
inline __host__ __device__ float linInterp(const float x, const float x_min, const float x_max, const float y_min,
                                           const float y_max)
{
  return (x - x_min) / (x_max - x_min) * (y_max - y_min) + y_min;
}

/**
 * Return the sign of a variable (1 for positive, -1 for negative, 0 for 0)
 **/
template <class T = float>
inline __host__ __device__ int sign(const T& a)
{
  return (a > 0) - (a < 0);
}

inline __host__ __device__ int int_ceil(const int& a, const int& b)
{
  return a == 0 ? a : (a - 1) / b + 1;
}

inline constexpr __host__ __device__ int int_ceil_const(const int& a, const int& b)
{
  return a == 0 ? a : (a - 1) / b + 1;
}

/**
 * @brief gives back the next multiple of b larger than or equal to a.
 * For example, if a = 3, b = 2, this method returns 4
 *
 * @param a - int to be larger than
 * @param b - int to be multiple of
 * @return int - the next multiple of b larger than a
 */
inline constexpr __host__ __device__ int int_multiple_const(const int& a, const int& b)
{
  return a == 0 ? a : ((a - 1) / b + 1) * b;
}

// Returns the int version of ceil(a/4)
inline __host__ __device__ int nearest_quotient_4(const int& a)
{
  return int_ceil(a, 4);
}

// Returns the next multiple of 4 larger than or equal to a, Useful for calculating aligned memory sizes
inline __host__ __device__ int nearest_multiple_4(const int& a)
{
  return int_ceil(a, 4) * 4;
}

/**
 * Calculates the normalized distance from the centerline
 * @param r - current radius
 * @param r_in - the inside radius of a track
 * @param r_out - the outside radius of a track
 * @return norm_dist - a normalized distance away from the centerline
 * norm_dist = 0 -> on the centerline
 * norm_dist = 1 -> on one of the track boundaries inner, or outer
 */
inline __host__ __device__ float normDistFromCenter(const float r, const float r_in, const float r_out)
{
  float r_center = (r_in + r_out) / 2.0f;
  float r_width = (r_out - r_in);
  float dist_from_center = fabsf(r - r_center);
  float norm_dist = dist_from_center / (r_width * 0.5f);
  return norm_dist;
}

/**
 * Multiply two quaternions together which gives you their rotations added together
 * q_3 = q_1 x q_2
 * Inputs:
 *  q_1 - first quaternion
 *  q_2 - second quaternion
 *  q_3 - output quaternion
 */
inline __host__ __device__ void QuatMultiply(const float q_1[4], const float q_2[4], float q_3[4],
                                             bool normalize = true)
{
  q_3[0] = q_1[0] * q_2[0] - q_1[1] * q_2[1] - q_1[2] * q_2[2] - q_1[3] * q_2[3];
  q_3[1] = q_1[1] * q_2[0] + q_1[0] * q_2[1] - q_1[3] * q_2[2] + q_1[2] * q_2[3];
  q_3[2] = q_1[2] * q_2[0] + q_1[3] * q_2[1] + q_1[0] * q_2[2] - q_1[1] * q_2[3];
  q_3[3] = q_1[3] * q_2[0] - q_1[2] * q_2[1] + q_1[1] * q_2[2] + q_1[0] * q_2[3];
  if (normalize)
  {
#ifdef __CUDA_ARCH__
    float inv_norm = rsqrtf(SQ(q_3[0]) + SQ(q_3[1]) + SQ(q_3[2]) + SQ(q_3[3]));
#else
    float inv_norm = 1.0f / sqrtf(SQ(q_3[0]) + SQ(q_3[1]) + SQ(q_3[2]) + SQ(q_3[3]));
#endif
    __UNROLL(4)
    for (int i = 0; i < 4; i++)
    {
      q_3[i] *= inv_norm;
    }
  }
}

inline __host__ __device__ void QuatInv(const float q[4], float q_inv[4])
{
#ifdef __CUDA_ARCH__
  float inv_norm = rsqrtf(SQ(q[0]) + SQ(q[1]) + SQ(q[2]) + SQ(q[3]));
#else
  float inv_norm = 1.0f / sqrtf(SQ(q[0]) + SQ(q[1]) + SQ(q[2]) + SQ(q[3]));
#endif
  q_inv[0] = q[0] * inv_norm;
  q_inv[1] = -q[1] * inv_norm;
  q_inv[2] = -q[2] * inv_norm;
  q_inv[3] = -q[3] * inv_norm;
}

/**
 * Calculate the rotation required to get from q_1 to q_2
 * In Euler angles, this would be direct subraction but
 * in quaternions, it doesn't quite work that way
 */
inline __device__ void QuatSubtract(float q_1[4], float q_2[4], float q_3[4])
{
  float q_1_inv[4];
  QuatInv(q_1, q_1_inv);
  QuatMultiply(q_2, q_1_inv, q_3);
}

/*
 * The Euler rotation sequence is 3-2-1 (roll, pitch, yaw) from Body to World
 */
inline __device__ void Euler2QuatNWU(const double& r, const double& p, const double& y, double q[4])
{
  double phi_2 = r / 2.0;
  double theta_2 = p / 2.0;
  double psi_2 = y / 2.0;
  double cos_phi_2 = cos(phi_2);
  double sin_phi_2 = sin(phi_2);
  double cos_theta_2 = cos(theta_2);
  double sin_theta_2 = sin(theta_2);
  double cos_psi_2 = cos(psi_2);
  double sin_psi_2 = sin(psi_2);

  q[0] = cos_phi_2 * cos_theta_2 * cos_psi_2 + sin_phi_2 * sin_theta_2 * sin_psi_2;
  q[1] = -cos_phi_2 * sin_theta_2 * sin_psi_2 + cos_theta_2 * cos_psi_2 * sin_phi_2;
  q[2] = cos_phi_2 * cos_psi_2 * sin_theta_2 + sin_phi_2 * cos_theta_2 * sin_psi_2;
  q[3] = cos_phi_2 * cos_theta_2 * sin_psi_2 - sin_phi_2 * cos_psi_2 * sin_theta_2;
}

/*
 * The Euler rotation sequence is 3-2-1 (roll, pitch, yaw) from Body to World
 */
inline __host__ __device__ void Euler2QuatNWU(const float& r, const float& p, const float& y, float q[4])
{
  float sin_phi_2, cos_phi_2, sin_theta_2, cos_theta_2, sin_psi_2, cos_psi_2;
#ifdef __CUDA_ARCH__
  float r_norm = angle_utils::normalizeAngle(r * 0.5f);
  float p_norm = angle_utils::normalizeAngle(p * 0.5f);
  float y_norm = angle_utils::normalizeAngle(y * 0.5f);
  __sincosf(r_norm, &sin_phi_2, &cos_phi_2);
  __sincosf(p_norm, &sin_theta_2, &cos_theta_2);
  __sincosf(y_norm, &sin_psi_2, &cos_psi_2);
#else
  sincosf(r * 0.5f, &sin_phi_2, &cos_phi_2);
  sincosf(p * 0.5f, &sin_theta_2, &cos_theta_2);
  sincosf(y * 0.5f, &sin_psi_2, &cos_psi_2);
#endif

  q[0] = cos_phi_2 * cos_theta_2 * cos_psi_2 + sin_phi_2 * sin_theta_2 * sin_psi_2;
  q[1] = -cos_phi_2 * sin_theta_2 * sin_psi_2 + cos_theta_2 * cos_psi_2 * sin_phi_2;
  q[2] = cos_phi_2 * cos_psi_2 * sin_theta_2 + sin_phi_2 * cos_theta_2 * sin_psi_2;
  q[3] = cos_phi_2 * cos_theta_2 * sin_psi_2 - sin_phi_2 * cos_psi_2 * sin_theta_2;
}

// (RPY rotation sequence)
/*
 * Returns an euler sequence 3-2-1 (roll pitch yaw) that when applied takes you from body to world
 */
inline __host__ __device__ void Quat2EulerNWU(const float q[4], float& r, float& p, float& y)
{
  r = atan2f(2.0f * q[3] * q[2] + 2.0f * q[0] * q[1], q[0] * q[0] + q[3] * q[3] - q[2] * q[2] - q[1] * q[1]);
  float temp = -2.0f * q[0] * q[2] + 2.0f * q[1] * q[3];
  // Clamp value between -1 and 1 to prevent NaNs
  p = -asinf(fmaxf(fminf(1.0f, temp), -1.0f));
  y = atan2f(2.0f * q[2] * q[1] + 2.0f * q[3] * q[0], q[0] * q[0] + q[1] * q[1] - q[2] * q[2] - q[3] * q[3]);
}

inline __host__ __device__ void Quat2DCM(const float q[4], float M[3][3])
{
  M[0][0] = SQ(q[0]) + SQ(q[1]) - SQ(q[2]) - SQ(q[3]);
  M[0][1] = 2 * (q[1] * q[2] - q[0] * q[3]);
  M[0][2] = 2 * (q[1] * q[3] + q[0] * q[2]);
  M[1][0] = 2 * (q[1] * q[2] + q[0] * q[3]);
  M[1][1] = SQ(q[0]) - SQ(q[1]) + SQ(q[2]) - SQ(q[3]);
  M[1][2] = 2 * (q[2] * q[3] - q[0] * q[1]);
  M[2][0] = 2 * (q[1] * q[3] - q[0] * q[2]);
  M[2][1] = 2 * (q[2] * q[3] + q[0] * q[1]);
  M[2][2] = SQ(q[0]) - SQ(q[1]) - SQ(q[2]) + SQ(q[3]);
}

inline __host__ __device__ void QuatSubtract(const Eigen::Quaternionf& q_1, const Eigen::Quaternionf& q_2,
                                             Eigen::Quaternionf& q_3)
{
  q_3 = q_2 * q_1.inverse();
}

inline __host__ __device__ void QuatMultiply(const Eigen::Quaternionf& q_1, const Eigen::Quaternionf& q_2,
                                             Eigen::Quaternionf& q_3, bool normalize = true)
{
  q_3 = q_1 * q_2;
  if (normalize)
  {
    q_3.normalize();
  }
}

inline __host__ __device__ void QuatInv(const Eigen::Quaternionf& q, Eigen::Quaternionf& q_f)
{
  q_f = q.inverse();
}

/*
 * rotates a point by the given quaternion
 */
inline __host__ __device__ void RotatePointByQuat(const float q[4], const float3& point, float3& output)
{
  // converts the point into a quaternion format
  float pq[4] = { 0.0f, point.x, point.y, point.z };
  float q_inv[4];
  float temp[4];
  QuatInv(q, q_inv);
  QuatMultiply(q, pq, temp, false);
  QuatMultiply(temp, q_inv, pq, false);
  // converts the quaternion back into a point
  output = make_float3(pq[1], pq[2], pq[3]);
}

/*
 * rotates a point by the given quaternion
 */
inline __host__ __device__ void RotatePointByQuat(const Eigen::Quaternionf& q, const float3& point, float3& output)
{
  // converts the point into a quaternion format
  Eigen::Quaternionf q_inv, temp, pq;
  pq.w() = 0.0f;
  pq.x() = point.x;
  pq.y() = point.y;
  pq.z() = point.z;
  QuatInv(q, q_inv);
  QuatMultiply(q, pq, temp, false);
  QuatMultiply(temp, q_inv, pq, false);
  // converts the quaternion back into a point
  output = make_float3(pq.x(), pq.y(), pq.z());
}
/*
 * rotates a point by the given quaternion
 */
inline __host__ __device__ void RotatePointByQuat(const Eigen::Quaternionf& q, const Eigen::Ref<Eigen::Vector3f>& point,
                                                  Eigen::Ref<Eigen::Vector3f> output)
{
  output = q * point;
}

/*
 * rotates a point by the given quaternion
 */
inline __host__ __device__ void RotatePointByQuat(const float q[4], float3& point)
{
  RotatePointByQuat(q, point, point);
}

/*
 * rotates a point by the given quaternion
 */
inline __host__ __device__ void RotatePointByQuat(const Eigen::Quaternionf& q, float3& point)
{
  RotatePointByQuat(q, point, point);
}

/*
 * rotates a point by the given quaternion
 */
inline __host__ __device__ void RotatePointByQuat(const Eigen::Quaternionf& q, Eigen::Ref<Eigen::Vector3f> point)
{
  RotatePointByQuat(q, point, point);
}

/*
 * rotates a point by the given DCM Matrix. Transpose is used as M stored in row-major
 */
inline __host__ __device__ void RotatePointByDCM(const float M[3][3], const float3& point, float3& output,
                                                 matrix::MAT_OP operation = matrix::MAT_OP::NONE)
{
  if (operation == matrix::MAT_OP::NONE)
  {
    matrix::gemm1<3, 3, 1, p1::Parallel1Dir::NONE>((float*)M, (const float*)&point, (float*)&output, 1.0f, 0.0f,
                                                   matrix::MAT_OP::TRANSPOSE);
  }
  else if (operation == matrix::MAT_OP::TRANSPOSE)
  {
    matrix::gemm1<3, 3, 1, p1::Parallel1Dir::NONE>((float*)M, (const float*)&point, (float*)&output, 1.0f, 0.0f,
                                                   matrix::MAT_OP::NONE);
  }
}

/*
 * rotates a point by the given DCM Matrix
 */
inline __host__ __device__ void RotatePointByDCM(const Eigen::Ref<Eigen::Matrix3f>& M, const float3& point,
                                                 float3& output, matrix::MAT_OP operation = matrix::MAT_OP::NONE)
{
  Eigen::Map<Eigen::Vector3f> point_eigen((float*)&point);
  Eigen::Map<Eigen::Vector3f> output_eigen((float*)&output);
  if (operation == matrix::MAT_OP::NONE)
  {
    output_eigen = M * point_eigen;
  }
  else if (operation == matrix::MAT_OP::TRANSPOSE)
  {
    output_eigen = M.transpose() * point_eigen;
  }
}

/*
 * rotates a point by the given DCM Matrix
 */
inline __host__ __device__ void RotatePointByDCM(const Eigen::Ref<Eigen::Matrix3f>& M,
                                                 const Eigen::Ref<Eigen::Vector3f>& point,
                                                 Eigen::Ref<Eigen::Vector3f> output,
                                                 matrix::MAT_OP operation = matrix::MAT_OP::NONE)
{
  if (operation == matrix::MAT_OP::NONE)
  {
    output = M * point;
  }
  else if (operation == matrix::MAT_OP::TRANSPOSE)
  {
    output = M.transpose() * point;
  }
}

/*
 * The Euler rotation sequence is 3-2-1 (roll, pitch, yaw) from Body to World
 */
inline __host__ __device__ void Euler2QuatNWU(const float& r, const float& p, const float& y, Eigen::Quaternionf& q)
{
  // double psi = clamp_radians(euler.roll);
  // double theta = clamp_radians(euler.pitch);
  // double phi = clamp_radians(euler.yaw);
  float sin_phi_2, cos_phi_2, sin_theta_2, cos_theta_2, sin_psi_2, cos_psi_2;
#ifdef __CUDA_ARCH__
  float r_norm = angle_utils::normalizeAngle(r * 0.5f);
  float p_norm = angle_utils::normalizeAngle(p * 0.5f);
  float y_norm = angle_utils::normalizeAngle(y * 0.5f);
  __sincosf(r_norm, &sin_phi_2, &cos_phi_2);
  __sincosf(p_norm, &sin_theta_2, &cos_theta_2);
  __sincosf(y_norm, &sin_psi_2, &cos_psi_2);
#else
  sincosf(r * 0.5f, &sin_phi_2, &cos_phi_2);
  sincosf(p * 0.5f, &sin_theta_2, &cos_theta_2);
  sincosf(y * 0.5f, &sin_psi_2, &cos_psi_2);
#endif

  q.w() = cos_phi_2 * cos_theta_2 * cos_psi_2 + sin_phi_2 * sin_theta_2 * sin_psi_2;
  q.x() = -cos_phi_2 * sin_theta_2 * sin_psi_2 + cos_theta_2 * cos_psi_2 * sin_phi_2;
  q.y() = cos_phi_2 * cos_psi_2 * sin_theta_2 + sin_phi_2 * cos_theta_2 * sin_psi_2;
  q.z() = cos_phi_2 * cos_theta_2 * sin_psi_2 - sin_phi_2 * cos_psi_2 * sin_theta_2;
}

/*
 * The Euler rotation sequence is 3-2-1 (roll, pitch, yaw) from Body to World
 */
inline __host__ __device__ void Euler2DCM_NWU(const float& r, const float& p, const float& y, float M[3][3])
{
  float sin_phi, cos_phi, sin_theta, cos_theta, sin_psi, cos_psi;
#ifdef __CUDA_ARCH__
  float r_norm = angle_utils::normalizeAngle(r);
  float p_norm = angle_utils::normalizeAngle(p);
  float y_norm = angle_utils::normalizeAngle(y);
  __sincosf(r_norm, &sin_phi, &cos_phi);
  __sincosf(p_norm, &sin_theta, &cos_theta);
  __sincosf(y_norm, &sin_psi, &cos_psi);
#else
  sincosf(r, &sin_phi, &cos_phi);
  sincosf(p, &sin_theta, &cos_theta);
  sincosf(y, &sin_psi, &cos_psi);
#endif

  M[0][0] = cos_theta * cos_psi;
  M[0][1] = sin_phi * sin_theta * cos_psi - cos_phi * sin_psi;
  M[0][2] = cos_phi * sin_theta * cos_psi + sin_phi * sin_psi;
  M[1][0] = cos_theta * sin_psi;
  M[1][1] = sin_phi * sin_theta * sin_psi + cos_phi * cos_psi;
  M[1][2] = cos_phi * sin_theta * sin_psi - sin_phi * cos_psi;
  M[2][0] = -sin_theta;
  M[2][1] = sin_phi * cos_theta;
  M[2][2] = cos_phi * cos_theta;
}

/*
 * The Euler rotation sequence is 3-2-1 (roll, pitch, yaw) from Body to World
 */
inline __host__ __device__ void Euler2DCM_NWU(const float& r, const float& p, const float& y,
                                              Eigen::Ref<Eigen::Matrix3f> M)
{
  float sin_phi, cos_phi, sin_theta, cos_theta, sin_psi, cos_psi;
#ifdef __CUDA_ARCH__
  float r_norm = angle_utils::normalizeAngle(r);
  float p_norm = angle_utils::normalizeAngle(p);
  float y_norm = angle_utils::normalizeAngle(y);
  __sincosf(r_norm, &sin_phi, &cos_phi);
  __sincosf(p_norm, &sin_theta, &cos_theta);
  __sincosf(y_norm, &sin_psi, &cos_psi);
#else
  sincosf(r, &sin_phi, &cos_phi);
  sincosf(p, &sin_theta, &cos_theta);
  sincosf(y, &sin_psi, &cos_psi);
#endif

  M(0, 0) = cos_theta * cos_psi;
  M(0, 1) = sin_phi * sin_theta * cos_psi - cos_phi * sin_psi;
  M(0, 2) = cos_phi * sin_theta * cos_psi + sin_phi * sin_psi;
  M(1, 0) = cos_theta * sin_psi;
  M(1, 1) = sin_phi * sin_theta * sin_psi + cos_phi * cos_psi;
  M(1, 2) = cos_phi * sin_theta * sin_psi - sin_phi * cos_psi;
  M(2, 0) = -sin_theta;
  M(2, 1) = sin_phi * cos_theta;
  M(2, 2) = cos_phi * cos_theta;
}

// (RPY rotation sequence)
/*
 * Returns an euler sequence 3-2-1 (roll pitch yaw) that when applied takes you from body to world
 */
inline void __host__ __device__ Quat2EulerNWU(const Eigen::Quaternionf& q, float& r, float& p, float& y)
{
  r = atan2f(2.0f * q.z() * q.y() + 2.0f * q.w() * q.x(),
             q.w() * q.w() + q.z() * q.z() - q.y() * q.y() - q.x() * q.x());
  float temp = -2.0f * q.w() * q.y() + 2.0f * q.x() * q.z();
  p = -asinf(fmaxf(-1.0f, fminf(temp, 1.0f)));
  y = atan2f(2.0f * q.y() * q.x() + 2.0f * q.z() * q.w(),
             q.w() * q.w() + q.x() * q.x() - q.y() * q.y() - q.z() * q.z());
}

inline void Quat2DCM(const Eigen::Quaternionf& q, Eigen::Ref<Eigen::Matrix3f> DCM)
{
  DCM = q.toRotationMatrix();
}

inline __device__ void omega2edot(const float p, const float q, const float r, const float e[4], float ed[4])
{
  ed[0] = 0.5f * (-p * e[1] - q * e[2] - r * e[3]);
  ed[1] = 0.5f * (p * e[0] - q * e[3] + r * e[2]);
  ed[2] = 0.5f * (p * e[3] + q * e[0] - r * e[1]);
  ed[3] = 0.5f * (-p * e[2] + q * e[1] + r * e[0]);
}

// Can't use Eigen::Ref on Quaternions
inline void omega2edot(const float p, const float q, const float r, const Eigen::Quaternionf& e, Eigen::Quaternionf& ed)
{
  ed.w() = 0.5f * (-p * e.x() - q * e.y() - r * e.z());
  ed.x() = 0.5f * (p * e.w() - q * e.z() + r * e.y());
  ed.y() = 0.5f * (p * e.z() + q * e.w() - r * e.x());
  ed.z() = 0.5f * (-p * e.y() + q * e.x() + r * e.w());
}

inline __host__ __device__ void bodyOffsetToWorldPoseQuat(const float3& offset, const float3& body_pose,
                                                          const float q[4], float3& output)
{
  // rotate body vector into world frame
  float3 rotated_offset = make_float3(offset.x, offset.y, offset.z);
  RotatePointByQuat(q, rotated_offset);
  // add offset to body pose
  output.x = body_pose.x + rotated_offset.x;
  output.y = body_pose.y + rotated_offset.y;
  output.z = body_pose.z + rotated_offset.z;
}

inline __host__ __device__ void bodyOffsetToWorldPoseQuat(const float3& offset, const float3& body_pose,
                                                          const Eigen::Quaternionf& q, float3& output)
{
  // rotate body vector into world frame
  float3 rotated_offset = make_float3(offset.x, offset.y, offset.z);
  RotatePointByQuat(q, rotated_offset);
  // add offset to body pose
  output.x = body_pose.x + rotated_offset.x;
  output.y = body_pose.y + rotated_offset.y;
  output.z = body_pose.z + rotated_offset.z;
}

inline __host__ __device__ void bodyOffsetToWorldPoseQuat(const Eigen::Ref<Eigen::Vector3f>& offset,
                                                          const Eigen::Ref<Eigen::Vector3f>& body_pose,
                                                          const Eigen::Quaternionf& q,
                                                          Eigen::Ref<Eigen::Vector3f> output)
{
  RotatePointByQuat(q, offset, output);
  // add offset to body pose
  output += body_pose;
}

inline __host__ __device__ void bodyOffsetToWorldPoseEuler(const float3& offset, const float3& body_pose,
                                                           const float3& rotation, float3& output)
{
  // convert RPY to Rotation Matrix
  float M[3][3];
  math::Euler2DCM_NWU(rotation.x, rotation.y, rotation.z, M);
  RotatePointByDCM(M, offset, output);

  // add offset to body pose
  output.x += body_pose.x;
  output.y += body_pose.y;
  output.z += body_pose.z;
}

inline __host__ __device__ void bodyOffsetToWorldPoseEuler(const float3& offset, const float3& body_pose,
                                                           const Eigen::Ref<Eigen::Vector3f>& rotation, float3& output)
{
  // convert RPY to Rotation Matrix
  float M[3][3];
  math::Euler2DCM_NWU(rotation.x(), rotation.y(), rotation.z(), M);
  RotatePointByDCM(M, offset, output);

  // add offset to body pose
  output.x += body_pose.x;
  output.y += body_pose.y;
  output.z += body_pose.z;
}

inline __host__ __device__ void bodyOffsetToWorldPoseEuler(const Eigen::Ref<Eigen::Vector3f>& offset,
                                                           const Eigen::Ref<Eigen::Vector3f>& body_pose,
                                                           const Eigen::Ref<Eigen::Vector3f>& rotation,
                                                           Eigen::Ref<Eigen::Vector3f> output)
{
  // convert RPY to Rotation Matrix
  Eigen::Matrix3f M;
  math::Euler2DCM_NWU(rotation.x(), rotation.y(), rotation.z(), M);
  RotatePointByDCM(M, offset, output);
  // add offset to body pose
  output += body_pose;
}

inline __host__ __device__ void bodyOffsetToWorldPoseDCM(const float3& offset, const float3& body_pose,
                                                         const float rotation[3][3], float3& output)
{
  RotatePointByDCM(rotation, offset, output);

  // add offset to body pose
  output.x += body_pose.x;
  output.y += body_pose.y;
  output.z += body_pose.z;
}

inline __host__ __device__ void bodyOffsetToWorldPoseDCM(const float3& offset, const float3& body_pose,
                                                         const Eigen::Ref<Eigen::Matrix3f>& rotation, float3& output)
{
  RotatePointByDCM(rotation, offset, output);

  // add offset to body pose
  output.x += body_pose.x;
  output.y += body_pose.y;
  output.z += body_pose.z;
}

inline __host__ __device__ void bodyOffsetToWorldPoseDCM(const Eigen::Ref<Eigen::Vector3f>& offset,
                                                         const Eigen::Ref<Eigen::Vector3f>& body_pose,
                                                         const Eigen::Ref<Eigen::Matrix3f>& rotation,
                                                         Eigen::Ref<Eigen::Vector3f> output)
{
  RotatePointByDCM(rotation, offset, output);

  // add offset to body pose
  output += body_pose;
}

inline __device__ __host__ Eigen::Matrix3f skewSymmetricMatrix(Eigen::Vector3f& v)
{
  Eigen::Matrix3f m;
  m << 0.0f, -v[2], v[1], v[2], 0.0f, -v[0], -v[1], v[0], 0.0f;
  return m;
}

inline __host__ double timeDiffms(const std::chrono::steady_clock::time_point& end,
                                  const std::chrono::steady_clock::time_point& start)
{
  return (end - start).count() / 1e6;
}

inline __host__ __device__ double normalCDF(double x)
{
  return 0.5 * erfc(-x * M_SQRT1_2);
}

inline __host__ std::vector<double> calculateCk(size_t steps)
{
  // calculate params only when more steps are required
  static std::vector<double> c_vec;
  if (c_vec.size() < steps)
  {
    c_vec.resize(steps, 0);
    c_vec[0] = 1.0;
    for (size_t k = 1; k <= steps; k++)
    {
      double c_k = 0;
      for (size_t m = 0; m < k; m++)
      {
        c_k += c_vec[m] * c_vec[k - 1 - m] / ((m + 1.0) * (2.0 * m + 1.0));
      }
      c_vec[k] = c_k;
    }
  }
  return c_vec;
}

/**
 * Implementation based on
 * https://en.wikipedia.org/wiki/Error_function#Inverse_functions and Horner's
 * method
 */
inline __host__ double inverseErrorFunc(double x, int num_precision = 5)
{
  std::vector<double> c_k = calculateCk(num_precision);
  double output = 0;
  for (int i = num_precision; i > 0; i--)
  {
    output = (c_k[i] / (2.0 * i + 1.0) + output) * x * x * M_PI / 4.0;
  }
  output = (output + c_k[0]) * x / M_2_SQRTPI;
  return output;
}

inline __host__ double inverseErrorFuncSlow(double x, int num_precision = 5)
{
  std::vector<double> c_k = calculateCk(num_precision);
  double slow_output = 0;
  for (int i = 0; i <= num_precision; i++)
  {
    slow_output += c_k[i] / (2.0 * i + 1.0) * std::pow(x / M_2_SQRTPI, 2 * i + 1);
  }
  return slow_output;
}

/**
 * https://en.wikipedia.org/wiki/Normal_distribution#Quantile_function
 */
inline __host__ double inverseNormalCDF(double x, int num_precision = 10)
{
  return M_SQRT2 * inverseErrorFunc(2.0 * x - 1.0, num_precision);
}

inline __host__ double inverseNormalCDFSlow(double x, int num_precision = 10)
{
  return M_SQRT2 * inverseErrorFuncSlow(2.0 * x - 1.0, num_precision);
}

inline __device__ __host__ float clamp(float value, float min, float max)
{
  return fminf(fmaxf(value, min), max);
}

inline __device__ __host__ float sign(float value)
{
  return value >= 0 ? 1 : -1;
}

}  // namespace math

}  // namespace mppi











/**
 * @class Managed managed.cuh
 * @brief Class for setting the stream to be used by dynamics and cost functions used
 * by MPPIController.
 *
 * This class has one variable, which is the CUDA stream, and a function which sets the
 * stream. It is meant to be inherited by costs and dynamics classes which are passed into
 * the MPPIController kernels. In the past, this class used unified memory (hence the managed name),
 * so that classes could be passed by reference to CUDA kernels. However, as of right now,
 * the difficulties of using unified memory with multi-threaded CPU programs make getting
 * good performance with unified memory difficult, so this has been removed. Future implementations
 * may bring back the unified memory feature.
 */
class Managed
{
public:
  cudaStream_t stream_ = 0;  ///< The CUDA Stream that the class is bound too. 0 is the default (NULL) stream.

  // true when allocated
  bool GPUMemStatus_ = false;

  Managed(cudaStream_t stream = 0)
  {
    this->bindToStream(stream);
    auto logger = std::make_shared<mppi::util::MPPILogger>();
    setLogger(logger);
  }

  /**
  @brief Sets the stream and synchronizes the device.
  @param stream is the CUDA stream that the object is assigned too.
  */
  void bindToStream(cudaStream_t stream)
  {
    stream_ = stream;
    cudaDeviceSynchronize();
  }

  // REQUIRED: basic interface, make sure to implement in each class
  // GPUSetup - allocates the GPU object
  // freeCudaMem -> deallocates what is setup in GPUSetup
  // getParams, setParams -> gets and sets the parameters
  // paramsToDevice -> copies the parameters over to the GPU side

  // OPTIONAL:
  // printParams
  // other printing methods

  __host__ void setLogger(const mppi::util::MPPILoggerPtr& logger)
  {
    logger_ = logger;
  }

  __host__ void setLogLevel(const mppi::util::LOG_LEVEL& level)
  {
    logger_->setLogLevel(level);
  }

  __host__ mppi::util::MPPILoggerPtr getLogger()
  {
    return logger_;
  }

  __host__ mppi::util::MPPILoggerPtr getLogger() const
  {
    return logger_;
  }

  __device__ __host__ int getGrdSharedSizeBytes() const
  {
    return SHARED_MEM_REQUEST_GRD_BYTES;
  }
  __device__ __host__ int getBlkSharedSizeBytes() const
  {
    return SHARED_MEM_REQUEST_BLK_BYTES;
  }


protected:
  template <class T>
  static T* GPUSetup(T* host_ptr)
  {
    // Allocate enough space on the GPU for the object
    T* device_ptr;
    cudaMalloc((void**)&device_ptr, sizeof(T));
    // Cudamemcpy
    HANDLE_ERROR(cudaMemcpyAsync(device_ptr, host_ptr, sizeof(T), cudaMemcpyHostToDevice, host_ptr->stream_));
    cudaDeviceSynchronize();
    host_ptr->GPUMemStatus_ = true;
    return device_ptr;
  }
  mppi::util::MPPILoggerPtr logger_ = nullptr;

  int SHARED_MEM_REQUEST_GRD_BYTES = 0;  ///< Amount of shared memory we need per BLOCK.
  int SHARED_MEM_REQUEST_BLK_BYTES = 0;  ///< Amount of shared memory we need per ROLLOUT.

  // TODO CRTP template this on the base class for allocation and dealloation
};













// CUDA barriers were first implemented in CUDA 11
#if defined(CMAKE_USE_CUDA_BARRIERS) && defined(CUDART_VERSION) && CUDART_VERSION > 11000
#include <cuda/barrier>
using barrier = cuda::barrier<cuda::thread_scope_block>;

// Turn on/off various CUDA barriers from CMake configuration
#ifdef CMAKE_USE_CUDA_BARRIERS_DYN
#define USE_CUDA_BARRIERS_DYN
#endif
#ifdef CMAKE_USE_CUDA_BARRIERS_COST
#define USE_CUDA_BARRIERS_COST
#endif
#ifdef CMAKE_USE_CUDA_BARRIERS_ROLLOUT
#define USE_CUDA_BARRIERS_ROLLOUT
#endif
#endif

#include <cooperative_groups.h>
namespace cg = cooperative_groups;
namespace mp1 = mppi::p1;


namespace mppi
{
namespace kernels
{
/*******************************************************************************************************************
 * Kernel functions
 *******************************************************************************************************************/
template <class COST_T, class SAMPLING_T, bool COALESCE = true>
__global__ void rolloutCostKernel(COST_T* __restrict__ costs, SAMPLING_T* __restrict__ sampling, float dt,
                                  const int num_timesteps, const int num_rollouts, float lambda, float alpha,
                                  const float* __restrict__ y_d, float* __restrict__ trajectory_costs_d);

template <class DYN_T, class COST_T, class SAMPLING_T>
__global__ void rolloutKernel(DYN_T* __restrict__ dynamics, SAMPLING_T* __restrict__ sampling,
                              COST_T* __restrict__ costs, float dt, const int num_timesteps, const int num_rollouts,
                              const float* __restrict__ init_x_d, float lambda, float alpha,
                              float* __restrict__ trajectory_costs_d);

template <class DYN_T, class SAMPLING_T>
__global__ void rolloutDynamicsKernel(DYN_T* __restrict__ dynamics, SAMPLING_T* __restrict__ sampling, float dt,
                                      const int num_timesteps, const int num_rollouts,
                                      const float* __restrict__ init_x_d, float* __restrict__ y_d);

template <class DYN_T, class COST_T, class SAMPLING_T>
__global__ void visualizeKernel(DYN_T* __restrict__ dynamics, SAMPLING_T* __restrict__ sampling,
                                COST_T* __restrict__ costs, float dt, const int num_timesteps, const int num_rollouts,
                                const float* __restrict__ init_x_d, float lambda, float alpha, float* __restrict__ y_d,
                                float* __restrict__ cost_traj_d, int* __restrict__ crash_status_d);

template <class COST_T, class SAMPLING_T, bool COALESCE = true>
__global__ void visualizeCostKernel(COST_T* __restrict__ costs, SAMPLING_T* __restrict__ sampling, float dt,
                                    const int num_timesteps, const int num_rollouts, const float lambda, float alpha,
                                    const float* __restrict__ y_d, float* __restrict__ cost_traj_d,
                                    int* __restrict__ crash_status_d);

template <int CONTROL_DIM>
__global__ void weightedReductionKernel(const float* __restrict__ exp_costs_d, const float* __restrict__ du_d,
                                        float* __restrict__ new_u_d, const float normalizer, const int num_timesteps,
                                        const int num_rollouts, const int sum_stride);

// Norm Exponential Kernel
__global__ void normExpKernel(int num_rollouts, float* trajectory_costs_d, float gamma, float baseline);
// Tsallis Kernel
__global__ void TsallisKernel(int num_rollouts, float* trajectory_costs_d, float gamma, float r, float baseline);

/*******************************************************************************************************************
 * RolloutKernel Helpers
 *******************************************************************************************************************/
/*
 * loadGlobalToShared
 * Copy global memory into shared memory
 *
 * Args:
 * state_dim: Number of states, defined in DYN_T
 * control_dim: Number of controls, defined in DYN_T
 * num_rollouts: Total number of rollouts
 * blocksize_y: Y dimension of each block of threads
 * global_idx: Current rollout index.
 * thread_idy: Current y index of block dimension.
 * thread_idz: Current z index of block dimension.
 * x0_device: initial condition in device memory
 * x_thread: state in shared memory
 * xdot_thread: state_dot in shared memory
 * u_thread: control / perturbed control in shared memory
 *
 */
template <int STATE_DIM, int CONTROL_DIM>
__device__ void loadGlobalToShared(const int num_rollouts, const int blocksize_y, const int global_idx,
                                   const int thread_idy, const int thread_idz, const float* __restrict__ x_device,
                                   float* __restrict__ x_thread, float* __restrict__ xdot_thread,
                                   float* __restrict__ u_thread);

/**
 * @brief Calculate the terminal cost and add it to the trajectories' overall cost
 *
 * @tparam COST_T - Cost Function class
 * @param num_rollouts
 * @param num_timesteps
 * @param global_idx - sample trajectory index
 * @param costs - GPU version of cost function
 * @param x_thread - terminal state x
 * @param running_cost - current cost of the sample trajectory
 * @param theta_c - shared memory for the cost function
 * @param cost_rollouts_device - global memory array storing the cost of each sample
 *
 * @return
 */
template <class COST_T>
__device__ void computeAndSaveCost(int num_rollouts, int num_timesteps, int global_idx, COST_T* costs, float* x_thread,
                                   float running_cost, float* theta_c, float* cost_rollouts_device);

/**
 * @brief conduct a warp reduction of addition s[tid * stride] += s[(tid + BLOCKSIZE) * stride]
 *
 * @tparam BLOCKSIZE - how many threads are doing the reduction
 * @param sdata - float array to do the reduction on
 * @param tid - current thread index
 * @param stride - how spaced out the summations should be
 *
 * @return
 */
template <int BLOCKSIZE>
__device__ void warpReduceAdd(volatile float* sdata, const int tid, const int stride = 1);

/**
 * @brief conduct a sum of floats in array through a GPU reduction algorithm
 *
 * @param running_cost - array of floats to be summed
 * @param start_size - number of items to be summed
 * @param index - GPU thread index
 * @param step - GPU step to avoid overlap of threads
 * @param catch_condition - when to stop summation
 * @param stride - how far apart the desired floats are in the array
 *
 * @return
 */
__device__ inline void costArrayReduction(float* running_cost, const int start_size, const int index, const int step,
                                          const bool catch_condition, const int stride = 1);

// Norm Exp Kernel Helpers
__device__ __host__ inline void normExpTransform(const int num_rollouts, float* __restrict__ trajectory_costs_d,
                                                 const float lambda_inv, const float baseline, const int global_idx,
                                                 const int rollout_idx_step);
// Tsallis Kernel Helpers
__device__ __host__ inline void TsallisTransform(const int num_rollouts, float* __restrict__ trajectory_costs_d,
                                                 const float gamma, float r, const float baseline, const int global_idx,
                                                 const int rollout_idx_step);
float computeBaselineCost(float* cost_rollouts_host, int num_rollouts);

float computeNormalizer(float* cost_rollouts_host, int num_rollouts);

float constructBestWeights(float* cost_rollouts_host, int num_rollouts);

int computeBestIndex(float* cost_rollouts_host, int num_rollouts);

/**
 * Calculates the free energy mean and variance from the different
 * cost trajectories after normExpKernel
 * Inputs:
 *  cost_rollouts_host - sampled cost trajectories
 *  num_rollouts - the number of sampled cost trajectories
 *  lambda - the lambda term from the definition of free energy
 *  baseline - minimum cost trajectory
 * Outputs:
 *  free_energy - the free energy of the samples
 *  free_energy_var - the variance of the free energy calculation
 */
void computeFreeEnergy(float& free_energy, float& free_energy_var, float& free_energy_modified,
                       float* cost_rollouts_host, int num_rollouts, float baseline, float lambda = 1.0);

/*******************************************************************************************************************
 * Weighted Reduction Kernel Helpers
 *******************************************************************************************************************/
/**
 * @brief set controls to zero
 *
 * @param control_dim
 * @param thread_idx - threadIdx.x
 * @param u - memory for control array
 * @param u_intermediate - shared memory for control array
 *
 * @return
 */
__device__ void setInitialControlToZero(int control_dim, int thread_idx, float* __restrict__ u,
                                        float* __restrict__ u_intermediate);

/**
 * @brief calculated the weighted sum of the controls
 *
 * @param num_rollouts
 * @param num_timesteps
 * @param sum_stride - how many summations to do in a single thread
 * @param thread_idx - threadIdx.x
 * @param block_idx - blockIdx.x
 * @param control_dim
 * @param exp_costs_d - global memory of the weights
 * @param normalizer - sum of all the weights to use as a normalizing term
 * @param du_d - global memory of all sampled controls
 * @param u - local memory to store a control from a single time
 * @param u_intermediate - shared memory containing the weighted sum
 *
 * @return
 */
__device__ void strideControlWeightReduction(const int num_rollouts, const int num_timesteps, const int sum_stride,
                                             const int thread_idx, const int block_idx, const int control_dim,
                                             const float* __restrict__ exp_costs_d, const float normalizer,
                                             const float* __restrict__ du_d, float* __restrict__ u,
                                             float* __restrict__ u_intermediate);

__device__ void rolloutWeightReductionAndSaveControl(int thread_idx, int block_idx, int num_rollouts, int num_timesteps,
                                                     int control_dim, int sum_stride, float* u, float* u_intermediate,
                                                     float* du_new_d);

/**
 * Launch Kernel Methods
 **/
template <class DYN_T, class COST_T, typename SAMPLING_T, bool COALESCE = true>
void launchSplitRolloutKernel(DYN_T* __restrict__ dynamics, COST_T* __restrict__ costs,
                              SAMPLING_T* __restrict__ sampling, float dt, const int num_timesteps,
                              const int num_rollouts, float lambda, float alpha, float* __restrict__ init_x_d,
                              float* __restrict__ y_d, float* __restrict__ trajectory_costs, dim3 dimDynBlock,
                              dim3 dimCostBlock, cudaStream_t stream, bool synchronize = true);

template <class DYN_T, class COST_T, typename SAMPLING_T>
void launchRolloutKernel(DYN_T* __restrict__ dynamics, COST_T* __restrict__ costs, SAMPLING_T* __restrict__ sampling,
                         float dt, const int num_timesteps, const int num_rollouts, float lambda, float alpha,
                         float* __restrict__ init_x_d, float* __restrict__ trajectory_costs, dim3 dimBlock,
                         cudaStream_t stream, bool synchronize = true);

template <class COST_T, class SAMPLING_T, bool COALESCE = true>
void launchVisualizeCostKernel(COST_T* __restrict__ costs, SAMPLING_T* __restrict__ sampling, float dt,
                               const int num_timesteps, const int num_rollouts, float lambda, float alpha,
                               float* __restrict__ y_d, int* __restrict__ sampled_crash_status_d,
                               float* __restrict__ cost_traj_result, dim3 dimBlock, cudaStream_t stream,
                               bool synchronize = true);

template <class DYN_T, class COST_T, typename SAMPLING_T>
void launchVisualizeKernel(DYN_T* __restrict__ dynamics, COST_T* __restrict__ costs, SAMPLING_T* __restrict__ sampling,
                           float dt, const int num_timesteps, const int num_rollouts, float lambda, float alpha,
                           float* __restrict__ init_x_d, float* __restrict__ y_d, float* __restrict__ trajectory_costs,
                           int* __restrict__ crash_status_d, dim3 dimVisBlock, cudaStream_t stream,
                           bool synchronize = true);

template <int CONTROL_DIM>
void launchWeightedReductionKernel(const float* __restrict__ exp_costs_d, const float* __restrict__ du_d,
                                   float* __restrict__ new_u_d, const float normalizer, const int num_timesteps,
                                   const int num_rollouts, const int sum_stride, cudaStream_t stream,
                                   bool synchronize = true);

void launchNormExpKernel(int num_rollouts, int blocksize_x, float* trajectory_costs_d, float lambda_inv, float baseline,
                         cudaStream_t stream, bool synchronize = true);

void launchTsallisKernel(int num_rollouts, int blocksize_x, float* trajectory_costs_d, float gamma, float r,
                         float baseline, cudaStream_t stream, bool synchronize = true);

template <class DYN_T, int NUM_ROLLOUTS, int SUM_STRIDE>
void launchWeightedReductionKernel(float* exp_costs_d, float* du_d, float* du_new_d, float normalizer,
                                   int num_timesteps, cudaStream_t stream, bool synchronize = true);

/*******************************************************************************************************************
 * Shared Memory Calculators for various kernels
 *******************************************************************************************************************/
template <class DYN_T, class SAMPLER_T>
unsigned calcRolloutDynamicsKernelSharedMemSize(const DYN_T* dynamics, const SAMPLER_T* sampler, dim3& dimBlock);

template <class COST_T, class SAMPLER_T>
unsigned calcRolloutCostKernelSharedMemSize(const COST_T* cost, const SAMPLER_T* sampler, dim3& dimBlock);

template <class DYN_T, class COST_T, class SAMPLER_T>
unsigned calcRolloutCombinedKernelSharedMemSize(const DYN_T* dynamics, const COST_T* cost, const SAMPLER_T* sampler,
                                                dim3& dimBlock);

template <class DYN_T, class COST_T, class SAMPLER_T>
unsigned calcVisualizeKernelSharedMemSize(const DYN_T* dynamics, const COST_T* cost, const SAMPLER_T* sampler,
                                          const int& num_timesteps, dim3& dimBlock);

template <class COST_T, class SAMPLER_T>
unsigned calcVisCostKernelSharedMemSize(const COST_T* cost, const SAMPLER_T* sampler, const int& num_timesteps,
                                        dim3& dimBlock);

template <class T>
__host__ __device__ inline unsigned calcClassSharedMemSize(const T* class_ptr, const dim3& dimBlock);
}  // namespace kernels
}  // namespace mppi



namespace mppi
{
namespace kernels
{
/*******************************************************************************************************************
 * Kernel Functions
 *******************************************************************************************************************/
template <class DYN_T, class COST_T, class SAMPLING_T>
__global__ void rolloutKernel(DYN_T* __restrict__ dynamics, SAMPLING_T* __restrict__ sampling,
                              COST_T* __restrict__ costs, float dt, const int num_timesteps, const int num_rollouts,
                              const float* __restrict__ init_x_d, float lambda, float alpha,
                              float* __restrict__ trajectory_costs_d)
{
  // Get thread and block id
  const int thread_idx = threadIdx.x;
  const int thread_idy = threadIdx.y;
  const int thread_idz = threadIdx.z;
  const int block_idx = blockIdx.x;
  const int global_idx = blockDim.x * block_idx + thread_idx;
  const int shared_idx = blockDim.x * thread_idz + thread_idx;
  const int distribution_idx = threadIdx.z;
  const int distribution_dim = blockDim.z;
  const int sample_dim = blockDim.x;
  // Ensure that there is enough room for the SHARED_MEM_REQUEST_GRD_BYTES and SHARED_MEM_REQUEST_BLK_BYTES portions to
  // be aligned to the float4 boundary.
  const int size_of_theta_s_bytes = calcClassSharedMemSize(dynamics, blockDim);
  const int size_of_theta_d_bytes = calcClassSharedMemSize(sampling, blockDim);
  const int size_of_theta_c_bytes = calcClassSharedMemSize(costs, blockDim);

  // Create shared state and control arrays
  extern __shared__ float entire_buffer[];

  float* x_shared = entire_buffer;
  float* x_next_shared = &x_shared[math::nearest_multiple_4(sample_dim * DYN_T::STATE_DIM * distribution_dim)];
  float* y_shared = &x_next_shared[math::nearest_multiple_4(sample_dim * DYN_T::STATE_DIM * distribution_dim)];
  float* x_dot_shared = &y_shared[math::nearest_multiple_4(sample_dim * DYN_T::OUTPUT_DIM * distribution_dim)];
  float* u_shared = &x_dot_shared[math::nearest_multiple_4(sample_dim * DYN_T::STATE_DIM * distribution_dim)];
  float* theta_s_shared = &u_shared[math::nearest_multiple_4(sample_dim * DYN_T::CONTROL_DIM * distribution_dim)];
  float* theta_d_shared = &theta_s_shared[size_of_theta_s_bytes / sizeof(float)];
  float* theta_c_shared = &theta_d_shared[size_of_theta_d_bytes / sizeof(float)];
  float* running_cost_shared = &theta_c_shared[size_of_theta_c_bytes / sizeof(float)];
  int* crash_status_shared = (int*)&running_cost_shared[math::nearest_multiple_4(blockDim.x * blockDim.y * blockDim.z)];

#ifdef USE_CUDA_BARRIERS_ROLLOUT
  barrier* barrier_shared = (barrier*)&crash_status_shared[math::nearest_multiple_4(sample_dim * distribution_dim)];
#endif

  // Create local state, state dot and controls
  int running_cost_index = thread_idx + blockDim.x * (thread_idy + blockDim.y * thread_idz);
  float* x = &x_shared[shared_idx * DYN_T::STATE_DIM];
  float* x_next = &x_next_shared[shared_idx * DYN_T::STATE_DIM];
  float* x_temp;
  float* xdot = &x_dot_shared[shared_idx * DYN_T::STATE_DIM];
  float* u = &u_shared[shared_idx * DYN_T::CONTROL_DIM];
  float* y = &y_shared[shared_idx * DYN_T::OUTPUT_DIM];
  float* running_cost = &running_cost_shared[running_cost_index];
  running_cost[0] = 0.0f;
  int* crash_status = &crash_status_shared[shared_idx];
  crash_status[0] = 0;  // We have not crashed yet as of the first trajectory.
#ifdef USE_CUDA_BARRIERS_ROLLOUT
  barrier* bar = &barrier_shared[shared_idx];
  if (thread_idy == 0)
  {
    init(bar, blockDim.y);
  }
#endif

  // Load global array to shared array
  loadGlobalToShared<DYN_T::STATE_DIM, DYN_T::CONTROL_DIM>(num_rollouts, blockDim.y, global_idx, thread_idy, thread_idz,
                                                           init_x_d, x, xdot, u);
  __syncthreads();

  /*<----Start of simulation loop-----> */
  dynamics->initializeDynamics(x, u, y, theta_s_shared, 0.0f, dt);
  sampling->initializeDistributions(y, 0.0f, dt, theta_d_shared);
  costs->initializeCosts(y, u, theta_c_shared, 0.0f, dt);
  __syncthreads();
  for (int t = 0; t < num_timesteps; t++)
  {
    // Load noise trajectories scaled by the exploration factor
    sampling->readControlSample(global_idx, t, distribution_idx, u, theta_d_shared, blockDim.y, thread_idy, y);
#ifdef USE_CUDA_BARRIERS_ROLLOUT
    bar->arrive_and_wait();
#else
    __syncthreads();
#endif

    // applies constraints as defined in dynamics.cuh see specific dynamics class for what happens here
    // usually just control clamping
    dynamics->enforceConstraints(x, u);
#ifdef USE_CUDA_BARRIERS_ROLLOUT
    bar->arrive_and_wait();
#else
    __syncthreads();
#endif
    // Copy control constraints back to global memory
    sampling->writeControlSample(global_idx, t, distribution_idx, u, theta_d_shared, blockDim.y, thread_idy, y);

    // Increment states
    dynamics->step(x, x_next, xdot, u, y, theta_s_shared, t, dt);
#ifdef USE_CUDA_BARRIERS_ROLLOUT
    bar->arrive_and_wait();
#else
    __syncthreads();
#endif
    running_cost[0] +=
        costs->computeRunningCost(y, u, t, theta_c_shared, crash_status) +
        sampling->computeLikelihoodRatioCost(u, theta_d_shared, global_idx, t, distribution_idx, lambda, alpha);
#ifdef USE_CUDA_BARRIERS_ROLLOUT
    bar->arrive_and_wait();
#else
    __syncthreads();
#endif
    x_temp = x;
    x = x_next;
    x_next = x_temp;
  }

  // Add all costs together
  running_cost = &running_cost_shared[thread_idx + blockDim.x * blockDim.y * thread_idz];
  __syncthreads();
  costArrayReduction(running_cost, blockDim.y, thread_idy, blockDim.y, thread_idy == 0, blockDim.x);
  // Compute terminal cost and the final cost for each thread
  computeAndSaveCost(num_rollouts, num_timesteps, global_idx, costs, y, running_cost[0] / (num_timesteps),
                     theta_c_shared, trajectory_costs_d);
}

template <class COST_T, class SAMPLING_T, bool COALESCE>
__global__ void rolloutCostKernel(COST_T* __restrict__ costs, SAMPLING_T* __restrict__ sampling, float dt,
                                  const int num_timesteps, const int num_rollouts, float lambda, float alpha,
                                  const float* __restrict__ y_d, float* __restrict__ trajectory_costs_d)
{
  // Get thread and block id
  const int thread_idx = threadIdx.x;
  const int thread_idy = threadIdx.y;
  const int thread_idz = threadIdx.z;
  const int global_idx = blockIdx.x;
  const int distribution_idx = threadIdx.z;
  const int size_of_theta_d_bytes = calcClassSharedMemSize(sampling, blockDim);
  const int size_of_theta_c_bytes = calcClassSharedMemSize(costs, blockDim);

  int running_cost_index = thread_idx + blockDim.x * (thread_idy + blockDim.y * thread_idz);
  // Create shared state and control arrays
  extern __shared__ float entire_buffer[];
  float* y_shared = entire_buffer;
  float* u_shared = &y_shared[math::nearest_multiple_4(blockDim.x * blockDim.z * COST_T::OUTPUT_DIM)];
  float* running_cost_shared = &u_shared[math::nearest_multiple_4(blockDim.x * blockDim.z * COST_T::CONTROL_DIM)];
  int* crash_status_shared = (int*)&running_cost_shared[math::nearest_multiple_4(blockDim.x * blockDim.y * blockDim.z)];
  float* theta_c = (float*)&crash_status_shared[math::nearest_multiple_4(blockDim.x * blockDim.z)];
  float* theta_d = &theta_c[size_of_theta_c_bytes / sizeof(float)];
#ifdef USE_CUDA_BARRIERS_COST
  barrier* barrier_shared = (barrier*)&theta_d[size_of_theta_d_bytes / sizeof(float)];
#endif

  // Create local state, state dot and controls
  float* y;
  float* u;
  int* crash_status;

  // Initialize running cost and total cost
  float* running_cost;
  int sample_time_offset = 0;

  // Load global array to shared array
  y = &y_shared[(blockDim.x * thread_idz + thread_idx) * COST_T::OUTPUT_DIM];
  u = &u_shared[(blockDim.x * thread_idz + thread_idx) * COST_T::CONTROL_DIM];
  crash_status = &crash_status_shared[thread_idz * blockDim.x + thread_idx];
  crash_status[0] = 0;  // We have not crashed yet as of the first trajectory.
  running_cost = &running_cost_shared[running_cost_index];
  running_cost[0] = 0.0f;
#ifdef USE_CUDA_BARRIERS_COST
  barrier* bar = &barrier_shared[(blockDim.x * thread_idz + thread_idx)];
  if (thread_idy == 0)
  {
    init(bar, blockDim.y);
  }
#endif

  /*<----Start of simulation loop-----> */
#ifdef USE_COST_WITH_OFF_NUM_TIMESTEPS
  const int max_time_iters = ceilf((float)(num_timesteps - 2) / blockDim.x);
#else
  const int max_time_iters = ceilf((float)num_timesteps / blockDim.x);
#endif
  costs->initializeCosts(y, u, theta_c, 0.0f, dt);
  sampling->initializeDistributions(y, 0.0f, dt, theta_d);
  __syncthreads();
  for (int time_iter = 0; time_iter < max_time_iters; ++time_iter)
  {
    int t = thread_idx + time_iter * blockDim.x;
    if (COALESCE)
    {  // Fill entire shared mem sequentially using sequential threads_idx
      int amount_to_fill = (time_iter + 1) * blockDim.x > num_timesteps ? num_timesteps % blockDim.x : blockDim.x;
      mp1::loadArrayParallel<mp1::Parallel1Dir::THREAD_XY>(
          y_shared, blockDim.x * thread_idz * COST_T::OUTPUT_DIM, y_d,
          ((num_rollouts * thread_idz + global_idx) * num_timesteps + time_iter * blockDim.x) * COST_T::OUTPUT_DIM,
          COST_T::OUTPUT_DIM * amount_to_fill);
    }
    else if (t < num_timesteps)
    {  // t = num_timesteps is the terminal state for outside this for-loop
      sample_time_offset = (num_rollouts * thread_idz + global_idx) * num_timesteps + t;
      mp1::loadArrayParallel<COST_T::OUTPUT_DIM>(y, 0, y_d, sample_time_offset * COST_T::OUTPUT_DIM);
    }
    if (t < num_timesteps)
    {  // load controls from t = 0 to t = num_timesteps - 1
      sampling->readControlSample(global_idx, t, distribution_idx, u, theta_d, blockDim.y, thread_idy, y);
    }
#ifdef USE_CUDA_BARRIERS_COST
    bar->arrive_and_wait();
#else
    __syncthreads();
#endif

    // Compute cost
    if (t < num_timesteps)
    {
      running_cost[0] +=
          costs->computeRunningCost(y, u, t, theta_c, crash_status) +
          sampling->computeLikelihoodRatioCost(u, theta_d, global_idx, t, distribution_idx, lambda, alpha);
    }
#ifdef USE_CUDA_BARRIERS_COST
    bar->arrive_and_wait();
#else
    __syncthreads();
#endif
  }

  // Add all costs together
  running_cost = &running_cost_shared[blockDim.x * blockDim.y * thread_idz];
  __syncthreads();
  costArrayReduction(running_cost, blockDim.x * blockDim.y, thread_idx + blockDim.x * thread_idy,
                     blockDim.x * blockDim.y, thread_idx == blockDim.x - 1 && thread_idy == 0);
  // point every thread to the last output at t = NUM_TIMESTEPS for terminal cost calculation
  const int last_y_index = (num_timesteps - 1) % blockDim.x;
  y = &y_shared[(blockDim.x * thread_idz + last_y_index) * COST_T::OUTPUT_DIM];
#ifdef USE_COST_WITH_OFF_NUM_TIMESTEPS
  // load last output array
  const int t = num_timesteps - 1;
  mp1::loadArrayParallel<mp1::Parallel1Dir::THREAD_XY>(
      y_shared, (blockDim.x * thread_idz + last_y_index) * COST_T::OUTPUT_DIM, y_d,
      ((global_idx + num_rollouts * thread_idz) * num_timesteps + t) * COST_T::OUTPUT_DIM, COST_T::OUTPUT_DIM);
  __syncthreads();
#endif
  // Compute terminal cost and the final cost for each thread
  computeAndSaveCost(num_rollouts, num_timesteps, global_idx, costs, y, running_cost[0] / (num_timesteps), theta_c,
                     trajectory_costs_d);
}

template <class DYN_T, class SAMPLING_T>
__global__ void rolloutDynamicsKernel(DYN_T* __restrict__ dynamics, SAMPLING_T* __restrict__ sampling, float dt,
                                      const int num_timesteps, const int num_rollouts,
                                      const float* __restrict__ init_x_d, float* __restrict__ y_d)
{
  // Get thread and block id
  const int thread_idx = threadIdx.x;
  const int thread_idy = threadIdx.y;
  const int thread_idz = threadIdx.z;
  const int block_idx = blockIdx.x;
  const int global_idx = blockDim.x * block_idx + thread_idx;
  const int shared_idx = blockDim.x * thread_idz + thread_idx;
  const int distribution_idx = threadIdx.z;
  const int distribution_dim = blockDim.z;
  const int sample_dim = blockDim.x;
  // Ensure that there is enough room for the SHARED_MEM_REQUEST_GRD_BYTES and SHARED_MEM_REQUEST_BLK_BYTES portions to
  // be aligned to the float4 boundary.
  const int size_of_theta_s_bytes = calcClassSharedMemSize(dynamics, blockDim);
  const int size_of_theta_d_bytes = calcClassSharedMemSize(sampling, blockDim);

  // Create shared state and control arrays
  extern __shared__ float entire_buffer[];

  float* x_shared = entire_buffer;
  float* x_next_shared = &x_shared[math::nearest_multiple_4(sample_dim * DYN_T::STATE_DIM * distribution_dim)];
  float* y_shared = &x_next_shared[math::nearest_multiple_4(sample_dim * DYN_T::STATE_DIM * distribution_dim)];
  float* x_dot_shared = &y_shared[math::nearest_multiple_4(sample_dim * DYN_T::OUTPUT_DIM * distribution_dim)];
  float* u_shared = &x_dot_shared[math::nearest_multiple_4(sample_dim * DYN_T::STATE_DIM * distribution_dim)];
  float* theta_s_shared = &u_shared[math::nearest_multiple_4(sample_dim * DYN_T::CONTROL_DIM * distribution_dim)];
  float* theta_d_shared = &theta_s_shared[size_of_theta_s_bytes / sizeof(float)];
#ifdef USE_CUDA_BARRIERS_DYN
  barrier* barrier_shared = (barrier*)&theta_d_shared[size_of_theta_d_bytes / sizeof(float)];
#endif

  // Create local state, state dot and controls
  float* x = &x_shared[shared_idx * DYN_T::STATE_DIM];
  float* x_next = &x_next_shared[shared_idx * DYN_T::STATE_DIM];
  float* x_temp;
  float* xdot = &x_dot_shared[shared_idx * DYN_T::STATE_DIM];
  float* u = &u_shared[shared_idx * DYN_T::CONTROL_DIM];
  float* y = &y_shared[shared_idx * DYN_T::OUTPUT_DIM];
#ifdef USE_CUDA_BARRIERS_DYN
  barrier* bar = &barrier_shared[shared_idx];
  if (thread_idy == 0)
  {
    init(bar, blockDim.y);
  }
#endif

  // Load global array to shared array
  loadGlobalToShared<DYN_T::STATE_DIM, DYN_T::CONTROL_DIM>(num_rollouts, blockDim.y, global_idx, thread_idy, thread_idz,
                                                           init_x_d, x, xdot, u);
  __syncthreads();

  /*<----Start of simulation loop-----> */
  dynamics->initializeDynamics(x, u, y, theta_s_shared, 0.0f, dt);
  sampling->initializeDistributions(y, 0.0f, dt, theta_d_shared);
  __syncthreads();
  for (int t = 0; t < num_timesteps; t++)
  {
    // Load noise trajectories scaled by the exploration factor
    sampling->readControlSample(global_idx, t, distribution_idx, u, theta_d_shared, blockDim.y, thread_idy, y);
#ifdef USE_CUDA_BARRIERS_DYN
    bar->arrive_and_wait();
#else
    __syncthreads();
#endif

    // applies constraints as defined in dynamics.cuh see specific dynamics class for what happens here
    // usually just control clamping
    dynamics->enforceConstraints(x, u);
#ifdef USE_CUDA_BARRIERS_DYN
    bar->arrive_and_wait();
#else
    __syncthreads();
#endif
    // Copy control constraints back to global memory
    sampling->writeControlSample(global_idx, t, distribution_idx, u, theta_d_shared, blockDim.y, thread_idy, y);

    // Increment states
    dynamics->step(x, x_next, xdot, u, y, theta_s_shared, t, dt);
#ifdef USE_CUDA_BARRIERS_DYN
    bar->arrive_and_wait();
#else
    __syncthreads();
#endif
    x_temp = x;
    x = x_next;
    x_next = x_temp;
    // Copy state to global memory
    int sample_time_offset = (num_rollouts * thread_idz + global_idx) * num_timesteps + t;
    mp1::loadArrayParallel<DYN_T::OUTPUT_DIM>(y_d, sample_time_offset * DYN_T::OUTPUT_DIM, y, 0);
  }
}

template <class DYN_T, class COST_T, class SAMPLING_T>
__global__ void visualizeKernel(DYN_T* __restrict__ dynamics, SAMPLING_T* __restrict__ sampling,
                                COST_T* __restrict__ costs, float dt, const int num_timesteps, const int num_rollouts,
                                const float* __restrict__ init_x_d, float lambda, float alpha, float* __restrict__ y_d,
                                float* __restrict__ cost_traj_d, int* __restrict__ crash_status_d)
{
  // Get thread and block id
  const int thread_idx = threadIdx.x;
  const int thread_idy = threadIdx.y;
  const int thread_idz = threadIdx.z;
  const int block_idx = blockIdx.x;
  const int global_idx = blockDim.x * block_idx + thread_idx;
  const int shared_idx = blockDim.x * thread_idz + thread_idx;
  const int distribution_idx = threadIdx.z;
  const int distribution_dim = blockDim.z;
  const int sample_dim = blockDim.x;
  // Ensure that there is enough room for the SHARED_MEM_REQUEST_GRD_BYTES and SHARED_MEM_REQUEST_BLK_BYTES portions to
  // be aligned to the float4 boundary.
  const int size_of_theta_s_bytes = calcClassSharedMemSize(dynamics, blockDim);
  const int size_of_theta_d_bytes = calcClassSharedMemSize(sampling, blockDim);
  const int size_of_theta_c_bytes = calcClassSharedMemSize(costs, blockDim);

  // Create shared state and control arrays
  extern __shared__ float entire_buffer[];

  float* x_shared = entire_buffer;
  float* x_next_shared = &x_shared[math::nearest_multiple_4(sample_dim * DYN_T::STATE_DIM * distribution_dim)];
  float* y_shared = &x_next_shared[math::nearest_multiple_4(sample_dim * DYN_T::STATE_DIM * distribution_dim)];
  float* x_dot_shared = &y_shared[math::nearest_multiple_4(sample_dim * DYN_T::OUTPUT_DIM * distribution_dim)];
  float* u_shared = &x_dot_shared[math::nearest_multiple_4(sample_dim * DYN_T::STATE_DIM * distribution_dim)];
  float* theta_s_shared = &u_shared[math::nearest_multiple_4(sample_dim * DYN_T::CONTROL_DIM * distribution_dim)];
  float* theta_d_shared = &theta_s_shared[size_of_theta_s_bytes / sizeof(float)];
  float* theta_c_shared = &theta_d_shared[size_of_theta_d_bytes / sizeof(float)];
  float* running_cost_shared = &theta_c_shared[size_of_theta_c_bytes / sizeof(float)];
  int* crash_status_shared =
      (int*)&running_cost_shared[math::nearest_multiple_4(num_timesteps * blockDim.y * blockDim.z)];

#ifdef USE_CUDA_BARRIERS_ROLLOUT
  barrier* barrier_shared = (barrier*)&crash_status_shared[math::nearest_multiple_4(sample_dim * distribution_dim)];
#endif

  // Create local state, state dot and controls
  float* x = &x_shared[shared_idx * DYN_T::STATE_DIM];
  float* x_next = &x_next_shared[shared_idx * DYN_T::STATE_DIM];
  float* x_temp;
  float* xdot = &x_dot_shared[shared_idx * DYN_T::STATE_DIM];
  float* u = &u_shared[shared_idx * DYN_T::CONTROL_DIM];
  float* y = &y_shared[shared_idx * DYN_T::OUTPUT_DIM];
  float* running_cost = &running_cost_shared[blockDim.x * (thread_idz * blockDim.y + thread_idy)];
  int* crash_status = &crash_status_shared[shared_idx];
  crash_status[0] = 0;  // We have not crashed yet as of the first trajectory.
  int cost_index;
#ifdef USE_CUDA_BARRIERS_ROLLOUT
  barrier* bar = &barrier_shared[shared_idx];
  if (thread_idy == 0)
  {
    init(bar, blockDim.y);
  }
#endif

  // Load global array to shared array
  loadGlobalToShared<DYN_T::STATE_DIM, DYN_T::CONTROL_DIM>(num_rollouts, blockDim.y, global_idx, thread_idy, thread_idz,
                                                           init_x_d, x, xdot, u);
  __syncthreads();

  /*<----Start of simulation loop-----> */
  dynamics->initializeDynamics(x, u, y, theta_s_shared, 0.0f, dt);
  sampling->initializeDistributions(y, 0.0f, dt, theta_d_shared);
  costs->initializeCosts(y, u, theta_c_shared, 0.0f, dt);
  __syncthreads();
  for (int t = 0; t < num_timesteps; t++)
  {
    // Load noise trajectories scaled by the exploration factor
    sampling->readVisControlSample(global_idx, t, distribution_idx, u, theta_d_shared, blockDim.y, thread_idy, y);
#ifdef USE_CUDA_BARRIERS_ROLLOUT
    bar->arrive_and_wait();
#else
    __syncthreads();
#endif

    // applies constraints as defined in dynamics.cuh see specific dynamics class for what happens here
    // usually just control clamping
    dynamics->enforceConstraints(x, u);
#ifdef USE_CUDA_BARRIERS_ROLLOUT
    bar->arrive_and_wait();
#else
    __syncthreads();
#endif

    // Increment states
    dynamics->step(x, x_next, xdot, u, y, theta_s_shared, t, dt);
#ifdef USE_CUDA_BARRIERS_ROLLOUT
    bar->arrive_and_wait();
#else
    __syncthreads();
#endif
    if (t > 0)
    {
      float cost =
          costs->computeRunningCost(y, u, t, theta_c_shared, crash_status) +
          sampling->computeLikelihoodRatioCost(u, theta_d_shared, global_idx, t, distribution_idx, lambda, alpha);
      running_cost[t - 1] = cost / (num_timesteps);
      crash_status_d[global_idx * num_timesteps + t] = crash_status[0];
    }
#ifdef USE_CUDA_BARRIERS_ROLLOUT
    bar->arrive_and_wait();
#else
    __syncthreads();
#endif
    x_temp = x;
    x = x_next;
    x_next = x_temp;
    // Copy state to global memory
    int sample_time_offset = (num_rollouts * thread_idz + global_idx) * num_timesteps + t;
    mp1::loadArrayParallel<DYN_T::OUTPUT_DIM>(y_d, sample_time_offset * DYN_T::OUTPUT_DIM, y, 0);
  }

  // Add all thread_y components of cost together
  running_cost = &running_cost_shared[thread_idx + blockDim.x * blockDim.y * thread_idz];
  __syncthreads();
  costArrayReduction(running_cost, blockDim.y, thread_idy, blockDim.y, thread_idy == blockDim.y - 1, blockDim.x);
  // Compute terminal cost for each thread
  if (threadIdx.x == 0 && threadIdx.y == 0)
  {
    cost_index = (threadIdx.z * num_rollouts + global_idx) * (num_timesteps + 1) + num_timesteps;
    cost_traj_d[cost_index] = costs->terminalCost(y, theta_c_shared) / (num_timesteps);
  }
  __syncthreads();
  // Copy to global memory
  int parallel_index, step;
  mp1::getParallel1DIndex<mp1::Parallel1Dir::THREAD_X>(parallel_index, step);
  if (num_timesteps % 4 == 0)
  {
    float4* cost_traj_d4 =
        reinterpret_cast<float4*>(&cost_traj_d[(thread_idz * num_rollouts + global_idx) * num_timesteps]);
    float4* running_cost_shared4 =
        reinterpret_cast<float4*>(&running_cost_shared[thread_idz * num_timesteps * blockDim.y]);
    for (int i = parallel_index; i < num_timesteps / 4; i += step)
    {
      cost_traj_d4[i] = running_cost_shared4[i];
    }
  }
  else if (num_timesteps % 2 == 0)
  {
    float2* cost_traj_d2 =
        reinterpret_cast<float2*>(&cost_traj_d[(thread_idz * num_rollouts + global_idx) * num_timesteps]);
    float2* running_cost_shared2 =
        reinterpret_cast<float2*>(&running_cost_shared[thread_idz * num_timesteps * blockDim.y]);
    for (int i = parallel_index; i < num_timesteps / 2; i += step)
    {
      cost_traj_d2[i] = running_cost_shared2[i];
    }
  }
  else
  {
    for (int i = parallel_index; i < num_timesteps; i += step)
    {
      cost_traj_d[(thread_idz * num_rollouts + global_idx) * num_timesteps + i] =
          running_cost_shared[thread_idz * num_timesteps * blockDim.y + i];
    }
  }
}

template <class COST_T, class SAMPLING_T, bool COALESCE>
__global__ void visualizeCostKernel(COST_T* __restrict__ costs, SAMPLING_T* __restrict__ sampling, float dt,
                                    const int num_timesteps, const int num_rollouts, const float lambda, float alpha,
                                    const float* __restrict__ y_d, float* __restrict__ cost_traj_d,
                                    int* __restrict__ crash_status_d)
{
  // Get thread and block id
  const int thread_idx = threadIdx.x;
  const int thread_idy = threadIdx.y;
  const int thread_idz = threadIdx.z;
  const int global_idx = blockIdx.x;
  const int shared_idx = blockDim.x * thread_idz + thread_idx;
  const int distribution_idx = threadIdx.z;

  const int size_of_theta_c_bytes = calcClassSharedMemSize(costs, blockDim);

  // Create shared state and control arrays
  extern __shared__ float entire_buffer[];

  float* y_shared = entire_buffer;
  float* u_shared = &y_shared[math::nearest_multiple_4(blockDim.x * blockDim.z * COST_T::OUTPUT_DIM)];
  float* running_cost_shared = &u_shared[math::nearest_multiple_4(blockDim.x * blockDim.z * COST_T::CONTROL_DIM)];
  int* crash_status_shared =
      (int*)&running_cost_shared[math::nearest_multiple_4(blockDim.y * blockDim.z * num_timesteps)];
  float* theta_c = (float*)&crash_status_shared[math::nearest_multiple_4(blockDim.x * blockDim.z)];
  float* theta_d = &theta_c[size_of_theta_c_bytes / sizeof(float)];
#ifdef USE_CUDA_BARRIERS_COST
  const int size_of_theta_d_bytes = calcClassSharedMemSize(sampling, blockDim);
  barrier* barrier_shared = (barrier*)&theta_d[size_of_theta_d_bytes / sizeof(float)];
#endif
  // Create local state, state dot and controls
  float* y;
  float* u;
  int* crash_status;

  // Initialize running cost and total cost
  float* running_cost;
  int sample_time_offset = 0;
  int cost_index = 0;

  // Load global array to shared array
  y = &y_shared[shared_idx * COST_T::OUTPUT_DIM];
  u = &u_shared[shared_idx * COST_T::CONTROL_DIM];
  crash_status = &crash_status_shared[shared_idx];
  crash_status[0] = 0;  // We have not crashed yet as of the first trajectory.
#ifdef USE_CUDA_BARRIERS_COST
  barrier* bar = &barrier_shared[(blockDim.x * thread_idz + thread_idx)];
  if (thread_idy == 0)
  {
    init(bar, blockDim.y);
  }
#endif

  /*<----Start of simulation loop-----> */
#ifdef USE_COST_WITH_OFF_NUM_TIMESTEPS
  const int max_time_iters = ceilf((float)(num_timesteps - 2) / blockDim.x);
#else
  const int max_time_iters = ceilf((float)num_timesteps / blockDim.x);
#endif
  costs->initializeCosts(y, u, theta_c, 0.0f, dt);
  sampling->initializeDistributions(y, 0.0f, dt, theta_d);
  __syncthreads();
  for (int time_iter = 0; time_iter < max_time_iters; ++time_iter)
  {
    int t = thread_idx + time_iter * blockDim.x;
    cost_index = (thread_idz * num_rollouts + global_idx) * (num_timesteps) + t - 1;
    running_cost = &running_cost_shared[blockDim.x * (thread_idz * blockDim.y + thread_idy) + t - 1];
    if (COALESCE)
    {  // Fill entire shared mem sequentially using sequential threads_idx
      int amount_to_fill = (time_iter + 1) * blockDim.x > num_timesteps ? num_timesteps % blockDim.x : blockDim.x;
      mp1::loadArrayParallel<mp1::Parallel1Dir::THREAD_XY>(
          y_shared, blockDim.x * thread_idz * COST_T::OUTPUT_DIM, y_d,
          ((num_rollouts * thread_idz + global_idx) * num_timesteps + time_iter * blockDim.x) * COST_T::OUTPUT_DIM,
          COST_T::OUTPUT_DIM * amount_to_fill);
    }
    else if (t < num_timesteps)
    {  // t = num_timesteps is the terminal state for outside this for-loop
      sample_time_offset = (num_rollouts * thread_idz + global_idx) * num_timesteps + t;
      mp1::loadArrayParallel<COST_T::OUTPUT_DIM>(y, 0, y_d, sample_time_offset * COST_T::OUTPUT_DIM);
    }
    if (t < num_timesteps)
    {  // load controls from t = 0 to t = num_timesteps - 1
      sampling->readVisControlSample(global_idx, t, distribution_idx, u, theta_d, blockDim.y, thread_idy, y);
    }
#ifdef USE_CUDA_BARRIERS_COST
    bar->arrive_and_wait();
#else
    __syncthreads();
#endif

    // Compute cost
    if (t < num_timesteps)
    {
      float cost = costs->computeRunningCost(y, u, t, theta_c, crash_status) +
                   sampling->computeLikelihoodRatioCost(u, theta_d, global_idx, t, distribution_idx, lambda, alpha);
      running_cost[0] = cost / (num_timesteps);
      crash_status_d[global_idx * num_timesteps + t] = crash_status[0];
    }
#ifdef USE_CUDA_BARRIERS_COST
    bar->arrive_and_wait();
#else
    __syncthreads();
#endif
  }
  // consolidate y threads into single cost
  running_cost = &running_cost_shared[thread_idx + blockDim.x * blockDim.y * thread_idz];
  __syncthreads();
  costArrayReduction(running_cost, blockDim.y, thread_idy, blockDim.y, thread_idy == blockDim.y - 1, blockDim.x);
  // point every thread to the last output at t = NUM_TIMESTEPS for terminal cost calculation
  const int last_y_index = (num_timesteps - 1) % blockDim.x;
  y = &y_shared[(blockDim.x * thread_idz + last_y_index) * COST_T::OUTPUT_DIM];
#ifdef USE_COST_WITH_OFF_NUM_TIMESTEPS
  // load last output array
  const int t = num_timesteps - 1;
  mp1::loadArrayParallel<mp1::Parallel1Dir::THREAD_XY>(
      y_shared, (blockDim.x * thread_idz + last_y_index) * COST_T::OUTPUT_DIM, y_d,
      ((global_idx + num_rollouts * thread_idz) * num_timesteps + t) * COST_T::OUTPUT_DIM, COST_T::OUTPUT_DIM);
  __syncthreads();
#endif
  // Compute terminal cost for each thread
  if (threadIdx.x == 0 && threadIdx.y == 0)
  {
    cost_index = (threadIdx.z * num_rollouts + global_idx) * (num_timesteps + 1) + num_timesteps;
    cost_traj_d[cost_index] = costs->terminalCost(y, theta_c) / (num_timesteps);
  }
  __syncthreads();
  // Copy to global memory
  if (num_timesteps % 4 == 0)
  {
    float4* cost_traj_d4 =
        reinterpret_cast<float4*>(&cost_traj_d[(thread_idz * num_rollouts + global_idx) * num_timesteps]);
    float4* running_cost_shared4 =
        reinterpret_cast<float4*>(&running_cost_shared[thread_idz * num_timesteps * blockDim.y]);
    for (int i = thread_idx; i < num_timesteps / 4; i += blockDim.x)
    {
      cost_traj_d4[i] = running_cost_shared4[i];
    }
  }
  else if (num_timesteps % 2 == 0)
  {
    float2* cost_traj_d2 =
        reinterpret_cast<float2*>(&cost_traj_d[(thread_idz * num_rollouts + global_idx) * num_timesteps]);
    float2* running_cost_shared2 =
        reinterpret_cast<float2*>(&running_cost_shared[thread_idz * num_timesteps * blockDim.y]);
    for (int i = thread_idx; i < num_timesteps / 2; i += blockDim.x)
    {
      cost_traj_d2[i] = running_cost_shared2[i];
    }
  }
  else
  {
    for (int i = thread_idx; i < num_timesteps; i += blockDim.x)
    {
      cost_traj_d[(thread_idz * num_rollouts + global_idx) * num_timesteps + i] =
          running_cost_shared[thread_idz * num_timesteps * blockDim.y + i];
    }
  }
}

__global__ void normExpKernel(int num_rollouts, float* trajectory_costs_d, float lambda_inv, float baseline)
{
  int global_idx = (blockDim.x * blockIdx.x + threadIdx.x) * blockDim.z + threadIdx.z;
  int global_step = blockDim.x * gridDim.x * blockDim.z * gridDim.z;
  // #if defined(CUDA_VERSION) && CUDA_VERSION > 11060
  //   auto block = cg::this_grid();
  //   int global_idx_b = block.thread_rank() + block.block_rank() * block.num_threads();
  //   int global_step_b = block.num_threads() * block.num_blocks();
  //   if (global_idx == 200 && threadIdx.y == 0 && threadIdx.z == 0)
  //   {
  //     printf("Global ind: %d, thread_rank: %d\n", global_idx, global_idx_b);
  //     printf("Global step: %d, thread_rank: %d\n", global_step, global_step_b);
  //   }
  // #endif
  normExpTransform(num_rollouts * blockDim.z, trajectory_costs_d, lambda_inv, baseline, global_idx, global_step);
}

__global__ void TsallisKernel(int num_rollouts, float* trajectory_costs_d, float gamma, float r, float baseline)
{
  int global_idx = (blockDim.x * blockIdx.x + threadIdx.x) * blockDim.z + threadIdx.z;
  int global_step = blockDim.x * gridDim.x * blockDim.z * gridDim.z;
  TsallisTransform(num_rollouts * blockDim.z, trajectory_costs_d, gamma, r, baseline, global_idx, global_step);
}

template <int CONTROL_DIM>
__global__ void weightedReductionKernel(const float* __restrict__ exp_costs_d, const float* __restrict__ du_d,
                                        float* __restrict__ new_u_d, const float normalizer, const int num_timesteps,
                                        const int num_rollouts, const int sum_stride)
{
  int thread_idx = threadIdx.x;  // Rollout index
  int block_idx = blockIdx.x;    // Timestep

  // Create a shared array for intermediate sums: CONTROL_DIM x NUM_THREADS
  extern __shared__ float u_intermediate[];

  float u[CONTROL_DIM];
  setInitialControlToZero(CONTROL_DIM, thread_idx, u, u_intermediate);

  __syncthreads();

  // Sum the weighted control variations at a desired stride
  strideControlWeightReduction(num_rollouts, num_timesteps, sum_stride, thread_idx, block_idx, CONTROL_DIM, exp_costs_d,
                               normalizer, du_d, u, u_intermediate);

  __syncthreads();

  // Sum all weighted control variations
  rolloutWeightReductionAndSaveControl(thread_idx, block_idx, num_rollouts, num_timesteps, CONTROL_DIM, sum_stride, u,
                                       u_intermediate, new_u_d);

  __syncthreads();
}

template <int CONTROL_DIM, int NUM_ROLLOUTS, int SUM_STRIDE>
__global__ void weightedReductionKernel(float* exp_costs_d, float* du_d, float* du_new_d,
                                        float2* baseline_and_normalizer_d, int num_timesteps)
{
  int thread_idx = threadIdx.x;  // Rollout index
  int block_idx = blockIdx.x;    // Timestep

  // Create a shared array for intermediate sums: CONTROL_DIM x NUM_THREADS
  __shared__ float u_intermediate[CONTROL_DIM * ((NUM_ROLLOUTS - 1) / SUM_STRIDE + 1)];

  float u[CONTROL_DIM];
  setInitialControlToZero(CONTROL_DIM, thread_idx, u, u_intermediate);

  __syncthreads();

  // Sum the weighted control variations at a desired stride
  strideControlWeightReduction(NUM_ROLLOUTS, num_timesteps, SUM_STRIDE, thread_idx, block_idx, CONTROL_DIM, exp_costs_d,
                               baseline_and_normalizer_d->y, du_d, u, u_intermediate);

  __syncthreads();

  // Sum all weighted control variations
  rolloutWeightReductionAndSaveControl(thread_idx, block_idx, NUM_ROLLOUTS, num_timesteps, CONTROL_DIM, SUM_STRIDE, u,
                                       u_intermediate, du_new_d);

  __syncthreads();
}

/*******************************************************************************************************************
 * Rollout Kernel Helpers
 *******************************************************************************************************************/
template <int STATE_DIM, int CONTROL_DIM>
__device__ void loadGlobalToShared(const int num_rollouts, const int blocksize_y, const int global_idx,
                                   const int thread_idy, const int thread_idz, const float* __restrict__ x_device,
                                   float* __restrict__ x_thread, float* __restrict__ xdot_thread,
                                   float* __restrict__ u_thread)
{
  // Transfer to shared memory
  int i;
  if (global_idx < num_rollouts)
  {
#if true
    mp1::loadArrayParallel<STATE_DIM>(x_thread, 0, x_device, STATE_DIM * thread_idz);
    if (STATE_DIM % 4 == 0)
    {
      float4* xdot4_t = reinterpret_cast<float4*>(xdot_thread);
      for (i = thread_idy; i < STATE_DIM / 4; i += blocksize_y)
      {
        xdot4_t[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      }
    }
    else if (STATE_DIM % 2 == 0)
    {
      float2* xdot2_t = reinterpret_cast<float2*>(xdot_thread);
      for (i = thread_idy; i < STATE_DIM / 2; i += blocksize_y)
      {
        xdot2_t[i] = make_float2(0.0f, 0.0f);
      }
    }
    else
    {
      for (i = thread_idy; i < STATE_DIM; i += blocksize_y)
      {
        xdot_thread[i] = 0.0f;
      }
    }

    if (CONTROL_DIM % 4 == 0)
    {
      float4* u4_t = reinterpret_cast<float4*>(u_thread);
      for (i = thread_idy; i < CONTROL_DIM / 4; i += blocksize_y)
      {
        u4_t[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      }
    }
    else if (CONTROL_DIM % 2 == 0)
    {
      float2* u2_t = reinterpret_cast<float2*>(u_thread);
      for (i = thread_idy; i < CONTROL_DIM / 2; i += blocksize_y)
      {
        u2_t[i] = make_float2(0.0f, 0.0f);
      }
    }
    else
    {
      for (i = thread_idy; i < CONTROL_DIM; i += blocksize_y)
      {
        u_thread[i] = 0.0f;
      }
    }
#else
    for (i = thread_idy; i < STATE_DIM; i += blocksize_y)
    {
      x_thread[i] = x_device[i + STATE_DIM * thread_idz];
      xdot_thread[i] = 0.0f;
    }
    for (i = thread_idy; i < CONTROL_DIM; i += blocksize_y)
    {
      u_thread[i] = 0.0f;
    }
#endif
  }
}

template <class COST_T>
__device__ void computeAndSaveCost(int num_rollouts, int num_timesteps, int global_idx, COST_T* costs, float* output,
                                   float running_cost, float* theta_c, float* cost_rollouts_device)
{
  // only want to save 1 cost per trajectory
  if (threadIdx.y == 0 && global_idx < num_rollouts)
  {
    cost_rollouts_device[global_idx + num_rollouts * threadIdx.z] =
        running_cost + costs->terminalCost(output, theta_c) / (num_timesteps);
  }
}

/*******************************************************************************************************************
 * NormExp Kernel Helpers
 *******************************************************************************************************************/
float computeBaselineCost(float* cost_rollouts_host, int num_rollouts)
{  // TODO if we use standard containers in MPPI, should this be replaced with a min algorithm?
  int best_idx = computeBestIndex(cost_rollouts_host, num_rollouts);
  return cost_rollouts_host[best_idx];
}

float constructBestWeights(float* cost_rollouts_host, int num_rollouts)
{
  int best_idx = computeBestIndex(cost_rollouts_host, num_rollouts);
  float best_cost = cost_rollouts_host[best_idx];

  for (int i = 0; i < num_rollouts; i++)
  {
    if (i == best_idx)
    {
      cost_rollouts_host[i] = 1.0;
    }
    else
    {
      cost_rollouts_host[i] = 0.0;
    }
  }

  // printf("Best idx: %d, cost: %f\n", best_cost_idx, best_cost);
  return best_cost;
}

int computeBestIndex(float* cost_rollouts_host, int num_rollouts)
{
  float best_cost = cost_rollouts_host[0];
  int best_cost_idx = 0;
  for (int i = 1; i < num_rollouts; i++)
  {
    if (cost_rollouts_host[i] < best_cost)
    {
      best_cost = cost_rollouts_host[i];
      best_cost_idx = i;
    }
  }

  // printf("Best idx: %d, cost: %f\n", best_cost_idx, best_cost);
  return best_cost_idx;
}

__device__ inline float computeBaselineCost(int num_rollouts, const float* __restrict__ trajectory_costs_d,
                                            float* __restrict__ reduction_buffer, int rollout_idx_global,
                                            int rollout_idx_step)
{
  // Copy costs to shared memory
  float min_cost = 0.0;
#if false
  // potential method to speed up copying costs
  int prev_size = min(blockDim.x, num_rollouts);
  float my_val = (rollout_idx_global < num_rollouts) ? trajectory_costs_d[rollout_idx_global] : INFINITY;
  for (int i = rollout_idx_global + rollout_idx_step; i < num_rollouts; i += rollout_idx_step)
  {
    my_val = min(trajectory_costs_d[i], my_val);
  }
  reduction_buffer[rollout_idx_global] = my_val;
  // __syncthreads();
  // if (threadIdx.x == 0)
  // {
  //   for (int i = 0; i < min(blockDim.x, num_rollouts); i++)
  //   {
  //     printf("buff %d: %f\n", i, reduction_buffer[i]);
  //   }
  //   printf("Num rollouts; %d\n", num_rollouts);
  // }
#else
  int prev_size = num_rollouts / 2;
  for (int i = rollout_idx_global; i < prev_size; i += rollout_idx_step)
  {
    reduction_buffer[i] = min(trajectory_costs_d[i], trajectory_costs_d[i + prev_size]);
  }
  if (num_rollouts - 2 * prev_size == 1 && threadIdx.x == blockDim.x - 1)
  {
    reduction_buffer[prev_size - 1] = min(reduction_buffer[num_rollouts - 1], reduction_buffer[prev_size - 1]);
  }
#endif

  __syncthreads();
  // find min along the entire array
  for (int size = prev_size / 2; size > 0; size /= 2)
  {
    for (int i = rollout_idx_global; i < size; i += rollout_idx_step)
    {
      reduction_buffer[i] = min(reduction_buffer[i], reduction_buffer[i + size]);
    }
    __syncthreads();
    if (prev_size - 2 * size == 1 && threadIdx.x == blockDim.x - 1)
    {
      reduction_buffer[size - 1] = min(reduction_buffer[size - 1], reduction_buffer[prev_size - 1]);
    }
    __syncthreads();
    prev_size = size;
  }
  min_cost = reduction_buffer[0];
  return min_cost;
}

__device__ __host__ inline void normExpTransform(int num_rollouts, float* __restrict__ trajectory_costs_d,
                                                 float lambda_inv, float baseline, int global_idx, int rollout_idx_step)
{
  for (int i = global_idx; i < num_rollouts; i += rollout_idx_step)
  {
    float cost_dif = trajectory_costs_d[i] - baseline;
    trajectory_costs_d[i] = expf(-lambda_inv * cost_dif);
  }
}

__device__ __host__ inline void TsallisTransform(int num_rollouts, float* __restrict__ trajectory_costs_d, float gamma,
                                                 float r, float baseline, int global_idx, int rollout_idx_step)
{
  for (int i = global_idx; i < num_rollouts; i += rollout_idx_step)
  {
    float cost_dif = trajectory_costs_d[i] - baseline;
    // trajectory_costs_d[i] = mppi::math::expr(-lambda_bar_inv * cost_dif);
    // trajectory_costs_d[i] = (cost_dif < gamma) * expf(logf(1.0 - cost_dif / gamma) / (r - 1));
    if (cost_dif < gamma)
    {
      trajectory_costs_d[i] = expf(logf(1.0 - cost_dif / gamma) / (r - 1));
    }
    else
    {
      trajectory_costs_d[i] = 0;
    }
  }
}

__device__ inline float computeNormalizer(int num_rollouts, const float* __restrict__ trajectory_costs_d,
                                          float* __restrict__ reduction_buffer, int rollout_idx_global,
                                          int rollout_idx_step)
{
  // Copy costs to shared memory
#if false
  // potential method to speed up copying costs
  int prev_size = min(blockDim.x, num_rollouts);
  float my_val = (rollout_idx_global < num_rollouts) ? trajectory_costs_d[rollout_idx_global] : 0;
  for (int i = rollout_idx_global + rollout_idx_step; i < num_rollouts; i += rollout_idx_step)
  {
    my_val += trajectory_costs_d[i];
  }
  reduction_buffer[rollout_idx_global] = my_val;
#else
  int prev_size = num_rollouts / 2;
  for (int i = rollout_idx_global; i < prev_size; i += rollout_idx_step)
  {
    reduction_buffer[i] = trajectory_costs_d[i] + trajectory_costs_d[i + prev_size];
  }
  if (num_rollouts - 2 * prev_size == 1 && threadIdx.x == blockDim.x - 1)
  {
    reduction_buffer[prev_size - 1] += reduction_buffer[num_rollouts - 1];
  }
#endif
  __syncthreads();
  // sum the entire array
  for (int size = prev_size / 2; size > 0; size /= 2)
  {
    for (int i = rollout_idx_global; i < size; i += rollout_idx_step)
    {
      reduction_buffer[i] += reduction_buffer[i + size];
    }
    __syncthreads();
    if (prev_size - 2 * size == 1 && threadIdx.x == blockDim.x - 1)
    {
      reduction_buffer[size - 1] += reduction_buffer[prev_size - 1];
    }
    __syncthreads();
    prev_size = size;
  }
  return reduction_buffer[0];
}

template <int NUM_ROLLOUTS, int BLOCKSIZE_X = 1024>
__global__ void fullGPUcomputeWeights(float* __restrict__ trajectory_costs_d, float lambda_inv,
                                      float2* __restrict__ output)
{
  __shared__ float reduction_buffer[NUM_ROLLOUTS];
  // int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
  // int better_global_idx = (blockIdx.x * blockDim.x + threadIdx.x) * blockDim.y + threadIdx.y;
  // int global_idx = (blockDim.x * blockIdx.x + threadIdx.x) * blockDim.z + threadIdx.z;
  // int global_step = blockDim.x * gridDim.x;
  // int better_global_step = blockDim.x * gridDim.x  * blockDim.y * gridDim.y;
  int global_idx = threadIdx.x;
  int global_step = blockDim.x;

  float baseline = computeBaselineCost(NUM_ROLLOUTS, trajectory_costs_d, reduction_buffer, global_idx, global_step);
  normExpTransform(NUM_ROLLOUTS, trajectory_costs_d, lambda_inv, baseline, global_idx, global_step);
  __syncthreads();
  float normalizer = computeNormalizer(NUM_ROLLOUTS, trajectory_costs_d, reduction_buffer, global_idx, global_step);
  __syncthreads();
  if (threadIdx.x == 0)
  {
    *output = make_float2(baseline, normalizer);
  }
}

float computeNormalizer(float* cost_rollouts_host, int num_rollouts)
{
  double normalizer = 0.0;
  for (int i = 0; i < num_rollouts; ++i)
  {
    normalizer += cost_rollouts_host[i];
  }
  return normalizer;
}

void computeFreeEnergy(float& free_energy, float& free_energy_var, float& free_energy_modified,
                       float* cost_rollouts_host, int num_rollouts, float baseline, float lambda)
{
  float var = 0;
  float norm = 0;
  for (int i = 0; i < num_rollouts; i++)
  {
    norm += cost_rollouts_host[i];
    var += SQ(cost_rollouts_host[i]);
  }
  norm /= num_rollouts;
  free_energy = -lambda * logf(norm) + baseline;
  free_energy_var = lambda * (var / num_rollouts - SQ(norm));
  // TODO Figure out the point of the following lines
  float weird_term = free_energy_var / (norm * sqrtf(1.0 * num_rollouts));
  free_energy_modified = lambda * (weird_term + 0.5 * SQ(weird_term));
}

/*******************************************************************************************************************
 * Weighted Reduction Kernel Helpers
 *******************************************************************************************************************/
__device__ void setInitialControlToZero(int control_dim, int thread_idx, float* __restrict__ u,
                                        float* __restrict__ u_intermediate)
{
  if (control_dim % 4 == 0)
  {
    for (int i = 0; i < control_dim / 4; i++)
    {
      reinterpret_cast<float4*>(u)[i] = make_float4(0, 0, 0, 0);
      reinterpret_cast<float4*>(&u_intermediate[thread_idx * control_dim])[i] = make_float4(0, 0, 0, 0);
    }
  }
  else if (control_dim % 2 == 0)
  {
    for (int i = 0; i < control_dim / 2; i++)
    {
      reinterpret_cast<float2*>(u)[i] = make_float2(0, 0);
      reinterpret_cast<float2*>(&u_intermediate[thread_idx * control_dim])[i] = make_float2(0, 0);
    }
  }
  else
  {
    for (int i = 0; i < control_dim; i++)
    {
      u[i] = 0;
      u_intermediate[thread_idx * control_dim + i] = 0;
    }
  }
}

__device__ void strideControlWeightReduction(const int num_rollouts, const int num_timesteps, const int sum_stride,
                                             const int thread_idx, const int block_idx, const int control_dim,
                                             const float* __restrict__ exp_costs_d, const float normalizer,
                                             const float* __restrict__ du_d, float* __restrict__ u,
                                             float* __restrict__ u_intermediate)
{
  // int index = thread_idx * sum_stride + i;
  for (int i = 0; i < sum_stride; ++i)
  {  // Iterate through the size of the subsection
    if ((thread_idx * sum_stride + i) < num_rollouts)
    {                                                                        // Ensure we do not go out of bounds
      float weight = exp_costs_d[thread_idx * sum_stride + i] / normalizer;  // compute the importance sampling weight
      for (int j = 0; j < control_dim; ++j)
      {  // Iterate through the control dimensions
        // Rollout index: (thread_idx*sum_stride + i)*(num_timesteps*control_dim)
        // Current timestep: block_idx*control_dim
        u[j] = du_d[(thread_idx * sum_stride + i) * (num_timesteps * control_dim) + block_idx * control_dim + j];
        u_intermediate[thread_idx * control_dim + j] += weight * u[j];
      }
    }
  }
}

__device__ void rolloutWeightReductionAndSaveControl(int thread_idx, int block_idx, int num_rollouts, int num_timesteps,
                                                     int control_dim, int sum_stride, float* u, float* u_intermediate,
                                                     float* du_new_d)
{
  if (thread_idx == 0 && block_idx < num_timesteps)
  {  // block index refers to the current timestep
    for (int i = 0; i < control_dim; ++i)
    {  // TODO replace with memset?
      u[i] = 0;
    }
    for (int i = 0; i < ((num_rollouts - 1) / sum_stride + 1); ++i)
    {  // iterate through the each subsection
      for (int j = 0; j < control_dim; ++j)
      {
        u[j] += u_intermediate[i * control_dim + j];
      }
    }
    for (int i = 0; i < control_dim; i++)
    {
      du_new_d[block_idx * control_dim + i] = u[i];
    }
  }
}

template <int BLOCKSIZE>
__device__ void warpReduceAdd(volatile float* sdata, const int tid, const int stride)
{
  if (BLOCKSIZE >= 64)
  {
    sdata[tid * stride] += sdata[(tid + 32) * stride];
  }
  if (BLOCKSIZE >= 32)
  {
    sdata[tid * stride] += sdata[(tid + 16) * stride];
  }
  if (BLOCKSIZE >= 16)
  {
    sdata[tid * stride] += sdata[(tid + 8) * stride];
  }
  if (BLOCKSIZE >= 8)
  {
    sdata[tid * stride] += sdata[(tid + 4) * stride];
  }
  if (BLOCKSIZE >= 4)
  {
    sdata[tid * stride] += sdata[(tid + 2) * stride];
  }
  if (BLOCKSIZE >= 2)
  {
    sdata[tid * stride] += sdata[(tid + 1) * stride];
  }
}

__device__ void costArrayReduction(float* running_cost, const int start_size, const int index, const int step,
                                   const bool catch_condition, const int stride)
{
  int prev_size = start_size;
  const bool block_power_of_2 = (prev_size & (prev_size - 1)) == 0;
  const int stop_condition = (block_power_of_2) ? 32 : 0;
  int size;
  int j;

  for (size = prev_size / 2; size > stop_condition; size /= 2)
  {
    for (j = index; j < size; j += step)
    {
      running_cost[j * stride] += running_cost[(j + size) * stride];
    }
    __syncthreads();
    if (prev_size - 2 * size == 1 && catch_condition)
    {
      running_cost[(size - 1) * stride] += running_cost[(prev_size - 1) * stride];
    }
    __syncthreads();
    prev_size = size;
  }
  switch (size * 2)
  {
    case 64:
      if (index < 32)
      {
        warpReduceAdd<64>(running_cost, index, stride);
      }
      break;
    case 32:
      if (index < 16)
      {
        warpReduceAdd<32>(running_cost, index, stride);
      }
      break;
    case 16:
      if (index < 8)
      {
        warpReduceAdd<16>(running_cost, index, stride);
      }
      break;
    case 8:
      if (index < 4)
      {
        warpReduceAdd<8>(running_cost, index, stride);
      }
      break;
    case 4:
      if (index < 2)
      {
        warpReduceAdd<4>(running_cost, index, stride);
      }
      break;
    case 2:
      if (index < 1)
      {
        warpReduceAdd<2>(running_cost, index, stride);
      }
      break;
  }
  __syncthreads();
}

/*******************************************************************************************************************
 * Launch Functions
 *******************************************************************************************************************/
template <class DYN_T, class COST_T, typename SAMPLING_T, bool COALESCE>
void launchSplitRolloutKernel(DYN_T* __restrict__ dynamics, COST_T* __restrict__ costs,
                              SAMPLING_T* __restrict__ sampling, float dt, const int num_timesteps,
                              const int num_rollouts, float lambda, float alpha, float* __restrict__ init_x_d,
                              float* __restrict__ y_d, float* __restrict__ trajectory_costs, dim3 dimDynBlock,
                              dim3 dimCostBlock, cudaStream_t stream, bool synchronize)
{
  if (num_rollouts % dimDynBlock.x != 0)
  {
    std::cerr << __FILE__ << " (" << __LINE__ << "): num_rollouts (" << num_rollouts
              << ") must be evenly divided by dynamics block size x (" << dimDynBlock.x << ")" << std::endl;
    exit(EXIT_FAILURE);
  }
  if (num_timesteps < dimCostBlock.x)
  {
    std::cerr << __FILE__ << " (" << __LINE__ << "): num_timesteps (" << num_timesteps
              << ") must be greater than or equal to cost block size x (" << dimCostBlock.x << ")" << std::endl;
    exit(EXIT_FAILURE);
  }
  // Run Dynamics
  const int gridsize_x = math::int_ceil(num_rollouts, dimDynBlock.x);
  dim3 dimGrid(gridsize_x, 1, 1);
  unsigned dynamics_shared_size = calcRolloutDynamicsKernelSharedMemSize(dynamics, sampling, dimDynBlock);
  HANDLE_ERROR(cudaFuncSetAttribute(rolloutDynamicsKernel<DYN_T, SAMPLING_T>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, dynamics_shared_size));
  rolloutDynamicsKernel<DYN_T, SAMPLING_T><<<dimGrid, dimDynBlock, dynamics_shared_size, stream>>>(
      dynamics->model_d_, sampling->sampling_d_, dt, num_timesteps, num_rollouts, init_x_d, y_d);
  HANDLE_ERROR(cudaGetLastError());
  // Run Costs
  dim3 dimCostGrid(num_rollouts, 1, 1);
  unsigned cost_shared_size = calcRolloutCostKernelSharedMemSize(costs, sampling, dimCostBlock);
  rolloutCostKernel<COST_T, SAMPLING_T, COALESCE><<<dimCostGrid, dimCostBlock, cost_shared_size, stream>>>(
      costs->cost_d_, sampling->sampling_d_, dt, num_timesteps, num_rollouts, lambda, alpha, y_d, trajectory_costs);
  HANDLE_ERROR(cudaGetLastError());
  if (synchronize)
  {
    HANDLE_ERROR(cudaStreamSynchronize(stream));
  }
}

template <class DYN_T, class COST_T, typename SAMPLING_T>
void launchRolloutKernel(DYN_T* __restrict__ dynamics, COST_T* __restrict__ costs, SAMPLING_T* __restrict__ sampling,
                         float dt, const int num_timesteps, const int num_rollouts, float lambda, float alpha,
                         float* __restrict__ init_x_d, float* __restrict__ trajectory_costs, dim3 dimBlock,
                         cudaStream_t stream, bool synchronize)
{
  if (num_rollouts % dimBlock.x != 0)
  {
    std::cerr << __FILE__ << " (" << __LINE__ << "): num_rollouts (" << num_rollouts
              << ") must be evenly divided by rollout thread block size x (" << dimBlock.x << ")" << std::endl;
    exit(EXIT_FAILURE);
  }

  const int gridsize_x = math::int_ceil(num_rollouts, dimBlock.x);
  dim3 dimGrid(gridsize_x, 1, 1);
  unsigned shared_mem_size = calcRolloutCombinedKernelSharedMemSize(dynamics, costs, sampling, dimBlock);
  HANDLE_ERROR(cudaFuncSetAttribute(rolloutKernel<DYN_T, COST_T, SAMPLING_T>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size));
  rolloutKernel<DYN_T, COST_T, SAMPLING_T><<<dimGrid, dimBlock, shared_mem_size, stream>>>(
      dynamics->model_d_, sampling->sampling_d_, costs->cost_d_, dt, num_timesteps, num_rollouts, init_x_d, lambda,
      alpha, trajectory_costs);
  HANDLE_ERROR(cudaGetLastError());
  if (synchronize)
  {
    HANDLE_ERROR(cudaStreamSynchronize(stream));
  }
}

template <class DYN_T, class COST_T, typename SAMPLING_T>
void launchVisualizeKernel(DYN_T* __restrict__ dynamics, COST_T* __restrict__ costs, SAMPLING_T* __restrict__ sampling,
                           float dt, const int num_timesteps, const int num_rollouts, float lambda, float alpha,
                           float* __restrict__ init_x_d, float* __restrict__ y_d, float* __restrict__ trajectory_costs,
                           int* __restrict__ crash_status_d, dim3 dimVisBlock, cudaStream_t stream, bool synchronize)
{
  if (num_rollouts <= 1)
  {  // Not enough samples to visualize
    std::cerr << "Not enough samples to visualize" << std::endl;
    return;
  }
  if (num_rollouts % dimVisBlock.x != 0)
  {
    std::cerr << __FILE__ << " (" << __LINE__ << "): num_rollouts (" << num_rollouts
              << ") must be evenly divided by vis block size x (" << dimVisBlock.x << ")" << std::endl;
    return;
  }

  const int gridsize_x = math::int_ceil(num_rollouts, dimVisBlock.x);
  dim3 dimGrid(gridsize_x, 1, 1);
  unsigned shared_mem_size = calcVisualizeKernelSharedMemSize(dynamics, costs, sampling, num_timesteps, dimVisBlock);
  HANDLE_ERROR(cudaFuncSetAttribute(rolloutKernel<DYN_T, COST_T, SAMPLING_T>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size));
  visualizeKernel<DYN_T, COST_T, SAMPLING_T><<<dimGrid, dimVisBlock, shared_mem_size, stream>>>(
      dynamics->model_d_, sampling->sampling_d_, costs->cost_d_, dt, num_timesteps, num_rollouts, init_x_d, lambda,
      alpha, y_d, trajectory_costs, crash_status_d);
  HANDLE_ERROR(cudaGetLastError());
  if (synchronize)
  {
    HANDLE_ERROR(cudaStreamSynchronize(stream));
  }
}

template <class COST_T, class SAMPLING_T, bool COALESCE>
void launchVisualizeCostKernel(COST_T* __restrict__ costs, SAMPLING_T* __restrict__ sampling, float dt,
                               const int num_timesteps, const int num_rollouts, float lambda, float alpha,
                               float* __restrict__ y_d, int* __restrict__ sampled_crash_status_d,
                               float* __restrict__ cost_traj_result, dim3 dimBlock, cudaStream_t stream,
                               bool synchronize)
{
  if (num_rollouts <= 1)
  {  // Not enough samples to visualize
    std::cerr << "Not enough samples to visualize" << std::endl;
    return;
  }

  dim3 dimCostGrid(num_rollouts, 1, 1);
  unsigned shared_mem_size = calcVisCostKernelSharedMemSize(costs, sampling, num_timesteps, dimBlock);
  visualizeCostKernel<COST_T, SAMPLING_T, COALESCE><<<dimCostGrid, dimBlock, shared_mem_size, stream>>>(
      costs->cost_d_, sampling->sampling_d_, dt, num_timesteps, num_rollouts, lambda, alpha, y_d, cost_traj_result,
      sampled_crash_status_d);
  HANDLE_ERROR(cudaGetLastError());
  if (synchronize)
  {
    HANDLE_ERROR(cudaStreamSynchronize(stream));
  }
}

template <int CONTROL_DIM>
void launchWeightedReductionKernel(const float* __restrict__ exp_costs_d, const float* __restrict__ du_d,
                                   float* __restrict__ new_u_d, const float normalizer, const int num_timesteps,
                                   const int num_rollouts, const int sum_stride, cudaStream_t stream, bool synchronize)
{
  dim3 dimBlock(math::int_ceil(num_rollouts, sum_stride), 1, 1);
  dim3 dimGrid(num_timesteps, 1, 1);
  unsigned shared_mem_size = math::nearest_multiple_4(CONTROL_DIM * dimBlock.x) * sizeof(float);
  weightedReductionKernel<CONTROL_DIM><<<dimGrid, dimBlock, shared_mem_size, stream>>>(
      exp_costs_d, du_d, new_u_d, normalizer, num_timesteps, num_rollouts, sum_stride);
  HANDLE_ERROR(cudaGetLastError());
  if (synchronize)
  {
    HANDLE_ERROR(cudaStreamSynchronize(stream));
  }
}

void launchNormExpKernel(int num_rollouts, int blocksize_x, float* trajectory_costs_d, float lambda_inv, float baseline,
                         cudaStream_t stream, bool synchronize)
{
  dim3 dimBlock(blocksize_x, 1, 1);
  dim3 dimGrid((num_rollouts - 1) / blocksize_x + 1, 1, 1);
  normExpKernel<<<dimGrid, dimBlock, 0, stream>>>(num_rollouts, trajectory_costs_d, lambda_inv, baseline);
  HANDLE_ERROR(cudaGetLastError());
  if (synchronize)
  {
    HANDLE_ERROR(cudaStreamSynchronize(stream));
  }
}

void launchTsallisKernel(int num_rollouts, int blocksize_x, float* trajectory_costs_d, float gamma, float r,
                         float baseline, cudaStream_t stream, bool synchronize)
{
  dim3 dimBlock(blocksize_x, 1, 1);
  dim3 dimGrid((num_rollouts - 1) / blocksize_x + 1, 1, 1);
  TsallisKernel<<<dimGrid, dimBlock, 0, stream>>>(num_rollouts, trajectory_costs_d, gamma, r, baseline);
  // CudaCheckError();
  HANDLE_ERROR(cudaGetLastError());
  if (synchronize)
  {
    HANDLE_ERROR(cudaStreamSynchronize(stream));
  }
}

template <int NUM_ROLLOUTS>
void launchWeightTransformKernel(float* __restrict__ costs_d, float2* __restrict__ baseline_and_norm_d,
                                 const float lambda_inv, const int num_systems, cudaStream_t stream, bool synchronize)
{
  // Figure out max size of threads from the device properties (slows down this method a lot)
  // int device_id = 0;
  // cudaDeviceProp deviceProp;
  // cudaGetDeviceProperties(&deviceProp, device_id);
  // int blocksize_x = deviceProp.maxThreadsDim[0];
  const int blocksize_x = 1024;
  dim3 dimBlock(blocksize_x, 1, 1);
  // Can't be split into multiple blocks because we want to do all the math in shared memory
  dim3 dimGrid(1, 1, 1);
  for (int i = 0; i < num_systems; i++)
  {
    fullGPUcomputeWeights<NUM_ROLLOUTS>
        <<<dimGrid, dimBlock, 0, stream>>>(costs_d + i * NUM_ROLLOUTS, lambda_inv, baseline_and_norm_d + i);
    HANDLE_ERROR(cudaGetLastError());
  }
  if (synchronize)
  {
    HANDLE_ERROR(cudaStreamSynchronize(stream));
  }
}

template <class DYN_T, int NUM_ROLLOUTS, int SUM_STRIDE>
void launchweightedReductionKernel(float* exp_costs_d, float* du_d, float* du_new_d, float2* baseline_and_normalizer_d,
                                   int num_timesteps, cudaStream_t stream, bool synchronize)
{
  dim3 dimBlock((NUM_ROLLOUTS - 1) / SUM_STRIDE + 1, 1, 1);
  dim3 dimGrid(num_timesteps, 1, 1);
  weightedReductionKernel<DYN_T::CONTROL_DIM, NUM_ROLLOUTS, SUM_STRIDE>
      <<<dimGrid, dimBlock, 0, stream>>>(exp_costs_d, du_d, du_new_d, baseline_and_normalizer_d, num_timesteps);
  // CudaCheckError();
  HANDLE_ERROR(cudaGetLastError());
  if (synchronize)
  {
    HANDLE_ERROR(cudaStreamSynchronize(stream));
  }
}

/*******************************************************************************************************************
 * Shared Memory Calculation Functions
 *******************************************************************************************************************/
template <class DYN_T, class SAMPLER_T>
unsigned calcRolloutDynamicsKernelSharedMemSize(const DYN_T* dynamics, const SAMPLER_T* sampler, dim3& dimBlock)
{
  const int dynamics_num_shared = dimBlock.x * dimBlock.z;
  unsigned dynamics_shared_size =
      sizeof(float) * (3 * math::nearest_multiple_4(dynamics_num_shared * DYN_T::STATE_DIM) +
                       math::nearest_multiple_4(dynamics_num_shared * DYN_T::OUTPUT_DIM) +
                       math::nearest_multiple_4(dynamics_num_shared * DYN_T::CONTROL_DIM)) +
      calcClassSharedMemSize<DYN_T>(dynamics, dimBlock) + calcClassSharedMemSize<SAMPLER_T>(sampler, dimBlock);
#ifdef USE_CUDA_BARRIERS_DYN
  dynamics_shared_size += math::int_multiple_const(dynamics_num_shared * sizeof(barrier), 16);
#endif
  return dynamics_shared_size;
}

template <class COST_T, class SAMPLER_T>
unsigned calcRolloutCostKernelSharedMemSize(const COST_T* cost, const SAMPLER_T* sampler, dim3& dimBlock)
{
  const int cost_num_shared = dimBlock.x * dimBlock.z;
  unsigned cost_shared_size = sizeof(float) * (math::nearest_multiple_4(cost_num_shared * COST_T::OUTPUT_DIM) +
                                               math::nearest_multiple_4(cost_num_shared * COST_T::CONTROL_DIM) +
                                               math::nearest_multiple_4(cost_num_shared * dimBlock.y)) +
                              sizeof(int) * math::nearest_multiple_4(cost_num_shared) +
                              calcClassSharedMemSize(cost, dimBlock) +
                              calcClassSharedMemSize<SAMPLER_T>(sampler, dimBlock);
#ifdef USE_CUDA_BARRIERS_COST
  cost_shared_size += math::int_multiple_const(cost_num_shared * sizeof(barrier), 16);
#endif
  return cost_shared_size;
}

template <class DYN_T, class COST_T, class SAMPLER_T>
unsigned calcRolloutCombinedKernelSharedMemSize(const DYN_T* dynamics, const COST_T* cost, const SAMPLER_T* sampler,
                                                dim3& dimBlock)
{
  const int num_shared = dimBlock.x * dimBlock.z;
  unsigned shared_mem_size = sizeof(float) * (3 * math::nearest_multiple_4(num_shared * DYN_T::STATE_DIM) +
                                              math::nearest_multiple_4(num_shared * DYN_T::OUTPUT_DIM) +
                                              math::nearest_multiple_4(num_shared * DYN_T::CONTROL_DIM) +
                                              math::nearest_multiple_4(num_shared * dimBlock.y)) +
                             sizeof(int) * math::nearest_multiple_4(num_shared) +
                             calcClassSharedMemSize(dynamics, dimBlock) + calcClassSharedMemSize(cost, dimBlock) +
                             calcClassSharedMemSize<SAMPLER_T>(sampler, dimBlock);
#ifdef USE_CUDA_BARRIERS_ROLLOUT
  shared_mem_size += math::int_multiple_const(num_shared * sizeof(barrier), 16);
#endif
  return shared_mem_size;
}

template <class DYN_T, class COST_T, class SAMPLER_T>
unsigned calcVisualizeKernelSharedMemSize(const DYN_T* dynamics, const COST_T* cost, const SAMPLER_T* sampler,
                                          const int& num_timesteps, dim3& dimBlock)
{
  const int num_shared = dimBlock.x * dimBlock.z;
  unsigned shared_mem_size = sizeof(float) * (3 * math::nearest_multiple_4(num_shared * DYN_T::STATE_DIM) +
                                              math::nearest_multiple_4(num_shared * DYN_T::OUTPUT_DIM) +
                                              math::nearest_multiple_4(num_shared * DYN_T::CONTROL_DIM) +
                                              math::nearest_multiple_4(num_timesteps * dimBlock.y * dimBlock.z)) +
                             sizeof(int) * math::nearest_multiple_4(num_shared) +
                             calcClassSharedMemSize(dynamics, dimBlock) + calcClassSharedMemSize(cost, dimBlock) +
                             calcClassSharedMemSize<SAMPLER_T>(sampler, dimBlock);
#ifdef USE_CUDA_BARRIERS_ROLLOUT
  shared_mem_size += math::int_multiple_const(num_shared * sizeof(barrier), 16);
#endif
  return shared_mem_size;
}

template <class COST_T, class SAMPLER_T>
unsigned calcVisCostKernelSharedMemSize(const COST_T* cost, const SAMPLER_T* sampler, const int& num_timesteps,
                                        dim3& dimBlock)
{
  const int shared_num = dimBlock.x * dimBlock.z;
  unsigned shared_mem_size = sizeof(float) * (math::nearest_multiple_4(shared_num * COST_T::OUTPUT_DIM) +
                                              math::nearest_multiple_4(shared_num * COST_T::CONTROL_DIM) +
                                              math::nearest_multiple_4(dimBlock.z * num_timesteps * dimBlock.y)) +
                             sizeof(int) * math::nearest_multiple_4(dimBlock.z * num_timesteps) +
                             calcClassSharedMemSize(cost, dimBlock) +
                             calcClassSharedMemSize<SAMPLER_T>(sampler, dimBlock);
#ifdef USE_CUDA_BARRIERS_COST
  shared_mem_size += math::int_multiple_const(shared_num * sizeof(barrier), 16);
#endif
  return shared_mem_size;
}

template <class T>
__host__ __device__ unsigned calcClassSharedMemSize(const T* class_ptr, const dim3& dimBlock)
{
  const int num_shared = dimBlock.x * dimBlock.z;
  unsigned shared_size = math::int_multiple_const(class_ptr->getGrdSharedSizeBytes(), sizeof(float4)) +
                         num_shared * math::int_multiple_const(class_ptr->getBlkSharedSizeBytes(), sizeof(float4));
  return shared_size;
}
}  // namespace kernels
}  // namespace mppi



#endif // MPPIGENERICCONTROLLER_H
