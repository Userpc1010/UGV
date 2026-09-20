#include <cuda.h>
#include <cuda_runtime.h>
#include <memory.h>
#include <iostream>
#include <librealsense2/rs.h>
#include <librealsense2/rsutil.h>
#include "assert.h"


#define RS2_CUDA_THREADS_PER_BLOCK 128
#define Map_length 16000000
#define Map_X 400
#define Map_Y 100
#define Map_Z 400
#define Map_center_X 199
#define Map_center_Y 49
#define Map_center_Z 199
#define Map_scale 10.0f

__device__ float Rotation_matrix[9] = {0.999048f, 0.018916f, -0.0393044f, 0.0f, 0.901077f, -0.433659f, 0.0436194f, 0.433246f, 0.900219f};
__device__ float Translation[3] = {-0.065f, 0.089f, 0.055f};


//__device__
//static inline
//uint8_t atomicCAS8( uint8_t * const address, uint8_t   const compare, uint8_t   const value )
//{
//    // Determine where in a byte-aligned 32-bit range our address of 8 bits occurs.
//    uint8_t    const     longAddressModulo = reinterpret_cast< size_t >( address ) & 0x3;
//    // Determine the base address of the byte-aligned 32-bit range that contains our address of 8 bits.
//    uint32_t * const     baseAddress       = reinterpret_cast< uint32_t * >( address - longAddressModulo );
//    uint32_t   constexpr byteSelection[]   = { 0x3214, 0x3240, 0x3410, 0x4210 }; // The byte position we work on is '4'.
//    uint32_t   const     byteSelector      = byteSelection[ longAddressModulo ];
//    uint32_t   const     longCompare       = compare;
//    uint32_t   const     longValue         = value;
//    uint32_t             longOldValue      = * baseAddress;
//    uint32_t             longAssumed;
//    uint8_t              oldValue;

//    do
//    {
//        // Select bytes from the old value and new value to construct a 32-bit value to use.
//        uint32_t const replacement = __byte_perm( longOldValue, longValue,   byteSelector );
//        uint32_t const comparison  = __byte_perm( longOldValue, longCompare, byteSelector );

//        longAssumed  = longOldValue;
//        // Use 32-bit atomicCAS() to try and set the 8-bits we care about.
//        longOldValue = ::atomicCAS( baseAddress, comparison, replacement );
//        // Grab the 8-bit portion we care about from the old value at address.
//        oldValue     = ( longOldValue >> ( 8 * longAddressModulo )) & 0xFF;
//    }
//    while ( compare == oldValue and longAssumed != longOldValue ); // Repeat until other three 8-bit values stabilize.

//    return oldValue;
//}

//__device__ uint16_t atomicAddShort(uint16_t* address, int16_t val)

//{

//    unsigned int *base_address = (unsigned int *)((size_t)address & ~2);

//    unsigned int long_val = ((size_t)address & 2) ? ((unsigned int)val << 16) : (int16_t)val;

//unsigned int long_old = atomicAdd(base_address, long_val);

//    if((size_t)address & 2) {

//        return (uint16_t)(long_old >> 16);

//    } else {

//        unsigned int overflow = ((long_old & 0xffff) + long_val) & 0xffff0000;

//        if (overflow)

//            atomicSub(base_address, overflow);

//        return (uint16_t)(long_old & 0xffff);

//    }
//}

__device__  uint32_t get_value (const uint16_t x, const uint16_t y, const uint16_t z) {
    uint32_t index = z * 40000 + y * 400 + x;
    return index;
}

__device__  uint32_t get_value_2d (const uint16_t x, const uint16_t y) {
    uint32_t index = y * 400 + x;
    return index;
}

__device__
float map_depth (float depth_scale, uint16_t z) {
    return depth_scale * z;
}

// CUDA kernel for threshold filtering, edits the source array
__global__ void threshold_filter(uint16_t* depth, size_t width, size_t height, uint16_t min_value, uint16_t max_value) {
    // Calculate the index of the current element on a two-dimensional grid
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        size_t idx = y * width + x;
        uint16_t val = depth[idx];
        if (val >= min_value && val <= max_value) {
            // Value in range - leave as is
            // Can be left unchanged or do something if necessary
        } else {
            // Value out of range - reset or assign another value
            depth[idx] = 0; // or another default value
        }
    }
}

__global__ void crop_depth_borders(uint16_t* depth, int width, int height, int border_left_right, int border_top_bottom)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    // Check if the pixel is in the edge area
    if (x < border_left_right || x >= (width - border_left_right) ||
        y < border_top_bottom || y >= (height - border_top_bottom))
    {
        depth[y * width + x] = 0;
    }
}

__global__ void crop_depth_borders( uint16_t* depth, int width, int height, int border_top, int border_bottom, int border_left, int border_right)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

     // Check if the pixel is in the edge area
    if (x < border_left || x >= (width - border_right) ||
        y < border_top || y >= (height - border_bottom))
    {
        depth[y * width + x] = 0;
    }
}

__device__
void deproject_pixel_to_point_cuda(float points[3], const struct rs2_intrinsics * intrin, const float pixel[2], float depth) {
    assert(intrin->model != RS2_DISTORTION_MODIFIED_BROWN_CONRADY); // Cannot deproject from a forward-distorted image
    assert(intrin->model != RS2_DISTORTION_FTHETA); // Cannot deproject to an ftheta image
    //assert(intrin->model != RS2_DISTORTION_BROWN_CONRADY); // Cannot deproject to an brown conrady model
    float x = (pixel[0] - intrin->ppx) / intrin->fx;
    float y = (pixel[1] - intrin->ppy) / intrin->fy;

    float xo = x;
    float yo = y;

    if (depth == 0.0f) return;

    if (intrin->model == RS2_DISTORTION_INVERSE_BROWN_CONRADY)
    {
        // need to loop until convergence
        // 10 iterations determined empirically
        for (int i = 0; i < 10; i++)
        {
            float r2 = x * x + y * y;
            float icdist = (float)1 / (float)(1 + ((intrin->coeffs[4] * r2 + intrin->coeffs[1])*r2 + intrin->coeffs[0])*r2);
            float xq = x / icdist;
            float yq = y / icdist;
            float delta_x = 2 * intrin->coeffs[2] * xq*yq + intrin->coeffs[3] * (r2 + 2 * xq*xq);
            float delta_y = 2 * intrin->coeffs[3] * xq*yq + intrin->coeffs[2] * (r2 + 2 * yq*yq);
            x = (xo - delta_x)*icdist;
            y = (yo - delta_y)*icdist;
        }
    }
    else if (intrin->model == RS2_DISTORTION_BROWN_CONRADY)
    {
        // need to loop until convergence
        // 10 iterations determined empirically
        for (int i = 0; i < 10; i++)
        {
            float r2 = x * x + y * y;
            float icdist = (float)1 / (float)(1 + ((intrin->coeffs[4] * r2 + intrin->coeffs[1])*r2 + intrin->coeffs[0])*r2);
            float delta_x = 2 * intrin->coeffs[2] * x*y + intrin->coeffs[3] * (r2 + 2 * x*x);
            float delta_y = 2 * intrin->coeffs[3] * x*y + intrin->coeffs[2] * (r2 + 2 * y*y);
            x = (xo - delta_x)*icdist;
            y = (yo - delta_y)*icdist;
        }
    }
    points[0] = depth * x;
    points[1] = depth * y;
    points[2] = depth;


}


__global__
void kernel_deproject_depth_cuda(uint16_t *ray_points, const rs2_intrinsics* intrin, const uint16_t * depth, float depth_scale, double w, double x, double y, double z)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= (*intrin).height * (*intrin).width) {
        return;
    }

    int stride = blockDim.x * gridDim.x;
    int a, b;

    const float ww = w * w;
    const float xx = x * x;
    const float yy = y * y;
    const float zz = z * z;
    const float wx = w * x;
    const float wy = w * y;
    const float wz = w * z;
    const float xy = x * y;
    const float xz = x * z;
    const float yz = y * z;
    const float two = 2.0f;

    for (int j = i; j < (*intrin).height * (*intrin).width; j += stride) {
        b = j / (*intrin).width;
        a = j - b * (*intrin).width;
        const float pixel[] = { (float)a, (float)b };

        float point[3] = {0}; float point_s[3] = {0}; uint16_t points[3] = {0};

        deproject_pixel_to_point_cuda(point, intrin, pixel, depth_scale * depth[j]);

        point_s[0] = (Rotation_matrix[0] * point[0] + Rotation_matrix[3] * point[1] + Rotation_matrix[6] * point[2]);
        point_s[1] = (Rotation_matrix[1] * point[0] + Rotation_matrix[4] * point[1] + Rotation_matrix[7] * point[2]);
        point_s[2] = (Rotation_matrix[2] * point[0] + Rotation_matrix[5] * point[1] + Rotation_matrix[8] * point[2]);

        point_s[0] += Translation[0];
        point_s[1] += Translation[1];
        point_s[2] += Translation[2];

        point_s[0] *= Map_scale;
        point_s[1] *= Map_scale;
        point_s[2] *= Map_scale;

        points[0] = __float2int_rd(ww*point_s[0] + two*wy*point_s[2] - two*wz*point_s[1] + xx*point_s[0] + two*xy*point_s[1] + two*xz*point_s[2] - zz*point_s[0] - yy*point_s[0]) + Map_center_X;
        points[1] = __float2int_rd(two*xy*point_s[0] + yy*point_s[1] + two*yz*point_s[2] + two*wz*point_s[0] - zz*point_s[1] + ww*point_s[1] - two*wx*point_s[2] - xx*point_s[1]) + Map_center_Y;
        points[2] = __float2int_rd(two*xz*point_s[0] + two*yz*point_s[1] + zz*point_s[2] - two*wy*point_s[0] - yy*point_s[2] + two*wx*point_s[1] - xx*point_s[2] + ww*point_s[2]) + Map_center_Z;

        if((points[0] < Map_X) && (points[1] < Map_Y) && (points[2] < Map_Z) && (points[0] > 0) && (points[1] > 0) && (points[2] > 0)) {
            ray_points[j * 3] = points[0];
            ray_points[1 + j * 3] = points[1];
            ray_points[2 + j * 3] = points[2];
        }
        else {
            ray_points[j * 3] = 0;
            ray_points[1 + j * 3] = 0;
            ray_points[2 + j * 3] = 0;
        }
   }
}

__device__ int16_t signum(int16_t x) {
  return x == 0 ? 0 : x < 0 ? -1 : 1;
}

__device__ float mod(float value, float modulus) {
  return fmod(fmod(value, modulus) + modulus, modulus);
}

__device__ float intbound(float s, float ds) {
  // Find the smallest positive t such that s+t*ds is an integer.
  if (ds < 0) {
    return intbound(-s, -ds);
  } else {
    s = mod(s, 1);
    // problem is now s+t*ds = 1
    return (1 - s) / ds;
  }
}

__device__
void Raycast_2_cuda(uint8_t * voxels, const uint16_t start[3], const uint16_t end[3], const uint16_t min[3], const uint16_t max[3])
{
  //    std::cout << start << ' ' << end << std::endl;
  // From "A Fast Voxel Traversal Algorithm for Ray Tracing"
  // by John Amanatides and Andrew Woo, 1987
  // <arhttp://www.cse.yorku.ca/~amana/resech/grid.pdf>
  // <http://citeseer.ist.psu.edu/viewdoc/summary?doi=10.1.1.42.3443>
  // Extensions to the described algorithm:
  //   • Imposed a distance limit.
  //   • The face passed through to reach the current cube is provided to
  //     the callback.

  // The foundation of this algorithm is a parameterized representation of
  // the provided ray,
  //                    origin + t * direction,
  // except that t is not actually stored; rather, at any given point in the
  // traversal, we keep track of the *greater* t values which we would have
  // if we took a step sufficient to cross a cube boundary along that axis
  // (i.e. change the integer part of the coordinate) in the variables
  // tMaxX, tMaxY, and tMaxZ.

    // Cube containing origin point.
    uint16_t x = start[0];
    uint16_t y = start[1];
    uint16_t z = start[2];
    uint16_t endX = end[0];
    uint16_t endY = end[1];
    uint16_t endZ = end[2];

    // Break out direction vector.
    int16_t dx = endX - x;
    int16_t dy = endY - y;
    int16_t dz = endZ - z;

    float direction_x = __int2float_rd(endX - start[0]);
    float direction_y = __int2float_rd(endY - start[1]);
    float direction_z = __int2float_rd(endZ - start[2]);

    float maxDist  = (direction_x * direction_x) + (direction_y * direction_y) + (direction_z * direction_z);

    // Direction to increment x,y,z when stepping.
    int16_t stepX = signum(dx);
    int16_t stepY = signum(dy);
    int16_t stepZ = signum(dz);

    float fdx = __int2float_rd(dx);
     float fdy = __int2float_rd(dy);
      float fdz = __int2float_rd(dz);

    // See description above. The initial values depend on the fractional
    // part of the origin.
    float tMaxX = intbound(__int2float_rd(start[0]), fdx);
    float tMaxY = intbound(__int2float_rd(start[1]), fdy);
    float tMaxZ = intbound(__int2float_rd(start[2]), fdz);

    // The change in t when taking a step (always positive).
    float tDeltaX = __int2float_rd(stepX) / fdx;
    float tDeltaY = __int2float_rd(stepY) / fdy;
    float tDeltaZ = __int2float_rd(stepZ) / fdz;

    // Avoids an infinite loop.
    if (stepX == 0 && stepY == 0 && stepZ == 0) return;

    // Safety limit — не более 3 * (Map_X + Map_Y + Map_Z) итераций
    int max_iterations = 3 * (Map_X + Map_Y + Map_Z);
    int iter = 0;

    while (iter++ < max_iterations) {

        if (x >= min[0] && x < max[0] && y >= min[1] && y < max[1] && z >= min[2] && z < max[2]) {

            float dir_x = __int2float_rd(x - start[0]);
            float dir_y = __int2float_rd(y - start[1]);
            float dir_z = __int2float_rd(z - start[2]);

            float curDist  = (dir_x * dir_x) + (dir_y * dir_y) + (dir_z * dir_z);

            if (curDist >= maxDist) return;

            // Нормальное условие выхода — достигли конца луча
            if (x == endX && y == endY && z == endZ) return;

            int ddx = (int)x - (int)endX;
            int ddy = (int)y - (int)endY;
            int ddz = (int)z - (int)endZ;
            if (ddx*ddx + ddy*ddy + ddz*ddz <= 9) return;   // радиус 3

            voxels[get_value(x,y,z)] = 0;

        }
        else return;

      // tMaxX stores the t-value at which we cross a cube boundary along the
      // X axis, and similarly for Y and Z. Therefore, choosing the least tMax
      // chooses the closest cube boundary. Only the first case of the four
      // has been commented in detail.
      if (tMaxX < tMaxY) {
        if (tMaxX < tMaxZ) {
          // Update which cube we are now in.
          x += stepX;
          // Adjust tMaxX to the next X-oriented boundary crossing.
          tMaxX += tDeltaX;
        } else {

          z += stepZ;
          tMaxZ += tDeltaZ;
        }
      } else {
        if (tMaxY < tMaxZ) {

          y += stepY;
          tMaxY += tDeltaY;
        } else {

          z += stepZ;
          tMaxZ += tDeltaZ;
        }
      }
  }
}


__global__
void voxelization_cuda( uint8_t * voxels, const uint16_t * ray_points, const unsigned int counter)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= counter) {
        return;
    }

    int stride = blockDim.x * gridDim.x;

    for (int j = i; j < counter; j += stride) {

        if((ray_points[j * 3] > 0) && (ray_points[1 + j * 3] > 0) && (ray_points[2 + j * 3] > 0)) {

            voxels[get_value(ray_points[j * 3], ray_points[1 + j * 3], ray_points[2 + j * 3])] = 1;
        }
    }
}

__global__
void ray_casting_map (uint8_t * voxels, const uint16_t *ray_points, const unsigned int counter)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= counter) {
        return;
    }

    int stride = blockDim.x * gridDim.x;

    for (int j = i; j < counter; j += stride) {

        if((ray_points[j * 3] > 0) && (ray_points[1 + j * 3] > 0) && (ray_points[2 + j * 3] > 0)) {

        uint16_t start[3] = {Map_center_X,Map_center_Y,Map_center_Z};  uint16_t end[3] = {ray_points[j * 3], ray_points[j * 3 + 1], ray_points[j * 3 + 2]}; uint16_t min[3] = {0,0,0}; uint16_t max[3] = {Map_X,Map_Y,Map_Z};

        Raycast_2_cuda(voxels, start, end, min, max);

        }
    }
}

// ============================================================================
// Вокселизация точек лидара (int16_t*, координаты в системе карты)
// ============================================================================
__global__
void voxelization_cuda_lidar(uint8_t *voxels, const int16_t *lidar_pcl, const uint32_t counter)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= counter) return;
    int stride = blockDim.x * gridDim.x;

    for (int j = i; j < counter; j += stride) {
        int16_t x = lidar_pcl[j * 3];
        int16_t y = lidar_pcl[j * 3 + 1];
        int16_t z = lidar_pcl[j * 3 + 2];

        if (x > 0 && x < Map_X && y > 0 && y < Map_Y && z > 0 && z < Map_Z) {
            voxels[get_value((uint16_t)x, (uint16_t)y, (uint16_t)z)] = 1;
        }
    }
}

// ============================================================================
// Ray casting для точек лидара (int16_t*, локальные координаты)
// ============================================================================
__global__
void ray_casting_map_lidar(uint8_t * voxels, const int16_t *lidar_pcl, const uint32_t counter)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= counter) return;
    int stride = blockDim.x * gridDim.x;

    for (int j = i; j < counter; j += stride) {
        int16_t points[3];
        points[0] = lidar_pcl[j * 3];
        points[1] = lidar_pcl[j * 3 + 1];
        points[2] = lidar_pcl[j * 3 + 2];

        // Ограничиваем координаты в [0, Map-1]
        if (points[0] < 0) points[0] = 0;
        if (points[0] >= Map_X) points[0] = Map_X - 1;
        if (points[1] < 0) points[1] = 0;
        if (points[1] >= Map_Y) points[1] = Map_Y - 1;
        if (points[2] < 0) points[2] = 0;
        if (points[2] >= Map_Z) points[2] = Map_Z - 1;

        const uint16_t start[3] = {Map_center_X, Map_center_Y, Map_center_Z};
        const uint16_t end[3]   = {(uint16_t)points[0], (uint16_t)points[1], (uint16_t)points[2]};
        const uint16_t min[3]   = {0, 0, 0};
        const uint16_t max[3]   = {Map_X, Map_Y, Map_Z};

        Raycast_2_cuda(voxels, start, end, min, max);
    }
}

// ============================================================================
// НОВЫЕ ЯДРА (взяты из 2-го проекта — лидарного)
// ============================================================================

// Блочная редукция для подсчёта занятых вокселей. Не обнуляет вход.
__global__ void count_map_size_2(uint8_t *voxels, uint32_t* d_result, uint32_t size)
{
    __shared__ int s_data[RS2_CUDA_THREADS_PER_BLOCK];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    s_data[tid] = (idx < size) ? voxels[idx] : 0;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && (idx + s) < size) {
            s_data[tid] += s_data[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        d_result[blockIdx.x] = s_data[0];
    }
}

// Сериализация: один поток — один элемент. Читает data[idx] == 1, пишет индекс в indices[pos]
// как float (битовая переинтерпретация uint32_t -> float). Не обнуляет data!
__global__ void serialization_cuda_2(const uint8_t *data, float* indices, uint32_t* count, uint32_t size)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    if (data[idx] == 1) {
        uint32_t pos = atomicAdd(count, 1);
        indices[pos] = __uint_as_float(idx);
    }
}

__global__
void offset_cuda_2(float *output, uint32_t *dev_world_counter, int16_t x_, int16_t y_, int16_t z_)
{
    uint32_t *output_u32 = reinterpret_cast<uint32_t*>(output);

    uint32_t i = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t stride = blockDim.x * gridDim.x;

    uint32_t total_points = *dev_world_counter;

    for (uint32_t j = i; j < total_points; j += stride)
    {
        uint32_t *point = &output_u32[j];

        int16_t orig_z = point[0] / (Map_X * Map_Y);
        uint32_t remaining = point[0] % (Map_X * Map_Y);
        int16_t orig_y  = remaining / Map_Z;
        int16_t orig_x  = point[0] % Map_Z;

        uint16_t z = orig_z - x_;
        uint16_t y = orig_y - z_;
        uint16_t x = orig_x - y_;

        point[0] = 0;

        if (x > 0 && y > 0 && z > 0 && x < Map_X && y < Map_Y && z < Map_Z)
        {
            point[0] = get_value(x, y, z);
        }
    }
}

// Восстановление dev_voxels из сдвинутого списка индексов.
// indices теперь float* — битовая интерпретация uint32_t (через reinterpret_cast).
__global__
void deserialization_cuda_2(const float *indices, uint8_t *voxels, uint32_t count)
{
    const uint32_t *indices_u32 = reinterpret_cast<const uint32_t*>(indices);

    uint32_t i = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t stride = blockDim.x * gridDim.x;

    for (uint32_t j = i; j < count; j += stride)
    {
        voxels[indices_u32[j]] = 1;
    }
}


// Вспомогательная функция для вычисления значения разницы высот
__device__ uint8_t calculate_height_value(int16_t layer_y, int16_t current_y) {
    return static_cast<uint8_t>(abs(layer_y - current_y));
}

// Ядро для поиска контуров в указанном слое Y
__global__ void findContoursKernel(uint8_t* data, int16_t layer_y, int16_t current_y) {
    // Вычисляем координаты x и z для текущего потока
    uint16_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint16_t z = blockIdx.y * blockDim.y + threadIdx.y;

    // Проверяем границы
    if (x >= 400 || z >= 400) return;

    uint32_t idx = get_value(x, layer_y, z);

    // Если текущая ячейка свободна - пропускаем
    if (data[idx] == 0) return;

     // Проверяем наложение блоков overlay filter
    if(data[get_value(x, layer_y + 1, z)]){ data[idx] = 0; return; }

    // Проверяем 4-связных соседей в плоскости XY
    int16_t dx[4] = {0, 0, -1, 1};
    int16_t dz[4] = {-1, 1, 0, 0};

    for (int8_t i = 0; i < 4; ++i) {
        int16_t nx = x + dx[i];
        int16_t nz = z + dz[i];

        // Если сосед за границами массива или свободен - это граница
        if (nx < 0 || nx >= 400 || nz < 0 || nz >= 400 || data[get_value(nx, layer_y, nz)] == 0) {

            data[idx] = calculate_height_value(layer_y, current_y);

            break;
        }
    }
}


// Основное ядро обработки
__global__ void processLayerKernel(uint8_t* data, int16_t layer_y, int16_t robot_height, int16_t current_y) {
    // Вычисляем координаты x и z для текущего потока
    uint16_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint16_t z = blockIdx.y * blockDim.y + threadIdx.y;

    // Проверяем границы
    if (x >= 400 || z >= 400) return;

    // 1. Обработка верхней области (y от layer_y + 1 до 100)
    uint32_t base_idx = get_value(x, layer_y, z);

    // Пропускаем свободные ячейки
    if (data[base_idx] == 0) return;

    // Проверяем наложение блоков overlay filter
   if(data[get_value(x, layer_y + 1, z)]){ data[base_idx] = 0; return; }

    bool is_contour = (data[base_idx] > 1 && data[base_idx] < 100);
    bool has_occupied_in_range = false;
    uint8_t occupied_count = 0;
    int16_t first_occupied_height = -1;
    robot_height += 1;

    // Проверяем ячейки выше
    for (int16_t y = layer_y + 1; y <= layer_y + robot_height; y++) {
        if (y >= 100) break;

        uint32_t idx = get_value(x, y, z);
        if (data[idx] != 0) {
            has_occupied_in_range = true;
            occupied_count++;
            if (first_occupied_height == -1) {
                first_occupied_height = y;
            }
        }
    }

    // Обработка в зависимости от типа ячейки
    if (is_contour) {
        // Случай 4: Ячейка контура
        if (has_occupied_in_range) {
            // 4.1: Есть занятые ячейки выше
            data[base_idx] = 254;

            // Удаляем занятые ячейки в диапазоне
            for (int16_t y = layer_y + 1; y <= layer_y + robot_height; y++) {
                if (y >= 100) break;
                uint32_t idx = get_value(x, y, z);
                data[idx] = 0;
            }
        } else {
            // 4.2: Нет занятых ячеек выше
            data[base_idx] = 1 + calculate_height_value(layer_y, current_y);
        }
    } else {
        // Обычная занятая ячейка
        if (!has_occupied_in_range) {
            // Случай 1: Нет занятых ячеек на высоте робота
            data[base_idx] = 1;
        } else if (occupied_count == 1 && first_occupied_height == layer_y + 1) {
            // Случай 2: Ровно одна занятая ячейка сразу над текущей
            uint32_t above_idx = get_value(x, layer_y + 1, z);
            data[above_idx] = 0;
            data[base_idx] = 1 + calculate_height_value(layer_y, current_y);
        } else {
            // Случай 3: Несколько занятых ячеек выше
            data[base_idx] = 254;

            // Удаляем все занятые ячейки в диапазоне
            for (int16_t y = layer_y + 1; y <= layer_y + robot_height; y++) {
                if (y >= 100) break;
                uint32_t idx = get_value(x, y, z);
                data[idx] = 0;
            }
        }
    }

    // Удаляем ячейки выше robot_height
       for (int16_t y = layer_y + robot_height + 1; y < 100; y++) {
           uint32_t idx = get_value(x, y, z);
           data[idx] = 0;
       }

    // 2. Обработка нижней области (y от 0 до current_y - 1)
    for (int16_t y = 0; y < layer_y - 1; y++) {
        uint32_t idx = get_value(x, y, z);
        data[idx] = 0;
    }
}

// Ядро для поиска контуров в 2д карте
__global__ void findContoursKernel_2d(uint8_t* data) {
    // Вычисляем координаты x и z для текущего потока
    uint16_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint16_t y = blockIdx.y * blockDim.y + threadIdx.y;

    // Проверяем границы
    if (x >= 400 || y >= 400) return;

    uint32_t idx = get_value_2d(x, y);

    // Если текущая ячейка свободна - пропускаем
    if (data[idx] == 0) return;

    // Проверяем 8-связных соседей в плоскости XY
    int16_t dx[8] = {0, 0, -1, 1, 1, -1,  1, -1};
    int16_t dy[8] = {-1, 1, 0, 0, 1, -1, -1,  1};

    uint8_t current_cell = data[idx];

    for (int8_t i = 0; i < 8; ++i) {
        int16_t nx = x + dx[i];
        int16_t ny = y + dy[i];

        // Если сосед за границами массива или свободен - это граница
        if (nx < 0 || nx >= 400 || ny < 0 || ny >= 400 || data[get_value_2d(nx, ny)] == 0 || 2 >= abs(nx - current_cell) || 2 >= abs(ny - current_cell)) {

            data[idx] = 254;

            break;
        }
    }
}

__global__ void Convert_3D_voxels_map_to_2d_cost_map(const uint8_t* input_3d,  uint8_t* output_2d,  bool mirror_x, bool mirror_z ) {
    // Вычисляем координаты x и z для текущего потока
    uint16_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint16_t z = blockIdx.y * blockDim.y + threadIdx.y;

    // Проверяем границы
    if (x >= 400 || z >= 400) return;

    // Применяем зеркальное отображение к координатам выхода
    uint16_t output_x = mirror_x ? (399 - x) : x;
    uint16_t output_y = mirror_z ? (399 - z) : z;

    // Проходим по высоте от 0 до 99
    for (uint16_t y = 0; y < 100; ++y) {
        // Вычисляем индекс в 3D массиве (оригинальные координаты)
        uint32_t index = z * 40000 + y * 400 + x;

        // Если нашли ненулевое значение
        if (input_3d[index] != 0) {
            // Записываем его в 2D массив в зеркальную позицию
            uint32_t output_index = output_y * 400 + output_x;
            output_2d[output_index] = input_3d[index];
            break;
        }
    }


}

__global__ void Normalize_2d_cost_map(uint8_t* output_2d)
{
    // Вычисляем координаты x и z для текущего потока
    uint16_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint16_t y = blockIdx.y * blockDim.y + threadIdx.y;

    // Проверяем границы
    if (x >= 400 || y >= 400) return;

    uint32_t idx = get_value_2d(x, y);

    if (output_2d[idx] == 0) output_2d[idx] = 255;

    if (output_2d[idx] == 1) output_2d[idx] = 0;
}


void compute_layer_y (uint8_t arr[100], int8_t start) {
    int8_t left = start;
    int8_t right = start;
    int8_t count = 0;

    arr[count++] = start;

    for (int8_t step = 1; count < 100; step++) {
        // Правая сторона (увеличиваем)
        right = start + step;
        if (right < 100 && count < 100) {
            arr[count++] = right;
        }

        // Левая сторона (уменьшаем)
        left = start - step;
        if (left >= 0 && count < 100) {
            arr[count++] = left;
        }
    }
}

// ============================================================================
// Инициализация / освобождение персистентных GPU-буферов
// ============================================================================

extern "C"
__host__
void init_cuda_map(uint8_t **dev_voxels, uint8_t **dev_costmap)
{
    cudaError_t result;

    result = cudaMalloc(dev_voxels, Map_length);
    assert(result == cudaSuccess);

    result = cudaMemset(*dev_voxels, 0, Map_length);
    assert(result == cudaSuccess);

    result = cudaMalloc(dev_costmap, Map_X * Map_Z);
    assert(result == cudaSuccess);

    result = cudaMemset(*dev_costmap, 0, Map_X * Map_Z);
    assert(result == cudaSuccess);

    result = cudaDeviceSynchronize();
    assert(result == cudaSuccess);

    std::cout << "[CUDA] init_cuda_map: dev_voxels=" << (void*)*dev_voxels
              << " dev_costmap=" << (void*)*dev_costmap << std::endl;
}

extern "C"
__host__
void cleanup_cuda_map(uint8_t *dev_voxels, uint8_t *dev_costmap)
{
    cudaError_t result;

    if (dev_voxels) {
        result = cudaFree(dev_voxels);
        assert(result == cudaSuccess);
    }
    if (dev_costmap) {
        result = cudaFree(dev_costmap);
        assert(result == cudaSuccess);
    }

    result = cudaDeviceSynchronize();
    assert(result == cudaSuccess);

    std::cout << "[CUDA] cleanup_cuda_map done" << std::endl;
}


extern "C"
__host__
void deproject_depth_cuda(float **serialization_point_1, uint32_t *world_counter,
                          uint8_t *dev_voxels,
                          const rs2_intrinsics &intrin, const uint16_t *depth, float depth_scale,
                          double w, double x, double y, double z,
                          int16_t x_, int16_t y_, int16_t z_,
                          const uint16_t *keyframe_pcl, uint32_t keyframe_size,
                          const int16_t  *lidar_pcl,    uint32_t lidar_size)
{
    cudaError_t result;

    int count = intrin.height * intrin.width;
    int numBlocks = count / RS2_CUDA_THREADS_PER_BLOCK;

    int border_top = 200;
    int border_bottom = 0;
    int border_left = 100;
    int border_right = 20;

    // ========================================================================
    // ЛОКАЛЬНЫЕ БУФЕРЫ (на один кадр)
    // ========================================================================
    uint16_t *dev_depth;
    rs2_intrinsics *dev_intrin;
    uint16_t *dev_ray_points;
    uint32_t *dev_local_counter;
    float *dev_serializ_point;

    result = cudaMalloc(&dev_ray_points, count * sizeof(uint16_t) * 3);
    std::cout << "Stage 1 (malloc ray_points)" << std::endl;
    assert(result == cudaSuccess);

    result = cudaMalloc(&dev_depth, count * sizeof(uint16_t));
    std::cout << "Stage 2 (malloc depth)" << std::endl;
    assert(result == cudaSuccess);

    result = cudaMalloc(&dev_intrin, sizeof(rs2_intrinsics));
    std::cout << "Stage 3 (malloc intrin)" << std::endl;
    assert(result == cudaSuccess);

    uint32_t blocks = (Map_length + RS2_CUDA_THREADS_PER_BLOCK - 1) / RS2_CUDA_THREADS_PER_BLOCK;

    result = cudaMalloc(&dev_local_counter, blocks * sizeof(uint32_t));
    std::cout << "Stage 4 (malloc local_counter, blocks=" << blocks << ")" << std::endl;
    assert(result == cudaSuccess);

    result = cudaMemset(dev_local_counter, 0, blocks * sizeof(uint32_t));
    std::cout << "Stage 5 (memset local_counter)" << std::endl;
    assert(result == cudaSuccess);

    result = cudaMalloc(&dev_serializ_point, Map_length * sizeof(float));
    std::cout << "Stage 6 (malloc serializ_point)" << std::endl;
    assert(result == cudaSuccess);

    // ========================================================================
    // ЗАГРУЗКА ГЛУБИНЫ И ИНТРИНСИКОВ
    // ========================================================================
    result = cudaMemcpy(dev_depth, depth, count * sizeof(uint16_t), cudaMemcpyHostToDevice);
    std::cout << "Stage 7 (memcpy depth)" << std::endl;
    assert(result == cudaSuccess);

    result = cudaMemcpy(dev_intrin, &intrin, sizeof(rs2_intrinsics), cudaMemcpyHostToDevice);
    std::cout << "Stage 8 (memcpy intrin)" << std::endl;
    assert(result == cudaSuccess);

    // ========================================================================
    // ФИЛЬТРАЦИЯ ГРАНИЦ ГЛУБИНЫ
    // ========================================================================
    dim3 blockSize(32, 32);
    dim3 gridSize((intrin.width + blockSize.x - 1) / blockSize.x,
                  (intrin.height + blockSize.y - 1) / blockSize.y);

    crop_depth_borders<<<gridSize, blockSize>>>(dev_depth, intrin.width, intrin.height,
                                                 border_top, border_bottom, border_left, border_right);
    result = cudaDeviceSynchronize();
    std::cout << "Stage 9 (crop_depth_borders)" << std::endl;
    assert(result == cudaSuccess);

    // ========================================================================
    // DEPROJECT
    // ========================================================================
    kernel_deproject_depth_cuda<<<numBlocks, RS2_CUDA_THREADS_PER_BLOCK>>>(
        dev_ray_points, dev_intrin, dev_depth,
        depth_scale, w, x, y, z);

    result = cudaDeviceSynchronize();
    std::cout << "Stage 10 (kernel_deproject_depth_cuda)" << std::endl;
    assert(result == cudaSuccess);

    cudaFree(dev_depth);
    cudaFree(dev_intrin);

    // ========================================================================
    // RAY CASTING (камера)
    // ========================================================================
    ray_casting_map<<<128, 128>>>(dev_voxels, dev_ray_points, count);

    result = cudaDeviceSynchronize();
    std::cout << "Stage 11 (ray_casting_map camera)" << std::endl;
    assert(result == cudaSuccess);

    // ========================================================================
    // ВОКСЕЛИЗАЦИЯ
    // ========================================================================
    voxelization_cuda<<<128, 128>>>(dev_voxels, dev_ray_points, count);

    result = cudaDeviceSynchronize();
    std::cout << "Stage 12 (voxelization_cuda)" << std::endl;
    assert(result == cudaSuccess);

    cudaFree(dev_ray_points);

    // ========================================================================
    // ОБРАБОТКА ЛИДАРА
    // ========================================================================

    if (lidar_size > 0) {
        int16_t *dev_lidar_pcl;
        result = cudaMalloc(&dev_lidar_pcl, lidar_size * 3 * sizeof(int16_t));
        std::cout << "Stage 13.1 (malloc lidar_pcl, size=" << lidar_size << ")" << std::endl;
        assert(result == cudaSuccess);

        result = cudaMemcpy(dev_lidar_pcl, lidar_pcl,
                            lidar_size * 3 * sizeof(int16_t), cudaMemcpyHostToDevice);
        std::cout << "Stage 13.2 (memcpy lidar_pcl)" << std::endl;
        assert(result == cudaSuccess);

        // Ray casting — зачищаем free space по актуальному скану
        ray_casting_map_lidar<<<128, 128>>>(dev_voxels, dev_lidar_pcl, lidar_size);
        result = cudaDeviceSynchronize();
        std::cout << "Stage 13.3 (ray_casting_map_lidar)" << std::endl;
        assert(result == cudaSuccess);

        cudaFree(dev_lidar_pcl);
    }

    if (keyframe_size > 0) {
        int16_t *dev_keyframe_pcl;
        result = cudaMalloc(&dev_keyframe_pcl, keyframe_size * 3 * sizeof(int16_t));
        std::cout << "Stage 13.4 (malloc keyframe_pcl, size=" << keyframe_size << ")" << std::endl;
        assert(result == cudaSuccess);

        result = cudaMemcpy(dev_keyframe_pcl, keyframe_pcl,
                            keyframe_size * 3 * sizeof(int16_t), cudaMemcpyHostToDevice);
        std::cout << "Stage 13.5 (memcpy keyframe_pcl)" << std::endl;
        assert(result == cudaSuccess);

        // voxelization — ставим occupied по накопленному keyframe
        voxelization_cuda_lidar<<<128, 128>>>(dev_voxels, dev_keyframe_pcl, keyframe_size);
        result = cudaDeviceSynchronize();
        std::cout << "Stage 13.6 (voxelization_cuda_lidar keyframe)" << std::endl;
        assert(result == cudaSuccess);

        cudaFree(dev_keyframe_pcl);
    }

    // ========================================================================
    // ПОДСЧЁТ ЗАНЯТЫХ ВОКСЕЛЕЙ
    // ========================================================================

    count_map_size_2<<<blocks, RS2_CUDA_THREADS_PER_BLOCK>>>(dev_voxels, dev_local_counter, Map_length);

    result = cudaDeviceSynchronize();
    std::cout << "Stage 14 (count_map_size_2)" << std::endl;
    assert(result == cudaSuccess);

    uint32_t *h_result = new uint32_t[blocks];
    result = cudaMemcpy(h_result, dev_local_counter, blocks * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    std::cout << "Stage 15 (memcpy count result)" << std::endl;
    assert(result == cudaSuccess);

    uint32_t total_sum = 0;
    for (uint32_t i = 0; i < blocks; ++i) total_sum += h_result[i];
    delete[] h_result;

    *world_counter = total_sum;
    std::cout << "Stage 16 (world_counter = " << total_sum << ")" << std::endl;

    // ========================================================================
    // СЕРИАЛИЗАЦИЯ
    // ========================================================================
    cudaMemset(dev_local_counter, 0, blocks * sizeof(uint32_t));
    std::cout << "Stage 17 (reset local_counter)" << std::endl;
    assert(result == cudaSuccess);

    serialization_cuda_2<<<blocks, RS2_CUDA_THREADS_PER_BLOCK>>>(
        dev_voxels, dev_serializ_point, dev_local_counter, Map_length);

    result = cudaDeviceSynchronize();
    std::cout << "Stage 18 (serialization_cuda_2)" << std::endl;
    assert(result == cudaSuccess);

    // ========================================================================
    // СДВИГ СПИСКА
    // ========================================================================
    offset_cuda_2<<<128, 128>>>(dev_serializ_point, dev_local_counter, x_, y_, z_);

    result = cudaDeviceSynchronize();
    std::cout << "Stage 19 (offset_cuda_2)" << std::endl;
    assert(result == cudaSuccess);

    // ========================================================================
    // ДЕСЕРИАЛИЗАЦИЯ
    // ========================================================================
    result = cudaMemset(dev_voxels, 0, Map_length);
    std::cout << "Stage 20 (memset dev_voxels)" << std::endl;
    assert(result == cudaSuccess);

    deserialization_cuda_2<<<blocks, RS2_CUDA_THREADS_PER_BLOCK>>>(
        dev_serializ_point, dev_voxels, total_sum);

    result = cudaDeviceSynchronize();
    std::cout << "Stage 21 (deserialization_cuda_2)" << std::endl;
    assert(result == cudaSuccess);

    // ========================================================================
    // КОПИЯ СПИСКА НА CPU
    // ========================================================================
    float *buff = (float*)malloc(total_sum * sizeof(float));
    result = cudaMemcpy(buff, dev_serializ_point, total_sum * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "Stage 22 (memcpy serializ_point to CPU)" << std::endl;
    assert(result == cudaSuccess);

    *serialization_point_1 = buff;

    // ========================================================================
    // ОСВОБОЖДЕНИЕ
    // ========================================================================
    cudaFree(dev_local_counter);
    cudaFree(dev_serializ_point);

    result = cudaDeviceSynchronize();
    std::cout << "Stage 23 (final sync)" << std::endl;
    assert(result == cudaSuccess);

}

// ============================================================================
// Построение 2D costmap из 3D воксельной карты
// ============================================================================

extern "C"
__host__
void process_costmap_cuda(uint8_t *dev_voxels, uint8_t *dev_costmap,
                          uint8_t *costmap_host, int16_t current_y)
{
    cudaError_t result;

    uint8_t layer_y[100];
    compute_layer_y(layer_y, current_y);

    dim3 blockSize_2d(16, 16);
    dim3 gridSize_2d((400 + 15) / 16, (400 + 15) / 16);

    assert(dev_voxels != nullptr && "dev_voxels is null");
    assert(dev_costmap != nullptr && "dev_costmap is null");

    cudaStream_t stream;
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;

    result = cudaStreamCreate(&stream);
    std::cout << "Costmap Stage 1 (stream create)" << std::endl;
    assert(result == cudaSuccess && "Failed to create CUDA stream");

    result = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    std::cout << "Costmap Stage 2 (begin capture)" << std::endl;
    assert(result == cudaSuccess && "Failed to begin stream capture");

    // =========================================================================
    // ВАЖНО: захват в цикле for не работает корректно. Нужно развернуть вручную.
    // =========================================================================

    findContoursKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[0], current_y);
    processLayerKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[0], current_y, 5);

    findContoursKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[1], current_y);
    processLayerKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[1], current_y, 5);

    findContoursKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[2], current_y);
    processLayerKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[2], current_y, 5);

    findContoursKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[3], current_y);
    processLayerKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[3], current_y, 5);

    findContoursKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[4], current_y);
    processLayerKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[4], current_y, 5);

    findContoursKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[5], current_y);
    processLayerKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[5], current_y, 5);

    findContoursKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[6], current_y);
    processLayerKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[6], current_y, 5);

    findContoursKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[7], current_y);
    processLayerKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[7], current_y, 5);

    findContoursKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[8], current_y);
    processLayerKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[8], current_y, 5);

    findContoursKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[9], current_y);
    processLayerKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[9], current_y, 5);

    findContoursKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[10], current_y);
    processLayerKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[10], current_y, 5);

    findContoursKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[11], current_y);
    processLayerKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[11], current_y, 5);

    findContoursKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[12], current_y);
    processLayerKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[12], current_y, 5);

    findContoursKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[13], current_y);
    processLayerKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[13], current_y, 5);

    findContoursKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[14], current_y);
    processLayerKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[14], current_y, 5);

    findContoursKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[15], current_y);
    processLayerKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[15], current_y, 5);

    findContoursKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[16], current_y);
    processLayerKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[16], current_y, 5);

    findContoursKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[17], current_y);
    processLayerKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[17], current_y, 5);

    findContoursKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[18], current_y);
    processLayerKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[18], current_y, 5);

    findContoursKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[19], current_y);
    processLayerKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[19], current_y, 5);

    findContoursKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[20], current_y);
    processLayerKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[20], current_y, 5);

    findContoursKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[21], current_y);
    processLayerKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[21], current_y, 5);

    findContoursKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[22], current_y);
    processLayerKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[22], current_y, 5);

    findContoursKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[23], current_y);
    processLayerKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[23], current_y, 5);

    findContoursKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[24], current_y);
    processLayerKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[24], current_y, 5);

    findContoursKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[25], current_y);
    processLayerKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[25], current_y, 5);

    findContoursKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[26], current_y);
    processLayerKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[26], current_y, 5);

    findContoursKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[27], current_y);
    processLayerKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[27], current_y, 5);

    findContoursKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[28], current_y);
    processLayerKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[28], current_y, 5);

    findContoursKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[29], current_y);
    processLayerKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[29], current_y, 5);

    findContoursKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[30], current_y);
    processLayerKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[30], current_y, 5);

    findContoursKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[31], current_y);
    processLayerKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[31], current_y, 5);

    findContoursKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[32], current_y);
    processLayerKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[32], current_y, 5);

    findContoursKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[33], current_y);
    processLayerKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[33], current_y, 5);

    findContoursKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[34], current_y);
    processLayerKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[34], current_y, 5);

    findContoursKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[35], current_y);
    processLayerKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[35], current_y, 5);

    findContoursKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[36], current_y);
    processLayerKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[36], current_y, 5);

    findContoursKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[37], current_y);
    processLayerKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[37], current_y, 5);

    findContoursKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[38], current_y);
    processLayerKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[38], current_y, 5);

    findContoursKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[39], current_y);
    processLayerKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[39], current_y, 5);

    findContoursKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[40], current_y);
    processLayerKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[40], current_y, 5);

    findContoursKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[41], current_y);
    processLayerKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[41], current_y, 5);

    findContoursKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[42], current_y);
    processLayerKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[42], current_y, 5);

    findContoursKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[43], current_y);
    processLayerKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[43], current_y, 5);

    findContoursKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[44], current_y);
    processLayerKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[44], current_y, 5);

    findContoursKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[45], current_y);
    processLayerKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[45], current_y, 5);

    findContoursKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[46], current_y);
    processLayerKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[46], current_y, 5);

    findContoursKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[47], current_y);
    processLayerKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[47], current_y, 5);

    findContoursKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[48], current_y);
    processLayerKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[48], current_y, 5);

    findContoursKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[49], current_y);
    processLayerKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[49], current_y, 5);

    findContoursKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[50], current_y);
    processLayerKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[50], current_y, 5);

    findContoursKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[51], current_y);
    processLayerKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[51], current_y, 5);

    findContoursKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[52], current_y);
    processLayerKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[52], current_y, 5);

    findContoursKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[53], current_y);
    processLayerKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[53], current_y, 5);

    findContoursKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[54], current_y);
    processLayerKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[54], current_y, 5);

    findContoursKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[55], current_y);
    processLayerKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[55], current_y, 5);

    findContoursKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[56], current_y);
    processLayerKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[56], current_y, 5);

    findContoursKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[57], current_y);
    processLayerKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[57], current_y, 5);

    findContoursKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[58], current_y);
    processLayerKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[58], current_y, 5);

    findContoursKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[59], current_y);
    processLayerKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[59], current_y, 5);

    findContoursKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[60], current_y);
    processLayerKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[60], current_y, 5);

    findContoursKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[61], current_y);
    processLayerKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[61], current_y, 5);

    findContoursKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[62], current_y);
    processLayerKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[62], current_y, 5);

    findContoursKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[63], current_y);
    processLayerKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[63], current_y, 5);

    findContoursKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[64], current_y);
    processLayerKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[64], current_y, 5);

    findContoursKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[65], current_y);
    processLayerKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[65], current_y, 5);

    findContoursKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[66], current_y);
    processLayerKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[66], current_y, 5);

    findContoursKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[67], current_y);
    processLayerKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[67], current_y, 5);

    findContoursKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[68], current_y);
    processLayerKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[68], current_y, 5);

    findContoursKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[69], current_y);
    processLayerKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[69], current_y, 5);

    findContoursKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[70], current_y);
    processLayerKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[70], current_y, 5);

    findContoursKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[71], current_y);
    processLayerKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[71], current_y, 5);

    findContoursKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[72], current_y);
    processLayerKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[72], current_y, 5);

    findContoursKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[73], current_y);
    processLayerKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[73], current_y, 5);

    findContoursKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[74], current_y);
    processLayerKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[74], current_y, 5);

    findContoursKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[75], current_y);
    processLayerKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[75], current_y, 5);

    findContoursKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[76], current_y);
    processLayerKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[76], current_y, 5);

    findContoursKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[77], current_y);
    processLayerKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[77], current_y, 5);

    findContoursKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[78], current_y);
    processLayerKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[78], current_y, 5);

    findContoursKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[79], current_y);
    processLayerKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[79], current_y, 5);

    findContoursKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[80], current_y);
    processLayerKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[80], current_y, 5);

    findContoursKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[81], current_y);
    processLayerKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[81], current_y, 5);

    findContoursKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[82], current_y);
    processLayerKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[82], current_y, 5);

    findContoursKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[83], current_y);
    processLayerKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[83], current_y, 5);

    findContoursKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[84], current_y);
    processLayerKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[84], current_y, 5);

    findContoursKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[85], current_y);
    processLayerKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[85], current_y, 5);

    findContoursKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[86], current_y);
    processLayerKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[86], current_y, 5);

    findContoursKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[87], current_y);
    processLayerKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[87], current_y, 5);

    findContoursKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[88], current_y);
    processLayerKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[88], current_y, 5);

    findContoursKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[89], current_y);
    processLayerKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[89], current_y, 5);

    findContoursKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[90], current_y);
    processLayerKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[90], current_y, 5);

    findContoursKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[91], current_y);
    processLayerKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[91], current_y, 5);

    findContoursKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[92], current_y);
    processLayerKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[92], current_y, 5);

    findContoursKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[93], current_y);
    processLayerKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[93], current_y, 5);

    findContoursKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[94], current_y);
    processLayerKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[94], current_y, 5);

    findContoursKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[95], current_y);
    processLayerKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[95], current_y, 5);

    findContoursKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[96], current_y);
    processLayerKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[96], current_y, 5);

    findContoursKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[97], current_y);
    processLayerKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[97], current_y, 5);

    findContoursKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[98], current_y);
    processLayerKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[98], current_y, 5);

    findContoursKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[99], current_y);
    processLayerKernel<<<gridSize_2d, blockSize_2d, 0, stream>>>(dev_voxels, layer_y[99], current_y, 5);


    result = cudaStreamEndCapture(stream, &graph);
    std::cout << "Costmap Stage 3 (end capture)" << std::endl;
    assert(result == cudaSuccess && "Failed to end stream capture");

    result = cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
    std::cout << "Costmap Stage 4 (graph instantiate)" << std::endl;
    assert(result == cudaSuccess && "Failed to instantiate graph");

    result = cudaGraphLaunch(graphExec, stream);
    std::cout << "Costmap Stage 5 (graph launch)" << std::endl;
    assert(result == cudaSuccess && "Failed to launch graph");

    result = cudaStreamSynchronize(stream);
    std::cout << "Costmap Stage 6 (stream sync)" << std::endl;
    assert(result == cudaSuccess && "Failed to synchronize stream after graph execution");

    result = cudaGetLastError();
    std::cout << "Costmap Stage 7 (get last error)" << std::endl;
    assert(result == cudaSuccess && "Kernel execution failed");

    result = cudaGraphExecDestroy(graphExec);
    result = cudaGraphDestroy(graph);
    result = cudaStreamDestroy(stream);

    // =========================================================================
    // 2D costmap
    // =========================================================================
    Convert_3D_voxels_map_to_2d_cost_map<<<gridSize_2d, blockSize_2d>>>(dev_voxels, dev_costmap, false, false);
    std::cout << "Costmap Stage 8 (Convert_3D_to_2d)" << std::endl;

    findContoursKernel_2d<<<gridSize_2d, blockSize_2d>>>(dev_costmap);
    std::cout << "Costmap Stage 9 (findContoursKernel_2d)" << std::endl;

    Normalize_2d_cost_map<<<gridSize_2d, blockSize_2d>>>(dev_costmap);
    std::cout << "Costmap Stage 10 (Normalize_2d_cost_map)" << std::endl;

    result = cudaGetLastError();
    assert(result == cudaSuccess && "costmap kernels failed");

    result = cudaMemcpy(costmap_host, dev_costmap, Map_X * Map_Z, cudaMemcpyDeviceToHost);
    std::cout << "Costmap Stage 11 (memcpy costmap to CPU)" << std::endl;
    assert(result == cudaSuccess);

    result = cudaDeviceSynchronize();
    std::cout << "Costmap Stage 12 (final sync)" << std::endl;
    assert(result == cudaSuccess);
}
