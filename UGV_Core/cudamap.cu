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

__device__ float Rotation_matrix[9] = {0.99973f, 0.00235f, -0.02316f, -0.00523f, 0.99212f, -0.12521f, 0.02269f, 0.12530f, 0.99186f};
__device__ float Translation[3] = {-0.085f, 0.090f, 0.090f};

//__device__ float Rotation_matrix[9] = {1.0f, 0.0f, 0.0f, 0.0f, 0.7071f, 0.7072f, 0.0f, -0.7071f, 0.7071f};
//__device__ float Translation[3] = {0.0f, 0.0f, 0.0f};

__device__ uint32_t vec_counter = 0;

__device__
static inline
uint8_t atomicCAS8( uint8_t * const address, uint8_t   const compare, uint8_t   const value )
{
    // Determine where in a byte-aligned 32-bit range our address of 8 bits occurs.
    uint8_t    const     longAddressModulo = reinterpret_cast< size_t >( address ) & 0x3;
    // Determine the base address of the byte-aligned 32-bit range that contains our address of 8 bits.
    uint32_t * const     baseAddress       = reinterpret_cast< uint32_t * >( address - longAddressModulo );
    uint32_t   constexpr byteSelection[]   = { 0x3214, 0x3240, 0x3410, 0x4210 }; // The byte position we work on is '4'.
    uint32_t   const     byteSelector      = byteSelection[ longAddressModulo ];
    uint32_t   const     longCompare       = compare;
    uint32_t   const     longValue         = value;
    uint32_t             longOldValue      = * baseAddress;
    uint32_t             longAssumed;
    uint8_t              oldValue;

    do
    {
        // Select bytes from the old value and new value to construct a 32-bit value to use.
        uint32_t const replacement = __byte_perm( longOldValue, longValue,   byteSelector );
        uint32_t const comparison  = __byte_perm( longOldValue, longCompare, byteSelector );

        longAssumed  = longOldValue;
        // Use 32-bit atomicCAS() to try and set the 8-bits we care about.
        longOldValue = ::atomicCAS( baseAddress, comparison, replacement );
        // Grab the 8-bit portion we care about from the old value at address.
        oldValue     = ( longOldValue >> ( 8 * longAddressModulo )) & 0xFF;
    }
    while ( compare == oldValue and longAssumed != longOldValue ); // Repeat until other three 8-bit values stabilize.

    return oldValue;
}

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
void kernel_deproject_depth_cuda(uint8_t * rays_filter, uint16_t *ray_points, const rs2_intrinsics* intrin, const uint16_t * depth, float depth_scale, double w, double x, double y, double z)
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

        // Проецирование глубины в облако точек

        deproject_pixel_to_point_cuda(point, intrin, pixel, depth_scale * depth[j]);

        //******************************************************************************* Матрица вращения и вектор смещения для лидара *********************************************************************************//

        point_s[0] = (Rotation_matrix[0] * point[0] + Rotation_matrix[3] * point[1] + Rotation_matrix[6] * point[2]);

        point_s[1] = (Rotation_matrix[1] * point[0] + Rotation_matrix[4] * point[1] + Rotation_matrix[7] * point[2]);

        point_s[2] = (Rotation_matrix[2] * point[0] + Rotation_matrix[5] * point[1] + Rotation_matrix[8] * point[2]);

        point_s[0] += Translation[0];

        point_s[1] += Translation[1];

        point_s[2] += Translation[2];

        //****************************************************************************** Масштаб карты *********************************************************************************************************************//

        point_s[0] *= Map_scale;

        point_s[1] *= Map_scale;

        point_s[2] *= Map_scale;

        //******************************************************************************** Фильтрация облака по дальности и вращение кватернионом ***************************************************************************//

        points[0] = __float2int_rd(ww*point_s[0] + two*wy*point_s[2] - two*wz*point_s[1] + xx*point_s[0] + two*xy*point_s[1] + two*xz*point_s[2] - zz*point_s[0] - yy*point_s[0]) + Map_center_X;

        points[1] = __float2int_rd(two*xy*point_s[0] + yy*point_s[1] + two*yz*point_s[2] + two*wz*point_s[0] - zz*point_s[1] + ww*point_s[1] - two*wx*point_s[2] - xx*point_s[1]) + Map_center_Y;

        points[2] = __float2int_rd(two*xz*point_s[0] + two*yz*point_s[1] + zz*point_s[2] - two*wy*point_s[0] - yy*point_s[2] + two*wx*point_s[1] - xx*point_s[2] + ww*point_s[2]) + Map_center_Z;

        // ****************************************************************************** Проверка на выход за пределы карты и фильтрация новых точек воксельным фильтром ****************************************************//

        if((points[0] < Map_X) && (points[1] < Map_Y) && (points[2] < Map_Z) && (points[0] > 0) && (points[1] > 0) && (points[2] > 0)) {

        if(!atomicCAS8(&rays_filter[get_value(points[0], points[1], points[2])], 0, 1))
        {
            ray_points[j * 3] = points[0];
            ray_points[1 + j * 3] = points[1];
            ray_points[2 + j * 3] = points[2];
        }
        }
   }
}

//__global__
//void kernel_voxelization_pcl_cuda(uint8_t * rays_filter, uint16_t *ray_points, uint32_t size, double w, double x, double y, double z)
//{
//    int i = blockDim.x * blockIdx.x + threadIdx.x;

//    if (i >= size) {
//        return;
//    }

//    int stride = blockDim.x * gridDim.x;
//    int a, b;

//    const float ww = w * w;
//    const float xx = x * x;
//    const float yy = y * y;
//    const float zz = z * z;
//    const float wx = w * x;
//    const float wy = w * y;
//    const float wz = w * z;
//    const float xy = x * y;
//    const float xz = x * z;
//    const float yz = y * z;
//    const float two = 2.0f;

//    for (int j = i; j < size; j += stride) {

//        float point_s[3] = {0}; uint16_t points[3] = {0};

//        //******************************************************************************** Фильтрация облака по дальности и вращение кватернионом ***************************************************************************//

//        points[0] = __float2int_rd(ww*point_s[0] + two*wy*point_s[2] - two*wz*point_s[1] + xx*point_s[0] + two*xy*point_s[1] + two*xz*point_s[2] - zz*point_s[0] - yy*point_s[0]) * Map_scale + Map_center_X;

//        points[1] = __float2int_rd(two*xy*point_s[0] + yy*point_s[1] + two*yz*point_s[2] + two*wz*point_s[0] - zz*point_s[1] + ww*point_s[1] - two*wx*point_s[2] - xx*point_s[1]) * Map_scale + Map_center_Y;

//        points[2] = __float2int_rd(two*xz*point_s[0] + two*yz*point_s[1] + zz*point_s[2] - two*wy*point_s[0] - yy*point_s[2] + two*wx*point_s[1] - xx*point_s[2] + ww*point_s[2]) * Map_scale  + Map_center_Z;

//        // ****************************************************************************** Проверка на выход за пределы карты и фильтрация новых точек воксельным фильтром ****************************************************//

//        if((points[0] < Map_X) && (points[1] < Map_Y) && (points[2] < Map_Z) && (points[0] > 0) && (points[1] > 0) && (points[2] > 0)) {

//        if(!atomicCAS8(&rays_filter[get_value(points[0], points[1], points[2])], 0, 1))
//        {
//            ray_points[j * 3] = points[0];
//            ray_points[1 + j * 3] = points[1];
//            ray_points[2 + j * 3] = points[2];
//        }
//        }
//   }
//}

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

  while (true) {

      if (x >= min[0] && x < max[0] && y >= min[1] && y < max[1] && z >= min[2] && z < max[2]) {


          atomicCAS8(&voxels[get_value(x,y,z)], 1, 0);

          float dir_x = __int2float_rd(x - start[0]);
          float dir_y = __int2float_rd(y - start[1]);
          float dir_z = __int2float_rd(z - start[2]);

          float curDist  = (dir_x * dir_x) + (dir_y * dir_y) + (dir_z * dir_z);

          if (curDist >= maxDist) return;

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

      if( 4 == (abs(x - endX) + abs(y - endY) + abs(z - endZ))) return;

      //if (x == endX && y == endY && z == endZ) return;
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

__global__
void count_map_size(uint32_t * dev_world_counter, uint8_t * voxels2)
{
    uint32_t i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= Map_length) {
        return;
    }

    vec_counter = 0;

    uint32_t stride = blockDim.x * gridDim.x;

    for (uint32_t j = i; j < Map_length; j += stride) {

    if(voxels2[j]){voxels2[j] = 0; atomicAdd(&vec_counter, 1);}
    }

    *dev_world_counter = vec_counter;
}

__global__
void serialization_cuda(uint8_t * voxels, uint16_t * output, uint32_t * dev_world_counter)
{
  uint32_t i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i >= Map_length) {
      return;
  }

  uint32_t stride = blockDim.x * gridDim.x;

  vec_counter = 0;

  for (uint32_t j = i; j < Map_length; j += stride) {

  if(voxels[j]){

  uint32_t ii = atomicAdd(&vec_counter, 1);

  if(ii > *dev_world_counter) return;

  /*z*/output[ii * 3 + 2] = j / (Map_X * Map_Y);
  uint32_t remaining = j % (Map_X * Map_Y);
  /*y*/output[ii * 3 + 1]= remaining / Map_Z;
  /*x*/output[ii * 3] = j % Map_Z;
  }
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


//__global__
//void  offset_cuda (uint16_t * output, uint32_t * dev_world_counter, int16_t x_, int16_t y_, int16_t z_)
//{
//    uint32_t i = blockDim.x * blockIdx.x + threadIdx.x;

//    if (i >= Map_length) {
//        return;
//    }

//    uint32_t stride = blockDim.x * gridDim.x;

//    for (uint32_t j = i; j < *dev_world_counter; j += stride) {

//         uint16_t z = output[j * 3 + 2] - x_;
//         uint16_t y = output[j * 3 + 1] - z_;
//         uint16_t x = output[j * 3] - y_;

//         output[j * 3 + 2] = 0;
//         output[j * 3 + 1] = 0;
//         output[j * 3] = 0;

//          if(x > 0 && y > 0 && z > 0 && x < Map_X && y < Map_Y && z < Map_Z )
//          {
//              output[j * 3 + 2] = z;
//              output[j * 3 + 1] = y;
//              output[j * 3] = x;
//          }
//          }
//}

__global__
void offset_cuda(uint16_t *output, uint32_t *dev_world_counter, int16_t x_, int16_t y_, int16_t z_)
{
    uint32_t i = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t stride = blockDim.x * gridDim.x;

    uint32_t total_points = *dev_world_counter;

    for (uint32_t j = i; j < total_points; j += stride)
    {
        // Get a pointer to the current element
        uint16_t* point = &output[j * 3];

        // Read the original coordinates
        int16_t orig_x = point[0];
        int16_t orig_y = point[1];
        int16_t orig_z = point[2];

        // Calculate new coordinates taking into account the offset
        int16_t z = orig_z - x_;
        int16_t y = orig_y - z_;
        int16_t x = orig_x - y_;

        // Zero out the original coordinates
        point[0] = 0;
        point[1] = 0;
        point[2] = 0;

        // Check the bounds and assign new values
        if (x >= 0 && y >= 0 && z >= 0 && x < Map_X && y < Map_Y && z < Map_Z)
        {
            point[2] = z;
            point[1] = y;
            point[0] = x;
        }
    }
}

__global__
void deserialization_cuda (uint16_t * input, uint8_t * voxels, uint32_t dev_world_counter)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= dev_world_counter) {
        return;
    }

    int stride = blockDim.x * gridDim.x;

    for (int j = i; j < dev_world_counter; j += stride) {

        if((input[j * 3] > 0) && (input[1 + j * 3] > 0) && (input[2 + j * 3] > 0)) {

            voxels[get_value(input[j * 3],input[1 + j * 3],input[2 + j * 3])] = 1;
        }
    }
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


extern "C"
__host__
void deproject_depth_cuda(uint16_t ** serialization_point_1, uint16_t *serialization_point_2, uint32_t * world_counter, uint8_t * voxels, uint8_t * costmap, const rs2_intrinsics & intrin, const uint16_t * depth, float depth_scale, double w, double x, double y, double z, int16_t x_, int16_t y_, int16_t z_)
{
    int count = intrin.height * intrin.width;
    int numBlocks = count / RS2_CUDA_THREADS_PER_BLOCK ;

    int border_top = 200; //up
    int border_bottom = 0;  //down
    int border_left = 100;
    int border_right = 20;

    int16_t current_y = 44;
    uint8_t layer_y [100];

    // Threshold parameters
    //uint16_t min_val = 500;
    //uint16_t max_val = 5500;

    dim3 blockSize(32, 32);
    dim3 gridSize((intrin.width + blockSize.x - 1) / blockSize.x,
                  (intrin.height + blockSize.y - 1) / blockSize.y);

    uint32_t *dev_world_counter;
    uint16_t *dev_serializ_point;
    uint16_t *dev_ray_points;
    uint8_t * dev_costmap;
    uint8_t * dev_voxels;
    uint8_t * dev_voxels_2;
    uint8_t * dev_ray_filter;
    uint16_t *dev_depth;
    rs2_intrinsics* dev_intrin;
    cudaError_t result;

    cudaStream_t stream;
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;

    result = cudaMalloc(&dev_ray_points, count * sizeof(uint16_t) * 3);
    //std::cout<<"Stage 1"<<std::endl;
    assert(result == cudaSuccess);

    result = cudaMalloc(&dev_ray_filter, Map_length);
    //std::cout<<"Stage 2"<<std::endl;
    assert(result == cudaSuccess);

    result = cudaMalloc(&dev_depth, count * sizeof(uint16_t));
    //std::cout<<"Stage 3"<<std::endl;
    assert(result == cudaSuccess);

    result = cudaMalloc(&dev_intrin, sizeof(rs2_intrinsics));
    //std::cout<<"Stage 4"<<std::endl;
    assert(result == cudaSuccess);

    result = cudaMemcpy(dev_depth, depth, count * sizeof(uint16_t), cudaMemcpyHostToDevice);
    //std::cout<<"Stage 8"<<std::endl;
    assert(result == cudaSuccess);

    result = cudaMemcpy(dev_intrin, &intrin, sizeof(rs2_intrinsics), cudaMemcpyHostToDevice);
    //std::cout<<"Stage 5"<<std::endl;
    assert(result == cudaSuccess);

    crop_depth_borders<<<gridSize, blockSize>>>(dev_depth, intrin.width, intrin.height, border_top, border_bottom, border_left, border_right);

    cudaDeviceSynchronize();

    //threshold_filter<<<gridSize, blockSize>>>(dev_depth, intrin.width, intrin.height, min_val, max_val);

    cudaDeviceSynchronize();

    //std::cout<<"Deproject_depth "<<count<<std::endl;

    kernel_deproject_depth_cuda<<<numBlocks, RS2_CUDA_THREADS_PER_BLOCK>>>(dev_ray_filter, dev_ray_points,  dev_intrin, dev_depth, depth_scale, w,x,y,z);

    cudaFree(dev_ray_filter);
    cudaFree(dev_depth);
    cudaFree(dev_intrin);

    cudaDeviceSynchronize();

    result = cudaMalloc(&dev_voxels, Map_length);
    //std::cout<<"Stage 6"<<std::endl;
    assert(result == cudaSuccess);

    result = cudaMalloc(&dev_serializ_point, *world_counter * sizeof(uint16_t) * 3);
    //std::cout<<"Stage 14"<<std::endl;
    assert(result == cudaSuccess);

    result = cudaMemcpy(dev_serializ_point, serialization_point_2, *world_counter * sizeof(uint16_t) * 3, cudaMemcpyHostToDevice);
    //std::cout<<"Stage 7"<<std::endl;
    assert(result == cudaSuccess);

    deserialization_cuda<<<128, 128>>>(dev_serializ_point, dev_voxels, *world_counter);

    cudaFree(dev_serializ_point);

    if (serialization_point_2 != NULL) {
       free(serialization_point_2);
       serialization_point_2 = NULL;
    }

    //result = cudaMemcpy(dev_voxels, voxels, Map_length, cudaMemcpyHostToDevice);
    //std::cout<<"Stage 7"<<std::endl;
    //assert(result == cudaSuccess);

    //std::cout<<"Voxelization "<<std::endl;

    voxelization_cuda<<<128, 128>>>(dev_voxels, dev_ray_points, count);

    cudaDeviceSynchronize();

    result = cudaMalloc(&dev_world_counter, sizeof(uint32_t));
    //std::cout<<"Stage 8"<<std::endl;
    assert(result == cudaSuccess);

    result = cudaMemcpy(dev_world_counter, world_counter, sizeof(uint32_t), cudaMemcpyHostToDevice);
    //std::cout<<"Stage 9"<<std::endl;
    assert(result == cudaSuccess);

   //std::cout<<"RayCasting "<<std::endl;

   ray_casting_map<<<128, 128>>>(dev_voxels, dev_ray_points, count);

   cudaDeviceSynchronize();

   cudaFree(dev_ray_points);

   result = cudaMalloc(&dev_voxels_2, Map_length);
   //std::cout<<"Stage 10"<<std::endl;
   assert(result == cudaSuccess);

   result = cudaMemcpy(dev_voxels_2, dev_voxels, Map_length, cudaMemcpyDeviceToDevice);
   //std::cout<<"Stage 11"<<std::endl;
   assert(result == cudaSuccess);

   count_map_size<<<128, 128>>>(dev_world_counter, dev_voxels_2);

   cudaDeviceSynchronize();

   cudaFree(dev_voxels_2);

   //result = cudaMemcpy(voxels, dev_voxels, Map_length, cudaMemcpyDeviceToHost);
   //std::cout<<"Stage 12"<<std::endl;
   //assert(result == cudaSuccess);

   result = cudaMemcpy (world_counter, dev_world_counter, sizeof(uint32_t), cudaMemcpyDeviceToHost);
   //std::cout<<"Stage 13"<<std::endl;
   assert(result == cudaSuccess);

   result = cudaMalloc(&dev_serializ_point, *world_counter * sizeof(uint16_t) * 3);
   //std::cout<<"Stage 14"<<std::endl;
   assert(result == cudaSuccess);

   uint16_t * test = (uint16_t*)malloc(*world_counter * sizeof(uint16_t) * 3);

   //std::cout<<"Serialization "<<std::endl;

   //Обнаружено при количестве потоков 256 есть рассинхроны с voxels массивом, хз чем больше потокв хуже 128 оптимально
   serialization_cuda<<<128, 128>>>(dev_voxels, dev_serializ_point, dev_world_counter);

   cudaDeviceSynchronize();

   offset_cuda<<<128, 128>>>(dev_serializ_point, dev_world_counter, x_, y_, z_);

   result = cudaMemcpy(test, dev_serializ_point, *world_counter * sizeof(uint16_t) * 3, cudaMemcpyDeviceToHost);
   // std::cout<<"Stage 15"<<std::endl;
   assert(result == cudaSuccess);

   *serialization_point_1 = test;

   cudaDeviceSynchronize();

   compute_layer_y(layer_y, current_y);

   dim3 blockSize_2d(16, 16);
   dim3 gridSize_2d((400 + 15) / 16, (400 + 15) / 16);

   assert(dev_voxels != nullptr && "dev_voxels is null");

   result = cudaGetLastError();
   assert(result == cudaSuccess && "serialization_cuda kernel launch failed");

   result = cudaStreamCreate(&stream);
   assert(result == cudaSuccess && "Failed to create CUDA stream");

   result = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
   assert(result == cudaSuccess && "Failed to begin stream capture");

   // Теперь в макросе передаем все параметры

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
   assert(result == cudaSuccess && "Failed to end stream capture");

   result = cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
   assert(result == cudaSuccess && "Failed to instantiate graph");

   result = cudaGraphLaunch(graphExec, stream);
   assert(result == cudaSuccess && "Failed to launch graph");

   result = cudaStreamSynchronize(stream);
   assert(result == cudaSuccess && "Failed to synchronize stream after graph execution");

   // Проверяем, не было ли ошибок выполнения ядер в графе
   result = cudaGetLastError();
   assert(result == cudaSuccess && "Kernel execution failed");

   result = cudaGraphExecDestroy(graphExec);
   assert(result == cudaSuccess && "Failed to destroy graph executable");
   result = cudaGraphDestroy(graph);
   assert(result == cudaSuccess && "Failed to destroy graph");
   result = cudaStreamDestroy(stream);
   assert(result == cudaSuccess && "Failed to destroy stream");

   result = cudaMalloc(&dev_costmap, Map_X * Map_Z);
   //std::cout<<"Stage 6"<<std::endl;
   assert(result == cudaSuccess);

   assert(dev_voxels != nullptr && "dev_voxels is null for Convert_3D_voxels_map_to_2d_cost_map");
   assert(costmap != nullptr && "costmap is null");

   // Запуск ядра
   Convert_3D_voxels_map_to_2d_cost_map<<<gridSize_2d, blockSize_2d>>>(dev_voxels, dev_costmap, false, false);

   findContoursKernel_2d<<<gridSize_2d, blockSize_2d>>>(dev_costmap);

   Normalize_2d_cost_map<<<gridSize_2d, blockSize_2d>>>(dev_costmap);

   result = cudaGetLastError();
   assert(result == cudaSuccess && "Convert_3D_voxels_map_to_2d_cost_map kernel launch failed");   

   result = cudaMemcpy(costmap, dev_costmap, Map_X * Map_Z, cudaMemcpyDeviceToHost);
   //std::cout<<"Stage 7"<<std::endl;
   assert(result == cudaSuccess);

   cudaFree(dev_voxels);
   cudaFree(dev_costmap);

   cudaFree(dev_world_counter);
   cudaFree(dev_serializ_point);

}
