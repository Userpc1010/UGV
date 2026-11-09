#include <cuda.h>
#include <cuda_runtime.h>
#include <memory.h>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <librealsense2/rs.h>
#include <librealsense2/rsutil.h>
#include "map_config.h"
#include "assert.h"

// ============================================================================
// Frustum binning parameters
// ============================================================================
#define FRUSTUM_MAX_LAYERS 256

static const float CAM_NEAR_MM      = 500.0f;
static const float CAM_FAR_MM       = 5000.0f;
static const float FRUSTUM_VOXEL_MM = 100.0f;

__constant__ int   c_Nz;
__constant__ int   c_Nx_per_layer[FRUSTUM_MAX_LAYERS];
__constant__ int   c_Ny_per_layer[FRUSTUM_MAX_LAYERS];
__constant__ int   c_layer_offset[FRUSTUM_MAX_LAYERS];
__constant__ float c_tan_half_h;
__constant__ float c_tan_half_v;
__constant__ float c_cam_near_mm;
__constant__ float c_cam_far_mm;

__device__  uint32_t get_value_2d (const uint16_t x, const uint16_t y) {
    uint32_t index = y * MAP_SIZE + x;
    return index;
}

__device__ __forceinline__
uint32_t get_value_frustum(const uint16_t iz,
                           const uint16_t iy,
                           const uint16_t ix)
{
    return (uint32_t)c_layer_offset[iz]
         + (uint32_t)iy * (uint32_t)c_Nx_per_layer[iz]
         + (uint32_t)ix;
}

__device__
float map_depth (float depth_scale, uint16_t z) {
    return depth_scale * z;
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
void kernel_deproject_depth_cuda(unsigned long long *ray_points_cam,
                                 const rs2_intrinsics* intrin,
                                 const uint16_t * depth,
                                 float depth_scale)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= (*intrin).height * (*intrin).width) return;
    int stride = blockDim.x * gridDim.x;

    for (int j = i; j < (*intrin).height * (*intrin).width; j += stride) {
        int b = j / (*intrin).width;
        int a = j - b * (*intrin).width;
        const float pixel[] = { (float)a, (float)b };

        float point[3] = {0};
        deproject_pixel_to_point_cuda(point, intrin, pixel, depth_scale * depth[j]);

        float xf = point[0] * 1000.0f;
        float yf = point[1] * 1000.0f;
        float zf = point[2] * 1000.0f;

        if (zf < c_cam_near_mm || zf >= c_cam_far_mm) continue;

        int iz = (int)((zf - c_cam_near_mm) / FRUSTUM_VOXEL_MM);
        if (iz < 0 || iz >= c_Nz) continue;

        float z_center = c_cam_near_mm + iz * FRUSTUM_VOXEL_MM + FRUSTUM_VOXEL_MM * 0.5f;
        float half_x = z_center * c_tan_half_h;
        float half_y = z_center * c_tan_half_v;

        if (xf < -half_x || xf >= half_x) continue;
        if (yf < -half_y || yf >= half_y) continue;

        int Nx = c_Nx_per_layer[iz];
        int Ny = c_Ny_per_layer[iz];

        int ix = (int)((xf + half_x) / FRUSTUM_VOXEL_MM);
        int iy = (int)((yf + half_y) / FRUSTUM_VOXEL_MM);
        if (ix < 0 || ix >= Nx) continue;
        if (iy < 0 || iy >= Ny) continue;

        int16_t x_mm = (int16_t)xf;
        int16_t y_mm = (int16_t)yf;
        int16_t z_mm = (int16_t)zf;

        // Центроид вокселя
        float cx = -half_x + (ix + 0.5f) * FRUSTUM_VOXEL_MM;
        float cy = -half_y + (iy + 0.5f) * FRUSTUM_VOXEL_MM;
        float cz = z_center;

        float dxn = xf - cx, dyn = yf - cy, dzn = zf - cz;
        uint32_t d_new = (uint32_t)(dxn*dxn + dyn*dyn + dzn*dzn);

        // Клампим dist² в 16 бит
        if (d_new > 0xFFFFu) d_new = 0xFFFFu;

        // Сдвигаем координаты в [0, 65535] (для беззнакового сравнения в atomicMin)
        uint64_t ux = (uint64_t)((int32_t)x_mm + 32768);
        uint64_t uy = (uint64_t)((int32_t)y_mm + 32768);
        uint64_t uz = (uint64_t)((int32_t)z_mm + 32768);

        // Упаковка: [63:48] dist² | [47:32] x | [31:16] y | [15:0] z
        unsigned long long packed = ((unsigned long long)d_new << 48)
                                  | ((unsigned long long)ux << 32)
                                  | ((unsigned long long)uy << 16)
                                  |  (unsigned long long)uz;

        uint32_t voxel_idx = get_value_frustum((uint16_t)iz, (uint16_t)iy, (uint16_t)ix);
        atomicMin(&ray_points_cam[voxel_idx], packed);
    }
}

__global__
void rotate_ray_points_to_world_zup(
    const unsigned long long *ray_points_cam,
    float *points_world,
    const unsigned int counter,
    const float R_cam_to_base[9],
    const float t_cam_to_base[3],
    const float R_base_to_map[9],
    const float shift[3])
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= counter) return;
    int stride = blockDim.x * gridDim.x;

    const unsigned long long MAX_PACKED = 0xFFFFFFFFFFFFFFFFULL;

    for (int j = i; j < counter; j += stride) {
        unsigned long long packed = ray_points_cam[j];
        if (packed == MAX_PACKED) {
            points_world[j*3] = points_world[1 + j*3] = points_world[2 + j*3] = 0.0f;
            continue;
        }

        unsigned long long ux = (packed >> 32) & 0xFFFFULL;
        unsigned long long uy = (packed >> 16) & 0xFFFFULL;
        unsigned long long uz =  packed        & 0xFFFFULL;

        int16_t cx = (int16_t)((int32_t)ux - 32768);
        int16_t cy = (int16_t)((int32_t)uy - 32768);
        int16_t cz = (int16_t)((int32_t)uz - 32768);

        if (cz == 0) {
            points_world[j*3] = points_world[1 + j*3] = points_world[2 + j*3] = 0.0f;
            continue;
        }

        float px = cx * 0.001f;
        float py = cy * 0.001f;
        float pz = cz * 0.001f;

        float bx = R_cam_to_base[0]*px + R_cam_to_base[3]*py + R_cam_to_base[6]*pz + t_cam_to_base[0];
        float by = R_cam_to_base[1]*px + R_cam_to_base[4]*py + R_cam_to_base[7]*pz + t_cam_to_base[1];
        float bz = R_cam_to_base[2]*px + R_cam_to_base[5]*py + R_cam_to_base[8]*pz + t_cam_to_base[2];

        float wx = R_base_to_map[0]*bx + R_base_to_map[1]*by + R_base_to_map[2]*bz;
        float wy = R_base_to_map[3]*bx + R_base_to_map[4]*by + R_base_to_map[5]*bz;
        float wz = R_base_to_map[6]*bx + R_base_to_map[7]*by + R_base_to_map[8]*bz;

        // ИСПРАВЛЕНО: + shift, как у лидара (p_world - center = wx + shift)
        points_world[j*3]     = wx + shift[0];
        points_world[1 + j*3] = wy + shift[1];
        points_world[2 + j*3] = wz + shift[2];
    }
}

// ============================================================================
// count_valid_ray_points_kernel: блочная редукция непустых packed-точек
// ============================================================================
__global__ void count_valid_ray_points_kernel(
    const unsigned long long* __restrict__ in,
    uint32_t* __restrict__ d_counts,
    uint32_t size)
{
    __shared__ int s_data[RS2_CUDA_THREADS_PER_BLOCK];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    s_data[tid] = (idx < size && in[idx] != 0xFFFFFFFFFFFFFFFFULL) ? 1 : 0;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) s_data[tid] += s_data[tid + s];
        __syncthreads();
    }
    if (tid == 0) d_counts[blockIdx.x] = s_data[0];
}

// ============================================================================
// compact_valid_ray_points_kernel: блочный compact без глобального atomicAdd
// ============================================================================
__global__ void compact_valid_ray_points_kernel(
    const unsigned long long* __restrict__ in,
    unsigned long long* __restrict__ out,
    const uint32_t* __restrict__ d_offsets,
    uint32_t size)
{
    __shared__ int s_scan[RS2_CUDA_THREADS_PER_BLOCK];
    __shared__ uint32_t s_block_offset;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    if (tid == 0) s_block_offset = d_offsets[blockIdx.x];
    __syncthreads();

    s_scan[tid] = (idx < size && in[idx] != 0xFFFFFFFFFFFFFFFFULL) ? 1 : 0;
    __syncthreads();

    for (int s = 1; s < blockDim.x; s <<= 1) {
        int val = (tid >= s) ? s_scan[tid - s] : 0;
        __syncthreads();
        if (tid >= s) s_scan[tid] += val;
        __syncthreads();
    }
    uint32_t local_pos = (tid > 0) ? (uint32_t)s_scan[tid - 1] : 0u;

    if (idx < size && in[idx] != 0xFFFFFFFFFFFFFFFFULL) {
        out[s_block_offset + local_pos] = in[idx];
    }
}

// ============================================================================
// rotate_compact_points_to_world: rotate БЕЗ sentinel-проверок, с + shift
// ============================================================================
__global__ void rotate_compact_points_to_world(
    const unsigned long long* __restrict__ ray_points_cam,
    float* __restrict__ points_world,
    uint32_t counter,
    const float R_cam_to_base[9],
    const float t_cam_to_base[3],
    const float R_base_to_map[9],
    const float shift[3])
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= counter) return;

    unsigned long long packed = ray_points_cam[i];

    unsigned long long ux = (packed >> 32) & 0xFFFFULL;
    unsigned long long uy = (packed >> 16) & 0xFFFFULL;
    unsigned long long uz =  packed        & 0xFFFFULL;

    int16_t cx = (int16_t)((int32_t)ux - 32768);
    int16_t cy = (int16_t)((int32_t)uy - 32768);
    int16_t cz = (int16_t)((int32_t)uz - 32768);

    float px = cx * 0.001f;
    float py = cy * 0.001f;
    float pz = cz * 0.001f;

    float bx = R_cam_to_base[0]*px + R_cam_to_base[3]*py + R_cam_to_base[6]*pz + t_cam_to_base[0];
    float by = R_cam_to_base[1]*px + R_cam_to_base[4]*py + R_cam_to_base[7]*pz + t_cam_to_base[1];
    float bz = R_cam_to_base[2]*px + R_cam_to_base[5]*py + R_cam_to_base[8]*pz + t_cam_to_base[2];

    float wx = R_base_to_map[0]*bx + R_base_to_map[1]*by + R_base_to_map[2]*bz;
    float wy = R_base_to_map[3]*bx + R_base_to_map[4]*by + R_base_to_map[5]*bz;
    float wz = R_base_to_map[6]*bx + R_base_to_map[7]*by + R_base_to_map[8]*bz;

    // + shift, без проверок границ — их сделает add_points_kernel_cuda
    points_world[i*3+0] = wx + shift[0];
    points_world[i*3+1] = wy + shift[1];
    points_world[i*3+2] = wz + shift[2];
}


// ============================================================================
// add_points_kernel_cuda: Калман + upper_bound + Ray Casting
// Логика полностью повторяет elevation_mapping_cupy (Z — высота)
// ============================================================================
__global__ void add_points_kernel_cuda(
     const float* points,        // [N][3] точки (x, y, z) в DLIO (метры)
     int num_points,
     const float* sensor_pos,    // [3] позиция сенсора в DLIO
     float* elevation_map,       // [7][400][400]
     float* new_map,             // [7][400][400]
     float* normal_map,          // [3][400][400]
     float resolution,           // 0.05 м
     float sensor_noise_factor,
     float mahalanobis_thresh,
     float outlier_variance,
     float wall_num_thresh,
     float max_ray_length,
     float cleanup_step,
     float min_valid_distance,
     float max_height_range,
     float cleanup_cos_thresh,
     float ramped_a, float ramped_b, float ramped_c,
     int enable_edge_sharpen,
     int enable_visibility_cleanup
 )
{
     int i = blockIdx.x * blockDim.x + threadIdx.x;
     if (i >= num_points) return;

     float x = points[i * 3 + 0];
     float y = points[i * 3 + 1];
     float z = points[i * 3 + 2];

     // === 3D-расстояние до сенсора ===
     float dx = x - sensor_pos[0];
     float dy = y - sensor_pos[1];
     float dz = z - sensor_pos[2];
     float d2 = dx*dx + dy*dy + dz*dz;

     // min_valid_distance: ближние точки отбрасываем
     if (d2 < min_valid_distance * min_valid_distance) return;

     // Exclusion Area (ramp): отсекаем нависающие
     float dist_xy = sqrtf(dx*dx + dy*dy);
     float dxy = fmaxf(dist_xy - ramped_b, 0.0f);
     if (dz > max_height_range || dz > dxy * ramped_a + ramped_c) return;

    // === Центрирование: DLIO (0,0) -> карта (200,200) ===
    float x_em = x / resolution + MAP_CENTER_F;
    float y_em = y / resolution + MAP_CENTER_F;
    int idx_x = (int)x_em;
    int idx_y = (int)y_em;
    if (idx_x <= 0 || idx_x >= MAP_LAST || idx_y <= 0 || idx_y >= MAP_LAST) return;
    int idx = idx_y * MAP_SIZE + idx_x;

    float rz = z - sensor_pos[2];
    float v_point = fmaxf(sensor_noise_factor * rz * rz, 1e-4f);

     // === Калман + upper_bound ===
     float map_h = elevation_map[0 * MAP_CELLS + idx];
     float map_v = elevation_map[1 * MAP_CELLS + idx];

        // Edge sharpening: если в ячейке уже много точек и текущая ниже оценки —
        // не обновляем, чтобы вертикальные стены оставались резкими.
     if (fabsf(map_h - z) > (map_v * mahalanobis_thresh)) {
           atomicAdd(&elevation_map[1 * MAP_CELLS + idx], outlier_variance);
       } else {

           float num_points_in_cell = new_map[4 * MAP_CELLS + idx];
           bool skip_due_to_edge =
               enable_edge_sharpen &&
               (num_points_in_cell > wall_num_thresh) &&
               (z < map_h - map_v * mahalanobis_thresh / fmaxf(num_points_in_cell, 1.0f));

           if (!skip_due_to_edge) {
               float new_h = (map_h * v_point + z * map_v) / (map_v + v_point);
               float new_v = (map_v * v_point) / (map_v + v_point);
               atomicAdd(&new_map[0 * MAP_CELLS + idx], new_h);
               atomicAdd(&new_map[1 * MAP_CELLS + idx], new_v);
               atomicAdd(&new_map[2 * MAP_CELLS + idx], 1.0f);

               elevation_map[2 * MAP_CELLS + idx] = 1.0f;
               elevation_map[4 * MAP_CELLS + idx] = 0.0f;
               elevation_map[5 * MAP_CELLS + idx] = new_h;
               elevation_map[6 * MAP_CELLS + idx] = 0.0f;
           }
       }

     // === Ray Casting (2D Брезенхем по X/Y) ===
     if (!enable_visibility_cleanup) return;

     // FIX 4: 3D-луч, как ray_vector в EMC. Шаг = resolution/sqrt(2).
     float rvx = x - sensor_pos[0];
     float rvy = y - sensor_pos[1];
     float rvz = z - sensor_pos[2];
     float ray_len = sqrtf(rvx*rvx + rvy*rvy + rvz*rvz);
     if (ray_len < 1e-6f) return;
     float ray_x = rvx / ray_len;
     float ray_y = rvy / ray_len;
     float ray_z_dir = rvz / ray_len;
     float ray_len_clamped = fminf(ray_len, max_ray_length);
     const float ray_step = resolution / 1.41421356f;

     int last_nidx = -1;
     for (float s = ray_step; s < ray_len_clamped; s += ray_step) {
         float nx = sensor_pos[0] + ray_x * s;
         float ny = sensor_pos[1] + ray_y * s;
         float nz = sensor_pos[2] + ray_z_dir * s;

         int cx = (int)(nx / resolution + MAP_CENTER_F);  // FIX 6: центр в 200
         int cy = (int)(ny / resolution + MAP_CENTER_F);
         if (cx <= 0 || cx >= MAP_LAST || cy <= 0 || cy >= MAP_LAST) continue;
         int nidx = cy * MAP_SIZE + cx;

         if (last_nidx == nidx) continue;  // пропускаем повторные попадания в ту же ячейку
         last_nidx = nidx;

         float ddx = x - nx, ddy = y - ny, ddz = z - nz;
         if (ddx*ddx + ddy*ddy + ddz*ddz < 0.1f) continue;

         float nmap_valid = elevation_map[2 * MAP_CELLS + nidx];
         float nmap_upper = elevation_map[5 * MAP_CELLS + nidx];
         float nmap_is_upper = elevation_map[6 * MAP_CELLS + nidx];

         if (nmap_valid < 0.5f) {
             if (nz < nmap_upper || nmap_is_upper < 0.5f) {
                 elevation_map[5 * MAP_CELLS + nidx] = nz;
                 elevation_map[6 * MAP_CELLS + nidx] = 1.0f;
             }
             continue;
         }

         float nmap_h = elevation_map[0 * MAP_CELLS + nidx];
         float nmap_v = elevation_map[1 * MAP_CELLS + nidx];
         float nmap_t = elevation_map[4 * MAP_CELLS + nidx];

         if (nmap_h > nz + 0.01f - fminf(nmap_v, 1.0f) * 0.05f) {
             float mnx = normal_map[0 * MAP_CELLS + nidx];
             float mny = normal_map[1 * MAP_CELLS + nidx];
             float mnz = normal_map[2 * MAP_CELLS + nidx];

            if (fabsf(ray_x*mnx + ray_y*mny + ray_z_dir*mnz) < cleanup_cos_thresh) continue;

             // FIX 1 (см. P0-1): new_map[3] теперь не zeroed → защита стен работает.
             float npts = new_map[3 * MAP_CELLS + nidx];
             if (npts > wall_num_thresh && nmap_t < 1.0f) continue;

             // FIX 5: в EMC cleanup = -step / (ray_len / max_ray).
             // В порте было умножение вместо деления → очистка шла в обратную сторону.
             float cleanup_factor = cleanup_step / fmaxf(ray_len_clamped / max_ray_length, 0.01f);
             atomicAdd(&elevation_map[2 * MAP_CELLS + nidx], -cleanup_factor);
             atomicAdd(&elevation_map[1 * MAP_CELLS + nidx], outlier_variance);

             if (nz < nmap_upper || nmap_is_upper < 0.5f) {
                 elevation_map[5 * MAP_CELLS + nidx] = nz;
                 elevation_map[6 * MAP_CELLS + nidx] = 1.0f;
             }
         }
     }
}

__global__ void average_map_kernel_cuda(
    float* new_map, float* elevation_map,
    float max_variance, float initial_variance
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= MAP_CELLS) return;

    float new_h = new_map[0 * MAP_CELLS + i];
    float new_v = new_map[1 * MAP_CELLS + i];
    float new_cnt = new_map[2 * MAP_CELLS + i];

    if (new_cnt > 0) {
        if (new_v / new_cnt > max_variance) {
            elevation_map[0 * MAP_CELLS + i] = 0.0f;
            elevation_map[1 * MAP_CELLS + i] = initial_variance;
            elevation_map[2 * MAP_CELLS + i] = 0.0f;
        } else {
            elevation_map[0 * MAP_CELLS + i] = new_h / new_cnt;
            elevation_map[1 * MAP_CELLS + i] = new_v / new_cnt;
            elevation_map[2 * MAP_CELLS + i] = 1.0f;
        }
        } else {
        // Как в оригинале: если ячейка невалидна и новых точек нет — сброс.
        float v = elevation_map[2 * MAP_CELLS + i];
        if (v < 0.5f) {
         elevation_map[0 * MAP_CELLS + i] = 0.0f;
         elevation_map[1 * MAP_CELLS + i] = initial_variance;
         elevation_map[2 * MAP_CELLS + i] = 0.0f;
         }
        }
        // Не даём L_IS_VALID уйти в минус после visibility cleanup.
        float v_ = elevation_map[2 * MAP_CELLS + i];
        if (v_ < 0.0f) {
        elevation_map[2 * MAP_CELLS + i] = 0.0f;
        }
}

__global__ void normal_filter_kernel_cuda(
        const float* __restrict__ dilated,        // 1 слой (dilated upper_bound)
        const float* __restrict__ elevation_map,  // 7 слоёв (для is_valid / is_upper)
        float* __restrict__ normal_map,
        float resolution
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x <= 0 || x >= MAP_LAST - 1 || y <= 0 || y >= MAP_LAST - 1) return;

    int idx = y * MAP_SIZE + x;
    float valid = elevation_map[2 * MAP_CELLS + idx];
    if (valid < 0.5f) return;

    int idx_x = y * MAP_SIZE + (x + 1);
    int idx_y = (y + 1) * MAP_SIZE + x;
    if (idx_x >= MAP_CELLS || idx_y >= MAP_CELLS) return;

    float h = dilated[idx];
    float dzdx = dilated[idx_x] - h;
    float dzdy = dilated[idx_y] - h;

    float nx = -dzdy / resolution;
    float ny = -dzdx / resolution;
    float nz = 1.0f;
    float norm = sqrtf(nx*nx + ny*ny + 1.0f);

    normal_map[0 * MAP_CELLS + idx] = nx / norm;
    normal_map[1 * MAP_CELLS + idx] = ny / norm;
    normal_map[2 * MAP_CELLS + idx] = nz / norm;
}

// ============================================================================
// error_counting_kernel_cuda: Считает среднюю ошибку по валидным точкам
// Вызывается ДО add_points_kernel_cuda. Логика из elevation_mapping_cupy.
// ============================================================================
__global__ void error_counting_kernel_cuda(
    const float* points, int num_points,
    const float* sensor_pos,
    float* elevation_map,
    float* error_sum, float* error_cnt,
    float* new_map,
    float resolution,
    float sensor_noise_factor,
    float mahalanobis_thresh,
    float outlier_variance,
    float traversability_inlier,
    float min_valid_distance,
    float max_height_range,
    float ramped_a, float ramped_b, float ramped_c
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_points) return;

    float x = points[i * 3 + 0];
    float y = points[i * 3 + 1];
    float z = points[i * 3 + 2];

    float dx = x - sensor_pos[0];
    float dy = y - sensor_pos[1];
    float dz = z - sensor_pos[2];
    float d2 = dx*dx + dy*dy + dz*dz;

    if (d2 < min_valid_distance * min_valid_distance) return;

    float dist_xy = sqrtf(dx*dx + dy*dy);
    float dxy = fmaxf(dist_xy - ramped_b, 0.0f);
    if (dz > max_height_range || dz > dxy * ramped_a + ramped_c) return;

    float x_em = x / resolution + MAP_CENTER;
    float y_em = y / resolution + MAP_CENTER;
    int idx_x = (int)x_em;
    int idx_y = (int)y_em;
    if (idx_x <= 0 || idx_x >= MAP_LAST || idx_y <= 0 || idx_y >= MAP_LAST) return;
    int idx = idx_y * MAP_SIZE + idx_x;

    float map_h = elevation_map[0 * MAP_CELLS + idx];
    float map_v = elevation_map[1 * MAP_CELLS + idx];
    float map_valid = elevation_map[2 * MAP_CELLS + idx];
    float map_trav = elevation_map[3 * MAP_CELLS + idx];

    // Условия: валидно, дисперсия мала, траверсабельность высокая
    if (map_valid > 0.5f &&
        (fabsf(map_h - z) < (map_v * mahalanobis_thresh)) &&
        map_v < outlier_variance / 2.0f &&
        map_trav > traversability_inlier) {

        float e = z - map_h;
        atomicAdd(&error_sum[0], e);
        atomicAdd(&error_cnt[0], 1.0f);
        atomicAdd(&new_map[3 * MAP_CELLS + idx], 1.0f);
    }
    atomicAdd(&new_map[4 * MAP_CELLS + idx], 1.0f);
}

// ============================================================================
// shift_map_z_kernel_cuda: Сдвигает слои L_HEIGHT и L_UPPER_BOUND на delta_z
// ============================================================================
// Сдвиг по Z: модифицирует L_HEIGHT и L_UPPER_BOUND.
// Вызывается при каждом изменении высоты робота (delta_z != 0).
__global__ void shift_map_z_kernel(
    float* __restrict__ elevation_map,
    float  delta_z)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= MAP_CELLS) return;

    // L_HEIGHT (0) и L_UPPER_BOUND (5) сдвигаются,
    // L_VARIANCE, L_IS_VALID, L_TRAVERS, L_TIME, L_IS_UPPER — нет.
    elevation_map[0 * MAP_CELLS + i] += delta_z;
    elevation_map[5 * MAP_CELLS + i] += delta_z;
}

__global__ void drift_comp_kernel(float* elevation_map, float delta_z) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= MAP_CELLS) return;
    elevation_map[0 * MAP_CELLS + i] += delta_z;
}


// Ядро заполнения «хвоста» после сдвига по XY.
// Используется как замена ручных циклов по краям (быстрее и без ошибок).
// Заполняет полосу [start_row, end_row) × [0, 400) значениями по слою.
__global__ void fill_vertical_band_kernel(
    float* __restrict__ elevation_map,
    int    start_col,
    int    end_col,
    float  val_height,
    float  val_variance,
    float  val_is_valid,
    float  val_traversability,
    float  val_time,
    float  val_upper_bound,
    float  val_is_upper)
{
    int x = start_col + blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= end_col || y >= MAP_SIZE) return;

    int idx = y * MAP_SIZE + x;
    elevation_map[0 * MAP_CELLS + idx] = val_height;
    elevation_map[1 * MAP_CELLS + idx] = val_variance;
    elevation_map[2 * MAP_CELLS + idx] = val_is_valid;
    elevation_map[3 * MAP_CELLS + idx] = val_traversability;
    elevation_map[4 * MAP_CELLS + idx] = val_time;
    elevation_map[5 * MAP_CELLS + idx] = val_upper_bound;
    elevation_map[6 * MAP_CELLS + idx] = val_is_upper;
}

// Аналогичное ядро для горизонтальных полос (по Y).
__global__ void fill_horizontal_band_kernel(
    float* __restrict__ elevation_map,
    int    start_row,
    int    end_row,          // исключительно
    float  val_height,
    float  val_variance,
    float  val_is_valid,
    float  val_traversability,
    float  val_time,
    float  val_upper_bound,
    float  val_is_upper)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = start_row + blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= MAP_SIZE || y >= end_row) return;

    int idx = y * MAP_SIZE + x;
    elevation_map[0 * MAP_CELLS + idx] = val_height;
    elevation_map[1 * MAP_CELLS + idx] = val_variance;
    elevation_map[2 * MAP_CELLS + idx] = val_is_valid;
    elevation_map[3 * MAP_CELLS + idx] = val_traversability;
    elevation_map[4 * MAP_CELLS + idx] = val_time;
    elevation_map[5 * MAP_CELLS + idx] = val_upper_bound;
    elevation_map[6 * MAP_CELLS + idx] = val_is_upper;
}


__global__ void fill_normal_vertical_band_kernel(
    float* __restrict__ normal_map, int start_col, int end_col)
{
    int x = start_col + blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= end_col || y >= MAP_SIZE) return;
    int idx = y * MAP_SIZE + x;
    normal_map[0 * MAP_CELLS + idx] = 0.0f;
    normal_map[1 * MAP_CELLS + idx] = 0.0f;
    normal_map[2 * MAP_CELLS + idx] = 0.0f;
}

__global__ void fill_normal_horizontal_band_kernel(
    float* __restrict__ normal_map, int start_row, int end_row)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = start_row + blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= MAP_SIZE || y >= end_row) return;
    int idx = y * MAP_SIZE + x;
    normal_map[0 * MAP_CELLS + idx] = 0.0f;
    normal_map[1 * MAP_CELLS + idx] = 0.0f;
    normal_map[2 * MAP_CELLS + idx] = 0.0f;
}

// ============================================================================
// overlap_clearance_kernel: Очищает ячейки, высота которых далеко от робота
// (для многоуровневых сред: робот поднялся по лестнице — старый этаж сбрасывается)
// ============================================================================
__global__ void overlap_clearance_kernel(
    float* elevation_map,
    float robot_z,           // текущая высота робота
    float clear_range_z,     // 2.0 м — диапазон очистки
    float initial_variance,  // 1000.0 — начальная дисперсия
    int center_x, int center_y,
    int clear_range_cells    // 4.0 м / 0.05 = 80 ячеек
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = (2 * clear_range_cells + 1) * (2 * clear_range_cells + 1);
    if (i >= total) return;

    int ix = i % (2 * clear_range_cells + 1);
    int iy = i / (2 * clear_range_cells + 1);

    int x = center_x - clear_range_cells + ix;
    int y = center_y - clear_range_cells + iy;

    if (x <= 0 || x >= MAP_LAST || y <= 0 || y >= MAP_LAST) return;

    int idx = y * MAP_SIZE + x;

    float h_min = robot_z - clear_range_z;
    float h_max = robot_z + clear_range_z;

    float h = elevation_map[0 * MAP_CELLS + idx];
    if (h < h_min || h > h_max) {
        elevation_map[0 * MAP_CELLS + idx] = 0.0f;
        elevation_map[1 * MAP_CELLS + idx] = initial_variance;
        elevation_map[2 * MAP_CELLS + idx] = 0.0f;
    }

    float ub = elevation_map[5 * MAP_CELLS + idx];
    if (ub < h_min || ub > h_max) {
        elevation_map[5 * MAP_CELLS + idx] = 0.0f;
        elevation_map[6 * MAP_CELLS + idx] = 0.0f;
    }
}

// ============================================================================
// min_filter_kernel: Заполняет невалидные ячейки минимальной высотой в окне
// ============================================================================
__global__ void min_filter_kernel(
    float* elevation_map,
    float* filtered_height,     // выходной буфер
    unsigned char* filtered_mask, // выходной флаг
    int dilation_size           // 3 (окно 7x7)
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= MAP_SIZE || y >= MAP_SIZE) return;

    int idx = y * MAP_SIZE + x;
    float h = elevation_map[0 * MAP_CELLS + idx];
    float valid = elevation_map[2 * MAP_CELLS + idx];

    if (valid > 0.5f) {
        filtered_height[idx] = h;
        filtered_mask[idx] = 1;
        return;
    }

    // Ищем минимальную валидную высоту в окне
    float min_val = 1e6f;
    for (int dy = -dilation_size; dy <= dilation_size; dy++) {
        for (int dx = -dilation_size; dx <= dilation_size; dx++) {
            int nx = x + dx;
            int ny = y + dy;
            if (nx <= 0 || nx >= MAP_LAST || ny <= 0 || ny >= MAP_LAST) continue;
            int nidx = ny * MAP_SIZE + nx;
            float nh = elevation_map[0 * MAP_CELLS + nidx];
            float nvalid = elevation_map[2 * MAP_CELLS + nidx];
            if (nvalid > 0.5f && nh < min_val) min_val = nh;
        }
    }

    if (min_val < 1e5f) {
        filtered_height[idx] = min_val;
        filtered_mask[idx] = 1;
    } else {
        filtered_height[idx] = 0.0f;
        filtered_mask[idx] = 0;
    }
}

// ============================================================================
// dilation_filter_kernel: заполнение невалидных ячеек upper_bound
// ближайшим валидным upper_bound в окне [−dilation_size, +dilation_size].
// Аналог elevation_mapping_cupy::dilation_filter_kernel.
// Вход: layer 5 (upper_bound), маска = layer 2 + layer 6.
// Выход: временный буфер (НЕ перезаписывает layer 5).
// ============================================================================
__global__ void dilation_filter_kernel(
    const float* __restrict__ upper_bound,   // layer 5
    const float* __restrict__ is_valid,       // layer 2
    const float* __restrict__ is_upper,       // layer 6
    float* __restrict__ dilated,              // выходной буфер
    int dilation_size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= MAP_SIZE || y >= MAP_SIZE) return;

    int idx = y * MAP_SIZE + x;
    // Маска = is_valid + is_upper (как в EMC: map[2] + map[6] > 0.5)
    float mask = is_valid[idx] + is_upper[idx];

    // По умолчанию — копия
    dilated[idx] = upper_bound[idx];

    if (mask < 0.5f) {
        float best_dist = 1e9f;
        float best_val  = 0.0f;
        bool  found     = false;

        for (int dy = -dilation_size; dy <= dilation_size; dy++) {
            for (int dx = -dilation_size; dx <= dilation_size; dx++) {
                int nx = x + dx;
                int ny = y + dy;
                if (nx <= 0 || nx >= MAP_LAST || ny <= 0 || ny >= MAP_LAST) continue;
                int nidx = ny * MAP_SIZE + nx;

                float nmask = is_valid[nidx] + is_upper[nidx];
                if (nmask < 0.5f) continue;

                // Манхэттенское расстояние (как в EMC: dx+dy без abs —
                // оставляем как есть для эквивалентности; при желании
                // заменить на fabsf(dx)+fabsf(dy))
                float d = fabsf((float)dx) + fabsf((float)dy);
                if (d < best_dist) {
                    best_dist = d;
                    best_val  = upper_bound[nidx];
                    found     = true;
                }
            }
        }
        if (found) dilated[idx] = best_val;
    }
}

// ============================================================================
// smooth_filter_kernel: Box blur 3x3 (сглаживание высот)
// ============================================================================
__global__ void smooth_filter_kernel(
    float* input, float* output
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x <= 0 || x >= MAP_LAST || y <= 0 || y >= MAP_LAST) return;

    float sum = 0.0f;
    int count = 0;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int nx = x + dx;
            int ny = y + dy;
            if (nx <= 0 || nx >= MAP_LAST || ny <= 0 || ny >= MAP_LAST) continue;
            int nidx = ny * MAP_SIZE + nx;
            float v = input[nidx];
            if (v > 0.0f) {  // только валидные
                sum += v;
                count++;
            }
        }
    }
    output[y * MAP_SIZE + x] = (count > 0) ? (sum / count) : 0.0f;
}

// ============================================================================
// combine_validity_kernel: is_valid + is_upper > 0.5 → 1.0, иначе 0.0
// Аналог (elevation_map[2] + elevation_map[6]) в EMC.
// ============================================================================
__global__ void combine_validity_kernel(
    const float* __restrict__ is_valid,
    const float* __restrict__ is_upper,
    float* __restrict__ out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= MAP_CELLS) return;
    out[i] = (is_valid[i] + is_upper[i] > 0.5f) ? 1.0f : 0.0f;
}

// ============================================================================
// getSlopeOfPoints: Least Squares для уклона в плоскости XY
// (вход: heights, xs, ys; выход: slopeX, slopeY)
// ============================================================================
__device__ void getSlopeOfPoints(
    const float* heights, const float* xs, const float* ys, int n,
    float& slopeX, float& slopeY
) {
    if (n < 3) { slopeX = 0.0f; slopeY = 0.0f; return; }

    float sumX = 0, sumY = 0, sumH = 0;
    float sumX2 = 0, sumY2 = 0, sumXY = 0;
    float sumXH = 0, sumYH = 0;

    for (int i = 0; i < n; ++i) {
        float x = xs[i], y = ys[i], h = heights[i];
        sumX += x; sumY += y; sumH += h;
        sumX2 += x * x; sumY2 += y * y; sumXY += x * y;
        sumXH += x * h; sumYH += y * h;
    }

    float det = sumX2 * (sumY2 * n - sumY * sumY)
              - sumXY * (sumXY * n - sumY * sumX)
              + sumX  * (sumXY * sumY - sumY2 * sumX);

    if (fabsf(det) < 1e-10f) { slopeX = 0.0f; slopeY = 0.0f; return; }

    float detA = sumXH * (sumY2 * n - sumY * sumY)
               - sumXY * (sumYH * n - sumY * sumH)
               + sumX  * (sumYH * sumY - sumY2 * sumH);

    float detB = sumX2 * (sumYH * n - sumY * sumH)
               - sumXH * (sumXY * n - sumY * sumX)
               + sumX  * (sumXY * sumH - sumYH * sumX);

    slopeX = detA / det;
    slopeY = detB / det;
}


__global__ void heightmap_to_slope_cuda(
    const float* __restrict__ dev_heightmap,
    const float* __restrict__ dev_is_valid,
    float* __restrict__ dev_slope,
    int estimation_size
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= Map_X || y >= Map_Y) return;

    // --- Локальный slope (окно 3×3) ---
    float hl[9], xl[9], yl[9];
    int nl = 0;
    float raw_min = 1e30f, raw_max = -1e30f;

    for (int dr = -1; dr <= 1; ++dr) {
        for (int dc = -1; dc <= 1; ++dc) {
            int nr = y + dr;
            int nc = x + dc;
            if (nr < 0 || nr >= Map_Y || nc < 0 || nc >= Map_X) continue;

            float h = dev_heightmap[nr * Map_X + nc];
            if (dev_is_valid[nr * Map_X + nc] < 0.5f) continue;  // unknown

            hl[nl] = h;
            xl[nl] = dc * SLOPE_CELL_SIZE;
            yl[nl] = dr * SLOPE_CELL_SIZE;
            nl++;

            if (h < raw_min) raw_min = h;
            if (h > raw_max) raw_max = h;
        }
    }

    if (nl < 3) {
        dev_slope[y * Map_X + x] = -1.0f;
        return;
    }

    float sx, sy;
    getSlopeOfPoints(hl, xl, yl, nl, sx, sy);
    float localSlope = sqrtf(sx * sx + sy * sy);

    float heightRange = (raw_max - raw_min) / SLOPE_CELL_SIZE;

    // --- Глобальный slope ---
    float globalSlope = 0.0f;
    if (estimation_size > 1) {
        const int MAX_N = 121;
        float hg[MAX_N], xg[MAX_N], yg[MAX_N];
        int ng = 0;

        for (int dr = -estimation_size; dr <= estimation_size; ++dr) {
            for (int dc = -estimation_size; dc <= estimation_size; ++dc) {
                if (ng >= MAX_N) break;
                int nr = y + dr;
                int nc = x + dc;
                if (nr < 0 || nr >= Map_Y || nc < 0 || nc >= Map_X) continue;

                float h = dev_heightmap[nr * Map_X + nc];
                if (dev_is_valid[nr*Map_X+nc] < 0.5f) continue;

                hg[ng] = h;
                xg[ng] = dc * SLOPE_CELL_SIZE;
                yg[ng] = dr * SLOPE_CELL_SIZE;
                ng++;
            }
        }

        if (ng >= 3) {
            float gx, gy;
            getSlopeOfPoints(hg, xg, yg, ng, gx, gy);
            globalSlope = sqrtf(gx * gx + gy * gy);
        }
    }

    // --- Итог = max(локальный, глобальный, размах) ---
    float result = localSlope;
    if (globalSlope > result) result = globalSlope;
    if (heightRange > result) result = heightRange;

    dev_slope[y * Map_X + x] = result;
}

__global__ void slope_to_costmap_cuda(
    const float* __restrict__ dev_slope,
    const float* __restrict__ dev_is_valid,
    uint8_t* __restrict__ dev_costmap,
    float max_slope, float flat_slope
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= Map_X || y >= Map_Y) return;


    int idx = y * Map_X + x;
    if (dev_is_valid[idx] < 0.5f) {
        dev_costmap[idx] = 255;   // unknown
        return;
    }
    float s = dev_slope[idx];
    uint8_t cost;

    if (s < 0.0f) cost = 255;              // unknown
    else if (s > max_slope) cost = 254;    // lethal
    else if (s < flat_slope) cost = 0;     // free
    else {
        float norm = (s - flat_slope) / (max_slope - flat_slope);
        float c = 252.0f * norm;
        cost = (uint8_t)(c < 1.0f ? 1.0f : (c > 252.0f ? 252.0f : c));
    }

    dev_costmap[idx] = cost;
}


// Ядро для поиска контуров в 2д карте
__global__ void findContoursKernel_2d(uint8_t* data) {
    uint16_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint16_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= MAP_SIZE || y >= MAP_SIZE) return;

    uint32_t idx = y * MAP_SIZE + x;
    uint8_t cell = data[idx];

    if (cell > 252) return;

    int16_t dx[4] = {0, 0, -1, 1};
    int16_t dy[4] = {-1, 1, 0, 0};  // <-- dx/dy (плоскость XY)

    for (int8_t i = 0; i < 4; ++i) {
        int16_t nx = x + dx[i];
        int16_t ny = y + dy[i];

        if (nx < 0 || nx >= MAP_SIZE || ny < 0 || ny >= MAP_SIZE) {
            data[idx] = 254;
            return;
        }

        uint8_t neighbor = data[ny * MAP_SIZE + nx];

        if (neighbor == 255) {
            data[idx] = 254;
            return;
        }
    }
}

__global__ void inflate_costmap_cuda(
    const uint8_t* __restrict__ dev_input,
    uint8_t* __restrict__ dev_output,
    int inscribed_radius,
    int inflation_radius,
    float cost_scaling_factor
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= Map_X || y >= Map_Y) return;

    int idx = y * MAP_SIZE + x;  //

    if (dev_input[idx] == 254) {
        dev_output[idx] = 254;
        return;
    }

    if (dev_input[idx] == 255) {
        dev_output[idx] = 255;
        return;
    }

    int min_dist_sq = (inflation_radius + 1) * (inflation_radius + 1);
    for (int dy = -inflation_radius; dy <= inflation_radius; dy++) {
        for (int dx = -inflation_radius; dx <= inflation_radius; dx++) {
            int nx = x + dx;
            int ny = y + dy;
            if (nx < 0 || nx >= Map_X || ny < 0 || ny >= Map_Y) continue;
            if (dev_input[ny * Map_X + nx] == 254) {
                int d_sq = dx * dx + dy * dy;
                if (d_sq < min_dist_sq) min_dist_sq = d_sq;
            }
        }
    }

    if (min_dist_sq > inflation_radius * inflation_radius) {
        dev_output[idx] = dev_input[idx];
        return;
    }

    float dist = sqrtf((float)min_dist_sq);
    if (dist <= (float)inscribed_radius) {
        dev_output[idx] = 253;
    } else {
        float dist_rel = dist - (float)inscribed_radius;
        float max_dist = (float)(inflation_radius - inscribed_radius);
        if (max_dist < 1e-6f) max_dist = 1e-6f;
        float val = 252.0f * expf(-cost_scaling_factor * (dist_rel / max_dist));
        dev_output[idx] = (uint8_t)(val < 1.0f ? 1.0f : (val > 252.0f ? 252.0f : val));
    }
}


// ============================================================================
// pack_heightmap_minmax_kernel: Упаковка heightmap float[400x400]
// в packed float[400x200] (2x uint16 в float), плоскость XY
// ============================================================================

__global__ void pack_heightmap_minmax_kernel(
    const float* __restrict__ heightmap,      // [400][400]
    const float* __restrict__ is_valid,       // [400][400] — слой L_IS_VALID
    float*       __restrict__ out_packed,     // [400][200]
    int*         __restrict__ d_minmax)
{
    int texCol = blockIdx.x * blockDim.x + threadIdx.x;
    int row    = blockIdx.y * blockDim.y + threadIdx.y;

    int lmin = 32767;
    int lmax = -32768;

    if (texCol < (MAP_SIZE / 2) && row < MAP_SIZE) {
        int col0 = texCol * 2;
        int col1 = col0 + 1;

        float fa = heightmap[row * MAP_SIZE + col0];
        float fb = heightmap[row * MAP_SIZE + col1];
        float va = is_valid[row * MAP_SIZE + col0];
        float vb = is_valid[row * MAP_SIZE + col1];

        // ✅ Отсекаем по is_valid, а не по знаку высоты
        int16_t a_raw = (va < 0.5f) ? -32768 : (int16_t)(fa * 1000.0f);
        int16_t b_raw = (vb < 0.5f) ? -32768 : (int16_t)(fb * 1000.0f);

        uint32_t packed = ((uint32_t)(uint16_t)a_raw << 16) | (uint32_t)(uint16_t)b_raw;
        out_packed[row * (MAP_SIZE / 2) + texCol] = __uint_as_float(packed);

        if (a_raw != -32768) { lmin = min(lmin, (int)a_raw); lmax = max(lmax, (int)a_raw); }
        if (b_raw != -32768) { lmin = min(lmin, (int)b_raw); lmax = max(lmax, (int)b_raw); }
    }

    // --- Редукция min/max (int) ---
    __shared__ int sMin[16][16];
    __shared__ int sMax[16][16];

    // КРИТИЧЕСКИЙ ФИКС: Абсолютно ВСЕ 256 потоков блока (даже если они за границей сетки 200x400)
    // обязаны инициализировать shared-память и дойти до барьера __syncthreads()
    sMin[threadIdx.y][threadIdx.x] = lmin;
    sMax[threadIdx.y][threadIdx.x] = lmax;

    // Теперь дедлок исключен, так как барьер выполняют все потоки
    __syncthreads();

    // Свертка строк внутри блока (по оси X)
    for (int s = 8; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (sMin[threadIdx.y][threadIdx.x + s] < sMin[threadIdx.y][threadIdx.x])
                sMin[threadIdx.y][threadIdx.x] = sMin[threadIdx.y][threadIdx.x + s];
            if (sMax[threadIdx.y][threadIdx.x + s] > sMax[threadIdx.y][threadIdx.x])
                sMax[threadIdx.y][threadIdx.x] = sMax[threadIdx.y][threadIdx.x + s];
        }
        __syncthreads();
    }

    // Свертка столбцов внутри блока (по оси Y)
    for (int s = 8; s > 0; s >>= 1) {
        if (threadIdx.y < s) {
            if (sMin[threadIdx.y + s][threadIdx.x] < sMin[threadIdx.y][threadIdx.x])
                sMin[threadIdx.y][threadIdx.x] = sMin[threadIdx.y + s][threadIdx.x];
            if (sMax[threadIdx.y + s][threadIdx.x] > sMax[threadIdx.y][threadIdx.x])
                sMax[threadIdx.y][threadIdx.x] = sMax[threadIdx.y + s][threadIdx.x];
        }
        __syncthreads();
    }

    // Финальная атомарная запись результатов блока
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        // Делаем атомик только если блок обработал хотя бы одну реальную точку высоты
        if (sMin[0][0] != 32767) {
            atomicMin(&d_minmax[0], sMin[0][0]);
        }
        if (sMax[0][0] != -32768) {
            atomicMax(&d_minmax[1], sMax[0][0]);
        }
    }
}

// ============================================================================
// colorize_costmap_kernel: раскраска costmap (uint8) в RGBA (uint32)
// Палитра взята из CostmapColorizer::createColorPalette
// ============================================================================
__global__ void colorize_costmap_kernel(
    const uint8_t* __restrict__ costmap,
    uint32_t* __restrict__ rgba_out,
    int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    uint8_t c = costmap[idx];

    uint8_t r, g, b;
    if (c == 0) {
        r = 192; g = 192; b = 192;              // free — светло-серый
    } else if (c <= 100) {
        int dist = abs((int)c - 49);             // blue -> red, центр 49
        r = (uint8_t)((255 * dist) / 49);
        b = 255 - r;
        g = 0;
    } else if (c <= 127) {
        r = 0; g = 255; b = 0;                   // illegal positive — зелёный
    } else if (c <= 252) {
        r = 255;                                  // illegal negative — красно-жёлтый
        g = (uint8_t)((255 * (c - 128)) / (252 - 128));
        b = 0;
    } else if (c == 253) {
        r = 0; g = 255; b = 255;                 // inscribed — циан
    } else if (c == 254) {
        r = 255; g = 0; b = 255;                 // lethal — пурпурный
    } else { // 255
        r = 0x70; g = 0x89; b = 0x86;            // unknown — серо-зелёный
    }

    // RGBA8888 (R в младшем байте)
    rgba_out[idx] = (uint32_t(255) << 24) |
                    (uint32_t(b)   << 16) |
                    (uint32_t(g)   << 8)  |
                     uint32_t(r);
}

__global__ void costmap_to_traversability_kernel(
    const uint8_t* __restrict__ dev_costmap,
    float* __restrict__ elevation_map)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= MAP_CELLS) return;
    int x = i % MAP_SIZE;
    int y = i / MAP_SIZE;
    if (x < 3 || x >= MAP_LAST - 3 || y < 3 || y >= MAP_LAST - 3) return;   // ← граница

    uint8_t c = dev_costmap[i];
    float trav = (c >= 253) ? 0.0f : 1.0f - (float)c / 252.0f;
    elevation_map[3 * MAP_CELLS + i] = trav;
}

// FIX 9: рост дисперсии во времени, как update_variance в EMC.
__global__ void update_variance_kernel(
    float* elevation_map, float time_variance, float max_variance)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= MAP_CELLS) return;
    if (elevation_map[2 * MAP_CELLS + i] < 0.5f) return;
    float v = elevation_map[1 * MAP_CELLS + i] + time_variance;
    elevation_map[1 * MAP_CELLS + i] = fminf(v, max_variance);
}

// ============================================================================
// update_time_kernel: L_TIME += dt для всех валидных ячеек.
// Аналог ElevationMap.update_time() в elevation_mapping_cupy.
// ============================================================================
__global__ void update_time_kernel(float* elevation_map, float dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= MAP_CELLS) return;
    // Растим время только у валидных ячеек (как в оригинале:
    // elevation_map[4] += time_interval * valid).
    float valid = elevation_map[2 * MAP_CELLS + i];
    elevation_map[4 * MAP_CELLS + i] += dt * (valid > 0.5f ? 1.0f : 0.0f);
}

extern "C" __host__
void update_time_cuda(float* dev_elevation_map, float dt)
{
    int blocks = (MAP_CELLS + 255) / 256;
    update_time_kernel<<<blocks, 256>>>(dev_elevation_map, dt);
    cudaDeviceSynchronize();
}

extern "C" __host__
void shift_map_z_cuda(float* dev_elevation_map, float delta_z)
{
    if (delta_z == 0.0f) return;
    int blocks = (MAP_CELLS + 255) / 256;
    shift_map_z_kernel<<<blocks, 256>>>(dev_elevation_map, delta_z);
    cudaError_t result = cudaDeviceSynchronize();
    assert(result == cudaSuccess);
}

extern "C" __host__
void shift_map_xy_cuda(
    float* dev_elevation_map,
    float* dev_normal_map,
    float* temp,
    int    shift_x,
    int    shift_y)
{
    if (shift_x == 0 && shift_y == 0) return;
    if (temp == nullptr) {
        std::cerr << "[shift_map_xy_cuda] temp is nullptr!" << std::endl;
        return;
    }

    cudaError_t result;

    // Клампим сдвиг
    if (shift_x >  Map_X) shift_x =  Map_X;
    if (shift_x < -Map_X) shift_x = -Map_X;
    if (shift_y >  Map_Y) shift_y =  Map_Y;
    if (shift_y < -Map_Y) shift_y = -Map_Y;

    int src_x0 = (shift_x > 0) ? 0         : -shift_x;
    int dst_x0 = (shift_x > 0) ? shift_x   : 0;
    int copy_w = (shift_x > 0) ? Map_X - shift_x : Map_X + shift_x;

    int src_y0 = (shift_y > 0) ? 0         : -shift_y;
    int dst_y0 = (shift_y > 0) ? shift_y   : 0;
    int copy_h = (shift_y > 0) ? Map_Y - shift_y : Map_Y + shift_y;

    if (copy_w <= 0 || copy_h <= 0) {
        float empty[7] = {0.0f, 1000.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f};
        dim3 block(16, 16);
        dim3 grid((Map_Y + 15) / 16, (Map_Y + 15) / 16);
        fill_vertical_band_kernel<<<grid, block>>>(
            dev_elevation_map, 0, Map_X,
            empty[0], empty[1], empty[2], empty[3], empty[4], empty[5], empty[6]);
        cudaDeviceSynchronize();
        return;
    }

    for (int layer = 0; layer < 7; ++layer) {
        float* layer_ptr = dev_elevation_map + layer * MAP_CELLS;
        result = cudaMemcpy(temp, layer_ptr, MAP_CELLS * sizeof(float), cudaMemcpyDeviceToDevice);
        assert(result == cudaSuccess);
        result = cudaMemcpy2D(
            layer_ptr + dst_y0 * Map_X + dst_x0, Map_X * sizeof(float),
            temp      + src_y0 * Map_X + src_x0, Map_X * sizeof(float),
            copy_w * sizeof(float), copy_h,
            cudaMemcpyDeviceToDevice);
        assert(result == cudaSuccess);
    }

    const float V_HEIGHT  = 0.0f;
    const float V_VAR     = 1000.0f;
    const float V_VALID   = 0.0f;
    const float V_TRAV    = 0.0f;
    const float V_TIME    = 0.0f;
    const float V_UPPER   = 0.0f;
    const float V_ISUP    = 0.0f;

    dim3 block(16, 16);

    if (shift_x > 0) {
        dim3 grid(((shift_x + 15) / 16), (Map_Y + 15) / 16);
        fill_vertical_band_kernel<<<grid, block>>>(dev_elevation_map, 0, shift_x, V_HEIGHT, V_VAR, V_VALID, V_TRAV, V_TIME, V_UPPER, V_ISUP);
    }
    if (shift_x < 0) {
        int start = Map_X + shift_x; if (start < 0) start = 0;
        dim3 grid(((Map_X - start + 15) / 16), (Map_Y + 15) / 16);
        fill_vertical_band_kernel<<<grid, block>>>(dev_elevation_map, start, Map_X, V_HEIGHT, V_VAR, V_VALID, V_TRAV, V_TIME, V_UPPER, V_ISUP);
    }
    if (shift_y > 0) {
        dim3 grid((Map_X + 15) / 16, ((shift_y + 15) / 16));
        fill_horizontal_band_kernel<<<grid, block>>>(dev_elevation_map, 0, shift_y, V_HEIGHT, V_VAR, V_VALID, V_TRAV, V_TIME, V_UPPER, V_ISUP);
    }
    if (shift_y < 0) {
        int start = Map_Y + shift_y; if (start < 0) start = 0;
        dim3 grid((Map_X + 15) / 16, ((Map_Y - start + 15) / 16));
        fill_horizontal_band_kernel<<<grid, block>>>(dev_elevation_map, start, Map_Y, V_HEIGHT, V_VAR, V_VALID, V_TRAV, V_TIME, V_UPPER, V_ISUP);
    }

    if (dev_normal_map != nullptr) {
        for (int comp = 0; comp < 3; ++comp) {
            float* comp_ptr = dev_normal_map + comp * MAP_CELLS;
            result = cudaMemcpy(temp, comp_ptr, MAP_CELLS * sizeof(float), cudaMemcpyDeviceToDevice);
            assert(result == cudaSuccess);
            result = cudaMemcpy2D(
                comp_ptr + dst_y0 * Map_X + dst_x0, Map_X * sizeof(float),
                temp      + src_y0 * Map_X + src_x0, Map_X * sizeof(float),
                copy_w * sizeof(float), copy_h,
                cudaMemcpyDeviceToDevice);
            assert(result == cudaSuccess);
        }
        // FIX 14: чистим полосы (иначе нормали — мусор, ломает wall-check)
        if (shift_x > 0) {
            dim3 g(((shift_x + 15) / 16), (Map_Y + 15) / 16);
            fill_normal_vertical_band_kernel<<<g, block>>>(dev_normal_map, 0, shift_x);
        }
        if (shift_x < 0) {
            int start = Map_X + shift_x; if (start < 0) start = 0;
            dim3 g(((Map_X - start + 15) / 16), (Map_Y + 15) / 16);
            fill_normal_vertical_band_kernel<<<g, block>>>(dev_normal_map, start, Map_X);
        }
        if (shift_y > 0) {
            dim3 g((Map_X + 15) / 16, ((shift_y + 15) / 16));
            fill_normal_horizontal_band_kernel<<<g, block>>>(dev_normal_map, 0, shift_y);
        }
        if (shift_y < 0) {
            int start = Map_Y + shift_y; if (start < 0) start = 0;
            dim3 g((Map_X + 15) / 16, ((Map_Y - start + 15) / 16));
            fill_normal_horizontal_band_kernel<<<g, block>>>(dev_normal_map, start, Map_Y);
        }
    }

    result = cudaDeviceSynchronize();
    assert(result == cudaSuccess);
}

// ============================================================================
// Инициализация / освобождение персистентных GPU-буферов
// ============================================================================

extern "C" __host__
void init_cuda_map(float** dev_elevation_map, float** dev_new_map,
                   float** dev_normal_map, float** dev_error,
                   float** dev_error_cnt,
                   uint8_t** dev_costmap,
                   float** dev_shift_temp)
{
    cudaError_t result;

    // 1. Основная карта
    result = cudaMalloc(dev_elevation_map, MAP_LAYERS * MAP_CELLS * sizeof(float));
    assert(result == cudaSuccess);
    result = cudaMemset(*dev_elevation_map, 0, MAP_LAYERS * MAP_CELLS * sizeof(float));
    assert(result == cudaSuccess);

    // 2. Буфер для atomicAdd
    result = cudaMalloc(dev_new_map, MAP_LAYERS * MAP_CELLS * sizeof(float));
    assert(result == cudaSuccess);
    result = cudaMemset(*dev_new_map, 0, MAP_LAYERS * MAP_CELLS * sizeof(float));
    assert(result == cudaSuccess);

    // 3. Нормали
    result = cudaMalloc(dev_normal_map, 3 * MAP_CELLS * sizeof(float));
    assert(result == cudaSuccess);
    result = cudaMemset(*dev_normal_map, 0, 3 * MAP_CELLS * sizeof(float));
    assert(result == cudaSuccess);

    // 4. Дрифт
    result = cudaMalloc(dev_error, sizeof(float));
    assert(result == cudaSuccess);
    result = cudaMalloc(dev_error_cnt, sizeof(float));
    assert(result == cudaSuccess);

    // 5. Costmap
    result = cudaMalloc(dev_costmap, Map_X * Map_Y);
    assert(result == cudaSuccess);
    result = cudaMemset(*dev_costmap, 0, Map_X * Map_Y);
    assert(result == cudaSuccess);

    // 6. Буфер для сдвига карты (shift_map_xy)
    result = cudaMalloc(dev_shift_temp, MAP_CELLS * sizeof(float));
    assert(result == cudaSuccess);
    result = cudaMemset(*dev_shift_temp, 0, MAP_CELLS * sizeof(float));
    assert(result == cudaSuccess);

    cudaDeviceSynchronize();
    std::cout << "[CUDA] init_cuda_map done (2.5D + shift temp)" << std::endl;
}


extern "C" __host__
void cleanup_cuda_map(float* dev_elevation_map, float* dev_new_map,
                      float* dev_normal_map, float* dev_error,
                      float* dev_error_cnt,
                      uint8_t* dev_costmap,
                      float* dev_shift_temp)   // ← НОВЫЙ ПАРАМЕТР
{
    if (dev_elevation_map) cudaFree(dev_elevation_map);
    if (dev_new_map)       cudaFree(dev_new_map);
    if (dev_normal_map)    cudaFree(dev_normal_map);
    if (dev_error)         cudaFree(dev_error);
    if (dev_error_cnt)     cudaFree(dev_error_cnt);
    if (dev_costmap)       cudaFree(dev_costmap);
    if (dev_shift_temp)    cudaFree(dev_shift_temp);

    cudaDeviceSynchronize();
    std::cout << "[CUDA] cleanup_cuda_map done" << std::endl;
}

// ============================================================================
// ИНИЦИАЛИЗАЦИЯ ДИСПЕРСИИ И ФЛАГОВ
// ============================================================================
__global__ void init_variance_kernel(float* elevation_map, float initial_variance)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= MAP_CELLS) return;

    elevation_map[0 * MAP_CELLS + i] = 0.0f;               // L_HEIGHT
    elevation_map[1 * MAP_CELLS + i] = initial_variance;   // L_VARIANCE
    elevation_map[2 * MAP_CELLS + i] = 0.0f;               // L_IS_VALID
    elevation_map[3 * MAP_CELLS + i] = 1.0f;               // L_TRAVERS
    elevation_map[4 * MAP_CELLS + i] = 0.0f;               // L_TIME
    elevation_map[5 * MAP_CELLS + i] = 0.0f;               // L_UPPER_BOUND
    elevation_map[6 * MAP_CELLS + i] = 0.0f;               // L_IS_UPPER
}

extern "C" __host__
void init_elevation_defaults(float* dev_elevation_map, float initial_variance)
{
    int blocks = (MAP_CELLS + 255) / 256;
    init_variance_kernel<<<blocks, 256>>>(dev_elevation_map, initial_variance);
    cudaDeviceSynchronize();
    std::cout << "[CUDA] init_elevation_defaults done" << std::endl;
}

extern "C" __host__
int init_frustum_cuda(const rs2_intrinsics &intrin)
{
    cudaError_t result;

    float fov[2];
    rs2_fov(&intrin, fov);   // градусы

    float fov_h_rad = fov[0] * (float)M_PI / 180.0f;
    float fov_v_rad = fov[1] * (float)M_PI / 180.0f;
    float tan_half_h = tanf(fov_h_rad / 2.0f);
    float tan_half_v = tanf(fov_v_rad / 2.0f);

    const float NEAR_MM = CAM_NEAR_MM;
    const float FAR_MM  = CAM_FAR_MM;
    const float VOX_MM  = FRUSTUM_VOXEL_MM;

    int Nz = (int)((FAR_MM - NEAR_MM) / VOX_MM);
    assert(Nz <= FRUSTUM_MAX_LAYERS && "frustum layers exceed FRUSTUM_MAX_LAYERS");

    int Nx_arr[FRUSTUM_MAX_LAYERS] = {0};
    int Ny_arr[FRUSTUM_MAX_LAYERS] = {0};
    int off_arr[FRUSTUM_MAX_LAYERS] = {0};

    int offset = 0;
    for (int iz = 0; iz < Nz; ++iz) {
        float z_center = NEAR_MM + iz * VOX_MM + VOX_MM * 0.5f;
        float half_x = z_center * tan_half_h;
        float half_y = z_center * tan_half_v;
        int Nx = (int)floorf(2.0f * half_x / VOX_MM);
        int Ny = (int)floorf(2.0f * half_y / VOX_MM);
        if (Nx < 1) Nx = 1;
        if (Ny < 1) Ny = 1;
        Nx_arr[iz] = Nx;
        Ny_arr[iz] = Ny;
        off_arr[iz] = offset;
        offset += Nx * Ny;
    }
    int total_voxels = offset;

    result = cudaMemcpyToSymbol(c_Nz, &Nz, sizeof(int)); assert(result == cudaSuccess);
    result = cudaMemcpyToSymbol(c_Nx_per_layer, Nx_arr, sizeof(int) * Nz); assert(result == cudaSuccess);
    result = cudaMemcpyToSymbol(c_Ny_per_layer, Ny_arr, sizeof(int) * Nz); assert(result == cudaSuccess);
    result = cudaMemcpyToSymbol(c_layer_offset, off_arr, sizeof(int) * Nz); assert(result == cudaSuccess);
    result = cudaMemcpyToSymbol(c_tan_half_h, &tan_half_h, sizeof(float)); assert(result == cudaSuccess);
    result = cudaMemcpyToSymbol(c_tan_half_v, &tan_half_v, sizeof(float)); assert(result == cudaSuccess);
    result = cudaMemcpyToSymbol(c_cam_near_mm, &NEAR_MM, sizeof(float)); assert(result == cudaSuccess);
    result = cudaMemcpyToSymbol(c_cam_far_mm, &FAR_MM, sizeof(float)); assert(result == cudaSuccess);

    std::cout << "[CUDA] init_frustum_cuda: Nz=" << Nz
              << " total_voxels=" << total_voxels
              << " fov_h=" << fov[0] << " fov_v=" << fov[1] << std::endl;

    return total_voxels;
}

extern "C" __host__
void deproject_depth_cuda(
    float* out_points_cpu,
    uint32_t* out_num_points,
    uint32_t total_voxels_frustum,
    const rs2_intrinsics& intrin,
    const uint16_t* depth, float depth_scale,
    const float R_cam_to_base[9],
    const float t_cam_to_base[3],
    const float R_base_to_map[9],
    const float shift[3])
{
    cudaError_t result;

    // ========================================================================
    // ПАРАМЕТРЫ
    // ========================================================================
    const int border_top    = 50;
    const int border_bottom = 0;
    const int border_left   = 50;
    const int border_right  = 20;

    const int count = intrin.height * intrin.width;
    const int numBlocks = count / RS2_CUDA_THREADS_PER_BLOCK;

    const uint32_t blocks_1d = (total_voxels_frustum + RS2_CUDA_THREADS_PER_BLOCK - 1)
                               / RS2_CUDA_THREADS_PER_BLOCK;

    // ========================================================================
    // ЛОКАЛЬНЫЕ GPU-БУФЕРЫ
    // ========================================================================
    uint16_t*           dev_depth;
    rs2_intrinsics*     dev_intrin;
    unsigned long long* dev_ray_points_cam;
    unsigned long long* dev_compact_packed;
    uint32_t*           dev_counts;
    uint32_t*           dev_offsets;
    float*              dev_points_world;

    // ========================================================================
    // ШАГ 1: Аллокация
    // ========================================================================
    result = cudaMalloc(&dev_depth, count * sizeof(uint16_t));
    assert(result == cudaSuccess);
    result = cudaMalloc(&dev_intrin, sizeof(rs2_intrinsics));
    assert(result == cudaSuccess);
    result = cudaMalloc(&dev_ray_points_cam,
                        (size_t)total_voxels_frustum * sizeof(unsigned long long));
    assert(result == cudaSuccess);
    result = cudaMalloc(&dev_compact_packed,
                        (size_t)total_voxels_frustum * sizeof(unsigned long long));
    assert(result == cudaSuccess);
    result = cudaMalloc(&dev_counts, blocks_1d * sizeof(uint32_t));
    assert(result == cudaSuccess);
    result = cudaMalloc(&dev_offsets, blocks_1d * sizeof(uint32_t));
    assert(result == cudaSuccess);
    result = cudaMalloc(&dev_points_world,
                        (size_t)total_voxels_frustum * 3 * sizeof(float));
    assert(result == cudaSuccess);

    // ========================================================================
    // ШАГ 2: Загрузка depth и интринсиков
    // ========================================================================
    result = cudaMemcpy(dev_depth, depth, count * sizeof(uint16_t), cudaMemcpyHostToDevice);
    assert(result == cudaSuccess);
    result = cudaMemcpy(dev_intrin, &intrin, sizeof(rs2_intrinsics), cudaMemcpyHostToDevice);
    assert(result == cudaSuccess);

    // ========================================================================
    // ШАГ 3: Crop границ глубины
    // ========================================================================
    dim3 blockSize(32, 32);
    dim3 gridSize((intrin.width + blockSize.x - 1) / blockSize.x,
                  (intrin.height + blockSize.y - 1) / blockSize.y);

    crop_depth_borders<<<gridSize, blockSize>>>(
        dev_depth, intrin.width, intrin.height,
        border_top, border_bottom, border_left, border_right);
    result = cudaDeviceSynchronize();
    assert(result == cudaSuccess);

    // ========================================================================
    // ШАГ 4: Депроекция + frustum binning (в СК камеры)
    // ========================================================================
    result = cudaMemset(dev_ray_points_cam, 0xFF,
                        (size_t)total_voxels_frustum * sizeof(unsigned long long));
    assert(result == cudaSuccess);

    kernel_deproject_depth_cuda<<<numBlocks, RS2_CUDA_THREADS_PER_BLOCK>>>(
        dev_ray_points_cam, dev_intrin, dev_depth, depth_scale);
    result = cudaDeviceSynchronize();
    assert(result == cudaSuccess);

    // depth и intrin больше не нужны
    cudaFree(dev_depth);    dev_depth = nullptr;
    cudaFree(dev_intrin);   dev_intrin = nullptr;

    // ========================================================================
    // ШАГ 5: Pass 1 — count непустых packed-точек (блочная редукция)
    // ========================================================================
    count_valid_ray_points_kernel<<<blocks_1d, RS2_CUDA_THREADS_PER_BLOCK>>>(
        dev_ray_points_cam, dev_counts, total_voxels_frustum);
    result = cudaDeviceSynchronize();
    assert(result == cudaSuccess);

    // ========================================================================
    // ШАГ 6: Prefix-sum на CPU + H2D offsets
    // ========================================================================
    uint32_t* h_counts = new uint32_t[blocks_1d];
    result = cudaMemcpy(h_counts, dev_counts, blocks_1d * sizeof(uint32_t),
                        cudaMemcpyDeviceToHost);
    assert(result == cudaSuccess);

    uint32_t* h_offsets = new uint32_t[blocks_1d];
    h_offsets[0] = 0;
    for (uint32_t b = 1; b < blocks_1d; ++b) {
        h_offsets[b] = h_offsets[b - 1] + h_counts[b - 1];
    }
    uint32_t total_count = h_offsets[blocks_1d - 1] + h_counts[blocks_1d - 1];
    delete[] h_counts;


    // ========================================================================
    // ШАГ 7: Pass 2 — блочный compact (без глобального atomicAdd)
    // ========================================================================
    if (total_count > 0) {
        result = cudaMemcpy(dev_offsets, h_offsets, blocks_1d * sizeof(uint32_t),
                            cudaMemcpyHostToDevice);
        assert(result == cudaSuccess);

        compact_valid_ray_points_kernel<<<blocks_1d, RS2_CUDA_THREADS_PER_BLOCK>>>(
            dev_ray_points_cam, dev_compact_packed, dev_offsets, total_voxels_frustum);

        result = cudaDeviceSynchronize();
        if (result != cudaSuccess) {
            std::cerr << "[DEPROJECT ERROR] compact failed: "
                      << cudaGetErrorString(result) << "\n";
            std::cerr << "  total_count = " << total_count << "\n";
        }

        assert(result == cudaSuccess);
    }
    delete[] h_offsets;

    // ray_points_cam больше не нужен
    cudaFree(dev_ray_points_cam);    dev_ray_points_cam = nullptr;
    cudaFree(dev_counts);            dev_counts = nullptr;
    cudaFree(dev_offsets);           dev_offsets = nullptr;

    // ========================================================================
    // ШАГ 8: Rotate только компактных (без sentinel-проверок)
    // ========================================================================

    *out_num_points = 0;

    if (total_count > 0) {
        const uint32_t rotate_blocks = (total_count + 255) / 256;

        // --- 8.1: Выделение device-буферов для параметров ядра ---
        float* dev_R_cam_base = nullptr;
        float* dev_t_cam_base = nullptr;
        float* dev_R_base_map = nullptr;
        float* dev_shift      = nullptr;

        result = cudaMalloc(&dev_R_cam_base, 9 * sizeof(float)); assert(result == cudaSuccess);
        result = cudaMalloc(&dev_t_cam_base, 3 * sizeof(float)); assert(result == cudaSuccess);
        result = cudaMalloc(&dev_R_base_map, 9 * sizeof(float)); assert(result == cudaSuccess);
        result = cudaMalloc(&dev_shift,      3 * sizeof(float)); assert(result == cudaSuccess);

        // --- 8.2: Копирование host → device ---
        result = cudaMemcpy(dev_R_cam_base, R_cam_to_base, 9 * sizeof(float),
                            cudaMemcpyHostToDevice); assert(result == cudaSuccess);
        result = cudaMemcpy(dev_t_cam_base, t_cam_to_base, 3 * sizeof(float),
                            cudaMemcpyHostToDevice); assert(result == cudaSuccess);
        result = cudaMemcpy(dev_R_base_map, R_base_to_map, 9 * sizeof(float),
                            cudaMemcpyHostToDevice); assert(result == cudaSuccess);
        result = cudaMemcpy(dev_shift,      shift,         3 * sizeof(float),
                            cudaMemcpyHostToDevice); assert(result == cudaSuccess);

        // --- 8.3: Запуск ядра с device-указателями ---
        rotate_compact_points_to_world<<<rotate_blocks, 256>>>(
            dev_compact_packed, dev_points_world, total_count,
            dev_R_cam_base, dev_t_cam_base, dev_R_base_map, dev_shift);

        result = cudaDeviceSynchronize();
        if (result != cudaSuccess) {
            std::cerr << "[DEPROJECT ERROR] rotate_compact_points_to_world: "
                      << cudaGetErrorString(result) << "\n";
            *out_num_points = 0;

            cudaFree(dev_R_cam_base);
            cudaFree(dev_t_cam_base);
            cudaFree(dev_R_base_map);
            cudaFree(dev_shift);
            cudaFree(dev_compact_packed);
            cudaFree(dev_points_world);
            cudaDeviceSynchronize();
            return;
        }

        // ====================================================================
        // ШАГ 9: D2H только компактных
        // ====================================================================
        result = cudaMemcpy(out_points_cpu, dev_points_world,
                            total_count * 3 * sizeof(float), cudaMemcpyDeviceToHost);
        assert(result == cudaSuccess);

        *out_num_points = total_count;


        cudaFree(dev_R_cam_base);
        cudaFree(dev_t_cam_base);
        cudaFree(dev_R_base_map);
        cudaFree(dev_shift);
    }

    // ========================================================================
    // ШАГ 10: Освобождение оставшихся буферов
    // ========================================================================
    cudaFree(dev_compact_packed);   dev_compact_packed = nullptr;
    cudaFree(dev_points_world);     dev_points_world = nullptr;

    cudaDeviceSynchronize();
}


extern "C" __host__
void process_costmap_cuda(
    const float* lidar_pcl, int num_lidar,
    const float* sensor_pos_lidar,
    const float* cam_pcl, int num_cam,
    const float* sensor_pos_cam,
    float* dev_elevation_map, float* dev_new_map, float* dev_normal_map,
    float* dev_error, float* dev_error_cnt,
    uint8_t* dev_costmap, uint8_t* costmap_host,
    uint32_t* costmap_rgba_host,
    float* heightmap_packed_host,
    float position_noise, float orientation_noise,
    float* out_h_min, float* out_h_max, float* out_rover_height_m)
{
    cudaError_t result;
    dim3 block(16, 16);
    dim3 grid((400 + 15) / 16, (400 + 15) / 16);

    // ========================================================================
    // ПАРАМЕТРЫ ЯДЕР (без изменений)
    // ========================================================================
    const float SENSOR_NOISE_FACTOR    = 0.05f;
    const float MAHALANOBIS_THRESH     = 2.0f;
    const float OUTLIER_VARIANCE       = 0.01f;
    const float DRIFT_COMP_VAR_INLIER  = 0.05f;
    const float MAX_VARIANCE           = 100.0f;
    const float INITIAL_VARIANCE       = 1000.0f;

    const float MAX_HEIGHT_RANGE       = 0.3f;
    const float RAMPED_A               = 0.3f;
    const float RAMPED_B               = 1.0f;
    const float RAMPED_C               = 0.2f;
    const float MIN_VALID_DISTANCE     = 0.5f;

    const float WALL_NUM_THRESH        = 20.0f;
    const float MAX_RAY_LENGTH         = 10.0f;
    const float CLEANUP_STEP           = 0.1f;
    const float CLEANUP_COS_THRESH     = 0.1f;

    const float DRIFT_ALPHA            = 0.1f;
    const float MAX_DRIFT              = 0.1f;
    const int   MIN_HEIGHT_DRIFT_CNT   = 100;
    const float POSITION_NOISE_THRESH  = 0.01f;
    const float ORIENTATION_NOISE_THRESH = 0.02f;

    const float VAR_INTERVAL           = 0.1f;
    const float TIME_VARIANCE          = 0.0001f;

    const float CLEAR_RANGE_Z          = 2.0f;
    const float CLEAR_RANGE_XY         = 4.0f;
    const float CLEAR_INITIAL_VARIANCE = 1000.0f;

    const int   DILATION_SIZE          = 3;

    //Угол	Тангенс	Уклон %
    //20°	0.36	36%
    //25°	0.47	47%
    //30°	0.58	58%
    //35°	0.70	70%
    //45°	1.0	    100%

    const float MAX_SLOPE              = 0.2f;
    const float FLAT_SLOPE             = 0.01f;
    const int   INSCRIBED_RADIUS       = 2;
    const int   INFLATION_RADIUS       = 4;
    const float COST_SCALING_FACTOR    = 15.0f;
    const int   ESTIMATION_SIZE_SLOPE  = 1;

    const int   enable_edge_sharpen        = 1;
    const int   enable_visibility_cleanup  = 1;
    const int   enable_drift_compensation  = 1;
    const int   enable_overlap_clearance   = 1;

    // ========================================================================
    // ЛОКАЛЬНЫЕ БУФЕРЫ
    // ========================================================================
    float* dev_points_lidar;
    float* dev_sensor_pos_lidar;
    float* dev_sensor_pos_cam;
    int*   dev_point_count_lidar;
    int*   dev_point_count_cam;
    float* dev_points_cam = nullptr;   // GPU-указатель, если камера есть

    float*   dev_slope;
    float*   dev_heightmap;
    float*   dev_heightmap_packed;
    int*     dev_minmax;
    uint8_t* dev_costmap_tmp;
    float*   dev_dilated;
    float*   dev_valid_combined;

    // ========================================================================
    // ШАГ 1: Аллокация локальных буферов
    // ========================================================================
    result = cudaMalloc(&dev_sensor_pos_lidar, 3 * sizeof(float));       assert(result == cudaSuccess);
    result = cudaMalloc(&dev_sensor_pos_cam,   3 * sizeof(float));       assert(result == cudaSuccess);
    result = cudaMalloc(&dev_points_lidar, num_lidar * 3 * sizeof(float)); assert(result == cudaSuccess);
    result = cudaMalloc(&dev_point_count_lidar, sizeof(int));            assert(result == cudaSuccess);
    result = cudaMalloc(&dev_point_count_cam,   sizeof(int));            assert(result == cudaSuccess);

    if (num_cam > 0 && cam_pcl != nullptr) {
        result = cudaMalloc(&dev_points_cam, num_cam * 3 * sizeof(float));
        assert(result == cudaSuccess);
    }

    result = cudaMalloc(&dev_slope,           MAP_CELLS * sizeof(float)); assert(result == cudaSuccess);
    result = cudaMalloc(&dev_heightmap,       MAP_CELLS * sizeof(float)); assert(result == cudaSuccess);
    result = cudaMalloc(&dev_heightmap_packed, 400 * 200 * sizeof(float)); assert(result == cudaSuccess);
    result = cudaMalloc(&dev_dilated,         MAP_CELLS * sizeof(float)); assert(result == cudaSuccess);
    result = cudaMalloc(&dev_valid_combined,  MAP_CELLS * sizeof(float)); assert(result == cudaSuccess);
    result = cudaMalloc(&dev_minmax,          2 * sizeof(unsigned int));   assert(result == cudaSuccess);
    result = cudaMalloc(&dev_costmap_tmp,     MAP_CELLS);                  assert(result == cudaSuccess);

    // ========================================================================
    // ШАГ 2: Тик L_TIME (10 Гц)
    // ========================================================================
    static auto last_var_update = std::chrono::steady_clock::now();
    static auto last_time_update = std::chrono::steady_clock::now();
    auto now = std::chrono::steady_clock::now();

    float dt_var = std::chrono::duration<float>(now - last_var_update).count();
    if (dt_var >= VAR_INTERVAL) {
        update_variance_kernel<<<(MAP_CELLS + 255) / 256, 256>>>(
            dev_elevation_map, TIME_VARIANCE, MAX_VARIANCE);
        last_var_update = now;
    }

    float dt_time = std::chrono::duration<float>(now - last_time_update).count();
    if (dt_time >= VAR_INTERVAL) {
        update_time_kernel<<<(MAP_CELLS + 255) / 256, 256>>>(
            dev_elevation_map, 0.1f);
        last_time_update = now;
    }

    // ========================================================================
    // ШАГ 3: Загрузка LiDAR-точек и позиции сенсора
    // ========================================================================
    result = cudaMemcpy(dev_point_count_lidar, &num_lidar, sizeof(int), cudaMemcpyHostToDevice);
    assert(result == cudaSuccess);
    result = cudaMemcpy(dev_points_lidar, lidar_pcl,
                        num_lidar * 3 * sizeof(float), cudaMemcpyHostToDevice);
    assert(result == cudaSuccess);
    result = cudaMemcpy(dev_sensor_pos_lidar, sensor_pos_lidar,
                        3 * sizeof(float), cudaMemcpyHostToDevice);
    assert(result == cudaSuccess);

    // Загрузка точек камеры (если есть)
    if (num_cam > 0 && cam_pcl != nullptr) {
        result = cudaMemcpy(dev_points_cam, cam_pcl,
                            num_cam * 3 * sizeof(float), cudaMemcpyHostToDevice);
        assert(result == cudaSuccess);
    }
    result = cudaMemcpy(dev_point_count_cam, &num_cam, sizeof(int), cudaMemcpyHostToDevice);
    assert(result == cudaSuccess);
    result = cudaMemcpy(dev_sensor_pos_cam, sensor_pos_cam,
                        3 * sizeof(float), cudaMemcpyHostToDevice);
    assert(result == cudaSuccess);

    // ========================================================================
    // ШАГ 4: Drift compensation (LiDAR)
    // ========================================================================
    cudaMemset(dev_error, 0, sizeof(float));
    cudaMemset(dev_error_cnt, 0, sizeof(float));
    cudaMemset(dev_new_map, 0, MAP_LAYERS * MAP_CELLS * sizeof(float));

    int blocks_err = (num_lidar + 255) / 256;
    error_counting_kernel_cuda<<<blocks_err, 256>>>(
        dev_points_lidar, num_lidar, dev_sensor_pos_lidar,
        dev_elevation_map, dev_error, dev_error_cnt, dev_new_map,
        MAP_RESOLUTION, SENSOR_NOISE_FACTOR, MAHALANOBIS_THRESH, DRIFT_COMP_VAR_INLIER,
        0.9f,
        MIN_VALID_DISTANCE, MAX_HEIGHT_RANGE,
        RAMPED_A, RAMPED_B, RAMPED_C);
    result = cudaDeviceSynchronize();
    assert(result == cudaSuccess);

    // Drift compensation (камера тоже участвует в error_counting — опционально)
    if (num_cam > 0) {
        int blocks_err_cam = (num_cam + 255) / 256;
        error_counting_kernel_cuda<<<blocks_err_cam, 256>>>(
            dev_points_cam, num_cam, dev_sensor_pos_cam,
            dev_elevation_map, dev_error, dev_error_cnt, dev_new_map,
            MAP_RESOLUTION, SENSOR_NOISE_FACTOR, MAHALANOBIS_THRESH, DRIFT_COMP_VAR_INLIER,
            0.9f,
            MIN_VALID_DISTANCE, MAX_HEIGHT_RANGE,
            RAMPED_A, RAMPED_B, RAMPED_C);
        result = cudaDeviceSynchronize();
        assert(result == cudaSuccess);
    }

    float h_error, h_error_cnt;
    cudaMemcpy(&h_error, dev_error, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_error_cnt, dev_error_cnt, sizeof(float), cudaMemcpyDeviceToHost);

    if (enable_drift_compensation &&
        h_error_cnt > MIN_HEIGHT_DRIFT_CNT &&
        (position_noise > POSITION_NOISE_THRESH || orientation_noise > ORIENTATION_NOISE_THRESH)) {
        float mean_error = h_error / h_error_cnt;
        if (fabsf(mean_error) < MAX_DRIFT) {
            drift_comp_kernel<<<(MAP_CELLS + 255) / 256, 256>>>(
                dev_elevation_map, mean_error * DRIFT_ALPHA);
            result = cudaDeviceSynchronize();
            assert(result == cudaSuccess);
        }
    }

    // ========================================================================
    // ШАГ 5: add_points (LiDAR с cleanup)
    // ========================================================================
    int blocks_add = (num_lidar + 255) / 256;
    add_points_kernel_cuda<<<blocks_add, 256>>>(
        dev_points_lidar, num_lidar, dev_sensor_pos_lidar,
        dev_elevation_map, dev_new_map, dev_normal_map,
        MAP_RESOLUTION, SENSOR_NOISE_FACTOR, MAHALANOBIS_THRESH, OUTLIER_VARIANCE,
        WALL_NUM_THRESH, MAX_RAY_LENGTH, CLEANUP_STEP,
        MIN_VALID_DISTANCE, MAX_HEIGHT_RANGE, CLEANUP_COS_THRESH,
        RAMPED_A, RAMPED_B, RAMPED_C,
        enable_edge_sharpen,
        enable_visibility_cleanup);
    result = cudaDeviceSynchronize();
    assert(result == cudaSuccess);

    // ========================================================================
    // ШАГ 6: add_points (камера БЕЗ cleanup)
    // ========================================================================
    if (num_cam > 0) {
        int blocks_add_cam = (num_cam + 255) / 256;
        add_points_kernel_cuda<<<blocks_add_cam, 256>>>(
            dev_points_cam, num_cam, dev_sensor_pos_cam,
            dev_elevation_map, dev_new_map, dev_normal_map,
            MAP_RESOLUTION, SENSOR_NOISE_FACTOR, MAHALANOBIS_THRESH, OUTLIER_VARIANCE,
            WALL_NUM_THRESH, MAX_RAY_LENGTH, CLEANUP_STEP,
            MIN_VALID_DISTANCE, MAX_HEIGHT_RANGE, CLEANUP_COS_THRESH,
            RAMPED_A, RAMPED_B, RAMPED_C,
            enable_edge_sharpen,
            enable_visibility_cleanup);
        result = cudaDeviceSynchronize();
        assert(result == cudaSuccess);
    }

    // ========================================================================
    // ШАГ 7: average_map (один раз для обоих источников)
    // ========================================================================
    average_map_kernel_cuda<<<(MAP_CELLS + 255) / 256, 256>>>(
        dev_new_map, dev_elevation_map, MAX_VARIANCE, INITIAL_VARIANCE);
    result = cudaDeviceSynchronize();
    assert(result == cudaSuccess);

    // ========================================================================
    // ШАГ 8: overlap_clearance
    // ========================================================================
    int clear_range_cells = ((int)(CLEAR_RANGE_XY / MAP_RESOLUTION)) / 2;
    int total = (2 * clear_range_cells + 1) * (2 * clear_range_cells + 1);
    if (enable_overlap_clearance) {
        overlap_clearance_kernel<<<(total + 255) / 256, 256>>>(
            dev_elevation_map, sensor_pos_lidar[2], CLEAR_RANGE_Z,
            CLEAR_INITIAL_VARIANCE, MAP_CENTER, MAP_CENTER, clear_range_cells);
        result = cudaDeviceSynchronize();
        assert(result == cudaSuccess);
    }

    // ========================================================================
    // ШАГ 9: combine_validity + dilation + normal
    // ========================================================================
    combine_validity_kernel<<<(MAP_CELLS + 255) / 256, 256>>>(
        dev_elevation_map + 2 * MAP_CELLS,
        dev_elevation_map + 6 * MAP_CELLS,
        dev_valid_combined);
    result = cudaDeviceSynchronize();
    assert(result == cudaSuccess);

    dilation_filter_kernel<<<grid, block>>>(
        dev_elevation_map + 5 * MAP_CELLS,
        dev_elevation_map + 2 * MAP_CELLS,
        dev_elevation_map + 6 * MAP_CELLS,
        dev_dilated,
        DILATION_SIZE);
    result = cudaDeviceSynchronize();
    assert(result == cudaSuccess);

    normal_filter_kernel_cuda<<<grid, block>>>(
        dev_dilated,
        dev_elevation_map,
        dev_normal_map,
        MAP_RESOLUTION);
    result = cudaDeviceSynchronize();
    assert(result == cudaSuccess);

    // ========================================================================
    // ШАГ 10: heightmap для визуализации (pack)
    // ========================================================================
    int init_mm[2] = { 32767, -32768 };
    cudaMemcpy(dev_minmax, init_mm, 2 * sizeof(int), cudaMemcpyHostToDevice);

    dim3 pack_grid(((MAP_SIZE / 2) + 15) / 16, (MAP_SIZE + 15) / 16);
    pack_heightmap_minmax_kernel<<<pack_grid, block>>>(
        dev_elevation_map, dev_elevation_map + 2 * MAP_CELLS,
        dev_heightmap_packed, dev_minmax);
    result = cudaDeviceSynchronize();
    assert(result == cudaSuccess);

    cudaMemcpy(heightmap_packed_host, dev_heightmap_packed,
               MAP_SIZE * (MAP_SIZE / 2) * sizeof(float), cudaMemcpyDeviceToHost);

    int h_minmax_u16[2];
    cudaMemcpy(h_minmax_u16, dev_minmax, 2 * sizeof(int), cudaMemcpyDeviceToHost);

    int h_min_int = h_minmax_u16[0];
    int h_max_int = h_minmax_u16[1];
    *out_h_min = (float)h_min_int * 0.001f;
    *out_h_max = (float)h_max_int * 0.001f;
    if (*out_h_max - *out_h_min < 0.05f) *out_h_max = *out_h_min + 0.5f;
    *out_rover_height_m = 0.0f;

    // ========================================================================
    // ШАГ 11: slope
    // ========================================================================

    heightmap_to_slope_cuda<<<grid, block>>>(
        dev_elevation_map, dev_valid_combined, dev_slope, ESTIMATION_SIZE_SLOPE);
    result = cudaDeviceSynchronize();
    assert(result == cudaSuccess);

    // ========================================================================
    // ШАГ 12: costmap + traversability
    // ========================================================================
    slope_to_costmap_cuda<<<grid, block>>>(
        dev_slope, dev_valid_combined, dev_costmap, MAX_SLOPE, FLAT_SLOPE);
    result = cudaDeviceSynchronize();
    assert(result == cudaSuccess);

    costmap_to_traversability_kernel<<<(MAP_CELLS + 255) / 256, 256>>>(
        dev_costmap, dev_elevation_map);
    result = cudaDeviceSynchronize();
    assert(result == cudaSuccess);

    // ========================================================================
    // ШАГ 13: findContours + inflate
    // ========================================================================
    findContoursKernel_2d<<<grid, block>>>(dev_costmap);
    result = cudaDeviceSynchronize();
    assert(result == cudaSuccess);

    cudaMemcpy(dev_costmap_tmp, dev_costmap, MAP_CELLS, cudaMemcpyDeviceToDevice);
    inflate_costmap_cuda<<<grid, block>>>(
        dev_costmap_tmp, dev_costmap, INSCRIBED_RADIUS, INFLATION_RADIUS, COST_SCALING_FACTOR);
    result = cudaDeviceSynchronize();
    assert(result == cudaSuccess);

    // ========================================================================
    // ШАГ 14: colorize → CPU
    // ========================================================================
    uint32_t* dev_rgba = nullptr;
    result = cudaMalloc(&dev_rgba, MAP_CELLS * sizeof(uint32_t));
    assert(result == cudaSuccess);

    colorize_costmap_kernel<<<grid, block>>>(dev_costmap, dev_rgba, Map_X, Map_Y);
    result = cudaDeviceSynchronize();
    assert(result == cudaSuccess);

    cudaMemcpy(costmap_rgba_host, dev_rgba,
               MAP_CELLS * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaFree(dev_rgba);

    // ========================================================================
    // ШАГ 15: копия costmap на CPU
    // ========================================================================
    cudaMemcpy(costmap_host, dev_costmap, MAP_CELLS, cudaMemcpyDeviceToHost);

    // ========================================================================
    // ШАГ 16: Освобождение локальных буферов
    // ========================================================================
    cudaFree(dev_sensor_pos_lidar);
    cudaFree(dev_sensor_pos_cam);
    cudaFree(dev_points_lidar);
    cudaFree(dev_point_count_lidar);
    cudaFree(dev_point_count_cam);
    if (dev_points_cam) cudaFree(dev_points_cam);
    cudaFree(dev_slope);
    cudaFree(dev_heightmap);
    cudaFree(dev_heightmap_packed);
    cudaFree(dev_dilated);
    cudaFree(dev_valid_combined);
    cudaFree(dev_minmax);
    cudaFree(dev_costmap_tmp);

    cudaDeviceSynchronize();
}
