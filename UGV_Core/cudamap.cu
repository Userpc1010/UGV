#include <cuda.h>
#include <cuda_runtime.h>
#include <memory.h>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <librealsense2/rs.h>
#include <librealsense2/rsutil.h>
#include "assert.h"


#define RS2_CUDA_THREADS_PER_BLOCK 128
#define MAP_CELLS 160000
#define MAP_LAYERS 7
#define Map_X 400
#define Map_Y 400
#define Map_Z 400
#define DEFAULT_ROBOT_Y 57
#define ROBOT_HEIGHT 8
#define MAP_W 400
#define MAP_H 400
#define MAP_CELLS_IDX(y, x) ((y) * MAP_W + (x))
#define TIME_INTERVAL 0.1f



__device__  uint32_t get_value_2d (const uint16_t x, const uint16_t y) {
    uint32_t index = y * 400 + x;
    return index;
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


// ============================================================================
// Конвертация лидара (int16_t, мм) -> float (метры) в DLIO
// ============================================================================
__global__ void convert_lidar_to_points_kernel(
    const int16_t* lidar_pcl, int num_points,
    float* out_points, int* out_count
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_points) return;

    // Лидар уже в DLIO (x — вперёд, y — влево, z — высота)
    float x = lidar_pcl[i * 3]     * 0.001f; // мм -> м
    float y = lidar_pcl[i * 3 + 1] * 0.001f;
    float z = lidar_pcl[i * 3 + 2] * 0.001f;

    int idx = atomicAdd(out_count, 1);
    out_points[idx * 3 + 0] = x;
    out_points[idx * 3 + 1] = y;
    out_points[idx * 3 + 2] = z;
}

// ============================================================================
// Депроекция RealSense depth -> точки в системе DLIO (x, y, z)
// ============================================================================
__global__ void deproject_to_2d_kernel(
    const uint16_t* depth, const rs2_intrinsics* intrin, float depth_scale,
    const float* R_cam_to_base,  // 3x3 (из extrinsics)
    const float* t_cam_to_base,  // 3
    const float* R_base_to_world, // 3x3 (из позы DLIO)
    const float* t_base_to_world, // 3
    float* out_points, int* out_count, int width, int height
) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;
    if (u >= width || v >= height) return;

    float d = depth[v * width + u] * depth_scale;
    if (d <= 0.0f) return;

    // 1. Депроекция в систему камеры (X вправо, Y вниз, Z вперёд)
    const float pixel[] = {(float)u, (float)v};
    float p_cam[3];
    deproject_pixel_to_point_cuda(p_cam, intrin, pixel, d);

    // 2. Камера -> база (DLIO: X вперёд, Y влево, Z вверх)
    float p_base[3];
    p_base[0] = R_cam_to_base[0]*p_cam[0] + R_cam_to_base[1]*p_cam[1] + R_cam_to_base[2]*p_cam[2] + t_cam_to_base[0];
    p_base[1] = R_cam_to_base[3]*p_cam[0] + R_cam_to_base[4]*p_cam[1] + R_cam_to_base[5]*p_cam[2] + t_cam_to_base[1];
    p_base[2] = R_cam_to_base[6]*p_cam[0] + R_cam_to_base[7]*p_cam[1] + R_cam_to_base[8]*p_cam[2] + t_cam_to_base[2];

    // 3. База -> мир (DLIO)
    float p_world[3];
    p_world[0] = R_base_to_world[0]*p_base[0] + R_base_to_world[1]*p_base[1] + R_base_to_world[2]*p_base[2] + t_base_to_world[0];
    p_world[1] = R_base_to_world[3]*p_base[0] + R_base_to_world[4]*p_base[1] + R_base_to_world[5]*p_base[2] + t_base_to_world[1];
    p_world[2] = R_base_to_world[6]*p_base[0] + R_base_to_world[7]*p_base[1] + R_base_to_world[8]*p_base[2] + t_base_to_world[2];

    // 4. Записываем
    int idx = atomicAdd(out_count, 1);
    out_points[idx * 3 + 0] = p_world[0];
    out_points[idx * 3 + 1] = p_world[1];
    out_points[idx * 3 + 2] = p_world[2]; // высота
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
     if (dz > max_height_range ||dz > (dist_xy - ramped_b) * ramped_a + ramped_c) return;

    // === Центрирование: DLIO (0,0) -> карта (200,200) ===
    float x_em = x / resolution + 200.0f;
    float y_em = y / resolution + 200.0f;
    int idx_x = (int)x_em;
    int idx_y = (int)y_em;
    if (idx_x <= 0 || idx_x >= 399 || idx_y <= 0 || idx_y >= 399) return;
    int idx = idx_y * 400 + idx_x;

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

         int cx = (int)(nx / resolution + 200.0f);  // FIX 6: центр в 200
         int cy = (int)(ny / resolution + 200.0f);
         if (cx <= 0 || cx >= 399 || cy <= 0 || cy >= 399) continue;
         int nidx = cy * 400 + cx;

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

             float nrm = sqrtf(mnx*mnx + mny*mny + mnz*mnz);
             if (nrm > 1e-3f) {
             if (fabsf(ray_x*mnx + ray_y*mny + ray_z_dir*mnz) < cleanup_cos_thresh) continue;
             }

             // FIX 1 (см. P0-1): new_map[3] теперь не zeroed → защита стен работает.
             float npts = new_map[3 * MAP_CELLS + nidx];
             if (npts > wall_num_thresh && nmap_t < 1.0f) continue;

             // FIX 5: в EMC cleanup = -step / (ray_len / max_ray).
             // В порте было умножение вместо деления → очистка шла в обратную сторону.
             float cleanup_factor = cleanup_step / fmaxf(ray_len_clamped / max_ray_length, 0.01f);
             atomicAdd(&elevation_map[2 * MAP_CELLS + nidx], -cleanup_factor);
             atomicAdd(&elevation_map[1 * MAP_CELLS + nidx], outlier_variance);
         }
     }
}

__global__ void clear_counts_kernel(float* new_map)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= MAP_CELLS) return;
    new_map[3 * MAP_CELLS + i] = 0.0f;  // inlier count
    new_map[4 * MAP_CELLS + i] = 0.0f;  // total count
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
    if (x <= 0 || x >= 399 || y <= 0 || y >= 399) return;

    int idx = y * 400 + x;
    float valid = elevation_map[2 * MAP_CELLS + idx] + elevation_map[6 * MAP_CELLS + idx];
    if (valid < 0.5f) return;

    int idx_x = y * 400 + (x + 1);
    int idx_y = (y + 1) * 400 + x;
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
    if (dz > max_height_range ||
        dz > (dist_xy - ramped_b) * ramped_a + ramped_c) return;

    float x_em = x / resolution + 200.0f;
    float y_em = y / resolution + 200.0f;
    int idx_x = (int)x_em;
    int idx_y = (int)y_em;
    if (idx_x <= 0 || idx_x >= 399 || idx_y <= 0 || idx_y >= 399) return;
    int idx = idx_y * 400 + idx_x;

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
    if (x >= end_col || y >= 400) return;

    int idx = y * 400 + x;
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
    if (x >= 400 || y >= end_row) return;

    int idx = y * 400 + x;
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
    if (x >= end_col || y >= 400) return;
    int idx = y * 400 + x;
    normal_map[0 * MAP_CELLS + idx] = 0.0f;
    normal_map[1 * MAP_CELLS + idx] = 0.0f;
    normal_map[2 * MAP_CELLS + idx] = 0.0f;
}

__global__ void fill_normal_horizontal_band_kernel(
    float* __restrict__ normal_map, int start_row, int end_row)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = start_row + blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= 400 || y >= end_row) return;
    int idx = y * 400 + x;
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

    if (x <= 0 || x >= 399 || y <= 0 || y >= 399) return;

    int idx = y * 400 + x;

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
    if (x >= 400 || y >= 400) return;

    int idx = y * 400 + x;
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
            if (nx <= 0 || nx >= 399 || ny <= 0 || ny >= 399) continue;
            int nidx = ny * 400 + nx;
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
    if (x >= 400 || y >= 400) return;

    int idx = y * 400 + x;
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
                if (nx <= 0 || nx >= 399 || ny <= 0 || ny >= 399) continue;
                int nidx = ny * 400 + nx;

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
    if (x <= 0 || x >= 399 || y <= 0 || y >= 399) return;

    float sum = 0.0f;
    int count = 0;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int nx = x + dx;
            int ny = y + dy;
            if (nx <= 0 || nx >= 399 || ny <= 0 || ny >= 399) continue;
            int nidx = ny * 400 + nx;
            float v = input[nidx];
            if (v > 0.0f) {  // только валидные
                sum += v;
                count++;
            }
        }
    }
    output[y * 400 + x] = (count > 0) ? (sum / count) : 0.0f;
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
    int y = blockIdx.y * blockDim.y + threadIdx.y;  // <-- Y вместо Z
    if (x >= Map_X || y >= Map_Y) return;

    const float cellSize = 100.0f;   // мм на ячейку

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
            xl[nl] = dc * cellSize;
            yl[nl] = dr * cellSize;
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
    getSlopeOfPoints(hl, xl, yl, nl, sx, sy);  // <-- sy вместо sz
    float localSlope = sqrtf(sx * sx + sy * sy);

    float heightRange = (raw_max - raw_min) / cellSize;

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
                xg[ng] = dc * cellSize;
                yg[ng] = dr * cellSize;
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
    int y = blockIdx.y * blockDim.y + threadIdx.y;  // <-- Y вместо Z
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
    if (x >= 400 || y >= 400) return;

    uint32_t idx = y * 400 + x;  // <-- y * 400 + x
    uint8_t cell = data[idx];

    if (cell > 252) return;

    int16_t dx[4] = {0, 0, -1, 1};
    int16_t dy[4] = {-1, 1, 0, 0};  // <-- dx/dy (плоскость XY)

    for (int8_t i = 0; i < 4; ++i) {
        int16_t nx = x + dx[i];
        int16_t ny = y + dy[i];

        if (nx < 0 || nx >= 400 || ny < 0 || ny >= 400) {
            data[idx] = 254;
            return;
        }

        uint8_t neighbor = data[ny * 400 + nx];  // <-- ny * 400 + nx

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
    int y = blockIdx.y * blockDim.y + threadIdx.y;  // <-- Y вместо Z
    if (x >= Map_X || y >= Map_Y) return;

    int idx = y * Map_X + x;  // <-- y * 400 + x

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

    if (texCol < 200 && row < 400) {
        int col0 = texCol * 2;
        int col1 = col0 + 1;

        float fa = heightmap[row * 400 + col0];
        float fb = heightmap[row * 400 + col1];
        float va = is_valid[row * 400 + col0];
        float vb = is_valid[row * 400 + col1];

        // ✅ Отсекаем по is_valid, а не по знаку высоты
        int16_t a_raw = (va < 0.5f) ? -32768 : (int16_t)(fa * 1000.0f);
        int16_t b_raw = (vb < 0.5f) ? -32768 : (int16_t)(fb * 1000.0f);

        uint32_t packed = ((uint32_t)(uint16_t)a_raw << 16) | (uint32_t)(uint16_t)b_raw;
        out_packed[row * 200 + texCol] = __uint_as_float(packed);

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


// ============================================================================
// debug_map_geometry_kernel: Выводит геометрию карты (профили, срезы)
// ============================================================================
__global__ void debug_map_geometry_kernel(
    const float* __restrict__ elevation_map,   // [7][400][400]
    int frame_counter
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    const float* h = elevation_map + 0 * MAP_CELLS;      // L_HEIGHT
    const float* valid = elevation_map + 2 * MAP_CELLS;  // L_IS_VALID

    printf("\n========== [MAP GEOMETRY frame %d] ==========\n", frame_counter);

    // --- 1. Профиль по X (центральная строка, y = 200) ---
    printf("PROFILE X (y=200, every 10 cells):\n");
    printf("  x: ");
    for (int x = 0; x < 400; x += 10) {
        printf("%4d ", x);
    }
    printf("\n  h: ");
    for (int x = 0; x < 400; x += 10) {
        int idx = 200 * 400 + x;
        if (valid[idx] < 0.5f) printf("  -- ");
        else printf("%4.2f ", h[idx]);
    }
    printf("\n");

    // --- 2. Профиль по Y (центральный столбец, x = 199) ---
    printf("PROFILE Y (x=200, every 10 cells):\n");
    printf("  y: ");
    for (int y = 0; y < 400; y += 10) {
        printf("%4d ", y);
    }
    printf("\n  h: ");
    for (int y = 0; y < 400; y += 10) {
        int idx = y * 400 + 200;
        if (valid[idx] < 0.5f) printf("  -- ");
        else printf("%4.2f ", h[idx]);
    }
    printf("\n");

    // --- 3. Срез карты (грубая сетка 40x40, шаг 10 ячеек) ---
    printf("SLICE 40x40 (every 10 cells):\n");
    printf("     ");
    for (int x = 0; x < 400; x += 10) {
        printf("%3d", x / 10);
    }
    printf("\n");
    for (int y = 0; y < 400; y += 10) {
        printf("%3d: ", y / 10);
        for (int x = 0; x < 400; x += 10) {
            int idx = y * 400 + x;
            if (valid[idx] < 0.5f) printf("  .");
            else {
                float val = h[idx];
                if (val < -0.5f) printf("  -");
                else if (val < 0.0f) printf("  0");
                else if (val < 0.5f) printf("  +");
                else if (val < 1.0f) printf("  1");
                else if (val < 1.5f) printf("  2");
                else printf("  3");
            }
        }
        printf("\n");
    }

    printf("============================================\n\n");
}

// ============================================================================
// debug_elevation_map_kernel: Статистика по dev_elevation_map[L_HEIGHT]
// ============================================================================
__global__ void debug_elevation_map_kernel(
    const float* __restrict__ elevation_map,
    const float* __restrict__ heightmap_packed,
    float rover_height_m,
    int frame_counter
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    const float* h = elevation_map + 0 * MAP_CELLS;      // L_HEIGHT
    const float* valid = elevation_map + 2 * MAP_CELLS;  // L_IS_VALID

    printf("\n========== [MAP DEBUG frame %d] ==========\n", frame_counter);
    printf("rover_height_m = %.4f\n", rover_height_m);

    int valid_count = 0, unknown_count = 0;
    float fmin = 1e30f, fmax = -1e30f;
    float sum = 0.0f;

    for (int i = 0; i < MAP_CELLS; i++) {
        if (valid[i] < 0.5f) { unknown_count++; continue; }
        float val = h[i];
        valid_count++;
        if (val < fmin) fmin = val;
        if (val > fmax) fmax = val;
        sum += val;
    }

    printf("MAP HEIGHT (float, метры):\n");
    printf("  valid = %d, unknown = %d\n", valid_count, unknown_count);
    if (valid_count > 0) {
        printf("  fmin = %.4f, fmax = %.4f, avg = %.4f\n", fmin, fmax, sum / valid_count);
    }

    // Примеры
    auto print_cell = [&](const char* label, int x, int y) {
        int idx = y * 400 + x;
        printf("  MAP[%s] (%d,%d) = %.4f (valid=%.1f)\n",
               label, x, y, h[idx], valid[idx]);
    };
    print_cell("center", 200, 200);
    print_cell("north",  200, 0);
    print_cell("south",  200, 399);
    print_cell("west",   0,   200);
    print_cell("east",   399, 200);

    // Статистика по упакованному
    int packed_valid = 0, packed_unknown = 0;
    int pmin = 32767, pmax = -32768;
    for (int row = 0; row < 400; row++) {
        for (int tc = 0; tc < 200; tc++) {
            float pf = heightmap_packed[row * 200 + tc];
            uint32_t packed = __float_as_uint(pf);
            int16_t a = (int16_t)((packed >> 16) & 0xFFFFu);
            int16_t b = (int16_t)(packed & 0xFFFFu);
            if (a != -32768) { packed_valid++; if (a < pmin) pmin = a; if (a > pmax) pmax = a; }
            else packed_unknown++;
            if (b != -32768) { packed_valid++; if (b < pmin) pmin = b; if (b > pmax) pmax = b; }
            else packed_unknown++;
        }
    }
    printf("PACKED (uint16, мм):\n");
    printf("  valid = %d, unknown = %d\n", packed_valid, packed_unknown);
    if (pmin != 0xFFFFFFFFu) {
        printf("  pmin = %d, pmax = %d\n", pmin, pmax);
    }

    printf("===========================================\n\n");
}

// ============================================================================
// debug_lidar_kernel: Статистика по сырому лидару (int16_t, мм)
// ============================================================================
__global__ void debug_lidar_kernel(
    const int16_t* lidar_pcl, int num_points,
    int frame_counter
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    printf("\n========== [LIDAR DEBUG frame %d] ==========\n", frame_counter);
    printf("num_points = %d\n", num_points);

    if (num_points == 0) {
        printf("LIDAR IS EMPTY!\n");
        return;
    }

    // --- 1. Статистика по x, y, z ---
    int16_t xmin = 32767, xmax = -32768;
    int16_t ymin = 32767, ymax = -32768;
    int16_t zmin = 32767, zmax = -32768;
    long sum_x = 0, sum_y = 0, sum_z = 0;
    int nan_count = 0;

    for (int i = 0; i < num_points; i++) {
        int16_t x = lidar_pcl[i * 3 + 0];
        int16_t y = lidar_pcl[i * 3 + 1];
        int16_t z = lidar_pcl[i * 3 + 2];

        if (x == 0 && y == 0 && z == 0) { nan_count++; continue; }

        if (x < xmin) xmin = x; if (x > xmax) xmax = x;
        if (y < ymin) ymin = y; if (y > ymax) ymax = y;
        if (z < zmin) zmin = z; if (z > zmax) zmax = z;
        sum_x += x; sum_y += y; sum_z += z;
    }

    printf("RAW LIDAR (int16_t, мм):\n");
    printf("  X: [%d, %d], avg = %ld\n", xmin, xmax, sum_x / num_points);
    printf("  Y: [%d, %d], avg = %ld\n", ymin, ymax, sum_y / num_points);
    printf("  Z: [%d, %d], avg = %ld\n", zmin, zmax, sum_z / num_points);
    printf("  zero_points = %d\n", nan_count);

    // --- 2. Примеры: первые 5 точек ---
    printf("  First 5 points (x, y, z):\n");
    for (int i = 0; i < min(5, num_points); i++) {
        printf("    [%d] = (%d, %d, %d)\n",
               i,
               lidar_pcl[i * 3 + 0],
               lidar_pcl[i * 3 + 1],
               lidar_pcl[i * 3 + 2]);
    }

    printf("=============================================\n\n");
}

// ============================================================================
// debug_points_kernel: Статистика по dev_points (float, метры)
// ============================================================================
__global__ void debug_points_kernel(
    const float* points, int* point_count,
    int frame_counter
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    int n = *point_count;
    printf("\n========== [POINTS DEBUG frame %d] ==========\n", frame_counter);
    printf("point_count = %d\n", n);

    if (n == 0) {
        printf("POINTS IS EMPTY!\n");
        return;
    }

    // --- 1. Статистика по x, y, z (в метрах) ---
    float xmin = 1e30f, xmax = -1e30f;
    float ymin = 1e30f, ymax = -1e30f;
    float zmin = 1e30f, zmax = -1e30f;
    float sum_x = 0, sum_y = 0, sum_z = 0;

    for (int i = 0; i < n; i++) {
        float x = points[i * 3 + 0];
        float y = points[i * 3 + 1];
        float z = points[i * 3 + 2];

        if (x < xmin) xmin = x; if (x > xmax) xmax = x;
        if (y < ymin) ymin = y; if (y > ymax) ymax = y;
        if (z < zmin) zmin = z; if (z > zmax) zmax = z;
        sum_x += x; sum_y += y; sum_z += z;
    }

    printf("POINTS (float, метры):\n");
    printf("  X: [%.3f, %.3f], avg = %.3f\n", xmin, xmax, sum_x / n);
    printf("  Y: [%.3f, %.3f], avg = %.3f\n", ymin, ymax, sum_y / n);
    printf("  Z: [%.3f, %.3f], avg = %.3f\n", zmin, zmax, sum_z / n);

    // --- 2. Примеры: первые 5 точек ---
    printf("  First 5 points (x, y, z):\n");
    for (int i = 0; i < min(5, n); i++) {
        printf("    [%d] = (%.3f, %.3f, %.3f)\n",
               i,
               points[i * 3 + 0],
               points[i * 3 + 1],
               points[i * 3 + 2]);
    }

    // --- 3. Сколько точек попадает в карту (idx_x, idx_y)? ---
    int inside = 0, outside = 0;
    for (int i = 0; i < n; i++) {
        float x = points[i * 3 + 0];
        float y = points[i * 3 + 1];
        float x_em = x / 0.1f + 200.0f;
        float y_em = y / 0.1f + 200.0f;
        int idx_x = (int)x_em;
        int idx_y = (int)y_em;
        if (idx_x > 0 && idx_x < 399 && idx_y > 0 && idx_y < 399) inside++;
        else outside++;
    }
    printf("  Inside map: %d, Outside map: %d\n", inside, outside);

    printf("============================================\n\n");
}

__global__ void costmap_to_traversability_kernel(
    const uint8_t* __restrict__ dev_costmap,
    float* __restrict__ elevation_map)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= MAP_CELLS) return;
    uint8_t c = dev_costmap[i];
    float trav;
    if (c >= 253) trav = 0.0f;                              // unknown / lethal / inscribed
    else          trav = 1.0f - (float)c / 252.0f;          // 0..252 → 1..0
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

    const int W = 400;
    const int H = 400;

    // Клампим сдвиг
    if (shift_x >  W) shift_x =  W;
    if (shift_x < -W) shift_x = -W;
    if (shift_y >  H) shift_y =  H;
    if (shift_y < -H) shift_y = -H;

    int src_x0 = (shift_x > 0) ? 0         : -shift_x;
    int dst_x0 = (shift_x > 0) ? shift_x   : 0;
    int copy_w = (shift_x > 0) ? W - shift_x : W + shift_x;

    int src_y0 = (shift_y > 0) ? 0         : -shift_y;
    int dst_y0 = (shift_y > 0) ? shift_y   : 0;
    int copy_h = (shift_y > 0) ? H - shift_y : H + shift_y;

    if (copy_w <= 0 || copy_h <= 0) {
        float empty[7] = {0.0f, 1000.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f};
        dim3 block(16, 16);
        dim3 grid((W + 15) / 16, (H + 15) / 16);
        fill_vertical_band_kernel<<<grid, block>>>(
            dev_elevation_map, 0, W,
            empty[0], empty[1], empty[2], empty[3], empty[4], empty[5], empty[6]);
        cudaDeviceSynchronize();
        return;
    }

    for (int layer = 0; layer < 7; ++layer) {
        float* layer_ptr = dev_elevation_map + layer * MAP_CELLS;
        result = cudaMemcpy(temp, layer_ptr, MAP_CELLS * sizeof(float), cudaMemcpyDeviceToDevice);
        assert(result == cudaSuccess);
        result = cudaMemcpy2D(
            layer_ptr + dst_y0 * W + dst_x0, W * sizeof(float),
            temp      + src_y0 * W + src_x0, W * sizeof(float),
            copy_w * sizeof(float), copy_h,
            cudaMemcpyDeviceToDevice);
        assert(result == cudaSuccess);
    }

    const float V_HEIGHT  = 0.0f;
    const float V_VAR     = 1000.0f;
    const float V_VALID   = 0.0f;
    const float V_TRAV    = 1.0f;
    const float V_TIME    = 0.0f;
    const float V_UPPER   = 0.0f;
    const float V_ISUP    = 0.0f;

    dim3 block(16, 16);

    if (shift_x > 0) {
        dim3 grid(((shift_x + 15) / 16), (H + 15) / 16);
        fill_vertical_band_kernel<<<grid, block>>>(dev_elevation_map, 0, shift_x, V_HEIGHT, V_VAR, V_VALID, V_TRAV, V_TIME, V_UPPER, V_ISUP);
    }
    if (shift_x < 0) {
        int start = W + shift_x; if (start < 0) start = 0;
        dim3 grid(((W - start + 15) / 16), (H + 15) / 16);
        fill_vertical_band_kernel<<<grid, block>>>(dev_elevation_map, start, W, V_HEIGHT, V_VAR, V_VALID, V_TRAV, V_TIME, V_UPPER, V_ISUP);
    }
    if (shift_y > 0) {
        dim3 grid((W + 15) / 16, ((shift_y + 15) / 16));
        fill_horizontal_band_kernel<<<grid, block>>>(dev_elevation_map, 0, shift_y, V_HEIGHT, V_VAR, V_VALID, V_TRAV, V_TIME, V_UPPER, V_ISUP);
    }
    if (shift_y < 0) {
        int start = H + shift_y; if (start < 0) start = 0;
        dim3 grid((W + 15) / 16, ((H - start + 15) / 16));
        fill_horizontal_band_kernel<<<grid, block>>>(dev_elevation_map, start, H, V_HEIGHT, V_VAR, V_VALID, V_TRAV, V_TIME, V_UPPER, V_ISUP);
    }

    if (dev_normal_map != nullptr) {
        for (int comp = 0; comp < 3; ++comp) {
            float* comp_ptr = dev_normal_map + comp * MAP_CELLS;
            result = cudaMemcpy(temp, comp_ptr, MAP_CELLS * sizeof(float), cudaMemcpyDeviceToDevice);
            assert(result == cudaSuccess);
            result = cudaMemcpy2D(
                comp_ptr + dst_y0 * W + dst_x0, W * sizeof(float),
                temp      + src_y0 * W + src_x0, W * sizeof(float),
                copy_w * sizeof(float), copy_h,
                cudaMemcpyDeviceToDevice);
            assert(result == cudaSuccess);
        }
        // FIX 14: чистим полосы (иначе нормали — мусор, ломает wall-check)
        if (shift_x > 0) {
            dim3 g(((shift_x + 15) / 16), (H + 15) / 16);
            fill_normal_vertical_band_kernel<<<g, block>>>(dev_normal_map, 0, shift_x);
        }
        if (shift_x < 0) {
            int start = W + shift_x; if (start < 0) start = 0;
            dim3 g(((W - start + 15) / 16), (H + 15) / 16);
            fill_normal_vertical_band_kernel<<<g, block>>>(dev_normal_map, start, W);
        }
        if (shift_y > 0) {
            dim3 g((W + 15) / 16, ((shift_y + 15) / 16));
            fill_normal_horizontal_band_kernel<<<g, block>>>(dev_normal_map, 0, shift_y);
        }
        if (shift_y < 0) {
            int start = H + shift_y; if (start < 0) start = 0;
            dim3 g((W + 15) / 16, ((H - start + 15) / 16));
            fill_normal_horizontal_band_kernel<<<g, block>>>(dev_normal_map, start, H);
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
void process_costmap_cuda(const int16_t* lidar_pcl, int num_points,
    const float* sensor_pos,
    float* dev_elevation_map, float* dev_new_map, float* dev_normal_map,
    float* dev_error, float* dev_error_cnt,
    uint8_t* dev_costmap, uint8_t* costmap_host,
    uint32_t* costmap_rgba_host,
    float* heightmap_packed_host,
    float position_noise, float orientation_noise,
    float *out_h_min, float *out_h_max, float *out_rover_height_m)
{
    cudaError_t result;
    dim3 block(16, 16);
    dim3 grid((400 + 15) / 16, (400 + 15) / 16);

    // ========================================================================
    // ПАРАМЕТРЫ ЯДЕР (все настройки в одном месте)
    // ========================================================================

    // --- Основные параметры карты ---
    const float RESOLUTION              = 0.1f;     // м/ячейка
    const int   MAP_CELL_N              = 400;      // ширина/высота карты
    const int   MAP_CENTER              = 200;      // центр карты (= MAP_CELL_N / 2)
    const float SENSOR_NOISE_FACTOR    = 0.05f;    // v = noise * rz^2

    // --- Параметры Калмана / выбросов ---
    const float MAHALANOBIS_THRESH     = 2.0f;     // порог outlier rejection
    const float OUTLIER_VARIANCE       = 0.01f;    // дисперсия выброса
    const float DRIFT_COMP_VAR_INLIER  = 0.05f;    // drift_compensation_variance_inlier
    const float MAX_VARIANCE           = 100.0f;   // max_variance
    const float INITIAL_VARIANCE       = 1000.0f;  // initial_variance

    // --- Параметры Exclusion Area (ramp) ---
    const float MAX_HEIGHT_RANGE       = 0.3f;     // max_height_range
    const float RAMPED_A               = 0.3f;     // ramped_height_range_a
    const float RAMPED_B               = 1.0f;     // ramped_height_range_b
    const float RAMPED_C               = 0.2f;     // ramped_height_range_c
    const float MIN_VALID_DISTANCE     = 0.5f;     // min_valid_distance

    // --- Параметры Ray Casting / visibility cleanup ---
    const float WALL_NUM_THRESH        = 20.0f;    // wall_num_thresh
    const float MAX_RAY_LENGTH         = 10.0f;    // max_ray_length
    const float CLEANUP_STEP           = 0.1f;     // cleanup_step
    const float CLEANUP_COS_THRESH     = 0.1f;     // cleanup_cos_thresh

    // --- Параметры drift compensation (timing) ---
    const float DRIFT_ALPHA            = 0.1f;     // drift_compensation_alpha
    const float MAX_DRIFT              = 0.1f;     // max_drift
    const int   MIN_HEIGHT_DRIFT_CNT   = 100;      // min_height_drift_cnt
    const float POSITION_NOISE_THRESH  = 0.01f;    // position_noise_thresh
    const float ORIENTATION_NOISE_THRESH = 0.02f;  // orientation_noise_thresh (2x EMC)

    // --- Параметры update_variance (рост дисперсии во времени) ---
    const float VAR_INTERVAL           = 0.2f;     // 5 Hz (= 1 / update_variance_fps)
    const float TIME_VARIANCE          = 0.0001f;  // time_variance

    // --- Параметры update_time (L_TIME) ---
    const float TIME_INTERVAL_MS       = 0.1f;     // time_interval (сек)

    // --- Параметры overlap clearance ---
    const float CLEAR_RANGE_Z          = 2.0f;     // overlap_clear_range_z
    const float CLEAR_RANGE_XY         = 4.0f;     // overlap_clear_range_xy
    const float CLEAR_INITIAL_VARIANCE = 1000.0f;  // = INITIAL_VARIANCE

    // --- Параметры dilation (перед normal) ---
    const int   DILATION_SIZE          = 3;        // dilation_size

    // --- Параметры slope / costmap ---
    const float MAX_SLOPE              = 2.0f;     // lethal slope
    const float FLAT_SLOPE             = 0.1f;     // free slope
    const int   INSCRIBED_RADIUS       = 2;        // радиус вписанной окружности
    const int   INFLATION_RADIUS       = 4;        // радиус инфляции
    const float COST_SCALING_FACTOR    = 15.0f;    // фактор масштабирования стоимости
    const int   ESTIMATION_SIZE_SLOPE  = 1;        // estimation_size для heightmap_to_slope

    // --- Feature toggles ---
    const int   enable_edge_sharpen        = 1;
    const int   enable_visibility_cleanup  = 1;
    const int   enable_drift_compensation  = 1;
    const int   enable_overlap_clearance   = 1;

    // ========================================================================
    // ЛОКАЛЬНЫЕ БУФЕРЫ
    // ========================================================================
    float* dev_points;
    int* dev_point_count;
    float* dev_slope;
    float* dev_heightmap;
    float* dev_heightmap_packed;
    int* dev_minmax;
    int16_t* dev_lidar_pcl;
    uint8_t* dev_costmap_tmp;
    float* dev_sensor_pos;
    float* dev_dilated;



    result = cudaMalloc(&dev_sensor_pos, 3 * sizeof(float));
    assert(result == cudaSuccess);
    result = cudaMalloc(&dev_lidar_pcl, num_points * 3 * sizeof(int16_t));
    assert(result == cudaSuccess);
    result = cudaMalloc(&dev_points, num_points * 3 * sizeof(float));
    assert(result == cudaSuccess);
    result = cudaMalloc(&dev_point_count, sizeof(int));
    assert(result == cudaSuccess);
    result = cudaMalloc(&dev_slope, MAP_CELLS * sizeof(float));
    assert(result == cudaSuccess);
    result = cudaMalloc(&dev_heightmap, MAP_CELLS * sizeof(float));
    assert(result == cudaSuccess);
    result = cudaMalloc(&dev_heightmap_packed, 400 * 200 * sizeof(float));
    assert(result == cudaSuccess);
    result = cudaMalloc(&dev_dilated, MAP_CELLS * sizeof(float));
    assert(result == cudaSuccess);
    result = cudaMalloc(&dev_minmax, 2 * sizeof(unsigned int));
    assert(result == cudaSuccess);
    result = cudaMalloc(&dev_costmap_tmp, MAP_CELLS);
    assert(result == cudaSuccess);


    // === Тик L_TIME ===
    static auto last_var_update = std::chrono::steady_clock::now();
    auto now_var = std::chrono::steady_clock::now();
    float dt_var = std::chrono::duration<float>(now_var - last_var_update).count();
    if (dt_var >= VAR_INTERVAL) {
        update_variance_kernel<<<(MAP_CELLS + 255) / 256, 256>>>(
            dev_elevation_map, TIME_VARIANCE, MAX_VARIANCE);
        last_var_update = now_var;
    }

    // ========================================================================
    // Шаг 1: Лидар -> точки
    // ========================================================================
    result = cudaMemset(dev_point_count, 0, sizeof(int));
    assert(result == cudaSuccess);
    result = cudaMemcpy(dev_lidar_pcl, lidar_pcl, num_points * 3 * sizeof(int16_t), cudaMemcpyHostToDevice);
    assert(result == cudaSuccess);

    int blocks_pts = (num_points + 255) / 256;
    convert_lidar_to_points_kernel<<<blocks_pts, 256>>>(
        dev_lidar_pcl, num_points, dev_points, dev_point_count
    );
    result = cudaDeviceSynchronize();
    assert(result == cudaSuccess);

    // ========================================================================
    // Шаг 2: Drift Compensation
    // ========================================================================
    cudaMemset(dev_error, 0, sizeof(float));
    cudaMemset(dev_error_cnt, 0, sizeof(float));
    cudaMemset(dev_new_map, 0, MAP_LAYERS * MAP_CELLS * sizeof(float));
    cudaMemcpy(dev_sensor_pos, sensor_pos, 3 * sizeof(float), cudaMemcpyHostToDevice);

    int blocks_err = (num_points + 255) / 256;
    error_counting_kernel_cuda<<<blocks_err, 256>>>(
        dev_points, num_points, dev_sensor_pos,
        dev_elevation_map, dev_error, dev_error_cnt, dev_new_map,
        RESOLUTION, SENSOR_NOISE_FACTOR, MAHALANOBIS_THRESH, DRIFT_COMP_VAR_INLIER,
        0.9f,  // traversability_inlier (пока фикс)
        MIN_VALID_DISTANCE, MAX_HEIGHT_RANGE,
        RAMPED_A, RAMPED_B, RAMPED_C
    );
    result = cudaDeviceSynchronize();
    assert(result == cudaSuccess);

    float h_error, h_error_cnt;
    cudaMemcpy(&h_error, dev_error, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_error_cnt, dev_error_cnt, sizeof(float), cudaMemcpyDeviceToHost);

    if (enable_drift_compensation &&
            h_error_cnt > MIN_HEIGHT_DRIFT_CNT &&
            (position_noise > POSITION_NOISE_THRESH || orientation_noise > ORIENTATION_NOISE_THRESH)) {
        float mean_error = h_error / h_error_cnt;
        if (fabsf(mean_error) < MAX_DRIFT) {
            shift_map_z_kernel<<<(MAP_CELLS + 255) / 256, 256>>>(dev_elevation_map, mean_error * DRIFT_ALPHA);
            result = cudaDeviceSynchronize();
            assert(result == cudaSuccess);
        }
    }

    // ========================================================================
    // Шаг 3: add_points (Калман + upper_bound + Ray Casting)
    // ========================================================================
    int blocks_add = (num_points + 255) / 256;
    add_points_kernel_cuda<<<blocks_add, 256>>>(
        dev_points, num_points, dev_sensor_pos,
        dev_elevation_map, dev_new_map, dev_normal_map,
        RESOLUTION, SENSOR_NOISE_FACTOR, MAHALANOBIS_THRESH, OUTLIER_VARIANCE,
        WALL_NUM_THRESH, MAX_RAY_LENGTH, CLEANUP_STEP,
        MIN_VALID_DISTANCE, MAX_HEIGHT_RANGE, CLEANUP_COS_THRESH,
        RAMPED_A, RAMPED_B, RAMPED_C,
        enable_edge_sharpen,
        enable_visibility_cleanup
    );
    result = cudaDeviceSynchronize();
    assert(result == cudaSuccess);

    // ========================================================================
    // Шаг 4: average_map
    // ========================================================================

    clear_counts_kernel<<<(MAP_CELLS + 255) / 256, 256>>>(dev_new_map);
    result = cudaDeviceSynchronize();
    assert(result == cudaSuccess);

    average_map_kernel_cuda<<<(MAP_CELLS + 255) / 256, 256>>>(
        dev_new_map, dev_elevation_map, MAX_VARIANCE, INITIAL_VARIANCE
    );
    result = cudaDeviceSynchronize();
    assert(result == cudaSuccess);

    // ========================================================================
    // Шаг 5: normal_filter
    // ========================================================================

    dilation_filter_kernel<<<grid, block>>>(
        dev_elevation_map + 5 * MAP_CELLS,   // upper_bound
        dev_elevation_map + 2 * MAP_CELLS,   // is_valid
        dev_elevation_map + 6 * MAP_CELLS,   // is_upper_bound
        dev_dilated,
        DILATION_SIZE
    );
    result = cudaDeviceSynchronize();
    assert(result == cudaSuccess);

    normal_filter_kernel_cuda<<<grid, block>>>(
            dev_dilated,          // 1 слой — для градиента
            dev_elevation_map,    // 7 слоёв — для is_valid / is_upper
            dev_normal_map,
            RESOLUTION
        );
    result = cudaDeviceSynchronize();
    assert(result == cudaSuccess);

    // ========================================================================
    // Шаг 6: overlap_clearance
    // ========================================================================
    // В process_costmap_cuda:
    int clear_range_cells = ((int)(CLEAR_RANGE_XY / RESOLUTION)) / 2;
    int total = (2 * clear_range_cells + 1) * (2 * clear_range_cells + 1);
    if (enable_overlap_clearance) {
        overlap_clearance_kernel<<<(total + 255) / 256, 256>>>(
        dev_elevation_map, sensor_pos[2], CLEAR_RANGE_Z, CLEAR_INITIAL_VARIANCE,
        MAP_CENTER, MAP_CENTER, clear_range_cells
        );
    }
    result = cudaDeviceSynchronize();
    assert(result == cudaSuccess);

    // ========================================================================
    // Шаг 7: min_filter (inpainting) — опционально
    // ========================================================================
    // for (int i = 0; i < 3; i++) {
    //     min_filter_kernel<<<grid, block>>>(
    //         dev_elevation_map, dev_heightmap, dev_costmap_tmp, 3
    //     );
    //     cudaDeviceSynchronize();
    //     cudaMemcpy(dev_elevation_map, dev_heightmap, MAP_CELLS * sizeof(float), cudaMemcpyDeviceToDevice);
    // }

    // ========================================================================
    // Шаг 8: smooth_filter — опционально
    // ========================================================================
    // smooth_filter_kernel<<<grid, block>>>(dev_elevation_map, dev_heightmap);
    // cudaDeviceSynchronize();
    // cudaMemcpy(dev_elevation_map, dev_heightmap, MAP_CELLS * sizeof(float), cudaMemcpyDeviceToDevice);

    // ========================================================================
    // Шаг 9: heightmap для визуализации
    // ========================================================================
    int init_mm[2] = { 32767, -32768 };
    cudaMemcpy(dev_minmax, init_mm, 2 * sizeof(int), cudaMemcpyHostToDevice);


    dim3 pack_grid((200 + 15) / 16, (400 + 15) / 16);
    pack_heightmap_minmax_kernel<<<pack_grid, block>>>( dev_elevation_map, dev_elevation_map + 2 * MAP_CELLS, dev_heightmap_packed, dev_minmax);
    result = cudaDeviceSynchronize();
    assert(result == cudaSuccess);

//    int num_elements = 400 * 200;
//    int blocks_fill = (num_elements + 255) / 256;
//    fill_heightmap_packed_kernel<<<blocks_fill, 256>>>(dev_heightmap_packed, num_elements, -32768);

    cudaMemcpy(heightmap_packed_host, dev_heightmap_packed,
               400 * 200 * sizeof(float), cudaMemcpyDeviceToHost);

    int h_minmax_u16[2];
    cudaMemcpy(h_minmax_u16, dev_minmax, 2 * sizeof(int), cudaMemcpyDeviceToHost);

    // === ВЫЧИСЛЯЕМ h_min, h_max (в метрах, знаковые) ===
    int h_min_int = h_minmax_u16[0];  // int (из int*)
    int h_max_int = h_minmax_u16[1];

    *out_h_min = (float)h_min_int * 0.001f;  // мм -> метры
    *out_h_max = (float)h_max_int * 0.001f;

    if (*out_h_max - *out_h_min < 0.05f) *out_h_max = *out_h_min + 0.5f;

    // === rover_height_m (просто 0, или sensor_pos[2]) ===
    *out_rover_height_m = 0.0f;  // временно

    // ========================================================================
    // Шаг 10: slope
    // ========================================================================
    heightmap_to_slope_cuda<<<grid, block>>>(
        dev_elevation_map, dev_elevation_map + 2 * MAP_CELLS, dev_slope,
        ESTIMATION_SIZE_SLOPE
         );
    result = cudaDeviceSynchronize();
    assert(result == cudaSuccess);

    // ========================================================================
    // Шаг 11: costmap из slope
    // ========================================================================
    slope_to_costmap_cuda<<<grid, block>>>(
        dev_slope, dev_elevation_map + 2 * MAP_CELLS, dev_costmap, MAX_SLOPE, FLAT_SLOPE
    );
    result = cudaDeviceSynchronize();
    assert(result == cudaSuccess);

    costmap_to_traversability_kernel<<<(MAP_CELLS + 255) / 256, 256>>>(
        dev_costmap, dev_elevation_map);
    result = cudaDeviceSynchronize();
    assert(result == cudaSuccess);

    // ========================================================================
    // Шаг 12: контуры
    // ========================================================================
    findContoursKernel_2d<<<grid, block>>>(dev_costmap);
    result = cudaDeviceSynchronize();
    assert(result == cudaSuccess);

    // ========================================================================
    // Шаг 13: инфляция
    // ========================================================================
    cudaMemcpy(dev_costmap_tmp, dev_costmap, MAP_CELLS, cudaMemcpyDeviceToDevice);
    inflate_costmap_cuda<<<grid, block>>>(
        dev_costmap_tmp, dev_costmap, INSCRIBED_RADIUS, INFLATION_RADIUS, COST_SCALING_FACTOR
    );
    result = cudaDeviceSynchronize();
    assert(result == cudaSuccess);


    // --- Раскраска costmap на GPU ---
    uint32_t* dev_rgba = nullptr;
    result = cudaMalloc(&dev_rgba, MAP_CELLS * sizeof(uint32_t));
    assert(result == cudaSuccess);

    colorize_costmap_kernel<<<grid, block>>>(
        dev_costmap, dev_rgba, 400, 400);
    result = cudaDeviceSynchronize();
    assert(result == cudaSuccess);

    // Копия RGBA на CPU (640 КБ вместо 160 КБ costmap)
    cudaMemcpy(costmap_rgba_host, dev_rgba,
               MAP_CELLS * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    cudaFree(dev_rgba);

    // ========================================================================
    // Шаг 14: копия costmap на CPU
    // ========================================================================
    cudaMemcpy(costmap_host, dev_costmap, MAP_CELLS, cudaMemcpyDeviceToHost);

    // === Отладка (раз в 100 кадров) ===
    static int debug_frame = 0;
    debug_frame++;
    if (debug_frame % 100 == 0) {
        std::cout << "\n========== [PARAMS DEBUG frame " << debug_frame << "] ==========\n";
        std::cout << "num_points = " << num_points << "\n";
        std::cout << "position_noise = " << position_noise << "\n";
        std::cout << "orientation_noise = " << orientation_noise << "\n";
        std::cout << "h_min = " << *out_h_min << ", h_max = " << *out_h_max << std::endl;
        std::cout << "=============================================\n\n";

        debug_lidar_kernel<<<1, 1>>>(dev_lidar_pcl, num_points, debug_frame);
        cudaDeviceSynchronize();

        debug_points_kernel<<<1, 1>>>(dev_points, dev_point_count, debug_frame);
        cudaDeviceSynchronize();

        debug_elevation_map_kernel<<<1, 1>>>(
            dev_elevation_map, dev_heightmap_packed, *out_rover_height_m, debug_frame
        );
        cudaDeviceSynchronize();

        debug_map_geometry_kernel<<<1, 1>>>(dev_elevation_map, debug_frame);
        cudaDeviceSynchronize();
    }

    // ========================================================================
    // Освобождение локальных буферов
    // ========================================================================
    cudaFree(dev_lidar_pcl);
    cudaFree(dev_sensor_pos);
    cudaFree(dev_points);
    cudaFree(dev_point_count);
    cudaFree(dev_slope);
    cudaFree(dev_heightmap);
    cudaFree(dev_heightmap_packed);
    cudaFree(dev_dilated);
    cudaFree(dev_minmax);
    cudaFree(dev_costmap_tmp);

    cudaDeviceSynchronize();
}
