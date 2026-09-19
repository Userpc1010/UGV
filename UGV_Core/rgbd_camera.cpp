#include "rgbd_camera.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>
#include <iostream>
#include <time.h>

extern "C"
void init_cuda_map(uint8_t **dev_voxels, uint8_t **dev_costmap);

extern "C"
void cleanup_cuda_map(uint8_t *dev_voxels, uint8_t *dev_costmap);

extern "C"
void deproject_depth_cuda(uint32_t **serialization_point_1, uint32_t *world_counter, uint8_t *dev_voxels,
                          const rs2_intrinsics &intrin, const uint16_t *depth, float depth_scale,
                          double w, double x, double y, double z,
                          int16_t x_, int16_t y_, int16_t z_,
                          const uint16_t *keyframe_pcl, uint32_t keyframe_size,
                          const int16_t  *lidar_pcl,    uint32_t lidar_size);

extern "C"
void process_costmap_cuda(uint8_t *dev_voxels, uint8_t *dev_costmap, uint8_t *costmap_host, int16_t current_y);


uint32_t floatBitsToUint_memcpy(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(float));
    return u;
}

float UintBitsToFloat_memcpy(uint32_t u) {
    float f;
    // Copy the raw bytes from the uint32_t to the float variable
    std::memcpy(&f, &u, sizeof(float));
    return f;
}

float pack_rgba_to_float(uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
    uint32_t packed = (r << 24) | (g << 16) | (b << 8) | a;
    return UintBitsToFloat_memcpy(packed);
}

int16_t Processor::signum(int16_t x) {
  return x == 0 ? 0 : x < 0 ? -1 : 1;
}

float Processor::mod(float value, float modulus) {
  return fmod(fmod(value, modulus) + modulus, modulus);
}

float Processor::intbound(float s, float ds) {
  // Find the smallest positive t such that s+t*ds is an integer.
  if (ds < 0) {
    return intbound(-s, -ds);
  } else {
    s = mod(s, 1);
    // problem is now s+t*ds = 1
    return (1 - s) / ds;
  }
}

void Processor::Raycast (const uint16_t start[3], const uint16_t end[3], const uint16_t min[3], const uint16_t max[3], std::vector<XYZ>* output)
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

    float direction_x = (float)(endX - start[0]);
    float direction_y = (float)(endY - start[1]);
    float direction_z = (float)(endZ - start[2]);

    float maxDist  = (direction_x * direction_x) + (direction_y * direction_y) + (direction_z * direction_z);

    // Direction to increment x,y,z when stepping.
    int16_t stepX = signum(dx);
    int16_t stepY = signum(dy);
    int16_t stepZ = signum(dz);

    float fdx = (float)(dx);
     float fdy = (float)(dy);
      float fdz = (float)(dz);

    // See description above. The initial values depend on the fractional
    // part of the origin.
    float tMaxX = intbound((float)(start[0]), fdx);
    float tMaxY = intbound((float)(start[1]), fdy);
    float tMaxZ = intbound((float)(start[2]), fdz);

    // The change in t when taking a step (always positive).
    float tDeltaX = (float)(stepX) / fdx;
    float tDeltaY = (float)(stepY) / fdy;
    float tDeltaZ = (float)(stepZ) / fdz;

  // Avoids an infinite loop.
  if (stepX == 0 && stepY == 0 && stepZ == 0) return;

  while (true) {

      if (x >= min[0] && x < max[0] && y >= min[1] && y < max[1] && z >= min[2] && z < max[2]) {

          XYZ data; data.x = x; data.y = y; data.z = z;

          output->push_back(data);

              float dir_x = (float)(x - start[0]);
              float dir_y = (float)(y - start[1]);
              float dir_z = (float)(z - start[2]);

             float dist  = (dir_x * dir_x) + (dir_y * dir_y) + (dir_z * dir_z);

             if (dist > maxDist) return;

                if (output->size() > 574) {
                  std::cerr << "Error, too many racyast voxels." << std::endl;
                  throw std::out_of_range("Too many raycast voxels");
                }
      }
      else break;

      if (x == endX && y == endY && z == endZ) break;

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

Processor::Processor(QObject *parent)
{
    qRegisterMetaType< std::vector<std::vector<cv::Point2f>> >("std::vector<std::vector<cv::Point2f>>");

    connect(this, SIGNAL(ArucoTrackerOut(std::vector<std::vector<cv::Point2f>>)), this, SLOT(ArucoTrackerIn(std::vector<std::vector<cv::Point2f>>)));

     //astar = new Astar(this);
     state_lattice_ = new StateLattice(this);

     colorizer = new CostmapColorizer();

    last_pos_x = 0.0;
    last_pos_y = 0.0;
    last_pos_z = 0.0;

    accumulated_dx = 0.0;
    accumulated_dy = 0.0;
    accumulated_dz = 0.0;

    // Last timestamp we visualized at
    last_visualization_timestamp = 0;

    process_rgbd = true;

    world_counter = 0;

    serialization_point = nullptr;
    dev_voxels = nullptr;
    dev_costmap = nullptr;
    costmap = nullptr;

   // Create the dictionary from the same dictionary the marker was generated.
//   cv::aruco::Dictionary dir = cv::aruco::getPredefinedDictionary(cv::aruco::PredefinedDictionaryType(cv::aruco::DICT_4X4_50));
//   dictionary = cv::Ptr<cv::aruco::Dictionary>(&dir);
}

Processor::~Processor()
{
 process_rgbd = false;
 pipe.stop();
 cleanup_cuda_map(dev_voxels, dev_costmap);
 dev_voxels = nullptr;
 dev_costmap = nullptr;
 if (costmap) { delete[] costmap; costmap = nullptr; }
 if (serialization_point) { free(serialization_point); serialization_point = nullptr; }
 if(ptr_guard_vertices_data){ ptr_guard_vertices_data = false; delete [] vertices_data;}
 if(ptr_guard_color_data){ptr_guard_color_data = false;  delete [] color_data;}
}

void Processor::IMU_Get_Euler_Angle(Quat buff)
{
    float CBn[5] = {0.0f};

    float q0q0 = buff.w * buff.w;

    //x-y-z
    CBn[0] = 2.0f * (q0q0 + buff.z * buff.z) - 1.0f;
    CBn[1] = 2.0f * (-buff.x * +buff.z - buff.w * -buff.y);
    CBn[2] = 2.0f * (-buff.x * buff.w - buff.z * -buff.y);
    //CBn[3] = 2.0f * (q1 * q2 - q0 * q3);
    //CBn[4] = 2.0f * (q0q0 + q2 * q2) - 1.0f;
    CBn[3] = 2.0f * (-buff.y * buff.z + buff.w * -buff.x);
    //CBn[6] = 2.0f * (q1 * q3 + q0 * q2);
    //CBn[7] = 2.0f * (q2 * q3 - q0 * q1);
    CBn[4] = 2.0f * (q0q0 + buff.z * buff.z) - 1.0f;

    //roll
    roll = atan2(-CBn[3], CBn[4]);

    if (roll > Quat_PI ) roll -= Quat_TWOPI;
    if (roll < -Quat_PI) roll += Quat_TWOPI;

    //pitch
    pitch = atan2(-CBn[2], CBn[4]);

    if (pitch >  Quat_PI) pitch -= Quat_TWOPI;
    if (pitch < -Quat_PI) pitch += Quat_TWOPI;

    //yaw
    yaw = atan2(CBn[1], CBn[0]);

    if (yaw >  Quat_PI) yaw -= Quat_TWOPI;
    if (yaw < -Quat_PI) yaw += Quat_TWOPI;

    roll = roll * RADTODEG;
    pitch = pitch * RADTODEG;
    yaw = yaw * RADTODEG;
}

void Processor::goalPositionSet(QVector3D position, QQuaternion rotation)
{
   IMU_Get_Euler_Angle(Quat(odom.w, -odom.y, -odom.z, odom.x));

   Finding_start_end_of_the_map_2d(position.x(), -position.z(), rotation.toEulerAngles().y());
}

void Processor::WindowState(uint8_t state)
{
    this->state = state;
}

void Processor::timesync(uint64_t timebase, uint64_t time)
{
    last_lidar_sync_time = time;
    last_scan_timebase = timebase;
    frame_counter_at_last_timesync = hardware_frame_counter;

    if(!has_lidar_time)
    {
    first_scan_lidar = time;
    first_frame_camera = hardware_frame_counter + 1;
    has_lidar_time = true;
    cam_offset_ms = 0.0;
    return;
    }

    // Вычисляем сырое время камеры на момент прихода пакета
    double raw_camera_time_ms = (hardware_frame_counter - first_frame_camera) * (1000.0 / 30.0);

    // Время лидара от первого пакета (мс)
    double lidar_time_ms = (time - first_scan_lidar) / 1e6;

    // Ошибка: положительное = камера опережает лидар
    cam_offset_ms = (lidar_time_ms - raw_camera_time_ms);
}

void Processor::point_cloud_lidar(const uint16_t * keyframe_pcl, uint32_t size_keyframe,
                                  const int16_t * lidar_pcl, uint32_t size_pcl)
{
        // === Забираем синхронизированный кадр камеры ===
        rs2::frame depth;
        float depth_scale;

        {
            std::unique_lock<std::mutex> lk(imu_mutex);

            if (image_ready_ < 1) {
                std::cout << "[SYNC] Дроп: лидар без кадра камеры" << std::endl;
                return;
            }
            depth = pending_depth_frame_;
            depth_scale = pending_depth_scale_;
            image_ready_ = 0;

            lk.unlock();
        }

        const uint16_t *depth_data = (const uint16_t*)depth.get_data();

        // === Освобождаем старые буферы визуализации ===
        if (ptr_guard_vertices_data) { delete[] vertices_data; ptr_guard_vertices_data = false; }
        if (ptr_guard_color_data)    { delete[] color_data;    ptr_guard_color_data = false; }

        // === Расчёт delta_cells ===
        int16_t delta_cells_x = 0;
        int16_t delta_cells_y = 0;
        int16_t delta_cells_z = 0;

        double delta_x = expRunningAverage(odom.x_pos, filter_integrate_x, filter_wegiht_x) - last_pos_x;
        accumulated_dx += delta_x;
        if (fabs(accumulated_dx) >= 0.1) {
            delta_cells_x = static_cast<int16_t>(floor(accumulated_dx * Map_scale));
            accumulated_dx -= delta_cells_x / Map_scale;
        }
        last_pos_x = odom.x_pos;

        double delta_y = expRunningAverage(odom.y_pos, filter_integrate_y, filter_wegiht_y) - last_pos_y;
        accumulated_dy += delta_y;
        if (fabs(accumulated_dy) >= 0.1) {
            delta_cells_y = static_cast<int16_t>(floor(accumulated_dy * Map_scale));
            accumulated_dy -= delta_cells_y / Map_scale;
        }
        last_pos_y = odom.y_pos;

        double delta_z = expRunningAverage(odom.z_pos, filter_integrate_z, filter_wegiht_z) - last_pos_z;
        accumulated_dz += delta_z;
        if (fabs(accumulated_dz) >= 0.1) {
            delta_cells_z = static_cast<int16_t>(floor(accumulated_dz * Map_scale));
            accumulated_dz -= delta_cells_z / Map_scale;
        }
        last_pos_z = odom.z_pos;

        // === CUDA: депроекция + вокселизация (камера + лидар) + сдвиг ===
        deproject_depth_cuda(&serialization_point, &world_counter,
                             dev_voxels,
                             intrinsics_depth, depth_data, depth_scale,
                             odom.w, -odom.y, -odom.z, odom.x,
                             delta_cells_x, -delta_cells_y, -delta_cells_z,
                             keyframe_pcl, size_keyframe,
                             lidar_pcl, size_pcl);

        // === Costmap ===
        int16_t current_y = 44;  // TODO: высота робота над землёй
//        process_costmap_cuda(dev_voxels, dev_costmap, costmap, current_y);

        // === Формирование массивов для визуализатора ===
        vertices_data = new GLfloat[world_counter]; ptr_guard_vertices_data = true;
        color_data    = new GLfloat[world_counter]; ptr_guard_color_data = true;

        for (uint32_t i = 0; i < world_counter; i++) {
            uint32_t idx = serialization_point[i];
            vertices_data[i] = UintBitsToFloat_memcpy(idx);
            color_data[i] = 0.0f;
        }

        // === Визуализация costmap ===
        QImage img = colorizer->colorizeCostmap(costmap, 400, 400);
        if (state == 2) DisplayingCostMap(img, QQuaternion(odom.w, -odom.y, odom.z, -odom.x));

        // === Визуализация вокселей ===
        if (state == 0) emit DisplayingCubes(vertices_data, color_data, world_counter,
                                              QQuaternion(odom.w, -odom.y, odom.z, -odom.x));

        // === Планировщик ===
        state_lattice_->updateCostmapFromData(costmap);

        // === Освобождение списка индексов ===
        free(serialization_point);
        serialization_point = nullptr;

        // === Аккумулируем смещение ===
        total_offset_x += delta_cells_x;
        total_offset_y += delta_cells_y;
        total_offset_z += delta_cells_z;

        QVector3D current_offset(
            static_cast<float>(total_offset_x),
            static_cast<float>(total_offset_y),
            static_cast<float>(total_offset_z)
        );

        if (total_offset_x != last_sent_offset_x ||
            total_offset_y != last_sent_offset_y ||
            total_offset_z != last_sent_offset_z) {
            emit updateMapOffset(current_offset);
            last_sent_offset_x = total_offset_x;
            last_sent_offset_y = total_offset_y;
            last_sent_offset_z = total_offset_z;
        }
}

//void Processor::ArucoTrackerIn(std::vector<std::vector<cv::Point2f>> corners)
//{
//        std::vector<cv::Vec3d> rvecs, tvecs;

//        //cv::aruco::estimatePoseSingleMarkers(corners, 0.065f, camera_matrix, dist_coeffs, rvecs, tvecs);

//        cv::aruco::estimatePoseSingleMarkers(corners, 0.2755f, camera_matrix, dist_coeffs, rvecs, tvecs);

//        //std::cout << "Translation: " << tvecs[0] << "\tRotation: " << rvecs[0] << "\n";

//        Aruco_point[0] = (Rotation_matrix[0] * tvecs[0](0) + Rotation_matrix[3] * tvecs[0](1) + Rotation_matrix[6] * tvecs[0](2));

//        Aruco_point[1] = (Rotation_matrix[1] * tvecs[0](0) + Rotation_matrix[4] * tvecs[0](1) + Rotation_matrix[7] * tvecs[0](2));

//        Aruco_point[2] = (Rotation_matrix[2] * tvecs[0](0) + Rotation_matrix[5] * tvecs[0](1) + Rotation_matrix[8] * tvecs[0](2));

//        Aruco_point[0] += Translation[0];

//        Aruco_point[1] += Translation[1];

//        Aruco_point[2] += Translation[2];

//        Aruco_point[0] *= Map_scale;

//        Aruco_point[1] *= Map_scale;

//        Aruco_point[2] *= Map_scale;

//        if(state == 0) DrawAruco(QVector3D(Aruco_point[0] + 199.0f, -Aruco_point[1] - 49.0f, -Aruco_point[2] - 199.0f));

//        Finding_start_end_of_the_map (Aruco_point[0], Aruco_point[1], Aruco_point[2]);

//}

void Processor::ArucoTrackerIn(std::vector<std::vector<cv::Point2f>> corners)
{
    static const float FOLLOW_DISTANCE = 1.5f;
    static const float SAFETY_DISTANCE = 1.0f;
    static const float HYSTERESIS = 0.2f;

    std::vector<cv::Vec3d> rvecs, tvecs;
    cv::aruco::estimatePoseSingleMarkers(corners, 0.206f, camera_matrix, dist_coeffs, rvecs, tvecs);

    // Получаем позицию маркера
    Aruco_point[0] = (Rotation_matrix[0] * tvecs[0](0) + Rotation_matrix[3] * tvecs[0](1) + Rotation_matrix[6] * tvecs[0](2));
    Aruco_point[1] = (Rotation_matrix[1] * tvecs[0](0) + Rotation_matrix[4] * tvecs[0](1) + Rotation_matrix[7] * tvecs[0](2));
    Aruco_point[2] = (Rotation_matrix[2] * tvecs[0](0) + Rotation_matrix[5] * tvecs[0](1) + Rotation_matrix[8] * tvecs[0](2));

    Aruco_point[0] += Translation[0];
    Aruco_point[1] += Translation[1];
    Aruco_point[2] += Translation[2];

    Aruco_point[0] *= Map_scale;
    Aruco_point[1] *= Map_scale;
    Aruco_point[2] *= Map_scale;

    //std::cout << "Translation: " << tvecs[0] << "\tRotation: " << rvecs[0] << "\n";

    if(state == 0) DrawAruco(QVector3D(Aruco_point[0] + 199.0f, -Aruco_point[1] - 49.0f, -Aruco_point[2] - 199.0f));

    if(state == 2) DrawAruco(QVector3D(Aruco_point[0] + 199.0f, -Aruco_point[1] - 49.0f, -Aruco_point[2] - 199.0f));

    float robot_x = Map_center_X;
    float robot_z = Map_center_Z;
    float marker_x = Aruco_point[0] + Map_center_X;
    float marker_z = Aruco_point[2] + Map_center_Z;

    float dx = marker_x - robot_x;
    float dz = marker_z - robot_z;
    float distance = sqrt(dx*dx + dz*dz);

    // Направление на маркер - РОБОТ ВСЕГДА ДОЛЖЕН СМОТРЕТЬ НА МАРКЕР
    float look_at_marker_yaw = atan2(dz, dx);

    float target_x, target_z;

    if (distance < SAFETY_DISTANCE * Map_scale) {
        // 1. СЛИШКОМ БЛИЗКО: отъезжаем задом, но продолжаем смотреть на маркер
        float back_distance = (SAFETY_DISTANCE * Map_scale - distance) + HYSTERESIS * Map_scale;
        target_x = robot_x - back_distance * (dx / distance);
        target_z = robot_z - back_distance * (dz / distance);
    }
    else if (distance > (FOLLOW_DISTANCE + HYSTERESIS) * Map_scale) {
        // 3. ДАЛЕКО: приближаемся вперед, смотрим на маркер
        target_x = marker_x - FOLLOW_DISTANCE * Map_scale * (dx / distance);
        target_z = marker_z - FOLLOW_DISTANCE * Map_scale * (dz / distance);
    }
    else if (distance < (FOLLOW_DISTANCE - HYSTERESIS) * Map_scale) {
        // 2. СЛИШКОМ БЛИЗКО (но не критично): отъезжаем до нужной дистанции
        float back_distance = (FOLLOW_DISTANCE * Map_scale - distance);
        target_x = robot_x - back_distance * (dx / distance);
        target_z = robot_z - back_distance * (dz / distance);
    }
    else {
        // 2. НОРМАЛЬНОЕ РАССТОЯНИЕ: стоим на месте, смотрим на маркер
        target_x = robot_x;
        target_z = robot_z;
    }

    // Всегда используем yaw = atan2(dz, dx) - смотреть на маркер
    Finding_start_end_of_the_map_2d(target_x, target_z, look_at_marker_yaw);
}


//void Processor::Finding_start_end_of_the_map(float target_pos_x, float target_pos_y, float target_pos_z)
//{
//    ClearLine();

//    std::vector<XYZ> output; XYZ buffer; std::vector<XYZ> ground;

//    uint16_t start[3] = {Map_center_X, Map_center_Y, Map_center_Z};  uint16_t end[3] = {target_pos_x + Map_center_X, target_pos_y + Map_center_Y, target_pos_z + Map_center_Z}; uint16_t min[3] = {0,0,0}; uint16_t max[3] = {Map_X,Map_Y,Map_Z};

////    qDebug()<<" x "<<target_pos_x + Map_center_X<<" y "<<target_pos_y + Map_center_Y<<" z "<< target_pos_z + Map_center_Z;

//      Raycast (start, end, min, max, &output);

//    for (uint16_t i = 0; i < output.size(); i++)
//    {
//     buffer = output[i];

//     for(uint8_t i = 0; i < Map_Y ; i++) if(voxels[get_value(buffer.x, i, buffer.z)]){ buffer.y = i - 1; ground.push_back(buffer); break;}

//    }

//    if(ground.size() > 3){

////       QVector3D point_1 = QVector3D(ground.front().x, -ground.front().y, -ground.front().z);
////       QVector3D point_2 = QVector3D(ground.back().x, -ground.back().y, -ground.back().z);

////       if(state == 0) emit DrawCube(point_1);
////       if(state == 0) emit DrawCube(point_2);

//      astar->GetPathFromGround(ground, voxels);
//    }
//}

void Processor::Finding_start_end_of_the_map_2d(float target_pos_x, float target_pos_y, float target_yaw)
{
    unsigned int start_x = Map_center_X;
    unsigned int start_y = Map_center_Z;
    unsigned int goal_x = static_cast<unsigned int>(target_pos_x);
    unsigned int goal_y = static_cast<unsigned int>(target_pos_y);

    // Проверяем границы цели
    if (goal_x >= Map_X || goal_y >= Map_Z) {
        qWarning() << "Цель выходит за границы карты";
        return;
    }

    // Функция проверки ячейки
    auto is_valid_cell = [this](unsigned int x, unsigned int y) -> bool {
        if (x >= Map_X || y >= Map_Z) return false;
        uint8_t cost = costmap[y * Map_X + x];
        return cost < 253;
    };

    // Вектор для точек луча
    std::vector<std::pair<unsigned int, unsigned int>> ray_points;

    // Собираем точки луча
    auto collect_points = [&ray_points, this](unsigned int offset) {
        ray_points.push_back({offset % Map_X, offset / Map_X});
    };

    // Трассируем луч
    try {
        raytraceLine(collect_points, start_x, start_y, goal_x, goal_y, Map_X);
    } catch (...) {
        qWarning() << "Ошибка при трассировке луча";
        return;
    }

    if (ray_points.empty()) {
        qWarning() << "Луч не содержит точек";
        return;
    }

    // Ищем первую валидную точку от начала
    std::pair<unsigned int, unsigned int> valid_start = {start_x, start_y};
    std::pair<unsigned int, unsigned int> valid_goal = {goal_x, goal_y};

    bool start_found = false;
    bool goal_found = false;

    // Прямой проход: ищем первую валидную точку от начала
    for (size_t i = 0; i < ray_points.size(); i++) {
        const auto& point = ray_points[i];

        // Ищем старт
        if (!start_found && is_valid_cell(point.first, point.second)) {
            valid_start = point;
            start_found = true;
        }

        // Если нашли старт, ищем цель с конца
        if (start_found && !goal_found) {
            // Ищем с конца до текущей позиции
            for (size_t j = ray_points.size() - 1; j >= i; j--) {
                const auto& end_point = ray_points[j];
                if (is_valid_cell(end_point.first, end_point.second)) {
                    valid_goal = end_point;
                    goal_found = true;
                    break;
                }
            }
            if (goal_found) break;
        }
    }

    // Если не нашли обе точки - выходим
    if (!start_found || !goal_found) {
        qWarning() << "Не найдены валидные точки на луче";
        return;
    }

    // Проверяем, что точки разные (хотя это нормально если совпадают)
    if (valid_start == valid_goal) {
        qDebug() << "Старт и цель совпадают после корректировки";
    }

    // Создаем Pose для планирования (используем Eigen::Vector2f)
    Pose start;
    Pose goal;

    // Eigen::Vector2f конструктор принимает (x, y)
    start.position = Eigen::Vector2f(
        static_cast<float>(valid_start.first),
        static_cast<float>(valid_start.second)
    );
    start.orientation = (90.0f + yaw) * DEGTORAD; // Текущий yaw робота в радианах

    goal.position = Eigen::Vector2f(
        static_cast<float>(valid_goal.first),
        static_cast<float>(valid_goal.second)
    );
    goal.orientation = (90.0f + target_yaw) * DEGTORAD; // Заданный yaw цели в радианах

    // Логирование для отладки
    qDebug() << "Скорректированные точки:";
    qDebug() << "  Start: (" << valid_start.first << "," << valid_start.second
             << ") yaw: " << start.orientation;
    qDebug() << "  Goal:  (" << valid_goal.first << "," << valid_goal.second
             << ") yaw: " << goal.orientation;

    // Запускаем планирование
    state_lattice_->asyncGetPathToTarget(start, goal);
}


COLOUR Processor::GetColour(float v, float vmin, float vmax)
{
    COLOUR c = {1.0f,1.0f,1.0f}; // white
      float dv;

      if (v < vmin)
         v = vmin;
      if (v > vmax)
         v = vmax;
      dv = vmax - vmin;

      if (v < (vmin + 0.25f * dv)) {
         c.r = 0.0f;
         c.g = 4.0f * (v - vmin) / dv;
      } else if (v < (vmin + 0.5f * dv)) {
         c.r = 0.0;
         c.b = 1.0f + 4.0f * (vmin + 0.25 * dv - v) / dv;
      } else if (v < (vmin + 0.75f * dv)) {
         c.r = 4.0f * (v - vmin - 0.5f * dv) / dv;
         c.b = 0.0f;
      } else {
         c.g = 1.0f + 4.0f * (vmin + 0.75f * dv - v) / dv;
         c.b = 0.0f;
      }

      return(c);
}

uint32_t Processor::get_value(const uint16_t x, const uint16_t y, const uint16_t z)
{
  uint32_t index = z * 40000 + y * 400 + x;
  return index;
}

double Processor::expRunningAverage(double &newVal, double &filVal, const double &wegiht)
{
  filVal += (newVal - filVal) * wegiht;
  return filVal;
}

rs2_vector Processor::interpolateMeasure(const double target_time, const rs2_vector current_data, const double current_time, const rs2_vector prev_data, const double prev_time)
{
    // If there are not previous information, the current data is propagated
      if(prev_time == 0)
      {
          return current_data;
      }

      rs2_vector increment;
      rs2_vector value_interp;

      if(target_time > current_time) {
          value_interp = current_data;
      }
      else if(target_time > prev_time)
      {
          increment.x = current_data.x - prev_data.x;
          increment.y = current_data.y - prev_data.y;
          increment.z = current_data.z - prev_data.z;

          double factor = (target_time - prev_time) / (current_time - prev_time);

          value_interp.x = prev_data.x + increment.x * factor;
          value_interp.y = prev_data.y + increment.y * factor;
          value_interp.z = prev_data.z + increment.z * factor;

          // zero interpolation
          value_interp = current_data;
      }
      else {
          value_interp = prev_data;
      }

      return value_interp;
}

void Processor::run()
{
    const cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);

    const cv::aruco::DetectorParameters detectorParams = cv::aruco::DetectorParameters();

    cv::FileStorage fs("/home/sencis/build-UGV-Desktop-Debug/UGV_Core/calibration_params.yml", cv::FileStorage::READ);

    fs["camera_matrix"] >> camera_matrix;

    fs["distortion_coefficients"] >> dist_coeffs;

    rs2::context ctx;
    rs2::device_list devices = ctx.query_devices();
    rs2::device selected_device;
    if (devices.size() == 0)
    {
        std::cerr << "No device connected, please connect a RealSense device" << std::endl;
        while(1)sleep(1000);
    }
    else selected_device = devices[0];

    std::vector<rs2::sensor> sensors = selected_device.query_sensors();
    // We can now iterate the sensors and print their names
    for (rs2::sensor sensor : sensors) {
        // Проверяем, является ли текущий сенсор модулем глубины
        if (auto depth_sensor = sensor.as<rs2::depth_sensor>()) {

            // Отключаем или настраиваем экспозицию
            depth_sensor.set_option(RS2_OPTION_DEPTH_AUTO_EXPOSURE_MODE, RS2_DEPTH_AUTO_EXPOSURE_ACCELERATED);
            depth_sensor.set_option(RS2_OPTION_ENABLE_AUTO_EXPOSURE, 1.0f);
            depth_sensor.set_option(RS2_OPTION_EMITTER_ENABLED, 1.0f);

            // Включаем Output Trigger (Режим Master для внешних устройств)
            if (depth_sensor.supports(RS2_OPTION_OUTPUT_TRIGGER_ENABLED)) {
                depth_sensor.set_option(RS2_OPTION_OUTPUT_TRIGGER_ENABLED, 1.0f);
                std::cout << "Output Trigger Enabled успешно включен!" << std::endl;
            }

            // Включаем Inter Cam Sync Mode: 1 (Master)
            if (depth_sensor.supports(RS2_OPTION_INTER_CAM_SYNC_MODE)) {
                depth_sensor.set_option(RS2_OPTION_INTER_CAM_SYNC_MODE, 1.0f);
                std::cout << "Inter Cam Sync Mode установлен в 1 (Master)" << std::endl;
            }
        }
        if (auto motion_sensor = sensor.as<rs2::motion_sensor>()) {
            motion_sensor.set_option(RS2_OPTION_ENABLE_MOTION_CORRECTION, 0);
        }
    }


  int width_img = 848, height_img = 480;

 // Declare RealSense pipeline, encapsulating the actual device and sensors
 //rs2::pipeline pipe;

 // Create a configuration for configuring the pipeline with a non default profile

 // RGB stream
 config.enable_stream(RS2_STREAM_COLOR, width_img, height_img,  RS2_FORMAT_BGR8, 30);

 // Depth stream
 config.enable_stream(RS2_STREAM_DEPTH, width_img, height_img, RS2_FORMAT_Z16, 30);

 //Stereo
 //config.enable_stream(RS2_STREAM_INFRARED, 1, 848, 480, RS2_FORMAT_Y8, 30);
 //config.enable_stream(RS2_STREAM_INFRARED, 2, 848, 480, RS2_FORMAT_Y8, 30);

 // IMU stream
 //config.enable_stream(RS2_STREAM_ACCEL, RS2_FORMAT_MOTION_XYZ32F, 200);
 //config.enable_stream(RS2_STREAM_GYRO, RS2_FORMAT_MOTION_XYZ32F, 200);

 // IMU callback

 bool reciv = false;


 double timestamp_image = -1.0;

 // start and stop just to get necessary profile
 rs2::pipeline_profile pipe_profile = pipe.start(config);
 pipe.stop();

 auto imu_callback = [&](const rs2::frame& frame)
 {


     if(rs2::frameset fs = frame.as<rs2::frameset>())
     {
         rs2::frame depth_f = fs.get_depth_frame();
         if (depth_f && depth_f.supports_frame_metadata(RS2_FRAME_METADATA_FRAME_COUNTER))
         {
             hardware_frame_counter = depth_f.get_frame_metadata(RS2_FRAME_METADATA_FRAME_COUNTER);
         }

         //Берем сырое текущее время камеры
         double raw_camera_time_ms = ( hardware_frame_counter - first_frame_camera ) * (1000.0 / 30.0); // 33.333 мс на кадр

         // Применяем смещение (обновляется в timesync)
         double current_frame_time_ms = raw_camera_time_ms;
         if (has_lidar_time)
         {
             current_frame_time_ms += cam_offset_ms;
         }

         // Переводим в наносекунды (абсолютное время, совместимое с last_lidar_sync_time)
         double current_frame_time = first_scan_lidar + current_frame_time_ms * 1e6;

         emit camsync (current_frame_time);

         // 4. Сдвиг относительно скорректированного времени
         uint64_t predicted_timebase = last_scan_timebase;
         if (has_lidar_time) {
             uint64_t frames_passed = hardware_frame_counter - frame_counter_at_last_timesync;
             predicted_timebase = last_scan_timebase + (frames_passed / 3) * 100000000ULL;
         }
         double independent_phase_ms = (current_frame_time - predicted_timebase) / 1e6 - 100.0;

         // Автоматическое определение целевого остатка
         if (has_lidar_time && target_remainder == -1 && hardware_frame_counter >= 20) {
             int rem = hardware_frame_counter % 3;
             phase_shifts[rem] += independent_phase_ms;
             phase_counts[rem]++;

             if (phase_counts[0] > 0 && phase_counts[1] > 0 && phase_counts[2] > 0) {
                 int best_rem = 0;
                 double best_err = std::numeric_limits<double>::max();
                 for (int i = 0; i < 3; i++) {
                     double avg = phase_shifts[i] / phase_counts[i];
                     double err = std::abs(avg - 50.0);  // цель — середина скана
                     if (err < best_err) {
                         best_err = err;
                         best_rem = i;
                     }
                 }
                 target_remainder = best_rem;
                 std::cout << "[SYSTEM] Фаза захвачена! target_remainder = " << best_rem
                           << ", фаза ≈ " << (phase_shifts[best_rem]/phase_counts[best_rem])
                           << " мс" << std::endl;
             }
         }

         // Логика фильтрации кадра по динамически определенному правилу
         bool is_target_frame = false;
         if (target_remainder != -1)
         {
             is_target_frame = ((hardware_frame_counter % 3) == target_remainder);

             if (is_target_frame)
             {
                 std::unique_lock<std::mutex> lock(imu_mutex);
                 pending_depth_frame_ = fs.get_depth_frame();
                 pending_depth_scale_ = fs.get_depth_frame().get_units();
                 image_ready_ = 1;
                 timestamp_image = fs.get_timestamp()*1e-3;
                 lock.unlock();

                 std::unique_lock<std::mutex> lock_c(color_mutex);
                 pending_color_frame_ = fs.get_color_frame();
                 lock_c.unlock();
             }
         }

         // В лог добавляем метку [TARGET], если это наш искомый кадр со сдвигом ~ 50 мс
//         std::cout << "[CAM_THREAD] кадр № " << hardware_frame_counter
//                          << (is_target_frame ? " [TARGET]" : "         ")
//                          << " | Время CAM: " << ((current_frame_time - first_scan_lidar) / 1000000.0) << " мс"
//                          << " | LIDAR_SYNC: " << ((last_lidar_sync_time - first_scan_lidar) / 1000000.0) << " мс"
//                          << " | Фаза: " << independent_phase_ms << " мс"
//                          << std::endl;
     }


     reciv = true;

 };

   pipe_profile = pipe.start(config, imu_callback);

   cam_stream = pipe_profile.get_stream(RS2_STREAM_DEPTH);

   intrinsics_depth = cam_stream.as<rs2::video_stream_profile>().get_intrinsics();

   init_cuda_map(&dev_voxels, &dev_costmap);

   costmap = new uint8_t [1600000];   // CPU-копия costmap (остаётся)


   cv::Mat Color;

 while(process_rgbd)
 {
   if(reciv) {


   {
   std::unique_lock<std::mutex> lk(color_mutex);
   if (image_ready_ == 1) {

   image_ready_ = 2;   // помечаем, что aruco забрал Color
   Color = cv::Mat(cv::Size(width_img, height_img), CV_8UC3,
                           (void*)pending_color_frame_.get_data(), cv::Mat::AUTO_STEP);
   }
     lk.unlock();
   }

   if(image_ready_ == 2) {

    std::vector<int> ids; bool marker_detected = false;

      std::vector<std::vector<cv::Point2f>> corners, rejectedCandidates;

      cv::aruco::ArucoDetector detector(dictionary, detectorParams);

      detector.detectMarkers(Color, corners, ids, rejectedCandidates);

      //if at least one marker detected

     for (size_t i = 0; i < ids.size(); ++i) if(ids[i] == 47){

      std::vector<std::vector<cv::Point2f>> corner; corner.push_back(corners[i]);

      marker_detected = true;

      if(state == 0) emit Invisible_Aruco(false);

      if(state == 2) emit Invisible_Aruco(false);

      emit ArucoTrackerOut(corners);

      }

     if(!marker_detected) {

       if(state == 0) emit Invisible_Aruco(true);

       if(state == 2) emit Invisible_Aruco(true);
     }
     }

     reciv = false;
   }
   else usleep(50);

 }

 pipe.stop();
}

void Processor::odometry_lidar(odometry data)
{
  odom = data;

  first_start = true;

  if(!first_start)
  {
      last_pos_x = odom.x_pos;
      last_pos_y = odom.y_pos;
      last_pos_z = odom.z_pos;

      filter_integrate_x = odom.x_pos;
      filter_integrate_y = odom.y_pos;
      filter_integrate_z = odom.z_pos;

  }
}


