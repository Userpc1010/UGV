#include "rgbd_camera.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>
#include <iostream>
#include <time.h>

extern "C" void init_cuda_map(float** dev_elevation_map, float** dev_new_map,
                              float** dev_normal_map, float** dev_error,
                              float** dev_error_cnt,
                              uint8_t** dev_costmap,
                              float** dev_shift_temp);

extern "C" void cleanup_cuda_map(float* dev_elevation_map, float* dev_new_map,
                                 float* dev_normal_map, float* dev_error,
                                 float* dev_error_cnt,
                                 uint8_t* dev_costmap,
                                 float* dev_shift_temp);

extern "C" void init_elevation_defaults(float* dev_elevation_map, float initial_variance);



extern "C" void process_costmap_cuda(
    const int16_t* lidar_pcl, int num_points,
    const float* sensor_pos,
    float* dev_elevation_map, float* dev_new_map, float* dev_normal_map,
    float* dev_error, float* dev_error_cnt,
    uint8_t* dev_costmap, uint8_t* costmap_host, uint32_t* costmap_rgba_host,
    float* heightmap_packed_host,
    float position_noise, float orientation_noise,
    float* out_h_min, float* out_h_max, float* out_rover_height_m
);

extern "C"
void shift_map_xy_cuda(float* dev_elevation_map, float* dev_normal_map,
                       float* temp,
                       int shift_x, int shift_y);
extern "C"
void shift_map_z_cuda(float* dev_elevation_map, float delta_z);

extern "C" void update_time_cuda(float* dev_elevation_map, float dt);

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


Processor::Processor(QObject *parent)
{
    qRegisterMetaType< std::vector<std::vector<cv::Point2f>> >("std::vector<std::vector<cv::Point2f>>");
    qRegisterMetaType<const float*>("const float*");

    connect(this, SIGNAL(ArucoTrackerOut(std::vector<std::vector<cv::Point2f>>)), this, SLOT(ArucoTrackerIn(std::vector<std::vector<cv::Point2f>>)));

     //astar = new Astar(this);
     state_lattice_ = new StateLattice(this);

    // Last timestamp we visualized at
    last_visualization_timestamp = 0;

    process_rgbd = true;

    dev_elevation_map = nullptr;
    dev_new_map = nullptr;
    dev_normal_map = nullptr;
    dev_error = nullptr;
    dev_error_cnt = nullptr;
    dev_costmap = nullptr;
    costmap = nullptr;
    heightmap_packed = nullptr;
    dev_shift_temp = nullptr;

   // Create the dictionary from the same dictionary the marker was generated.
//   cv::aruco::Dictionary dir = cv::aruco::getPredefinedDictionary(cv::aruco::PredefinedDictionaryType(cv::aruco::DICT_4X4_50));
//   dictionary = cv::Ptr<cv::aruco::Dictionary>(&dir);
}

Processor::~Processor()
{
 process_rgbd = false;
 pipe.stop();
 cleanup_cuda_map(dev_elevation_map, dev_new_map, dev_normal_map,
                  dev_error, dev_error_cnt,
                  dev_costmap,
                  dev_shift_temp);
 dev_shift_temp = nullptr;
 dev_elevation_map = nullptr;
 dev_new_map = nullptr;
 dev_normal_map = nullptr;
 dev_error = nullptr;
 dev_error_cnt = nullptr;
 dev_costmap = nullptr;
 if (costmap_rgba) { delete[] costmap_rgba; costmap_rgba = nullptr; }
 if (costmap) { delete[] costmap; costmap = nullptr; }
 if (heightmap_packed) { delete[] heightmap_packed; heightmap_packed = nullptr; }
}

void Processor::goalPositionSet(QVector3D position, QQuaternion rotation)
{
   Finding_start_end_of_the_map_2d(position.x(), -position.z(), rotation.toEulerAngles().y());
}

void Processor::WindowState(uint8_t state)
{
    this->state = state;
}

void Processor::timesync(uint64_t timebase, uint64_t time)
{
    if(time_stabilized >= 1 )time_stabilized --;
    else{
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
}

void Processor::point_cloud_lidar(const int16_t * keyframe_pcl, uint32_t size_keyframe,
                                  const int16_t * lidar_pcl, uint32_t size_pcl)
{
        // === Забираем синхронизированный кадр камеры ===
        rs2::frame depth;
        float depth_scale;

        {
            std::unique_lock<std::mutex> lk(depth_mutex);

//            if (!depth_ready) {
//                std::cout << "[SYNC] Дроп: лидар без кадра камеры" << std::endl;
//                return;
//            }
            depth = pending_depth_frame_2;
            depth_scale = pending_depth_scale_2;
            depth_ready = false;

            lk.unlock();
        }

        const uint16_t *depth_data = (const uint16_t*)depth.get_data();

        // === Расчёт сдвига карты (как в elevation_mapping_cupy) ===
        const float RESOLUTION = 0.1f;

        float position_x = static_cast<float>(odom.x_pos);
        float position_y = static_cast<float>(odom.y_pos);
        float position_z = static_cast<float>(odom.z_pos);

        float delta_x = position_x - center_x;
        float delta_y = position_y - center_y;
        float delta_z = position_z - center_z;

        int16_t delta_cells_x = static_cast<int16_t>(round(delta_x / RESOLUTION));
        int16_t delta_cells_y = static_cast<int16_t>(round(delta_y / RESOLUTION));
        int16_t delta_cells_z = 0;

        center_x += delta_cells_x * RESOLUTION;
        center_y += delta_cells_y * RESOLUTION;
        center_z += delta_z;

        if (delta_cells_x != 0 || delta_cells_y != 0) {
            shift_map_xy_cuda(dev_elevation_map, dev_normal_map, dev_shift_temp, -delta_cells_x, -delta_cells_y);
        }
        if (fabsf(delta_z) > 1e-6f) {
            shift_map_z_cuda(dev_elevation_map, -delta_z);
        }

        // === Компенсация точек: переводим их в систему координат центра карты ===
        float shift_x = position_x - center_x;
        float shift_y = position_y - center_y;
        float shift_z = position_z - center_z;

        std::vector<int16_t> points_shifted(size_pcl * 3);
        for (uint32_t i = 0; i < size_pcl; i++) {
            points_shifted[i*3+0] = lidar_pcl[i*3+0] + static_cast<int16_t>(shift_x * 1000.0f);
            points_shifted[i*3+1] = lidar_pcl[i*3+1] + static_cast<int16_t>(shift_y * 1000.0f);
            points_shifted[i*3+2] = lidar_pcl[i*3+2] + static_cast<int16_t>(shift_z * 1000.0f);
        }

        // === position_noise из одометрии ===
        const float alpha_pos = 0.2f;   // position_lowpass_alpha из YAML
        const float alpha_ori = 0.2f;   // orientation_lowpass_alpha из YAML

        float qw = (float)odom.w, qx = (float)odom.x, qy = (float)odom.y, qz = (float)odom.z;

        if (!lpf_init) {
            lpf_x = position_x; lpf_y = position_y; lpf_z = position_z;
            lpf_qw = qw; lpf_qx = qx; lpf_qy = qy; lpf_qz = qz;
            lpf_init = true;
        }

        // position_noise — отклонение от lowpass
        float pnx = position_x - lpf_x;
        float pny = position_y - lpf_y;
        float pnz = position_z - lpf_z;
        float position_noise = sqrtf(pnx*pnx + pny*pny + pnz*pnz);

        // orientation_noise — угол между текущим и lowpass-кватернионом
        float dot = qw*lpf_qw + qx*lpf_qx + qy*lpf_qy + qz*lpf_qz;
        dot = fminf(fmaxf(fabsf(dot), 0.0f), 1.0f);
        float orientation_noise = 2.0f * acosf(dot);   // radians

        // Обновляем lowpass (после измерения ошибки)
        lpf_x = alpha_pos * position_x + (1.0f - alpha_pos) * lpf_x;
        lpf_y = alpha_pos * position_y + (1.0f - alpha_pos) * lpf_y;
        lpf_z = alpha_pos * position_z + (1.0f - alpha_pos) * lpf_z;
        lpf_qw = alpha_ori * qw + (1.0f - alpha_ori) * lpf_qw;
        lpf_qx = alpha_ori * qx + (1.0f - alpha_ori) * lpf_qx;
        lpf_qy = alpha_ori * qy + (1.0f - alpha_ori) * lpf_qy;
        lpf_qz = alpha_ori * qz + (1.0f - alpha_ori) * lpf_qz;
        float qnorm = sqrtf(lpf_qw*lpf_qw + lpf_qx*lpf_qx + lpf_qy*lpf_qy + lpf_qz*lpf_qz);
        if (qnorm > 1e-6f) { lpf_qw /= qnorm; lpf_qx /= qnorm; lpf_qy /= qnorm; lpf_qz /= qnorm; }

        // === Локальная позиция сенсора (робот в центре карты) ===
        float sensor_pos[3] = { shift_x, shift_y, shift_z };

        float h_min = 0.0f, h_max = 1.5f, rover_height_m = 0.0f;

        process_costmap_cuda(points_shifted.data(), size_pcl, sensor_pos,
                             dev_elevation_map, dev_new_map, dev_normal_map,
                             dev_error, dev_error_cnt,
                             dev_costmap, costmap, costmap_rgba,
                             heightmap_packed,
                             position_noise, orientation_noise,
                             &h_min, &h_max, &rover_height_m);

        // === Визуализация costmap ===
        QImage img((const uchar*)costmap_rgba, 400, 400, QImage::Format_RGBA8888);
        if (state == 2) DisplayingCostMap(img, QQuaternion(odom.w, -odom.y, odom.z, -odom.x));

        // === Визуализация heightmap ===
        if (state == 0) emit DisplayingHeightMap(heightmap_packed, h_min, h_max, rover_height_m, QQuaternion(odom.w, -odom.y, odom.z, -odom.x));

        // === Планировщик ===
        state_lattice_->updateCostmapFromData(costmap);
}

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
    float robot_z = Map_center_Y;
    float marker_x = Aruco_point[0] + Map_center_X;
    float marker_z = Aruco_point[2] + Map_center_Y;

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

void Processor::Finding_start_end_of_the_map_2d(float target_pos_x, float target_pos_y, float target_yaw)
{
    unsigned int start_x = Map_center_X;
    unsigned int start_y = Map_center_Y;
    unsigned int goal_x = static_cast<unsigned int>(target_pos_x);
    unsigned int goal_y = static_cast<unsigned int>(target_pos_y);

    // Проверяем границы цели
    if (goal_x >= Map_X || goal_y >= Map_Y) {
        qWarning() << "Цель выходит за границы карты";
        return;
    }

    // Функция проверки ячейки
    auto is_valid_cell = [this](unsigned int x, unsigned int y) -> bool {
        if (x >= Map_X || y >= Map_Y) return false;
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
    start.orientation = (90.0f + 0.0f) * DEGTORAD; // Текущий yaw робота в радианах

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

 double timestamp_image = -1.0;

 // start and stop just to get necessary profile
 rs2::pipeline_profile pipe_profile = pipe.start(config);
 pipe.stop();

 auto imu_callback = [&](const rs2::frame& frame)
 {
   std::unique_lock<std::mutex> lock(imu_mutex);

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
         if (has_lidar_time && target_remainder == -1 && hardware_frame_counter >= 64) {
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

                 pending_depth_frame_ = fs.get_depth_frame();
                 pending_depth_scale_ = fs.get_depth_frame().get_units();
                 pending_color_frame_ = fs.get_color_frame();
                 image_ready_ = 1;
                 timestamp_image = fs.get_timestamp()*1e-3;


             }
         }

        //  В лог добавляем метку [TARGET], если это наш искомый кадр со сдвигом ~ 50 мс
//        std::cout << "[CAM_THREAD] кадр № " << hardware_frame_counter
//                          << (is_target_frame ? " [TARGET]" : "         ")
//                          << " | Время CAM: " << ((current_frame_time - first_scan_lidar) / 1000000.0) << " мс"
//                          << " | LIDAR_SYNC: " << ((last_lidar_sync_time - first_scan_lidar) / 1000000.0) << " мс"
//                          << " | Фаза: " << independent_phase_ms << " мс"
//                          << std::endl;
     }

   lock.unlock();
 };

   pipe_profile = pipe.start(config, imu_callback);

   cam_stream = pipe_profile.get_stream(RS2_STREAM_DEPTH);

   intrinsics_depth = cam_stream.as<rs2::video_stream_profile>().get_intrinsics();

   init_cuda_map(&dev_elevation_map, &dev_new_map, &dev_normal_map,
                 &dev_error, &dev_error_cnt,
                 &dev_costmap,
                 &dev_shift_temp);

   // Инициализация дисперсии (1000.0 — как в elevation_mapping_cupy)
   init_elevation_defaults(dev_elevation_map, 1000.0f);

   costmap = new uint8_t [160000];   // CPU-копия costmap

   heightmap_packed = new float[400 * 200];   // CPU-копия упакованного heightmap

   costmap_rgba = new uint32_t[160000];   // 640 КБ

   cv::Mat Color;

   rs2::frame pending_depth_frame;
   float pending_depth_scale = 0.0f;

 while(process_rgbd)
 {
   if(image_ready_) {


   if (image_ready_ == 1) {
    std::unique_lock<std::mutex> lk(imu_mutex);

    image_ready_ = 2;   // помечаем, что aruco забрал Color
    Color = cv::Mat(cv::Size(width_img, height_img), CV_8UC3, (void*)pending_color_frame_.get_data(), cv::Mat::AUTO_STEP);
    pending_depth_frame = pending_depth_frame_;
    pending_depth_scale = pending_depth_scale_;

    lk.unlock();
   }

    if (image_ready_ == 2) {
    std::unique_lock<std::mutex> lk(depth_mutex);
    pending_depth_frame_2 = pending_depth_frame;
    pending_depth_scale_2 = pending_depth_scale;
    depth_ready = true;
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


     image_ready_ = 0;

   }
   else usleep(50);

 }

 pipe.stop();
}

void Processor::odometry_lidar(odometry data)
{
  odom = data;
}


