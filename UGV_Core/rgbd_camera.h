#ifndef RGBD_CAMERA_H
#define RGBD_CAMERA_H

#include <GLFW/glfw3.h>
#include <QThread>
#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include <librealsense2-gl/rs_processing_gl.hpp> // Include GPU-Processing API
#include <librealsense2/rsutil.h>
#include "LivoxData.hpp"
#include "nav2_costmap_2d/raytrace_line_2d.hpp"
#include "statelattice.h"
#include <opencv2/core.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#define GL_SILENCE_DEPRECATION
#define GLFW_INCLUDE_GLU

#ifndef PI_
#define PI_  3.14159265358979323846
#define PI_FL  3.141592f
#endif

#define DEGTORAD 0.0174532925199432957f
#define RADTODEG 57.295779513082320876f

#define Quat_HALFPI 1.5707963267948966192313216916398f
#define Quat_PI 3.1415926535897932384626433832795f
#define Quat_TWOPI 6.283185307179586476925286766559f
#define Quat_TODEG(x) ((x) * 57.2957796f)

#define MAP_CELL_N 400
#define MAP_CELLS  160000  // 400 * 400
#define MAP_LAYERS 7

// Индексы слоёв в dev_elevation_map (каждый слой — 400x400 float)
#define L_HEIGHT      0   // Высота (Z в DLIO)
#define L_VARIANCE    1   // Дисперсия (для Калмана)
#define L_IS_VALID    2   // Флаг валидности (1.0 = валидно)
#define L_TRAVERS     3   // Проходимость (0..1)
#define L_TIME        4   // Время с последнего обновления
#define L_UPPER_BOUND 5   // Верхняя граница (для нависающих препятствий)
#define L_IS_UPPER    6   // Флаг: 1.0 = граница, 0.0 = земля

// Оставляем для costmap и min/max
#define Map_X 400
#define Map_Y 400
#define Map_center_X 200
#define Map_center_Y 200

#define Map_scale 10.0f

typedef struct {
    float r = 0.0, g = 0.0, b = 0.0;
} COLOUR;

class Processor:public QThread
{
    Q_OBJECT

public:

 Processor(QObject *parent = nullptr);

 ~Processor();

 StateLattice* state_lattice_;

 std::atomic<bool> lattice_planning_in_progress_{false};

 float* dev_elevation_map;    // [7][400][400] — основная 2.5D карта
 float* dev_new_map;          // [7][400][400] — буфер для atomicAdd (Калман)
 float* dev_normal_map;       // [3][400][400] — нормали поверхности
 float* dev_error;            // [1] — сумма ошибок для drift compensation
 float* dev_error_cnt;        // [1] — счётчик точек для drift compensation
 uint8_t  *dev_costmap;       // GPU-указатель на 2D costmap (1.6 МБ)
 uint8_t  *costmap;           // CPU-копия costmap (остаётся)
 float *heightmap_packed;     // CPU-копия упакованного heightmap float[400*200]
 float * dev_shift_temp;      // Временный буфер для сдига карты
 uint32_t* costmap_rgba;     // CPU-копия RGBA для QImage

private:

 const float Rotation_matrix[9] = {0.999048f, 0.018916f, -0.0393044f, 0.0f, 0.901077f, -0.433659f, 0.0436194f, 0.433246f, 0.900219f};
 const float Translation[3] = {-0.005f, 0.089f, 0.055f}; //Ваше старое значение было -0.065. Когда мы прибавили к нему +0.06 (В вашем выводе из SDK было четко сказано: Tbc = 0.005), мы получили -0.025.

 //ArucoPosition
 float Aruco_point[3] = {0};

 //ArucoTracker RGB camera
 cv::Mat camera_matrix, dist_coeffs;

 // Declare pointcloud object, for calculating pointclouds and texture mappings
 // rs2::gl::pointcloud pc;
 // used to explicitly copy frame to the GPU
 // rs2::gl::uploader upload;
 // similar to rs2::colorizer
 // rs2::gl::colorizer  colorizer;

 rs2::config config;
 // Declare RealSense pipeline, encapsulating the actual device and sensors
 rs2::pipeline pipe;
 // We want the points object to be persistent so we can display the last cloud when a frame drops
// rs2::points points;

 rs2::stream_profile cam_stream;

 rs2_intrinsics intrinsics_depth;

 // The pointcloud will use colorized depth as a texture:
 // rs2::frame depth_texture;

 // GLFWwindow* win;

 // Declare object that handles camera pose calculations

 odometry odom;

 // Центр карты в мировых координатах (аналог self.center в elevation_mapping_cupy)
 float center_x = 0.0f;
 float center_y = 0.0f;
 float center_z = 0.0f;

 float lpf_x = 0.0f, lpf_y = 0.0f, lpf_z = 0.0f;
 float lpf_qw = 1.0f, lpf_qx = 0.0f, lpf_qy = 0.0f, lpf_qz = 0.0f;
 bool lpf_init = false;

 // Last timestamp we visualized at
 double last_visualization_timestamp;

 bool process_rgbd;

 double cam_offset_ms = 0.0;

 uint8_t time_stabilized = 15;

 uint64_t last_lidar_sync_time = 0;

 uint64_t first_scan_lidar = 0;

 uint64_t first_frame_camera = 0;

 uint64_t last_scan_timebase = 0;

 uint64_t frame_counter_at_last_timesync = 0;

 bool has_lidar_time = false;

 double phase_shifts [3] = {0};

 uint64_t phase_counts [3] = {0};

 // Переменная для хранения надежного аппаратного счетчика
 unsigned long long hardware_frame_counter = 0;

 int target_remainder = -1;       // -1 означает, что фаза еще не определена

  uint8_t state = 0;

  // === Буфер синхронизированного кадра камеры (RGB-D) ===
  std::mutex imu_mutex;
  std::mutex depth_mutex;
  rs2::frame pending_color_frame_;
  rs2::frame pending_depth_frame_;
  rs2::frame pending_depth_frame_2;
  float pending_depth_scale_ = 0.0f;
  float pending_depth_scale_2 = 0.0f;
  std::atomic<uint8_t> image_ready_{0};   // 0 = нет, 1 = готов (записан в pending_*), 2 = забран для aruco
  bool depth_ready = false;

private:

 void Finding_start_end_of_the_map_2d (float target_pos_x, float target_pos_y, float target_yaw);

protected:

    void run() override;

 signals:

 void DisplayingCostMap(const QImage &map, QQuaternion rotation);

 void Invisible_Aruco (bool invisible);

 void DrawAruco (QVector3D pos);

 void ArucoTrackerOut(std::vector<std::vector<cv::Point2f>> corners);

 void DisplayingHeightMap(const float *data, float min, float max, float targetHeight, QQuaternion rotation);

 void updateMapOffset(QVector3D offset);

 void DrawCube(QVector3D pos);

 void camsync(double time);

public slots:

   void point_cloud_lidar (const int16_t * keyframe_pcl, uint32_t size_keyframe, const int16_t * lidar_pcl, uint32_t size_pcl);

   void ArucoTrackerIn (std::vector<std::vector<cv::Point2f>> corners);

   void odometry_lidar (odometry data);

   void goalPositionSet(QVector3D position, QQuaternion rotation);

   void WindowState (uint8_t state);

   void timesync(uint64_t timebase, uint64_t time);

};

#endif
