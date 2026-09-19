#ifndef RGBD_CAMERA_H
#define RGBD_CAMERA_H

#include <GLFW/glfw3.h>
#include <QThread>
#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include <librealsense2-gl/rs_processing_gl.hpp> // Include GPU-Processing API
#include <librealsense2/rsutil.h>
#include "oglwidget.h"
#include "LivoxData.hpp"
#include "data.h"
#include "CostmapColorizer.h"
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

#define Map_length 16000000
#define Map_X 400
#define Map_Y 100
#define Map_Z 400
#define Map_center_X 199
#define Map_center_Y 49
#define Map_center_Z 199
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

CostmapColorizer * colorizer;

uint32_t world_counter;

 uint8_t  *dev_voxels;        // GPU-указатель на воксельную карту (16 МБ)
 uint8_t  *dev_costmap;       // GPU-указатель на 2D costmap (1.6 МБ)
 uint32_t *serialization_point;  // CPU-копия списка индексов для визуализации
 uint8_t  *costmap;           // CPU-копия costmap (остаётся)

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

 const int zero = 0;

 odometry odom;

 int32_t total_offset_x = 0;
 int32_t total_offset_y = 0;
 int32_t total_offset_z = 0;

 int32_t last_sent_offset_x = 0;
 int32_t last_sent_offset_y = 0;
 int32_t last_sent_offset_z = 0;

 double accumulated_dx;
 double accumulated_dy;
 double accumulated_dz;

 double filter_integrate_x = 0.0;
 double filter_integrate_y = 0.0;
 double filter_integrate_z = 0.0;

 double filter_wegiht_x = 0.9;
 double filter_wegiht_y = 0.9;
 double filter_wegiht_z = 0.5;

 double last_pos_x;
 double last_pos_y;
 double last_pos_z;

 float pitch = 0.0f, roll= 0.0f, yaw = 0.0f;

 // Last timestamp we visualized at
 double last_visualization_timestamp;

 bool process_rgbd;

 bool first_start = true;   //First Start True!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

 double cam_offset_ms = 0.0;

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

  GLfloat * vertices_data;
  bool ptr_guard_vertices_data = false;
  GLfloat * color_data;
  bool ptr_guard_color_data = false;

  QVector3D m_accumulatedOffset;  // Накопленное смещение

  QVector3D m_lastMapOffset;      // Последнее отправленное смещение

  // === Буфер синхронизированного кадра камеры (RGB-D) ===
  std::mutex imu_mutex;
  std::mutex color_mutex;
  rs2::frame pending_depth_frame_;
  rs2::frame pending_color_frame_;
  float pending_depth_scale_ = 0.0f;
  std::atomic<uint8_t> image_ready_{0};   // 0 = нет, 1 = готов (записан в pending_*), 2 = забран для aruco

private:

 void IMU_Get_Euler_Angle(Quat buff);

 void Raycast (const uint16_t start[3], const uint16_t end[3], const uint16_t min[3], const uint16_t max[3], std::vector<XYZ>* output);

 int16_t signum(int16_t x);

 float mod(float value, float modulus);

 float intbound(float s, float ds);

 COLOUR GetColour(float v,float vmin,float vmax);

 COLOUR JetColor(float v,float vmin,float vmax);

 uint32_t get_value (const uint16_t x, const uint16_t y, const uint16_t z);

 double expRunningAverage(double &newVal, double &filVal, const double &wegiht);

 void Finding_start_end_of_the_map (float target_pos_x, float target_pos_y, float target_pos_z);

 void Finding_start_end_of_the_map_2d (float target_pos_x, float target_pos_y, float target_yaw);

 rs2_vector interpolateMeasure(const double target_time, const rs2_vector current_data, const double current_time, const rs2_vector prev_data, const double prev_time);

protected:

    void run() override;

 signals:

 void DisplayingCostMap(const QImage &map, QQuaternion rotation);

 void Invisible_Aruco (bool invisible);

 void DrawAruco (QVector3D pos);

 void ArucoTrackerOut(std::vector<std::vector<cv::Point2f>> corners);

 void DisplayingCubes(GLfloat* vertices_buffer, GLfloat* color_buffer, unsigned long long counter, QQuaternion rotation);

 void DrawCube(QVector3D pos);

 void updateMapOffset(QVector3D offset);

 void grab_imu (rs2_vector v_gyro_data, double v_gyro_timestamp,rs2_vector v_accel_data, double v_accel_timestamp);

 void grab_stereo_camera(cv::Mat im, cv::Mat imRight, double timestamp, uint16_t width_img, uint16_t height_img);

 void input_track (double timestamp);

 void camsync(double time);

// void rotation (QQuaternion rotation);

public slots:

   void point_cloud_lidar (const uint16_t * keyframe_pcl, uint32_t size_keyframe, const int16_t * lidar_pcl, uint32_t size_pcl);

   void ArucoTrackerIn (std::vector<std::vector<cv::Point2f>> corners);

   void odometry_lidar (odometry data);

   void goalPositionSet(QVector3D position, QQuaternion rotation);

   void WindowState (uint8_t state);

   void timesync(uint64_t timebase, uint64_t time);

};

#endif
