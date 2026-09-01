#ifndef LIVOXDATA_H
#define LIVOXDATA_H

#include <stdint.h>
#include <vector>

typedef struct Point
{
  uint64_t offset_time = 0;      // offset time relative to the base time
  float x = 0.0f;               // X axis, unit:m
  float y = 0.0f;               // Y axis, unit:m
  float z = 0.0f;               // Z axis, unit:m
  uint8_t reflectivity = 0;      // reflectivity, 0~255
  uint8_t tag = 0;               // livox tag
  uint8_t line = 0;              // laser number in lidar

} CustomPoint;

typedef struct Msg
{
  double header = 0.0; //Time to Sec!! .stamp.toSec()
  uint32_t point_num = 0;                 // Total number of pointclouds
  uint8_t  lidar_id = 0;                  // Lidar device id number
  uint64_t timebase = 0;                     // The time of first point
  std::vector<CustomPoint> points;    // Pointcloud data
} CustomMsg;

// the preintegrated Lidar states at the time of IMU measurements in a frame
typedef struct Pose6D
{
    double offset_time = 0.0; // the offset time of IMU measurement w.r.t the first lidar point
    double acc [3] = {0.0};      // the preintegrated total acceleration (global frame) at the Lidar origin
    double gyr [3] = {0.0};      // the unbiased angular velocity (body frame) at the Lidar origin
    double vel [3] = {0.0};      // the preintegrated velocity (global frame) at the Lidar origin
    double pos [3] = {0.0};      // the preintegrated position (global frame) at the Lidar origin
    double rot [9] = {0.0};      // the preintegrated rotation (global frame) at the Lidar origin
}Pose6D;

typedef struct ImuConst
{
    ImuConst(): header(),angular_velocity(), linear_acceleration()
    {
    }
    ~ImuConst(){}

  double header = 0.0; //Time to Sec!! .stamp.toSec()

  double linear_acceleration [3] = {0.0};
  double angular_velocity [3] = {0.0};

  void reset()
  {
   header = 0.0;

   linear_acceleration[0] = 0.0;
   linear_acceleration[1] = 0.0;
   linear_acceleration[2] = 0.0;

   angular_velocity[0] = 0.0;
   angular_velocity[1] = 0.0;
   angular_velocity[2] = 0.0;
  }

} ImuConstPtr;


typedef struct odometry_lidar
{
  double timestamp = 0.0;

  double x_pos = 0.0;
  double y_pos = 0.0;
  double z_pos = 0.0;

  double x_vel = 0.0;
  double y_vel = 0.0;
  double z_vel = 0.0;

  double c_pose_covariance [36] = {0.0};

  double w=1.0,x=0.0,y=0.0,z=0.0;

  odometry_lidar(double w_ = 1.0,double x_ = 0.0, double y_ = 0.0,double z_ = 0.0)
  {
    w=w_;
    x=x_;
    y=y_;
    z=z_;
  }
}odometry;

/** 8bytes stamp to uint64_t stamp */
typedef union {
  struct {
    uint32_t low;
    uint32_t high;
  } stamp_word;

  uint8_t stamp_bytes[8];
  int64_t stamp;
} LdsStamp;

// SDK related
/** Timestamp sync mode define. */
typedef enum {
  kTimestampTypeNoSync = 0, /**< No sync signal mode. */
  kTimestampTypeGptpOrPtp = 1,    /**< gPTP or PTP sync mode */
  kTimestampTypeGps = 2   /**< GPS sync mode. */
} TimestampType;

#endif // LIVOXDATA_H
