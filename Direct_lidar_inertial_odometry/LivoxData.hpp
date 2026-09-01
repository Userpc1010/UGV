#ifndef LIVOXDATA_H
#define LIVOXDATA_H

#pragma once

#include <stdint.h>
#include <vector>
#include <limits.h>
#include <cmath>
#include <stdexcept>

template<class T>
 class TimeBase
 {
 public:
   uint32_t sec, nsec;

   TimeBase() : sec(0), nsec(0) { }
   TimeBase(uint32_t _sec, uint32_t _nsec) : sec(_sec), nsec(_nsec)
   {
     normalizeSecNSec(sec, nsec);
   }
   explicit TimeBase(double t) { fromSec(t); }
   ~TimeBase() {}
   bool operator==(const T &rhs) const;
   inline bool operator!=(const T &rhs) const { return !(*static_cast<const T*>(this) == rhs); }
   bool operator>(const T &rhs) const;
   bool operator<(const T &rhs) const;
   bool operator>=(const T &rhs) const;
   bool operator<=(const T &rhs) const;

   double toSec()  const { return (double)sec + 1e-9*(double)nsec; };
   T& fromSec(double t) { sec = (uint32_t)floor(t); nsec = (uint32_t)round((t-sec) * 1e9);  return *static_cast<T*>(this);}

   uint64_t toNSec() const {return (uint64_t)sec*1000000000ull + (uint64_t)nsec;  }
   T& fromNSec(uint64_t t);

   void normalizeSecNSec(uint64_t& sec, uint64_t& nsec)
    {
      uint64_t nsec_part = nsec % 1000000000UL;
      uint64_t sec_part = nsec / 1000000000UL;

      if (sec_part > UINT_MAX)
        throw std::runtime_error("Time is out of dual 32-bit range");

      sec += sec_part;
      nsec = nsec_part;
    }

    void normalizeSecNSec(uint32_t& sec, uint32_t& nsec)
    {
      uint64_t sec64 = sec;
      uint64_t nsec64 = nsec;

      normalizeSecNSec(sec64, nsec64);

      sec = (uint32_t)sec64;
      nsec = (uint32_t)nsec64;
    }

   inline bool isZero() const { return sec == 0 && nsec == 0; }
   inline bool is_zero() const { return isZero(); }
 };

 class Time : public TimeBase<Time>
 {
 public:
   Time()
   : TimeBase<Time>()
   {}

   Time(uint32_t _sec, uint32_t _nsec)
   : TimeBase<Time>(_sec, _nsec)
   {}

   explicit Time(double t) { fromSec(t); }

   static Time now();
   static bool sleepUntil(const Time& end);

   static void init();
   static void shutdown();
   static void setNow(const Time& new_now);
   static bool useSystemTime();
   static bool isSimTime();
   static bool isSystemTime();

   static bool isValid();
   static bool waitForValid();
};

typedef struct
{
 uint32_t msg_seq = 0;
 Time stamp;
}Header;

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
  Header header;                           //Time to Sec!! .stamp.toSec()
  uint32_t point_num = 0;                 // Total number of pointclouds
  uint8_t  lidar_id = 0;                  // Lidar device id number
  uint64_t timebase = 0;                     // The time of first point
  std::vector<CustomPoint> points;    // Pointcloud data
} CustomMsg;

typedef struct ImuConst
{
    ImuConst(): header(),angular_velocity(), linear_acceleration()
    {
    }
    ~ImuConst(){}

  Header header; //Time to Sec!! .stamp.toSec()

  double linear_acceleration [3] = {0.0};
  double angular_velocity [3] = {0.0};

  void reset()
  {
   header.stamp.nsec = 0;
   header.stamp.sec = 0;

   header.msg_seq = 0;

   linear_acceleration[0] = 0.0;
   linear_acceleration[1] = 0.0;
   linear_acceleration[2] = 0.0;

   angular_velocity[0] = 0.0;
   angular_velocity[1] = 0.0;
   angular_velocity[2] = 0.0;
  }

} ImuConst;

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
