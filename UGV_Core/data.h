#ifndef DATA_H
#define DATA_H

#include <stdint.h>

//Струтура бинарного протокола GPS NAV_PVT Автор Iforce2d

const unsigned char UBX_HEADER[]     = { 0xB5, 0x62 };
const unsigned char NAV_PVT_HEADER[] = { 0x01, 0x07 };
const unsigned char NAV_SOL_HEADER[] = { 0x01, 0x06 };

struct NAV_PVT {
  uint8_t cls;
  uint8_t id;
  uint16_t len;
  uint32_t iTOW;          /**< GPS Time of Week [ms] */
  uint16_t year;          /**< Year (UTC)*/
      uint8_t  month;         /**< Month, range 1..12 (UTC) */
      uint8_t  day;           /**< Day of month, range 1..31 (UTC) */
      uint8_t  hour;          /**< Hour of day, range 0..23 (UTC) */
      uint8_t  min;           /**< Minute of hour, range 0..59 (UTC) */
      uint8_t  sec;           /**< Seconds of minute, range 0..60 (UTC) */
      uint8_t  valid;         /**< Validity flags (see UBX_RX_NAV_PVT_VALID_...) */
      uint32_t tAcc;          /**< Time accuracy estimate (UTC) [ns] */
      int32_t  nano;          /**< Fraction of second (UTC) [-1e9...1e9 ns] */
      uint8_t  fixType;       /**< GNSSfix type: 0 = No fix, 1 = Dead Reckoning only, 2 = 2D fix, 3 = 3d-fix, 4 = GNSS + dead reckoning, 5 = time only fix */
      uint8_t  flags;         /**< Fix Status Flags (see UBX_RX_NAV_PVT_FLAGS_...) */
      uint8_t  reserved1;
      uint8_t  numSV;         /**< Number of SVs used in Nav Solution */
      int32_t  lon;           /**< Longitude [1e-7 deg] */
      int32_t  lat;           /**< Latitude [1e-7 deg] */
      int32_t  height;        /**< Height above ellipsoid [mm] */
      int32_t  hMSL;          /**< Height above mean sea level [mm] */
      uint32_t hAcc;          /**< Horizontal accuracy estimate [mm] */
      uint32_t vAcc;          /**< Vertical accuracy estimate [mm] */
      int32_t  velN;          /**< NED north velocity [mm/s]*/
      int32_t  velE;          /**< NED east velocity [mm/s]*/
      int32_t  velD;          /**< NED down velocity [mm/s]*/
      int32_t  gSpeed;        /**< Ground Speed (2-D) [mm/s] */
      int32_t  headMot;       /**< Heading of motion (2-D) [1e-5 deg] */
      uint32_t sAcc;          /**< Speed accuracy estimate [mm/s] */
      uint32_t headAcc;       /**< Heading accuracy estimate (motion and vehicle) [1e-5 deg] */
      uint16_t pDOP;          /**< Position DOP [0.01] */
      uint16_t reserved2;
      uint32_t reserved3;
      int32_t  headVeh;       /**< (ubx8+ only) Heading of vehicle (2-D) [1e-5 deg] */
      uint32_t reserved4;     /**< (ubx8+ only) */
};

//struct NAV_SOL {
//  unsigned long iTOW;          // Время недели эпохи навигации
//  long fTOW;                   // Дробная часть iTOW
//  short week;                  // Номер недели эпохи навигации
//  unsigned char gpsFix;        // Тип позиции GPS 2D-Fix, 3D-Fix, GNSS + dead reckoning
//  char flags;                  // Флаги валидности
//  long ECEF_X;                 // Координата X ECEF (m, cm)
//  long ECEF_Y;                 // Координата Y ECEF (m, cm)
//  long ECEF_Z;                 // Координата Z ECEF (m, cm)
//  unsigned long pAcc;          // Оценка Горизонтальной Точности (m, cm)
//  long ECEF_VEL_X;             // Скорость X ECEF (m, cm)
//  long ECEF_VEL_Y;             // Скорость Y ECEF (m, cm)
//  long ECEF_VEL_Z;             // Скорость Z ECEF (m, cm)
//  unsigned long sAcc;          // Оценка Точности Скорости (m, cm)
//  unsigned short pDOP;         // Погрешность определения планового положения
//  unsigned char reserved1;     // Зарезервировано
//  unsigned char numSV;         // Количество спутников, используемых в решении навигации
//  unsigned long reserved2;     // Зарезервtatusировано
//};

//union UBXMessage {
//  NAV_PVT pvt;
//  NAV_SOL sol;
//};

struct SI { // Перевод в СИ

  double lon = 0.0;
  double lat = 0.0;
  double raw_lon = 0.0;
  double raw_lat = 0.0;
  float heading = 0.0f;
  float headingAcc = 0.0f;
  float g_speed = 0.0f;

  float N_vel = 0.0f;
  float E_vel = 0.0f;
  float D_vel = 0.0f;

  float h_acc = 0.0f;

};

// Режимы работы (как на STM32)
typedef enum {
    CTRL_MODE_MANUAL = 0,
    CTRL_MODE_MPPI = 1,
    CTRL_MODE_VPC = 2
} ctrl_mode_t;

// Структура для PID-режима (как на STM32)
struct PCA9685 {
    float w_cmd;       // угловая скорость (рад/с)
    float setpoint;    // линейная скорость (м/с)
    uint8_t stop;      // остановка/тормоз
};

// Структура для MPPI-режима (как на STM32)
struct MPPI_Ctrl {
    float steer_angle;   // рад
    float throttle;      // -1..1
    float brake;         // 0..1
};

// Структура телеметрии от STM32
#pragma pack(push, 1)
struct TelePacket
{
    uint8_t header[4];      // "TELE"
    float w_yaw;            // угловая скорость (рад/с)
    float stering_angle;    // угол сервы (градусы)
    float velosity_1d_mps;  // линейная скорость (м/с)
};
#pragma pack(pop)

typedef struct XYZ
{
    uint16_t x = 0, y = 0, z = 0;

    XYZ(uint16_t x_ = 0, uint16_t y_ = 0, uint16_t z_ = 0)
    {
     x = x_; y = y_; z = z_;
    }
}XYZ;

typedef struct telemetry
{
 double Vel_X = 0.0;
 double Vel_Y = 0.0;
 double Vel_Z = 0.0;

 double X = 0.0;
 double Y = 0.0;
 double Z = 0.0;

 float pitch = 0.0f, roll= 0.0f, yaw = 0.0f;

 double target_X = 0.0, target_Y = 0.0, target_Z = 0.0;

 float course = 0.0f;
 float target_course = 0.0f;

} GUI_telem;

typedef struct Quat
{
   double w,x,y,z=0.0;

    Quat(double w_ = 0.0,double x_ = 0.0, double y_ = 0.0,double z_ = 0.0)
    {
      w=w_;
      x=x_;
      y=y_;
      z=z_;
    }
}Quat;

#endif // DATA_H
