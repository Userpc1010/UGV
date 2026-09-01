#ifndef FASTLIO2_H
#define FASTLIO2_H

#include "livox_lidar_def.h"
#include "livox_lidar_api.h"

#include <unistd.h>
#include <arpa/inet.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <thread>
#include <chrono>
#include <iostream>
#include <mutex>
#include <condition_variable>

#include <QObject>
#include "LivoxData.hpp"

/** Device Line Number **/
const uint8_t kLineNumberDefault = 1;
const uint8_t kLineNumberMid360 = 4;
const uint8_t kLineNumberHAP = 6;

/*****************************/
/* About Extrinsic Parameter */
typedef struct {
  float roll;  /**< Roll angle, unit: degree. */
  float pitch; /**< Pitch angle, unit: degree. */
  float yaw;   /**< Yaw angle, unit: degree. */
  int32_t x;   /**< X translation, unit: mm. */
  int32_t y;   /**< Y translation, unit: mm. */
  int32_t z;   /**< Z translation, unit: mm. */
} ExtParameter;

typedef float TranslationVector[3]; /**< x, y, z translation, unit: mm. */
typedef float RotationMatrix[3][3];

typedef struct {
  TranslationVector trans;
  RotationMatrix rotation;
} ExtParameterDetailed;

class FASTLIO_2 : public QObject
{
  Q_OBJECT

public:

    FASTLIO_2(const std::string config_path, QObject *parent = nullptr);

    ~FASTLIO_2();

private:

void SetLidarsExtParam(ExtParameter lidar_param);

ExtParameterDetailed extrinsic_ = {
    {0, 0, 0},
    {
      {1, 0, 0},
      {0, 1, 1},
      {0, 0, 1}
    }
};

std::string path;

std::atomic_bool is_set_extrinsic_params_;

bool extrinsic_enable;

signals:

void imu_cbk(const ImuConstPtr msg_in);
void livox_pcl_cbk(const CustomMsg msg);

};

#endif // FASTLIO2_H
