#include "fast-lio2.hpp"
#include <boost/math/special_functions/round.hpp>
#include <QDebug>
#include <cmath>
#include "timer.hpp"

class FASTLIO_2;
FASTLIO_2 * fast_lio_odject = nullptr;

uint64_t last = 0.0;

CustomMsg customMsg;

bool lock = false;

uint32_t sec = 0, nsec = 0;

bool start_l = true;

std::mutex mtx_buffer;
std::condition_variable sig_buffer;

double fromSec(double t)
{
  sec = (uint32_t)floor(t);
  nsec = (uint32_t)boost::math::round((t-sec) * 1e9);
  // avoid rounding errors
  sec += (nsec / 1000000000ul);
  nsec %= 1000000000ul;
  return (double)sec + 1e-9*(double)nsec;
}

uint64_t GetEthPacketTimestamp(uint8_t timestamp_type, uint8_t* time_stamp, uint8_t size)
{
    LdsStamp time;
    memcpy(time.stamp_bytes, time_stamp, size);

    if (timestamp_type == kTimestampTypeGptpOrPtp || timestamp_type == kTimestampTypeGps) return time.stamp;

    return std::chrono::high_resolution_clock::now().time_since_epoch().count();
}

void PointCloudCallback2(uint32_t handle, const uint8_t dev_type, LivoxLidarEthernetPacket *data, void *client_data) // 10Hz
{
    if (data == nullptr) {
       return;
     }

     if(lock){lock = false; customMsg.points.clear(); customMsg.point_num = 0;}

    // printf("point cloud handle: %u, data_num: %d, data_type: %d, length: %d, frame_counter: %d\n",handle, data->dot_num, data->data_type, data->length, data->frame_cnt);

     //kLivoxLidarCartesianCoordinateHighData Это облако точек в декартовой системе координат в миллиметрах и int32_t
     //kLivoxLidarCartesianCoordinateLowData  Это облако точек в декартовой системе координат в сантиметрах и int16_t
     //kLivoxLidarSphericalCoordinateData Это в полярной системе координат в углах пси и тета uint16_t и высоте (глубине) uint32_t

     if (data->data_type == kLivoxLidarCartesianCoordinateHighData) {

       mtx_buffer.lock();

       uint64_t timestamp = GetEthPacketTimestamp(data->time_type, data->timestamp, sizeof(data->timestamp));

       LivoxLidarCartesianHighRawPoint *p_point_data = (LivoxLidarCartesianHighRawPoint *)data->data;

        if(start_l){start_l = false; last = timestamp;}

       for (uint32_t i = 0; i < data->dot_num; i++) {

         CustomPoint point;

         point.x = p_point_data[i].x / 1000.0;
         point.y = p_point_data[i].y / 1000.0;
         point.z = p_point_data[i].z / 1000.0;

         point.reflectivity = p_point_data[i].reflectivity;
         point.line = i % kLineNumberMid360;
         point.tag = p_point_data[i].tag;
         point.offset_time = timestamp + i * (data->time_interval * 100 / data->dot_num /*ns*/);

         customMsg.points.push_back(point);
       }

       customMsg.point_num += data->dot_num;

    if (timestamp - last >= 100000000)
    {
      customMsg.header = fromSec((double)timestamp / 1000000000.0);  //To Sec ROS
      lock = true;
      customMsg.lidar_id = handle;
      customMsg.timebase = customMsg.points.at(0).offset_time;

     emit fast_lio_odject->livox_pcl_cbk(customMsg);

     last = timestamp;
    }

    mtx_buffer.unlock();
    sig_buffer.notify_all();

   }
}

void ImuDataCallback2(uint32_t handle, const uint8_t dev_type, LivoxLidarEthernetPacket *data, void *client_data)
{
    if (data == nullptr) {
        return;
      }

     //printf("Imu data callback handle:%u, data_num:%u, data_type:%u, length:%u, frame_counter:%u.\n", handle, data->dot_num, data->data_type, data->length, data->frame_cnt);

    if (data->data_type == kLivoxLidarImuData) {

         mtx_buffer.lock();

        uint64_t timestamp = GetEthPacketTimestamp(data->time_type, data->timestamp, sizeof(data->timestamp));

         ImuConstPtr imuPtr;

        LivoxLidarImuRawPoint *p_point_data = (LivoxLidarImuRawPoint *)data->data;

        imuPtr.header = fromSec((double)timestamp / 1000000000.0);  //To Sec ROS

        imuPtr.angular_velocity[0] = p_point_data->gyro_x;
        imuPtr.angular_velocity[1] = p_point_data->gyro_y;
        imuPtr.angular_velocity[2] = p_point_data->gyro_z;

        imuPtr.linear_acceleration[0] = p_point_data->acc_x;
        imuPtr.linear_acceleration[1] = p_point_data->acc_y;
        imuPtr.linear_acceleration[2] = p_point_data->acc_z;

        //printf("Imu data callback acc_x:%f, acc_y:%f, acc_z:%f, time:%u, dot_num:%u.\n", handle, p_point_data->acc_x, p_point_data->acc_y, p_point_data->acc_z, timestamp, data->dot_num);

        emit fast_lio_odject->imu_cbk(imuPtr);

        mtx_buffer.unlock();
        sig_buffer.notify_all();

    }
}

void WorkModeCallback2(livox_status status, uint32_t handle, LivoxLidarAsyncControlResponse *response, void *client_data)
{
    if (response == nullptr) {
       return;
     }

}

void RebootCallback2(livox_status status, uint32_t handle, LivoxLidarRebootResponse *response, void *client_data)
{
    if (response == nullptr) {
        return;
      }
}

void SetIpInfoCallback2(livox_status status, uint32_t handle, LivoxLidarAsyncControlResponse *response, void *client_data)
{
  if (response == nullptr) {
   return;
  }
  // printf("LivoxLidarIpInfoCallback, status:%u, handle:%u, ret_code:%u, error_key:%u",status, handle, response->ret_code, response->error_key);

  if (response->ret_code == 0 && response->error_key == 0) {
     //LivoxLidarRequestReboot(handle, RebootCallback, nullptr);
  }
}

void QueryInternalInfoCallback2(livox_status status, uint32_t handle, LivoxLidarDiagInternalInfoResponse *response, void *client_data)
{
    if (status != kLivoxLidarStatusSuccess) {
        //printf("Query lidar internal info failed.\n");
        //QueryLivoxLidarInternalInfo(handle, QueryInternalInfoCallback, nullptr);
        return;
      }

      if (response == nullptr) {
        return;
      }

      uint8_t host_point_ipaddr[4] {0};
      uint16_t host_point_port = 0;
      uint16_t lidar_point_port = 0;

      uint8_t host_imu_ipaddr[4] {0};
      uint16_t host_imu_data_port = 0;
      uint16_t lidar_imu_data_port = 0;

      uint16_t off = 0;
      for (uint8_t i = 0; i < response->param_num; ++i) {
        LivoxLidarKeyValueParam* kv = (LivoxLidarKeyValueParam*)&response->data[off];
        if (kv->key == kKeyLidarPointDataHostIpCfg) {
          memcpy(host_point_ipaddr, &(kv->value[0]), sizeof(uint8_t) * 4);
          memcpy(&(host_point_port), &(kv->value[4]), sizeof(uint16_t));
          memcpy(&(lidar_point_port), &(kv->value[6]), sizeof(uint16_t));
        } else if (kv->key == kKeyLidarImuHostIpCfg) {
          memcpy(host_imu_ipaddr, &(kv->value[0]), sizeof(uint8_t) * 4);
          memcpy(&(host_imu_data_port), &(kv->value[4]), sizeof(uint16_t));
          memcpy(&(lidar_imu_data_port), &(kv->value[6]), sizeof(uint16_t));
        }
        off += sizeof(uint16_t) * 2;
        off += kv->length;
      }

      //printf("Host point cloud ip addr:%u.%u.%u.%u, host point cloud port:%u, lidar point cloud port:%u.\n",host_point_ipaddr[0], host_point_ipaddr[1], host_point_ipaddr[2], host_point_ipaddr[3], host_point_port, lidar_point_port);

      //printf("Host imu ip addr:%u.%u.%u.%u, host imu port:%u, lidar imu port:%u.\n",host_imu_ipaddr[0], host_imu_ipaddr[1], host_imu_ipaddr[2], host_imu_ipaddr[3], host_imu_data_port, lidar_imu_data_port);
}

void LidarInfoChangeCallback2(const uint32_t handle, const LivoxLidarInfo *info, void *client_data)
{
    if (info == nullptr) {
       //printf("lidar info change callback failed, the info is nullptr.\n");
       return;
     }
     //printf("LidarInfoChangeCallback Lidar handle: %u SN: %s\n", handle, info->sn);

     // set the work mode to kLivoxLidarNormal, namely start the lidar
     //SetLivoxLidarWorkMode(handle, kLivoxLidarNormal, WorkModeCallback, nullptr);

     //QueryLivoxLidarInternalInfo(handle, QueryInternalInfoCallback, nullptr);

     // LivoxLidarIpInfo lidar_ip_info;
     // strcpy(lidar_ip_info.ip_addr, "192.168.1.10");
     // strcpy(lidar_ip_info.net_mask, "255.255.255.0");
     // strcpy(lidar_ip_info.gw_addr, "192.168.1.1");
     // SetLivoxLidarLidarIp(handle, &lidar_ip_info, SetIpInfoCallback, nullptr);
}

void LivoxLidarPushMsgCallback2(const uint32_t handle, const uint8_t dev_type, const char *info, void *client_data)
{
   struct in_addr tmp_addr;
   tmp_addr.s_addr = handle;
   //std::cout << "handle: " << handle << ", ip: " << inet_ntoa(tmp_addr) << ", push msg info: " << std::endl;
   //std::cout << info << std::endl;
   return;
}

FASTLIO_2::FASTLIO_2(const std::string config_path, QObject *parent)
{
  path = config_path; fast_lio_odject = this;

  // REQUIRED, to init Livox SDK2
  if (!LivoxLidarSdkInit(path.c_str())) {
      printf("Livox Init Failed\n");
      LivoxLidarSdkUninit();
      while(1)sleep(1000);
 }

  // REQUIRED, to get point cloud data via 'PointCloudCallback'
  SetLivoxLidarPointCloudCallBack(PointCloudCallback2, nullptr);

  // OPTIONAL, to get imu data via 'ImuDataCallback'
  // some lidar types DO NOT contain an imu component
  SetLivoxLidarImuDataCallback(ImuDataCallback2, nullptr);

  SetLivoxLidarInfoCallback(LivoxLidarPushMsgCallback2, nullptr);

  // REQUIRED, to get a handle to targeted lidar and set its work mode to NORMAL
  SetLivoxLidarInfoChangeCallback(LidarInfoChangeCallback2, nullptr);
}

FASTLIO_2::~FASTLIO_2()
{
   fast_lio_odject = nullptr;
   LivoxLidarSdkUninit();
   printf("Livox End!\n");
}

void FASTLIO_2::SetLidarsExtParam(ExtParameter lidar_param)
{
    if (is_set_extrinsic_params_) {
        return;
      }
      extrinsic_.trans[0] = lidar_param.x;
      extrinsic_.trans[1] = lidar_param.y;
      extrinsic_.trans[2] = lidar_param.z;

      double cos_roll = cos(static_cast<double>(lidar_param.roll * M_PI / 180.0));
      double cos_pitch = cos(static_cast<double>(lidar_param.pitch * M_PI / 180.0));
      double cos_yaw = cos(static_cast<double>(lidar_param.yaw * M_PI / 180.0));
      double sin_roll = sin(static_cast<double>(lidar_param.roll * M_PI / 180.0));
      double sin_pitch = sin(static_cast<double>(lidar_param.pitch * M_PI / 180.0));
      double sin_yaw = sin(static_cast<double>(lidar_param.yaw * M_PI / 180.0));

      extrinsic_.rotation[0][0] = cos_pitch * cos_yaw;
      extrinsic_.rotation[0][1] = sin_roll * sin_pitch * cos_yaw - cos_roll * sin_yaw;
      extrinsic_.rotation[0][2] = cos_roll * sin_pitch * cos_yaw + sin_roll * sin_yaw;

      extrinsic_.rotation[1][0] = cos_pitch * sin_yaw;
      extrinsic_.rotation[1][1] = sin_roll * sin_pitch * sin_yaw + cos_roll * cos_yaw;
      extrinsic_.rotation[1][2] = cos_roll * sin_pitch * sin_yaw - sin_roll * cos_yaw;

      extrinsic_.rotation[2][0] = -sin_pitch;
      extrinsic_.rotation[2][1] = sin_roll * cos_pitch;
      extrinsic_.rotation[2][2] = cos_roll * cos_pitch;

      is_set_extrinsic_params_ = true;

//      if (extrinsic_enable) {
//           point.x = p_point_data[i].x / 1000.0;
//           point.y = p_point_data[i].y / 1000.0;
//           point.z = p_point_data[i].z / 1000.0;
//         } else {
//           point.x = (p_point_data[i].x * extrinsic_.rotation[0][0] + p_point_data[i].y * extrinsic_.rotation[0][1] + p_point_data[i].z * extrinsic_.rotation[0][2] + extrinsic_.trans[0]) / 1000.0;
//           point.y = (p_point_data[i].x * extrinsic_.rotation[1][0] + p_point_data[i].y * extrinsic_.rotation[1][1] + p_point_data[i].z * extrinsic_.rotation[1][2] + extrinsic_.trans[1]) / 1000.0;
//           point.z = (p_point_data[i].x * extrinsic_.rotation[2][0] + p_point_data[i].y * extrinsic_.rotation[2][1] + p_point_data[i].z * extrinsic_.rotation[2][2] + extrinsic_.trans[2]) / 1000.0;
      //         }
}
