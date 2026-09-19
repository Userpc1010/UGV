#include "livoxsdk.h"


class LivoxSDK;
LivoxSDK * mapping = nullptr;

LivoxSDK::LivoxSDK(const std::string config_path, QObject *parent)
{  
  time_stabilized = false;
  last = 0;
  start_l = true;
  lock_ = false;
  Last_upd = 0;

  pcl_thread = std::thread( &LivoxSDK::callbackPointCloud, this);
  imu_thread = std::thread( &LivoxSDK::callbackImu, this);

  usleep(1000);

  path = config_path;

  // REQUIRED, to init Livox SDK2
  if (!LivoxLidarSdkInit(path.c_str())) {
        printf("Livox Init Failed\n");
        LivoxLidarSdkUninit();
        while(1)sleep(1000);
  }

  // REQUIRED, to get point cloud data via 'PointCloudCallback'
  SetLivoxLidarPointCloudCallBack(PointCloudCallback, nullptr);

  // OPTIONAL, to get imu data via 'ImuDataCallback'
  // some lidar types DO NOT contain an imu component
  SetLivoxLidarImuDataCallback(ImuDataCallback, nullptr);

  mapping = this;
}

LivoxSDK::~LivoxSDK()
{
  mapping = nullptr;
  LivoxLidarSdkUninit();
  printf("Livox End!\n");

  if(pcl_thread.joinable()) pcl_thread.detach();
  if(pcl_thread.joinable()) imu_thread.detach();
}

void LivoxSDK::camsync(double time)
{
    cam_offset = time;
    time_stabilized = true;
}

void LivoxSDK::callbackPointCloud2(const CustomMsg msg)
{
 std::lock_guard<std::mutex> lock(mutex);
 pc_queue_.push_back(msg);
 cv_pc_.notify_one();

}

void LivoxSDK::callbackImu2(const ImuConst imu)
{
  std::lock_guard<std::mutex> lock(mutex);
  imu_queue_.push_back(imu);
  cv_imu_.notify_one();
}


void LivoxSDK::callbackPointCloud()
{
  nice(19);
  while(1){

  std::unique_lock<std::mutex>lock(mutex);

  cv_pc_.wait(lock, [this]{return !pc_queue_.empty();});
  CustomMsg msg = pc_queue_.front();
  pc_queue_.pop_front();
  lock.unlock();

  Node.callbackPointCloud(msg);
  }
}

void LivoxSDK::callbackImu()
{
 nice(19);
 while(1){

 std::unique_lock<std::mutex>lock(mutex);

 cv_imu_.wait(lock, [this]{return !imu_queue_.empty();});
 ImuConst imu_raw = imu_queue_.front();
 imu_queue_.pop_front();
 lock.unlock();

 Node.callbackImu(imu_raw);

 }
}

void LivoxSDK::PointCloudCallback(uint32_t handle, const uint8_t dev_type, LivoxLidarEthernetPacket *data, void *client_data)
{
    if (data == nullptr) {
       return;
     }

    if (!time_stabilized) {
      // Ждём скачка, ничего не накапливаем
      return;
    }

     if(lock_){lock_ = false; customMsg.points.clear(); customMsg.point_num = 0;}

    // printf("point cloud handle: %u, data_num: %d, data_type: %d, length: %d, frame_counter: %d\n",handle, data->dot_num, data->data_type, data->length, data->frame_cnt);

     //kLivoxLidarCartesianCoordinateHighData Это облако точек в декартовой системе координат в миллиметрах и int32_t
     //kLivoxLidarCartesianCoordinateLowData  Это облако точек в декартовой системе координат в сантиметрах и int16_t
     //kLivoxLidarSphericalCoordinateData Это в полярной системе координат в углах пси и тета uint16_t и высоте (глубине) uint32_t

     if (data->data_type == kLivoxLidarCartesianCoordinateHighData) {

       uint64_t timestamp = GetEthPacketTimestamp(handle, data->time_type, data->timestamp, sizeof(data->timestamp));

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

         uint64_t scan_start_time = customMsg.points.at(0).offset_time;

         emit mapping->timesync(scan_start_time, (scan_start_time + 100000000) + (int64_t)TARGET_PHASE_NS);

         customMsg.header.stamp = Time(timestamp / 1000000000.0);

         //customMsg.header = fromSec_pcl(static_cast<double>(timestamp) / 1000000000.0);  //To Sec ROS

         //customMsg.stamp = static_cast<double>(timestamp) / 1000000000.0;

         lock_ = true;
         customMsg.lidar_id = handle;
         customMsg.header.msg_seq++;
         customMsg.timebase = scan_start_time;

         emit mapping->callbackPointCloud2(customMsg);

         last += 100000000;
       }

   }
}


void LivoxSDK::ImuDataCallback(uint32_t handle, const uint8_t dev_type, LivoxLidarEthernetPacket *data, void *client_data)
{
    if (data == nullptr) {
        return;
      }

     //printf("Imu data callback handle:%u, data_num:%u, data_type:%u, length:%u, frame_counter:%u.\n", handle, data->dot_num, data->data_type, data->length, data->frame_cnt);

    if (data->data_type == kLivoxLidarImuData) {

        uint64_t timestamp = GetEthPacketTimestamp(handle, data->time_type, data->timestamp, sizeof(data->timestamp));

        if (!time_stabilized) {
               // Ждём скачка, ничего не накапливаем
               return;
        }

        if(((timestamp - Last_upd) / 1000.0) < 2000 ) { Last_upd += 5000000; timestamp += 2000 - ((timestamp - Last_upd) / 1000.0); }
        else Last_upd += 5000000;

        LivoxLidarImuRawPoint *p_point_data = (LivoxLidarImuRawPoint *)data->data;

        ImuConst imu;

        imu.header.stamp = Time(timestamp / 1000000000.0);

        imu.header.msg_seq++;

        //imu.header = fromSec_imu(static_cast<double>(timestamp) / 1000000000.0);  //To Sec ROS

        //imu.stamp = static_cast<double>(timestamp) / 1000000000.0;

        imu.angular_velocity[0] = p_point_data->gyro_x;
        imu.angular_velocity[1] = p_point_data->gyro_y;
        imu.angular_velocity[2] = p_point_data->gyro_z;

        imu.linear_acceleration[0] = p_point_data->acc_x;
        imu.linear_acceleration[1] = p_point_data->acc_y;
        imu.linear_acceleration[2] = p_point_data->acc_z;

        emit mapping->callbackImu2(imu);

        //printf("Imu data callback acc_x:%f, acc_y:%f, acc_z:%f, time:%u, dot_num:%u.\n", handle, p_point_data->acc_x, p_point_data->acc_y, p_point_data->acc_z, timestamp, data->dot_num);
    }
}

uint64_t LivoxSDK::GetEthPacketTimestamp(uint32_t handle, uint8_t timestamp_type, uint8_t *time_stamp, uint8_t size)
{
    LdsStamp time;
    memcpy(time.stamp_bytes, time_stamp, size);

    if (time_stabilized)
    {
        // Дельта = время камеры - текущее время лидара
        double delta_ns = cam_offset - time.stamp;

        // Корректируем время лидара на эту дельту + фазовый сдвиг
        return time.stamp + (int64_t)delta_ns - (int64_t)TARGET_PHASE_NS;
    }

    return time.stamp;
}
