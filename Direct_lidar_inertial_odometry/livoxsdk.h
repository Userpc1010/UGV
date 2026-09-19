#ifndef LIVOXSDK_H
#define LIVOXSDK_H

#include <QObject>
#include <livox_lidar_def.h>
#include <livox_lidar_api.h>
#include <iostream>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <thread>
#include "odom.h"

class LivoxSDK: public QObject
{
    Q_OBJECT

public:

    LivoxSDK(const std::string config_path, QObject *parent = nullptr);
    ~LivoxSDK();

    dlio::OdomNode Node;

signals:

    void timesync(uint64_t timebase, uint64_t time);

public slots:

     void camsync(double time);

private:

    void callbackPointCloud2(const CustomMsg msg);
    void callbackImu2(const ImuConst imu);

    void callbackPointCloud();
    void callbackImu();

    static void PointCloudCallback(uint32_t handle, const uint8_t dev_type, LivoxLidarEthernetPacket *data, void *client_data);
    static void ImuDataCallback(uint32_t handle, const uint8_t dev_type, LivoxLidarEthernetPacket *data, void *client_data);

    static uint64_t GetEthPacketTimestamp(uint32_t handle, uint8_t timestamp_type, uint8_t *time_stamp, uint8_t size);

    // Threads
    std::thread imu_thread;
    std::thread pcl_thread;

   std::deque<CustomMsg> pc_queue_;
   std::deque<ImuConst> imu_queue_;

   std::mutex mutex;

   std::condition_variable cv_pc_;
   std::condition_variable cv_imu_;

   static inline uint64_t Last_upd = 0;

    //Livox SDK
    static inline const uint8_t kLineNumberDefault = 1;
    static inline const uint8_t kLineNumberMid360 = 4;
    static inline const uint8_t kLineNumberHAP = 6;

    static inline uint64_t last = 0;

    static inline double cam_offset = 0;

    static inline CustomMsg customMsg;

    static inline bool lock_ = false;

    static inline bool start_l = true;

    static inline bool time_stabilized = false;

    // Целевой сдвиг фазы: 11.5 мс = 11 500 000 наносекунд
    static inline const uint64_t TARGET_PHASE_NS = 11500000ULL;

    std::string path;

};

#endif // LIVOXSDK_H
