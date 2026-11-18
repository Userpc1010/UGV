#ifndef MDETECTORNODE_H
#define MDETECTORNODE_H

#include <QObject>
#include <memory>
#include <deque>
#include <cstring>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include "m-detector/DynObjFilter.h"

class MDetectorNode : public QObject {
    Q_OBJECT
public:
    explicit MDetectorNode(QObject* parent = nullptr);
    ~MDetectorNode();

public slots:
    void onPointCloud(const float* body_pcl, uint32_t size_pcl,
                      const float* T_arr_16, double scan_stamp);
    void WindowState(uint8_t state);

signals:
    void staticPointsReady(const float* pcl_data, uint32_t size, const float* T_arr_16);
    void dynamicPointsReady(const float* pcl_data, uint32_t size, const float* T_arr_16);

private:
    std::shared_ptr<DynObjFilter> filter_;
    std::deque<std::vector<float>> stead_accu_;
    float T_arr_saved_[16];
    float* stead_data_ = nullptr;
    bool   stead_guard_ = false;
    uint32_t stead_size_ = 0;
    float* dyn_data_ = nullptr;
    bool   dyn_guard_ = false;
    uint32_t dyn_size_ = 0;
    uint8_t state_windosw = 0;
};

#endif
