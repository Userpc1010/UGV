#include "MDetectorNode.h"
#include <QDebug>

MDetectorNode::MDetectorNode(QObject* parent) : QObject(parent) {
    filter_ = std::make_shared<DynObjFilter>();
    MDetectorConfig cfg;
    filter_->init(cfg);
}

MDetectorNode::~MDetectorNode() {
    if (stead_guard_) delete[] stead_data_;
    if (dyn_guard_)   delete[] dyn_data_;
}

void MDetectorNode::onPointCloud(const float* body_pcl, uint32_t size_pcl,
                                 const float* T_arr_16, double scan_stamp)
{
    if (size_pcl == 0 || !body_pcl || !T_arr_16) return;

    // ===== DEBUG: вход в onPointCloud =====
    static int frame_cnt = 0;
    if (frame_cnt % 10 == 0) {
        std::cout << "[MDetNode] frame=" << frame_cnt
                  << " size=" << size_pcl
                  << " scan_stamp=" << scan_stamp
                  << std::endl;
    }
    frame_cnt++;
    // ===== /DEBUG =====

    // 1. Копируем T_arr_16 к себе
    std::memcpy(T_arr_saved_, T_arr_16, 16 * sizeof(float));

    // 2. R, t из T_arr_saved_
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            T(i, j) = T_arr_saved_[i*4 + j];
    Eigen::Matrix3d R = T.block<3,3>(0,0);
    Eigen::Vector3d t = T.block<3,1>(0,3);

    // 3. float* → PCL
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr body_pc(new pcl::PointCloud<pcl::PointXYZINormal>);
    body_pc->reserve(size_pcl);
    for (uint32_t i = 0; i < size_pcl; i++) {
        pcl::PointXYZINormal pt;
        pt.x = body_pcl[i*3+0]; pt.y = body_pcl[i*3+1]; pt.z = body_pcl[i*3+2];
        pt.intensity = 0; pt.curvature = 0;
        body_pc->push_back(pt);
    }

    // 4. Фильтр
    filter_->filter(body_pc, R, t, scan_stamp);

    // ===== DEBUG: после filter() =====
    if (frame_cnt % 10 == 0) {
        std::cout << "[MDetNode] after filter: dyn=" << filter_->laserCloudDynObj->size()
                  << " stead=" << filter_->laserCloudSteadObj->size()
                  << " dyn_clus=" << filter_->laserCloudDynObj_clus->size()
                  << " stead_clus=" << filter_->laserCloudSteadObj_clus->size()
                  << " cluster_coupled=" << filter_->cluster_coupled
                  << " cluster_future=" << filter_->cluster_future
                  << std::endl;
    }
    // ===== /DEBUG =====

    // 5. Статика (world) → body
    auto stead_world = (filter_->cluster_coupled || filter_->cluster_future)
        ? pcl::make_shared<pcl::PointCloud<pcl::PointXYZINormal>>(*filter_->laserCloudSteadObj_clus)
        : pcl::make_shared<pcl::PointCloud<pcl::PointXYZINormal>>(*filter_->laserCloudSteadObj);

    if (stead_guard_) { delete[] stead_data_; stead_guard_ = false; }
    stead_size_ = stead_world->size();
    stead_data_ = new float[stead_size_ * 3];
    for (size_t i = 0; i < stead_world->size(); i++) {
        const auto& p = stead_world->points[i];
        Eigen::Vector3d p_b = (Eigen::Vector3d(p.x, p.y, p.z) - t); // world → body
        stead_data_[i*3+0] = (float)p_b.x();
        stead_data_[i*3+1] = (float)p_b.y();
        stead_data_[i*3+2] = (float)p_b.z();
    }
    stead_guard_ = true;

    // ===== DEBUG: перед emit =====
        if (frame_cnt % 10 == 0) {
            std::cout << "[MDetNode] emit static: size=" << stead_size_
                      << " state_windosw=" << (int)state_windosw
                      << std::endl;
        }
        // ===== /DEBUG =====
        emit staticPointsReady(stead_data_, stead_size_, T_arr_saved_);

    // 6. Динамика
    if (state_windosw == 2) {
        auto dyn_world = (filter_->cluster_coupled || filter_->cluster_future)
            ? pcl::make_shared<pcl::PointCloud<pcl::PointXYZINormal>>(*filter_->laserCloudDynObj_clus)
            : pcl::make_shared<pcl::PointCloud<pcl::PointXYZINormal>>(*filter_->laserCloudDynObj_world);

        if (dyn_guard_) { delete[] dyn_data_; dyn_guard_ = false; }
        dyn_size_ = dyn_world->size();
        dyn_data_ = new float[dyn_size_ * 3];
        for (size_t i = 0; i < dyn_world->size(); i++) {
            const auto& p = dyn_world->points[i];

            dyn_data_[i*3+0] = p.x;
            dyn_data_[i*3+1] = p.y;
            dyn_data_[i*3+2] = p.z;
        }
        dyn_guard_ = true;
        // ===== DEBUG: перед emit dynamic =====
        if (frame_cnt % 10 == 0) {
            std::cout << "[MDetNode] emit dynamic: size=" << dyn_size_
                      << std::endl;
        }
        // ===== /DEBUG =====
        emit dynamicPointsReady(dyn_data_, dyn_size_, T_arr_saved_);
    }
}


void MDetectorNode::WindowState(uint8_t state) {
    this->state_windosw = state;
}
