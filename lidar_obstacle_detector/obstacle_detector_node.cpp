#include "obstacle_detector_node.h"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>
#include <QDebug>

#include "config.hpp"
#include "obstacle_detector.hpp"

namespace {

using PointT = pcl::PointXYZI;

// Точка внутри бокса в BEV с запасом margin [м] (z не проверяем —
// бокс уже отфильтрован по высоте самим детектором)
inline bool insideBEV(const lidar_obstacle_detector::Box& b,
                      const Eigen::Vector3f& p, float margin) {
    const Eigen::Vector2f d = p.head<2>() - b.center.head<2>();
    const float c = std::cos(b.yaw), s = std::sin(b.yaw);
    return std::abs(c * d.x() + s * d.y()) <= 0.5f * b.size.x() + margin &&
           std::abs(-s * d.x() + c * d.y()) <= 0.5f * b.size.y() + margin;
}

}  // namespace

struct ObstacleDetectorNode::Impl {
    lidar_obstacle_detector::PipelineConfig cfg;
    lidar_obstacle_detector::ObstacleDetector<PointT> detector;

    // Буферы под сигналы (как в старом MDetectorNode): живут до следующего
    // onPointCloud, guard-флаги — чтобы не течь между вызовами
    float* stead_data_ = nullptr;
    bool   stead_guard_ = false;
    uint32_t stead_size_ = 0;

    float* dyn_data_ = nullptr;
    bool   dyn_guard_ = false;
    uint32_t dyn_size_ = 0;

    float T_arr_saved_[16] = {0};

    int frame_count = 0;

    Impl() : cfg(lidar_obstacle_detector::PipelineConfig::preset("fast")),
             detector(makeConfig(cfg)) {}

    static lidar_obstacle_detector::PipelineConfig makeConfig(
        lidar_obstacle_detector::PipelineConfig c) {
        c.occupancy.enabled = false;   // для замены M-detector не нужно
        c.preprocess.remove_ego = false; // MID-360 на ровере установлен на башне и свой корпус не видит
        c.ground.sensor_height = 0.70f;  // MID-360 на ровере
        c.preprocess.voxel_size = 0.1f;
        c.tracker.accel_noise = 0.5f;
        c.tracker.meas_noise = 0.5f;
        c.tracker.min_hits = 3;
        c.tracker.max_age = 1.0f;
        c.cluster.min_size = 20;
        c.cluster.object_max_length = 8.0f;
        c.cluster.object_max_width = 4.0f;
        return c;
    }
};

ObstacleDetectorNode::ObstacleDetectorNode(QObject* parent)
    : QObject(parent), impl_(std::make_unique<Impl>()) {}

ObstacleDetectorNode::~ObstacleDetectorNode() {
    if (impl_->stead_guard_ && impl_->stead_data_) delete[] impl_->stead_data_;
    if (impl_->dyn_guard_   && impl_->dyn_data_)   delete[] impl_->dyn_data_;
}

void ObstacleDetectorNode::onPointCloud(const float* body_pcl, uint32_t size_pcl,
                                        const float* T_arr_16, double scan_stamp) {
    if (!body_pcl || size_pcl == 0 || !T_arr_16) {
        qDebug() << "[ODN] skip: empty input";
        return;
    }

    const bool log_verbose = (impl_->frame_count % 10 == 0);
    impl_->frame_count++;

    // 1. Сохраняем T (как старый MDetectorNode) — пробросим в сигнал
    std::memcpy(impl_->T_arr_saved_, T_arr_16, 16 * sizeof(float));

    Eigen::Matrix4f T;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            T(i, j) = impl_->T_arr_saved_[i * 4 + j];
    const Eigen::Matrix3f R = T.block<3, 3>(0, 0);
    const float yaw_deg = std::atan2(R(1, 0), R(0, 0)) * 180.f / float(M_PI);

    if (log_verbose) {
    qDebug().nospace() << "[ODN] in: " << size_pcl << " pts"
                       << " stamp=" << scan_stamp
                       << " T_t=(" << T(0,3) << "," << T(1,3) << "," << T(2,3) << ")"
                       << " yaw=" << yaw_deg << "deg";
    }

    // 2. float* -> PCL (body-фрейм, вход детектора)
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);
    cloud->reserve(size_pcl);
    for (uint32_t i = 0; i < size_pcl; ++i) {
        PointT p;
        p.x = body_pcl[i * 3 + 0];
        p.y = body_pcl[i * 3 + 1];
        p.z = body_pcl[i * 3 + 2];
        p.intensity = 0.f;
        cloud->push_back(p);
    }

    // 3. Пайплайн. tracking_T_sensor = T (world_T_body): трекинг в мировой
    //    системе, скорости мировые
    Eigen::Isometry3f tracking_T_sensor = Eigen::Isometry3f::Identity();
    tracking_T_sensor.matrix() = T;
    const auto res = impl_->detector.process(cloud, scan_stamp, tracking_T_sensor);

    const bool ground_empty = (!res.ground || res.ground->empty());

    if (log_verbose) {
    qDebug().nospace() << "[ODN] out:"
                       << " boxes=" << res.boxes.size()
                       << " structures=" << res.structure_boxes.size()
                       << " ground=" << (res.ground ? res.ground->size() : 0)
                       << " obstacles=" << (res.obstacles ? res.obstacles->size() : 0)
                       << " t=" << res.timings.total << "ms"
                       << (ground_empty ? "  << GROUND EMPTY >>" : "");
    }

    // 4. Отбираем движущиеся боксы
    std::vector<const lidar_obstacle_detector::Box*> moving;
    float moving_area = 0.f;
    moving.reserve(res.boxes.size());
    for (const auto& b : res.boxes) {
        if (b.tracked && b.has_velocity &&
            b.velocity.norm() > kMinDynamicSpeed) {
            moving.push_back(&b);
            moving_area += b.size.x() * b.size.y();
        }
    }

    if (log_verbose) {
    // Лог каждого движущегося бокса
    for (const auto* b : moving) {
        qDebug().nospace() << "[ODN]   MOVING id=" << b->id
                           << " |v|=" << b->velocity.norm()
                           << " size=(" << b->size.x() << "," << b->size.y() << "," << b->size.z() << ")"
                           << " label=" << lidar_obstacle_detector::objectClassName(b->label)
                           << " tracked=" << int(b->tracked)
                           << " age=" << b->age;
    }
    }

    int n_tracked = 0, n_unconfirmed = 0;
    for (const auto& b : res.boxes) { if (b.tracked) ++n_tracked; else ++n_unconfirmed; }

    if (log_verbose) {
    qDebug().nospace() << "[ODN] tracks=" << n_tracked
                       << " (unconfirmed=" << n_unconfirmed << ")"
                       << " moving(>" << kMinDynamicSpeed << "m/s)=" << moving.size()
                       << " moving_area=" << moving_area << "m2";
    }

    // 5. Разметка точек: inside -> dynamic (world), outside -> static
    //    (R * p_body, как у MDetectorNode)
    constexpr float kMargin = 0.2f;

    if (impl_->stead_guard_ && impl_->stead_data_) {
        delete[] impl_->stead_data_; impl_->stead_data_ = nullptr;
        impl_->stead_guard_ = false;
    }
    if (impl_->dyn_guard_ && impl_->dyn_data_) {
        delete[] impl_->dyn_data_; impl_->dyn_data_ = nullptr;
        impl_->dyn_guard_ = false;
    }

    impl_->stead_data_ = new float[3 * size_pcl];
    impl_->dyn_data_   = new float[3 * size_pcl];
    impl_->stead_size_ = 0;
    impl_->dyn_size_   = 0;

    for (uint32_t i = 0; i < size_pcl; ++i) {
        const Eigen::Vector3f p_b(body_pcl[i * 3 + 0],
                                  body_pcl[i * 3 + 1],
                                  body_pcl[i * 3 + 2]);
        bool is_dyn = false;
        for (const auto* b : moving) {
            if (insideBEV(*b, p_b, kMargin)) { is_dyn = true; break; }
        }

        if (is_dyn) {
            // dynamic -> world
            const Eigen::Vector3f p_w = R * p_b + T.block<3, 1>(0, 3);
            const uint32_t k = impl_->dyn_size_++;
            impl_->dyn_data_[k * 3 + 0] = p_w.x();
            impl_->dyn_data_[k * 3 + 1] = p_w.y();
            impl_->dyn_data_[k * 3 + 2] = p_w.z();
        } else {
            // static -> R * p_body (без смещения t, совместимо с Processor)
            const Eigen::Vector3f p_s = R * p_b;
            const uint32_t k = impl_->stead_size_++;
            impl_->stead_data_[k * 3 + 0] = p_s.x();
            impl_->stead_data_[k * 3 + 1] = p_s.y();
            impl_->stead_data_[k * 3 + 2] = p_s.z();
        }
    }

    if (log_verbose) {
    qDebug().nospace() << "[ODN] split:"
                       << " static=" << impl_->stead_size_
                       << " dynamic=" << impl_->dyn_size_
                       << " dyn_frac=" << (impl_->stead_size_ + impl_->dyn_size_ > 0
                                             ? 100.f * float(impl_->dyn_size_) /
                                               float(impl_->stead_size_ + impl_->dyn_size_)
                                             : 0.f)
                       << "%"
                       << " window=" << int(state_windosw);
    }

    impl_->stead_guard_ = impl_->stead_size_ > 0;
    impl_->dyn_guard_   = impl_->dyn_size_   > 0;

    // 6. Эмиссия (те же сигнатуры, что у MDetectorNode)
    emit staticPointsReady(impl_->stead_data_, impl_->stead_size_,
                           impl_->T_arr_saved_);
    if (state_windosw == 2) {
        emit dynamicPointsReady(impl_->dyn_data_, impl_->dyn_size_,
                                impl_->T_arr_saved_);
    }
}

void ObstacleDetectorNode::WindowState(uint8_t state) {
    state_windosw = state;
    qDebug() << "[ObstacleDetectorNode] window state ->" << int(state);
}
