#ifndef OBSTACLE_DETECTOR_NODE_H
#define OBSTACLE_DETECTOR_NODE_H

#include <QObject>
#include <cstdint>
#include <memory>

/**
 * Qt-обёртка над lidar_obstacle_detector::ObstacleDetector
 * (замена старого MDetectorNode).
 *
 * Вход (слот onPointCloud):
 *   body_pcl  — точки в body-фрейме, float[3*N] (x,y,z)
 *   size_pcl  — количество точек
 *   T_arr_16  — world_T_body, row-major 4x4
 *   scan_stamp— время скана [s]
 *
 * Выход (сигналы):
 *   staticPointsReady  — точки вне движущихся боксов, в «безопасном» виде
 *                        (R · p_body, как выдавал M-detector для совместимости
 *                        с Processor)
 *   dynamicPointsReady — точки внутри движущихся боксов, в world-фрейме
 *
 * Движущийся объект = трек ObstacleDetector с
 *   box.tracked && box.has_velocity && |velocity| > kMinDynamicSpeed.
 *
 * ВНИМАНИЕ: указатели в сигналах живут до следующего вызова onPointCloud
 * (та же модель, что была в MDetectorNode). Для Queued-соединения между
 * потоками в MainWindow.cpp нужен
 *   qRegisterMetaType<const float*>("const float*");
 * либо Qt::DirectConnection.
 */
class ObstacleDetectorNode : public QObject {
    Q_OBJECT
public:
    // Порог скорости, выше которого трек считается движущимся [м/с]
    static constexpr float kMinDynamicSpeed = 1.0f;

    explicit ObstacleDetectorNode(QObject* parent = nullptr);
    ~ObstacleDetectorNode() override;

public slots:
    void onPointCloud(const float* body_pcl, uint32_t size_pcl,
                      const float* T_arr_16, double scan_stamp);
    void WindowState(uint8_t state);

signals:
    void staticPointsReady(const float* pcl_data, uint32_t size,
                           const float* T_arr_16);
    void dynamicPointsReady(const float* pcl_data, uint32_t size,
                            const float* T_arr_16);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    uint8_t state_windosw = 0;  // 0 — только static; 2 — static + dynamic
};

#endif  // OBSTACLE_DETECTOR_NODE_H
