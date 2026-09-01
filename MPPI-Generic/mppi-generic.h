#ifndef MPPIGENERIC_H
#define MPPIGENERIC_H

#include <QObject>
#include <QTimer>
#include <QElapsedTimer>

#include <vector>
#include <cstdint>
#include <memory>
#include <atomic>
#include <mutex>

// ============================================================================
// Структуры данных для MPPI (соответствуют C-интерфейсу)
// ============================================================================

#pragma pack(push, 1)
struct MPPI_Input {
    float state_x;          // Позиция X [м]
    float state_y;          // Позиция Y [м]
    float state_psi;        // Курс [рад]
    float state_v;          // Скорость [м/с]
    float state_steer;      // Угол поворота колёс [рад]
    uint32_t reserved;      // Зарезервировано для выравнивания
};

struct MPPI_Output {
    float control_v;        // Команда скорости [м/с]
    float control_w;        // Команда угловой скорости [рад/с]
    float cost;             // Стоимость оптимальной траектории
    uint32_t iterations;    // Количество итераций
    float free_energy;      // Свободная энергия (метрика здоровья)
};
#pragma pack(pop)

// ============================================================================
// Структура точки пути
// ============================================================================
struct PathPoint {
    float x, y, yaw;
};

// ============================================================================
// MPPIController - высокоуровневая обёртка над GPU-библиотекой MPPI-Generic
// ============================================================================
class MPPIController : public QObject
{
    Q_OBJECT

public:
    explicit MPPIController(QObject* parent = nullptr);
    ~MPPIController();

    bool initialize(float dt, float lambda, float desired_speed,
                    int costmap_width, int costmap_height,
                    float costmap_resolution, float costmap_origin_x, float costmap_origin_y);

    // Отправка costmap из CPU-памяти (копирует на GPU и вызывает setCostmapGPU)
    void setCostmapCPU(const uint8_t* cpu_data);

    std::vector<float> getControlSequence();

    std::vector<float> getSampledTrajectories();

    std::vector<float> getStateTrajectory();

    void start(int frequency_hz = 20);
    void stop();

    bool isInitialized() const { return initialized_; }
    bool isRunning() const { return running_; }

    // Синхронное вычисление управления (для тестирования)
    MPPI_Output computeControlSync(float x, float y, float psi, float v, float steer);

    int getNumTimesteps() const;
    float getDt() const;

public slots:
    void setCurrentState(float x, float y, float psi, float v, float steer);
    void setCostmapGPU(uint8_t* gpu_costmap);
    void setGoal(float goal_x, float goal_y, float goal_yaw);
    void updatePath(const std::vector<PathPoint>& path);
    void controlLoop();
    void emergencyStop();

signals:
    void controlReady(float v_cmd, float w_cmd);
    void statisticsUpdated(float cost, float free_energy, float compute_time_ms);
    void error(const QString& message);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;

    QTimer* timer_;
    std::atomic<bool> initialized_;
    std::atomic<bool> running_;

    mutable std::mutex state_mutex_;
    float current_x_, current_y_, current_psi_, current_v_, current_steer_;
    bool state_received_;

    std::atomic<bool> emergency_stop_flag_;
};

#endif // MPPIGENERIC_H
