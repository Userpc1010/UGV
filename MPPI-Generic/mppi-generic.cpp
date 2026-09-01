#include "mppi-generic.h"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <QDebug>
#include <QElapsedTimer>

#include <cmath>
#include <cstring>
#include <algorithm>
#include <cstdio>

// ============================================================================
// Внешние C-функции из скомпилированной GPU-библиотеки
// ============================================================================
extern "C" {
    bool mppi_initialize(float dt, float lambda, float desired_speed,
                         int costmap_width, int costmap_height,
                         float costmap_resolution, float costmap_origin_x, float costmap_origin_y);
    void mppi_shutdown();
    bool mppi_is_initialized();
    void mppi_set_costmap(const uint8_t* costmap_data);
    void mppi_set_costmap_gpu(uint8_t* gpu_costmap);
    void mppi_update_goal(float goal_x, float goal_y, float goal_yaw);
    void mppi_update_full_path(const float* path_x, const float* path_y,
                               const float* path_yaws, uint32_t path_size);
    void Run_MPPI_Core(uint8_t* input, uint8_t* output);
    void mppi_slide_control_sequence(int steps);
    int mppi_get_control_sequence(float* buffer, int max_timesteps);
    int mppi_get_sampled_state_trajectories(float* buffer, int max_trajectories);
    int mppi_get_state_trajectory(float* buffer, int max_timesteps);

    int mppi_get_num_timesteps();
    float mppi_get_dt();
}

// ============================================================================
// MPPIController::Impl - ПОЛНОЕ ОПРЕДЕЛЕНИЕ (должно быть перед использованием)
// ============================================================================
class MPPIController::Impl {
public:
    bool initialize(float dt, float lambda, float desired_speed,
                    int costmap_width, int costmap_height,
                    float costmap_resolution, float costmap_origin_x, float costmap_origin_y)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        bool result = mppi_initialize(dt, lambda, desired_speed,
                                      costmap_width, costmap_height,
                                      costmap_resolution, costmap_origin_x, costmap_origin_y);
        if (result) {
            initialized_ = true;
        }
        return result;
    }

    void shutdown() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (initialized_) {
            mppi_shutdown();
            initialized_ = false;
        }
    }

    bool isInitialized() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return initialized_ && mppi_is_initialized();
    }

    void setCostmapGPU(uint8_t* gpu_costmap) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!initialized_) return;
        mppi_set_costmap_gpu(gpu_costmap);
    }

    void setCostmapCPU(const uint8_t* cpu_data) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!initialized_) return;
        mppi_set_costmap(cpu_data);  // вызов существующей C-функции
    }

    void setGoal(float goal_x, float goal_y, float goal_yaw) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!initialized_) return;
        mppi_update_goal(goal_x, goal_y, goal_yaw);
    }

    void updateFullPath(const std::vector<PathPoint>& path) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!initialized_ || path.empty()) return;

        std::vector<float> path_x, path_y, path_yaws;
        path_x.reserve(path.size());
        path_y.reserve(path.size());
        path_yaws.reserve(path.size());

        for (const auto& p : path) {
            path_x.push_back(p.x);
            path_y.push_back(p.y);
            path_yaws.push_back(p.yaw);
        }

        mppi_update_full_path(path_x.data(), path_y.data(), path_yaws.data(),
                              static_cast<uint32_t>(path_x.size()));
    }

    std::vector<float> getControlSequence() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!initialized_) return {};

        const int max_timesteps = getNumTimesteps();
        std::vector<float> buffer(max_timesteps * 2);

        int timesteps = mppi_get_control_sequence(buffer.data(), max_timesteps);
        buffer.resize(timesteps * 2);
        return buffer;
    }

    std::vector<float> getSampledTrajectories() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!initialized_) return {};

        int timesteps = getNumTimesteps();
        if (timesteps <= 0) return {};

        int num_traj = mppi_get_sampled_state_trajectories(nullptr, 0);
        if (num_traj <= 0) return {};

        // Исправлено: буфер под timesteps-1 точек
        int valid_timesteps = timesteps - 1;
        std::vector<float> buffer(num_traj * valid_timesteps * 3);
        mppi_get_sampled_state_trajectories(buffer.data(), num_traj);

        return buffer;

    }

    std::vector<float> getStateTrajectory() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!initialized_) return {};

        int timesteps = getNumTimesteps();
        if (timesteps <= 0) return {};

        std::vector<float> buffer(timesteps * 3);
        mppi_get_state_trajectory(buffer.data(), timesteps);
        return buffer;
    }

    int getNumTimesteps() const { return mppi_get_num_timesteps(); }
    float getDt() const { return mppi_get_dt(); }

    MPPI_Output computeControl(float x, float y, float psi, float v, float steer) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!initialized_) {
            return MPPI_Output{0};
        }

        MPPI_Input input;
        input.state_x = x;
        input.state_y = y;
        input.state_psi = psi;
        input.state_v = v;
        input.state_steer = steer;
        input.reserved = 0;

        MPPI_Output output;
        Run_MPPI_Core(reinterpret_cast<uint8_t*>(&input), reinterpret_cast<uint8_t*>(&output));

        if (std::isnan(output.control_v) || std::isnan(output.control_w)) {
            std::fprintf(stderr, "[MPPI Impl] NaN detected in control output, forcing zero.\n");
            output.control_v = 0.0f;
            output.control_w = 0.0f;
        }

        return output;
    }

private:
    mutable std::mutex mutex_;
    bool initialized_ = false;
};

// ============================================================================
// MPPIController реализация
// ============================================================================
MPPIController::MPPIController(QObject* parent)
    : QObject(parent)
    , impl_(std::make_unique<Impl>())
    , timer_(new QTimer(this))
    , initialized_(false)
    , running_(false)
    , state_received_(false)
    , emergency_stop_flag_(false)
    , current_x_(0), current_y_(0), current_psi_(0), current_v_(0), current_steer_(0)
{
    timer_->setSingleShot(false);
    connect(timer_, &QTimer::timeout, this, &MPPIController::controlLoop);
}

MPPIController::~MPPIController()
{
    stop();
    impl_->shutdown();
}

bool MPPIController::initialize(float dt, float lambda, float desired_speed,
                                int costmap_width, int costmap_height,
                                float costmap_resolution, float costmap_origin_x, float costmap_origin_y)
{
    if (initialized_) {
        qDebug() << "[MPPI] Already initialized";
        return true;
    }

    bool success = impl_->initialize(dt, lambda, desired_speed,
                                     costmap_width, costmap_height,
                                     costmap_resolution, costmap_origin_x, costmap_origin_y);
    if (success) {
        initialized_ = true;
        qDebug() << "[MPPI] Initialized successfully. Map size:" << costmap_width << "x" << costmap_height;
    } else {
        emit error("MPPI initialization failed");
    }
    return success;
}

void MPPIController::start(int frequency_hz)
{
    if (!initialized_) {
        emit error("MPPI not initialized");
        return;
    }

    if (running_) return;

    int interval_ms = 1000 / frequency_hz;
    timer_->start(interval_ms);
    running_ = true;
    emergency_stop_flag_ = false;
    state_received_ = false;

    qDebug() << "[MPPI] Started at" << frequency_hz << "Hz (interval:" << interval_ms << "ms)";
}

void MPPIController::stop()
{
    if (!running_) return;

    timer_->stop();
    running_ = false;
    state_received_ = false;

    qDebug() << "[MPPI] Stopped";
}

MPPI_Output MPPIController::computeControlSync(float x, float y, float psi, float v, float steer)
{
    return impl_->computeControl(x, y, psi, v, steer);
}

void MPPIController::setCurrentState(float x, float y, float psi, float v, float steer)
{
    std::lock_guard<std::mutex> lock(state_mutex_);
    current_x_ = x;
    current_y_ = y;
    current_psi_ = psi;
    current_v_ = v;
    current_steer_ = steer;
    state_received_ = true;
}

void MPPIController::setCostmapGPU(uint8_t* gpu_costmap)
{
    if (!initialized_) return;
    impl_->setCostmapGPU(gpu_costmap);
}

void MPPIController::setCostmapCPU(const uint8_t* cpu_data) {
    if (!initialized_) return;
    impl_->setCostmapCPU(cpu_data);
}

void MPPIController::setGoal(float goal_x, float goal_y, float goal_yaw)
{
    if (!initialized_) return;
    impl_->setGoal(goal_x, goal_y, goal_yaw);
}

void MPPIController::updatePath(const std::vector<PathPoint>& path)
{
    if (!initialized_) return;
    if (path.empty()) {
        qDebug() << "[MPPI] Received empty path";
        return;
    }
    impl_->updateFullPath(path);
    qDebug() << "[MPPI] Path updated:" << path.size() << "points";
}

void MPPIController::controlLoop()
{
    if (!initialized_ || !running_) return;
    if (emergency_stop_flag_) {
        emit controlReady(0.0f, 0.0f);
        return;
    }

    float x, y, psi, v, steer;
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        if (!state_received_) return;
        x = current_x_;
        y = current_y_;
        psi = current_psi_;
        v = current_v_;
        steer = current_steer_;
    }

    QElapsedTimer timer;
    timer.start();

    MPPI_Output output = impl_->computeControl(x, y, psi, v, steer);

    float compute_time_ms = timer.elapsed();

    emit controlReady(output.control_v, output.control_w);
    emit statisticsUpdated(output.cost, output.free_energy, compute_time_ms);
}

void MPPIController::emergencyStop()
{
    emergency_stop_flag_ = true;
    emit controlReady(0.0f, 0.0f);
    qDebug() << "[MPPI] Emergency stop activated!";
}

std::vector<float> MPPIController::getControlSequence() {
    return impl_->getControlSequence();
}

std::vector<float> MPPIController::getSampledTrajectories() {
    return impl_->getSampledTrajectories();
}

std::vector<float> MPPIController::getStateTrajectory() {
    return impl_->getStateTrajectory();
}

int MPPIController::getNumTimesteps() const {
    return impl_->getNumTimesteps();
}

float MPPIController::getDt() const {
    return impl_->getDt();
}
