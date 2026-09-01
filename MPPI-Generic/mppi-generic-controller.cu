#include <mppi/dynamics/dynamics.h>
#include <mppi/cost_functions/cost.h>
#include <mppi/controllers/controller.h>
#include <mppi/sampling_distributions/sampling_distribution.h>
#include <mppi/feedback_controllers/feedback.h>
#include <mppi/core/mppi_common.h>
#include <mppi/utils/angle_utils.h>
#include <mppi/utils/math_utils.h>
#include <mppi/utils/cuda_math_utils.h>

#include <mppi/controllers/MPPI/mppi_controller.h>
#include <mppi/feedback_controllers/DDP/ddp.h>
#include <mppi/sampling_distributions/gaussian/gaussian.h>
#include <mppi/core/base_plant.h>

#include <mppi/dynamics/AckermannDynamics.h>
#include <mppi/cost_functions/NAV2_Compatible_CostFunction.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>

#include <memory>
#include <atomic>
#include <chrono>
#include <thread>
#include <mutex>
#include <vector>
#include <cstring>
#include <cstdio>
#include <cmath>

// ============================================================================
// Конфигурация MPPI
// ============================================================================
constexpr int NUM_TIMESTEPS = 300;   // Горизонт 5 секунд
constexpr int NUM_ROLLOUTS = 1024;   // Оптимально для 3050 Ti
constexpr int DYN_BLOCK_X = 64;
constexpr int COST_BLOCK_X = 32;     // <= NUM_TIMESTEPS, степень двойки

using DYN_T = AckermannDynamics;
using COST_T = Nav2MinimalCost;
using FB_T = DDPFeedback<DYN_T, NUM_TIMESTEPS>;
using SAMPLING_T = mppi::sampling_distributions::GaussianDistribution<typename DYN_T::DYN_PARAMS_T>;
using CONTROLLER_T = VanillaMPPIController<DYN_T, COST_T, FB_T, NUM_TIMESTEPS, NUM_ROLLOUTS, SAMPLING_T>;

// ============================================================================
// Структуры данных
// ============================================================================
#pragma pack(push, 1)
struct MPPI_Input {
    float state_x;
    float state_y;
    float state_psi;
    float state_v;
    float state_steer;
};

struct MPPI_Output {
    float control_v;
    float control_w;
    float cost;
    uint32_t iterations;
    float free_energy;
};
#pragma pack(pop)

// ============================================================================
// CUDA Error Checking
// ============================================================================
namespace {
    inline void checkCudaError(cudaError_t err, const char* file, int line) {
        if (err != cudaSuccess) {
            printf("[CUDA ERROR] %s:%d - %s\n", file, line, cudaGetErrorString(err));
            throw std::runtime_error(cudaGetErrorString(err));
        }
    }
}
#define CUDA_CHECK(err) checkCudaError(err, __FILE__, __LINE__)

// ============================================================================
// Ядро для SDF
// ============================================================================
__global__ void computePathSDFKernel(
    float* sdf_map,
    const float* path_x,
    const float* path_y,
    int path_size,
    int width, int height,
    float resolution,
    float origin_x, float origin_y)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= width || iy >= height) return;

    float wx = origin_x + (ix + 0.5f) * resolution;
    float wy = origin_y + (iy + 0.5f) * resolution;

    float min_dist_sq = 1e10f;
    for (int p = 0; p < path_size; p++) {
        float dx = wx - path_x[p];
        float dy = wy - path_y[p];
        float dist_sq = dx * dx + dy * dy;
        if (dist_sq < min_dist_sq) min_dist_sq = dist_sq;
    }
    sdf_map[iy * width + ix] = sqrtf(min_dist_sq);
}

// ============================================================================
// MPPI Core
// ============================================================================
class MPPICore {
public:
    static MPPICore& instance() {
        static MPPICore instance;
        return instance;
    }

    MPPICore() = default;
    ~MPPICore() { shutdown(); }

    bool initialize(float dt, float lambda, float desired_speed,
                    int costmap_width, int costmap_height,
                    float costmap_resolution, float costmap_origin_x, float costmap_origin_y)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (initialized_) return true;

        try {
            CUDA_CHECK(cudaStreamCreate(&stream_));

            dynamics_ = std::make_shared<DYN_T>(stream_);
            dynamics_->GPUSetup();

            cost_ = std::make_shared<COST_T>(stream_);
            cost_->GPUSetup();

            auto cost_params = cost_->getParams();
            cost_params.goal_weight = 10.0f;
            cost_params.goal_angle_weight = 10.0f;
            cost_params.obstacle_weight = 10.0f;
            cost_->setParams(cost_params);

            fb_controller_ = std::make_shared<FB_T>(dynamics_.get(), dt);
            fb_controller_->bindToStream(stream_);
            fb_controller_->GPUSetup();

            typename SAMPLING_T::SAMPLING_PARAMS_T sampler_params;
            sampler_params.num_rollouts = NUM_ROLLOUTS;
            sampler_params.num_timesteps = NUM_TIMESTEPS;
            sampler_params.num_distributions = 1;
            sampler_params.std_dev[0] = 1.0f;
            sampler_params.std_dev[1] = 0.2f;

            sampler_ = std::make_shared<SAMPLING_T>(sampler_params, stream_);
            sampler_->GPUSetup();

            typename CONTROLLER_T::TEMPLATED_PARAMS controller_params;
            controller_params.dt_ = dt;
            controller_params.lambda_ = lambda;
            controller_params.alpha_ = 0.0f;
            controller_params.num_timesteps_ = NUM_TIMESTEPS;
            controller_params.num_iters_ = 1;
            controller_params.dynamics_rollout_dim_ = dim3(DYN_BLOCK_X, DYN_T::STATE_DIM, 1);
            controller_params.cost_rollout_dim_ = dim3(COST_BLOCK_X, 1, 1);
            controller_params.init_control_traj_ = Eigen::Matrix<float, DYN_T::CONTROL_DIM, NUM_TIMESTEPS>::Zero();

            controller_ = std::make_shared<CONTROLLER_T>(
                dynamics_.get(), cost_.get(), fb_controller_.get(),
                sampler_.get(), controller_params, stream_);

            controller_->disableFeedbackController();
            controller_->setPercentageSampledControlTrajectories(0.125f);
            controller_->setTopNSampledControlTrajectoriesHelper(0);

            dt_ = dt;
            lambda_ = lambda;
            desired_speed_ = desired_speed;
            costmap_width_ = costmap_width;
            costmap_height_ = costmap_height;
            costmap_resolution_ = costmap_resolution;
            costmap_origin_x_ = costmap_origin_x;
            costmap_origin_y_ = costmap_origin_y;

            initialized_ = true;
            printf("[MPPICore] Initialized successfully\n");
            printf("[MPPICore] Initialized: res=%.4f, origin=(%.2f, %.2f)\n",
                   costmap_resolution_, costmap_origin_x_, costmap_origin_y_);
            return true;
        } catch (const std::exception& e) {
            printf("[MPPICore] Init failed: %s\n", e.what());
            return false;
        }
    }

    void shutdown() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!initialized_) return;
        if (current_sdf_) { cudaFree(current_sdf_); current_sdf_ = nullptr; }
        controller_.reset();
        sampler_.reset();
        fb_controller_.reset();
        cost_.reset();
        dynamics_.reset();
        if (stream_) { cudaStreamDestroy(stream_); stream_ = nullptr; }
        initialized_ = false;
    }

    void setCostmap(const uint8_t* costmap_data) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!initialized_ || !cost_) return;
        cost_->initCostmap(costmap_data, costmap_width_, costmap_height_,
                          costmap_resolution_, costmap_origin_x_, costmap_origin_y_);
    }

    void updateGoal(float goal_x, float goal_y, float goal_yaw) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!initialized_ || !cost_) return;
        cost_->updateGoal(goal_x, goal_y, goal_yaw);
    }

    MPPI_Output computeControl(const MPPI_Input& input) {
        std::lock_guard<std::mutex> lock(mutex_);
        MPPI_Output output = {0};
        if (!initialized_ || !controller_) return output;

        DYN_T::state_array state = DYN_T::state_array::Zero();
        using StateIdx = typename DYN_T::params_t::StateIndex;
        state[static_cast<int>(StateIdx::POS_X)] = input.state_x;
        state[static_cast<int>(StateIdx::POS_Y)] = input.state_y;
        state[static_cast<int>(StateIdx::PSI)] = input.state_psi;
        state[static_cast<int>(StateIdx::V)] = input.state_v;
        state[static_cast<int>(StateIdx::STEER_ANGLE)] = input.state_steer;

        auto start = std::chrono::steady_clock::now();
        controller_->computeControl(state, 1);
        controller_->calculateSampledStateTrajectories();
        auto end = std::chrono::steady_clock::now();
        auto dur_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        auto ctrl = controller_->getControlSeq();
        using ControlIdx = typename DYN_T::params_t::ControlIndex;
        output.control_v = ctrl(static_cast<int>(ControlIdx::V_CMD), 0);
        output.control_w = ctrl(static_cast<int>(ControlIdx::W_CMD), 0);
        output.cost = controller_->getBaselineCost();
        output.iterations = controller_->getNumIters();
        output.free_energy = controller_->getFreeEnergyStatistics().real_sys.freeEnergyMean;

//        static int fc = 0; static float tt = 0;
//        fc++; tt += dur_us.count() / 1000.0f;
//        if (fc % 100 == 0) {
//            printf("[MPPICore] %d: time=%.2f ms, cost=%.2f, energy=%.4f\n",
//                   fc, tt / fc, output.cost, output.free_energy);
//        }
        return output;
    }

    void setCostmapGPU(uint8_t* gpu_costmap) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!initialized_ || !cost_) return;

        cost_->initCostmapFromGPU(gpu_costmap, costmap_width_, costmap_height_,
                                  costmap_resolution_, costmap_origin_x_, costmap_origin_y_);
    }

    void setCostmapCPU(const uint8_t* cpu_costmap) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!initialized_ || !cost_) return;

        cost_->initCostmap(cpu_costmap, costmap_width_, costmap_height_,
                           costmap_resolution_, costmap_origin_x_, costmap_origin_y_);  // <-- данные на CPU
    }

    auto getOptimalControlSequence() {
        if (!controller_) return decltype(controller_->getControlSeq())();
        return controller_->getControlSeq();
    }

    int getNumTimesteps() const { return NUM_TIMESTEPS; }

    auto getController() const { return controller_.get(); }

    float getDt() const { return dt_; }

    bool isInitialized() const { return initialized_; }

private:
    MPPICore(const MPPICore&) = delete;
    MPPICore& operator=(const MPPICore&) = delete;

    mutable std::mutex mutex_;
    bool initialized_ = false;
    cudaStream_t stream_ = nullptr;

    std::shared_ptr<DYN_T> dynamics_;
    std::shared_ptr<COST_T> cost_;
    std::shared_ptr<FB_T> fb_controller_;
    std::shared_ptr<SAMPLING_T> sampler_;
    std::shared_ptr<CONTROLLER_T> controller_;

    float dt_ = 0.1f, lambda_ = 1.0f, desired_speed_ = 10.0f;
    int costmap_width_ = 800, costmap_height_ = 800;
    float costmap_resolution_ = 0.489f, costmap_origin_x_ = 0.0f, costmap_origin_y_ = 0.0f;
    float* current_sdf_ = nullptr;
};

// ============================================================================
// Глобальный экземпляр
// ============================================================================
static std::unique_ptr<MPPICore> g_mppi_core;

// ============================================================================
// C-интерфейс
// ============================================================================
extern "C" {

bool mppi_initialize(float dt, float lambda, float desired_speed,
                     int w, int h, float res, float ox, float oy) {
    g_mppi_core = std::make_unique<MPPICore>();
    return g_mppi_core->initialize(dt, lambda, desired_speed, w, h, res, ox, oy);
}

void mppi_shutdown() {
    if (g_mppi_core) { g_mppi_core->shutdown(); g_mppi_core.reset(); }
}

bool mppi_is_initialized() {
    return g_mppi_core && g_mppi_core->isInitialized();
}

void mppi_set_costmap(const uint8_t* data) {
    if (g_mppi_core) g_mppi_core->setCostmap(data);
}

//void mppi_set_costmap_gpu(uint8_t* gpu_costmap) {
//    if (g_mppi_core) {
        // Приведение типов: GPU-указатель на costmap
//        g_mppi_core->setCostmap(reinterpret_cast<const uint8_t*>(gpu_costmap));
//    }
//}

void mppi_set_costmap_gpu(uint8_t* gpu_costmap) {
    if (!g_mppi_core) return;
    g_mppi_core->setCostmapGPU(gpu_costmap);
}

void mppi_update_goal(float gx, float gy, float gyaw) {
    if (g_mppi_core) g_mppi_core->updateGoal(gx, gy, gyaw);
}

void mppi_update_full_path(const float* path_x, const float* path_y,
                           const float* path_yaws, uint32_t path_size) {
    // В минимальной версии Nav2MinimalCost нет полной поддержки пути
    // Можно либо игнорировать, либо использовать для обновления цели
    if (g_mppi_core && path_size > 0) {
        // Пока просто используем последнюю точку как цель
        g_mppi_core->updateGoal(path_x[path_size - 1],
                                path_y[path_size - 1],
                                path_yaws[path_size - 1]);
    }
}

void Run_MPPI_Core(uint8_t* input, uint8_t* output) {
    if (!g_mppi_core) return;
    MPPI_Input* in = reinterpret_cast<MPPI_Input*>(input);
    MPPI_Output out = g_mppi_core->computeControl(*in);
    memcpy(output, &out, sizeof(MPPI_Output));
}

void mppi_slide_control_sequence(int steps) {
    // В текущей реализации не используется
    (void)steps;
}

int mppi_get_control_sequence(float* buffer, int max_timesteps) {
    if (!g_mppi_core) return 0;

    auto ctrl = g_mppi_core->getOptimalControlSequence();
    int timesteps = std::min(max_timesteps, g_mppi_core->getNumTimesteps());

    for (int t = 0; t < timesteps; t++) {
        buffer[t * 2 + 0] = ctrl(0, t);
        buffer[t * 2 + 1] = ctrl(1, t);
    }
    return timesteps;
}

int mppi_get_state_trajectory(float* buffer, int max_timesteps) {
    if (!g_mppi_core) return 0;

    auto state_traj = g_mppi_core->getController()->getTargetStateSeq();
    int timesteps = std::min(max_timesteps, g_mppi_core->getNumTimesteps());

    if (buffer) {
        for (int t = 0; t < timesteps; t++) {
            buffer[t*3 + 0] = state_traj(0, t);  // X
            buffer[t*3 + 1] = state_traj(1, t);  // Y
            buffer[t*3 + 2] = state_traj(2, t);  // PSI
        }
    }
    return timesteps;
}

int mppi_get_sampled_state_trajectories(float* buffer, int max_trajectories) {
    if (!g_mppi_core) return 0;

    auto trajectories = g_mppi_core->getController()->getSampledOutputTrajectories();
    int num_traj = trajectories.size();

    if (buffer == nullptr) {
        return num_traj;
    }

    int timesteps = g_mppi_core->getNumTimesteps();
    int count = std::min(num_traj, max_trajectories);

    // Баг MPPI-Generic: последний timestep (t=timesteps-1) содержит нули
    // Копируем только timesteps-1 точек
    int valid_timesteps = timesteps - 1;

    for (int i = 0; i < count; i++) {
        for (int t = 0; t < valid_timesteps; t++) {
            int idx = (i * valid_timesteps + t) * 3;
            buffer[idx + 0] = trajectories[i](0, t);
            buffer[idx + 1] = 0.51f;
            buffer[idx + 2] = trajectories[i](1, t);
        }
    }
    return count;
}

int mppi_get_num_timesteps() {
    return g_mppi_core ? g_mppi_core->getNumTimesteps() : 0;
}

float mppi_get_dt() {
    return g_mppi_core ? g_mppi_core->getDt() : 0.0f;
}

} // extern "C"
