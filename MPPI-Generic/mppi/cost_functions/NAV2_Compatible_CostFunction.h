
/*
#ifndef NAV2_COMPATIBLE_COSTFUNCTION_H
#define NAV2_COMPATIBLE_COSTFUNCTION_H

#pragma once

#ifndef NAV2_AGGREGATED_COST_CUH_
#define NAV2_AGGREGATED_COST_CUH_

#include <mppi/cost_functions/cost.h>
#include <mppi/dynamics/AckermannDynamics.h>
#include <mppi/utils/math_utils.h>
#include <mppi/utils/angle_utils.h>
#include <mppi/utils/cuda_math_utils.h>
#include <mppi/utils/texture_helpers/texture_helper.h>
#include <cuda_runtime.h>
#include <vector>
#include <Eigen/Dense>

#ifndef M_PI_2
#define M_PI_2 1.57079632679f
#endif

// ============================================================================
// Параметры функции стоимости (агрегация всех критиков Nav2)
// ============================================================================
struct Nav2AggregatedCostParams : public CostParams<2>  // 2 канала управления: V_CMD, W_CMD
{
    // ------------------------------------------------------------------------
    // Общие параметры
    // ------------------------------------------------------------------------
    float dt = 0.05f;                    // шаг времени [с]
    int power = 1;                       // степень для нелинейности стоимости
    float discount = 1.0f;               // дисконтирующий фактор (из базового класса)
    float wheelbase = 0.46f;             // Колёсная база
    float eps_v = 0.05f;                 // Мин. округление

    // ------------------------------------------------------------------------
    // ConstraintCritic (ограничения скорости и угловой скорости)
    // ------------------------------------------------------------------------
    bool enable_constraint = true;
    float max_speed = 2.0f;              // vx_max из динамики
    float min_speed = -0.5f;             // vx_min из динамики
    float max_wz = 2.5f;                 // wz_max из динамики
    float constraint_weight = 4.0f;      // вес штрафа за нарушение ограничений

    // ------------------------------------------------------------------------
    // ObstaclesCritic (препятствия на карте)
    // ------------------------------------------------------------------------
    bool enable_obstacles = true;
    bool consider_footprint = true;
    float footprint_radius = 0.5f;       // радиус робота [м]
    float collision_cost = 1000000.0f;   // стоимость столкновения
    float near_collision_threshold = 253.0f;  // порог "близкого" препятствия (costmap value)
    float critical_weight = 20.0f;       // вес критической близости
    float repulsion_weight = 1.5f;       // вес отталкивания
    float collision_margin = 0.1f;       // запас до препятствия [м]
    float obstacles_weight = 3.81f;      // общий вес препятствий

    // Параметры инфляции (для вычисления расстояния из costmap)
    float inflation_radius = 0.5f;       // радиус инфляции [м]
    float cost_scaling_factor = 10.0f;   // фактор масштабирования
    float inscribed_radius = 0.3f;       // вписанный радиус робота [м]

    // ------------------------------------------------------------------------
    // GoalCritic (достижение цели)
    // ------------------------------------------------------------------------
    bool enable_goal = true;
    float goal_threshold = 0.5f;         // расстояние до цели для активации [м]
    float goal_position_weight = 5.0f;   // вес позиционного штрафа

    // ------------------------------------------------------------------------
    // GoalAngleCritic (ориентация на цель)
    // ------------------------------------------------------------------------
    bool enable_goal_angle = true;
    bool symmetric_angle_tolerance = false;
    float goal_angle_weight = 3.0f;      // вес углового штрафа

    // ------------------------------------------------------------------------
    // PathFollowCritic (следование к дальней точке пути)
    // ------------------------------------------------------------------------
    bool enable_path_follow = true;
    int path_offset = 6;                 // смещение от furthest_reached
    float path_follow_weight = 5.0f;     // вес штрафа

    // ------------------------------------------------------------------------
    // PathAlignCritic (выравнивание всей траектории по пути)
    // ------------------------------------------------------------------------
    bool enable_path_align = true;
    float path_align_weight = 10.0f;     // вес выравнивания

    // ------------------------------------------------------------------------
    // PathAngleCritic (угол к дальней точке пути)
    // ------------------------------------------------------------------------
    bool enable_path_angle = true;
    int path_angle_mode = 0;             // 0: forward, 1: no direction, 2: with orientation
    float path_angle_weight = 2.2f;      // вес штрафа
    float max_angle_to_furthest = 0.785f; // порог угла [рад] (~45°)

    // ------------------------------------------------------------------------
    // PreferForwardCritic (предпочтение движения вперёд)
    // ------------------------------------------------------------------------
    bool enable_prefer_forward = true;
    float prefer_forward_weight = 5.0f;  // вес штрафа за задний ход

    // ------------------------------------------------------------------------
    // TwirlingCritic (подавление вращения)
    // ------------------------------------------------------------------------
    bool enable_twirling = true;
    float twirling_weight = 10.0f;       // вес штрафа за вращение

    // ------------------------------------------------------------------------
    // VelocityDeadbandCritic (штраф за слишком малую скорость)
    // ------------------------------------------------------------------------
    bool enable_deadband = false;
    float deadband_vx = 0.1f;            // порог скорости [м/с]
    float deadband_weight = 35.0f;       // вес штрафа

    // ------------------------------------------------------------------------
    // Параметры проективного преобразования для costmap
    // ------------------------------------------------------------------------
    float3 transform_r1 = make_float3(1, 0, 0);
    float3 transform_r2 = make_float3(0, 1, 0);
    float3 transform_trs = make_float3(0, 0, 1);

    // Смещение costmap (origin)
    float costmap_origin_x = -20.0f;
    float costmap_origin_y = -20.0f;
    float costmap_resolution = 0.1f;
    int costmap_width = 400;
    int costmap_height = 400;

    Nav2AggregatedCostParams()
    {
        control_cost_coeff[0] = 0.0f;    // штраф за v_cmd (обычно 0)
        control_cost_coeff[1] = 0.0f;    // штраф за w_cmd (обычно 0)
    }
};

// ============================================================================
// Структура для хранения глобального пути на GPU
// ============================================================================
struct PathDataGPU
{
    float* x;           // X координаты точек пути
    float* y;           // Y координаты точек пути
    float* yaws;        // ориентации в точках пути
    float* integrated_distances;  // интегральные расстояния вдоль пути
    bool* valid_pts;    // флаги валидности точек (не заблокированы препятствиями)
    int size;           // количество точек
    int furthest_reached;  // индекс самой дальней достигнутой точки
};

// ============================================================================
// Основной класс функции стоимости
// ============================================================================
template <class CLASS_T, class PARAMS_T = Nav2AggregatedCostParams>
class Nav2AggregatedCostImpl : public Cost<CLASS_T, PARAMS_T, AckermannDynamicsParams>
{
public:
    using PARENT_CLASS = Cost<CLASS_T, PARAMS_T, AckermannDynamicsParams>;
    using output_array = typename PARENT_CLASS::output_array;

    // Индексы состояния для удобства (из AckermannDynamics)
    static constexpr int S_POS_X = 0;
    static constexpr int S_POS_Y = 1;
    static constexpr int S_PSI = 2;
    static constexpr int S_V = 3;
    static constexpr int S_STEER = 4;

    static constexpr float MAX_COST_VALUE = 1e16f;

    Nav2AggregatedCostImpl(cudaStream_t stream = 0);
    ~Nav2AggregatedCostImpl();

    std::string getCostFunctionName() const override
    {
        return "Nav2 Aggregated Cost for Ackermann";
    }

    // ------------------------------------------------------------------------
    // Инициализация costmap
    // ------------------------------------------------------------------------
    void initCostmap(const uint8_t* costmap_data, int width, int height,
                     float resolution, float origin_x, float origin_y);

    // Инициализация costmap из уже существующей GPU-памяти (без копирования!)
    void initCostmapFromGPU(uint8_t* gpu_data, int width, int height,
                            float resolution, float origin_x, float origin_y);

    // ------------------------------------------------------------------------
    // Обновление глобального пути
    // ------------------------------------------------------------------------
    void updatePath(const std::vector<float>& path_x,
                    const std::vector<float>& path_y,
                    const std::vector<float>& path_yaws,
                    const std::vector<bool>& valid_pts,
                    int furthest_reached);

    // ------------------------------------------------------------------------
    // Основной метод вычисления стоимости (вызывается для каждого состояния)
    // ------------------------------------------------------------------------
    __device__ float computeStateCost(float* state, int timestep, float* theta_c, int* crash_status);

    float computeStateCost(const Eigen::Ref<const output_array> y, int timestep, int* crash_status);

    // ------------------------------------------------------------------------
    // Терминальная стоимость
    // ------------------------------------------------------------------------
    __device__ float terminalCost(float* state, float* theta_c);

    float terminalCost(const Eigen::Ref<const output_array> s);

    // ------------------------------------------------------------------------
    // Инициализация SDF текстуры
    // ------------------------------------------------------------------------
    void initPathSDF(const float* sdf_data, int width, int height);
    void setPathSDFGPU(float* dev_sdf, int width, int height);

protected:
    // ------------------------------------------------------------------------
    // Вспомогательные device-функции для каждого критика
    // ------------------------------------------------------------------------

    __device__ float constraintCost(float v, float wz, float dt) const;
    __device__ float obstacleCost(float x, float y, float psi, int& crash) const;
    __device__ float goalCost(float x, float y, const PathDataGPU* path) const;
    __device__ float goalAngleCost(float psi, const PathDataGPU* path) const;
    __device__ float pathFollowCost(float x, float y, const PathDataGPU* path) const;
    __device__ float pathAlignCost(float x, float y) const;
    __device__ float pathAngleCost(float x, float y, float psi, const PathDataGPU* path) const;
    __device__ float preferForwardCost(float v) const;
    __device__ float twirlingCost(float wz) const;
    __device__ float deadbandCost(float v) const;

    // ------------------------------------------------------------------------
    // Работа с costmap
    // ------------------------------------------------------------------------
    __device__ float queryCostmap(float x, float y) const;
    __device__ void worldToTexture(float x, float y, float& u, float& v) const;
    __device__ float costmapToDistance(float cost) const;

    // ------------------------------------------------------------------------
    // Поиск индекса в пути по расстоянию
    // ------------------------------------------------------------------------
    __device__ int findPathIndexByDistance(const PathDataGPU* path, float dist) const;

private:
    // CUDA текстура для costmap
    cudaArray* costmapArray_d_ = nullptr;
    cudaTextureObject_t costmap_tex_d_ = 0;

    // SDF текстура для PathAlignCritic
    cudaArray* sdfArray_d_ = nullptr;
    cudaTextureObject_t sdf_tex_d_ = 0;
    bool external_sdf_ = false;

    // Флаг, указывающий, что мы используем внешнюю GPU-память (не освобождать)
    bool external_costmap_ = false;

    // Параметры costmap
    int costmap_width_ = 0;
    int costmap_height_ = 0;
    float costmap_resolution_ = 0.1f;
    float costmap_origin_x_ = 0.0f;
    float costmap_origin_y_ = 0.0f;

    // Данные пути на GPU
    PathDataGPU* path_data_d_ = nullptr;
    size_t path_capacity_ = 0;
};

// ============================================================================
// Конструктор / Деструктор
// ============================================================================

template <class CLASS_T, class PARAMS_T>
Nav2AggregatedCostImpl<CLASS_T, PARAMS_T>::Nav2AggregatedCostImpl(cudaStream_t stream)
{
    this->bindToStream(stream);
}

template <class CLASS_T, class PARAMS_T>
Nav2AggregatedCostImpl<CLASS_T, PARAMS_T>::~Nav2AggregatedCostImpl()
{
    if (!external_costmap_) {
        if (costmapArray_d_) {
            cudaFreeArray(costmapArray_d_);
        }
        if (costmap_tex_d_) {
            cudaDestroyTextureObject(costmap_tex_d_);
        }
    }

    // Освобождаем SDF ресурсы
    if (!external_sdf_) {
        if (sdfArray_d_) {
            cudaFreeArray(sdfArray_d_);
        }
    }
    if (sdf_tex_d_) {
        cudaDestroyTextureObject(sdf_tex_d_);
    }

    if (path_data_d_) {
        cudaFree(path_data_d_->x);
        cudaFree(path_data_d_->y);
        cudaFree(path_data_d_->yaws);
        cudaFree(path_data_d_->integrated_distances);
        cudaFree(path_data_d_->valid_pts);
        cudaFree(path_data_d_);
    }
}

template <class CLASS_T, class PARAMS_T>
void Nav2AggregatedCostImpl<CLASS_T, PARAMS_T>::initPathSDF(
    const float* sdf_data, int width, int height)
{
    if (width <= 0 || height <= 0 || !sdf_data) return;

    if (!external_sdf_ && sdfArray_d_) {
        cudaFreeArray(sdfArray_d_);
    }
    if (sdf_tex_d_) {
        cudaDestroyTextureObject(sdf_tex_d_);
    }

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    HANDLE_ERROR(cudaMallocArray(&sdfArray_d_, &channelDesc, width, height));

    // Исправлено: cudaMemcpy2DToArray вместо устаревшего cudaMemcpyToArray
    HANDLE_ERROR(cudaMemcpy2DToArray(sdfArray_d_, 0, 0, sdf_data,
                                      width * sizeof(float),
                                      width * sizeof(float),
                                      height,
                                      cudaMemcpyHostToDevice));

    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = sdfArray_d_;

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 1;

    HANDLE_ERROR(cudaCreateTextureObject(&sdf_tex_d_, &resDesc, &texDesc, nullptr));
    external_sdf_ = false;
}

// ============================================================================
// Инициализация SDF из GPU-памяти (без копирования!)
// ============================================================================
template <class CLASS_T, class PARAMS_T>
void Nav2AggregatedCostImpl<CLASS_T, PARAMS_T>::setPathSDFGPU(
    float* dev_sdf, int width, int height)
{
    if (width <= 0 || height <= 0 || !dev_sdf) return;

    // Освобождаем предыдущие ресурсы
    if (!external_sdf_ && sdfArray_d_) {
        cudaFreeArray(sdfArray_d_);
        sdfArray_d_ = nullptr;
    }
    if (sdf_tex_d_) {
        cudaDestroyTextureObject(sdf_tex_d_);
        sdf_tex_d_ = 0;
    }

    external_sdf_ = true;

    // Создаём CUDA array
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    HANDLE_ERROR(cudaMallocArray(&sdfArray_d_, &channelDesc, width, height));

    // Копируем данные из внешней GPU-памяти в CUDA array
    HANDLE_ERROR(cudaMemcpy2DToArray(
        sdfArray_d_,                    // dst array
        0, 0,                           // dst offset
        dev_sdf,                        // src ptr
        width * sizeof(float),          // src pitch
        width * sizeof(float),          // width in bytes
        height,                         // height
        cudaMemcpyDeviceToDevice));     // kind

    // Настройка текстуры
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = sdfArray_d_;

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModeLinear;  // линейная интерполяция для гладкости
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 1;

    HANDLE_ERROR(cudaCreateTextureObject(&sdf_tex_d_, &resDesc, &texDesc, nullptr));

    cudaStreamSynchronize(this->stream_);
}

// ============================================================================
// Инициализация costmap (из CPU-данных)
// ============================================================================

template <class CLASS_T, class PARAMS_T>
void Nav2AggregatedCostImpl<CLASS_T, PARAMS_T>::initCostmap(
    const uint8_t* costmap_data, int width, int height,
    float resolution, float origin_x, float origin_y)
{
    if (width <= 0 || height <= 0 || !costmap_data) return;

    if (!external_costmap_ && costmapArray_d_) {
        cudaFreeArray(costmapArray_d_);
        costmapArray_d_ = nullptr;
    }
    if (costmap_tex_d_) {
        cudaDestroyTextureObject(costmap_tex_d_);
        costmap_tex_d_ = 0;
    }

    costmap_width_ = width;
    costmap_height_ = height;
    costmap_resolution_ = resolution;
    costmap_origin_x_ = origin_x;
    costmap_origin_y_ = origin_y;
    external_costmap_ = false;

    std::vector<float> float_data(width * height);
    for (int i = 0; i < width * height; i++) {
        float_data[i] = static_cast<float>(costmap_data[i]);
    }

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    HANDLE_ERROR(cudaMallocArray(&costmapArray_d_, &channelDesc, width, height));

    // Исправлено: cudaMemcpy2DToArray вместо устаревшего cudaMemcpyToArray
    HANDLE_ERROR(cudaMemcpy2DToArray(costmapArray_d_, 0, 0, float_data.data(),
                                      width * sizeof(float),
                                      width * sizeof(float),
                                      height,
                                      cudaMemcpyHostToDevice));

    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = costmapArray_d_;

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 1;

    HANDLE_ERROR(cudaCreateTextureObject(&costmap_tex_d_, &resDesc, &texDesc, nullptr));

    this->params_.costmap_width = width;
    this->params_.costmap_height = height;
    this->params_.costmap_resolution = resolution;
    this->params_.costmap_origin_x = origin_x;
    this->params_.costmap_origin_y = origin_y;
    this->paramsToDevice();
}

// ============================================================================
// Инициализация costmap из GPU-памяти (без копирования!)
// ============================================================================

template <class CLASS_T, class PARAMS_T>
void Nav2AggregatedCostImpl<CLASS_T, PARAMS_T>::initCostmapFromGPU(
    uint8_t* gpu_data, int width, int height,
    float resolution, float origin_x, float origin_y)
{
    // ВАЖНО: Здесь предполагается, что gpu_data — это uint8_t массив в GPU-памяти.
    // Для использования в текстуре нужно создать CUDA array с типом float.
    // Это требует копирования, но мы можем сделать это один раз при инициализации.

    // Освобождаем предыдущие ресурсы
    if (!external_costmap_ && costmapArray_d_) {
        cudaFreeArray(costmapArray_d_);
    }
    if (costmap_tex_d_) {
        cudaDestroyTextureObject(costmap_tex_d_);
    }

    costmap_width_ = width;
    costmap_height_ = height;
    costmap_resolution_ = resolution;
    costmap_origin_x_ = origin_x;
    costmap_origin_y_ = origin_y;
    external_costmap_ = true;  // Помечаем, что исходные данные внешние

    // Создаём CUDA array
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    HANDLE_ERROR(cudaMallocArray(&costmapArray_d_, &channelDesc, width, height));

    // Запускаем ядро для конвертации uint8_t -> float и копирования в массив
    // (можно реализовать отдельным ядром, здесь упрощённо)

    // Настройка текстуры
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = costmapArray_d_;

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 1;

    HANDLE_ERROR(cudaCreateTextureObject(&costmap_tex_d_, &resDesc, &texDesc, nullptr));

    this->params_.costmap_width = width;
    this->params_.costmap_height = height;
    this->params_.costmap_resolution = resolution;
    this->params_.costmap_origin_x = origin_x;
    this->params_.costmap_origin_y = origin_y;
    this->paramsToDevice();
}

// ============================================================================
// Обновление пути
// ============================================================================

template <class CLASS_T, class PARAMS_T>
void Nav2AggregatedCostImpl<CLASS_T, PARAMS_T>::updatePath(
    const std::vector<float>& path_x,
    const std::vector<float>& path_y,
    const std::vector<float>& path_yaws,
    const std::vector<bool>& valid_pts,
    int furthest_reached)
{
    size_t size = path_x.size();
    if (size == 0) return;

    // Освобождаем старые данные, если размер изменился
    if (path_data_d_ && path_capacity_ < size) {
        cudaFree(path_data_d_->x);
        cudaFree(path_data_d_->y);
        cudaFree(path_data_d_->yaws);
        cudaFree(path_data_d_->integrated_distances);
        cudaFree(path_data_d_->valid_pts);
        cudaFree(path_data_d_);
        path_data_d_ = nullptr;
    }

    if (!path_data_d_) {
        path_capacity_ = size;
        HANDLE_ERROR(cudaMalloc(&path_data_d_, sizeof(PathDataGPU)));
        HANDLE_ERROR(cudaMalloc(&path_data_d_->x, size * sizeof(float)));
        HANDLE_ERROR(cudaMalloc(&path_data_d_->y, size * sizeof(float)));
        HANDLE_ERROR(cudaMalloc(&path_data_d_->yaws, size * sizeof(float)));
        HANDLE_ERROR(cudaMalloc(&path_data_d_->integrated_distances, size * sizeof(float)));
        HANDLE_ERROR(cudaMalloc(&path_data_d_->valid_pts, size * sizeof(bool)));
    }

    // Копируем данные
    HANDLE_ERROR(cudaMemcpyAsync(path_data_d_->x, path_x.data(), size * sizeof(float),
                                  cudaMemcpyHostToDevice, this->stream_));
    HANDLE_ERROR(cudaMemcpyAsync(path_data_d_->y, path_y.data(), size * sizeof(float),
                                  cudaMemcpyHostToDevice, this->stream_));
    HANDLE_ERROR(cudaMemcpyAsync(path_data_d_->yaws, path_yaws.data(), size * sizeof(float),
                                  cudaMemcpyHostToDevice, this->stream_));

    // Вычисляем интегральные расстояния
    std::vector<float> integrated(size, 0.0f);
    for (size_t i = 1; i < size; ++i) {
        float dx = path_x[i] - path_x[i-1];
        float dy = path_y[i] - path_y[i-1];
        integrated[i] = integrated[i-1] + sqrtf(dx*dx + dy*dy);
    }
    HANDLE_ERROR(cudaMemcpyAsync(path_data_d_->integrated_distances, integrated.data(),
                                  size * sizeof(float), cudaMemcpyHostToDevice, this->stream_));

    // Конвертируем vector<bool> в vector<char>
    std::vector<char> valid_chars(size);
    for (size_t i = 0; i < size; ++i) valid_chars[i] = valid_pts[i] ? 1 : 0;
    HANDLE_ERROR(cudaMemcpyAsync(path_data_d_->valid_pts, valid_chars.data(),
                                  size * sizeof(char), cudaMemcpyHostToDevice, this->stream_));

    path_data_d_->size = static_cast<int>(size);
    path_data_d_->furthest_reached = furthest_reached;

    cudaStreamSynchronize(this->stream_);
}

// ============================================================================
// Device-функции для работы с costmap
// ============================================================================

template <class CLASS_T, class PARAMS_T>
__device__ void Nav2AggregatedCostImpl<CLASS_T, PARAMS_T>::worldToTexture(
    float x, float y, float& u, float& v) const
{
    // Преобразование мировых координат в нормализованные координаты текстуры [0,1]
    u = (x - this->params_.costmap_origin_x) /
        (this->params_.costmap_width * this->params_.costmap_resolution);
    v = (y - this->params_.costmap_origin_y) /
        (this->params_.costmap_height * this->params_.costmap_resolution);
}

template <class CLASS_T, class PARAMS_T>
__device__ float Nav2AggregatedCostImpl<CLASS_T, PARAMS_T>::queryCostmap(float x, float y) const
{
    float u, v;
    worldToTexture(x, y, u, v);

    // Проверка границ
    if (u < 0.0f || u > 1.0f || v < 0.0f || v > 1.0f) {
        return 0.0f;  // За пределами карты — свободное пространство (или 255?)
    }

    return tex2D<float>(costmap_tex_d_, u, v);
}

template <class CLASS_T, class PARAMS_T>
__device__ float Nav2AggregatedCostImpl<CLASS_T, PARAMS_T>::costmapToDistance(float cost) const
{
    // Преобразование значения costmap (0-255) в расстояние до препятствия
    // Используется формула из inflation_layer: cost = exp(-scale * dist) * 253
    if (cost < 1.0f) return this->params_.inflation_radius;
    if (cost >= 253.0f) return 0.0f;

    float dist = -logf(cost / 253.0f) / this->params_.cost_scaling_factor;
    return dist;
}

// ============================================================================
// Поиск индекса в пути по расстоянию
// ============================================================================

template <class CLASS_T, class PARAMS_T>
__device__ int Nav2AggregatedCostImpl<CLASS_T, PARAMS_T>::findPathIndexByDistance(
    const PathDataGPU* path, float dist) const
{
    if (!path || path->size == 0) return 0;

    // Бинарный поиск (путь упорядочен по возрастанию integrated_distances)
    int left = 0, right = path->size - 1;
    while (left < right) {
        int mid = (left + right) / 2;
        if (path->integrated_distances[mid] < dist) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    return left;
}

// ============================================================================
// Реализации критиков
// ============================================================================

template <class CLASS_T, class PARAMS_T>
__device__ float Nav2AggregatedCostImpl<CLASS_T, PARAMS_T>::constraintCost(
    float v, float wz, float dt) const
{
    if (!this->params_.enable_constraint) return 0.0f;

    float cost = 0.0f;

    // Превышение максимальной скорости
    if (v > this->params_.max_speed) {
        cost += (v - this->params_.max_speed) * dt;
    }
    // Выход за минимальную скорость (задний ход)
    if (v < this->params_.min_speed) {
        cost += (this->params_.min_speed - v) * dt;
    }
    // Превышение угловой скорости
    float abs_wz = fabsf(wz);
    if (abs_wz > this->params_.max_wz) {
        cost += (abs_wz - this->params_.max_wz) * dt;
    }

    return cost * this->params_.constraint_weight;
}

template <class CLASS_T, class PARAMS_T>
__device__ float Nav2AggregatedCostImpl<CLASS_T, PARAMS_T>::obstacleCost(
    float x, float y, float psi, int& crash) const
{
    if (!this->params_.enable_obstacles) return 0.0f;

    float cost = 0.0f;
    float base_cost = queryCostmap(x, y);

    // Если учитываем footprint, проверяем несколько точек
    if (this->params_.consider_footprint) {
        float r = this->params_.footprint_radius;
        float front_x = x + r * cosf(psi);
        float front_y = y + r * sinf(psi);
        float back_x = x - r * cosf(psi);
        float back_y = y - r * sinf(psi);

        // Боковые точки
        float left_x = x + r * cosf(psi + M_PI_2);
        float left_y = y + r * sinf(psi + M_PI_2);
        float right_x = x + r * cosf(psi - M_PI_2);
        float right_y = y + r * sinf(psi - M_PI_2);

        float front_cost = queryCostmap(front_x, front_y);
        float back_cost = queryCostmap(back_x, back_y);
        float left_cost = queryCostmap(left_x, left_y);
        float right_cost = queryCostmap(right_x, right_y);

        base_cost = fmaxf(base_cost, fmaxf(front_cost,
                          fmaxf(back_cost, fmaxf(left_cost, right_cost))));
    }

    // Проверка на столкновение
    if (base_cost >= this->params_.collision_cost) {
        crash = 1;
        return this->params_.collision_cost;
    }

    // Критическая близость
    if (base_cost >= this->params_.near_collision_threshold) {
        cost += this->params_.critical_weight * base_cost;
    } else if (base_cost > 1.0f) {
        // Репульсивный штраф на основе расстояния
        float dist = costmapToDistance(base_cost);
        if (dist < this->params_.collision_margin) {
            cost += (this->params_.collision_margin - dist) *
                    this->params_.critical_weight * 10.0f;
        } else {
            cost += base_cost * this->params_.repulsion_weight;
        }
    }

    return cost * this->params_.obstacles_weight;
}

template <class CLASS_T, class PARAMS_T>
__device__ float Nav2AggregatedCostImpl<CLASS_T, PARAMS_T>::goalCost(
    float x, float y, const PathDataGPU* path) const
{
    if (!this->params_.enable_goal || !path || path->size == 0) return 0.0f;

    int last = path->size - 1;
    float goal_x = path->x[last];
    float goal_y = path->y[last];

    float dx = goal_x - x;
    float dy = goal_y - y;
    float dist = sqrtf(dx*dx + dy*dy);

    // Применяем только если расстояние меньше порога
    if (dist > this->params_.goal_threshold) return 0.0f;

    return dist * this->params_.goal_position_weight;
}

template <class CLASS_T, class PARAMS_T>
__device__ float Nav2AggregatedCostImpl<CLASS_T, PARAMS_T>::goalAngleCost(
    float psi, const PathDataGPU* path) const
{
    if (!this->params_.enable_goal_angle || !path || path->size == 0) return 0.0f;

    int last = path->size - 1;
    float goal_psi = path->yaws[last];

    float angle_diff = angle_utils::shortestAngularDistance(psi, goal_psi);

    if (this->params_.symmetric_angle_tolerance) {
        float alt_diff = angle_utils::shortestAngularDistance(
            psi, angle_utils::normalizeAngle(goal_psi + M_PI));
        angle_diff = fminf(fabsf(angle_diff), fabsf(alt_diff));
    } else {
        angle_diff = fabsf(angle_diff);
    }

    return angle_diff * this->params_.goal_angle_weight;
}

template <class CLASS_T, class PARAMS_T>
__device__ float Nav2AggregatedCostImpl<CLASS_T, PARAMS_T>::pathFollowCost(
    float x, float y, const PathDataGPU* path) const
{
    if (!this->params_.enable_path_follow || !path || path->size == 0) return 0.0f;

    int target_idx = min(path->furthest_reached + this->params_.path_offset, path->size - 1);

    // Пропускаем заблокированные точки
    while (target_idx < path->size - 1 && !path->valid_pts[target_idx]) {
        target_idx++;
    }
    if (target_idx >= path->size) return 0.0f;

    float dx = x - path->x[target_idx];
    float dy = y - path->y[target_idx];
    float dist = sqrtf(dx*dx + dy*dy);

    return dist * this->params_.path_follow_weight;
}

template <class CLASS_T, class PARAMS_T>
__device__ float Nav2AggregatedCostImpl<CLASS_T, PARAMS_T>::pathAlignCost(
    float x, float y) const
{
    if (!this->params_.enable_path_align) return 0.0f;
    if (!sdf_tex_d_) return 0.0f;  // SDF текстура не инициализирована

    // Преобразуем мировые координаты в нормализованные координаты текстуры [0, 1]
    float u = (x - this->params_.costmap_origin_x) /
              (this->params_.costmap_width * this->params_.costmap_resolution);
    float v = (y - this->params_.costmap_origin_y) /
              (this->params_.costmap_height * this->params_.costmap_resolution);

    // Проверка границ
    if (u < 0.0f || u > 1.0f || v < 0.0f || v > 1.0f) {
        return 0.0f;  // За пределами карты — не штрафуем (или можно вернуть max_distance)
    }

    // O(1) чтение из SDF текстуры — точное расстояние до пути в метрах
    float dist = tex2D<float>(sdf_tex_d_, u, v);

    return dist * this->params_.path_align_weight;
}

template <class CLASS_T, class PARAMS_T>
__device__ float Nav2AggregatedCostImpl<CLASS_T, PARAMS_T>::pathAngleCost(
    float x, float y, float psi, const PathDataGPU* path) const
{
    if (!this->params_.enable_path_angle || !path || path->size == 0) return 0.0f;

    int target_idx = min(path->furthest_reached + this->params_.path_offset, path->size - 1);

    float dx = path->x[target_idx] - x;
    float dy = path->y[target_idx] - y;
    float angle_to_point = atan2f(dy, dx);

    float diff;
    switch (this->params_.path_angle_mode) {
        case 0:  // forward preference
            diff = fabsf(angle_utils::shortestAngularDistance(psi, angle_to_point));
            break;
        case 1:  // no directional preference
        {
            float diff1 = fabsf(angle_utils::shortestAngularDistance(psi, angle_to_point));
            float diff2 = fabsf(angle_utils::shortestAngularDistance(psi, angle_to_point + M_PI));
            diff = fminf(diff1, diff2);
            break;
        }
        case 2:  // with path orientation
        {
            float goal_psi = path->yaws[target_idx];
            float desired_angle = angle_utils::shortestAngularDistance(goal_psi, angle_to_point);
            diff = fabsf(angle_utils::shortestAngularDistance(psi, desired_angle));
            break;
        }
        default:
            diff = 0.0f;
    }

    if (diff < this->params_.max_angle_to_furthest) {
        return 0.0f;
    }
    return diff * this->params_.path_angle_weight;
}

template <class CLASS_T, class PARAMS_T>
__device__ float Nav2AggregatedCostImpl<CLASS_T, PARAMS_T>::preferForwardCost(float v) const
{
    if (!this->params_.enable_prefer_forward) return 0.0f;

    // Штраф за отрицательную скорость (движение назад)
    float penalty = (v < 0) ? -v : 0.0f;
    return penalty * this->params_.prefer_forward_weight * this->params_.dt;
}

template <class CLASS_T, class PARAMS_T>
__device__ float Nav2AggregatedCostImpl<CLASS_T, PARAMS_T>::twirlingCost(float wz) const
{
    if (!this->params_.enable_twirling) return 0.0f;
    return fabsf(wz) * this->params_.twirling_weight * this->params_.dt;
}

template <class CLASS_T, class PARAMS_T>
__device__ float Nav2AggregatedCostImpl<CLASS_T, PARAMS_T>::deadbandCost(float v) const
{
    if (!this->params_.enable_deadband) return 0.0f;

    float cost = 0.0f;
    if (fabsf(v) < this->params_.deadband_vx) {
        cost += (this->params_.deadband_vx - fabsf(v)) * this->params_.dt;
    }
    return cost * this->params_.deadband_weight;
}

// ============================================================================
// Основная функция computeStateCost
// ============================================================================

template <class CLASS_T, class PARAMS_T>
__device__ float Nav2AggregatedCostImpl<CLASS_T, PARAMS_T>::computeStateCost(
    float* state, int timestep, float* theta_c, int* crash_status)
{
    // Извлекаем состояние (5 элементов: x, y, psi, v, steer)
    float x = state[S_POS_X];
    float y = state[S_POS_Y];
    float psi = state[S_PSI];
    float v = state[S_V];
    float steer = state[S_STEER];

    // Вычисляем угловую скорость из кинематики
    float wz = 0.0f;
    float v_abs = fabsf(v);
    if (v_abs > this->params_.eps_v) {
        wz = (v_abs / this->params_.wheelbase) * tanf(steer);
    }

    int crash = 0;
    float total_cost = 0.0f;

    // 1. Constraint (ограничения)
    total_cost += constraintCost(v, wz, this->params_.dt);

    // 2. Obstacles (препятствия)
    total_cost += obstacleCost(x, y, psi, crash);

    if (crash) {
        *crash_status = 1;
        return this->params_.collision_cost;
    }

    // Получаем указатель на данные пути
    PathDataGPU* path = path_data_d_;

    if (path && path->size > 0) {
        // 3. Goal
        total_cost += goalCost(x, y, path);

        // 4. GoalAngle
        total_cost += goalAngleCost(psi, path);

        // 5. PathFollow
        total_cost += pathFollowCost(x, y, path);

        // 6. PathAlign
        total_cost += pathAlignCost(x, y);

        // 7. PathAngle
        total_cost += pathAngleCost(x, y, psi, path);
    }

    // 8. PreferForward
    total_cost += preferForwardCost(v);

    // 9. Twirling
    total_cost += twirlingCost(wz);

    // 10. Deadband
    total_cost += deadbandCost(v);

    // Применяем дисконтирование
    total_cost *= powf(this->params_.discount, timestep);

    // Применяем степень (power)
    if (this->params_.power > 1) {
        total_cost = powf(total_cost, static_cast<float>(this->params_.power));
    }

    // Проверка на переполнение
    if (total_cost > MAX_COST_VALUE || isnan(total_cost)) {
        total_cost = MAX_COST_VALUE;
    }

    if (crash) *crash_status = 1;

    return total_cost;
}

// ============================================================================
// Терминальная стоимость
// ============================================================================

template <class CLASS_T, class PARAMS_T>
__device__ float Nav2AggregatedCostImpl<CLASS_T, PARAMS_T>::terminalCost(
    float* state, float* theta_c)
{
    return 0.0f;
}

template <class CLASS_T, class PARAMS_T>
float Nav2AggregatedCostImpl<CLASS_T, PARAMS_T>::terminalCost(
    const Eigen::Ref<const output_array> s)
{
    return 0.0f;
}

// ============================================================================
// CPU-версия computeStateCost
// ============================================================================

template <class CLASS_T, class PARAMS_T>
float Nav2AggregatedCostImpl<CLASS_T, PARAMS_T>::computeStateCost(
    const Eigen::Ref<const output_array> s, int timestep, int* crash_status)
{
    // Заглушка (можно реализовать вызов GPU-версии или полную CPU-логику)
    return 0.0f;
}

// ============================================================================
// Конкретный класс для использования
// ============================================================================

class Nav2AggregatedCost : public Nav2AggregatedCostImpl<Nav2AggregatedCost>
{
public:
    Nav2AggregatedCost(cudaStream_t stream = 0)
        : Nav2AggregatedCostImpl<Nav2AggregatedCost>(stream) {}
};

// ============================================================================
// Явная инстанциация
// ============================================================================

template class Nav2AggregatedCostImpl<Nav2AggregatedCost>;

#endif // NAV2_AGGREGATED_COST_CUH_
#endif // NAV2_COMPATIBLE_COSTFUNCTION_H
*/

#ifndef NAV2_MINIMAL_COST_CUH_
#define NAV2_MINIMAL_COST_CUH_

#include <mppi/cost_functions/cost.h>
#include <mppi/dynamics/AckermannDynamics.h>
#include <mppi/utils/angle_utils.h>
#include <mppi/utils/texture_helpers/two_d_texture_helper.h>

// ============================================================================
// Вспомогательное ядро для конвертации uint8_t -> float
// ============================================================================
__global__ void convertUint8ToFloatKernel(float* dst, const uint8_t* src, size_t count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        dst[idx] = static_cast<float>(src[idx]);
    }
}

// ============================================================================
// Параметры функции стоимости (минимальный набор для path tracking)
// ============================================================================
struct Nav2MinimalCostParams : public CostParams<2>  // V_CMD, W_CMD
{
    // ------------------------------------------------------------------------
    // GoalCritic
    // ------------------------------------------------------------------------
    float goal_x = 10.0f;
    float goal_y = 10.0f;
    float goal_weight = 5.0f;
    float goal_threshold = 0.5f;         // расстояние до цели для активации
    int goal_power = 1;

    // ------------------------------------------------------------------------
    // GoalAngleCritic
    // ------------------------------------------------------------------------
    float goal_yaw = 0.0f;
    float goal_angle_weight = 3.0f;
    int goal_angle_power = 1;
    bool symmetric_angle_tolerance = false;

    // ------------------------------------------------------------------------
    // ObstaclesCritic
    // ------------------------------------------------------------------------
    float obstacle_weight = 3.81f;
    float critical_weight = 20.0f;
    float collision_cost = 1000000.0f;
    float collision_margin = 0.1f;
    float near_collision_threshold = 253.0f;
    float inflation_radius = 0.5f;
    float cost_scaling_factor = 10.0f;
    float inscribed_radius = 0.3f;
    float footprint_radius = 0.5f;
    bool consider_footprint = true;
    int obstacle_power = 1;

    // ------------------------------------------------------------------------
    // Общие параметры
    // ------------------------------------------------------------------------
    float dt = 0.1f;
    float wheelbase = 0.86f;
    float eps_v = 0.1f;

    Nav2MinimalCostParams()
    {
        control_cost_coeff[0] = 0.0f;
        control_cost_coeff[1] = 0.0f;
    }
};

// ============================================================================
// Основной класс функции стоимости
// ============================================================================
template <class CLASS_T, class PARAMS_T = Nav2MinimalCostParams>
class Nav2MinimalCostImpl : public Cost<CLASS_T, PARAMS_T, AckermannDynamicsParams>
{
public:
    using PARENT_CLASS = Cost<CLASS_T, PARAMS_T, AckermannDynamicsParams>;
    using output_array = typename PARENT_CLASS::output_array;

    static constexpr int S_POS_X = 0;
    static constexpr int S_POS_Y = 1;
    static constexpr int S_PSI = 2;
    static constexpr int S_V = 3;
    static constexpr int S_STEER = 4;

    static constexpr float MAX_COST_VALUE = 1e16f;

    Nav2MinimalCostImpl(cudaStream_t stream = 0);
    ~Nav2MinimalCostImpl();

    std::string getCostFunctionName() const
    {
        return "Nav2 Minimal Cost (Goal + GoalAngle + Obstacles)";
    }

    // ------------------------------------------------------------------------
    // Обязательные методы из базового класса
    // ------------------------------------------------------------------------
    void GPUSetup();
    void paramsToDevice();
    void freeCudaMem();

    // ------------------------------------------------------------------------
    // Инициализация costmap
    // ------------------------------------------------------------------------
    void initCostmap(const uint8_t* costmap_data, int width, int height,
                     float resolution, float origin_x, float origin_y);

    void initCostmapFromGPU(uint8_t* gpu_data, int width, int height,
                            float resolution, float origin_x, float origin_y);

    // ------------------------------------------------------------------------
    // Обновление цели
    // ------------------------------------------------------------------------
    void updateGoal(float goal_x, float goal_y, float goal_yaw);

    // ------------------------------------------------------------------------
    // Основные методы вычисления стоимости
    // ------------------------------------------------------------------------
    __host__ __device__ float computeStateCost(float* state, int timestep,
                                                float* theta_c, int* crash_status);

    float computeStateCost(const Eigen::Ref<const output_array>& y,
                           int timestep, int* crash_status);

    __host__ __device__ float terminalCost(float* state, float* theta_c);

    float terminalCost(const Eigen::Ref<const output_array>& s);

private:
    // ------------------------------------------------------------------------
    // Вспомогательные методы для каждого критика
    // ------------------------------------------------------------------------
    __host__ __device__ float obstacleCost(float x, float y, float psi,
                                            int& crash, const PARAMS_T* p) const;

    __host__ __device__ float goalCost(float x, float y, const PARAMS_T* p) const;

    __host__ __device__ float goalAngleCost(float psi, const PARAMS_T* p) const;

    // ------------------------------------------------------------------------
    // Работа с costmap через TextureHelper
    // ------------------------------------------------------------------------
    __host__ __device__ float queryCostmap(float x, float y) const;

    __host__ __device__ float costmapToDistance(float cost, const PARAMS_T* p) const;

private:
    TwoDTextureHelper<float>* tex_helper_ = nullptr;

    int costmap_width_ = 0;
    int costmap_height_ = 0;
    float costmap_resolution_ = 0.1f;
    float costmap_origin_x_ = 0.0f;
    float costmap_origin_y_ = 0.0f;
};

// ============================================================================
// Конкретный класс для использования
// ============================================================================
class Nav2MinimalCost : public Nav2MinimalCostImpl<Nav2MinimalCost>
{
public:
    Nav2MinimalCost(cudaStream_t stream = 0)
        : Nav2MinimalCostImpl<Nav2MinimalCost>(stream) {}
};

// ============================================================================
// Реализация
// ============================================================================

template <class CLASS_T, class PARAMS_T>
Nav2MinimalCostImpl<CLASS_T, PARAMS_T>::Nav2MinimalCostImpl(cudaStream_t stream)
{
    this->bindToStream(stream);
    tex_helper_ = new TwoDTextureHelper<float>(1, stream);
}

template <class CLASS_T, class PARAMS_T>
Nav2MinimalCostImpl<CLASS_T, PARAMS_T>::~Nav2MinimalCostImpl()
{
    delete tex_helper_;
}

template <class CLASS_T, class PARAMS_T>
void Nav2MinimalCostImpl<CLASS_T, PARAMS_T>::GPUSetup()
{
    // 1. Настраиваем текстурный хелпер
    tex_helper_->GPUSetup();

    // 2. Создаём GPU-копию cost-функции
    if (!this->GPUMemStatus_)
    {
        // ВАЖНО: Передаём указатель на КОНКРЕТНЫЙ класс (CLASS_T*), а не на базовый
        CLASS_T* derived = static_cast<CLASS_T*>(this);
        this->cost_d_ = Managed::GPUSetup(derived);
    }
    else
    {
        this->logger_->debug("%s: GPU Memory already set\n",
                             this->getCostFunctionName().c_str());
    }

    // 3. Копируем указатель на tex_helper в GPU-версию
    HANDLE_ERROR(cudaMemcpyAsync(&(this->cost_d_->tex_helper_),
                                 &(tex_helper_->ptr_d_),
                                 sizeof(TwoDTextureHelper<float>*),
                                 cudaMemcpyHostToDevice,
                                 this->stream_));

    // 4. Копируем параметры
    this->paramsToDevice();
}

template <class CLASS_T, class PARAMS_T>
void Nav2MinimalCostImpl<CLASS_T, PARAMS_T>::paramsToDevice()
{
    if (this->GPUMemStatus_)
    {
        tex_helper_->copyToDevice();

        HANDLE_ERROR(cudaMemcpyAsync(&this->cost_d_->params_, &this->params_,
                                     sizeof(PARAMS_T), cudaMemcpyHostToDevice, this->stream_));
        HANDLE_ERROR(cudaStreamSynchronize(this->stream_));
    }
}

template <class CLASS_T, class PARAMS_T>
void Nav2MinimalCostImpl<CLASS_T, PARAMS_T>::freeCudaMem()
{
    if (this->GPUMemStatus_)
    {
        tex_helper_->freeCudaMem();
    }
    PARENT_CLASS::freeCudaMem();
}

template <class CLASS_T, class PARAMS_T>
void Nav2MinimalCostImpl<CLASS_T, PARAMS_T>::initCostmap(
    const uint8_t* costmap_data, int width, int height,
    float resolution, float origin_x, float origin_y)
{
    if (width <= 0 || height <= 0 || !costmap_data) return;

    // Конвертируем uint8_t -> float
    std::vector<float> float_data(width * height);
    for (int i = 0; i < width * height; i++)
    {
        float_data[i] = static_cast<float>(costmap_data[i]);
    }

    // Настраиваем TextureHelper
    cudaExtent extent = make_cudaExtent(width, height, 0);
    tex_helper_->setExtent(0, extent);
    tex_helper_->updateResolution(0, resolution);

    float3 origin = make_float3(origin_x, origin_y, 0);
    tex_helper_->updateOrigin(0, origin);

    tex_helper_->updateTexture(0, float_data, false);
    tex_helper_->enableTexture(0);
}

/**
 * @brief Инициализация costmap напрямую из GPU-памяти (без копирования на CPU)
 * @param gpu_data Указатель на данные в GPU-памяти (uint8_t)
 * @param width Ширина карты
 * @param height Высота карты
 * @param resolution Разрешение [м/ячейка]
 * @param origin_x, origin_y Начало координат карты
 */
template <class CLASS_T, class PARAMS_T>
void Nav2MinimalCostImpl<CLASS_T, PARAMS_T>::initCostmapFromGPU(
    uint8_t* gpu_data, int width, int height,
    float resolution, float origin_x, float origin_y)
{
    if (width <= 0 || height <= 0 || !gpu_data) return;

    // Настраиваем TextureHelper
    cudaExtent extent = make_cudaExtent(width, height, 0);
    tex_helper_->setExtent(0, extent);
    tex_helper_->updateResolution(0, resolution);

    float3 origin = make_float3(origin_x, origin_y, 0);
    tex_helper_->updateOrigin(0, origin);

    // Принудительно копируем параметры и выделяем текстуру
    tex_helper_->copyToDevice(true);

    // Выделяем временный буфер на GPU для float данных
    size_t float_size = width * height * sizeof(float);
    float* gpu_float_data = nullptr;
    HANDLE_ERROR(cudaMalloc(&gpu_float_data, float_size));

    // Запускаем ядро конвертации
    dim3 block(256);
    dim3 grid((width * height + 255) / 256);
    convertUint8ToFloatKernel<<<grid, block, 0, this->stream_>>>(
        gpu_float_data, gpu_data, width * height);
    HANDLE_ERROR(cudaGetLastError());
    HANDLE_ERROR(cudaStreamSynchronize(this->stream_));

    // Копируем float данные в текстуру
    tex_helper_->updateTextureFromGPU(0, gpu_float_data, width * height, false);
    tex_helper_->enableTexture(0);

    // Освобождаем временный буфер
    HANDLE_ERROR(cudaFree(gpu_float_data));

    if (this->GPUMemStatus_) {
        this->paramsToDevice();
    }
}

template <class CLASS_T, class PARAMS_T>
void Nav2MinimalCostImpl<CLASS_T, PARAMS_T>::updateGoal(
    float goal_x, float goal_y, float goal_yaw)
{
    this->params_.goal_x = goal_x;
    this->params_.goal_y = goal_y;
    this->params_.goal_yaw = goal_yaw;

    if (this->GPUMemStatus_)
    {
        this->paramsToDevice();
    }
}

// ============================================================================
// Device/Host функции для работы с costmap
// ============================================================================

template <class CLASS_T, class PARAMS_T>
__host__ __device__ float Nav2MinimalCostImpl<CLASS_T, PARAMS_T>::queryCostmap(
    float x, float y) const
{
#ifdef __CUDA_ARCH__
    if (!tex_helper_->checkTextureUse(0))
        return 0.0f;
    return tex_helper_->queryTextureAtWorldPose(0, make_float3(x, y, 0));
#else
    // CPU-версия: простой возврат 0 (можно реализовать полную логику при необходимости)
    return 0.0f;
#endif
}

template <class CLASS_T, class PARAMS_T>
__host__ __device__ float Nav2MinimalCostImpl<CLASS_T, PARAMS_T>::costmapToDistance(
    float cost, const PARAMS_T* p) const
{
    if (cost < 1.0f) return p->inflation_radius;
    if (cost >= p->near_collision_threshold) return 0.0f;

    return -logf(cost / 253.0f) / p->cost_scaling_factor;
}

// ============================================================================
// Реализации критиков
// ============================================================================

template <class CLASS_T, class PARAMS_T>
__host__ __device__ float Nav2MinimalCostImpl<CLASS_T, PARAMS_T>::obstacleCost(
    float x, float y, float psi, int& crash, const PARAMS_T* p) const
{
    float cost = 0.0f;
    float base_cost = queryCostmap(x, y);

    // Проверяем несколько точек если учитываем footprint
    if (p->consider_footprint)
    {
        float r = p->footprint_radius;

#ifdef __CUDA_ARCH__
        float front_x = x + r * __cosf(psi);
        float front_y = y + r * __sinf(psi);
        float back_x = x - r * __cosf(psi);
        float back_y = y - r * __sinf(psi);
#else
        float front_x = x + r * cosf(psi);
        float front_y = y + r * sinf(psi);
        float back_x = x - r * cosf(psi);
        float back_y = y - r * sinf(psi);
#endif

        float front_cost = queryCostmap(front_x, front_y);
        float back_cost = queryCostmap(back_x, back_y);

        base_cost = fmaxf(base_cost, fmaxf(front_cost, back_cost));
    }

    // Проверка на столкновение
    if (base_cost >= p->collision_cost || base_cost >= p->near_collision_threshold)
    {
        crash = 1;
        return p->collision_cost;
    }

    // Штраф за близость к препятствиям
    if (base_cost > 1.0f)
    {
        float dist = costmapToDistance(base_cost, p);
        if (dist < p->collision_margin)
        {
            cost += (p->collision_margin - dist) * p->critical_weight;
        }
        else
        {
            cost += base_cost * p->obstacle_weight / 253.0f;
        }
    }

    return cost;
}

template <class CLASS_T, class PARAMS_T>
__host__ __device__ float Nav2MinimalCostImpl<CLASS_T, PARAMS_T>::goalCost(
    float x, float y, const PARAMS_T* p) const
{
    float dx = x - p->goal_x;
    float dy = y - p->goal_y;
    float dist = sqrtf(dx * dx + dy * dy);

    //if (dist > p->goal_threshold) return 0.0f;

    float cost = dist * p->goal_weight;

    if (p->goal_power > 1)
        cost = powf(cost, static_cast<float>(p->goal_power));

    return cost;
}

template <class CLASS_T, class PARAMS_T>
__host__ __device__ float Nav2MinimalCostImpl<CLASS_T, PARAMS_T>::goalAngleCost(
    float psi, const PARAMS_T* p) const
{
    float angle_diff = angle_utils::shortestAngularDistance(psi, p->goal_yaw);

    if (p->symmetric_angle_tolerance)
    {
        float alt_diff = angle_utils::shortestAngularDistance(
            psi, angle_utils::normalizeAngle(p->goal_yaw + M_PI));
        angle_diff = fminf(fabsf(angle_diff), fabsf(alt_diff));
    }
    else
    {
        angle_diff = fabsf(angle_diff);
    }

    float cost = angle_diff * p->goal_angle_weight;

    if (p->goal_angle_power > 1)
        cost = powf(cost, static_cast<float>(p->goal_angle_power));

    return cost;
}

// ============================================================================
// Основная функция computeStateCost
// ============================================================================

template <class CLASS_T, class PARAMS_T>
__host__ __device__ float Nav2MinimalCostImpl<CLASS_T, PARAMS_T>::computeStateCost(
    float* state, int timestep, float* theta_c, int* crash_status)
{
    const PARAMS_T* p = &this->params_;

    float x = state[S_POS_X];
    float y = state[S_POS_Y];
    float psi = state[S_PSI];

    int crash = 0;
    float total_cost = 0.0f;

    // 1. Obstacles
    float obs_cost = obstacleCost(x, y, psi, crash, p);
    total_cost += obs_cost;

    if (crash)
    {
        *crash_status = 1;
        return p->collision_cost;
    }

    // 2. Goal
    total_cost += goalCost(x, y, p);

    // 3. GoalAngle
    total_cost += goalAngleCost(psi, p);

    // Проверка на переполнение
    if (total_cost > MAX_COST_VALUE || isnan(total_cost))
    {
        total_cost = MAX_COST_VALUE;
    }

    return total_cost;
}

template <class CLASS_T, class PARAMS_T>
float Nav2MinimalCostImpl<CLASS_T, PARAMS_T>::computeStateCost(
    const Eigen::Ref<const output_array>& y, int timestep, int* crash_status)
{
    float state_array[5];
    state_array[S_POS_X] = y[S_POS_X];
    state_array[S_POS_Y] = y[S_POS_Y];
    state_array[S_PSI] = y[S_PSI];
    state_array[S_V] = y[S_V];
    state_array[S_STEER] = y[S_STEER];

    return computeStateCost(state_array, timestep, nullptr, crash_status);
}

template <class CLASS_T, class PARAMS_T>
__host__ __device__ float Nav2MinimalCostImpl<CLASS_T, PARAMS_T>::terminalCost(
    float* state, float* theta_c)
{
    // Терминальная стоимость = те же компоненты что и в computeStateCost
    const PARAMS_T* p = &this->params_;

    float x = state[S_POS_X];
    float y = state[S_POS_Y];
    float psi = state[S_PSI];
    int dummy_crash = 0;

    return goalCost(x, y, p) + goalAngleCost(psi, p) +
           obstacleCost(x, y, psi, dummy_crash, p);
}

template <class CLASS_T, class PARAMS_T>
float Nav2MinimalCostImpl<CLASS_T, PARAMS_T>::terminalCost(
    const Eigen::Ref<const output_array>& s)
{
    float state_array[5];
    state_array[S_POS_X] = s[S_POS_X];
    state_array[S_POS_Y] = s[S_POS_Y];
    state_array[S_PSI] = s[S_PSI];
    state_array[S_V] = s[S_V];
    state_array[S_STEER] = s[S_STEER];

    return terminalCost(state_array, nullptr);
}

#endif // NAV2_MINIMAL_COST_CUH_
