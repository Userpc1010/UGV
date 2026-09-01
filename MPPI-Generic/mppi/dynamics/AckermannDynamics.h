//#ifndef ACKERMANNDYNAMICS_H
//#define ACKERMANNDYNAMICS_H

//#include <mppi/dynamics/dynamics.h>
//#include <mppi/utils/angle_utils.h>
//#include <Eigen/Dense>
//#include <map>
//#include <string>
//#include <cmath>

//// ============================================================================
//// Параметры динамики Ackermann
//// ============================================================================
//struct AckermannDynamicsParams : public DynamicsParams
//{
//    enum class StateIndex : int
//    {
//        POS_X = 0,          // позиция X [м]
//        POS_Y,              // позиция Y [м]
//        PSI,                // угол рысканья (курс) [рад]
//        V,                  // продольная скорость [м/с]
//        STEER_ANGLE,        // фактический угол поворота колёс [рад]
//        NUM_STATES
//    };

//    enum class ControlIndex : int
//    {
//        V_CMD = 0,          // желаемая скорость [м/с]
//        W_CMD,              // желаемая угловая скорость корпуса [рад/с]
//        NUM_CONTROLS
//    };

//    enum class OutputIndex : int
//    {
//        POS_X = 0,
//        POS_Y,
//        PSI,
//        V,
//        STEER_ANGLE,
//        NUM_OUTPUTS
//    };

//    // Геометрия
//    float wheelbase = 0.89f;                // колёсная база [м]
//    float ackermann_compensation = 0.85f;  // 1.0 = без компенсации, <1.0 = уменьшить эффективную базу

//    // Ограничения по скорости
//    float vx_max = 20.0f;                   // макс. скорость вперёд [м/с]
//    float vx_min = -10.0f;                  // макс. скорость назад [м/с]

//    // Ограничения рулевого управления
//    float max_steer_angle = 0.436332f;      // макс. угол поворота колёс [рад] ≈ 25°
//    float max_steer_speed = 1.5f;          // макс. скорость поворота руля [рад/с]

//    // Коэффициенты времени
//    float kv = 0.1f;                      // коэффициент реакции скорости
//    float kv_steer = 1.0f;                // коэффициент реакции руля

//    // Порог скорости для обратной кинематики
//    float eps_v = 1.0f;                   // зона нечувствительности по скорости [м/с]

//    //Ограничение на срыв колёс
//    float max_centrifugal = 4.0f;      // макс. допустимое боковое ускорение (м/с²) ≈0.4g

//    // Минимальный угол поворота при трогании с места
//    float min_steer_at_start = 0.01f;      // минимальный угол для начала движения [рад]

//    AckermannDynamicsParams() = default;
//};

//// ============================================================================
//// Класс динамики Ackermann
//// ============================================================================
//class AckermannDynamics : public MPPI_internal::Dynamics<AckermannDynamics, AckermannDynamicsParams>
//{
//public:
//    using PARENT_CLASS = MPPI_internal::Dynamics<AckermannDynamics, AckermannDynamicsParams>;
//    using state_array   = typename PARENT_CLASS::state_array;
//    using control_array = typename PARENT_CLASS::control_array;
//    using output_array  = typename PARENT_CLASS::output_array;
//    using params_t      = AckermannDynamicsParams;

//    AckermannDynamics(cudaStream_t stream = 0);

//    std::string getDynamicsModelName() const override
//    {
//        return "Ackermann Dynamics";
//    }

//    // ------------------------------------------------------------------------
//    // Хост-версия computeDynamics
//    // ------------------------------------------------------------------------
//    void computeDynamics(const Eigen::Ref<const state_array>& state,
//                         const Eigen::Ref<const control_array>& control,
//                         Eigen::Ref<state_array> state_der);

//    // ------------------------------------------------------------------------
//    // Хост-версия computeKinematics (оставлена пустой для совместимости)
//    // ------------------------------------------------------------------------
//    void computeKinematics(const Eigen::Ref<const state_array>& state,
//                           Eigen::Ref<state_array> state_der);

//    // ------------------------------------------------------------------------
//    // Device-версия computeDynamics (GPU)
//    // ------------------------------------------------------------------------
//    __device__ void computeDynamics(float* state, float* control,
//                                    float* state_der, float* theta_s = nullptr);

//    // ------------------------------------------------------------------------
//    // Device-версия computeKinematics (GPU) - оставлена пустой
//    // ------------------------------------------------------------------------
//    __device__ void computeKinematics(float* state, float* state_der);

//    // ------------------------------------------------------------------------
//    // Принудительное применение ограничений к управлению
//    // ------------------------------------------------------------------------
//    void enforceConstraints(Eigen::Ref<state_array> state,
//                            Eigen::Ref<control_array> control);

//    __device__ void enforceConstraints(float* state, float* control);

//    // ------------------------------------------------------------------------
//    // Хост-версия updateState
//    // ------------------------------------------------------------------------

//    void updateState(const Eigen::Ref<const state_array>& state,
//                     Eigen::Ref<state_array> next_state,
//                     Eigen::Ref<state_array> state_der,
//                     const float dt)
//    {
//        next_state = state + state_der * dt;
//        next_state[S_INDEX(PSI)] = angle_utils::normalizeAngle(next_state[S_INDEX(PSI)]);
//        next_state[S_INDEX(STEER_ANGLE)] = fmaxf(-params_.max_steer_angle,
//                                                  fminf(params_.max_steer_angle,
//                                                        next_state[S_INDEX(STEER_ANGLE)]));
//    }

//    __device__ void updateState(float* state, float* next_state, float* state_der, const float dt)
//    {
//        for (int i = 0; i < STATE_DIM; i++) {
//            next_state[i] = state[i] + state_der[i] * dt;
//        }
//        next_state[S_INDEX(PSI)] = angle_utils::normalizeAngle(next_state[S_INDEX(PSI)]);
//        next_state[S_INDEX(STEER_ANGLE)] = fmaxf(-params_.max_steer_angle,
//                                                  fminf(params_.max_steer_angle,
//                                                        next_state[S_INDEX(STEER_ANGLE)]));
//    }

//    // ------------------------------------------------------------------------
//    // Интерполяция состояния
//    // ------------------------------------------------------------------------
//    state_array interpolateState(const Eigen::Ref<const state_array>& s1,
//                                 const Eigen::Ref<const state_array>& s2,
//                                 const float alpha) const;

//    // ------------------------------------------------------------------------
//    // Преобразование карты (ROS-совместимость)
//    // ------------------------------------------------------------------------
//    state_array stateFromMap(const std::map<std::string, float>& map);

//    // ------------------------------------------------------------------------
//    // Получение нулевого состояния
//    // ------------------------------------------------------------------------
//    state_array getZeroState() const
//    {
//        return state_array::Zero();
//    }

//    // ------------------------------------------------------------------------
//    // Вспомогательная функция для вычисления желаемого угла поворота
//    // ------------------------------------------------------------------------
//    inline __host__ __device__ float computeTargetSteer(float v, float w_cmd, const params_t& p) const
//    {
//        float target_steer = 0.0f;

//        float v_abs = fabsf(v);

//        if (v_abs > p.eps_v) {

//            target_steer = atan2f(w_cmd * p.wheelbase, v_abs);

//            // Ограничение по максимальному углу
//            target_steer = fmaxf(-p.max_steer_angle, fminf(p.max_steer_angle, target_steer));

//            float v2 = v_abs * v_abs;
//            float safe_tan = (p.max_centrifugal * p.wheelbase) / v2;
//            float safe_angle = std::atan(safe_tan);
//            target_steer = fmaxf(-safe_angle, fminf(safe_angle, target_steer));
//        }

//        if (v_abs < 3.0f) {
//            // Линейная интерполяция: 0 м/с → 5°, 3 м/с → полный угол (25°)
//            float max_steer_low_speed = 0.087f + (p.max_steer_angle - 0.087f) * (v_abs / 3.0f);
//            target_steer = fmaxf(-max_steer_low_speed, fminf(max_steer_low_speed, target_steer));
//        }

//        return target_steer;
//    }

//private:
//    // Общая реализация computeDynamics (используется и хостом, и устройством)
//    template<typename T_state, typename T_control, typename T_der>
//    __host__ __device__ inline void computeDynamicsImpl(
//        const T_state* state, const T_control* control, T_der* state_der, const params_t* p) const
//    {
//        constexpr int POS_X = static_cast<int>(params_t::StateIndex::POS_X);
//        constexpr int POS_Y = static_cast<int>(params_t::StateIndex::POS_Y);
//        constexpr int PSI   = static_cast<int>(params_t::StateIndex::PSI);
//        constexpr int V     = static_cast<int>(params_t::StateIndex::V);
//        constexpr int STEER = static_cast<int>(params_t::StateIndex::STEER_ANGLE);
//        constexpr int V_CMD = static_cast<int>(params_t::ControlIndex::V_CMD);
//        constexpr int W_CMD = static_cast<int>(params_t::ControlIndex::W_CMD);

//        // Нормализуем углы ПЕРЕД использованием (критически важно!)
//#ifdef __CUDA_ARCH__
//        float psi = angle_utils::normalizeAngle(state[PSI]);
//        float steer = angle_utils::normalizeAngle(state[STEER]);
//#else
//        float psi = angle_utils::normalizeAngle(state[PSI]);
//        float steer = angle_utils::normalizeAngle(state[STEER]);
//#endif

//        steer = fmaxf(-p->max_steer_angle, fminf(p->max_steer_angle, steer));

//        float v = state[V];

//        float v_cmd = control[V_CMD];
//        float w_cmd = control[W_CMD];

//        // Динамика скорости
//        float target_v = fmaxf(p->vx_min, fminf(p->vx_max, v_cmd));
//        state_der[V] = (target_v - v) * p->kv;

//        // Динамика рулевого управления
//        float target_steer = computeTargetSteer(v, w_cmd, *p);
//        float steer_rate = (target_steer - steer) * p->kv_steer;
//        steer_rate = fmaxf(-p->max_steer_speed, fminf(p->max_steer_speed, steer_rate));
//        state_der[STEER] = steer_rate;

//        // Кинематика
//#ifdef __CUDA_ARCH__
//        state_der[POS_X] = v * __cosf(psi);
//        state_der[POS_Y] = v * __sinf(psi);
//#else
//        state_der[POS_X] = v * cosf(psi);
//        state_der[POS_Y] = v * sinf(psi);
//#endif

//        float wz = 0.0f;
//        float v_abs = fabsf(v);

//        if (v_abs > p->eps_v) {
//            wz = (v / (p->wheelbase)) * tanf(steer);

//            if (v < 0) {
//                wz = -wz;
//            }
//        }

//        state_der[PSI] = wz;
//    }
//};

//// ============================================================================
//// Конструктор
//// ============================================================================
//inline AckermannDynamics::AckermannDynamics(cudaStream_t stream)
//    : PARENT_CLASS(stream)
//{
//    // Не используем разделяемую память для параметров (как в BicycleDynamics)
//    this->SHARED_MEM_REQUEST_GRD_BYTES = 0;
//    this->SHARED_MEM_REQUEST_BLK_BYTES = 0;
//}

//// ============================================================================
//// Хост-версии (Eigen)
//// ============================================================================
//inline void AckermannDynamics::computeKinematics(
//    const Eigen::Ref<const state_array>& state,
//    Eigen::Ref<state_array> state_der)
//{
//    // Кинематика включена в computeDynamics
//    (void)state; (void)state_der;
//}

//inline void AckermannDynamics::computeDynamics(
//    const Eigen::Ref<const state_array>& state,
//    const Eigen::Ref<const control_array>& control,
//    Eigen::Ref<state_array> state_der)
//{
//    computeDynamicsImpl(state.data(), control.data(), state_der.data(), &params_);
//}

//inline void AckermannDynamics::enforceConstraints(
//    Eigen::Ref<state_array> state,
//    Eigen::Ref<control_array> control)
//{
//    const params_t* p = &params_;

//    control[C_INDEX(V_CMD)] = fmaxf(p->vx_min, fminf(p->vx_max, control[C_INDEX(V_CMD)]));

//    float v = fabsf(state[S_INDEX(V)]);
//    float max_wz = (fmaxf(v, p->eps_v) / p->wheelbase) * tanf(p->max_steer_angle);
//    max_wz = fmaxf(max_wz, 0.1f);

//    control[C_INDEX(W_CMD)] = fmaxf(-max_wz, fminf(max_wz, control[C_INDEX(W_CMD)]));
//}

//// ============================================================================
//// Device-версии (GPU)
//// ============================================================================
//__device__ inline void AckermannDynamics::computeKinematics(
//    float* state, float* state_der)
//{
//    (void)state; (void)state_der;
//}

//__device__ inline void AckermannDynamics::computeDynamics(
//    float* state, float* control, float* state_der, float* theta_s)
//{
//    const params_t* p = &params_;
//    (void)theta_s;  // Не используем shared memory
//    computeDynamicsImpl(state, control, state_der, p);
//}

//__device__ inline void AckermannDynamics::enforceConstraints(
//    float* state, float* control)
//{
//    const params_t* p = &params_;

//    control[C_INDEX(V_CMD)] = fmaxf(p->vx_min, fminf(p->vx_max, control[C_INDEX(V_CMD)]));

//    float v = fabsf(state[S_INDEX(V)]);
//    float max_wz = (fmaxf(v, p->eps_v) / p->wheelbase) * tanf(p->max_steer_angle);
//    max_wz = fmaxf(max_wz, 0.1f);

//    control[C_INDEX(W_CMD)] = fmaxf(-max_wz, fminf(max_wz, control[C_INDEX(W_CMD)]));
//}

//// ============================================================================
//// Интерполяция состояния
//// ============================================================================
//inline AckermannDynamics::state_array AckermannDynamics::interpolateState(
//    const Eigen::Ref<const state_array>& s1,
//    const Eigen::Ref<const state_array>& s2,
//    const float alpha) const
//{
//    state_array result = s1 + alpha * (s2 - s1);
//    result[S_INDEX(PSI)] = angle_utils::normalizeAngle(result[S_INDEX(PSI)]);
//    result[S_INDEX(STEER_ANGLE)] = angle_utils::normalizeAngle(result[S_INDEX(STEER_ANGLE)]);
//    return result;
//}

//// ============================================================================
//// Преобразование карты
//// ============================================================================
//inline AckermannDynamics::state_array AckermannDynamics::stateFromMap(
//    const std::map<std::string, float>& map)
//{
//    state_array s = state_array::Zero();
//    s[S_INDEX(POS_X)] = map.at("POS_X");
//    s[S_INDEX(POS_Y)] = map.at("POS_Y");
//    s[S_INDEX(PSI)] = map.at("YAW");
//    s[S_INDEX(V)] = map.at("VEL_X");
//    s[S_INDEX(STEER_ANGLE)] = map.at("STEER_ANGLE");
//    return s;
//}

//#endif // ACKERMANNDYNAMICS_H

#ifndef ACKERMANNDYNAMICS_H
#define ACKERMANNDYNAMICS_H

#include <mppi/dynamics/dynamics.h>
#include <mppi/utils/angle_utils.h>
#include <Eigen/Dense>
#include <map>
#include <string>
#include <cmath>

// ============================================================================
// Параметры динамики Ackermann (улучшенная версия, синхронизированная с симуляцией)
// ============================================================================
struct AckermannDynamicsParams : public DynamicsParams
{
    enum class StateIndex : int
    {
        POS_X = 0,          // позиция X [м]
        POS_Y,              // позиция Y [м]
        PSI,                // угол рысканья (курс) [рад]
        V,                  // продольная скорость [м/с]
        STEER_ANGLE,        // фактический угол поворота колёс [рад]
        NUM_STATES
    };

    enum class ControlIndex : int
    {
        V_CMD = 0,          // желаемая скорость [м/с]
        W_CMD,              // желаемая угловая скорость корпуса [рад/с]
        NUM_CONTROLS
    };

    enum class OutputIndex : int
    {
        POS_X = 0,
        POS_Y,
        PSI,
        V,
        STEER_ANGLE,
        NUM_OUTPUTS
    };

    // ========================================================================
    // Геометрия (из simulation.h car_config)
    // ========================================================================
    float wheelbase = 7.107f;               // колёсная база [м] (из simulation.h)
    float lf = 3.55204f;                    // расстояние от ЦМ до передней оси [м] (из wheelOffsets[0].z())
    float lr = 3.55538f;                    // расстояние от ЦМ до задней оси [м] (из wheelOffsets[2].z())

    // ========================================================================
    // Масса и инерция (из simulation.h car_config)
    // ========================================================================
    float mass = 2200.0f;                   // масса [кг]

    // ========================================================================
    // Ограничения по скорости
    // ========================================================================
    float vx_max = 20.0f;                   // макс. скорость вперёд [м/с]
    float vx_min = -10.0f;                  // макс. скорость назад [м/с]

    // ========================================================================
    // Ограничения рулевого управления (из simulation.h car_config)
    // ========================================================================
    float max_steer_angle = 0.523599f;      // макс. угол поворота колёс [рад] ≈ 25° (max_sterr_rad)
    float max_steer_speed = 2.0f;           // макс. скорость поворота руля [рад/с] (max_sterr_speed)

    // ========================================================================
    // Коэффициенты времени реакции (отражают динамику CarController)
    // ========================================================================
    float kv = 0.1f;                        // коэффициент реакции скорости
    float kv_steer = 1.0f;                  // коэффициент реакции руля

    // ========================================================================
    // Порог скорости для обратной кинематики
    // ========================================================================
    float eps_v = 1.0f;                     // зона нечувствительности по скорости [м/с]

    // ========================================================================
    // Модель шин
    // ========================================================================
    float mu = 4.0f;                        // коэффициент трения

    // ========================================================================
    // Модель двигателя и сопротивления (из simulation.h car_config)
    // ========================================================================
    float throttle_force = 8000.0f;         // максимальная сила тяги [Н] (throttle)
    float rolling_resistance = 50.0f;       // сопротивление качению [Н] (rolling_resistance)

    // Аэродинамика: 0.5 * ρ * Cd * A (из simulation.h)
    // ρ = 1.225 кг/м³, Cd*A = 0.35 * 2.2 = 0.77 м²
    float aero_coeff = 0.471625f;           // 0.5 * 1.225 * 0.35 * 2.2 [кг/м]

    // ========================================================================
    // Ограничение центробежного ускорения (для безопасности)
    // ========================================================================
    float max_centrifugal = 4.0f;           // макс. допустимое боковое ускорение [м/с²] ≈0.4g

    // ========================================================================
    // Ограничения ускорения (из физики двигателя и тормозов)
    // ========================================================================
    float max_accel = 9.636f;               // 8000Н / 2200кг ≈ 3.64 м/с²
    float max_brake = 11.364f;              // 25000Н / 2200кг ≈ 11.36 м/с² (brake force)

    AckermannDynamicsParams() = default;
};

// ============================================================================
// Класс динамики Ackermann (улучшенная версия с моделью шин из симуляции)
// ============================================================================
class AckermannDynamics : public MPPI_internal::Dynamics<AckermannDynamics, AckermannDynamicsParams>
{
public:
    using PARENT_CLASS = MPPI_internal::Dynamics<AckermannDynamics, AckermannDynamicsParams>;
    using state_array   = typename PARENT_CLASS::state_array;
    using control_array = typename PARENT_CLASS::control_array;
    using output_array  = typename PARENT_CLASS::output_array;
    using params_t      = AckermannDynamicsParams;

    AckermannDynamics(cudaStream_t stream = 0);

    std::string getDynamicsModelName() const override
    {
        return "Ackermann Dynamics (Simulation-Synced Tire Model)";
    }

    // ------------------------------------------------------------------------
    // Хост-версия computeDynamics
    // ------------------------------------------------------------------------
    void computeDynamics(const Eigen::Ref<const state_array>& state,
                         const Eigen::Ref<const control_array>& control,
                         Eigen::Ref<state_array> state_der);

    // ------------------------------------------------------------------------
    // Хост-версия computeKinematics (оставлена пустой для совместимости)
    // ------------------------------------------------------------------------
    void computeKinematics(const Eigen::Ref<const state_array>& state,
                           Eigen::Ref<state_array> state_der);

    // ------------------------------------------------------------------------
    // Device-версия computeDynamics (GPU)
    // ------------------------------------------------------------------------
    __device__ void computeDynamics(float* state, float* control,
                                    float* state_der, float* theta_s = nullptr);

    // ------------------------------------------------------------------------
    // Device-версия computeKinematics (GPU) - оставлена пустой
    // ------------------------------------------------------------------------
    __device__ void computeKinematics(float* state, float* state_der);

    // ------------------------------------------------------------------------
    // Принудительное применение ограничений к управлению
    // ------------------------------------------------------------------------
    void enforceConstraints(Eigen::Ref<state_array> state,
                            Eigen::Ref<control_array> control);

    __device__ void enforceConstraints(float* state, float* control);

    // ------------------------------------------------------------------------
    // Обновление состояния
    // ------------------------------------------------------------------------
    void updateState(const Eigen::Ref<const state_array>& state,
                     Eigen::Ref<state_array> next_state,
                     Eigen::Ref<state_array> state_der,
                     const float dt)
    {
        next_state = state + state_der * dt;
        next_state[S_INDEX(PSI)] = angle_utils::normalizeAngle(next_state[S_INDEX(PSI)]);
        next_state[S_INDEX(STEER_ANGLE)] = fmaxf(-params_.max_steer_angle,
                                                  fminf(params_.max_steer_angle,
                                                        next_state[S_INDEX(STEER_ANGLE)]));
    }

    __device__ void updateState(float* state, float* next_state, float* state_der, const float dt)
    {
        for (int i = 0; i < STATE_DIM; i++) {
            next_state[i] = state[i] + state_der[i] * dt;
        }
        next_state[S_INDEX(PSI)] = angle_utils::normalizeAngle(next_state[S_INDEX(PSI)]);
        next_state[S_INDEX(STEER_ANGLE)] = fmaxf(-params_.max_steer_angle,
                                                  fminf(params_.max_steer_angle,
                                                        next_state[S_INDEX(STEER_ANGLE)]));
    }

    // ------------------------------------------------------------------------
    // Интерполяция состояния
    // ------------------------------------------------------------------------
    state_array interpolateState(const Eigen::Ref<const state_array>& s1,
                                 const Eigen::Ref<const state_array>& s2,
                                 const float alpha) const;

    // ------------------------------------------------------------------------
    // Преобразование карты (ROS-совместимость)
    // ------------------------------------------------------------------------
    state_array stateFromMap(const std::map<std::string, float>& map);

    // ------------------------------------------------------------------------
    // Получение нулевого состояния
    // ------------------------------------------------------------------------
    state_array getZeroState() const
    {
        return state_array::Zero();
    }

    // ------------------------------------------------------------------------
    // Вспомогательная функция: вычисление целевого угла поворота
    // (с ограничениями, идентичными CarController)
    // ------------------------------------------------------------------------
    inline __host__ __device__ float computeTargetSteer(float v, float w_cmd, const params_t& p) const
    {
        float target_steer = 0.0f;
        float v_abs = fabsf(v);

        if (v_abs > p.eps_v) {
            // Кинематический feedforward
            target_steer = atan2f(w_cmd * p.wheelbase, v_abs);
            if (v < 0) target_steer = -target_steer;

            // Ограничение по максимальному углу
            target_steer = fmaxf(-p.max_steer_angle, fminf(p.max_steer_angle, target_steer));

            // Ограничение по центробежной силе
            float v2 = v_abs * v_abs;
            float safe_tan = (p.max_centrifugal * p.wheelbase) / v2;
            float safe_angle = atanf(safe_tan);
            target_steer = fmaxf(-safe_angle, fminf(safe_angle, target_steer));
        }

        return target_steer;
    }

private:
    // Общая реализация computeDynamics (используется и хостом, и устройством)
    template<typename T_state, typename T_control, typename T_der>
    __host__ __device__ inline void computeDynamicsImpl(
        const T_state* state, const T_control* control, T_der* state_der, const params_t* p) const
    {
        // Индексы состояний
        constexpr int POS_X = static_cast<int>(params_t::StateIndex::POS_X);
        constexpr int POS_Y = static_cast<int>(params_t::StateIndex::POS_Y);
        constexpr int PSI   = static_cast<int>(params_t::StateIndex::PSI);
        constexpr int V     = static_cast<int>(params_t::StateIndex::V);
        constexpr int STEER = static_cast<int>(params_t::StateIndex::STEER_ANGLE);
        constexpr int V_CMD = static_cast<int>(params_t::ControlIndex::V_CMD);
        constexpr int W_CMD = static_cast<int>(params_t::ControlIndex::W_CMD);

        // Нормализация углов
#ifdef __CUDA_ARCH__
        float psi = angle_utils::normalizeAngle(state[PSI]);
        float steer = angle_utils::normalizeAngle(state[STEER]);
#else
        float psi = angle_utils::normalizeAngle(state[PSI]);
        float steer = angle_utils::normalizeAngle(state[STEER]);
#endif
        steer = fmaxf(-p->max_steer_angle, fminf(p->max_steer_angle, steer));

        // Текущие значения
        float v = state[V];
        float v_cmd = control[V_CMD];
        float w_cmd = control[W_CMD];

        // ====================================================================
        // ШАГ 1: Ограничение w_cmd (идентично CarController)
        // ====================================================================
        float v_abs = fabsf(v);
        float max_wz;

        if (v_abs > p->eps_v) {
            // Максимальная угловая скорость из кинематики руля
            max_wz = (v_abs / p->wheelbase) * tanf(p->max_steer_angle);

            // Дополнительное ограничение по центробежной силе
            if (v_abs > 1.0f) {
                float v2 = v_abs * v_abs;
                float safe_tan = (p->max_centrifugal * p->wheelbase) / v2;
                float safe_steer = fminf(p->max_steer_angle, atanf(safe_tan));
                float safe_wz = (v_abs / p->wheelbase) * tanf(safe_steer);
                max_wz = fminf(max_wz, safe_wz);
            }
        } else {
            max_wz = 0.1f;
        }
        max_wz = fmaxf(max_wz, 0.1f);

        // Применяем ограничение
        float w_cmd_limited = fmaxf(-max_wz, fminf(max_wz, w_cmd));

        // ====================================================================
        // ШАГ 2: Улучшенная динамика продольной скорости
        // ====================================================================

        // Ограничение целевой скорости
        float target_v = fmaxf(p->vx_min, fminf(p->vx_max, v_cmd));

        // Базовая динамика первого порядка (реакция CarController)
        float speed_error = target_v - v;
        float base_accel = speed_error * p->kv;

        // Модель тяги двигателя (из simulation.h: throttle_force * ctrl.throttle)
        // ctrl.throttle = 1.0 при полном газе
        float engine_force = 0.0f;
        if (fabsf(v_cmd) > 0.01f) {
            // Пропорционально команде скорости (0..1 от vx_max)
            float throttle_ratio = fminf(fabsf(v_cmd) / p->vx_max, 1.0f);
            engine_force = p->throttle_force * throttle_ratio;
            if (v_cmd < 0) engine_force = -engine_force;
        }

        // Сопротивление качению (из simulation.h: rolling_resistance * sign(omega))
        float rolling_force = p->rolling_resistance * tanhf(v * 5.0f);

        // Аэродинамическое сопротивление (из simulation.h: 0.5 * ρ * Cd * A * v²)
        float aero_force = p->aero_coeff * v * fabsf(v);

        // Суммарное продольное ускорение
        float total_accel = base_accel + (engine_force - rolling_force - aero_force) / p->mass;


        // =============================================================================
        // ШАГ 3: Эффект плуга (поворот колёс создаёт сопративление) через КРУГ КАММА
        // =============================================================================
        if (v_abs > 0.5f) {
            float Fx_demand = fabsf(total_accel) * p->mass;
            float F_max_total = p->mass * 9.81f * p->mu;

            // Гладкая мера "заполненности" круга трения
            float Fy_demand = p->mass * v_abs * fabsf(w_cmd_limited) + 0.1f;
            float usage = (Fx_demand * Fx_demand + Fy_demand * Fy_demand) / (F_max_total * F_max_total + 1.0f);

            // Сигмоида: плавно падает от 1 до 0 когда usage > 1
            float smooth_scale = 1.0f / (1.0f + usage * usage);

            // Применяем ко всему
            total_accel *= smooth_scale;

            float Fy_available = F_max_total * smooth_scale;
            float max_w_kamm = Fy_available / (p->mass * v_abs + 0.1f);
            w_cmd_limited = fmaxf(-max_w_kamm, fminf(max_w_kamm, w_cmd_limited));
        }

        // Ограничение ускорения (из физических пределов)
        total_accel = fmaxf(-p->max_brake, fminf(p->max_accel, total_accel));

        state_der[V] = total_accel;

        // ====================================================================
        // ШАГ 4: Динамика рулевого управления
        // ====================================================================
        float target_steer = computeTargetSteer(v, w_cmd_limited, *p);
        float steer_rate = (target_steer - steer) * p->kv_steer;

        // Ограничение скорости руля (max_sterr_speed из simulation.h)
        steer_rate = fmaxf(-p->max_steer_speed, fminf(p->max_steer_speed, steer_rate));
        state_der[STEER] = steer_rate;

        // ====================================================================
        // ШАГ 5: Кинематика
        // ====================================================================
#ifdef __CUDA_ARCH__
        state_der[POS_X] = v * __cosf(psi);
        state_der[POS_Y] = v * __sinf(psi);
#else
        state_der[POS_X] = v * cosf(psi);
        state_der[POS_Y] = v * sinf(psi);
#endif

        // Угловая скорость из кинематики Аккермана
        float wz = 0.0f;
        if (v_abs > p->eps_v) {
            wz = (v / p->wheelbase) * tanf(steer);
            if (v < 0) wz = -wz;
        }
        state_der[PSI] = wz;
    }
};

// ============================================================================
// Конструктор
// ============================================================================
inline AckermannDynamics::AckermannDynamics(cudaStream_t stream)
    : PARENT_CLASS(stream)
{
    this->SHARED_MEM_REQUEST_GRD_BYTES = 0;
    this->SHARED_MEM_REQUEST_BLK_BYTES = 0;
}

// ============================================================================
// Хост-версии (Eigen)
// ============================================================================
inline void AckermannDynamics::computeKinematics(
    const Eigen::Ref<const state_array>& state,
    Eigen::Ref<state_array> state_der)
{
    (void)state; (void)state_der;
}

inline void AckermannDynamics::computeDynamics(
    const Eigen::Ref<const state_array>& state,
    const Eigen::Ref<const control_array>& control,
    Eigen::Ref<state_array> state_der)
{
    computeDynamicsImpl(state.data(), control.data(), state_der.data(), &params_);
}

inline void AckermannDynamics::enforceConstraints(
    Eigen::Ref<state_array> state,
    Eigen::Ref<control_array> control)
{
    const params_t* p = &params_;

    // Ограничение скорости
    control[C_INDEX(V_CMD)] = fmaxf(p->vx_min, fminf(p->vx_max, control[C_INDEX(V_CMD)]));

    // Ограничение угловой скорости (как в CarController)
    float v = fabsf(state[S_INDEX(V)]);
    float max_wz = (fmaxf(v, p->eps_v) / p->wheelbase) * tanf(p->max_steer_angle);
    max_wz = fmaxf(max_wz, 0.1f);

    // Дополнительное ограничение по центробежной силе
    if (v > 1.0f) {
        float v2 = v * v;
        float safe_tan = (p->max_centrifugal * p->wheelbase) / v2;
        float safe_wz = (v / p->wheelbase) * safe_tan;
        max_wz = fminf(max_wz, safe_wz);
    }

    control[C_INDEX(W_CMD)] = fmaxf(-max_wz, fminf(max_wz, control[C_INDEX(W_CMD)]));
}

// ============================================================================
// Device-версии (GPU)
// ============================================================================
__device__ inline void AckermannDynamics::computeKinematics(
    float* state, float* state_der)
{
    (void)state; (void)state_der;
}

__device__ inline void AckermannDynamics::computeDynamics(
    float* state, float* control, float* state_der, float* theta_s)
{
    const params_t* p = &params_;
    (void)theta_s;
    computeDynamicsImpl(state, control, state_der, p);
}

__device__ inline void AckermannDynamics::enforceConstraints(
    float* state, float* control)
{
    const params_t* p = &params_;

    control[C_INDEX(V_CMD)] = fmaxf(p->vx_min, fminf(p->vx_max, control[C_INDEX(V_CMD)]));

    float v = fabsf(state[S_INDEX(V)]);
    float max_wz = (fmaxf(v, p->eps_v) / p->wheelbase) * tanf(p->max_steer_angle);
    max_wz = fmaxf(max_wz, 0.1f);

    if (v > 1.0f) {
        float v2 = v * v;
        float safe_tan = (p->max_centrifugal * p->wheelbase) / v2;
        float safe_wz = (v / p->wheelbase) * safe_tan;
        max_wz = fminf(max_wz, safe_wz);
    }

    control[C_INDEX(W_CMD)] = fmaxf(-max_wz, fminf(max_wz, control[C_INDEX(W_CMD)]));
}

// ============================================================================
// Интерполяция состояния
// ============================================================================
inline AckermannDynamics::state_array AckermannDynamics::interpolateState(
    const Eigen::Ref<const state_array>& s1,
    const Eigen::Ref<const state_array>& s2,
    const float alpha) const
{
    state_array result = s1 + alpha * (s2 - s1);
    result[S_INDEX(PSI)] = angle_utils::normalizeAngle(result[S_INDEX(PSI)]);
    result[S_INDEX(STEER_ANGLE)] = angle_utils::normalizeAngle(result[S_INDEX(STEER_ANGLE)]);
    return result;
}

// ============================================================================
// Преобразование карты
// ============================================================================
inline AckermannDynamics::state_array AckermannDynamics::stateFromMap(
    const std::map<std::string, float>& map)
{
    state_array s = state_array::Zero();
    s[S_INDEX(POS_X)] = map.at("POS_X");
    s[S_INDEX(POS_Y)] = map.at("POS_Y");
    s[S_INDEX(PSI)] = map.at("YAW");
    s[S_INDEX(V)] = map.at("VEL_X");
    s[S_INDEX(STEER_ANGLE)] = map.at("STEER_ANGLE");
    return s;
}

#endif // ACKERMANNDYNAMICS_H
