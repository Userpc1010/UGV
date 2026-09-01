#ifndef VECTOR_PURSUIT_CONTROLLER_HPP
#define VECTOR_PURSUIT_CONTROLLER_HPP

#include "datatypes_smac_planner.h"
#include "vector_pursuit_controller_collision_checker.hpp"
#include <Eigen/Dense>
#include <vector>
#include <cstdint>
#include <cmath>
#include <limits>
#include <algorithm>

namespace vpc
{

struct VPCOutput
{
    float linear_vel = 0.0f;
    float angular_vel = 0.0f;
    float steering_angle = 0.0f;
    bool is_rotating_to_heading = false;
};

struct VPCParams
{
    // Vector Pursuit tuning
    float k = 8.0f;                          // коэффициент VPC (обычно 8-10)

    // Скорости
    float desired_linear_vel = 2.0f;
    float min_linear_vel = 0.1f;
    float min_approach_linear_velocity = 0.1f;
    float approach_velocity_scaling_dist = 5.0f;

    // Lookahead
    float lookahead_dist = 3.0f;
    float min_lookahead_dist = 1.0f;
    float max_lookahead_dist = 10.0f;
    float lookahead_time = 3.5f;
    bool use_velocity_scaled_lookahead_dist = true;
    bool use_interpolation = true;
    bool use_heading_from_path = true;

    // Ограничения
    float min_turning_radius = 2.3f;
    float max_lateral_accel = 6.0f;
    float max_linear_accel = 9.0f;
    float max_angular_accel = 1.3f;

    // Поворот к курсу
    bool use_rotate_to_heading = false;
    float rotate_to_heading_angular_vel = 0.25f;
    float rotate_to_heading_min_angle = 0.785f;

    // Коллизии
    bool use_collision_detection = true;
    float max_allowed_time_to_collision_up_to_target = 1.0f;

    // Cost scaling
    bool use_cost_regulated_linear_velocity_scaling = false;
    float cost_scaling_dist = 2.0f;
    float cost_scaling_gain = 1.0f;
    float inflation_cost_scaling_factor = 3.0f;

    // Реверс
    bool allow_reversing = true;

    // Ackermann
    float wheelbase = 0.7f;

    // Goal tolerance
    float goal_dist_tol = 0.15f;

    // Control duration
    float control_duration = 0.1f;
};

class VectorPursuitController
{
public:
    VectorPursuitController();
    ~VectorPursuitController() = default;

    void configure(
        const VPCParams& params,
        const uint8_t* costmap_data = nullptr,
        int costmap_width = 0, int costmap_height = 0,
        float costmap_resolution = 1.0f,
        float costmap_origin_x = 0.0f,
        float costmap_origin_y = 0.0f);

    void setPath(const Path& path);

    VPCOutput computeVelocityCommands(
        float robot_x, float robot_y, float robot_yaw,
        float current_linear_vel, float current_angular_vel,
        float dt);

    void updateCostmap(
        const uint8_t* costmap_data,
        int width, int height,
        float resolution,
        float origin_x, float origin_y);

    void reset();
    void getLookaheadPoint(float& lx, float& ly) const {
        lx = m_lookaheadX; ly = m_lookaheadY;
    }

    // ✅ ДИАГНОСТИКА
        const Path& getCurrentPath() const { return m_path; }
        const Path& getTransformedPath() const { return m_transformedPath; }
        float getLookaheadDist() const { return m_lookaheadDist; }
        float getLookaheadHeading() const { return m_lookaheadHeading; }
        float getTurningRadius() const { return m_lastTurningRadius; }
        float getCurvature() const { return m_lastCurvature; }
        float getSign() const { return m_lastSign; }
        int getPathSize() const { return static_cast<int>(m_path.poses.size()); }
        int getTransformedPathSize() const { return static_cast<int>(m_transformedPath.poses.size()); }

private:
    VPCParams m_params;
    vpc::CollisionChecker m_collisionChecker;
    Path m_path;
    Path m_transformedPath;

    float m_lookaheadX = 0.0f;
    float m_lookaheadY = 0.0f;
    float m_lookaheadHeading = 0.0f;

    float m_lastLinearVel = 0.0f;
    float m_lastAngularVel = 0.0f;
    bool m_isRotatingToHeading = false;

    // ✅ ДИАГНОСТИКА
    float m_lookaheadDist = 0.0f;
    float m_lastTurningRadius = 0.0f;
    float m_lastCurvature = 0.0f;
    float m_lastSign = 1.0f;

    // Вспомогательные методы
    void transformPathToRobotFrame(float robot_x, float robot_y, float robot_yaw);
    float getLookAheadDistance(float current_speed);
    bool findLookAheadPoint(float lookahead_dist, float& point_x, float& point_y, float& heading);
    float calculateTurningRadius(float lx, float ly, float heading) const;

    // Ограничения скорости
    float curvatureConstraint(float raw_linear_vel, float curvature) const;
    float costConstraint(float raw_linear_vel, float pose_cost) const;
    float approachVelocityConstraint(float constrained_linear_vel) const;
    void applyConstraints(float curvature, float pose_cost, float& linear_vel, float& sign);

    // Поворот к курсу
    bool shouldRotateToPath(float& angle_to_path, float x_vel_sign);
    bool shouldRotateToGoalHeading(float robot_x, float robot_y, float goal_yaw);
    void rotateToHeading(float& linear_vel, float& angular_vel,
                         float angle_to_path, float current_angular_vel, float dt);

    // Cusp detection
    float getCuspDist(const Path& transformed_plan) const;

    // Ориентация
    float getOrientation(float x1, float y1, float x2, float y2) const;
    float normalizeAngle(float angle) const;
    float normalizeAnglePositive(float angle) const;
};

}  // namespace vpc

#endif  // VECTOR_PURSUIT_CONTROLLER_HPP
