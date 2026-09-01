#include "vector_pursuit_controller.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>

namespace vpc
{

VectorPursuitController::VectorPursuitController()
{
}

void VectorPursuitController::configure(
    const VPCParams& params,
    const uint8_t* costmap_data,
    int costmap_width, int costmap_height,
    float costmap_resolution,
    float costmap_origin_x, float costmap_origin_y)
{
    m_params = params;

    if (costmap_data && costmap_width > 0 && costmap_height > 0) {
        updateCostmap(costmap_data, costmap_width, costmap_height,
                      costmap_resolution, costmap_origin_x, costmap_origin_y);
    }
}

void VectorPursuitController::updateCostmap(
    const uint8_t* costmap_data,
    int width, int height,
    float resolution,
    float origin_x, float origin_y)
{
    // Используем тот же CollisionChecker, что и RPP
    m_collisionChecker.configure(
        costmap_data, width, height, resolution,
        origin_x, origin_y,
        3.0f,   // inscribed_radius
        4.2f);  // circumscribed_radius
}

void VectorPursuitController::setPath(const Path& path) {

    m_path = path;
}

void VectorPursuitController::reset()
{
    m_lastLinearVel = 0.0f;
    m_lastAngularVel = 0.0f;
    m_isRotatingToHeading = false;
}

void VectorPursuitController::transformPathToRobotFrame(
    float robot_x, float robot_y, float robot_yaw)
{
    m_transformedPath.poses.clear();

    if (m_path.poses.empty()) return;

    // 1. Найти ближайшую точку к роботу
    int closest_idx = 0;
    float min_dist = std::numeric_limits<float>::max();

    for (size_t i = 0; i < m_path.poses.size(); ++i) {
        float dx = m_path.poses[i].position.x() - robot_x;
        float dy = m_path.poses[i].position.y() - robot_y;
        float dist = std::hypot(dx, dy);
        if (dist < min_dist) {
            min_dist = dist;
            closest_idx = static_cast<int>(i);
        }
    }

    // 2. УДАЛЯЕМ пройденные точки из m_path (как в оригинале)
    if (closest_idx > 0) {
        m_path.poses.erase(m_path.poses.begin(), m_path.poses.begin() + closest_idx);
    }

    // 3. Трансформируем все оставшиеся точки
    float cos_yaw = std::cos(-robot_yaw);
    float sin_yaw = std::sin(-robot_yaw);

    for (const auto& pose : m_path.poses) {
        float dx = pose.position.x() - robot_x;
        float dy = pose.position.y() - robot_y;

        float x_local = dx * cos_yaw - dy * sin_yaw;
        float y_local = dx * sin_yaw + dy * cos_yaw;

        Pose transformed;
        transformed.position.x() = x_local;
        transformed.position.y() = y_local;
        transformed.orientation = pose.orientation - robot_yaw;

        m_transformedPath.poses.push_back(transformed);
    }
}


float VectorPursuitController::getLookAheadDistance(float current_speed)
{
    float lookahead_dist = m_params.lookahead_dist;

    if (m_params.use_velocity_scaled_lookahead_dist) {
        lookahead_dist = std::fabs(current_speed) * m_params.lookahead_time;
        lookahead_dist = std::clamp(lookahead_dist,
                                    m_params.min_lookahead_dist,
                                    m_params.max_lookahead_dist);
    }

    return lookahead_dist;
}

float VectorPursuitController::normalizeAngle(float angle) const
{
    while (angle > M_PI) angle -= 2.0f * M_PI;
    while (angle < -M_PI) angle += 2.0f * M_PI;
    return angle;
}

float VectorPursuitController::normalizeAnglePositive(float angle) const
{
    while (angle < 0.0f) angle += 2.0f * M_PI;
    while (angle >= 2.0f * M_PI) angle -= 2.0f * M_PI;
    return angle;
}

float VectorPursuitController::getOrientation(float x1, float y1, float x2, float y2) const
{
    return std::atan2(y2 - y1, x2 - x1);
}

bool VectorPursuitController::findLookAheadPoint(
    float lookahead_dist, float& point_x, float& point_y, float& heading)
{
    if (m_transformedPath.poses.empty()) return false;

    // Найти первую точку, которая дальше lookahead_dist
    auto goal_pose_it = std::find_if(
        m_transformedPath.poses.begin(), m_transformedPath.poses.end(),
        [&](const Pose& ps) {
            return std::hypot(ps.position.x(), ps.position.y()) >= lookahead_dist;
        });

    point_x = 0.0f;
    point_y = 0.0f;
    heading = 0.0f;

    // Если все точки ближе, берём последнюю
    if (goal_pose_it == m_transformedPath.poses.end()) {
        const auto& last = m_transformedPath.poses.back();
        point_x = last.position.x();
        point_y = last.position.y();
        heading = last.orientation;
        return true;
    }

    // Если первая точка уже дальше, берём её
    if (goal_pose_it == m_transformedPath.poses.begin()) {
        point_x = goal_pose_it->position.x();
        point_y = goal_pose_it->position.y();
        heading = goal_pose_it->orientation;
        return true;
    }

    // Иначе интерполируем между prev и goal
    auto prev_pose_it = std::prev(goal_pose_it);

    if (m_params.use_interpolation) {
        // Circle-segment intersection (как в оригинальном VPC)
        float x1 = prev_pose_it->position.x();
        float y1 = prev_pose_it->position.y();
        float x2 = goal_pose_it->position.x();
        float y2 = goal_pose_it->position.y();

        float dx = x2 - x1;
        float dy = y2 - y1;
        float dr2 = dx * dx + dy * dy;
        float D = x1 * y2 - x2 * y1;

        float d1 = x1 * x1 + y1 * y1;
        float d2 = x2 * x2 + y2 * y2;
        float dd = d2 - d1;

        float sqrt_term = std::sqrt(lookahead_dist * lookahead_dist * dr2 - D * D);
        point_x = (D * dy + std::copysign(1.0f, dd) * dx * sqrt_term) / dr2;
        point_y = (-D * dx + std::copysign(1.0f, dd) * dy * sqrt_term) / dr2;

        // Интерполяция heading
        if (m_params.use_heading_from_path) {
            float goal_yaw = normalizeAnglePositive(goal_pose_it->orientation);
            float prev_yaw = normalizeAnglePositive(prev_pose_it->orientation);
            heading = normalizeAngle((goal_yaw + prev_yaw) / 2.0f);
        } else {
            heading = getOrientation(x1, y1, x2, y2);
        }
    } else {
        point_x = goal_pose_it->position.x();
        point_y = goal_pose_it->position.y();

        if (m_params.use_heading_from_path) {
            heading = goal_pose_it->orientation;
        } else {
            heading = getOrientation(prev_pose_it->position.x(), prev_pose_it->position.y(),
                                     goal_pose_it->position.x(), goal_pose_it->position.y());
        }
    }

    return true;
}

float VectorPursuitController::calculateTurningRadius(
    float lx, float ly, float heading) const
{
    float distance = std::hypot(lx, ly);
    float turning_radius;

    if (m_params.allow_reversing || lx >= 0.0f) {
        if (std::fabs(ly) > 1e-6f) {
            float phi_1 = std::atan2(
                (2.0f * ly * ly - distance * distance),
                (2.0f * lx * ly));
            float term_2 = distance * distance / (2.0f * ly);
            float phi_2 = std::atan2(term_2, 0.0f);
            float phi = normalizeAnglePositive(phi_1 - phi_2);
            phi = std::max(phi, 1.0e-9f);

            float term_1 = (m_params.k * phi) / (((m_params.k - 1.0f) * phi) + heading);
            turning_radius = std::fabs(term_1 * term_2);
        } else {
            turning_radius = std::numeric_limits<float>::max();
        }
    } else {
        turning_radius = m_params.min_turning_radius;
    }

    turning_radius = std::max(turning_radius, m_params.min_turning_radius);
    return turning_radius;
}

float VectorPursuitController::curvatureConstraint(
    float raw_linear_vel, float curvature) const
{
    if (std::fabs(curvature) < 1e-6f) return raw_linear_vel;

    // Ограничение по боковому ускорению
    float max_vel_for_curve = std::sqrt(m_params.max_lateral_accel / std::fabs(curvature));
    return std::min(raw_linear_vel, max_vel_for_curve);
}

float VectorPursuitController::costConstraint(
    float raw_linear_vel, float pose_cost) const
{
    if (pose_cost <= 0.0f || pose_cost >= 254.0f) return raw_linear_vel;

    const float inscribed_radius = 3.0f;
    float min_dist = (m_params.inflation_cost_scaling_factor * inscribed_radius -
                      std::log(pose_cost) + std::log(253.0f)) /
                     m_params.inflation_cost_scaling_factor;

    if (min_dist < m_params.cost_scaling_dist) {
        return raw_linear_vel *
               (m_params.cost_scaling_gain * min_dist / m_params.cost_scaling_dist);
    }

    return raw_linear_vel;
}

float VectorPursuitController::approachVelocityConstraint(
    float constrained_linear_vel) const
{
    if (m_transformedPath.poses.empty()) return constrained_linear_vel;

    float remaining_dist = 0.0f;
    for (size_t i = 1; i < m_transformedPath.poses.size(); ++i) {
        float dx = m_transformedPath.poses[i].position.x() -
                   m_transformedPath.poses[i - 1].position.x();
        float dy = m_transformedPath.poses[i].position.y() -
                   m_transformedPath.poses[i - 1].position.y();
        remaining_dist += std::hypot(dx, dy);
    }

    if (remaining_dist < m_params.approach_velocity_scaling_dist) {
        float dist_to_goal = std::hypot(m_transformedPath.poses.back().position.x(),
                                        m_transformedPath.poses.back().position.y());
        float velocity_scaling = dist_to_goal / m_params.approach_velocity_scaling_dist;
        float approach_vel = constrained_linear_vel * velocity_scaling;

        approach_vel = std::max(approach_vel, m_params.min_approach_linear_velocity);
        return std::min(constrained_linear_vel, approach_vel);
    }

    return constrained_linear_vel;
}

//float VectorPursuitController::approachVelocityConstraint(
//    float constrained_linear_vel) const
//{
//    if (m_transformedPath.poses.empty()) return constrained_linear_vel;

//    // ✅ Как в RPP: расстояние до последней точки (цели)
//    const auto& last = m_transformedPath.poses.back();
//    float dist_to_goal = std::hypot(last.position.x(), last.position.y());

//    if (dist_to_goal < m_params.approach_velocity_scaling_dist) {
//        float velocity_scaling = dist_to_goal / m_params.approach_velocity_scaling_dist;
//        float approach_vel = constrained_linear_vel * velocity_scaling;

//        approach_vel = std::max(approach_vel, m_params.min_approach_linear_velocity);
//        return std::min(constrained_linear_vel, approach_vel);
//    }

//    return constrained_linear_vel;
//}

void VectorPursuitController::applyConstraints(
    float curvature, float pose_cost,
    float& linear_vel, float& sign)
{
    float curvature_vel = linear_vel;
    float cost_vel = linear_vel;

    curvature_vel = curvatureConstraint(linear_vel, curvature);

    if (m_params.use_cost_regulated_linear_velocity_scaling) {
        cost_vel = costConstraint(linear_vel, pose_cost);
    }

    linear_vel = std::min(cost_vel, curvature_vel);

    // Ограничение по ускорению
    linear_vel = std::min(linear_vel,
                          std::fabs(m_lastLinearVel) + m_params.max_linear_accel * m_params.control_duration);

    linear_vel = approachVelocityConstraint(linear_vel);

    linear_vel = std::max(linear_vel, m_params.min_linear_vel);
    linear_vel = std::clamp(linear_vel, 0.0f, m_params.desired_linear_vel);
    linear_vel = sign * linear_vel;
}

bool VectorPursuitController::shouldRotateToPath(
    float& angle_to_path, float x_vel_sign)
{
    if (m_transformedPath.poses.empty()) return false;

    const auto& carrot = m_transformedPath.poses.front();
    angle_to_path = std::atan2(carrot.position.y(), carrot.position.x());

    if (x_vel_sign < 0.0f) {
        angle_to_path = normalizeAngle(angle_to_path + M_PI);
    }

    return m_params.use_rotate_to_heading &&
           std::fabs(angle_to_path) > m_params.rotate_to_heading_min_angle;
}

bool VectorPursuitController::shouldRotateToGoalHeading(
    float robot_x, float robot_y, float goal_yaw)
{
    if (!m_params.use_rotate_to_heading) return false;
    if (m_transformedPath.poses.empty()) return false;

    float dist_to_goal = std::hypot(m_transformedPath.poses.back().position.x(),
                                    m_transformedPath.poses.back().position.y());

    return dist_to_goal < m_params.goal_dist_tol;
}

void VectorPursuitController::rotateToHeading(
    float& linear_vel, float& angular_vel,
    float angle_to_path, float current_angular_vel, float dt)
{
    linear_vel = 0.0f;
    float sign = angle_to_path > 0.0f ? 1.0f : -1.0f;
    angular_vel = sign * m_params.rotate_to_heading_angular_vel;

    float min_feasible = current_angular_vel - m_params.max_angular_accel * dt;
    float max_feasible = current_angular_vel + m_params.max_angular_accel * dt;
    angular_vel = std::clamp(angular_vel, min_feasible, max_feasible);

    float max_vel_to_stop = std::sqrt(2.0f * m_params.max_angular_accel * std::fabs(angle_to_path));
    if (std::fabs(angular_vel) > max_vel_to_stop) {
        angular_vel = sign * max_vel_to_stop;
    }
}

float VectorPursuitController::getCuspDist(const Path& transformed_plan) const
{
    for (size_t pose_id = 1; pose_id < transformed_plan.poses.size() - 1; ++pose_id) {
        float oa_x = transformed_plan.poses[pose_id].position.x() -
                     transformed_plan.poses[pose_id - 1].position.x();
        float oa_y = transformed_plan.poses[pose_id].position.y() -
                     transformed_plan.poses[pose_id - 1].position.y();
        float ab_x = transformed_plan.poses[pose_id + 1].position.x() -
                     transformed_plan.poses[pose_id].position.x();
        float ab_y = transformed_plan.poses[pose_id + 1].position.y() -
                     transformed_plan.poses[pose_id].position.y();

        float dot_prod = (oa_x * ab_x) + (oa_y * ab_y);
        if (dot_prod < 0.0f) {
            return std::hypot(transformed_plan.poses[pose_id].position.x(),
                              transformed_plan.poses[pose_id].position.y());
        }
    }

    return std::numeric_limits<float>::max();
}

VPCOutput VectorPursuitController::computeVelocityCommands(
    float robot_x, float robot_y, float robot_yaw,
    float current_linear_vel, float current_angular_vel,
    float dt)
{
    VPCOutput output;

    if (m_path.poses.empty()) {
        output.linear_vel = 0.0f;
        output.angular_vel = 0.0f;
        return output;
    }

    if (dt <= 0.0f) dt = 0.1f;

    // Трансформируем путь в систему координат робота
    transformPathToRobotFrame(robot_x, robot_y, robot_yaw);

    // Lookahead distance
    float lookahead_dist = getLookAheadDistance(current_linear_vel);

    // После getLookAheadDistance:
    m_lookaheadDist = lookahead_dist;

    // Cusp check
    const float dist_to_cusp = getCuspDist(m_transformedPath);
    if (dist_to_cusp < lookahead_dist) {
        lookahead_dist = dist_to_cusp;
    }

    // Находим lookahead point
    float lookahead_x, lookahead_y, lookahead_heading;
    if (!findLookAheadPoint(lookahead_dist, lookahead_x, lookahead_y, lookahead_heading)) {
        output.linear_vel = 0.0f;
        output.angular_vel = 0.0f;
        return output;
    }

    m_lookaheadX = lookahead_x;
    m_lookaheadY = lookahead_y;
    m_lookaheadHeading = lookahead_heading;

    // Определяем направление движения
    float sign = 1.0f;
    if (m_params.allow_reversing) {
        sign = lookahead_x >= 0.0f ? 1.0f : -1.0f;
    }

    float linear_vel = m_params.desired_linear_vel;
    float angular_vel = 0.0f;

    // Поворот к цели или пути
    float angle_to_path;
    if (shouldRotateToGoalHeading(robot_x, robot_y, m_transformedPath.poses.back().orientation)) {
        m_isRotatingToHeading = true;
        float angle_to_goal = m_transformedPath.poses.back().orientation;
        rotateToHeading(linear_vel, angular_vel, angle_to_goal, current_angular_vel, dt);
    } else if (shouldRotateToPath(angle_to_path, sign)) {
        m_isRotatingToHeading = true;
        rotateToHeading(linear_vel, angular_vel, angle_to_path, current_angular_vel, dt);
    } else {
        m_isRotatingToHeading = false;

        // Вычисляем turning radius через Vector Pursuit
        float turning_radius = calculateTurningRadius(lookahead_x, lookahead_y, lookahead_heading);
        float curvature = 1.0f / turning_radius;

        // Стоимость в текущей позиции
        float pose_cost = m_collisionChecker.costAtPose(robot_x, robot_y);

        // Применяем ограничения
        applyConstraints(curvature, pose_cost, linear_vel, sign);

        // После calculateTurningRadius:
        m_lastTurningRadius = turning_radius;
        m_lastCurvature = curvature;
        m_lastSign = sign;

        // Вычисляем angular velocity
        angular_vel = linear_vel / turning_radius;
        if (lookahead_y < 0) {
            angular_vel *= -1;
        }
    }

    // Проверка коллизий
    if (m_params.use_collision_detection) {
        float target_dist = std::hypot(lookahead_x, lookahead_y);
        if (m_collisionChecker.isCollisionImminent(
                robot_x, robot_y, robot_yaw,
                linear_vel, angular_vel,
                target_dist,
                m_params.max_allowed_time_to_collision_up_to_target,
                dt)) {
            std::cerr << "[VPC] Collision imminent! Stopping." << std::endl;
            output.linear_vel = 0.0f;
            output.angular_vel = 0.0f;
            return output;
        }
    }

    // Формируем выход
    output.linear_vel = linear_vel;
    output.angular_vel = angular_vel;
    output.steering_angle = std::atan2(m_params.wheelbase * angular_vel,
                                        std::max(std::fabs(linear_vel), 0.1f));
    output.is_rotating_to_heading = m_isRotatingToHeading;

    m_lastLinearVel = linear_vel;
    m_lastAngularVel = angular_vel;


    return output;
}

}  // namespace vpc
