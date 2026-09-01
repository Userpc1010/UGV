#include "vector_pursuit_controller_collision_checker.hpp"
#include <cmath>
#include <algorithm>

namespace vpc  // ✅ namespace vpc
{

CollisionChecker::CollisionChecker()
{
}

void CollisionChecker::configure(
    const uint8_t* costmap_data,
    int width, int height,
    float resolution,
    float origin_x, float origin_y,
    float inscribed_radius,
    float circumscribed_radius)
{
    m_costmapData = costmap_data;
    m_width = width;
    m_height = height;
    m_resolution = resolution;
    m_originX = origin_x;
    m_originY = origin_y;
    m_inscribedRadius = inscribed_radius;
    m_circumscribedRadius = circumscribed_radius;
}

bool CollisionChecker::worldToMap(float wx, float wy, int& mx, int& my) const
{
    if (!m_costmapData) return false;

    mx = static_cast<int>((wx - m_originX) / m_resolution);
    my = static_cast<int>((wy - m_originY) / m_resolution);

    return (mx >= 0 && mx < m_width && my >= 0 && my < m_height);
}

float CollisionChecker::footprintCostAtPose(float x, float y, float theta) const
{
    if (!m_costmapData) return 255.0f;

    const float cos_th = std::cos(theta);
    const float sin_th = std::sin(theta);

    // Проверка центра
    int cx, cy;
    if (worldToMap(x, y, cx, cy)) {
        uint8_t center_cost = m_costmapData[cy * m_width + cx];
        if (center_cost >= 254) return 254.0f;
    }

    // Проверка точек на описанной окружности (4 точки)
    const float check_radius = m_circumscribedRadius * 0.7f;
    const float angles[4] = {0.0f, M_PI_2, M_PI, 3.0f * M_PI_2};

    float max_cost = 0.0f;
    for (int i = 0; i < 4; ++i) {
        float px = x + check_radius * std::cos(theta + angles[i]);
        float py = y + check_radius * std::sin(theta + angles[i]);

        int mx, my;
        if (worldToMap(px, py, mx, my)) {
            float cost = static_cast<float>(m_costmapData[my * m_width + mx]);
            max_cost = std::max(max_cost, cost);
        }
    }

    return max_cost;
}

bool CollisionChecker::inCollision(float x, float y, float theta) const
{
    if (!m_costmapData) return false;

    int mx, my;
    if (!worldToMap(x, y, mx, my)) return false;

    float cost = footprintCostAtPose(x, y, theta);

    return cost >= 254.0f;
}

uint8_t CollisionChecker::costAtPose(float x, float y) const
{
    if (!m_costmapData) return 0;

    int mx, my;
    if (!worldToMap(x, y, mx, my)) return 254;  // за границей — препятствие

    return m_costmapData[my * m_width + mx];
}

bool CollisionChecker::isCollisionImminent(
    float robot_x, float robot_y, float robot_yaw,
    float linear_vel, float angular_vel,
    float carrot_dist, float max_time, float dt)
{
    if (!m_costmapData) return false;

    if (inCollision(robot_x, robot_y, robot_yaw)) {
        return true;
    }

    if (dt <= 0.0f) dt = 0.05f;

    float projection_time = 0.0f;
    if (std::fabs(linear_vel) < 0.01f && std::fabs(angular_vel) > 0.01f) {
        double max_radius = m_circumscribedRadius;
        projection_time = 2.0 * std::sin((m_resolution / 2.0) / max_radius) / std::fabs(angular_vel);
    } else {
        projection_time = m_resolution / std::fabs(std::max(linear_vel, 0.1f));
    }

    float curr_x = robot_x;
    float curr_y = robot_y;
    float curr_theta = robot_yaw;

    float total_distance = 0.0f;
    float sim_distance_limit = carrot_dist;

    while (projection_time < max_time) {
        curr_x += projection_time * (linear_vel * std::cos(curr_theta));
        curr_y += projection_time * (linear_vel * std::sin(curr_theta));
        curr_theta += projection_time * angular_vel;

        total_distance = std::hypot(curr_x - robot_x, curr_y - robot_y);
        if (total_distance > sim_distance_limit) {
            break;
        }

        if (inCollision(curr_x, curr_y, curr_theta)) {
            return true;
        }

        projection_time += dt;
    }

    return false;
}

}  // namespace vpc
