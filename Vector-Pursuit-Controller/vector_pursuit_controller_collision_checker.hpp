#ifndef VECTOR_PURSUIT_CONTROLLER_COLLISION_CHECKER_HPP
#define VECTOR_PURSUIT_CONTROLLER_COLLISION_CHECKER_HPP

#include <datatypes_smac_planner.h>
#include <vector>
#include <functional>
#include <cstdint>

namespace vpc  // ✅ namespace vpc
{

/**
 * @brief Упрощённый коллизия-чекер для Vector Pursuit Controller
 *
 * Прямой доступ к данным costmap
 */
class CollisionChecker
{
public:
    CollisionChecker();

    /**
     * @brief Инициализация данными costmap
     * @param costmap_data Указатель на данные costmap (0-254)
     * @param width Ширина costmap в ячейках
     * @param height Высота costmap в ячейках
     * @param resolution Разрешение (метров на ячейку)
     * @param origin_x X-координата левого нижнего угла в мировых координатах
     * @param origin_y Y-координата левого нижнего угла в мировых координатах
     * @param inscribed_radius Радиус вписанной окружности робота (м)
     * @param circumscribed_radius Радиус описанной окружности робота (м)
     */
    void configure(
        const uint8_t* costmap_data,
        int width, int height,
        float resolution,
        float origin_x, float origin_y,
        float inscribed_radius,
        float circumscribed_radius);

    /**
     * @brief Проверка на неизбежное столкновение
     * @param robot_x, robot_y Позиция робота в мировых координатах
     * @param robot_yaw Ориентация робота (рад)
     * @param linear_vel Линейная скорость
     * @param angular_vel Угловая скорость
     * @param carrot_dist Расстояние до carrot
     * @param max_time Максимальное время проекции
     * @param dt Шаг времени
     * @return true если столкновение неизбежно
     */
    bool isCollisionImminent(
        float robot_x, float robot_y, float robot_yaw,
        float linear_vel, float angular_vel,
        float carrot_dist, float max_time, float dt);

    /**
     * @brief Проверка столкновения в точке
     * @param x, y Мировые координаты
     * @param theta Ориентация
     * @return true если столкновение
     */
    bool inCollision(float x, float y, float theta) const;

    /**
     * @brief Стоимость в точке
     * @param x, y Мировые координаты
     * @return Стоимость (0-254)
     */
    uint8_t costAtPose(float x, float y) const;

private:
    const uint8_t* m_costmapData = nullptr;
    int m_width = 0;
    int m_height = 0;
    float m_resolution = 1.0f;
    float m_originX = 0.0f;
    float m_originY = 0.0f;
    float m_inscribedRadius = 0.3f;
    float m_circumscribedRadius = 0.5f;

    bool worldToMap(float wx, float wy, int& mx, int& my) const;
    float footprintCostAtPose(float x, float y, float theta) const;
};

}  // namespace vpc

#endif  // VECTOR_PURSUIT_CONTROLLER_COLLISION_CHECKER_HPP
