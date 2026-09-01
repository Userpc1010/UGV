

// Copyright (c) 2020, Samsung Research America
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License. Reserved.

#ifndef NAV2_SMAC_PLANNER__TYPES_HPP_
#define NAV2_SMAC_PLANNER__TYPES_HPP_

#include <vector>
#include <utility>
#include <string>
#include "constants.hpp"
#include <iostream>


typedef std::pair<float, uint32_t> NodeHeuristicPair;


struct SmacPlanner2dDynamicParams {
  float tolerance = 0.125f;                      // допуск достижения цели в метрах (точность позиционирования)
  float cost_travel_multiplier = 1.0f;           // множитель штрафа за стоимость клетки (влияет на предпочтение маршрутов)
  float max_planning_time = 2.0f;                // максимальное время планирования в секундах (таймаут алгоритма)
  bool downsample_costmap = false;               // включение даунсемплинга карты стоимости (уменьшение разрешения для ускорения)
  bool allow_unknown = false;                     // разрешение планирования через неизвестные области (unknown space)
  bool use_final_approach_orientation = false;   // использование финальной ориентации подхода к цели
  int downsampling_factor = 1;                   // фактор уменьшения разрешения карты стоимости (во сколько раз уменьшать)
  int max_iterations = 1000000;                  // максимальное количество итераций поиска (предотвращение бесконечного цикла)
  int max_on_approach_iterations = 1000;         // максимальное количество итераций при приближении к цели (тонкая настройка у цели)
  int terminal_checking_interval = 5000;         // интервал проверки условий завершения (частота проверки критериев остановки)
};

struct SmacPlannerHybridDynamicParams {
  // Float parameters
  float max_planning_time = 5.0f;                // максимальное время планирования в секундах
  float tolerance = 0.25f;                       // допуск достижения цели в метрах
  float lookup_table_size = 20.0f;               // размер таблицы поиска эвристик в метрах
  float minimum_turning_radius = 1.4f;           // минимальный радиус поворота робота в метрах
  float reverse_penalty = 2.0f;                  // штраф за движение задним ходом (умножается на стоимость при реверсе)
  float change_penalty = 0.0f;                   // штраф за смену направления движения
  float non_straight_penalty = 1.2f;             // штраф за непрямолинейное движение (повороты)
  float cost_penalty = 2.0f;                     // штраф за стоимость клетки (влияет на обход препятствий)
  float analytic_expansion_ratio = 3.5f;         // коэффициент аналитического расширения (отношение аналитических/дискретных расширений)
  float analytic_expansion_max_length = 3.0f;    // максимальная длина аналитического расширения в метрах
  float analytic_expansion_max_cost = 200.0f;    // максимальная стоимость для аналитического расширения
  float resolution = 0.10f;                      // разрешение карты в метрах на клетку (специальный случай для динамического обновления)

  // Boolean parameters
  bool downsample_costmap = false;               // включение даунсемплинга карты стоимости
  bool allow_unknown = false;                     // разрешение планирования через неизвестные области
  bool cache_obstacle_heuristic = false;         // кэширование эвристики препятствий (ускорение повторных поисков)
  bool allow_primitive_interpolation = false;    // разрешение интерполяции примитивов движения
  bool smooth_path = false;                       // включение сглаживания результирующего пути
  bool analytic_expansion_max_cost_override = false; // переопределение максимальной стоимости аналитического расширения
  bool use_quadratic_cost_penalty = false;       // использование квадратичного штрафа стоимости
  bool downsample_obstacle_heuristic = false;     // даунсемплинг эвристики препятствий
  bool debug_visualizations = false;             // визуализация отладки

  // Integer parameters
  int downsampling_factor = 1;                   // фактор уменьшения разрешения карты стоимости
  int max_iterations = 1000000;                  // максимальное количество итераций поиска
  int max_on_approach_iterations = 1000;         // максимальное количество итераций при приближении к цели
  int terminal_checking_interval = 5000;         // интервал проверки условий завершения
  int angle_quantization_bins = 72;              // количество бинов для квантования углов (дискретизация ориентации)
  int coarse_search_resolution = 1;              // разрешение грубого поиска (шаг угла для первоначального поиска)

  // String parameters
  std::string motion_model_for_search = "DUBIN"; // модель движения для поиска ("DUBIN", "REEDS_SHEPP", etc)
  std::string goal_heading_mode = "DEFAULT";     // режим ориентации цели ("DEFAULT", "BIDIRECTIONAL", "ALL_DIRECTION")
};

struct SmacPlannerLatticeDynamicParams {
  // Float parameters
  float max_planning_time = 5.0f;                // максимальное время планирования в секундах
  float tolerance = 0.25f;                       // допуск достижения цели в метрах
  float lookup_table_size = 20.0f;               // размер таблицы поиска эвристик в метрах
  float reverse_penalty = 2.0f;                  // штраф за движение задним ходом
  float change_penalty = 0.05f;                  // штраф за смену направления движения
  float non_straight_penalty = 1.05f;            // штраф за непрямолинейное движение
  float cost_penalty = 2.0f;                     // штраф за стоимость клетки
  float rotation_penalty = 2.5f;                 // штраф за вращение (специфично для решетчатых моделей)
  float analytic_expansion_ratio = 3.5f;         // коэффициент аналитического расширения
  float analytic_expansion_max_length = 3.0f;    // максимальная длина аналитического расширения в метрах
  float analytic_expansion_max_cost = 200.0f;    // максимальная стоимость для аналитического расширения
  float retrospective_penalty = 0.015f;          // ретроспективный штраф

  // Boolean parameters
  bool allow_unknown = false;                     // разрешение планирования через неизвестные области
  bool cache_obstacle_heuristic = false;         // кэширование эвристики препятствий
  bool allow_reverse_expansion = true;          // разрешение обратного расширения (специфично для решеток)
  bool smooth_path = false;                       // включение сглаживания результирующего пути
  bool analytic_expansion_max_cost_override = false; // переопределение максимальной стоимости аналитического расширения
  bool debug_visualizations = false;             // визуализация отладки

  // Integer parameters
  int max_iterations = 1000000;                  // максимальное количество итераций поиска
  int max_on_approach_iterations = 1000;         // максимальное количество итераций при приближении к цели
  int terminal_checking_interval = 5000;         // интервал проверки условий завершения
  int coarse_search_resolution = 1;              // разрешение грубого поиска

  // String parameters
  std::string lattice_filepath = "";             // путь к файлу с примитивами решетки (JSON файл с движениями)
  std::string goal_heading_mode = "DEFAULT";     // режим ориентации цели ("DEFAULT", "BIDIRECTIONAL", "ALL_DIRECTION")
};

struct SmacPlanner2DParams {
  // Основные параметры планирования
  float tolerance = 0.125f;                    // допуск достижения цели в метрах
  float cost_penalty = 1.0f;                   // множитель штрафа за стоимость клетки
  float max_planning_time = 2.0;              // максимальное время планирования в секундах

  // Параметры оптимизации и поиска
  bool downsample_costmap = false;             // включение даунсемплинга карты стоимости
  int downsampling_factor = 1;                 // фактор уменьшения разрешения карты стоимости
  bool allow_unknown = false;                   // разрешение планирования через неизвестные области
  int max_iterations = 1000000;                // максимальное количество итераций поиска
  int max_on_approach_iterations = 1000;       // макс. итераций при приближении к цели
  int terminal_checking_interval = 5000;       // интервал проверки условий завершения

  // Параметры поведения
  bool use_final_approach_orientation = false; // использование финальной ориентации подхода
  MotionModel motion_model = MotionModel::TWOD; // модель движения (2D для плоского планирования)
};

struct SmacPlannerHybridParams {

  // Параметры даунсемплинга карты стоимости
  bool downsample_costmap = false;    // включение даунсемплинга карты стоимости
  int downsampling_factor = 1;        // фактор даунсемплинга

  // Параметры квантования углов
  int angle_quantization_bins = 72;   // количество бинов для квантования углов

  // Основные параметры планировщика
  float tolerance = 0.25f;            // допуск достижения цели
  bool allow_unknown = false;          // разрешение неизвестных областей
  int max_iterations = 1000000;       // максимальное количество итераций
  int max_on_approach_iterations = 1000; // итерации при приближении
  int terminal_checking_interval = 5000; // интервал проверки завершения
  bool smooth_path = true;            // сглаживание пути

  // Параметры движения и радиуса поворота
  float minimum_turning_radius = 0.4f; // минимальный радиус поворота

  // Параметры поиска и эвристик
  bool allow_primitive_interpolation = false; // интерполяция примитивов
  bool cache_obstacle_heuristic = false;      // кэширование эвристики препятствий
  float reverse_penalty = 2.0f;               // штраф за движение назад
  float change_penalty = 0.0f;                // штраф за смену направления
  float non_straight_penalty = 1.2f;          // штраф за непрямое движение
  float cost_penalty = 2.0f;                  // штраф за стоимость
  float retrospective_penalty = 0.015f;       // ретроспективный штраф
  float analytic_expansion_ratio = 3.5f;      // коэффициент аналитического расширения
  float analytic_expansion_max_cost = 200.0f; // максимальная стоимость аналитического расширения
  bool analytic_expansion_max_cost_override = false; // переопределение максимальной стоимости
  bool use_quadratic_cost_penalty = false;    // использование квадратичного штрафа стоимости
  bool downsample_obstacle_heuristic = false;  // даунсемплинг эвристики препятствий

  // Параметры аналитического расширения
  float analytic_expansion_max_length = 3.0f; // максимальная длина аналитического расширения

  // Параметры времени планирования и таблицы поиска
  double max_planning_time = 5.0;     // максимальное время планирования
  float lookup_table_size = 20.0;    // размер таблицы поиска

  // Параметры отладки и визуализации
  bool debug_visualizations = false;  // визуализация отладки

  // Параметры модели движения и режима цели
  std::string motion_model_for_search = "DUBIN"; // модель движения для поиска

  // Параметры заголовка цели
  std::string goal_heading_mode = "DEFAULT"; // режим ориентации цели

  // Параметры грубого поиска
  int coarse_search_resolution = 1;   // разрешение грубого поиска
};


struct SmacPlannerLatticeParams {
  // Основные параметры планировщика
  float tolerance = 0.25f;                       // ".tolerance" - допуск достижения цели в метрах
  bool allow_unknown = false;                     // ".allow_unknown" - разрешение неизвестных областей
  int max_iterations = 1000000;                  // ".max_iterations" - максимальное количество итераций поиска
  int max_on_approach_iterations = 1000;         // ".max_on_approach_iterations" - итерации при приближении к цели
  int terminal_checking_interval = 5000;         // ".terminal_checking_interval" - интервал проверки завершения
  bool smooth_path = false;                       // ".smooth_path" - сглаживание результирующего пути

  // Параметры решетчатой модели (lattice)
  std::string lattice_filepath = "";             // ".lattice_filepath" - путь к файлу с примитивами решетки

  // Параметры эвристик и штрафов
  bool cache_obstacle_heuristic = false;         // ".cache_obstacle_heuristic" - кэширование эвристики препятствий
  float reverse_penalty = 2.0f;                  // ".reverse_penalty" - штраф за движение задним ходом
  float change_penalty = 0.05f;                  // ".change_penalty" - штраф за смену направления
  float non_straight_penalty = 1.05f;            // ".non_straight_penalty" - штраф за непрямолинейное движение
  float cost_penalty = 2.0f;                     // ".cost_penalty" - штраф за стоимость клетки
  float retrospective_penalty = 0.015f;          // ".retrospective_penalty" - ретроспективный штраф
  float rotation_penalty = 2.5f;                 // ".rotation_penalty" - штраф за вращение
  float analytic_expansion_ratio = 3.5f;         // ".analytic_expansion_ratio" - коэффициент аналитического расширения
  float analytic_expansion_max_cost = 200.0f;    // ".analytic_expansion_max_cost" - максимальная стоимость аналитического расширения
  bool analytic_expansion_max_cost_override = false; // ".analytic_expansion_max_cost_override" - переопределение максимальной стоимости

  // Параметры аналитического расширения
  float analytic_expansion_max_length = 3.0f;    // ".analytic_expansion_max_length" - максимальная длина аналитического расширения в метрах

  // Параметры производительности и времени
  float max_planning_time = 5.0f;                // ".max_planning_time" - максимальное время планирования
  float lookup_table_size = 20.0f;               // ".lookup_table_size" - размер таблицы поиска
  bool allow_reverse_expansion = true;          // ".allow_reverse_expansion" - разрешение обратного расширения

  // Параметры отладки
  bool debug_visualizations = false;             // ".debug_visualizations" - визуализация отладки

  // Параметры цели
  std::string goal_heading_mode = "DEFAULT";     // ".goal_heading_mode" - режим ориентации цели

  // Параметры поиска
  int coarse_search_resolution = 1;              // ".coarse_search_resolution" - разрешение грубого поиска
};


/**
 * @struct nav2_smac_planner::SearchInfo
 * @brief Search properties and penalties
 */
struct SearchInfo
{
  float minimum_turning_radius{8.0f};
  float non_straight_penalty{1.05f};
  float change_penalty{0.0f};
  float reverse_penalty{2.0f};
  float cost_penalty{2.0f};
  float retrospective_penalty{0.015f};
  float rotation_penalty{5.0f};
  float analytic_expansion_ratio{3.5f};
  float analytic_expansion_max_length{60.0f};
  float analytic_expansion_max_cost{200.0f};
  bool analytic_expansion_max_cost_override{false};
  std::string lattice_filepath;
  bool cache_obstacle_heuristic{false};
  bool allow_reverse_expansion{false};
  bool allow_primitive_interpolation{false};
  bool downsample_obstacle_heuristic{true};
  bool use_quadratic_cost_penalty{false};
};

struct CostmapConfig {
  // Основные параметры карты
  float resolution = 0.10f;                    // Разрешение карты (метры на ячейку)
  unsigned int width = 400;                     // Ширина карты в ячейках
  unsigned int height = 400;                    // Высота карты в ячейках
  float origin_x = 0.0f;                      // X координата начала карты в мире
  float origin_y = 0.0f;                      // Y координата начала карты в мире

  // Параметры робота
  float robot_radius = 1.1f;                  // Радиус робота для круговой аппроксимации
  float footprint_padding = 0.01f;            // Отступ от контура робота
  std::string footprint = "[]";               // Геометрический контур робота в JSON
  std::string robot_base_frame = "base_link"; // Фрейм основания робота
  bool subscribe_to_stamped_footprint = false;// Подписка на stamped footprint из топика

  // Фреймы и трансформации
  std::string global_frame = "map";           // Глобальная система координат карты
  float transform_tolerance = 0.3f;           // Допустимая задержка трансформаций (сек)
  float initial_transform_timeout = 60.0f;    // Таймаут получения начальных трансформаций


  // Флаги режимов работы
  bool rolling_window = false;                // Режим "скользящего окна" (для local costmap)
  bool track_unknown_space = false;           // Отслеживать неизвестное пространство отдельно

  float cost_scaling_factor = 10.0f; //это параметр, который определяет, насколько быстро уменьшается стоимость ячеек по мере удаления от препятствий в зоне инфляции
  float inflation_radius = 3.5f;    //это радиус, на который "раздуваются" препятствия на карте стоимости для создания буферной зоны вокруг них

  // Визуализация
  float map_vis_z = 0.0f;                     // Z-координата для визуализации карты в RViz
};

/**
 * @struct nav2_smac_planner::SmootherParams
 * @brief Parameters for the smoother
 */

struct SmootherParams {
  float tolerance = 1e-7f;          // допуск сходимости алгоритма сглаживания
  int max_iterations = 1000;         // максимальное количество итераций оптимизации
  float w_data = 0.2f;               // вес сохранения исходных данных (привязка к оригинальному пути)
  float w_smooth = 0.3f;             // вес сглаживания траектории
  bool is_holonomic = true;          // флаг голономности робота (возможность движения в любом направлении)
  bool do_refinement = true;         // включение дополнительного улучшения траектории
  int refinement_num = 2;            // количество дополнительных проходов улучшения (рефинментов)
};


/**
 * @struct nav2_smac_planner::TurnDirection
 * @brief A struct with the motion primitive's direction embedded
 */
enum class TurnDirection
{
  UNKNOWN = 0,
  FORWARD = 1,
  LEFT = 2,
  RIGHT = 3,
  REVERSE = 4,
  REV_LEFT = 5,
  REV_RIGHT = 6
};

/**
 * @struct nav2_smac_planner::MotionPose
 * @brief A struct for poses in motion primitives
 */
struct MotionPose
{
  /**
   * @brief A constructor for nav2_smac_planner::MotionPose
   */
  MotionPose() {}

  /**
   * @brief A constructor for nav2_smac_planner::MotionPose
   * @param x X pose
   * @param y Y pose
   * @param theta Angle of pose
   * @param TurnDirection Direction of the primitive's turn
   */
  MotionPose(const float & x, const float & y, const float & theta, const TurnDirection & turn_dir)
  : _x(x), _y(y), _theta(theta), _turn_dir(turn_dir)
  {}

  MotionPose operator-(const MotionPose & p2)
  {
    return MotionPose(
      this->_x - p2._x, this->_y - p2._y, this->_theta - p2._theta, TurnDirection::UNKNOWN);
  }

  float _x;
  float _y;
  float _theta;
  TurnDirection _turn_dir;
};

typedef std::vector<MotionPose> MotionPoses;

/**
 * @struct nav2_smac_planner::LatticeMetadata
 * @brief A struct of all lattice metadata
 */
struct LatticeMetadata
{
  float min_turning_radius;
  float grid_resolution;
  unsigned int number_of_headings;
  std::vector<float> heading_angles;
  unsigned int number_of_trajectories;
  std::string motion_model;
};

/**
 * @struct nav2_smac_planner::MotionPrimitive
 * @brief A struct of all motion primitive data
 */
struct MotionPrimitive
{
  unsigned int trajectory_id;
  float start_angle;
  float end_angle;
  float turning_radius;
  float trajectory_length;
  float arc_length;
  float straight_length;
  bool left_turn;
  MotionPoses poses;
};

/**
 * @struct nav2_smac_planner::GoalState
 * @brief A struct to store the goal state
 */
template<typename NodeT>
struct GoalState
{
  NodeT * goal = nullptr;
  bool is_valid = true;
};

#endif  // NAV2_SMAC_PLANNER__TYPES_HPP_
