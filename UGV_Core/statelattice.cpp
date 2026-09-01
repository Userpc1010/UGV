#include "statelattice.h"
#include "rgbd_camera.h"
#include "nav2_costmap_2d/costmap_2d_ros.hpp"
#include <QFileDialog>


StateLattice::StateLattice(QObject* parent) : QObject(parent)
{
    // Инициализация параметров по умолчанию
    costmap_config_.resolution = Map_scale / 100.0f;
    costmap_config_.width = Map_X;
    costmap_config_.height = Map_Z;
    costmap_config_.origin_x = 0.0f;  // Центрируем карту
    costmap_config_.origin_y = 0.0f;
    costmap_config_.robot_radius = 3.0f;
    costmap_config_.footprint = "[[3.0,3.0],[3.0,-3.0],[-3.0,-3.0],[-3.0,3.0]]";
    costmap_config_.global_frame = "map";
    costmap_config_.robot_base_frame = "base_link";
    costmap_config_.transform_tolerance = 0.3f;
    costmap_config_.footprint_padding = 1.5f;
    costmap_config_.inflation_radius = 15.0f;
    costmap_config_.cost_scaling_factor = 10.0f;

    // Параметры Lattice планировщика
    planner_params_.tolerance = 1.0f;
    planner_params_.allow_unknown = false;
    planner_params_.max_iterations = 10000000;
    planner_params_.max_on_approach_iterations = 1000;
    planner_params_.lattice_filepath = "";  // Путь к файлу решетки
    planner_params_.reverse_penalty = 2.0f;
    planner_params_.change_penalty = 0.05f;
    planner_params_.non_straight_penalty = 1.05f;
    planner_params_.cost_penalty = 2.0f;
    planner_params_.retrospective_penalty = 0.015f;
    planner_params_.rotation_penalty = 2.5f;  // Специфично для lattice
    planner_params_.analytic_expansion_ratio = 3.5f;
    planner_params_.analytic_expansion_max_cost = 200.0f;
    planner_params_.analytic_expansion_max_cost_override = false;
    planner_params_.max_planning_time = 5.0f;
    planner_params_.lookup_table_size = 60.0f;
    planner_params_.allow_reverse_expansion = true;
    planner_params_.smooth_path = false;
    planner_params_.debug_visualizations = false;
    planner_params_.goal_heading_mode = "DEFAULT";
    planner_params_.coarse_search_resolution = 1;
    planner_params_.cache_obstacle_heuristic = false;
    planner_params_.debug_visualizations = true;

    // Копирование параметров для динамических
    Dyn_planner_params_.tolerance = planner_params_.tolerance;
    Dyn_planner_params_.allow_unknown = planner_params_.allow_unknown;
    Dyn_planner_params_.max_iterations = planner_params_.max_iterations;
    Dyn_planner_params_.max_on_approach_iterations = planner_params_.max_on_approach_iterations;
    Dyn_planner_params_.lattice_filepath = planner_params_.lattice_filepath;
    Dyn_planner_params_.reverse_penalty = planner_params_.reverse_penalty;
    Dyn_planner_params_.change_penalty = planner_params_.change_penalty;
    Dyn_planner_params_.non_straight_penalty = planner_params_.non_straight_penalty;
    Dyn_planner_params_.cost_penalty = planner_params_.cost_penalty;
    Dyn_planner_params_.retrospective_penalty = planner_params_.retrospective_penalty;
    Dyn_planner_params_.rotation_penalty = planner_params_.rotation_penalty;
    Dyn_planner_params_.analytic_expansion_ratio = planner_params_.analytic_expansion_ratio;
    Dyn_planner_params_.analytic_expansion_max_cost = planner_params_.analytic_expansion_max_cost;
    Dyn_planner_params_.analytic_expansion_max_cost_override = planner_params_.analytic_expansion_max_cost_override;
    Dyn_planner_params_.max_planning_time = planner_params_.max_planning_time;
    Dyn_planner_params_.lookup_table_size = planner_params_.lookup_table_size;
    Dyn_planner_params_.allow_reverse_expansion = planner_params_.allow_reverse_expansion;
    Dyn_planner_params_.smooth_path = planner_params_.smooth_path;
    Dyn_planner_params_.debug_visualizations = planner_params_.debug_visualizations;
    Dyn_planner_params_.goal_heading_mode = planner_params_.goal_heading_mode;
    Dyn_planner_params_.coarse_search_resolution = planner_params_.coarse_search_resolution;
    Dyn_planner_params_.cache_obstacle_heuristic = planner_params_.cache_obstacle_heuristic;

    // Параметры сглаживания
    smoother_params_.w_data = 0.2;
    smoother_params_.w_smooth = 0.3;
    smoother_params_.max_iterations = 1500;

    // Создание объектов
    costmap_master_ = std::make_shared<Costmap2D_Master>();
    planner_ = std::make_unique<SmacPlannerLattice>();

    QString latticePath = QDir::homePath() + "/UGV/nav2_smac_planner/lattice_primitives/output.json";

        // Сначала проверяем, существует ли файл и читаем ли он
        QFileInfo fileInfo(latticePath);
        if (fileInfo.exists() && fileInfo.isReadable()) {
            // Пробуем прочитать файл
            QFile file(latticePath);
            if (file.open(QIODevice::ReadOnly | QIODevice::Text)) {
                QByteArray content = file.read(200);
                file.close();

                if (content.contains("{") && content.contains("lattice_metadata")) {
                    planner_params_.lattice_filepath = latticePath.toStdString();
                    std::cout << "[INFO] Using existing lattice file: " << latticePath.toStdString() << std::endl;
                } else {
                    std::cout << "[WARNING] Lattice file exists but not valid JSON, using built-in primitives" << std::endl;
                    planner_params_.lattice_filepath = "";
                }
            } else {
                std::cout << "[WARNING] Cannot open lattice file, using built-in primitives" << std::endl;
                planner_params_.lattice_filepath = "";
            }
        } else {
            std::cout << "[INFO] Lattice file not found, using built-in primitives" << std::endl;
            planner_params_.lattice_filepath = "";
        }


     costmap_master_->on_configure(costmap_config_);
     planner_->configure(smoother_params_, planner_params_, costmap_master_);

     // Инициализация указателей
     output_vertices_close_node = nullptr;
     output_vertices_path = nullptr;
}

StateLattice::~StateLattice()
{
    cleanup();
}

void StateLattice::cleanup()
{
    // Освобождаем память вершин
    if (ptr_guard_output_vertices_close_node) {
        delete[] output_vertices_close_node;
        output_vertices_close_node = nullptr;
        output_vertices_close_node_size = 0;
        ptr_guard_output_vertices_close_node = 0;
    }

    if (ptr_guard_output_vertices_path) {
        delete[] output_vertices_path;
        output_vertices_path = nullptr;
        output_vertices_path_size = 0;
        ptr_guard_output_vertices_path = 0;
    }
}

void StateLattice::drawExpansionsAsLines(const std::vector<std::tuple<float, float, float>>& expansions)
{
    // Очищаем предыдущие данные
    if (ptr_guard_output_vertices_close_node) {
        delete[] output_vertices_close_node;
        output_vertices_close_node = nullptr;
        ptr_guard_output_vertices_close_node = 0;
    }

    // Проверяем, есть ли что рисовать
    if (expansions.empty()) {
        std::cout << "[StateLattice] Нет экспансий для визуализации" << std::endl;
        return;
    }

    // Каждая экспансия = линия (2 точки) × 3 координаты
    output_vertices_close_node_size = expansions.size() * 2;  // 2 точки на линию
    output_vertices_close_node = new GLfloat[output_vertices_close_node_size * 3];
    ptr_guard_output_vertices_close_node = 2;

    const float LINE_LENGTH = 0.5f;
    uint32_t vertex_index = 0;

    // ОДИН цикл - сразу в GLfloat массив
    for (const auto& exp : expansions) {
        float x = std::get<0>(exp);
        float y = std::get<1>(exp);
        float theta = std::get<2>(exp);

        // Начальная точка линии
        output_vertices_close_node[vertex_index++] = x;
        output_vertices_close_node[vertex_index++] = -53.9f;      // Инверсия Y
        output_vertices_close_node[vertex_index++] = -y;          // Z=0

        // Конечная точка линии (ориентация)
        float end_x = x + LINE_LENGTH * std::cos(theta);
        float end_y = y + LINE_LENGTH * std::sin(theta);

        output_vertices_close_node[vertex_index++] = end_x;
        output_vertices_close_node[vertex_index++] = -53.9f;    // Инверсия Y
        output_vertices_close_node[vertex_index++] = -end_y;    // Z=0
    }

    std::cout << "[StateLattice] Создано " << expansions.size()
              << " линий экспансий (" << output_vertices_close_node_size << " вершин)" << std::endl;

    if(state == 2) emit DrawLines(output_vertices_close_node, blue, output_vertices_close_node_size, ptr_guard_output_vertices_close_node);
}

void StateLattice::drawPathAsLines(const Path& path)
{
    // Очищаем предыдущие данные
    if (ptr_guard_output_vertices_path) {
        delete[] output_vertices_path;
        output_vertices_path = nullptr;
        ptr_guard_output_vertices_path = 0;
    }

    // Проверяем путь
    if (path.poses.size() < 2) {
        std::cout << "[StateLattice] Недостаточно точек в пути для визуализации" << std::endl;
        return;
    }

    // Количество линий = (точек - 1) × 2 точки на линию
    output_vertices_path_size = (path.poses.size() - 1) * 2;
    output_vertices_path = new GLfloat[output_vertices_path_size * 3];
    ptr_guard_output_vertices_path = 1;

    uint32_t vertex_index = 0;

    // ОДИН цикл - сразу в GLfloat массив
    for (size_t i = 0; i < path.poses.size() - 1; i++) {
        const Pose& pose1 = path.poses[i];
        const Pose& pose2 = path.poses[i + 1];

        // Первая точка линии
        output_vertices_path[vertex_index++] = pose1.position.x();
        output_vertices_path[vertex_index++] = -53.9f;                              // Инверсия Y
        output_vertices_path[vertex_index++] = -pose1.position.y();                 // Z=0

        // Вторая точка линии
        output_vertices_path[vertex_index++] = pose2.position.x();
        output_vertices_path[vertex_index++] = -53.9f;                              // Инверсия Y
        output_vertices_path[vertex_index++] = -pose2.position.y();                 // Z=0
    }

    std::cout << "[StateLattice] Создано " << (path.poses.size() - 1)
              << " линий пути (" << output_vertices_path_size << " вершин)" << std::endl;

    if(state == 2) emit DrawLines(output_vertices_path, green, output_vertices_path_size, ptr_guard_output_vertices_path);
}

void StateLattice::updateCostmapFromData(const uint8_t* data)
{
    std::lock_guard<std::mutex> lock(costmap_mutex_);
    costmap_master_->updateMap(data);

    emit costmapUpdated(
        data,
        costmap_config_.width,
        costmap_config_.height,
        costmap_config_.resolution,
        costmap_config_.origin_x,
        costmap_config_.origin_y
    );
}

void StateLattice::GetPathToTarget(Pose & start_pose, Pose & goal_pose, const uint8_t *costmap2D)
{

    costmap_master_->updateMap(costmap2D);

    // Замер времени начала
    auto start_time = std::chrono::high_resolution_clock::now();
    auto planning_start_time = std::chrono::steady_clock::now();

    std::cout << "[TIMING] Lattice path planning started at: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(
                    planning_start_time.time_since_epoch()).count()
              << " ms" << std::endl;

    // Планирование пути
    try {
        std::cout << "[INFO] Starting lattice path planning..." << std::endl;
        Path path = planner_->createPlan(
            start_pose,
            goal_pose,
            []() { return false; }
        );

        // Замер времени конца
        auto end_time = std::chrono::high_resolution_clock::now();
        auto planning_end_time = std::chrono::steady_clock::now();

        // Вычисляем разницу
        auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        auto planning_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            planning_end_time - planning_start_time);

        std::cout << "[TIMING] ===========================================" << std::endl;
        std::cout << "[TIMING] LATTICE PLANNING TIME: " << duration_ms.count() << " milliseconds" << std::endl;
        std::cout << "[TIMING] (" << (duration_ms.count() / 1000.0) << " seconds)" << std::endl;
        std::cout << "[TIMING] Pure planning time: " << planning_duration.count() << " ms" << std::endl;
        std::cout << "[TIMING] ===========================================" << std::endl;

        if (path.poses.empty()) {
            QString msg = "Путь не найден!";
            std::cout << "[ERROR] " << msg.toStdString() << std::endl;
            return;
        }

        // Визуализация пути
        drawPathAsLines(path);

        if(Dyn_planner_params_.debug_visualizations ) {

        // Получаем и визуализируем экспансии
        expansions_ = planner_->getLastExpansions();

        drawExpansionsAsLines(expansions_);
        }

        // Обновляем статус с временем
        QString msg = QString("Путь найден (Lattice)! Длина: %1 точек | Время: %2 мс (%3 сек)")
            .arg(path.poses.size())
            .arg(duration_ms.count())
            .arg(duration_ms.count() / 1000.0, 0, 'f', 3);

        std::cout << "[SUCCESS] " << msg.toStdString() << std::endl;

    } catch (const std::exception& e) {
        // Замер времени для ошибки
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        std::cout << "[TIMING] Lattice planning FAILED after: " << duration_ms.count() << " ms" << std::endl;

        std::cerr << "[EXCEPTION] " << e.what() << std::endl;
    }
}

// В реализации:
void StateLattice::asyncGetPathToTarget(Pose start_pose, Pose goal_pose)
{
    // ПРОСТОЙ ПОДХОД: если планировщик занят - пропускаем запрос
    if (is_planning_in_progress_.load()) {
        // Просто пропускаем без всяких проверок на зависание
        // Планировщик сам отменится по таймауту (5 сек в параметрах)
        std::cout << "[StateLattice] Планировщик занят, пропускаем запрос" << std::endl;
        return;
    }

    // Устанавливаем флаг
    is_planning_in_progress_.store(true);
    should_cancel_.store(false);

    // Запускаем в отдельном потоке через std::async
    planning_future_ = std::async(std::launch::async, [this, start_pose, goal_pose]() {
        this->runPlanning(start_pose, goal_pose);
    });
}


void StateLattice::runPlanning(Pose start_pose, Pose goal_pose)
{
    auto planning_start_time = std::chrono::steady_clock::now();

    try {
        std::cout << "[StateLattice] Асинхронное планирование запущено" << std::endl;

        // Просто вызываем планировщик
        Path path_result = planner_->createPlan(
            start_pose,
            goal_pose,
            [this]() { return should_cancel_.load(); }
        );

        auto planning_end_time = std::chrono::steady_clock::now();
        auto planning_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            planning_end_time - planning_start_time);

        if (should_cancel_.load()) {
            std::cout << "[StateLattice] Планирование отменено" << std::endl;

            //QMetaObject::invokeMethod(this, [this]() {
                emit planningCancelled();
                is_planning_in_progress_.store(false);
            //});
            return;
        }

        if (path_result.poses.empty()) {
            throw std::runtime_error("Путь не найден");
        }

        // Получаем экспансии если нужно
        std::vector<std::tuple<float, float, float>> expansions_result;
        if (Dyn_planner_params_.debug_visualizations) {
            expansions_result = planner_->getLastExpansions();
        }

        std::cout << "[StateLattice] Планирование успешно за "
                  << planning_duration.count() << " мс" << std::endl;

        // Отправляем сигнал в основной поток
        //QMetaObject::invokeMethod(this, [this, path_result, expansions_result, planning_duration]() {
            emit planningFinished(true, path_result, expansions_result,
                                  static_cast<double>(planning_duration.count()));
            is_planning_in_progress_.store(false);

            drawPathAsLines(path_result);

            // Автоматическая отрисовка если нужно
            if (!expansions_result.empty()) {
                drawExpansionsAsLines(expansions_result);
            }

        //});

    } catch (const std::exception& e) {
        auto planning_end_time = std::chrono::steady_clock::now();
        auto planning_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            planning_end_time - planning_start_time);

        std::cerr << "[StateLattice] Ошибка: " << e.what()
                  << " (за " << planning_duration.count() << " мс)" << std::endl;

        //QMetaObject::invokeMethod(this, [this, e]() {
            emit planningFailed(QString::fromStdString(e.what()));
            is_planning_in_progress_.store(false);
        //});
    }
}

void StateLattice::WindowState(uint8_t state)
{
     this->state = state;
}

void StateLattice::cancelPlanning()
{
    if (is_planning_in_progress_.load()) {
        should_cancel_.store(true);
        std::cout << "[StateLattice] Запрос на отмену планирования отправлен" << std::endl;
    }
}
