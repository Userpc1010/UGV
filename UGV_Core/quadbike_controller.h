#ifndef QUADBIKE_CONTROLLER_H
#define QUADBIKE_CONTROLLER_H

#include "data.h"
#include "rgbd_camera.h"
#include <QObject>
#include <QTimer>
#include <QByteArray>
#include <QVector>
#include <QVector3D>
#include <memory>
#include <vector>
#include <atomic>
#include <mutex>

// VPC
#include "vector_pursuit_controller.hpp"

// MPPI
#include "mppi-generic.h"

class quadbike_controller : public QObject
{
    Q_OBJECT

public:
    quadbike_controller(QObject *parent = nullptr);
    ~quadbike_controller();

signals:
    void send_telem(GUI_telem data);
    void Controller_to_Server(QByteArray data);

private:
    // UART
    int fd4;
    QTimer * timer_uart;

    // Структуры для STM32
    PCA9685 pca;
    MPPI_Ctrl mppi_ctrl;
    ctrl_mode_t current_mode = CTRL_MODE_MANUAL;

    // Телеметрия
    TelePacket telem;

    // Буфер UART
    uint8_t UART_RX_BUFFER[256] = {0};

    // Одометрия
    odometry data_;
    double last_yaw = 0.0;
    double last_angular_vel = 0.0;

    // VPC контроллер
    std::unique_ptr<vpc::VectorPursuitController> m_vpcController;
    Path m_vpcPath;
    bool m_hasVpcPath = false;

    // MPPI контроллер
    std::unique_ptr<MPPIController> m_mppiController;
    bool m_mppiInitialized = false;

    // Costmap
    std::vector<uint8_t> m_costmapBuffer;
    int m_costmapWidth = 0;
    int m_costmapHeight = 0;
    float m_costmapResolution = 1.0f;
    float m_costmapOriginX = 0.0f;
    float m_costmapOriginY = 0.0f;
    bool m_hasCostmap = false;

    // Цель
    float m_goalX = 0.0f;
    float m_goalY = 0.0f;
    float m_goalYaw = 0.0f;
    bool m_hasGoal = false;

    // Вспомогательные
    int init_comport(const char *comport, int baud);
    int read_com(int fd, int len, int timeout, uint8_t * buff);
    int write_com(int fd, uint8_t * buf, size_t size, int timeout);
    int serialDataAvail(const int fd);
    void set_blocking(int fd, int should_block);

    // Методы отправки на STM32
    void sendMode(ctrl_mode_t mode);
    void sendPCA();
    void sendMPPI();

    // Вычисление yaw из кватерниона
    float computeYaw(const odometry& odom);
    float computeLinearVel(const odometry& odom, float yaw);

    // VPC и MPPI
    void updateVPC();
    void updateMPPI();

protected slots:
    void UART4_timer();

public slots:

    void onModeChanged(uint8_t mode);
    void manual_points(QVector<QVector3D> data);
    void odometry_lidar(odometry data);
    void PathToTarget(std::vector<PathPoint> points);
    void Server_to_Controller(QByteArray data);
    void updateCostmap(const uint8_t* data, int width, int height,
                       float resolution, float origin_x, float origin_y);
    void onPlanningFinished(bool success, const Path& path,
                            const std::vector<std::tuple<float, float, float>>& expansions,
                            double planning_time_ms);
    void setGoal(float x, float y, float yaw);
};

#endif // QUADBIKE_CONTROLLER_H
