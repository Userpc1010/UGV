#include "quadbike_controller.h"
#include <QDebug>
#include <cstring>
#include <cmath>
#include <stdio.h>  /* Standard input/output definitions */
#include <string.h> /* String function definitions */
#include <unistd.h> /* UNIX standard function definitions */
#include <fcntl.h> /* File control definitions */
#include <errno.h> /* Error number definitions */
#include <sys/poll.h>
#include <sys/types.h>
#include <stdint.h>
#include <stdbool.h>
#include <termios.h> /* POSIX terminal control definitions */
#include <linux/serial.h> // for RS-485
#include <sys/ioctl.h>

#define FNDELAY	O_NDELAY
#define RTS_SET	1
#define RTS_CLR 0

// converts integer baud to Linux define
static int get_baud(int baud)
{
    switch (baud) {
    case 9600: return B9600;
    case 19200: return B19200;
    case 38400: return B38400;
    case 57600: return B57600;
    case 115200: return B115200;
    case 230400: return B230400;
    case 460800: return B460800;
    case 500000: return B500000;
    case 576000: return B576000;
    case 921600: return B921600;
    case 1000000: return B1000000;
    case 1152000: return B1152000;
    case 1500000: return B1500000;
    case 2000000: return B2000000;
    case 2500000: return B2500000;
    case 3000000: return B3000000;
    case 3500000: return B3500000;
    case 4000000: return B4000000;
    default: return -1;
    }
}

// ========== КОНСТРУКТОР ==========

quadbike_controller::quadbike_controller(QObject *parent)
    : QObject(parent)
{
    // Инициализация UART
//    fd4 = init_comport("/dev/ttyUSB0", 2000000);
//    if (fd4 < 0) {
//        qWarning() << "Failed to open UART port";
//    }

    // Инициализация структур
    pca.w_cmd = 0.0f;
    pca.setpoint = 0.0f;
    pca.stop = 0;

    mppi_ctrl.steer_angle = 0.0f;
    mppi_ctrl.throttle = 0.0f;
    mppi_ctrl.brake = 0.0f;

    current_mode = CTRL_MODE_MANUAL;

    // Инициализация VPC
    m_vpcController = std::make_unique<vpc::VectorPursuitController>();

    vpc::VPCParams vpc_params;

    m_vpcController->configure(vpc_params);

    // Инициализация MPPI
    m_mppiController = std::make_unique<MPPIController>();

    bool mppi_ok = m_mppiController->initialize(
        0.1f,      // dt
        1.0f,      // lambda
        2.0f,      // desired_speed (м/с)
        800, 800,  // costmap width, height
        1.0f,      // resolution
        0, 0       // origin
    );

    if (mppi_ok) {
        m_mppiInitialized = true;
        qDebug() << "[quadbike_controller] MPPI initialized";
    } else {
        qWarning() << "[quadbike_controller] MPPI initialization failed";
    }

    // Таймер UART (200 Гц = 5 мс)
//    timer_uart = new QTimer(this);
//    connect(timer_uart, &QTimer::timeout, this, &quadbike_controller::UART4_timer);
//    timer_uart->start(5);

    // Сразу отправляем режим MANUAL
    sendMode(CTRL_MODE_MANUAL);
}

// ========== ДЕСТРУКТОР ==========

quadbike_controller::~quadbike_controller()
{
    if (timer_uart) {
        timer_uart->stop();
        delete timer_uart;
    }
    if (fd4 >= 0) {
        close(fd4);
    }
}

// ========== ВЫЧИСЛЕНИЕ YAW И СКОРОСТИ ==========

float quadbike_controller::computeYaw(const odometry& odom)
{
    // Кватернион в yaw (поворот вокруг Z)
    Eigen::Quaternionf q(odom.w, odom.x, odom.y, odom.z);
    Eigen::Vector3f forward = q * Eigen::Vector3f::UnitZ();  // или UnitX
    return std::atan2(forward.x(), forward.z());
}

float quadbike_controller::computeLinearVel(const odometry& odom, float yaw)
{
    // Проекция скорости на направление движения
    Eigen::Vector3f vel(odom.x_vel, odom.y_vel, odom.z_vel);
    Eigen::Vector3f forward(std::cos(yaw), 0.0f, std::sin(yaw));
    return vel.dot(forward);
}

// ========== ОБНОВЛЕНИЕ VPC ==========

void quadbike_controller::updateVPC()
{
    if (!m_hasVpcPath || !m_vpcController) return;

    // Вычисляем yaw из одометрии
    float yaw = computeYaw(data_);
    float v_current = computeLinearVel(data_, yaw);
    float w_current = last_angular_vel;

    // Вызов VPC
    vpc::VPCOutput vpc_out = m_vpcController->computeVelocityCommands(
        data_.x_pos, data_.z_pos, yaw,
        v_current, w_current,
        0.05f  // dt
    );

    // Отправляем в STM32 через PCA (ручной режим)
    pca.setpoint = vpc_out.linear_vel;
    pca.w_cmd = vpc_out.angular_vel;
    pca.stop = 0;
}

// ========== ОБНОВЛЕНИЕ MPPI ==========

void quadbike_controller::updateMPPI()
{
    if (!m_mppiInitialized || !m_hasCostmap || !m_hasGoal) return;

    // Вычисляем yaw
    float yaw = computeYaw(data_);
    float v_current = computeLinearVel(data_, yaw);

    // Позиция робота в costmap (центр 400,400)
    float robot_cx = 400.0f;
    float robot_cz = 400.0f;

    // Цель в costmap координатах
    float goal_cx = 400.0f + (m_goalX - m_costmapOriginX) / m_costmapResolution;
    float goal_cz = 400.0f + (m_goalY - m_costmapOriginY) / m_costmapResolution;

    // Обновляем costmap и цель
    m_mppiController->setCostmapCPU(m_costmapBuffer.data());
    m_mppiController->setGoal(goal_cx, goal_cz, m_goalYaw);

    // Вызов MPPI
    MPPI_Output mppi_out = m_mppiController->computeControlSync(
        robot_cx, robot_cz,
        yaw,
        v_current,
        0.0f  // steer angle
    );

    // Отправляем в STM32 через MPI (MPPI режим)
    mppi_ctrl.steer_angle = mppi_out.control_w;  // или angle
    mppi_ctrl.throttle = mppi_out.control_v;
    mppi_ctrl.brake = 0.0f;
}

// ========== ПРИЁМ ОТ TcpServer ==========

void quadbike_controller::Server_to_Controller(QByteArray data)
{
    if (data.size() < 3) return;

    // Смена режима: "MODE" + байт
    if (data[0] == 'M' && data[1] == 'O' && data[2] == 'D' && data[3] == 'E')
    {
        if (data.size() >= 5)
        {
            uint8_t requested_mode = data[4];

            if (requested_mode == 0) {
                current_mode = CTRL_MODE_MANUAL;
                qDebug() << "Mode: MANUAL";
            } else if (requested_mode == 1) {
                current_mode = CTRL_MODE_MPPI;
                qDebug() << "Mode: MPPI";
            } else if (requested_mode == 2) {
                current_mode = CTRL_MODE_VPC;
                qDebug() << "Mode: VPC";
            } else {
                qWarning() << "Unknown mode:" << requested_mode;
                return;
            }

            // Отправляем режим на STM32
            // Для STM32: 0=MANUAL, 1=MPPI (VPC тоже через PCA)
            if (current_mode == CTRL_MODE_MPPI) {
                sendMode(CTRL_MODE_MPPI);
            } else {
                sendMode(CTRL_MODE_MANUAL);
            }
        }
        return;
    }

    // Ручной режим: "MAN" + float v_cmd + float w_cmd
    if (data[0] == 'M' && data[1] == 'A' && data[2] == 'N')
    {
        if (data.size() >= 11)
        {
            float v_cmd, w_cmd;
            memcpy(&v_cmd, data.constData() + 3, sizeof(float));
            memcpy(&w_cmd, data.constData() + 7, sizeof(float));

            pca.setpoint = v_cmd;
            pca.w_cmd = w_cmd;
            pca.stop = 0;

            if (current_mode != CTRL_MODE_MANUAL) {
                current_mode = CTRL_MODE_MANUAL;
                sendMode(CTRL_MODE_MANUAL);
            }

            qDebug() << "MAN: v =" << v_cmd << "w =" << w_cmd;
        }
        return;
    }
}

// ========== ТАЙМЕР UART ==========

void quadbike_controller::UART4_timer()
{
        if (fd4 < 0) return;

        // Heartbeat + отправка в зависимости от режима
        if (current_mode == CTRL_MODE_MANUAL) {
            sendPCA();
        } else if (current_mode == CTRL_MODE_MPPI) {
            sendMPPI();
        } else if (current_mode == CTRL_MODE_VPC) {
            sendPCA();  // VPC использует PCA, потому что STM32 в MANUAL режиме
        }

    // Приём телеметрии
    int avail = serialDataAvail(fd4);
    if (avail <= 0) return;

    int len = read_com(fd4, sizeof(UART_RX_BUFFER), 10, UART_RX_BUFFER);

    if (len >= (int)sizeof(TelePacket))
    {
        if (UART_RX_BUFFER[0] == 'T' && UART_RX_BUFFER[1] == 'E' &&
            UART_RX_BUFFER[2] == 'L' && UART_RX_BUFFER[3] == 'E')
        {
            TelePacket tele_packet;
            memcpy(&tele_packet, UART_RX_BUFFER, sizeof(TelePacket));

            telem = tele_packet;

            QByteArray telData;
            telData.append("TEL");
            telData.append(reinterpret_cast<const char*>(&tele_packet.velosity_1d_mps), sizeof(float));
            telData.append(reinterpret_cast<const char*>(&tele_packet.w_yaw), sizeof(float));

            emit Controller_to_Server(telData);
        }
    }
}


// ========== СЛОТЫ ==========

void quadbike_controller::manual_points(QVector<QVector3D> data)
{
    Q_UNUSED(data)
}

void quadbike_controller::PathToTarget(std::vector<PathPoint> points)
{
    Q_UNUSED(points)
}

void quadbike_controller::onModeChanged(uint8_t mode)
{
    switch (mode) {
        case 0:
            current_mode = CTRL_MODE_MANUAL;
            sendMode(CTRL_MODE_MANUAL);
            qDebug() << "Mode: MANUAL";
            break;
        case 1:
            current_mode = CTRL_MODE_MPPI;
            sendMode(CTRL_MODE_MPPI);
            qDebug() << "Mode: MPPI";
            break;
        case 2:
            current_mode = CTRL_MODE_VPC;
            sendMode(CTRL_MODE_MANUAL);  // VPC работает через PCA (MANUAL на STM32)
            qDebug() << "Mode: VPC";
            break;
        default:
            qWarning() << "Unknown mode:" << mode;
            break;
    }
}

void quadbike_controller::odometry_lidar(odometry data)
{
    data_ = data;

    last_yaw = computeYaw(data);
    last_angular_vel = 0.0f;  // TODO

    // Вызываем контроллер в зависимости от режима
    if (current_mode == CTRL_MODE_VPC) {
        updateVPC();
    } else if (current_mode == CTRL_MODE_MPPI) {
        updateMPPI();
    }
    // MANUAL — ничего не делаем
}

void quadbike_controller::updateCostmap(const uint8_t* data, int width, int height,
                                        float resolution, float origin_x, float origin_y)
{
    if (!data || width <= 0 || height <= 0) return;

    m_costmapBuffer.assign(data, data + width * height);
    m_costmapWidth = width;
    m_costmapHeight = height;
    m_costmapResolution = resolution;
    m_costmapOriginX = origin_x;
    m_costmapOriginY = origin_y;
    m_hasCostmap = true;

    // Обновляем VPC
    if (m_vpcController) {
        m_vpcController->updateCostmap(
            m_costmapBuffer.data(),
            m_costmapWidth, m_costmapHeight,
            m_costmapResolution,
            m_costmapOriginX, m_costmapOriginY
        );
    }

    // Обновляем MPPI
    if (m_mppiInitialized) {
        m_mppiController->setCostmapCPU(m_costmapBuffer.data());
    }
}

void quadbike_controller::onPlanningFinished( bool success, const Path& path, const std::vector<std::tuple<float, float, float>>& expansions, double planning_time_ms)
{
    Q_UNUSED(expansions)
    Q_UNUSED(planning_time_ms)

    if (!success) {
        qWarning() << "[quadbike_controller] Planning failed";
        return;
    }

    if (path.poses.empty()) {
        qWarning() << "[quadbike_controller] Empty path";
        return;
    }

    // Сохраняем путь
    m_vpcPath = path;
    m_hasVpcPath = true;

    // Передаём путь в VPC
    if (m_vpcController) {
        m_vpcController->setPath(path);
    }

    qDebug() << "[quadbike_controller] VPC path updated:" << path.poses.size() << "poses";
}

void quadbike_controller::setGoal(float x, float y, float yaw)
{
    m_goalX = x;
    m_goalY = y;
    m_goalYaw = yaw;
    m_hasGoal = true;
}

// ========== UART FUNCTIONS ==========

void quadbike_controller::sendMode(ctrl_mode_t mode)
{
    if (fd4 < 0) return;

    uint8_t buffer[5];
    buffer[0] = 'M';
    buffer[1] = 'O';
    buffer[2] = 'D';
    buffer[3] = 'E';
    buffer[4] = (uint8_t)mode;

    write_com(fd4, buffer, 5, 100);
    qDebug() << "MODE sent:" << (int)mode;
}

void quadbike_controller::sendPCA()
{
    if (fd4 < 0) return;

    uint8_t buffer[3 + sizeof(PCA9685)];
    buffer[0] = 'P';
    buffer[1] = 'C';
    buffer[2] = 'A';

    memcpy(buffer + 3, &pca, sizeof(PCA9685));

    write_com(fd4, buffer, sizeof(buffer), 100);
}

void quadbike_controller::sendMPPI()
{
    if (fd4 < 0) return;

    uint8_t buffer[3 + sizeof(MPPI_Ctrl)];
    buffer[0] = 'M';
    buffer[1] = 'P';
    buffer[2] = 'I';

    memcpy(buffer + 3, &mppi_ctrl, sizeof(MPPI_Ctrl));

    write_com(fd4, buffer, sizeof(buffer), 100);
}

/**
  * @brief  inittialize comport
  * @param  comport : string value represents com device, examle: "/dev/ttyUSB0"
  * @retval file decriptor of initialized port (<0 if failed)
*/
int quadbike_controller::init_comport(const char *comport, int baud)
{
    int fd = 0;
    struct termios options;
    fd = open(comport, O_RDWR | O_NOCTTY | O_NDELAY);

    if (fd < 0) {
        fprintf(stderr, "open_port: Unable to open %s - %s\n", comport, strerror(errno));
        return fd;
    }

    /* Configure port reading */
    //fcntl(fd, F_SETFL, 0); 	//read com-port is the bloking
    fcntl(fd, F_SETFL, FNDELAY);  //read com-port not bloking
    //fcntl(fd, F_SETFL, O_NDELAY);  //read com-port not bloking

    //ioctl(fd, FIOASYNC, 1);

    /* Get the current options for the port */
    tcgetattr(fd, &options);
    if (0 != baud) {
        cfsetispeed(&options, get_baud(baud));
    } else {
        cfsetispeed(&options, B2000000);
    }

    /* Enable the receiver and set local mode */
    options.c_cflag |= (CLOCAL | CREAD);
    options.c_cflag &= ~PARENB;          /* Mask the character size to 8 bits, no parity */
    options.c_cflag &= ~CSTOPB;          //one stop bit
    //options.c_cflag |= CSTOPB;	     //two stop bit
    options.c_cflag &= ~CSIZE;           /* Select 8 data bits */
    options.c_cflag |= CS8;
    options.c_cflag &= ~CRTSCTS;         /* Disable hardware flow control */
    options.c_oflag &= ~OPOST;           /* Disable postprocessing */
    options.c_iflag &= ~(IXON | IXOFF | IXANY); /* Software flow control is disabled */
    options.c_lflag &= ~(ICANON | ECHO | ISIG); /* Enable data to be processed as raw input */

    /* Set the new options for the port */
    options.c_cc[VMIN]  = 0;
    options.c_cc[VTIME] = 10;

    options.c_iflag |= IGNBRK;
    options.c_iflag &= ~ICRNL;
    options.c_oflag &= ~ONLCR;
    options.c_lflag &= ~IEXTEN;
    options.c_lflag &= ~ECHOE;
    options.c_lflag &= ~ECHOK;

    tcsetattr(fd, TCSANOW, &options);

    return fd;
}

/**
    * @brief  reads from comport into ring buffer
    * @param  fd : comport file descriptor
    * @param  len :
    * @param  timeout : time to wait for data, ms
    * @buff  buffer for read
    * @retval size of data peing placed in ringbuffer
    */
int quadbike_controller::read_com(int fd, int len, int timeout, uint8_t *buff)
{
    int ret = 0;

    struct pollfd fds;
    fds.fd = fd;
    fds.events = POLLIN;
    poll(&fds, 1, timeout);
    if (fds.revents & POLLIN) {
        ret = read(fd, buff, len);
    }
    if (ret < 0) {
        ret = 0;
    }
    return ret;
}

/**
    * @brief  writes into comport
    * @param  fd : comport file descriptor
    * @param  buf : pointer to data
    * @param  size : data size
    * @param  timeout : time to wait for port to be free, ms
    * @retval 1 - success, 0 - fail
    */
int quadbike_controller::write_com(int fd, uint8_t *buf, size_t size, int timeout)
{
    int ret = 0;

    struct pollfd fds;
    fds.fd = fd;
    fds.events = POLLOUT;

    poll(&fds, 1, timeout);
    if (fds.revents & POLLOUT) {
        ret = write(fd, (uint8_t*)buf, size);
        tcdrain(fd);
    }

    if (ret != (int)size) return 0;
    return 1;
}

void quadbike_controller::set_blocking(int fd, int should_block)
{
    struct termios tty;
    memset(&tty, 0, sizeof tty);
    if (tcgetattr(fd, &tty) != 0) {
        printf("Error tcgetattr: %s\n", strerror(errno));
        return;
    }

    tty.c_cc[VMIN]  = should_block ? 1 : 0;
    tty.c_cc[VTIME] = 5;  // 0.5 seconds read timeout

    if (tcsetattr(fd, TCSANOW, &tty) != 0)
        printf("Error tcsetattr: %s\n", strerror(errno));
}

/*
 * serialDataAvail:
 *	Return the number of bytes of data avalable to be read in the serial port
 *********************************************************************************
 */
int quadbike_controller::serialDataAvail(const int fd)
{
    int result;
    if (ioctl(fd, FIONREAD, &result) == -1)
        return -1;
    return result;
}
