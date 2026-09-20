#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QApplication>
#include <QThread>

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
  ui->setupUi(this);

  QSurfaceFormat format;
  format.setSamples(16);
  format.setDepthBufferSize(24);
  format.setRenderableType(QSurfaceFormat::OpenGL);
  QSurfaceFormat::setDefaultFormat(format);

//  std::string config_path = "/home/sencis/build-UGV-Desktop-Debug/UAV_Core/estimator_config.yaml";
  std::string config_path = "/home/sencis/build-UGV-Desktop-Debug/FAST-LIO2/MID360_config.json";

  thread_Lidar = new QThread(this);
  SDK = new LivoxSDK(config_path, nullptr);
  SDK->moveToThread(thread_Lidar);
  thread_Lidar->start(QThread::HighPriority);

// this->setCentralWidget(processor->Wiget);
// processor->Wiget->setFocusPolicy(Qt::StrongFocus);
// processor->start(QThread::HighPriority);

//  widgetCamera = new OGLWidget(this, 1);
//  widgetLidar = new OGLWidget(this, 2);
//  widgetCostMap = new OGLWidget(this, 0);

  widgetCamera = new WidgetCamera(this);
  widgetLidar = new WidgetLidar(this);
  widgetCostMap = new WidgetCostMap(this);

  stackedWidget = new QStackedWidget(this);

  stackedWidget->addWidget(widgetCamera);
  stackedWidget->addWidget(widgetLidar);
  stackedWidget->addWidget(widgetCostMap);

  stackedWidget->setFocusPolicy(Qt::StrongFocus);
  widgetLidar->setFocusPolicy(Qt::StrongFocus);
  widgetCamera->setFocusPolicy(Qt::StrongFocus);
  widgetCostMap->setFocusPolicy(Qt::StrongFocus);
  this->setFocusPolicy(Qt::StrongFocus);

  this->setCentralWidget(stackedWidget);

  qRegisterMetaType< cv::Mat >("cv::Mat");
  qRegisterMetaType< rs2_vector >("rs2_vector");
  qRegisterMetaType< odometry >("odometry");
  qRegisterMetaType< uint16_t >("uint16_t");
  qRegisterMetaType< uint32_t >("uint32_t");
  qRegisterMetaType< uint64_t >("uint64_t");
  qRegisterMetaType< int64_t >("int64_t");
  qRegisterMetaType< GUI_telem >("GUI_telem");
  qRegisterMetaType< uint8_t >("uint8_t");
  qRegisterMetaType< GLfloat >("GLfloat");
  qRegisterMetaType< std::vector<PathPoint> >("std::vector<PathPoint>");
  qRegisterMetaType< QVector<QVector3D> >("QVector<QVector3D");
  qRegisterMetaType<Path>("Path");
  qRegisterMetaType<std::vector<std::tuple<float, float, float>>>("std::vector<std::tuple<float,float,float>>");
 // qRegisterMetaType< ImuConst >("ImuConst");
 // qRegisterMetaType< CustomMsg >("CustomMsg");

    thread_controller = new QThread(this);
    controller = new quadbike_controller(nullptr);
    controller->moveToThread(thread_controller);
    thread_controller->start(QThread::NormalPriority);

    thread_server = new QThread(this);
    Server = new TcpServer(nullptr);
    Server->moveToThread(thread_server);
    thread_server->start(QThread::NormalPriority);

//  I2C_Ports * I2C = new I2C_Ports(this);
//  GPS * gps = new GPS(this);
//  GSM * gsm = new GSM(this);
//  Lidar_TFmini * Lidar = new Lidar_TFmini(this);

//    thread_Lidar = new QThread(this);
//    lidarmap = new LidarMapping(config_path, nullptr);
//    lidarmap->moveToThread(thread_Lidar);
//    lidarmap->start(QThread::NormalPriority);
//    thread_Lidar->start(QThread::NormalPriority);

    thread_Processor = new QThread(this);
    processor = new Processor (nullptr);
    processor->moveToThread(thread_Processor);
    processor->start(QThread::NormalPriority);
    thread_Processor->start(QThread::NormalPriority);


    connect(&SDK->Node, SIGNAL(DisplayingPoint(GLfloat*, GLfloat*,unsigned long long,QQuaternion, QVector3D)),                                   widgetLidar, SLOT(DisplayingPoint(GLfloat*,GLfloat*,unsigned long long,QQuaternion, QVector3D)));

    connect(&SDK->Node, SIGNAL(DisplayingPointMap(GLfloat*, GLfloat*,unsigned long long)),                                                       widgetLidar, SLOT(DisplayingMapPoint(GLfloat*,GLfloat*,unsigned long long)));

    connect(SDK, SIGNAL(timesync(uint64_t, uint64_t)),                                                                                           processor, SLOT(timesync(uint64_t, uint64_t)));

    connect(processor, SIGNAL(camsync(double)),                                                                                                  SDK, SLOT(camsync(double)));

    connect(this, SIGNAL(destroyed(QObject*)),                                                                                                   thread_controller, SLOT(quit()));

    connect(this, SIGNAL(destroyed(QObject*)),                                                                                                   thread_Processor, SLOT(quit()));

    connect(&SDK->Node, SIGNAL(point_cloud_lidar(const uint16_t*, uint32_t, const int16_t*, uint32_t)),                                          processor, SLOT(point_cloud_lidar(const uint16_t*, uint32_t, const int16_t*, uint32_t)));

    connect(&SDK->Node, SIGNAL(odometry_lidar(odometry)),                                                                                        processor,SLOT(odometry_lidar(odometry)));

    connect(&SDK->Node, SIGNAL(odometry_lidar(odometry)),                                                                                        controller,SLOT(odometry_lidar(odometry)));

    connect(processor->state_lattice_, SIGNAL(costmapUpdated(const uint8_t*, int, int, float, float, float)),                                    controller, SLOT(updateCostmap(const uint8_t*, int, int, float, float, float)));

    connect(processor->state_lattice_, SIGNAL(planningFinished(bool, const Path&, const std::vector<std::tuple<float, float, float>>&, double)), controller, SLOT(onPlanningFinished(bool, const Path&, const std::vector<std::tuple<float, float, float>>&, double)));

    connect(processor, SIGNAL(DisplayingCubes(GLfloat*, GLfloat*, unsigned long long, QQuaternion)),                                             widgetCamera, SLOT (DisplayingCubes(GLfloat*, GLfloat*, unsigned long long, QQuaternion)));

    connect(processor, SIGNAL(DisplayingCostMap(QImage, QQuaternion)),                                                                           widgetCostMap, SLOT(DisplayingCostMap(QImage, QQuaternion)));

    connect(processor->state_lattice_, SIGNAL(DrawLines(GLfloat*,QVector3D,unsigned long long,uint16_t)),                                        widgetCostMap, SLOT(DrawLines(GLfloat*,QVector3D,unsigned long long,uint16_t)));

    connect(widgetCostMap, SIGNAL(goalPositionSet(QVector3D,QQuaternion)),                                                                       processor, SLOT(goalPositionSet(QVector3D,QQuaternion)));

    connect(processor,SIGNAL(DrawCube(QVector3D)),                                                                                               widgetCamera,SLOT(DrawCube(QVector3D)));

    connect(processor,SIGNAL(updateMapOffset(QVector3D)),                                                                                        widgetCamera,SLOT(updateMapOffset(QVector3D)));

    connect(processor, SIGNAL(DrawAruco(QVector3D)),                                                                                             widgetCamera, SLOT(DrawAruco(QVector3D)));

    connect(processor, SIGNAL(Invisible_Aruco(bool)),                                                                                            widgetCamera, SLOT(Invisible_Aruco(bool)));

    connect(processor, SIGNAL(DrawAruco(QVector3D)),                                                                                             widgetCostMap, SLOT(DrawAruco(QVector3D)));

    connect(processor, SIGNAL(Invisible_Aruco(bool)),                                                                                            widgetCostMap, SLOT(Invisible_Aruco(bool)));

    connect(controller, SIGNAL(send_telem(GUI_telem)),                                                                                           this, SLOT(send_telem(GUI_telem)));

    connect(Server, SIGNAL(Server_to_Controller(QByteArray)),                                                                                    controller, SLOT(Server_to_Controller(QByteArray)));

    connect(this, SIGNAL(WindowState(uint8_t)),                                                                                                  processor, SLOT(WindowState(uint8_t)));

    connect(this, SIGNAL(WindowState(uint8_t)),                                                                                                  processor->state_lattice_, SLOT(WindowState(uint8_t)));

    connect(this, SIGNAL(ModeChanged(uint8_t)),                                                                                                  controller, SLOT(onModeChanged(uint8_t)));

    connect(this, SIGNAL(WindowState(uint8_t)),                                                                                                  &SDK->Node, SLOT(WindowState(uint8_t)));


//  connect (Server, SIGNAL(Server_to_GPS(QByteArray)),                                                  gps, SLOT(Server_to_GPS(QByteArray)));

//  connect (gps, SIGNAL(GPS_to_Server(QByteArray)),                                                     Server, SLOT(GPS_to_Server(QByteArray)));

//  connect (gsm, SIGNAL(GSM_to_Server(QByteArray)),                                                     Server, SLOT(GSM_to_Server(QByteArray)));

//  connect (gsm, SIGNAL(GSM_to_GPS()),                                                                  gps, SLOT(GSM_to_GPS()));

//  connect (gps, SIGNAL(GPS_to_GSM(double, double)),                                                    gsm, SLOT(GPS_to_GSM(double, double)));

//  connect (Server, SIGNAL(Server_to_I2C(QByteArray)),                                                  I2C, SLOT(Server_to_I2C(QByteArray)));

//  connect (I2C, SIGNAL(I2C_to_Server(QByteArray)),                                                     Server, SLOT(I2C_to_Server(QByteArray)));

//  connect (gps, SIGNAL(GPS_to_I2C(QByteArray)),                                                        I2C, SLOT(GPS_to_I2C(QByteArray)));

//  connect (gps, SIGNAL(GPS_to_SLAM(QByteArray)),                                                       widget, SLOT(I2C_to_SLAM(QByteArray)));

//  connect (I2C, SIGNAL(I2C_to_GPS(QByteArray)),                                                        gps, SLOT(I2C_to_GPS(QByteArray)));

//  connect (Lidar, SIGNAL(Lidar_to_I2C(int)),                                                           I2C, SLOT(Lidar_to_I2C(int)));

}

MainWindow::~MainWindow()
{
    processor->deleteLater();
    controller->deleteLater();
//    lidarmap->deleteLater();
    widgetCamera->deleteLater();
    widgetLidar->deleteLater();
    Server->deleteLater();

    //usleep(1);


    thread_Lidar->terminate();
    thread_Processor->terminate();
    thread_controller->terminate();
    thread_server->terminate();

    delete ui;
}

void MainWindow::send_telem(GUI_telem data)
{
  telem = data;

  ui->curent_course->setText(QString("%1").arg(telem.course, 0, 'f', 2));

  ui->current_roll->setText(QString("%1").arg(telem.roll, 0, 'f', 2));

  ui->current_pitch->setText(QString("%1").arg(telem.pitch, 0, 'f', 2));

  ui->xyz_position->setText(QString("%1  ").arg(telem.X, 0, 'f', 2) + QString("%1  ").arg(telem.Y, 0, 'f', 2) + QString("%1").arg(telem.Z, 0, 'f', 2));

  ui->next_position_xyz->setText(QString("%1  ").arg(telem.target_X, 0, 'f', 2) + QString("%1  ").arg(telem.target_Y, 0, 'f', 2) + QString("%1").arg(telem.target_Z, 0, 'f', 2));

  ui->target_course->setText(QString("%1").arg(telem.target_course, 0, 'f', 2));

}

//void MainWindow::future_image(cv::Mat img_history)
//{
//    if (img_history.empty() != true)
//    {
//      QImage qOriginalImage((uchar*)img_history.data, img_history.cols, img_history.rows, img_history.step, QImage::Format_RGB888);
//      ui->future_image->setPixmap(QPixmap::fromImage(qOriginalImage));
//    }
//}

void MainWindow::on_pushButton_Widget_clicked()
{
    if(state > 2) state = 0;
    if(state == 0) {
     stackedWidget->setCurrentIndex(state);
     emit WindowState(state);
    }

    if(state == 1) {
     stackedWidget->setCurrentIndex(state);
     emit WindowState(state);
    }
    if(state == 2){
     stackedWidget->setCurrentIndex(state);
     emit WindowState(state);
     this->resize(800, 1080);
    }

    state++;
}

void MainWindow::on_pushButton_Mode_clicked()
{
    static uint8_t mode = 0;
    mode++;
    if (mode > 2) mode = 0;

    switch (mode) {
        case 0:
            ui->pushButton_Mode->setText("MANUAL");
            break;
        case 1:
            ui->pushButton_Mode->setText("MPPI");
            break;
        case 2:
            ui->pushButton_Mode->setText("VPC");
            break;
    }

    // Отправляем режим в контроллер
    emit ModeChanged(mode);

    qDebug() << "Mode changed to:" << mode;
}
