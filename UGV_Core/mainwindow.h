#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QKeyEvent>
#include <QStackedWidget>
#include "myserver.h"
#include "WidgetCamera.h"
#include "WidgetLidar.h"
#include "WidgetCostMap.h"
#include "rgbd_camera.h"
#include "quadbike_controller.h"
#include "livoxsdk.h"
#include "odom.h"
#include "data.h"
#include <opencv4/opencv2/opencv.hpp>

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

public slots:

    void send_telem (GUI_telem data);

//    void future_image (cv::Mat img_history);

signals:

    void WindowState (uint8_t state);
    void ModeChanged(uint8_t mode);

private:

    uint8_t state = 1;

    GUI_telem telem;

    QThread * thread_Processor;
    QThread * thread_controller;
    QThread * thread_server;
    QThread * thread_Lidar;
    QThread * thread_map;

    Processor * processor;
    quadbike_controller * controller;
    LivoxSDK * SDK;
    TcpServer * Server;

    Ui::MainWindow *ui;
    QStackedWidget * stackedWidget;
    WidgetCostMap * widgetCostMap;
    WidgetCamera * widgetCamera;
    WidgetLidar * widgetLidar;

private slots:

    void on_pushButton_Widget_clicked();
    void on_pushButton_Mode_clicked();
};

#endif // MAINWINDOW_H
