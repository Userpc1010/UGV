#ifndef OPENVINS_H
#define OPENVINS_H


#include "core/VioManager.h"
#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include <opencv4/opencv2/opencv.hpp>
#include <QObject>
#include <QTimer>

typedef struct Data
{
 double timestamp;

 double Vel_X = 0.0;
 double Vel_Y = 0.0;
 double Vel_Z = 0.0;

 double X = 0.0;
 double Y = 0.0;
 double Z = 0.0;

 double W = -0.5;
 double I = 0.5;
 double J = -0.5;
 double K = 0.5;
} OpenVins_Data;

class OpenVins : public QObject
{
  Q_OBJECT

public:
    OpenVins(std::string config_path, QObject *parent = nullptr);
    ~OpenVins();

public slots:

    void grab_imu (rs2_vector v_gyro_data, double v_gyro_timestamp,rs2_vector v_accel_data, double v_accel_timestamp);

    void grab_stereo_camera(cv::Mat im, cv::Mat imRight, double timestamp, uint16_t width_img, uint16_t height_img);

    void input_track (double timestamp);

    void openvins_timestamp (double * data);

    void futures_image_update ();

signals:

    void output_track (OpenVins_Data data);

    void future_image (cv::Mat img_history);

public :

    std::vector<Eigen::Vector3d> output_features_MSCKF ();

    std::vector<Eigen::Vector3d> output_features_SLAM ();

    std::vector<Eigen::Vector3d> output_features_ARUCO ();

private:

    cv::Mat frame_to_mat(const rs2::frame& f);

    std::shared_ptr<ov_msckf::VioManager> sys;

    // Verbosity
    std::string verbosity = "INFO";

    // Create our VIO system
    ov_msckf::VioManagerOptions params;

    Eigen::Matrix<double, 13, 1> state_plus;

    Eigen::Matrix<double, 12, 12> cov_plus;

    OpenVins_Data data;

    QTimer * timer;

    double last_timestamp_image = 0.0;
    double last_timestamp_gyro = 0.0;
    double last_timestamp_acc = 0.0;

    bool initOK = false;
};


#endif // OPENVINS_H
