

#include "openvins.h"
#include "state/State.h"
#include "state/Propagator.h"


OpenVins::OpenVins(std::string config_path, QObject *parent): QObject(parent)
{
    timer = new QTimer(this);

    connect(timer, SIGNAL(timeout()), this, SLOT(futures_image_update()));

    timer->start(33);

    state_plus = Eigen::Matrix<double, 13, 1>::Zero();
    cov_plus = Eigen::Matrix<double, 12, 12>::Zero();

    // Load the config
    auto parser = std::make_shared<ov_core::YamlParser>(config_path);

    parser->parse_config("verbosity", verbosity);
    ov_core::Printer::setPrintLevel(verbosity);

    params.print_and_load(parser);

    params.use_multi_threading_pubs = true;
    params.use_multi_threading_subs = true;

    sys = std::make_shared<ov_msckf::VioManager>(params);

    // Ensure we read in all parameters required
    if (!parser->successful()) {
      PRINT_ERROR(RED "unable to parse all parameters, please fix\n" RESET);
      std::exit(EXIT_FAILURE);
    }

}

OpenVins::~OpenVins()
{
  timer->stop();
  timer->deleteLater();
}

void OpenVins::grab_imu(rs2_vector v_gyro_data, double v_gyro_timestamp,rs2_vector v_accel_data, double v_accel_timestamp)
{
  if((abs(v_accel_timestamp-last_timestamp_acc)<0.001) && (abs(v_gyro_timestamp-last_timestamp_gyro)<0.001)) return;

  last_timestamp_acc = v_accel_timestamp;   last_timestamp_gyro = v_gyro_timestamp;

  ov_core::ImuData message_imu;

  message_imu.timestamp=v_accel_timestamp; //*1e-9;
  Eigen::Matrix<double, 3, 1> wm = {v_gyro_data.x,v_gyro_data.y,v_gyro_data.z};
  Eigen::Matrix<double, 3, 1> am = {v_accel_data.x,v_accel_data.y,v_accel_data.z};
  message_imu.wm = wm;
  message_imu.am = am;

  sys->feed_measurement_imu(message_imu);
}

void OpenVins::grab_stereo_camera(cv::Mat im, cv::Mat imRight, double timestamp, uint16_t width_img, uint16_t height_img)
{  
    if(abs(timestamp-last_timestamp_image)<0.001) return;

    last_timestamp_image = timestamp;

//    cv::imshow("img_l", im);
//    cv::imshow("img_r", imRight);

    // Else lets track this image
    ov_core::CameraData message;
    message.timestamp = timestamp;
    message.sensor_ids.push_back(0);
    message.sensor_ids.push_back(1);
    message.images.push_back(im);
    message.images.push_back(imRight);
    message.masks.push_back(cv::Mat::zeros(cv::Size(width_img, height_img), CV_8UC1));
    message.masks.push_back(cv::Mat::zeros(cv::Size(width_img, height_img), CV_8UC1));

    sys->feed_measurement_camera(message);
}

void OpenVins::input_track(double timestamp)
{
  if(sys->initialized()){

     // Get fast propagate state at the desired timestamp
     std::shared_ptr<ov_msckf::State> state = sys->get_state();
     Eigen::Matrix<double, 13, 1> state_plus_ = Eigen::Matrix<double, 13, 1>::Zero();
     Eigen::Matrix<double, 12, 12> cov_plus_ = Eigen::Matrix<double, 12, 12>::Zero();
     if (sys->get_propagator()->fast_state_propagate(state, timestamp, state_plus_, cov_plus_))
     {
        initOK = true;

        state_plus = state_plus_; cov_plus = cov_plus_;

//       std::cout<<std::endl;

//       // The POSE component (orientation and position)
//       std::cout<<"quat: "<<std::endl;
//       std::cout<<"x: "<< state_plus(0) <<std::endl;
//       std::cout<<"y: "<< state_plus(1) <<std::endl;
//       std::cout<<"z: "<< state_plus(2) <<std::endl;
//       std::cout<<"w: "<< state_plus(3) <<std::endl;
//       std::cout<<"pose: "<<std::endl;
//       std::cout<<"x: "<< state_plus(4) <<std::endl;
//       std::cout<<"y: "<< state_plus(5) <<std::endl;
//       std::cout<<"z: "<< state_plus(6) <<std::endl;

//       // The TWIST component (angular and linear velocities)
//       std::cout<<"linear velocitie: "<<std::endl;
//       std::cout<<"linear x: "<< state_plus(7) <<std::endl; // vel in local frame
//       std::cout<<"linear y: "<< state_plus(8) <<std::endl; // vel in local frame
//       std::cout<<"linear z: "<< state_plus(9) <<std::endl; // vel in local frame
//       std::cout<<"angular velocitie: "<<std::endl;
//       std::cout<<"x: "<< state_plus(10) <<std::endl; // we do not estimate this...
//       std::cout<<"y: "<< state_plus(11) <<std::endl; // we do not estimate this...
//       std::cout<<"z: "<< state_plus(12) <<std::endl; // we do not estimate this...

//       std::cout<<std::endl;
   }

   data.Vel_X = state_plus(7);
   data.Vel_Y = state_plus(8);
   data.Vel_Z = state_plus(9);

   data.X = state_plus(4);
   data.Y = state_plus(5);
   data.Z = state_plus(6);

   data.W = state_plus(3);
   data.I = state_plus(0);
   data.J = state_plus(1);
   data.K = state_plus(2);

   data.timestamp = timestamp;

   emit output_track(data);

  }
}

std::vector<Eigen::Vector3d> OpenVins::output_features_MSCKF()
{
 std::vector<Eigen::Vector3d> feats_msckf = sys->get_good_features_MSCKF();

 return feats_msckf;
}

std::vector<Eigen::Vector3d> OpenVins::output_features_SLAM()
{
 std::vector<Eigen::Vector3d> feats_slam = sys->get_features_SLAM();

 return feats_slam;
}

std::vector<Eigen::Vector3d> OpenVins::output_features_ARUCO()
{
  std::vector<Eigen::Vector3d> feats_aruco = sys->get_features_ARUCO();

  return feats_aruco;
}

void OpenVins::openvins_timestamp(double * data)
{
  data = &sys->get_state()->_timestamp;
}

cv::Mat OpenVins::frame_to_mat(const rs2::frame &f)
{
    using namespace cv;
    using namespace rs2;

    auto vf = f.as<video_frame>();
    const int w = vf.get_width();
    const int h = vf.get_height();

    if (f.get_profile().format() == RS2_FORMAT_BGR8)
    {
        return Mat(Size(w, h), CV_8UC3, (void*)f.get_data(), Mat::AUTO_STEP);
    }
    else if (f.get_profile().format() == RS2_FORMAT_Y16)
    {
        return Mat(Size(w, h), CV_16UC1, (void*)f.get_data(), Mat::AUTO_STEP);
    }
    else if (f.get_profile().format() == RS2_FORMAT_RGB8)
    {
        auto r_rgb = Mat(Size(w, h), CV_8UC3, (void*)f.get_data(), Mat::AUTO_STEP);
        Mat r_bgr;
        cvtColor(r_rgb, r_bgr, COLOR_RGB2BGR);
        return r_bgr;
    }
    else if (f.get_profile().format() == RS2_FORMAT_Z16)
    {
        return Mat(Size(w, h), CV_16UC1, (void*)f.get_data(), Mat::AUTO_STEP);
    }
    else if (f.get_profile().format() == RS2_FORMAT_Y8)
    {
        return Mat(Size(w, h), CV_8UC1, (void*)f.get_data(), Mat::AUTO_STEP);
    }
    else if (f.get_profile().format() == RS2_FORMAT_DISPARITY32)
    {
        return Mat(Size(w, h), CV_32FC1, (void*)f.get_data(), Mat::AUTO_STEP);
    }

    throw std::runtime_error("Frame format is not supported yet!");
}

void OpenVins::futures_image_update()
{
 if(initOK){
  // Get our image of history tracks
  cv::Mat img_history = sys->get_historical_viz_image();

  if (img_history.empty()) return;

  cv::Mat result (120, 480, CV_8UC3, cv::Scalar(10, 100, 150));

  cv::resize(img_history, result, result.size(), 0, 0, cv::INTER_CUBIC);

  emit future_image(result);
 }
}

