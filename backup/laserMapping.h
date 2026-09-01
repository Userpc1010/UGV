#ifndef LASERMAPPING_H
#define LASERMAPPING_H
#include <omp.h>
#include <mutex>
#include <math.h>
#include <thread>
#include <fstream>
#include <csignal>
#include <unistd.h>
//#include <Python.h>
#include <so3_math.h>
//#include <ros/ros.h>
#include <Eigen/Core>
#include "IMU_Processing.hpp"
//#include <nav_msgs/Odometry.h>
//#include <nav_msgs/Path.h>
//#include <visualization_msgs/Marker.h>
//#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
//#include <sensor_msgs/PointCloud2.h>
//#include <tf/transform_datatypes.h>
//#include <tf/transform_broadcaster.h>
//#include <geometry_msgs/Vector3.h>
//#include <livox_ros_driver2/CustomMsg.h>
#include "preprocess.h"
#include <ikd-Tree/ikd_Tree.h>
#include "data.h"

void livox_pcl_cbk(const CustomMsg &msg);
void imu_cbk(const ImuConstPtr &msg_in);

#endif // LASERMAPPING_H
