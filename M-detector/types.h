#ifndef TYPES_H
#define TYPES_H

#include <Eigen/Eigen>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>



typedef pcl::PointXYZINormal MD_PointType;
typedef pcl::PointCloud<MD_PointType> PointCloudXYZI;
typedef std::vector<MD_PointType, Eigen::aligned_allocator<MD_PointType>>  PointVector;
typedef Eigen::Vector2f V2F;
typedef Eigen::Vector3d V3D;
typedef Eigen::Matrix3d M3D;
typedef Eigen::Vector3f V3F;
typedef Eigen::Matrix3f M3F;

#define MD(a,b)  Eigen::Matrix<double, (a), (b)>
#define VD(a)    Eigen::Matrix<double, (a), 1>
#define MF(a,b)  Eigen::Matrix<float, (a), (b)>
#define VF(a)    Eigen::Matrix<float, (a), 1>



#endif
