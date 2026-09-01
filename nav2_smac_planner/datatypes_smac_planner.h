#ifndef DATATYPES_SMAC_PLANNER_H
#define DATATYPES_SMAC_PLANNER_H

#include <Eigen/Core>
#include <Eigen/Geometry>

// Аналог geometry_msgs/msg/Pose
struct Pose
{
    Eigen::Vector2f position;
    float orientation;
//    Eigen::Quaternionf orientation;

//    float getYawFromQuaternion() const
//    {
//        Eigen::Matrix3f rotation_matrix = orientation.toRotationMatrix();
//        return std::atan2(rotation_matrix(1, 0), rotation_matrix(0, 0));
//    }
};

struct Path
{
    std::vector<Pose> poses;

};

struct Polygon
{
  std::vector<Eigen::Vector2f> points;
};

struct PolygonStamped
{
  Polygon polygon;
};

struct PoseArray
{
  std::vector<Pose> poses;
};

#endif // DATATYPES_SMAC_PLANNER_H
