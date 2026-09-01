#ifndef ODOMETRY_H
#define ODOMETRY_H
#include <QObject>
#include "lidarmapping.hpp"
#include "OdomIMUKalmanFilter/OdomAccKalman.h"

class Odometry : public QObject
{

    Q_OBJECT

public:
    Odometry(QObject *parent = nullptr);
    ~Odometry();
private:
    const float bit = 1.0f;
    const float zero = 0.0f;

    GPSAccKalmanFilter_t *kf2;

    float pitch = zero, roll= zero, yaw = zero;

    // Keeps the arrival time of previous gyro frame
    double last_ts_gyro = zero;

    uint16_t counter = 0;

    bool init = false;

    float invSqrtf (float x);

    double invSqrt(double x);

    void AHRSupdateIMU (double sampleFreq, double gx, double gy, double gz, Quat_f data);

    Pose RotationPoint (const ImuConstPtr msg_in);

    void init_filter(double LastTimestampLidar, Pose posotion, Velocity vel);

public:

     double q0 = bit, q1 = zero, q2 = zero, q3 = zero; // quaternion of sensor frame relative to auxiliary frame

    void IMU_Get_Euler_Angle(void);
    void IMU_Get_Euler_Angle2(Quat_f data);

public slots:

    // Function to calculate the change in angle of motion based on data from gyro
    void process_imu(const ImuConstPtr msg_in, Quat_f data);

    void process_odom(double LastTimestampLidar, Pose posotion, Velocity vel);

};

#endif // ODOMETRY_H
