#include "odometry.h"
#include "LivoxData.hpp"

#ifndef PI
#define PI  3.14159265358979323846
#define PI_FL  3.141592f
#endif

#define DEGTORAD 0.0174532925199432957f
#define RADTODEG 57.295779513082320876f

#define Quat_HALFPI 1.5707963267948966192313216916398f
#define Quat_PI 3.1415926535897932384626433832795f
#define Quat_TWOPI 6.283185307179586476925286766559f
#define Quat_TODEG(x) ((x) * 57.2957796f)
#define pow2C(a) ((a) * (a))

Odometry::Odometry(QObject *parent): QObject(parent)
{

}

Odometry::~Odometry()
{

}

float Odometry::invSqrtf(float x) {
    float halfx = 0.5f * x;
    float y = x;
    long i = *(long*)&y;
    i = 0x5f3759df - (i>>1);
    y = *(float*)&i;
    y = y * (1.5f - (halfx * y * y));
    return y;
}

double Odometry::invSqrt(double x) {
    double halfx = 0.5 * x;
    double y = x;
    std::int64_t i = *(std::int64_t*)&y;
    i = 0x5fe6eb50c7b537a9 - (i>>1);
    y = *(double*)&i;
    y = y * (1.5 - (halfx * y * y));
    return y;
}

void Odometry::process_imu(const ImuConstPtr msg_in, Quat_f data)
{
    if(last_ts_gyro /*&& init*/){

        // Compute the difference between arrival times of previous and current gyro frames
        double sampleFreq = ( msg_in.header - last_ts_gyro);

        AHRSupdateIMU(sampleFreq, msg_in.angular_velocity[0]/*Roll*/, msg_in.angular_velocity[1]/*Pitch*/, msg_in.angular_velocity[2]/*Yaw*/, data); // Обьеденяем кватернион от гироскопа и лидара 10Гц и 200Гц

        Pose AccToWorld = RotationPoint(msg_in); //Вращаем кватернионом 200Гц ускорения в мировые координаты из СК ИДУ

        qDebug()<<" quat1 "<<q0<<" "<<q1<<"  "<<q2<<" "<<q3<<" Quat2 "<<data.w<<"  "<<data.x<<"  "<<data.y<<"  "<<data.z;

        //IMU_Get_Euler_Angle2(data); IMU_Get_Euler_Angle();

       // GPSAccKalmanPredict(kf2, msg_in.header, AccToWorld.x, AccToWorld.y, AccToWorld.z); //Интегрируем позицию лидара по времени за счёт ускорения получаем 200Гц

        //Итого получаем позицию + вращение не 10 а 200Гц

    }

    last_ts_gyro = msg_in.header;
}

void Odometry::process_odom(double LastTimestampLidar, Pose posotion, Velocity vel)
{
  if(init){

    double posErr = std::sqrt(pow2C(posotion.x - GPSAccKalmanGetX(kf2)) + pow2C(posotion.y - GPSAccKalmanGetY(kf2)) + pow2C(posotion.z - GPSAccKalmanGetZ(kf2)));

    GPSAccKalmanUpdate(kf2, LastTimestampLidar, posotion.x, posotion.y, posotion.z, vel.x, vel.y, vel.z, std::abs(posErr)); //Сбрасываем дрейф новой позицией от лидара 10Гц
  }
  else
  {
    init_filter(LastTimestampLidar,posotion, vel);  init = true;
  }
}

void Odometry::AHRSupdateIMU(double sampleFreq, double gx, double gy, double gz, Quat_f data) {

    double recipNorm = zero;

    // Integrate rate of change of quaternion
    q0 += ( -q1  * gx  -  q2   *  gy - q3   *  gz) * (0.5 * sampleFreq);
    q1 += (  q0  * gx  +  q2   *  gz - q3   *  gy) * (0.5 * sampleFreq);
    q2 += (  q0  * gy  -  q1   *  gz + q3   *  gx) * (0.5 * sampleFreq);
    q3 += (  q0  * gz  +  q1   *  gy - q2   *  gx) * (0.5 * sampleFreq);

    // Normalise quaternion
    recipNorm = invSqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3);

    q0 *= recipNorm;
    q1 *= recipNorm;
    q2 *= recipNorm;
    q3 *= recipNorm;

    q0 += q0 * 0.999 + data.w * 0.001;
    q1 += q1 * 0.999 + data.x * 0.001;
    q2 += q2 * 0.999 + data.y * 0.001;
    q3 += q3 * 0.999 + data.z * 0.001;

    // Normalise quaternion
    recipNorm = invSqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3);

    q0 *= recipNorm;
    q1 *= recipNorm;
    q2 *= recipNorm;
    q3 *= recipNorm;

}

Pose Odometry::RotationPoint(const ImuConstPtr msg_in)
{
    const float ww = q0 * q0;
    const float xx = q1 * q1;
    const float yy = q2 * q2;
    const float zz = q3 * q3;
    const float wx = q0 * q1;
    const float wy = q0 * q2;
    const float wz = q0 * q3;
    const float xy = q1 * q2;
    const float xz = q1 * q3;
    const float yz = q2 * q3;
    const float two = 2.0f;

    Pose accWorld;

    accWorld.x = ww*msg_in.linear_acceleration[0] + two*wy*msg_in.linear_acceleration[2] - two*wz*msg_in.linear_acceleration[1] + xx*msg_in.linear_acceleration[0] + two*xy*msg_in.linear_acceleration[1] + two*xz*msg_in.linear_acceleration[2] - zz*msg_in.linear_acceleration[0] - yy*msg_in.linear_acceleration[0];

    accWorld.y = two*xy*msg_in.linear_acceleration[0] + yy*msg_in.linear_acceleration[1] + two*yz*msg_in.linear_acceleration[2] + two*wz*msg_in.linear_acceleration[0] - zz*msg_in.linear_acceleration[1] + ww*msg_in.linear_acceleration[1] - two*wx*msg_in.linear_acceleration[2] - xx*msg_in.linear_acceleration[1];

    accWorld.z = two*xz*msg_in.linear_acceleration[0] + two*yz*msg_in.linear_acceleration[1] + zz*msg_in.linear_acceleration[2] - two*wy*msg_in.linear_acceleration[0] - yy*msg_in.linear_acceleration[2] + two*wx*msg_in.linear_acceleration[1] - xx*msg_in.linear_acceleration[2] + ww*msg_in.linear_acceleration[2];
}

void Odometry::init_filter(double LastTimestampLidar, Pose posotion, Velocity vel)
{
    static const double accDev = 0.32;

    static const int low = 800;
      static const int high = 1500;
      double noiseX = RandomBetween2Vals(low, high) / 1000000.0;
      double noiseY = RandomBetween2Vals(low, high) / 1000000.0;
      double noiseZ = RandomBetween2Vals(low, high) / 1000000.0;
      double posErr = 0.0;

      if (rand() & 0x01) return;
      if (rand() & 0x01) return;

      noiseX *= rand() & 0x01 ? -1.0 : 1.0;
      noiseY *= rand() & 0x01 ? -1.0 : 1.0;
      noiseZ *= rand() & 0x01 ? -1.0 : 1.0;

      posErr = std::sqrt(pow2C(posotion.x - (posotion.x + noiseX)) + pow2C(posotion.y - (posotion.y + noiseY)) + pow2C(posotion.z - (posotion.z + noiseZ)));

    kf2 = GPSAccKalmanAlloc(
    posotion.x,
    posotion.y,
    posotion.z,
    vel.x,
    vel.y,
    vel.z,
    accDev,
    posErr,
    LastTimestampLidar);
}

void Odometry::IMU_Get_Euler_Angle()
{
    float CBn[5] = {zero};

    float q0q0 = q0 * q0;

    //x-y-z
    CBn[0] = 2.0f * (q0q0 + q1 * q1) - 1.0f;
    CBn[1] = 2.0f * (q1 * q2 + q0 * q3);
    CBn[2] = 2.0f * (q1 * q3 - q0 * q2);
    //CBn[3] = 2.0f * (q1 * q2 - q0 * q3);
    //CBn[4] = 2.0f * (q0q0 + q2 * q2) - 1.0f;
    CBn[3] = 2.0f * (q2 * q3 + q0 * q1);
    //CBn[6] = 2.0f * (q1 * q3 + q0 * q2);
    //CBn[7] = 2.0f * (q2 * q3 - q0 * q1);
    CBn[4] = 2.0f * (q0q0 + q3 * q3) - 1.0f;

    //roll
    roll = atan2(-CBn[3], CBn[4]);

    if (roll > Quat_PI ) roll -= Quat_TWOPI;
    if (roll < -Quat_PI) roll += Quat_TWOPI;

    //pitch
    pitch = atan2(-CBn[2], CBn[4]);

    if (pitch >  Quat_PI) pitch -= Quat_TWOPI;
    if (pitch < -Quat_PI) pitch += Quat_TWOPI;

    //yaw
    yaw = atan2(CBn[1], -CBn[0]);

    if (yaw >  Quat_PI) yaw -= Quat_TWOPI;
    if (yaw < -Quat_PI) yaw += Quat_TWOPI;

    roll = Quat_TODEG(roll);
    pitch = Quat_TODEG(pitch);
    yaw = Quat_TODEG(yaw);

    qDebug()<<"Pitch "<<pitch<<" Roll "<<roll<<" Yaw "<<yaw;
}

void Odometry::IMU_Get_Euler_Angle2(Quat_f data)
{
    float CBn[5] = {zero};

    float q0q0 = data.w * data.w;

    //x-y-z
    CBn[0] = 2.0f * (q0q0 + data.x * data.x) - 1.0f;
    CBn[1] = 2.0f * (data.x * data.y + data.w * data.z);
    CBn[2] = 2.0f * (data.x * data.z - data.w * data.y);
    //CBn[3] = 2.0f * (q1 * q2 - q0 * q3);
    //CBn[4] = 2.0f * (q0q0 + q2 * q2) - 1.0f;
    CBn[3] = 2.0f * (data.y * data.z + data.w * data.x);
    //CBn[6] = 2.0f * (q1 * q3 + q0 * q2);
    //CBn[7] = 2.0f * (q2 * q3 - q0 * q1);
    CBn[4] = 2.0f * (q0q0 + data.z * data.z) - 1.0f;

    //roll
    roll = atan2(-CBn[3], CBn[4]);

    if (roll > Quat_PI ) roll -= Quat_TWOPI;
    if (roll < -Quat_PI) roll += Quat_TWOPI;

    //pitch
    pitch = atan2(-CBn[2], CBn[4]);

    if (pitch >  Quat_PI) pitch -= Quat_TWOPI;
    if (pitch < -Quat_PI) pitch += Quat_TWOPI;

    //yaw
    yaw = atan2(CBn[1], -CBn[0]);

    if (yaw >  Quat_PI) yaw -= Quat_TWOPI;
    if (yaw < -Quat_PI) yaw += Quat_TWOPI;

    roll = Quat_TODEG(roll);
    pitch = Quat_TODEG(pitch);
    yaw = Quat_TODEG(yaw);

    qDebug()<<"Pitch "<<pitch<<" Roll "<<roll<<" Yaw "<<yaw;
}
