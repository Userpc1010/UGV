#ifndef TOOLS_H
#define TOOLS_H

#include <inttypes.h>
#include <math.h>
#include <QQuaternion>
#include <QVector3D>

#define fir_filterLength 5
#define STOPPER 0                                      /* Smaller than any datum */
#define NUM_READ 15

#define Radius 6371008.8f //Радиус земли километров
#define DEGTORAD 0.0174532925199432957f
#define RADTODEG 57.295779513082320876f
#define kSemimajorAxis 6378137
#define kSemiminorAxis 6356752.3142
#define kFirstEccentricitySquared 6.69437999014 * 0.001
#define kSecondEccentricitySquared 6.73949674228 * 0.001
#define kFlattening 1 / 298.257223563

typedef struct vec3 {

  vec3(float x, float y, float z)
  {
   x_ = x;
   y_ = y;
   z_ = z;
  }

  ~vec3(){}

private:

  float x_ = 0.0f;
  float y_ = 0.0f;
  float z_ = 0.0f;

public:

  float X (){return x_;}

  float Y () {return y_;}

  float Z (){return z_;}

} vec3;



struct float3 {
    float x, y, z;
    float3 operator*(float t)
    {
        return { x * t, y * t, z * t };
    }

    float3 operator-(float t)
    {
        return { x - t, y - t, z - t };
    }

    void operator*=(float t)
    {
        x = x * t;
        y = y * t;
        z = z * t;
    }

    void operator=(float3 other)
    {
        x = other.x;
        y = other.y;
        z = other.z;
    }

    void add(float t1, float t2, float t3)
    {
        x += t1;
        y += t2;
        z += t3;
    }
};

float normalize_180_deg (float set_point)
{
  if (set_point < -180.0f) set_point += 360.0f;
  if (set_point >  180.0f) set_point -= 360.0f;

  return set_point;
}

float normalize_360_deg (float set_point)
{
  if (set_point < 0.0f) set_point += 360.0f;
  if (set_point > 360.0f) set_point -= 360.0f;

  return set_point;
}

float normalize_360_rad (float set_point)
{
  if (set_point < 0.0f) set_point += 6.28318530718f;
  if (set_point > 6.28318530718f) set_point -= 6.28318530718f;

  return set_point;
}

float normalize_180_rad (float set_point)
{
  if (set_point < -3.14159265358979323846f) set_point += 6.28318530718f;
  if (set_point >  3.14159265358979323846f) set_point -= 6.28318530718f;

  return set_point;
}


typedef struct
{
double lat = 0.0;
double lon = 0.0;
} vec_2;

class Tools
{
 public:

    float pow_2 (float data)
    {
     return  data * data;
    }

    void filterUpdateFIR( float * shiftBuf, float newSample)
    {
        // Shift history buffer and push new sample
        for (int16_t i = fir_filterLength - 1; i > 0; i--)

        shiftBuf[i] = shiftBuf[i - 1];

        shiftBuf[0] = newSample;
    }

    float filterApplyFIR( float * shiftBuf, float current_atti ,float commonMultiplier)
    {
       float accum = 0.0f;

       for (int16_t i = 0; i < fir_filterLength; i++)

       accum += (current_atti - shiftBuf[i]) * coeffBuf[i];

       return (accum / fir_filterLength) * commonMultiplier;
    }

    float Exp_filter (float input, float &output, float wegiht)
    {
      output = wegiht * input + (1.0f - wegiht) * output;

      return output;
    }

    float LPF_filter (float input, float &output, float wegiht)
    {
      output = output + wegiht * (input - output);

      return output;
    }

    float moving_average (float avg, float * moving_avg_buf, uint8_t Length_avg)
    {
      moving_avg_buf[0] = avg;
      for ( int i = 1; Length_avg > i; i++ )moving_avg_buf[i] = moving_avg_buf[i - 1];
      for ( int i = 1; Length_avg > i; i++ )avg += moving_avg_buf[i];
      return avg / Length_avg;
    }

    // http://www.movable-type.co.uk/scripts/latlong.html

    double geoDistance (double n_lat, double n_lon, double c_lat, double c_lon) {  //Расстояние
      //const float R = 6371008.0f; // km
      double p1 = c_lat * DEGTORAD;
      double p2 = n_lat * DEGTORAD;
      double dp = (n_lat - c_lat) * DEGTORAD;
      double dl = (n_lon - c_lon) * DEGTORAD;

      double x = sin(dp/2) * sin(dp/2) + cos(p1) * cos(p2) * sin(dl/2) * sin(dl/2);
      double y = 2 * atan2(sqrt(x), sqrt(1-x));

      return Radius * y;
    }

    double CoordLongitudeToMeters(double lon) {
      double distance = geoDistance(0.0, lon, 0.0, 0.0);
      return distance * (lon < 0.0 ? -1.0 : 1.0);
    }

    double CoordLatitudeToMeters(double lat) {
      double distance = geoDistance(lat, 0.0, 0.0, 0.0);
      return distance * (lat < 0.0 ? -1.0 : 1.0);
    }

    float geoBearing (float n_lat, float n_lon, float c_lat, float c_lon) {  //Азимут на цель
      float y = sinf(n_lon - c_lon) * cosf(n_lat);
      float x = cosf(c_lat) * sinf(n_lat) - sinf(c_lat) * cosf(n_lat)* cosf(n_lon - c_lon);

      return atan2(y, x) * RADTODEG; //Результат вида  +180 /-180
    }

    vec_2 Point_Distance_Direct_to_Point(double lon, double lat, float distance, float bering)
    {
      vec_2 pos;

      distance = distance / Radius;

      bering = bering * DEGTORAD;

      lat = lat * DEGTORAD;
      lon = lon * DEGTORAD;

      double Lat = asin( sin(lat) * cos(distance) + cos(lat) * sin(distance) * cos(bering) );
      double Lon = lon + atan2( sin(bering) * sin(distance) * cos(lat), cos(distance) - sin(lat) * sin(Lat));
      Lon = fmod(Lon + 3.0*M_PI, 2.0*M_PI) - M_PI;

      pos.lat = Lat * RADTODEG;
      pos.lon = Lon * RADTODEG;

      return pos;
    }

    vec_2 pointPlusDistanceEast(double lon, double lat, double distance) {
      return Point_Distance_Direct_to_Point(lon, lat, distance, 90.0);
    }

    vec_2 pointPlusDistanceNorth(double lon, double lat, double distance) {
      return Point_Distance_Direct_to_Point(lon, lat, distance, 0.0);
    }
    //////////////////////////////////////////////////////////////////////////

    vec_2 CoordMetersToGeopoint(double lonMeters,
                                double latMeters) {
      vec_2 point = {0.0, 0.0};
      vec_2 pointEast = pointPlusDistanceEast(point.lon, point.lat, lonMeters);
      vec_2 pointNorthEast = pointPlusDistanceNorth(pointEast.lon, pointEast.lat, latMeters);
      return pointNorthEast;
    }


    void geodetic_to_Ecef(const double latitude, const double longitude, const double altitude, double* x,
                         double* y, double* z)
      {
        // Convert geodetic coordinates to ECEF.
        // http://code.google.com/p/pysatel/source/browse/trunk/coord.py?r=22
        double lat_rad = latitude * DEGTORAD;
        double lon_rad = longitude * DEGTORAD;
        double xi = sqrt(1 - kFirstEccentricitySquared * sin(lat_rad) * sin(lat_rad));
        *x = (kSemimajorAxis / xi + altitude) * cos(lat_rad) * cos(lon_rad);
        *y = (kSemimajorAxis / xi + altitude) * cos(lat_rad) * sin(lon_rad);
        *z = (kSemimajorAxis / xi * (1 - kFirstEccentricitySquared) + altitude) * sin(lat_rad);
      }

      void ecef_to_Geodetic(const double x, const double y, const double z, double* latitude,
                         double* longitude, double* altitude)
      {
        // Convert ECEF coordinates to geodetic coordinates.
        // J. Zhu, "Conversion of Earth-centered Earth-fixed coordinates
        // to geodetic coordinates," IEEE Transactions on Aerospace and
        // Electronic Systems, vol. 30, pp. 957-961, 1994.

        double r = sqrt(x * x + y * y);
        double Esq = kSemimajorAxis * kSemimajorAxis - kSemiminorAxis * kSemiminorAxis;
        double F = 54 * kSemiminorAxis * kSemiminorAxis * z * z;
        double G = r * r + (1 - kFirstEccentricitySquared) * z * z - kFirstEccentricitySquared * Esq;
        double C = (kFirstEccentricitySquared * kFirstEccentricitySquared * F * r * r) / pow(G, 3);
        double S = cbrt(1 + C + sqrt(C * C + 2 * C));
        double P = F / (3 * pow((S + 1 / S + 1), 2) * G * G);
        double Q = sqrt(1 + 2 * kFirstEccentricitySquared * kFirstEccentricitySquared * P);
        double r_0 = -(P * kFirstEccentricitySquared * r) / (1 + Q)
            + sqrt(
                0.5 * kSemimajorAxis * kSemimajorAxis * (1 + 1.0 / Q)
                    - P * (1 - kFirstEccentricitySquared) * z * z / (Q * (1 + Q)) - 0.5 * P * r * r);
        double U = sqrt(pow((r - kFirstEccentricitySquared * r_0), 2) + z * z);
        double V = sqrt(
            pow((r - kFirstEccentricitySquared * r_0), 2) + (1 - kFirstEccentricitySquared) * z * z);
        double Z_0 = kSemiminorAxis * kSemiminorAxis * z / (kSemimajorAxis * V);
        *altitude = U * (1 - kSemiminorAxis * kSemiminorAxis / (kSemimajorAxis * V));
        *latitude = atan((z + kSecondEccentricitySquared * Z_0) / r) * RADTODEG;
        *longitude = atan2(y, x) * RADTODEG;
      }


    float rk4 (float accel, float vell, float dt, float * buffer/* 12 */, bool update = false)
        {

              if(update) buffer[1] /*velocity_l */ = vell;

/*velocity */ buffer[0] = buffer[1] /*velocity_l */;  buffer[2]/* k1_v */ = accel;

/*velocity */ buffer[0] = buffer[1] /*velocity_l */ + buffer[2]/* k1_v */ * (dt * 0.5f); buffer[3]/* k2_v */ = accel;

/*velocity */ buffer[0] = buffer[1] /*velocity_l */ + buffer[3]/* k2_v */ * (dt * 0.5f); buffer[4]/* k3_v*/ = accel;

/*velocity */ buffer[0] = buffer[1] /*velocity_l */ + buffer[4]/* k3_v*/ * dt; buffer[5]/* k4_v*/ = accel;

              float v_dt = 1.0f / 6.0f * ( buffer[2]/* k1_v */ + 2.0f * ( buffer[3]/* k2_v */ + buffer[4]/* k3_v*/ ) + buffer[5]/* k4_v*/);

              buffer[1] /*velocity_l */ += v_dt * dt; //Аккумулируем скорость

/*position */ buffer[6] = buffer[7] /*position_l */;  buffer[8] /* k1_p*/ = buffer[1] /*velocity_l */;

/*position */ buffer[6] = buffer[7] /*position_l */ + buffer[8] /* k1_p*/ * (dt *0.5f); buffer[9] /* k2_p*/ = buffer[1] /*velocity_l */;

/*position */ buffer[6] = buffer[7] /*position_l */ + buffer[9] /* k2_p*/ * (dt *0.5f); buffer[10]/* k3_p*/ = buffer[1] /*velocity_l */;

/*position */ buffer[6] = buffer[7] /*position_l */ + buffer[10]/* k3_p*/ * dt; buffer[11]/* k4_p*/ = buffer[1] /*velocity_l */;

              float p_dt = 1.0f / 6.0f * ( buffer[8] /* k1_p*/ + 2.0f * ( buffer[9] /* k2_p*/ + buffer[10]/* k3_p*/ ) + buffer[11]/* k4_p*/ );

              buffer[7] /*position_l */ = p_dt * dt;

              return buffer[7] /*position_l */;
        }

    float find_Median (float newVal, float * buffer, const uint8_t size_buffer) {

      uint8_t count = 0;
      buffer[count] = newVal;

      if ((count < size_buffer - 1) and (buffer[count] > buffer[count + 1])) {
        for (int i = count; i < size_buffer - 1; i++) {
          if (buffer[i] > buffer[i + 1]) {
            float buff = buffer[i];
            buffer[i] = buffer[i + 1];
            buffer[i + 1] = buff;
          }
        }
      } else {
        if ((count > 0) and (buffer[count - 1] > buffer[count])) {
          for (int i = count; i > 0; i--) {
            if (buffer[i] < buffer[i - 1]) {
              float buff = buffer[i];
              buffer[i] = buffer[i - 1];
              buffer[i - 1] = buff;
            }
          }
        }
      }
      if (++count >= size_buffer) count = 0;
      return buffer[(int)size_buffer / 2];
    }

    QVector3D Quaternion_rotate(QQuaternion r, float * v)
    {
        //assert(output != NULL);
        QVector3D result;

        QQuaternion q;

        q.setScalar(r.y()); q.setX(-r.x()); q.setY(-r.scalar()); q.setZ(-r.z());

        //qDebug()<<q;

        float ww = q.scalar() * q.scalar();
        float xx = q.x() * q.x();
        float yy = q.y() * q.y();
        float zz = q.z() * q.z();
        float wx = q.scalar() * q.x();
        float wy = q.scalar() * q.y();
        float wz = q.scalar() * q.z();
        float xy = q.x() * q.y();
        float xz = q.x() * q.z();
        float yz = q.y() * q.z();

        // Formula from http://www.euclideanspace.com/maths/algebra/realNormedAlgebra/quaternions/transforms/index.htm
        // p2.x = w*w*p1.x + 2*y*w*p1.z - 2*z*w*p1.y + x*x*p1.x + 2*y*x*p1.y + 2*z*x*p1.z - z*z*p1.x - y*y*p1.x;
        // p2.y = 2*x*y*p1.x + y*y*p1.y + 2*z*y*p1.z + 2*w*z*p1.x - z*z*p1.y + w*w*p1.y - 2*x*w*p1.z - x*x*p1.y;
        // p2.z = 2*x*z*p1.x + 2*y*z*p1.y + z*z*p1.z - 2*w*y*p1.x - y*y*p1.z + 2*w*x*p1.y - x*x*p1.z + w*w*p1.z;

        result.setX(ww*v[1] + 2*wy*v[0] - 2*wz*v[2] + xx*v[1] + 2*xy*v[2] + 2*xz*v[0] - zz*v[1] - yy*v[1]);

        result.setZ(2*xy*v[1] + yy*v[2] + 2*yz*v[0] + 2*wz*v[1] - zz*v[2] + ww*v[2] - 2*wx*v[0] - xx*v[2]);

        result.setY(2*xz*v[1] + 2*yz*v[2] + zz*v[0] - 2*wy*v[1] - yy*v[0] + 2*wx*v[2] - xx*v[0] + ww*v[0]);

        return result;
    }


    // PT1 Low Pass filter (when no dT specified it will be calculated from the cycleTime)
    float LPF_filterApply(float input, float f_cut, /* default 17Hz, Range 1-50Hz */ float dT)
     {
       // Pre calculate and store RC
       if (!RC) {
        RC = 1.0f / ( 2.0f * (float)M_PI * f_cut );
       }

       state = state + dT / (RC + dT) * (input - state);
       return state;
      }

private:

//FIR фильтр производной расчитывается по пяти точкам взят из INAV 1.1
const int8_t coeffBuf [fir_filterLength] = {5, 2, -8, -2, 3};

float RC = 0;
float state = 0;
};

#endif // TOOLS_H
