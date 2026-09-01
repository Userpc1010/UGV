#include "rgbd_camera.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>
#include <iostream>
#include <time.h>

extern "C"
void deproject_depth_cuda(uint16_t **serialization_point_1, uint16_t *serialization_point_2, uint32_t * world_counter, uint8_t * voxels, uint8_t * costmap, const rs2_intrinsics & intrin, const uint16_t * depth, float depth_scale, double w, double x, double y, double z, int16_t x_, int16_t y_, int16_t z_);

uint32_t floatBitsToUint_memcpy(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(float));
    return u;
}

float UintBitsToFloat_memcpy(uint32_t u) {
    float f;
    // Copy the raw bytes from the uint32_t to the float variable
    std::memcpy(&f, &u, sizeof(float));
    return f;
}

float pack_rgba_to_float(uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
    uint32_t packed = (r << 24) | (g << 16) | (b << 8) | a;
    return UintBitsToFloat_memcpy(packed);
}

int16_t Processor::signum(int16_t x) {
  return x == 0 ? 0 : x < 0 ? -1 : 1;
}

float Processor::mod(float value, float modulus) {
  return fmod(fmod(value, modulus) + modulus, modulus);
}

float Processor::intbound(float s, float ds) {
  // Find the smallest positive t such that s+t*ds is an integer.
  if (ds < 0) {
    return intbound(-s, -ds);
  } else {
    s = mod(s, 1);
    // problem is now s+t*ds = 1
    return (1 - s) / ds;
  }
}

void Processor::Raycast (const uint16_t start[3], const uint16_t end[3], const uint16_t min[3], const uint16_t max[3], std::vector<XYZ>* output)
{
  //    std::cout << start << ' ' << end << std::endl;
  // From "A Fast Voxel Traversal Algorithm for Ray Tracing"
  // by John Amanatides and Andrew Woo, 1987
  // <arhttp://www.cse.yorku.ca/~amana/resech/grid.pdf>
  // <http://citeseer.ist.psu.edu/viewdoc/summary?doi=10.1.1.42.3443>
  // Extensions to the described algorithm:
  //   • Imposed a distance limit.
  //   • The face passed through to reach the current cube is provided to
  //     the callback.

  // The foundation of this algorithm is a parameterized representation of
  // the provided ray,
  //                    origin + t * direction,
  // except that t is not actually stored; rather, at any given point in the
  // traversal, we keep track of the *greater* t values which we would have
  // if we took a step sufficient to cross a cube boundary along that axis
  // (i.e. change the integer part of the coordinate) in the variables
  // tMaxX, tMaxY, and tMaxZ.

    // Cube containing origin point.
    uint16_t x = start[0];
    uint16_t y = start[1];
    uint16_t z = start[2];
    uint16_t endX = end[0];
    uint16_t endY = end[1];
    uint16_t endZ = end[2];

    // Break out direction vector.
    int16_t dx = endX - x;
    int16_t dy = endY - y;
    int16_t dz = endZ - z;

    float direction_x = (float)(endX - start[0]);
    float direction_y = (float)(endY - start[1]);
    float direction_z = (float)(endZ - start[2]);

    float maxDist  = (direction_x * direction_x) + (direction_y * direction_y) + (direction_z * direction_z);

    // Direction to increment x,y,z when stepping.
    int16_t stepX = signum(dx);
    int16_t stepY = signum(dy);
    int16_t stepZ = signum(dz);

    float fdx = (float)(dx);
     float fdy = (float)(dy);
      float fdz = (float)(dz);

    // See description above. The initial values depend on the fractional
    // part of the origin.
    float tMaxX = intbound((float)(start[0]), fdx);
    float tMaxY = intbound((float)(start[1]), fdy);
    float tMaxZ = intbound((float)(start[2]), fdz);

    // The change in t when taking a step (always positive).
    float tDeltaX = (float)(stepX) / fdx;
    float tDeltaY = (float)(stepY) / fdy;
    float tDeltaZ = (float)(stepZ) / fdz;

  // Avoids an infinite loop.
  if (stepX == 0 && stepY == 0 && stepZ == 0) return;

  while (true) {

      if (x >= min[0] && x < max[0] && y >= min[1] && y < max[1] && z >= min[2] && z < max[2]) {

          XYZ data; data.x = x; data.y = y; data.z = z;

          output->push_back(data);

              float dir_x = (float)(x - start[0]);
              float dir_y = (float)(y - start[1]);
              float dir_z = (float)(z - start[2]);

             float dist  = (dir_x * dir_x) + (dir_y * dir_y) + (dir_z * dir_z);

             if (dist > maxDist) return;

                if (output->size() > 574) {
                  std::cerr << "Error, too many racyast voxels." << std::endl;
                  throw std::out_of_range("Too many raycast voxels");
                }
      }
      else break;

      if (x == endX && y == endY && z == endZ) break;

      // tMaxX stores the t-value at which we cross a cube boundary along the
      // X axis, and similarly for Y and Z. Therefore, choosing the least tMax
      // chooses the closest cube boundary. Only the first case of the four
      // has been commented in detail.
      if (tMaxX < tMaxY) {
        if (tMaxX < tMaxZ) {
          // Update which cube we are now in.
          x += stepX;
          // Adjust tMaxX to the next X-oriented boundary crossing.
          tMaxX += tDeltaX;
        } else {

          z += stepZ;
          tMaxZ += tDeltaZ;
        }
      } else {
        if (tMaxY < tMaxZ) {

          y += stepY;
          tMaxY += tDeltaY;
        } else {

          z += stepZ;
          tMaxZ += tDeltaZ;
        }
      }
  }
}

Processor::Processor(QObject *parent)
{
    qRegisterMetaType< std::vector<std::vector<cv::Point2f>> >("std::vector<std::vector<cv::Point2f>>");

    connect (this, SIGNAL(VoxelsMapOut(const uint16_t*,float)), this, SLOT (VoxelsMapIn(const uint16_t*,float)));

    connect(this, SIGNAL(ArucoTrackerOut(std::vector<std::vector<cv::Point2f>>)), this, SLOT(ArucoTrackerIn(std::vector<std::vector<cv::Point2f>>)));

     //astar = new Astar(this);
     state_lattice_ = new StateLattice(this);

     colorizer = new CostmapColorizer();

    last_pos_x = 0.0;
    last_pos_y = 0.0;
    last_pos_z = 0.0;

    accumulated_dx = 0.0;
    accumulated_dy = 0.0;
    accumulated_dz = 0.0;

    // Last timestamp we visualized at
    last_visualization_timestamp = 0;

    process_rgbd = true;

    world_counter = 0;

    serialization_point_1 = NULL;
    serialization_point_2 = NULL;

   // Create the dictionary from the same dictionary the marker was generated.
//   cv::aruco::Dictionary dir = cv::aruco::getPredefinedDictionary(cv::aruco::PredefinedDictionaryType(cv::aruco::DICT_4X4_50));
//   dictionary = cv::Ptr<cv::aruco::Dictionary>(&dir);
}

Processor::~Processor()
{
 process_rgbd = false;
 pipe.stop();
 delete [] voxels;
 if(sizeof(serialization_point_1)) delete [] serialization_point_1;
 if(sizeof(serialization_point_2)) delete [] serialization_point_2;
 if(ptr_guard_vertices_data){ ptr_guard_vertices_data = false; delete [] vertices_data;}
 if(ptr_guard_color_data){ptr_guard_color_data = false;  delete [] color_data;}
}

void Processor::IMU_Get_Euler_Angle(Quat buff)
{
    float CBn[5] = {0.0f};

    float q0q0 = buff.w * buff.w;

    //x-y-z
    CBn[0] = 2.0f * (q0q0 + buff.z * buff.z) - 1.0f;
    CBn[1] = 2.0f * (-buff.x * +buff.z - buff.w * -buff.y);
    CBn[2] = 2.0f * (-buff.x * buff.w - buff.z * -buff.y);
    //CBn[3] = 2.0f * (q1 * q2 - q0 * q3);
    //CBn[4] = 2.0f * (q0q0 + q2 * q2) - 1.0f;
    CBn[3] = 2.0f * (-buff.y * buff.z + buff.w * -buff.x);
    //CBn[6] = 2.0f * (q1 * q3 + q0 * q2);
    //CBn[7] = 2.0f * (q2 * q3 - q0 * q1);
    CBn[4] = 2.0f * (q0q0 + buff.z * buff.z) - 1.0f;

    //roll
    roll = atan2(-CBn[3], CBn[4]);

    if (roll > Quat_PI ) roll -= Quat_TWOPI;
    if (roll < -Quat_PI) roll += Quat_TWOPI;

    //pitch
    pitch = atan2(-CBn[2], CBn[4]);

    if (pitch >  Quat_PI) pitch -= Quat_TWOPI;
    if (pitch < -Quat_PI) pitch += Quat_TWOPI;

    //yaw
    yaw = atan2(CBn[1], CBn[0]);

    if (yaw >  Quat_PI) yaw -= Quat_TWOPI;
    if (yaw < -Quat_PI) yaw += Quat_TWOPI;

    roll = roll * RADTODEG;
    pitch = pitch * RADTODEG;
    yaw = yaw * RADTODEG;
}

void Processor::RayCastPosition(QVector3D CameraPos, QVector2D CameraRot)
{
  int dist = 0; float x = CameraPos.x(), y = -CameraPos.y(), z = -CameraPos.z();

  int x_last = 0, y_last = 0, z_last = 0;

  while(dist <90){

    dist++;

    x+= sin((CameraRot.y() + 90.0f)/180.0f * PI_) / 3; x = x/ 1.0f;
            y+= -tan((CameraRot.x())/180.0f * PI_) / 3; y = y/ 1.0f;
                 z+= cos((CameraRot.y() + 90.0f)/180.0f * PI_) / 3; z = z / 1.0f;

                 if(voxels[get_value(x,y,z)]){

                     astar_points.push_back(QVector3D(x_last, y_last, z_last));
                     if(state == 0) DrawCube(QVector3D(x_last, -y_last, -z_last));
                     break;
                 }

                 x_last = x; y_last = y; z_last = z;
  }
}

void Processor::goalPositionSet(QVector3D position, QQuaternion rotation)
{
   IMU_Get_Euler_Angle(Quat(odom.w, -odom.y, -odom.z, odom.x));

   Finding_start_end_of_the_map_2d(position.x(), -position.z(), rotation.toEulerAngles().y());
}


void Processor::reset()
{
  astar_points.clear();

  emit ClearLine();
}

void Processor::WindowState(uint8_t state)
{
   this->state = state;
}

void Processor::VoxelsMapIn (const uint16_t * depth, float depth_scale)
{
     if(ptr_guard_vertices_data){ ptr_guard_vertices_data = false; delete [] vertices_data;}
     if(ptr_guard_color_data){ptr_guard_color_data = false;  delete [] color_data;}


//     double x = odom.x_pos * Map_scale; int16_t x_ = x - last_pos_x; /*вперёд */ if((x_ - last_x) != 0){last_pos_x = x; last_x = x_;}

//     double y = odom.y_pos * Map_scale; int16_t y_ = y - last_pos_y;             if((y_ - last_y) != 0){last_pos_y = y; last_y = y_;}

//     double z = odom.z_pos * Map_scale; int16_t z_ = z - last_pos_z; /*высота*/  if((z_ - last_z) != 0){last_pos_z = z; last_z = z_; }

     int16_t delta_cells_x = 0;
     int16_t delta_cells_y = 0;
     int16_t delta_cells_z = 0;

     double delta_x = expRunningAverage( odom.x_pos,  filter_integrate_x, filter_wegiht_x ) - last_pos_x;

     accumulated_dx += delta_x;


     if (fabs(accumulated_dx) >= 0.1) {
         delta_cells_x = static_cast<int16_t>(floor(accumulated_dx * Map_scale));
         accumulated_dx -= delta_cells_x / Map_scale;
     }

     last_pos_x = odom.x_pos;


     double delta_y = expRunningAverage( odom.y_pos,  filter_integrate_y, filter_wegiht_y ) - last_pos_y;


     accumulated_dy += delta_y;


     if (fabs(accumulated_dy) >= 0.1) {
         delta_cells_y = static_cast<int16_t>(floor(accumulated_dy * Map_scale));
         accumulated_dy -= delta_cells_y / Map_scale;
     }

     last_pos_y = odom.y_pos;


     double delta_z = expRunningAverage( odom.z_pos,  filter_integrate_z, filter_wegiht_z ) - last_pos_z;


     accumulated_dz += delta_z;


     if (fabs(accumulated_dz) >= 0.1) {
         delta_cells_z = static_cast<int16_t>(floor(accumulated_dz * Map_scale));
         accumulated_dz -= delta_cells_z / Map_scale;
     }

     last_pos_z = odom.z_pos;


//    qDebug()<<" State_plus "<<x_<<" "<<y_<<" "<<z_;

//    qDebug()<<" Last_pos "<<last_pos_x<<" "<<last_pos_y<<" "<<last_pos_z;

//     deproject_depth_cuda(&serialization_point, &world_counter, voxels, intrinsics_depth, depth, depth_scale, odom.w, -odom.y, -odom.z, odom.x, x_, -y_, -z_);

     deproject_depth_cuda(&serialization_point_1, serialization_point_2, &world_counter, voxels, costmap, intrinsics_depth, depth, depth_scale, odom.w, -odom.y, -odom.z, odom.x, delta_cells_x, -delta_cells_y, -delta_cells_z);

     serialization_point_2 = serialization_point_1; serialization_point_1 = NULL;

     //memset(voxels, 0, 16000000); //COLOUR c; uint16_t min = 29, max = 79;

//   #pragma omp parallel for num_threads(10)

//   for( uint32_t i = 0; i < world_counter; i++)
//   {
//     //if (max < serialization_point[i * 3 + 1]) max = serialization_point[i * 3 + 1]; if (min > serialization_point[i * 3 + 1]) min = serialization_point[i * 3 + 1];
//   }

     //qDebug()<<" World size "<<world_counter;

     vertices_data = new GLfloat [world_counter]; ptr_guard_vertices_data = true;

     color_data = new GLfloat [world_counter]; ptr_guard_color_data = true;


     for(uint32_t i = 0; i < world_counter; i++)
     {
           uint16_t x = serialization_point_2[i * 3];
           uint16_t y = serialization_point_2[i * 3 + 1];
           uint16_t z = serialization_point_2[i * 3 + 2];

           uint32_t buff = get_value(x, y, z);

           //voxels[buff] = 1;

           vertices_data[i] = UintBitsToFloat_memcpy(buff);

            //если вдруг захочется покрасить кубик
           //color_data_int[i] = pack_rgba_to_float(255,0,0,0);
           color_data[i] = 0;

           //vertices_data[i * 3] = x;
           //vertices_data[i * 3 + 1] = -y;
           //vertices_data[i * 3 + 2] = -z;

           //c = GetColour(y, min, max);

           //если вдруг захочется покрасить кубик

//           color_data[i * 3] = 0.0f;
//           color_data[i * 3 + 1] = 0.0f;
//           color_data[i * 3 + 2] = 0.0f;

    }

    QImage img = colorizer->colorizeCostmap(costmap, 400, 400);

    if(state == 2) DisplayingCostMap(img, QQuaternion(odom.w, -odom.y, odom.z, -odom.x));

    if(state == 0) emit DisplayingCubes(vertices_data, color_data, world_counter, QQuaternion(odom.w, -odom.y, odom.z, -odom.x));

    state_lattice_->updateCostmapFromData(costmap);

//   if(astar_points.size() > 1){

   //x = 487 y = -515 z = -531   x= 505 y= -514 z = 544

//    uint16_t point_1 [3] = {astar_points[0].x(),astar_points[0].y(),astar_points[0].z()};    uint16_t point_2 [3] = {astar_points[1].x(),astar_points[1].y(),astar_points[1].z()};

   //astar->GetPathToTarget( point_1, point_2, voxels);

   //emit manual_points(astar_points);

//   astar_points.clear();
//   }

   // Аккумулируем общее смещение
   total_offset_x += delta_cells_x;
   total_offset_y += delta_cells_y;
   total_offset_z += delta_cells_z;

    // Создаем QVector3D
    QVector3D current_offset(
    static_cast<float>(total_offset_x),
    static_cast<float>(total_offset_y),
    static_cast<float>(total_offset_z)
    );

       // Проверяем изменение (сравниваем как целые для надежности)
       if (total_offset_x != last_sent_offset_x ||
           total_offset_y != last_sent_offset_y ||
           total_offset_z != last_sent_offset_z) {

           emit updateMapOffset(current_offset);

           last_sent_offset_x = total_offset_x;
           last_sent_offset_y = total_offset_y;
           last_sent_offset_z = total_offset_z;
       }
}


//void Processor::ArucoTrackerIn(std::vector<std::vector<cv::Point2f>> corners)
//{
//        std::vector<cv::Vec3d> rvecs, tvecs;

//        //cv::aruco::estimatePoseSingleMarkers(corners, 0.065f, camera_matrix, dist_coeffs, rvecs, tvecs);

//        cv::aruco::estimatePoseSingleMarkers(corners, 0.2755f, camera_matrix, dist_coeffs, rvecs, tvecs);

//        //std::cout << "Translation: " << tvecs[0] << "\tRotation: " << rvecs[0] << "\n";

//        Aruco_point[0] = (Rotation_matrix[0] * tvecs[0](0) + Rotation_matrix[3] * tvecs[0](1) + Rotation_matrix[6] * tvecs[0](2));

//        Aruco_point[1] = (Rotation_matrix[1] * tvecs[0](0) + Rotation_matrix[4] * tvecs[0](1) + Rotation_matrix[7] * tvecs[0](2));

//        Aruco_point[2] = (Rotation_matrix[2] * tvecs[0](0) + Rotation_matrix[5] * tvecs[0](1) + Rotation_matrix[8] * tvecs[0](2));

//        Aruco_point[0] += Translation[0];

//        Aruco_point[1] += Translation[1];

//        Aruco_point[2] += Translation[2];

//        Aruco_point[0] *= Map_scale;

//        Aruco_point[1] *= Map_scale;

//        Aruco_point[2] *= Map_scale;

//        if(state == 0) DrawAruco(QVector3D(Aruco_point[0] + 199.0f, -Aruco_point[1] - 49.0f, -Aruco_point[2] - 199.0f));

//        Finding_start_end_of_the_map (Aruco_point[0], Aruco_point[1], Aruco_point[2]);

//}

void Processor::ArucoTrackerIn(std::vector<std::vector<cv::Point2f>> corners)
{
    static const float FOLLOW_DISTANCE = 1.5f;
    static const float SAFETY_DISTANCE = 1.0f;
    static const float HYSTERESIS = 0.2f;

    std::vector<cv::Vec3d> rvecs, tvecs;
    cv::aruco::estimatePoseSingleMarkers(corners, 0.206f, camera_matrix, dist_coeffs, rvecs, tvecs);

    // Получаем позицию маркера
    Aruco_point[0] = (Rotation_matrix[0] * tvecs[0](0) + Rotation_matrix[3] * tvecs[0](1) + Rotation_matrix[6] * tvecs[0](2));
    Aruco_point[1] = (Rotation_matrix[1] * tvecs[0](0) + Rotation_matrix[4] * tvecs[0](1) + Rotation_matrix[7] * tvecs[0](2));
    Aruco_point[2] = (Rotation_matrix[2] * tvecs[0](0) + Rotation_matrix[5] * tvecs[0](1) + Rotation_matrix[8] * tvecs[0](2));

    Aruco_point[0] += Translation[0];
    Aruco_point[1] += Translation[1];
    Aruco_point[2] += Translation[2];

    Aruco_point[0] *= Map_scale;
    Aruco_point[1] *= Map_scale;
    Aruco_point[2] *= Map_scale;

    //std::cout << "Translation: " << tvecs[0] << "\tRotation: " << rvecs[0] << "\n";

    if(state == 0) DrawAruco(QVector3D(Aruco_point[0] + 199.0f, -Aruco_point[1] - 49.0f, -Aruco_point[2] - 199.0f));

    if(state == 2) DrawAruco(QVector3D(Aruco_point[0] + 199.0f, -Aruco_point[1] - 49.0f, -Aruco_point[2] - 199.0f));

    float robot_x = Map_center_X;
    float robot_z = Map_center_Z;
    float marker_x = Aruco_point[0] + Map_center_X;
    float marker_z = Aruco_point[2] + Map_center_Z;

    float dx = marker_x - robot_x;
    float dz = marker_z - robot_z;
    float distance = sqrt(dx*dx + dz*dz);

    // Направление на маркер - РОБОТ ВСЕГДА ДОЛЖЕН СМОТРЕТЬ НА МАРКЕР
    float look_at_marker_yaw = atan2(dz, dx);

    float target_x, target_z;

    if (distance < SAFETY_DISTANCE * Map_scale) {
        // 1. СЛИШКОМ БЛИЗКО: отъезжаем задом, но продолжаем смотреть на маркер
        float back_distance = (SAFETY_DISTANCE * Map_scale - distance) + HYSTERESIS * Map_scale;
        target_x = robot_x - back_distance * (dx / distance);
        target_z = robot_z - back_distance * (dz / distance);
    }
    else if (distance > (FOLLOW_DISTANCE + HYSTERESIS) * Map_scale) {
        // 3. ДАЛЕКО: приближаемся вперед, смотрим на маркер
        target_x = marker_x - FOLLOW_DISTANCE * Map_scale * (dx / distance);
        target_z = marker_z - FOLLOW_DISTANCE * Map_scale * (dz / distance);
    }
    else if (distance < (FOLLOW_DISTANCE - HYSTERESIS) * Map_scale) {
        // 2. СЛИШКОМ БЛИЗКО (но не критично): отъезжаем до нужной дистанции
        float back_distance = (FOLLOW_DISTANCE * Map_scale - distance);
        target_x = robot_x - back_distance * (dx / distance);
        target_z = robot_z - back_distance * (dz / distance);
    }
    else {
        // 2. НОРМАЛЬНОЕ РАССТОЯНИЕ: стоим на месте, смотрим на маркер
        target_x = robot_x;
        target_z = robot_z;
    }

    // Всегда используем yaw = atan2(dz, dx) - смотреть на маркер
    Finding_start_end_of_the_map_2d(target_x, target_z, look_at_marker_yaw);
}


//void Processor::Finding_start_end_of_the_map(float target_pos_x, float target_pos_y, float target_pos_z)
//{
//    ClearLine();

//    std::vector<XYZ> output; XYZ buffer; std::vector<XYZ> ground;

//    uint16_t start[3] = {Map_center_X, Map_center_Y, Map_center_Z};  uint16_t end[3] = {target_pos_x + Map_center_X, target_pos_y + Map_center_Y, target_pos_z + Map_center_Z}; uint16_t min[3] = {0,0,0}; uint16_t max[3] = {Map_X,Map_Y,Map_Z};

////    qDebug()<<" x "<<target_pos_x + Map_center_X<<" y "<<target_pos_y + Map_center_Y<<" z "<< target_pos_z + Map_center_Z;

//      Raycast (start, end, min, max, &output);

//    for (uint16_t i = 0; i < output.size(); i++)
//    {
//     buffer = output[i];

//     for(uint8_t i = 0; i < Map_Y ; i++) if(voxels[get_value(buffer.x, i, buffer.z)]){ buffer.y = i - 1; ground.push_back(buffer); break;}

//    }

//    if(ground.size() > 3){

////       QVector3D point_1 = QVector3D(ground.front().x, -ground.front().y, -ground.front().z);
////       QVector3D point_2 = QVector3D(ground.back().x, -ground.back().y, -ground.back().z);

////       if(state == 0) emit DrawCube(point_1);
////       if(state == 0) emit DrawCube(point_2);

//      astar->GetPathFromGround(ground, voxels);
//    }
//}

void Processor::Finding_start_end_of_the_map_2d(float target_pos_x, float target_pos_y, float target_yaw)
{
    unsigned int start_x = Map_center_X;
    unsigned int start_y = Map_center_Z;
    unsigned int goal_x = static_cast<unsigned int>(target_pos_x);
    unsigned int goal_y = static_cast<unsigned int>(target_pos_y);

    // Проверяем границы цели
    if (goal_x >= Map_X || goal_y >= Map_Z) {
        qWarning() << "Цель выходит за границы карты";
        return;
    }

    // Функция проверки ячейки
    auto is_valid_cell = [this](unsigned int x, unsigned int y) -> bool {
        if (x >= Map_X || y >= Map_Z) return false;
        uint8_t cost = costmap[y * Map_X + x];
        return cost < 253;
    };

    // Вектор для точек луча
    std::vector<std::pair<unsigned int, unsigned int>> ray_points;

    // Собираем точки луча
    auto collect_points = [&ray_points, this](unsigned int offset) {
        ray_points.push_back({offset % Map_X, offset / Map_X});
    };

    // Трассируем луч
    try {
        raytraceLine(collect_points, start_x, start_y, goal_x, goal_y, Map_X);
    } catch (...) {
        qWarning() << "Ошибка при трассировке луча";
        return;
    }

    if (ray_points.empty()) {
        qWarning() << "Луч не содержит точек";
        return;
    }

    // Ищем первую валидную точку от начала
    std::pair<unsigned int, unsigned int> valid_start = {start_x, start_y};
    std::pair<unsigned int, unsigned int> valid_goal = {goal_x, goal_y};

    bool start_found = false;
    bool goal_found = false;

    // Прямой проход: ищем первую валидную точку от начала
    for (size_t i = 0; i < ray_points.size(); i++) {
        const auto& point = ray_points[i];

        // Ищем старт
        if (!start_found && is_valid_cell(point.first, point.second)) {
            valid_start = point;
            start_found = true;
        }

        // Если нашли старт, ищем цель с конца
        if (start_found && !goal_found) {
            // Ищем с конца до текущей позиции
            for (size_t j = ray_points.size() - 1; j >= i; j--) {
                const auto& end_point = ray_points[j];
                if (is_valid_cell(end_point.first, end_point.second)) {
                    valid_goal = end_point;
                    goal_found = true;
                    break;
                }
            }
            if (goal_found) break;
        }
    }

    // Если не нашли обе точки - выходим
    if (!start_found || !goal_found) {
        qWarning() << "Не найдены валидные точки на луче";
        return;
    }

    // Проверяем, что точки разные (хотя это нормально если совпадают)
    if (valid_start == valid_goal) {
        qDebug() << "Старт и цель совпадают после корректировки";
    }

    // Создаем Pose для планирования (используем Eigen::Vector2f)
    Pose start;
    Pose goal;

    // Eigen::Vector2f конструктор принимает (x, y)
    start.position = Eigen::Vector2f(
        static_cast<float>(valid_start.first),
        static_cast<float>(valid_start.second)
    );
    start.orientation = (90.0f + yaw) * DEGTORAD; // Текущий yaw робота в радианах

    goal.position = Eigen::Vector2f(
        static_cast<float>(valid_goal.first),
        static_cast<float>(valid_goal.second)
    );
    goal.orientation = (90.0f + target_yaw) * DEGTORAD; // Заданный yaw цели в радианах

    // Логирование для отладки
    qDebug() << "Скорректированные точки:";
    qDebug() << "  Start: (" << valid_start.first << "," << valid_start.second
             << ") yaw: " << start.orientation;
    qDebug() << "  Goal:  (" << valid_goal.first << "," << valid_goal.second
             << ") yaw: " << goal.orientation;

    // Запускаем планирование
    state_lattice_->asyncGetPathToTarget(start, goal);
}


COLOUR Processor::GetColour(float v, float vmin, float vmax)
{
    COLOUR c = {1.0f,1.0f,1.0f}; // white
      float dv;

      if (v < vmin)
         v = vmin;
      if (v > vmax)
         v = vmax;
      dv = vmax - vmin;

      if (v < (vmin + 0.25f * dv)) {
         c.r = 0.0f;
         c.g = 4.0f * (v - vmin) / dv;
      } else if (v < (vmin + 0.5f * dv)) {
         c.r = 0.0;
         c.b = 1.0f + 4.0f * (vmin + 0.25 * dv - v) / dv;
      } else if (v < (vmin + 0.75f * dv)) {
         c.r = 4.0f * (v - vmin - 0.5f * dv) / dv;
         c.b = 0.0f;
      } else {
         c.g = 1.0f + 4.0f * (vmin + 0.75f * dv - v) / dv;
         c.b = 0.0f;
      }

      return(c);
}

uint32_t Processor::get_value(const uint16_t x, const uint16_t y, const uint16_t z)
{
  uint32_t index = z * 40000 + y * 400 + x;
  return index;
}

double Processor::expRunningAverage(double &newVal, double &filVal, const double &wegiht)
{
  filVal += (newVal - filVal) * wegiht;
  return filVal;
}

rs2_vector Processor::interpolateMeasure(const double target_time, const rs2_vector current_data, const double current_time, const rs2_vector prev_data, const double prev_time)
{
    // If there are not previous information, the current data is propagated
      if(prev_time == 0)
      {
          return current_data;
      }

      rs2_vector increment;
      rs2_vector value_interp;

      if(target_time > current_time) {
          value_interp = current_data;
      }
      else if(target_time > prev_time)
      {
          increment.x = current_data.x - prev_data.x;
          increment.y = current_data.y - prev_data.y;
          increment.z = current_data.z - prev_data.z;

          double factor = (target_time - prev_time) / (current_time - prev_time);

          value_interp.x = prev_data.x + increment.x * factor;
          value_interp.y = prev_data.y + increment.y * factor;
          value_interp.z = prev_data.z + increment.z * factor;

          // zero interpolation
          value_interp = current_data;
      }
      else {
          value_interp = prev_data;
      }

      return value_interp;
}

void Processor::run()
{
    const cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);

    const cv::aruco::DetectorParameters detectorParams = cv::aruco::DetectorParameters();

    cv::FileStorage fs("/home/sencis/build-UGV-Desktop-Debug/UGV_Core/calibration_params.yml", cv::FileStorage::READ);

    fs["camera_matrix"] >> camera_matrix;

    fs["distortion_coefficients"] >> dist_coeffs;

    rs2::context ctx;
    rs2::device_list devices = ctx.query_devices();
    rs2::device selected_device;
    if (devices.size() == 0)
    {
        std::cerr << "No device connected, please connect a RealSense device" << std::endl;
        while(1)sleep(1000);
    }
    else selected_device = devices[0];

    std::vector<rs2::sensor> sensors = selected_device.query_sensors();
    int index = 0;
    // We can now iterate the sensors and print their names
    for (rs2::sensor sensor : sensors)
        if (sensor.supports(RS2_CAMERA_INFO_NAME)) {
            ++index;
            if (index == 1) {
                sensor.set_option(RS2_OPTION_DEPTH_AUTO_EXPOSURE_MODE, RS2_DEPTH_AUTO_EXPOSURE_ACCELERATED);
                sensor.set_option(RS2_OPTION_ENABLE_AUTO_EXPOSURE, 1.0f);
                sensor.set_option(RS2_OPTION_EMITTER_ENABLED, 1.0f); // switch off emitter
            }
            // std::cout << "  " << index << " : " << sensor.get_info(RS2_CAMERA_INFO_NAME) << std::endl;
            if (index == 2){
                // RGB camera (not used here...)
                //sensor.set_option(RS2_OPTION_EXPOSURE,100.f);
            }

            if (index == 3){
                sensor.set_option(RS2_OPTION_ENABLE_MOTION_CORRECTION,0);
            }

        }

  int width_img = 848, height_img = 480;

 // Declare RealSense pipeline, encapsulating the actual device and sensors
 //rs2::pipeline pipe;

 // Create a configuration for configuring the pipeline with a non default profile

 // RGB stream
 config.enable_stream(RS2_STREAM_COLOR, width_img, height_img,  RS2_FORMAT_BGR8, 5);

 // Depth stream
 config.enable_stream(RS2_STREAM_DEPTH, width_img, height_img, RS2_FORMAT_Z16, 5);

 //Stereo
 //config.enable_stream(RS2_STREAM_INFRARED, 1, 848, 480, RS2_FORMAT_Y8, 30);
 //config.enable_stream(RS2_STREAM_INFRARED, 2, 848, 480, RS2_FORMAT_Y8, 30);

 // IMU stream
 //config.enable_stream(RS2_STREAM_ACCEL, RS2_FORMAT_MOTION_XYZ32F, 200);
 //config.enable_stream(RS2_STREAM_GYRO, RS2_FORMAT_MOTION_XYZ32F, 200);

 // IMU callback
 std::mutex imu_mutex;
 //std::condition_variable cond_image_rec;

 double v_gyro_timestamp = 0;
 rs2_vector v_gyro_data;

 double v_accel_timestamp = 0;
 rs2_vector v_accel_data;
 //double prev_accel_timestamp = 0;
 //rs2_vector prev_accel_data;

 uint8_t accel= 0, gyro = 0;

 bool reciv = false;


 double timestamp_image = -1.0;
 uint8_t image_ready = 0;

 // start and stop just to get necessary profile
 rs2::pipeline_profile pipe_profile = pipe.start(config);
 pipe.stop();

 rs2::frameset fsCam;

 auto imu_callback = [&](const rs2::frame& frame)
 {
     std::unique_lock<std::mutex> lock(imu_mutex);

     if(rs2::frameset fs = frame.as<rs2::frameset>())
     {
         fsCam = fs;

         timestamp_image = fs.get_timestamp()*1e-3;
         image_ready = 1;

     }
     if (rs2::motion_frame m_frame = frame.as<rs2::motion_frame>())
     {
         if (m_frame.get_profile().stream_name() == "Gyro")
         {
             gyro = 1;
             // It runs at 200Hz
             v_gyro_data = m_frame.get_motion_data();
             v_gyro_timestamp = m_frame.get_timestamp()*1e-3;

             //algo.process_gyro(v_gyro_data, v_gyro_timestamp);
         }
         if (m_frame.get_profile().stream_name() == "Accel")
         {
             accel = 1;
             // It runs at 200Hz

             //prev_accel_timestamp = v_accel_timestamp;
             //prev_accel_data = v_accel_data;

             v_accel_data = m_frame.get_motion_data();
             v_accel_timestamp = m_frame.get_timestamp()*1e-3;

             //rs2_vector interp_data = interpolateMeasure(v_gyro_timestamp, v_accel_data, v_accel_timestamp, prev_accel_data, prev_accel_timestamp);

             //v_accel_data = interp_data;
             //v_accel_timestamp = v_gyro_timestamp;

             //algo.process_accel(v_accel_data);
         }
     }

     reciv = true;
     lock.unlock();
     //cond_image_rec.notify_all();
 };

   pipe_profile = pipe.start(config, imu_callback);

   cam_stream = pipe_profile.get_stream(RS2_STREAM_DEPTH);

   intrinsics_depth = cam_stream.as<rs2::video_stream_profile>().get_intrinsics();

   voxels = new uint8_t [16000000];

   costmap = new uint8_t [1600000];

//   rs2::stream_profile cam_left = pipe_profile.get_stream(RS2_STREAM_INFRARED, 1);
//   rs2::stream_profile cam_right = pipe_profile.get_stream(RS2_STREAM_INFRARED, 2);

//   rs2::stream_profile imu_stream = pipe_profile.get_stream(RS2_STREAM_GYRO);
//   float* Rbc0 = cam_left.get_extrinsics_to(imu_stream).rotation;
//   float* tbc0 = cam_left.get_extrinsics_to(imu_stream).translation;
//   std::cout << "Tbc (left 1) = " << std::endl;
//   for(int i = 0; i<3; i++){
//       for(int j = 0; j<3; j++)
//           std::cout << Rbc0[i*3 + j] << ", ";
//       std::cout << tbc0[i] << "\n";
//   }

//   std::cout<<std::endl;

//   float* Rbc1 = cam_right.get_extrinsics_to(imu_stream).rotation;
//   float* tbc1 = cam_right.get_extrinsics_to(imu_stream).translation;
//   std::cout << "Tbc (Right 2) = " << std::endl;
//   for(int i = 0; i<3; i++){
//       for(int j = 0; j<3; j++)
//           std::cout << Rbc1[i*3 + j] << ", ";
//       std::cout << tbc1[i] << "\n";
//   }

//   std::cout<<std::endl;

//   float* Rlr = cam_right.get_extrinsics_to(cam_left).rotation;
//   float* tlr = cam_right.get_extrinsics_to(cam_left).translation;
//   std::cout << "T left-right  = " << std::endl;
//   for(int i = 0; i<3; i++){
//       for(int j = 0; j<3; j++)
//           std::cout << Rlr[i*3 + j] << ", ";
//       std::cout << tlr[i] << "\n";
//   }

//   std::cout<<std::endl;

//   rs2_intrinsics intrinsics_left = cam_left.as<rs2::video_stream_profile>().get_intrinsics();
//   width_img = intrinsics_left.width;
//   height_img = intrinsics_left.height;
//   std::cout << "Left camera 1: \n";
//   std::cout << " fx = " << intrinsics_left.fx << std::endl;
//   std::cout << " fy = " << intrinsics_left.fy << std::endl;
//   std::cout << " cx = " << intrinsics_left.ppx << std::endl;
//   std::cout << " cy = " << intrinsics_left.ppy << std::endl;
//   std::cout << " height = " << intrinsics_left.height << std::endl;
//   std::cout << " width = " << intrinsics_left.width << std::endl;
//   std::cout << " Coeff = " << intrinsics_left.coeffs[0] << ", " << intrinsics_left.coeffs[1] << ", " <<
//       intrinsics_left.coeffs[2] << ", " << intrinsics_left.coeffs[3] << ", " << intrinsics_left.coeffs[4] << ", " << std::endl;
//   std::cout << " Model = " << intrinsics_left.model << std::endl;

//   std::cout<<std::endl;

//   rs2_intrinsics intrinsics_right = cam_right.as<rs2::video_stream_profile>().get_intrinsics();
//   width_img = intrinsics_right.width;
//   height_img = intrinsics_right.height;
//   std::cout << "Right camera 2: \n";
//   std::cout << " fx = " << intrinsics_right.fx << std::endl;
//   std::cout << " fy = " << intrinsics_right.fy << std::endl;
//   std::cout << " cx = " << intrinsics_right.ppx << std::endl;
//   std::cout << " cy = " << intrinsics_right.ppy << std::endl;
//   std::cout << " height = " << intrinsics_right.height << std::endl;
//   std::cout << " width = " << intrinsics_right.width << std::endl;
//   std::cout << " Coeff = " << intrinsics_right.coeffs[0] << ", " << intrinsics_right.coeffs[1] << ", " <<
//       intrinsics_right.coeffs[2] << ", " << intrinsics_right.coeffs[3] << ", " << intrinsics_right.coeffs[4] << ", " << std::endl;
//   std::cout << " Model = " << intrinsics_right.model << std::endl;

//   std::cout<<std::endl;

// double timestamp;
// cv::Mat im, imRight;

   cv::Mat Color;

 while(process_rgbd)
 {
   if(reciv) {

//     rs2_vector vGyro;
//     double vGyro_times;
//     rs2_vector vAccel;
//     double vAccel_times;
     rs2::frame depth;
     float depth_scale;

     rs2::frameset fs;
     {
         std::unique_lock<std::mutex> lk(imu_mutex);

         //cond_image_rec.wait(lk);

         if(image_ready == 1){

         fs = fsCam;

//         rs2::video_frame ir_frameL = fs.get_infrared_frame(1);
//         rs2::video_frame ir_frameR = fs.get_infrared_frame(2);

           rs2::video_frame color_frame = fs.get_color_frame();

           Color = cv::Mat(cv::Size(width_img, height_img), CV_8UC3, (void*)(color_frame.get_data()), cv::Mat::AUTO_STEP);

//         im= cv::Mat(cv::Size(width_img, height_img), CV_8UC1, (void*)(ir_frameL.get_data()), cv::Mat::AUTO_STEP);
//         imRight = cv::Mat(cv::Size(width_img, height_img), CV_8UC1, (void*)(ir_frameR.get_data()), cv::Mat::AUTO_STEP);

         // Image
//         timestamp = timestamp_image;

         depth = fs.get_depth_frame();

         depth_scale = fs.get_depth_frame().get_units();

         image_ready = 2;

         }
//         if(gyro == 1 && accel == 1){

         // Copy the IMU data
//         vGyro = v_gyro_data;
//         vGyro_times = v_gyro_timestamp;
//         vAccel = v_accel_data;
//         vAccel_times = v_accel_timestamp;

//         gyro = 2; accel = 2;
//         }

         lk.unlock();
     }

//     if(gyro == 2 && accel == 2){

//      emit grab_imu (vGyro, vGyro_times, vAccel, vAccel_times);

//      emit input_track(vAccel_times);

//      gyro = 0; accel = 0;
//     }

     if(image_ready == 2) {

      //emit grab_stereo_camera(im, imRight, timestamp, width_img, height_img);

      auto depth_data = (uint16_t*)depth.get_data();

      if(first_start) emit VoxelsMapOut (depth_data, depth_scale);

      std::vector<int> ids; bool marker_detected = false;
      std::vector<std::vector<cv::Point2f>> corners, rejectedCandidates;

      cv::aruco::ArucoDetector detector(dictionary, detectorParams);

      detector.detectMarkers(Color, corners, ids, rejectedCandidates);

      //if at least one marker detected

     for (size_t i = 0; i < ids.size(); ++i) if(ids[i] == 47){

      std::vector<std::vector<cv::Point2f>> corner; corner.push_back(corners[i]);

      marker_detected = true;

      if(state == 0) emit Invisible_Aruco(false);

      if(state == 2) emit Invisible_Aruco(false);

      emit ArucoTrackerOut(corners);

      }

     if(!marker_detected) {

       if(state == 0) emit Invisible_Aruco(true);

       if(state == 2) emit Invisible_Aruco(true);
     }

      // if(counter <= 1){ VoxelsMaps(intrinsics_depth.height * intrinsics_depth.width); /*pipe.stop(); */counter = 15; }
      // counter--;

      image_ready = 0;

     }

     reciv = false;

     }
     else usleep(50);
 }

 pipe.stop();
}

//void Processor::output_track(OpenVins_Data data)
//{
//    vio_data = data;
//}

void Processor::odometry_lidar(odometry data)
{
  odom = data;

  first_start = true;

  if(!first_start)
  {
      last_pos_x = odom.x_pos;
      last_pos_y = odom.y_pos;
      last_pos_z = odom.z_pos;

      filter_integrate_x = odom.x_pos;
      filter_integrate_y = odom.y_pos;
      filter_integrate_z = odom.z_pos;

  }
}


