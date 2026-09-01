#include <QVector3D>
#include <stdlib.h>
#include <cmath>
#include <vector>
#include <queue>
#include <memory>
#include <cmath>
#include <algorithm>
#include "data.h"
#include "WidgetCamera.h"
#include <QObject>

#ifndef ASTAR_H
#define ASTAR_H

typedef struct PathPoint
{
  public:

  // текущая точка
  uint16_t point [3] = {0};
  // расстояние от старта
  float pathLenghtFromStart = 0;
  // примерное расстояние до цели
  float heuristicEstimatePathLenght = 0;
  // еврестическое расстояние до цели
  float estimateFullPathLenght () const
  {
    return this->heuristicEstimatePathLenght + this->pathLenghtFromStart;
  }
  // точка из которой пришли сюда
  PathPoint* cameFrom = nullptr;
} PathPoints;

struct CompareNode {
    bool operator()(const PathPoint& a, const PathPoint& b) const {
        return a.estimateFullPathLenght() > b.estimateFullPathLenght();
    }
};

 const float LATERAL_WEIGHT = 1.0f;
 const float DIAGONAL_WEIGHT = 1.4f;
 const float UP_WEIGHT = 2.0f;
 const float DOWN_WEIGHT = 1.8f;
 const float UP_DIAGONAL_WEIGHT = (float) (sqrt(3) + UP_WEIGHT);
 const float DOWN_DIAGONAL_WEIGHT = (float) (sqrt(3) + DOWN_WEIGHT);


 const int8_t directions[24][3] {
        {1, 0, 0},   // Move right in x
        {-1, 0, 0},  // Move left in x
        {0, 0, 1},   // Move forward in z
        {0, 0, -1},  // Move backward in z

        {1, 0, 1},   // Move right-forward in xz plane
        {1, 0, -1},  // Move right-backward in xz plane
        {-1, 0, 1},  // Move left-forward in xz plane
        {-1, 0, -1}, // Move left-backward in xz plane

        {1, 1, 0},   // Move right-up in xy plane
        {1, -1, 0},  // Move right-down in xy plane
        {-1, 1, 0},  // Move left-up in xy plane
        {-1, -1, 0}, // Move left-down in xy plane

        {0, 1, 1},   // Move up-forward in yz plane
        {0, 1, -1},  // Move up-backward in yz plane
        {0, -1, 1},  // Move down-forward in yz plane
        {0, -1, -1}, // Move down-backward in yz plane

        {1, 1, 1},   // Move diagonally up-right-forward
        {1, 1, -1},  // Move diagonally up-right-backward
        {1, -1, 1},  // Move diagonally down-right-forward
        {1, -1, -1}, // Move diagonally down-right-backward

        {-1, 1, 1},  // Move diagonally up-left-forward
        {-1, 1, -1}, // Move diagonally up-left-backward
        {-1, -1, 1}, // Move diagonally down-left-forward
        {-1, -1, -1} // Move diagonally down-left-backward
 };

 const float Neighbor_weight[24] {
        LATERAL_WEIGHT,    // Move right in x
        LATERAL_WEIGHT,  // Move left in x
        LATERAL_WEIGHT,   // Move forward in z
        LATERAL_WEIGHT,  // Move backward in z

        DIAGONAL_WEIGHT,   // Move right-forward in xz plane
        DIAGONAL_WEIGHT,  // Move right-backward in xz plane
        DIAGONAL_WEIGHT,  // Move left-forward in xz plane
        DIAGONAL_WEIGHT, // Move left-backward in xz plane

        UP_WEIGHT,   // Move right-up in xy plane
        DOWN_WEIGHT,  // Move right-down in xy plane
        UP_WEIGHT,  // Move left-up in xy plane
        DOWN_WEIGHT, // Move left-down in xy plane

        UP_WEIGHT,   // Move up-forward in yz plane
        UP_WEIGHT,  // Move up-backward in yz plane
        DOWN_WEIGHT,  // Move down-forward in yz plane
        DOWN_WEIGHT, // Move down-backward in yz plane

        UP_DIAGONAL_WEIGHT,   // Move diagonally up-right-forward
        UP_DIAGONAL_WEIGHT,  // Move diagonally up-right-backward
        DOWN_DIAGONAL_WEIGHT,  // Move diagonally down-right-forward
        DOWN_DIAGONAL_WEIGHT, // Move diagonally down-right-backward

        UP_DIAGONAL_WEIGHT,  // Move diagonally up-left-forward
        UP_DIAGONAL_WEIGHT, // Move diagonally up-left-backward
        DOWN_DIAGONAL_WEIGHT, // Move diagonally down-left-forward
        DOWN_DIAGONAL_WEIGHT // Move diagonally down-left-backward
 };

 class Astar : public QObject
 {
   Q_OBJECT

   public:

   Astar(QObject *parent = nullptr);
   ~Astar();

   private:
   bool stopFlag;
   bool target_detected;
   float maxEstimatePath;
   uint16_t maxNodes;

   uint8_t state = 0;
   const float fZero= 0.0f;

   GLfloat *output_vertices_open_node;
   uint32_t output_vertices_open_node_size = 0;
   uint8_t ptr_guard_output_vertices_open_node = 0;

   GLfloat *output_vertices_close_node;
   uint32_t output_vertices_close_node_size = 0;
   uint8_t ptr_guard_output_vertices_close_node = 0;

   GLfloat *output_vertices_path;
   uint32_t output_vertices_path_size = 0;
   uint8_t ptr_guard_output_vertices_path = 0;

   const QVector3D blue = QVector3D(0.0f, 1.0f, 0.0f);
   const QVector3D green = QVector3D(0.0f, 0.0f, 1.0f);
   const QVector3D white = QVector3D(1.0f, 1.0f, 1.0f);

   public:

   PathPoint NewPathPoint(uint16_t& x, uint16_t& y, uint16_t& z, float pathLenghtFromStart, float heuristicEstimatePathLenght);

   PathPoint NewPathPoint(uint16_t& x, uint16_t& y, uint16_t& z, float pathLenghtFromStart, float heuristicEstimatePathLenght, const PathPoint& ppoint);

   // находит индекс точки ближайшей к точке назначения
   int GetMinEstimate( std::vector<PathPoint>& points);

    // проверяет найдена ли цель
   bool FinishFounded( std::vector<PathPoint>& points);

    // Возвращает список точек от старта до финиша
   std::vector<PathPoint> GetNodeToTarget( std::vector<PathPoint>& points);

   bool InList (const std::vector<PathPoint>& data, uint16_t& x, uint16_t& y, uint16_t& z);

   bool InList (const std::priority_queue<PathPoint, std::vector<PathPoint>, CompareNode>& data, uint16_t& x, uint16_t& y, uint16_t& z);

   bool CanStand (uint16_t& x, uint16_t& y, uint16_t& z, uint8_t voxels[16000000]);

   bool isCube (uint16_t& x, uint16_t& y, uint16_t& z, uint8_t voxels[16000000]);

   bool noCube (uint16_t& x, uint16_t& y, uint16_t& z, uint8_t voxels[16000000]);

   float Distance (uint16_t& x, uint16_t& y, uint16_t& z, uint16_t tgt [3]);

   float GetTravelCost(uint16_t x, uint16_t y, uint16_t z, uint8_t voxels[16000000], float &weight);

   uint32_t get_value (const uint16_t x, const uint16_t y, const uint16_t z);

   std::priority_queue<PathPoint,std::vector<PathPoint>,CompareNode> ClosePoint(PathPoint& lastPoint, std::priority_queue<PathPoint,std::vector<PathPoint>,CompareNode>& openPoints, std::vector<PathPoint>& closedPoints, uint16_t targetPoint [3], uint8_t voxels[16000000]);

   void OutputPathToTarget(std::priority_queue<PathPoint,std::vector<PathPoint>,CompareNode>&  openPoints, std::vector<PathPoint>&  closedPoints, std::vector<PathPoint>& path);

   void DrawPath(std::vector<PathPoint>& points, const QVector3D& color, GLfloat * vertices, uint8_t guard, uint32_t size, const uint8_t& index);

   void DrawPath(std::priority_queue<PathPoint,std::vector<PathPoint>,CompareNode>& points, const QVector3D& color, GLfloat * vertices, uint8_t guard, uint32_t size, const uint8_t& index);

   void GetPathToTarget(uint16_t * beginPoint, uint16_t * targetPoint, uint8_t * voxels);

   void GetPathFromGround(std::vector<XYZ> ground, uint8_t * voxels);

   void AddNeighborIfValid(std::priority_queue<PathPoint,std::vector<PathPoint>,CompareNode>& newOpenPoints, std::vector<PathPoint>& closedPoints, const PathPoint& lastPoint, uint16_t& dx, uint16_t& dy, uint16_t& dz, float weight, uint16_t targetPoint [3], uint8_t voxels[16000000]);

   public slots:

   void WindowState (uint8_t state);

   void manual_points (std::vector<QVector3D>);

   signals:

   void DrawCube(QVector3D pos);

   void DrawLines(GLfloat* vertices_buffer, QVector3D color_buffer, unsigned long long counter, uint16_t index);

   void PathToTarget (std::vector<PathPoint> points);
 };

#endif // ASTAR_H
