#ifndef ASTAR_H
#define ASTAR_H

#include <stdlib.h>
#include <cmath>
#include <QObject>
#include "oglwidget.h"

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
  float estimateFullPathLenght ()
  {
    return this->heuristicEstimatePathLenght + this->pathLenghtFromStart;
  }
 // точка из которой пришли сюда
 PathPoint* cameFrom = nullptr;
} PathPoints;

class Astars : public QObject
{
    Q_OBJECT

public:
    Astars(QObject *parent = nullptr);

private:

const float LATERAL_WEIGHT = 1.0f;
const float DIAGONAL_WEIGHT = 1.4f;
const float UP_WEIGHT = 2.0f;
const float DOWN_WEIGHT = 1.8f;
const float UP_DIAGONAL_WEIGHT = (float) (sqrt(3) + UP_WEIGHT);
const float DOWN_DIAGONAL_WEIGHT = (float) (sqrt(3) + DOWN_WEIGHT);

bool stopFlag;
float maxEstimatePath;
int maxNodes;

public:

PathPoint NewPathPoint(uint16_t x, uint16_t y, uint16_t z, float pathLenghtFromStart, float heuristicEstimatePathLenght);

PathPoint NewPathPoint(uint16_t x, uint16_t y, uint16_t z, float pathLenghtFromStart, float heuristicEstimatePathLenght, PathPoint ppoint);

// находит индекс точки ближайшей к точке назначения
int GetMinEstimate( QVector<PathPoint> points);

 // проверяет найдена ли цель
bool FinishFounded( QVector<PathPoint> points);

 // Возвращает список точек от старта до финиша
QVector<PathPoint> GetNodeToTarget( QVector<PathPoint> points);

bool InList (QVector<PathPoint> data, uint16_t x, uint16_t y, uint16_t z);

bool CanStand (uint16_t x, uint16_t y, uint16_t z, uint8_t voxels[16000000]);

bool isCube (uint16_t x, uint16_t y, uint16_t z, uint8_t voxels[16000000]);

bool noCube (uint16_t x, uint16_t y, uint16_t z, uint8_t voxels[16000000]);

float Distance (uint16_t x, uint16_t y, uint16_t z, uint16_t tgt [3]);

float GetTravelCost(uint16_t x, uint16_t y, uint16_t z, uint8_t voxels[16000000], float weight);

uint32_t get_value (const uint16_t x, const uint16_t y, const uint16_t z);

QVector<PathPoint> ClosePoint(int index, QVector<PathPoint> openPoints, QVector<PathPoint> closedPoints, uint16_t targetPoint [3], uint8_t voxels[16000000]);

void OutputPathToTarget(QVector<PathPoint>  openPoints, QVector<PathPoint>  closedPoints, QVector<PathPoint> path);

void DrawPath(QVector<PathPoint> points, QVector3D color);

public slots:

//void GetPathToTarget(uint16_t beginPoint [3], uint16_t targetPoint [3], uint8_t voxels[16000000]);

signals:

void DrawLines(GLfloat* vertices_buffer, QVector3D color_buffer, unsigned long long counter);

};

#endif // ASTAR_H
