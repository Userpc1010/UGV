#include "astar.h"

constexpr uint8_t Zero = 0;
constexpr uint8_t One  = 1;
constexpr uint8_t Two = 2;
constexpr uint8_t Three = 3;

Astar::Astar(QObject *parent): QObject(parent)
{
    // Устанавливаем ограничители чтобы избежать проверки всей карты
    maxEstimatePath = 1500;
    maxNodes = 6000;
}

Astar::~Astar()
{
 if(ptr_guard_output_vertices_open_node){ptr_guard_output_vertices_open_node = 0; delete[] output_vertices_open_node;}
 if(ptr_guard_output_vertices_close_node){ptr_guard_output_vertices_close_node = 0; delete[] output_vertices_close_node;}
 if(ptr_guard_output_vertices_path){ptr_guard_output_vertices_path = 0; delete[] output_vertices_path;}
 this->deleteLater();
}

PathPoint Astar::NewPathPoint(uint16_t& x, uint16_t& y, uint16_t& z, float pathLenghtFromStart, float heuristicEstimatePathLenght)
{
    PathPoint a;
    a.point[Zero] = x;
    a.point[One] = y;
    a.point[Two] = z;
    a.pathLenghtFromStart = pathLenghtFromStart;
    a.heuristicEstimatePathLenght = heuristicEstimatePathLenght;
    a.cameFrom = nullptr;

    return a;
}

PathPoint Astar::NewPathPoint(uint16_t& x, uint16_t& y, uint16_t& z, float pathLenghtFromStart, float heuristicEstimatePathLenght, const PathPoint& ppoint)
{
    PathPoint a;
    a.point[Zero] = x;
    a.point[One] = y;
    a.point[Two] = z;
    a.pathLenghtFromStart = pathLenghtFromStart;
    a.heuristicEstimatePathLenght = heuristicEstimatePathLenght;

    a.cameFrom = new PathPoint();
    a.cameFrom->heuristicEstimatePathLenght = ppoint.heuristicEstimatePathLenght;
    a.cameFrom->pathLenghtFromStart = ppoint.pathLenghtFromStart;
    a.cameFrom->point[Zero] = ppoint.point[Zero];
    a.cameFrom->point[One] = ppoint.point[One];
    a.cameFrom->point[Two] = ppoint.point[Two];
    a.cameFrom->cameFrom = ppoint.cameFrom;

    return a;
}

int Astar::GetMinEstimate(std::vector<PathPoint>& points)
{
    uint16_t min = Zero;
    for (uint16_t i = Zero; i < points.size(); i++)
    {
        if (points[i].estimateFullPathLenght() < points[Zero].estimateFullPathLenght())
            min = i;
    }
    return min;
}

bool Astar::FinishFounded(std::vector<PathPoint>& points)
{
    for (uint16_t i = Zero; i < points.size(); i++)
    {
        if (points[i].heuristicEstimatePathLenght <= 0.0f)
            return true;
    }
    return false;
}

std::vector<PathPoint> Astar::GetNodeToTarget(std::vector<PathPoint>& points)
{
    std::vector<PathPoint> path;
    uint16_t targetIndex = Zero;
    for (uint16_t i = Zero; i < points.size(); i++)
    {
        if (points[i].heuristicEstimatePathLenght <= 0.0f)
            targetIndex = i;
    }
    PathPoint ppoint;
    ppoint = points[targetIndex];//Нашли первую точку с наименьшим расстоянием

    while (ppoint.pathLenghtFromStart > 0.0f)
    {
        path.push_back(ppoint);
        ppoint = *ppoint.cameFrom; //Лезем в камфром искть откуда она пришла
    }

    return path;
}

bool Astar::InList(const std::vector<PathPoint>&  data, uint16_t& x, uint16_t& y, uint16_t& z)
{
    for (uint16_t i = Zero; i < data.size(); i++ )
    {
        if ((data[i].point[Zero] == x) && (data[i].point[One] == y) && (data[i].point[Two] == z))
            return true;
    }

    return false;
}

bool Astar::InList(const std::priority_queue<PathPoint, std::vector<PathPoint>, CompareNode>& data, uint16_t& x, uint16_t& y, uint16_t& z)
{

    std::priority_queue<PathPoint, std::vector<PathPoint>, CompareNode> tempQueue = data;

    while (!tempQueue.empty()) {

    const PathPoint& point = tempQueue.top();

    if (point.point[0] == x && point.point[One] == y && point.point[Two] == z) return true;

    tempQueue.pop();
    }
    return false;
}

bool Astar::CanStand(uint16_t& x, uint16_t& y, uint16_t& z, uint8_t voxels[16000000]) {
    const uint32_t idx_current = get_value(x, y, z);

    if (voxels[idx_current]) {
        return false;
    }

    const uint32_t idx_y_plus = get_value(x, y + One, z);
    const uint32_t idx_y_minus_1 = get_value(x, y - One, z);
    const uint32_t idx_y_minus_2 = get_value(x, y - Two, z);


    if (voxels[idx_y_plus]) {
        return true;
    }


    if (voxels[idx_y_minus_1] || voxels[idx_y_minus_2]) {
        return false;
    }

    return false;
}

bool Astar::isCube(uint16_t& x, uint16_t& y, uint16_t& z, uint8_t voxels[16000000])
{
   const uint32_t idx_xp = get_value(x + One, y, z);
   const uint32_t idx_xm = get_value(x - One, y, z);
   const uint32_t idx_zp = get_value(x, y, z + One);
   const uint32_t idx_zm = get_value(x, y, z - One);
   const uint32_t idx_xp_zp = get_value(x + One, y, z + One);
   const uint32_t idx_xm_zp = get_value(x - One, y, z + One);
   const uint32_t idx_xm_zm = get_value(x - One, y, z - One);

   return (voxels[idx_xp] ||
           voxels[idx_xm] ||
           voxels[idx_zp] ||
           voxels[idx_zm] ||
           voxels[idx_xp_zp] ||
           voxels[idx_xm_zp] ||
           voxels[idx_xm_zm]);

}

bool Astar::noCube(uint16_t& x, uint16_t& y, uint16_t& z, uint8_t voxels[16000000]) {

    const uint32_t idx_xp = get_value(x + One, y, z);
    const uint32_t idx_xm = get_value(x - One, y, z);
    const uint32_t idx_zp = get_value(x, y, z + One);
    const uint32_t idx_zm = get_value(x, y, z - One);
    const uint32_t idx_xp_zp = get_value(x + One, y, z + One);
    const uint32_t idx_xm_zp = get_value(x - One, y, z + One);
    const uint32_t idx_xm_zm = get_value(x - One, y, z - One);

    return (!voxels[idx_xp] ||
            !voxels[idx_xm] ||
            !voxels[idx_zp] ||
            !voxels[idx_zm] ||
            !voxels[idx_xp_zp] ||
            !voxels[idx_xm_zp] ||
            !voxels[idx_xm_zm]);
}


float Astar::Distance(uint16_t& x, uint16_t& y, uint16_t& z, uint16_t tgt [Three])
{
    //return (abs(src.x() - tgt.x()) + abs(src.y() - tgt.y()) + abs(src.z() - tgt.z())); //Манхетоновское расстояние

    return std::max( std::max(abs(x - tgt[Zero]), abs(y - tgt[One])), std::max(abs(z - tgt[Two]), abs(y - tgt[One]))); //Чебышев

}

float Astar::GetTravelCost(uint16_t x, uint16_t y, uint16_t z, uint8_t voxels[16000000], float &weight){ //значение эвристической функции "расстояние + стоимость" для вершины x

    //по идее тут ещё должна быть проверка на различные условия на крате

    return weight;
}

uint32_t Astar::get_value(const uint16_t x, const uint16_t y, const uint16_t z)
{
  uint32_t index = z * 40000 + y * 400 + x;
  return index;
}

//std::vector<std::vector<PathPoint>> Astar::GetPathToTarget(uint16_t beginPoint [3], uint16_t targetPoint[3], uint8_t voxels[16000000])
//{
//    std::vector<std::vector<PathPoint>> output;

//    std::vector<PathPoint> path;
//    // Список не проверенных нодов
//    std::vector<PathPoint> openPoints;
//    // Список проверенных нодов
//    std::vector<PathPoint> closedPoints;

//    // Добавляем к открытым начальную точку
//    openPoints.push_back(NewPathPoint(beginPoint[0], beginPoint[1], beginPoint[2], 0, Distance(beginPoint[0], beginPoint[1], beginPoint[2], targetPoint)));
//    // закрываем точку
//    closedPoints.push_back(openPoints[0]);
//    // открываем новые точки и удаляем закрытую
//    openPoints = ClosePoint(0, openPoints, closedPoints, targetPoint, voxels);

//    // "Стоп сигнал" для нашего поиска
//    bool stopFlag = true;
//    // Устанавливаем ограничители чтобы избежать проверки всей карты
//    float maxEstimatePath = 1500;
//    int maxNodes = 6000;
//    while (stopFlag)
//    {
//        // Находим индекс точки с минимальным евристическим расстоянием
//        int minIndex = GetMinEstimate(openPoints);

//        if (openPoints.size() > 0)
//            if (openPoints[minIndex].estimateFullPathLenght() < maxEstimatePath)
//            {
//                // закрываем точку
//                closedPoints.push_back(openPoints[minIndex]);
//                // добавляем новые если есть и удаляем minIndex
//                openPoints = ClosePoint(minIndex, openPoints, closedPoints, targetPoint, voxels);
//            }
//            else
//            {
//                // если расстояние достигло пределов поиска
//                // просто закрываем точку без добавления новых
//                closedPoints.push_back(openPoints[minIndex]);
//                openPoints.removeAt(minIndex);
//            }

//        // Функция проверяет найден ли финиш
//        if (FinishFounded(closedPoints))
//        {
//            qDebug()<<"Финиш найден!";
//            path = GetPathToTarget(closedPoints);
//            stopFlag = false; // остановка цикла если найдена цель
//        }

//        if (openPoints.size() <= 0)
//            stopFlag = false; // остановка цикла если открытых точек нет

//        if ((openPoints.size()>= maxNodes) ||(closedPoints.size()>= maxNodes))
//            stopFlag = false; // остановка цикла слишком много нодов
//    }
//    qDebug()<<"Nodes created "<<closedPoints.size();

//     output.push_back(openPoints);

//     output.push_back(closedPoints);

//     output.push_back(path);

//    return output;
//}

//=======================================================================================================================================================================
//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
//----Для прокладки маршрута по спутниковой карте можно убрать проверку наличия кубов под нодой либо делать проверки проходимости от маршрутной линии scr до робота tgt--
//----В будущем будет нужна проверка колизии робота с стенами и потолком,------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
//=======================================================================================================================================================================


void Astar::AddNeighborIfValid(std::priority_queue<PathPoint,std::vector<PathPoint>,CompareNode>& newOpenPoints, std::vector<PathPoint>& closedPoints, const PathPoint& lastPoint,uint16_t& dx, uint16_t& dy, uint16_t& dz, float weight, uint16_t targetPoint [Three], uint8_t voxels[16000000]) {

    uint16_t nx = lastPoint.point[Zero] + dx;
    uint16_t ny = lastPoint.point[One] + dy;
    uint16_t nz = lastPoint.point[Two] + dz;

    if (CanStand(nx, ny, nz, voxels)) {

        if (!InList(newOpenPoints, nx, ny, nz) && !InList(closedPoints, nx, ny, nz)) {
            float gCost = lastPoint.pathLenghtFromStart + GetTravelCost(lastPoint.point[Zero] +  nx, lastPoint.point[One] + ny, lastPoint.point[Two] + nz, voxels, weight);
            float hCost = Distance(nx, ny, nz, targetPoint);

             newOpenPoints.push(NewPathPoint(nx, ny, nz, gCost, hCost, lastPoint));
        }
    }
}

std::priority_queue<PathPoint,std::vector<PathPoint>,CompareNode> Astar::ClosePoint(PathPoint& lastPoint, std::priority_queue<PathPoint,std::vector<PathPoint>,CompareNode>& openPoints, std::vector<PathPoint>& closedPoints, uint16_t targetPoint [Three], uint8_t voxels[16000000])
{
    //std::priority_queue<PathPoint,std::vector<PathPoint>,CompareNode> newOpenPoints = openPoints;

    // если блок может быть занят
    if (CanStand(lastPoint.point[Zero], lastPoint.point[One], lastPoint.point[Two], voxels))
    {
        for (uint8_t i = 0; i < 24; ++i) {
            uint16_t dx = directions[i][Zero];
            uint16_t dy = directions[i][One];
            uint16_t dz = directions[i][Two];

            AddNeighborIfValid(openPoints, closedPoints, lastPoint ,dx ,dy ,dz , Neighbor_weight[i] , targetPoint, voxels);
     }
    }

    return openPoints;
}

void Astar::DrawPath(std::vector<PathPoint> &points,const QVector3D& color, GLfloat * vertices, uint8_t guard, uint32_t size,const uint8_t& index)
{
    if(guard){guard = Zero; delete[] vertices;}

    std::vector<QVector3D> buffer;

    while (!points.empty())
    {
        PathPoint ppoint = points.back();

        points.pop_back();

        if (ppoint.cameFrom != nullptr){

         QVector3D vec_1 = QVector3D(ppoint.point[Zero],ppoint.point[One],ppoint.point[Two]);

         QVector3D vec_2 = QVector3D(ppoint.cameFrom->point[Zero], ppoint.cameFrom->point[One],ppoint.cameFrom->point[Two]);

        // if(vec_1.isNull() || vec_2.isNull()) continue;

         buffer.push_back(vec_1);

         buffer.push_back(vec_2);

       }
    } //есть идея запухнуть размер массива в 0й его элемент что-бы ограничится передачей лишь его самого

    size = buffer.size(); //if(size %2!= 0){buffer.pop_back(); size = buffer.size();}

    vertices = new GLfloat[size * Three]; guard = index;

    for (uint32_t i = Zero; i < size; i++){

     vertices[i * Three] = buffer[i].x();
     vertices[i * Three + One] = -buffer[i].y();
     vertices[i * Three + Two] = -buffer[i].z();
    }

    if(state == Zero) emit DrawLines(vertices, color, size, guard);
}

void Astar::DrawPath(std::priority_queue<PathPoint, std::vector<PathPoint>, CompareNode> &points, const QVector3D& color, GLfloat *vertices, uint8_t guard, uint32_t size,const uint8_t& index)
{
    if(guard){guard = Zero; delete[] vertices;}

    std::vector<QVector3D> buffer;

    while (!points.empty())
    {
        PathPoint ppoint = points.top();

        points.pop();

        if (ppoint.cameFrom != nullptr){

         QVector3D vec_1 = QVector3D(ppoint.point[Zero],ppoint.point[One],ppoint.point[Two]);

         QVector3D vec_2 = QVector3D(ppoint.cameFrom->point[Zero], ppoint.cameFrom->point[One],ppoint.cameFrom->point[Two]);

        // if(vec_1.isNull() || vec_2.isNull()) continue;

         buffer.push_back(vec_1);

         buffer.push_back(vec_2);

       }
    } //есть идея запухнуть размер массива в 0й его элемент что-бы ограничится передачей лишь его самого

    size = buffer.size(); //if(size %2!= 0){buffer.pop_back(); size = buffer.size();}

    vertices = new GLfloat[size * Three]; guard = index;

    for (uint32_t i = Zero; i < size; i++){

     vertices[i * Three] = buffer[i].x();
     vertices[i * Three + One] = -buffer[i].y();
     vertices[i * Three + Two] = -buffer[i].z();
    }

    if(state == Zero) emit DrawLines(vertices, color, size, guard);
}

void Astar::GetPathToTarget(uint16_t * beginPoint, uint16_t * targetPoint, uint8_t * voxels)
{
    std::vector<PathPoint> path;
    // Список не проверенных нодов
    //std::vector<PathPoint> openPoints;
    std::priority_queue<PathPoint,std::vector<PathPoint>,CompareNode> openPoints;
    // Список проверенных нодов
    std::vector<PathPoint> closedPoints;


    // Добавляем к открытым начальную точку
    openPoints.push(NewPathPoint(beginPoint[Zero], beginPoint[One], beginPoint[Two], fZero, Distance(beginPoint[Zero], beginPoint[One], beginPoint[Two], targetPoint)));
    // Находим индекс точки с минимальным евристическим расстоянием
    PathPoint currentNode = openPoints.top();
    // закрываем точку
    closedPoints.push_back(currentNode);
    // открываем новые точки и удаляем закрытую
    openPoints = ClosePoint(currentNode, openPoints, closedPoints, targetPoint, voxels);

    // "Стоп сигнал" для нашего поиска
    stopFlag = true;

    while (stopFlag)
    {
        // Находим индекс точки с минимальным евристическим расстоянием
        //int minIndex = GetMinEstimate(openPoints);
        currentNode = openPoints.top();
        openPoints.pop();


        if (openPoints.size() > Zero){

            if (currentNode.estimateFullPathLenght() < maxEstimatePath)
            {
                // закрываем точку
                closedPoints.push_back(currentNode);
                // добавляем новые если есть и удаляем minIndex
                openPoints = ClosePoint(currentNode, openPoints, closedPoints, targetPoint, voxels);
            }
            else
            {
                // если расстояние достигло пределов поиска
                // просто закрываем точку без добавления новых
                closedPoints.push_back(currentNode);
            }
        }

        // Функция проверяет найден ли финиш
        if (FinishFounded(closedPoints))
        {
            qDebug()<<"Финиш найден!";
            path = GetNodeToTarget(closedPoints);
            stopFlag = false; // остановка цикла если найдена цель
        }

        if (openPoints.size() <= Zero)
            stopFlag = false; // остановка цикла если открытых точек нет

        if ((openPoints.size()>= maxNodes) ||(closedPoints.size()>= maxNodes))
            stopFlag = false; // остановка цикла слишком много нодов
    }
    qDebug()<<"Nodes created "<<closedPoints.size();

    //GLfloat data[6] = {astar_points[0].x()  ,-astar_points[0].y(),-astar_points[0].z(), astar_points[1].x(), -astar_points[1].y(), -astar_points[1].z()};

   //emit PathToTarget(path);

     DrawPath(openPoints, blue,output_vertices_open_node, ptr_guard_output_vertices_open_node, output_vertices_open_node_size, One);
     DrawPath(closedPoints, green, output_vertices_close_node, ptr_guard_output_vertices_close_node, output_vertices_close_node_size, Two);
     DrawPath(path, white, output_vertices_path, ptr_guard_output_vertices_path, output_vertices_path_size, Three);
}

void Astar::GetPathFromGround(std::vector<XYZ> ground, uint8_t *voxels)
{
    std::vector<PathPoint> path;
    // Список не проверенных нодов
     std::priority_queue<PathPoint,std::vector<PathPoint>,CompareNode> openPoints;
    // Список проверенных нодов
    std::vector<PathPoint> closedPoints;

    // "Стоп сигнал" для нашего поиска
    stopFlag = true;
    target_detected = false;

    for( uint16_t i = Zero; !target_detected && i < Two; i++ ){
    for( uint16_t ii = ground.size() - One; !target_detected && ii > Three; ii--){

    uint16_t targetPoint[Three] = {ground[ii].x, ground[ii].y, ground[ii].z};


    // Добавляем к открытым начальную точку
    openPoints.push(NewPathPoint(ground[i].x, ground[i].y, ground[i].z, fZero, Distance(ground[i].x, ground[i].y, ground[i].z, targetPoint)));
    // Находим индекс точки с минимальным евристическим расстоянием
    PathPoint currentNode = openPoints.top();
    // закрываем точку
    closedPoints.push_back(currentNode);
    // открываем новые точки и удаляем закрытую
    openPoints = ClosePoint(currentNode, openPoints, closedPoints, targetPoint, voxels);

    while (stopFlag)
    {
        // Находим индекс точки с минимальным евристическим расстоянием
        //int minIndex = GetMinEstimate(openPoints);
        currentNode = openPoints.top();
        openPoints.pop();


        if (openPoints.size() > Zero){

            if (currentNode.estimateFullPathLenght() < maxEstimatePath)
            {
                // закрываем точку
                closedPoints.push_back(currentNode);
                // добавляем новые если есть и удаляем minIndex
                openPoints = ClosePoint(currentNode, openPoints, closedPoints, targetPoint, voxels);
            }
            else
            {
                // если расстояние достигло пределов поиска
                // просто закрываем точку без добавления новых
                closedPoints.push_back(currentNode);
            }
        }

        // Функция проверяет найден ли финиш
        if (FinishFounded(closedPoints))
        {
            //qDebug()<<"Финиш найден!";
            path = GetNodeToTarget(closedPoints);
            stopFlag = false; // остановка цикла если найдена цель
            target_detected = true;
        }

        if (openPoints.size() <= 0)
            stopFlag = false; // остановка цикла если открытых точек нет

        if ((openPoints.size()>= maxNodes) ||(closedPoints.size()>= maxNodes))
            stopFlag = false; // остановка цикла слишком много нодов
    }
    }
    }

     //qDebug()<<"Nodes created "<<closedPoints.size();


    //GLfloat data[6] = {astar_points[0].x()  ,-astar_points[0].y(),-astar_points[0].z(), astar_points[1].x(), -astar_points[1].y(), -astar_points[1].z()};

    emit PathToTarget(path);

    DrawPath(openPoints, blue, output_vertices_open_node, ptr_guard_output_vertices_open_node, output_vertices_open_node_size, One);
    DrawPath(closedPoints, green, output_vertices_close_node, ptr_guard_output_vertices_close_node, output_vertices_close_node_size, Two);
    DrawPath(path, white, output_vertices_path, ptr_guard_output_vertices_path, output_vertices_path_size, Three);
}

void Astar::WindowState(uint8_t state)
{
    this->state = state;
}

void Astar::manual_points(std::vector<QVector3D>)
{

}
