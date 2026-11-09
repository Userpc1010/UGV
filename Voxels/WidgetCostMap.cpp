#include "WidgetCostMap.h"
#include "costmap2d.h"
#include "cube.h"
#include "multidrawline.h"
#include <QMouseEvent>
#include <QDebug>
#include <QFile>
#include <QTextStream>
#include <QtMath>
#include <cmath>

WidgetCostMap::WidgetCostMap(QWidget *parent)
    : OGLWidgetBase(parent)
    , m_costmap(new CostmapRander())
    , m_cube(new Cube())
    , m_MultiLineRender(new MultiDrawLine())
    , m_Aruco_mobile(new Cube())
    , m_GoalRover(nullptr)
    , m_GoalRoverPlacing(false)
{
    // Загружаем модель для цели
    loadGoalRoverModel(":/Moon_Rover/14014_Moon_Rover_V1_l1.obj", QImage(":/Moon_Rover/14014_Moon_Rover_V1_Atlas.png"));
}

WidgetCostMap::~WidgetCostMap()
{
    delete m_costmap;
    delete m_cube;
    delete m_MultiLineRender;
    delete m_Aruco_mobile;
    if (m_GoalRover) delete m_GoalRover;
}

void WidgetCostMap::initializeSpecific()
{
    m_Aruco_mobile->initCube(1.0f, QImage(":/Aruco/4x4_1000-47.png"));
    m_Aruco_mobile->Invisible_AllCube(true);
    m_Aruco_mobile->setApplyMapOffset(false);
    m_costmap->init_fast();
}

void WidgetCostMap::paintSpecific()
{
    m_costmap->draw(&m_Program, function);
    m_cube->draw(&m_Program, function);
    m_Aruco_mobile->draw(&m_Program, function);
    m_MultiLineRender->draw_lines(&m_Program, function);

    // Отрисовываем цель, если она существует
    if (m_GoalRover) {
        m_GoalRover->draw(&m_Program, function);
    }
}

Eigen::Vector3f WidgetCostMap::getPlaneCoordinatesFromMouse(const QVector2D& mousePos)
{
   // Используем существующий метод, но переименуем для ясности
   QVector3D p = screenCoordsToPlaneCoords(mousePos);
   return Eigen::Vector3f(p.x(), -p.z(), p.y());
}

void WidgetCostMap::onMousePress(QMouseEvent *event)
{
    // ЛЕВАЯ кнопка мыши - начало установки цели
    if (event->button() == Qt::LeftButton) {
        // Получаем координаты на плоскости
        m_GoalRoverPosition = getPlaneCoordinatesFromMouse(QVector2D(event->pos()));

        // Берем текущее вращение реального ровера и извлекаем только рыскание
        Eigen::Quaternionf realRoverRot = m_Rover->GetRot();
        Eigen::Matrix3f R = realRoverRot.toRotationMatrix();

        // Достаем yaw из матрицы и строим вращение вокруг Y
        float yaw_rad = std::atan2(R(1, 0), R(0, 0));

        m_GoalRoverRotation = Eigen::Quaternionf(Eigen::AngleAxisf(yaw_rad, Eigen::Vector3f::UnitZ()));

        // Создаем объект цели
        if (!m_GoalRoverVertexData.isEmpty() && !m_GoalRoverIndexData.isEmpty()) {
            // Удаляем старую цель, если есть
            if (m_GoalRover) {
                delete m_GoalRover;
                m_GoalRover = nullptr;
            }

            m_GoalRover = new SimpleObject3D(m_GoalRoverVertexData,
                                            m_GoalRoverIndexData,
                                            m_GoalRoverTexture);
            m_GoalRover->translate(m_GoalRoverPosition);
            m_GoalRover->rotate(m_GoalRoverRotation);
            m_GoalRover->scale(m_RoverScale);

            m_GoalRoverPlacing = true; // Включаем режим вращения
            qDebug() << "Goal rover placing started at:"<< m_GoalRoverPosition.x() << m_GoalRoverPosition.y() << m_GoalRoverPosition.z();
        }

        update();
        event->accept();
        return;
    }

    // Средний клик - сброс цели и очистка линий
    if (event->button() == Qt::MiddleButton) {
        // Сброс цели
        if (m_GoalRover) {
            delete m_GoalRover;
            m_GoalRover = nullptr;
        }
        m_GoalRoverPlacing = false;

        // Очистка линий (существующая функциональность)
        m_MultiLineRender->Clear_Lines();
        m_cube->delete_cube();
        update();
        event->accept();
        return;
    }

    // Для правой кнопки - передаем управление базовому классу (вращение камеры)
    OGLWidgetBase::onMousePress(event);
}

void WidgetCostMap::onMouseMove(QMouseEvent *event)
{
    // Вращение цели при зажатой левой кнопке (если мы в режиме установки)
    if (m_GoalRoverPlacing && (event->buttons() & Qt::LeftButton)) {
        // Получаем текущую позицию мыши на плоскости
        Eigen::Vector3f currentPlanePos = getPlaneCoordinatesFromMouse(QVector2D(event->pos()));

        // Направление от цели к мыши в горизонтальной плоскости
        Eigen::Vector3f direction = currentPlanePos - m_GoalRoverPosition;
        direction.z() = 0.0f;

        if (direction.squaredNorm() > 0.001f) {
        direction.normalize();
        m_GoalRoverRotation = calculateLookAtRotation(
        m_GoalRoverPosition,
        m_GoalRoverPosition + direction,
        m_GoalRoverRotation);
        m_GoalRover->rotate(m_GoalRoverRotation);
        update();
        }
        event->accept();
        return;
    }

    // Базовый обработчик для вращения камеры (только при правой кнопке)
    OGLWidgetBase::onMouseMove(event);
}

void WidgetCostMap::onMouseRelease(QMouseEvent *event)
{
    // Отпускание левой кнопки - фиксация цели
    if (event->button() == Qt::LeftButton && m_GoalRoverPlacing) {
        m_GoalRoverPlacing = false;

        // Испускаем сигнал с позицией и вращением цели
        emit goalPositionSet(m_GoalRoverPosition, m_GoalRoverRotation);
        qDebug() << "Goal position set:"
        << m_GoalRoverPosition.x() << m_GoalRoverPosition.y() << m_GoalRoverPosition.z()
        << "Rotation (wxyz):" << m_GoalRoverRotation.w()
        << m_GoalRoverRotation.x() << m_GoalRoverRotation.y() << m_GoalRoverRotation.z();

        event->accept();
        return;
    }

    OGLWidgetBase::onMouseRelease(event);
}

Eigen::Quaternionf WidgetCostMap::calculateLookAtRotation(const Eigen::Vector3f& from,
                                                          const Eigen::Vector3f& to,
                                                          const Eigen::Quaternionf& currentRotation)
{
    // Вычисляем направление к цели
    Eigen::Vector3f direction = to - from;
    if (direction.squaredNorm() < 0.001f) {
        return currentRotation;
    }
    direction.normalize();

    Eigen::Vector3f currentForward = currentRotation * Eigen::Vector3f(0.0f, -1.0f, 0.0f);
    currentForward.z() = 0.0f;
    if (currentForward.squaredNorm() < 0.001f) {
        currentForward = Eigen::Vector3f(0.0f, -1.0f, 0.0f);
    }
    currentForward.normalize();

    Eigen::Vector3f axis = currentForward.cross(direction);
    if (axis.squaredNorm() < 0.001f) {
       float dot = currentForward.dot(direction);
       if (dot < -0.999f) {
       Eigen::Quaternionf flip(Eigen::AngleAxisf(float(M_PI), Eigen::Vector3f::UnitZ()));
       return flip * currentRotation;
       }
       return currentRotation;
    }
    axis.normalize();

    float dot = std::min(std::max(currentForward.dot(direction), -1.0f), 1.0f);
    float angle = std::acos(dot);
    Eigen::Quaternionf deltaRotation(Eigen::AngleAxisf(angle, axis));
    return deltaRotation * currentRotation;
}

bool WidgetCostMap::loadGoalRoverModel(const QString &path, const QImage &img)
{
    QFile objfile(path);
    if(!objfile.exists()) {
        qCritical() << "File not exist:" << path;
        return false;
    }
    if(!objfile.open(QFile::ReadOnly)) {
        qCritical() << "File not opened:" << path;
        return false;
    }

    QTextStream input(&objfile);
    QVector<QVector3D> coords;
    QVector<QVector2D> texturcoords;
    QVector<QVector3D> normals;
    m_GoalRoverVertexData.clear();
    m_GoalRoverIndexData.clear();

    bool ok = true;
    while(!input.atEnd() && ok)
    {
        auto str = input.readLine();
        if(str.isEmpty()) continue;
        auto strlist = str.split(' ');
        strlist.removeAll("");
        auto key = strlist.at(0);

        if(key.toLower() == "v" && strlist.size() > 3)
        {
            coords.append(QVector3D(strlist.at(1).toFloat(&ok),
                                   strlist.at(2).toFloat(&ok),
                                   strlist.at(3).toFloat(&ok)));
        }
        else if(key.toLower() == "vt" && strlist.size() > 2)
        {
            texturcoords.append(QVector2D(strlist.at(1).toFloat(&ok),
                                         strlist.at(2).toFloat(&ok)));
        }
        else if(key.toLower() == "vn" && strlist.size() == 4)
        {
            normals.append(QVector3D(strlist.at(1).toFloat(&ok),
                                    strlist.at(2).toFloat(&ok),
                                    strlist.at(3).toFloat(&ok)));
        }
        else if(key.toLower() == "f")
        {
            for(int i = 1; i < strlist.size(); i++)
            {
                auto v = strlist.at(i).split('/');
                if(v.size() == 3 && !v.at(1).isEmpty() && !v.at(2).isEmpty())
                {
                    m_GoalRoverVertexData.append(VertexData(
                        coords.at(v.at(0).toInt(&ok, 10) - 1),
                        texturcoords.at(v.at(1).toInt(&ok, 10) - 1),
                        normals.at(v.at(2).toInt(&ok, 10) - 1)));
                    m_GoalRoverIndexData.append(static_cast<GLuint>(m_GoalRoverIndexData.size()));
                }
            }
        }
    }

    objfile.close();
    if(!ok) return false;

    m_GoalRoverTexture = img;
    return true;
}

void WidgetCostMap::DisplayingCostMap(const QImage &map, Eigen::Quaternionf rotation)
{
    m_costmap->update_map(map);
    m_Rover->rotate(rotation);
    update();
}

void WidgetCostMap::DrawLines(GLfloat* vertices_buffer, Eigen::Vector3f color_buffer, unsigned long long counter, uint16_t index)
{
    m_MultiLineRender->Add_Lines(vertices_buffer, color_buffer, counter, index);
    update();
}

void WidgetCostMap::DrawCube(Eigen::Vector3f pos)
{
    m_cube->add_cube(pos);
    update();
}

void WidgetCostMap::DrawAruco(Eigen::Vector3f pos)
{
    m_Aruco_mobile->translation(pos);
    update();
}

void WidgetCostMap::Invisible_Aruco(bool invisible)
{
    m_Aruco_mobile->Invisible_AllCube(invisible);
}

void WidgetCostMap::ClearLine()
{
    m_MultiLineRender->Clear_Lines();
    m_cube->delete_cube();
}
