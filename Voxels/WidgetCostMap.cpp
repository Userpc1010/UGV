#include "WidgetCostMap.h"
#include "costmap2d.h"
#include "cube.h"
#include "multidrawline.h"
#include <QMouseEvent>
#include <QDebug>
#include <QFile>
#include <QTextStream>
#include <QtMath>

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

QVector3D WidgetCostMap::getPlaneCoordinatesFromMouse(const QVector2D& mousePos)
{
    // Используем существующий метод, но переименуем для ясности
    return screenCoordsToPlaneCoords(mousePos);
}

void WidgetCostMap::onMousePress(QMouseEvent *event)
{
    // ЛЕВАЯ кнопка мыши - начало установки цели
    if (event->button() == Qt::LeftButton) {
        // Получаем координаты на плоскости
        m_GoalRoverPosition = getPlaneCoordinatesFromMouse(QVector2D(event->pos()));

        // Берем текущее вращение реального ровера и извлекаем только рыскание
        QQuaternion realRoverRot = m_Rover->GetRot();
        QVector3D euler = realRoverRot.toEulerAngles();
        m_GoalRoverRotation = QQuaternion::fromEulerAngles(0, euler.y(), 0);

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
            qDebug() << "Goal rover placing started at:" << m_GoalRoverPosition;
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
        QVector3D currentPlanePos = getPlaneCoordinatesFromMouse(QVector2D(event->pos()));

        // Вычисляем направление от позиции цели к текущей позиции мыши
        QVector3D direction = currentPlanePos - m_GoalRoverPosition;
        direction.setY(0); // Проецируем на горизонтальную плоскость

        if (direction.lengthSquared() > 0.001f) {
            direction.normalize();

            // Вычисляем новое вращение для цели
            m_GoalRoverRotation = calculateLookAtRotation(
                m_GoalRoverPosition,
                m_GoalRoverPosition + direction,
                m_GoalRoverRotation);

            // Обновляем вращение объекта цели
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
        qDebug() << "Goal position set:" << m_GoalRoverPosition
                 << "Rotation:" << m_GoalRoverRotation.toEulerAngles();

        event->accept();
        return;
    }

    OGLWidgetBase::onMouseRelease(event);
}

QQuaternion WidgetCostMap::calculateLookAtRotation(const QVector3D& from,
                                                 const QVector3D& to,
                                                 const QQuaternion& currentRotation)
{
    // Вычисляем направление к цели
    QVector3D direction = to - from;

    if (direction.lengthSquared() < 0.001f) {
        return currentRotation; // Оставляем текущее вращение
    }

    direction.normalize();

    // Получаем текущее "вперед" направление цели
    QVector3D currentForward = currentRotation * QVector3D(0.0f, 0.0f, -1.0f);
    currentForward.setY(0); // Проецируем на горизонтальную плоскость
    if (currentForward.lengthSquared() < 0.001f) {
        currentForward = QVector3D(0.0f, 0.0f, -1.0f);
    }
    currentForward.normalize();

    // Вычисляем ось вращения
    QVector3D axis = QVector3D::crossProduct(currentForward, direction);

    // Если векторы коллинеарны
    if (axis.lengthSquared() < 0.001f) {
        float dot = QVector3D::dotProduct(currentForward, direction);
        if (dot < -0.999f) {
            // Противоположные направления - разворот на 180°
            QQuaternion flip = QQuaternion::fromAxisAndAngle(QVector3D(0,1,0), 180.0f);
            return flip * currentRotation;
        }
        return currentRotation; // Уже смотрит в нужном направлении
    }

    axis.normalize();

    // Вычисляем угол
    float dot = QVector3D::dotProduct(currentForward, direction);
    float angle = qRadiansToDegrees(qAcos(qMin(qMax(dot, -1.0f), 1.0f)));

    // Создаем кватернион для поворота от currentForward к direction
    QQuaternion deltaRotation = QQuaternion::fromAxisAndAngle(axis, angle);

    // Применяем дельта-вращение к текущему вращению
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

void WidgetCostMap::DisplayingCostMap(const QImage &map, QQuaternion rotation)
{
    m_costmap->update_map(map);
    m_Rover->rotate(rotation);
    update();
}

void WidgetCostMap::DrawLines(GLfloat* vertices_buffer, QVector3D color_buffer,
                             unsigned long long counter, uint16_t index)
{
    m_MultiLineRender->Add_Lines(vertices_buffer, color_buffer, counter, index);
    update();
}

void WidgetCostMap::DrawCube(QVector3D pos)
{
    m_cube->add_cube(pos);
    update();
}

void WidgetCostMap::DrawAruco(QVector3D pos)
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
