#include "WidgetLidar.h"
#include "pointrender.h"
#include "simpleobject3d.h"

WidgetLidar::WidgetLidar(QWidget *parent)
    : OGLWidgetBase(parent)
    , m_PointRender(new PointRender())
    , m_PointRenderMap(new PointRender())
{
    // Увеличиваем масштаб ровера для режима лидара
    m_RoverScale = 10.0f;
}

WidgetLidar::~WidgetLidar()
{
    delete m_PointRender;
    delete m_PointRenderMap;
}

void WidgetLidar::initializeSpecific()
{
    m_PointRender->init();
    m_PointRenderMap->init();
}

void WidgetLidar::paintSpecific()
{
    m_PointRender->draw(&m_Program, function, 9.0);
    m_PointRenderMap->draw(&m_Program, function, 3.0);
}

void WidgetLidar::DisplayingPoint(GLfloat* vertices_buffer, GLfloat* color_buffer,
                                 unsigned long long counter, QQuaternion rotation,
                                 QVector3D translation)
{
    m_PointRender->translate(vertices_buffer, color_buffer, counter);
    m_Rover->rotate(rotation);
    m_Rover->translate(translation);
    update();
}

void WidgetLidar::DisplayingMapPoint(GLfloat* vertices_buffer, GLfloat* color_buffer,
                                    unsigned long long counter)
{
    m_PointRenderMap->translate(vertices_buffer, color_buffer, counter);
    update();
}
