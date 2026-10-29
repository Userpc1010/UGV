#include "WidgetCamera.h"
#include "camera_3d.h"
#include "heightmaprenderer.h"
#include "multidrawline.h"
#include "cube.h"

WidgetCamera::WidgetCamera(QWidget *parent)
    : OGLWidgetBase(parent)
    , m_HeightMap(new HeightMapRenderer())
    , m_Aruco_mobile(new Cube())
    , m_cube(new Cube())
    , m_MultiLineRender(new MultiDrawLine())
{
}

WidgetCamera::~WidgetCamera()
{
    delete m_HeightMap;
    delete m_Aruco_mobile;
    delete m_cube;
    delete m_MultiLineRender;
}

void WidgetCamera::initializeSpecific()
{
    m_Aruco_mobile->initCube(1.0f, QImage(":/Aruco/4x4_1000-47.png"));
    m_Aruco_mobile->Invisible_AllCube(true);
    m_Aruco_mobile->setApplyMapOffset(false);
    m_HeightMap->init_fast();
}

void WidgetCamera::paintSpecific()
{
    m_HeightMap->draw(&m_Program, function, m_camera->GetPositionCamera());
    m_Aruco_mobile->draw(&m_Program, function);
    m_cube->draw(&m_Program, function);
    m_MultiLineRender->draw_lines(&m_Program, function);
}


void WidgetCamera::DisplayingHeightMap(const float* data,  float min, float max, float targetHeight, QQuaternion rotation)
{
    m_HeightMap->update_map(data, min, max);
    m_Rover->rotate(rotation);
    m_Rover->translate(QVector3D(0.0f, targetHeight, 0.0f));
    update();
}

void WidgetCamera::DrawAruco(QVector3D pos)
{
    m_Aruco_mobile->translation(pos);
    update();
}

void WidgetCamera::Invisible_Aruco(bool invisible)
{
    m_Aruco_mobile->Invisible_AllCube(invisible);
}

void WidgetCamera::Drone_scale(float s)
{
    m_RoverScale = s;
    m_Rover->scale(s);
    update();
}

void WidgetCamera::DrawLines(GLfloat *vertices_buffer, QVector3D color_buffer, uint32_t counter, uint16_t index)
{
    m_MultiLineRender->Add_Lines(vertices_buffer, color_buffer, counter, index);
    update();
}

void WidgetCamera::DrawCube(QVector3D pos)
{
    m_cube->add_cube(pos);
    update();
}

void WidgetCamera::ClearLine()
{
    m_MultiLineRender->Clear_Lines();
    m_cube->delete_cube();
}

void WidgetCamera::updateMapOffset(QVector3D offset)
{
  this->setMapOffset(offset);
}

//void WidgetCamera::onMousePress(QMouseEvent *event)
//{
//    if(event->buttons() == Qt::LeftButton)
//    {
//        auto cam_data = m_camera->GetRaycastPosition();
//        emit RayCastPosition(QVector3D(cam_data[0], cam_data[1], cam_data[2]), QVector2D(cam_data[3], cam_data[4]));
//    }

//    if(event->buttons() == Qt::MidButton)
//    {
//        emit reset();
//    }
//}
