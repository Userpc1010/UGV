#include "oglwidget.h"
#include "simpleobject3d.h"
#include "camera_3d.h"
#include "cube.h"
#include "cuberander.h"
#include "costmap2d.h"
#include "drawline.h"
#include "pointrender.h"
#include "multidrawline.h"
#include <QMouseEvent>
#include <QOpenGLContext>
#include <QtMath>

OGLWidget::OGLWidget(QWidget *parent, uint8_t state)
    : QOpenGLWidget(parent)
{
   init_pos = QVector3D(199.0f, -54.0f, -199.0f);
   function = new QOpenGLFunctions_4_5_Core();
   m_camera = new Camera_3D(init_pos);
   m_cube = new Cube();
   m_Aruco_mobile = new Cube();
   m_CubeRander = new CubeRander();
   m_costmap = new CostmapRander();
   m_PointRender = new PointRender();
   m_PointRenderMap = new PointRender();
   m_MultiLineRender = new MultiDrawLine();
   m_Rover = new SimpleObject3D();
   _state = state;
}

OGLWidget::~OGLWidget()
{
   delete m_Rover;
   delete m_camera;
   delete m_cube;
   delete m_Aruco_mobile;
   delete m_CubeRander;
   delete m_costmap;
   delete m_PointRender;
   delete m_PointRenderMap;
   delete function;
   delete m_MultiLineRender;
   this->deleteLater();
}

void OGLWidget::initializeGL()
{
    function->initializeOpenGLFunctions();
    function->glEnable(GL_DEPTH_TEST);
    function->glDepthFunc(GL_LEQUAL);
    function->glEnable(GL_CULL_FACE);
    function->glCullFace(GL_BACK);
    function->glFrontFace(GL_CCW);
    function->glLineWidth(6.0f);
    function->glPointSize(6.0f);

    initShaders();

    if (_state == 0) {

        if(initObj(":/Moon_Rover/14014_Moon_Rover_V1_l1.obj", QImage(":/Moon_Rover/14014_Moon_Rover_V1_Atlas.png")))
        {
          m_Rover->translate(init_pos);

          m_Rover->scale(scale);
        }

         m_Aruco_mobile->initCube(1.0f, QImage(":/Aruco/4x4_1000-47.png"));

         m_costmap->init_fast();
    }

    if (_state == 1){

        if(initObj(":/Moon_Rover/14014_Moon_Rover_V1_l1.obj", QImage(":/Moon_Rover/14014_Moon_Rover_V1_Atlas.png")))
        {
          m_Rover->translate(init_pos);

          m_Rover->scale(scale);
        }

         m_Aruco_mobile->initCube(1.0f, QImage(":/Aruco/4x4_1000-47.png"));

         m_CubeRander->init_fast();

    }
    if (_state == 2){

        if(initObj(":/Moon_Rover/14014_Moon_Rover_V1_l1.obj", QImage(":/Moon_Rover/14014_Moon_Rover_V1_Atlas.png")))
        {
          m_Rover->translate(init_pos);

          m_Rover->scale(10.0f);
        }

         m_PointRender->init();

         m_PointRenderMap->init();
    }
}

void OGLWidget::resizeGL(int w, int h)
{
    float aspect = w / (h? static_cast<float>(h) : 1);
    m_PojectionMatrix.setToIdentity();
    m_PojectionMatrix.perspective(80, aspect, 1.0f, 2000.0f);
}

void OGLWidget::paintGL()
{
    function->glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    function->glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    m_Program.bind();
    m_Program.setUniformValue("u_projectionMatrix", m_PojectionMatrix);
    m_CubeRander->draw(&m_Program, function, m_camera->GetPositionCamera());
    m_costmap->draw(&m_Program, function);
    m_PointRender->draw(&m_Program, function, 9.0);
    m_PointRenderMap->draw(&m_Program, function, 3.0);
    m_MultiLineRender->draw_lines(&m_Program, function);
    m_cube->draw(&m_Program, function);
    m_Aruco_mobile ->draw(&m_Program, function);
    m_Rover->draw(&m_Program, function);
    m_camera->draw(&m_Program);
    m_Program.release();
}

void OGLWidget::mousePressEvent(QMouseEvent *event)
{
    if(event->buttons() == Qt::LeftButton)
    {
      cam_data = m_camera->GetRaycastPosition();

      RayCastPosition(QVector3D(cam_data[0], cam_data[1], cam_data[2]), QVector2D(cam_data[3],cam_data[4]));

       if (_state == 0) {

        m_cube->add_cube(screenCoordsToPlaneCoords(QVector2D(event->localPos())));

        update();

       }

    }

    if(event->buttons() == Qt::MidButton)
    {
     reset();

     m_cube->delete_cube();

     m_MultiLineRender->Clear_Lines();
    }

    if(event->buttons() == Qt::RightButton)
    {

     lastX =  event->localPos().x();
     lastY =  event->localPos().y();

    }

    update();

    event->accept();
}

void OGLWidget::mouseReleaseEvent(QMouseEvent *event)
{
    if(event->buttons() == Qt::RightButton)
    {
     camera = true;
    }
    event->accept();
}


void OGLWidget::mouseMoveEvent(QMouseEvent *event)
{

    if(event->buttons() == Qt::RightButton)
    {
        if (camera){

          lastX = event->localPos().x();
          lastY = event->localPos().y();
          camera = false;
        }
         double xoffset = event->localPos().x() - lastX;
         double yoffset = lastY - event->localPos().y();

         lastX = event->localPos().x();
         lastY = event->localPos().y();

         m_camera->rotate_camera(xoffset, yoffset);
    }

    update();

    event->accept();

}

void OGLWidget::wheelEvent(QWheelEvent *event)
{
  if ( event->delta() > 0 ) m_camera->camera_zoom(true);
  if ( event->delta() < 0 ) m_camera->camera_zoom(false);

  update();

  event->accept();
}

void OGLWidget::keyPressEvent(QKeyEvent *event)
{
     //qDebug()<< event->key();
     if (event->key() == Qt::Key_W) m_camera->Front_move();
     if (event->key() == Qt::Key_S) m_camera->Back_move();
     if (event->key() == Qt::Key_A) m_camera->right_move();
     if (event->key() == Qt::Key_D) m_camera->left_move();
     if (event->key() == Qt::Key_Space)m_camera->Up_move();
     if (event->key() == Qt::Key_Alt) m_camera->Down_move();

     update();

     event->accept();
}

void OGLWidget::DisplayingCubes(GLfloat* vertices_buffer, GLfloat* color_buffer, unsigned long long counter, QQuaternion rotation)
{
 m_CubeRander->translate(vertices_buffer, color_buffer, counter);

 m_Rover->rotate(rotation);

 update();
}

void OGLWidget::DisplayingPoint(GLfloat *vertices_buffer, GLfloat *color_buffer, unsigned long long counter, QQuaternion rotation, QVector3D translation)
{
  m_PointRender->translate(vertices_buffer, color_buffer, counter);

  m_Rover->rotate(rotation);

  m_Rover->translate(translation);

  update();
}

void OGLWidget::DisplayingMapPoint(GLfloat *vertices_buffer, GLfloat *color_buffer, unsigned long long counter)
{

    m_PointRenderMap->translate(vertices_buffer, color_buffer, counter);

    update();
}

void OGLWidget::DisplayingCostMap(const QImage &map)
{
  m_costmap->update_map(map);

  update();
}

void OGLWidget::DrawLines(GLfloat *vertices_buffer, QVector3D color_buffer, unsigned long long counter, uint16_t index)
{
  m_MultiLineRender->Add_Lines(vertices_buffer, color_buffer, counter, index);

  update();
}

void OGLWidget::DrawCube(QVector3D pos)
{
  m_cube->add_cube(pos);

  update();
}

void OGLWidget::DrawAruco(QVector3D pos)
{
 m_Aruco_mobile->translation(pos);

 update();
}

void OGLWidget::OffsetCube(QVector3D pos)
{
  m_cube->Offset_AllCube(pos);

  update();
}

void OGLWidget::Invisible_Aruco(bool invisible)
{
  m_Aruco_mobile->Invisible_AllCube(invisible);
}

void OGLWidget::ClearLine()
{
   m_MultiLineRender->Clear_Lines();

   m_cube->delete_cube();
}

QVector3D OGLWidget::screenCoordsToPlaneCoords(const QVector2D &mouse_pose)
{
  QVector4D mouse_norm(2.0f * mouse_pose.x()/width() - 1.0f, -2.0f * mouse_pose.y() / height() + 1.0f, -1.0f, 1.0f );

  QMatrix4x4 offset_matrix = m_PojectionMatrix.inverted();

  QVector4D mouse_to_cam_coord ((offset_matrix * mouse_norm).toVector2D(), -1.0f, 0.0f);

  offset_matrix = m_camera->ViewMatrix().inverted();

  QVector3D Ray_Direction ((offset_matrix * mouse_to_cam_coord).toVector3D().normalized());

  QVector3D CamPosWorld ((offset_matrix * QVector4D(0.0f, 0.0f, 0.0f, 1.0f)).toVector3D());

  //Ax + By + Cz + D общее уравнение плоскости
  //Norm = (A, B, C) Нахождение вектора нормали к плоскости
  // P * N - P0 * N = 0  Расчётуравнения плоскости в векторонм виде,  D = P0 * N Растояние до центра системы координат = Любая точка в плоскости P0 * N нормаль
  // O + D * t Уравнение луча в векторном виде (описывает все точки луча) это расчёт P, O это точка откуда исходит луч, t это число
  // (O + D * t) * N - P0 * N = 0 Уравнение плоскости в векторонм виде
  // t = (P0 * N - O * N) / (D * N)
  // result = O + D * t // Точка пересечения плоскости

  QVector3D Normal_to_Plane (0.0, 1.0f, 0.0);
  QVector3D Center_of_coord_plane (0.0f, -53.0f, 0.0f);

  float t = (QVector3D::dotProduct(Center_of_coord_plane, Normal_to_Plane) - QVector3D::dotProduct(CamPosWorld, Normal_to_Plane)) / QVector3D::dotProduct(Ray_Direction, Normal_to_Plane);

  return CamPosWorld + Ray_Direction * t;
}

void OGLWidget::initShaders()
{
    if(! m_Program.addShaderFromSourceFile(QOpenGLShader::Vertex, ":/vertshader.vsh"))
        this->close();
    if(! m_Program.addShaderFromSourceFile(QOpenGLShader::Fragment, ":/fragshader.fsh"))
        this->close();

    if(! m_Program.link())
        this->close();
}

bool OGLWidget::initObj(const QString &path, const QImage &img)
{
    QFile objfile(path);

    if(! objfile.exists()) { qCritical() << "File not exist:" << path; return false; }
    if(! objfile.open(QFile::ReadOnly))  { qCritical() << "File not opened:" << path; return false; }

    QTextStream input(&objfile);
    QVector<QVector3D> coords;
    QVector<QVector2D> texturcoords;
    QVector<QVector3D> normals;

    QVector<VertexData> vertexes;
    QVector<GLuint> indexes;

    qDebug() << "Reading" << path << "...";

    bool ok = true;
    while(!input.atEnd() && ok)
    {
        auto str = input.readLine(); if(str.isEmpty()) continue;
        auto strlist = str.split(' '); strlist.removeAll("");
        auto key = strlist.at(0);

        if (key == "#") { qDebug() << str; }
        else if(key == "mtllib")
        {
            qDebug() << str;
        }
        else if(key.toLower() == "o")
        {
            qDebug() << str;
        }
        else if(key.toLower() == "g")
        {
            qDebug() << str;
        }
        else if(key.toLower() == "s")
        {
            qDebug() << str;
        }
        else if(key.toLower() == "v")
        {
            if(strlist.size() > 3)
            {
                coords.append(QVector3D(strlist.at(1).toFloat(&ok),
                                        strlist.at(2).toFloat(&ok),
                                        strlist.at(3).toFloat(&ok)));
                if(!ok) { qCritical() << "Error at line (format):" << str; }
            }
            else { qCritical() << "Error at line (count):" << str; ok = false; }
        }
        else if(key.toLower() == "vt")
        {
            if(strlist.size() > 2)
            {
                texturcoords.append(QVector2D(strlist.at(1).toFloat(&ok),
                                              strlist.at(2).toFloat(&ok)));
                if(!ok) { qCritical() << "Error at line (format):" << str; }
            }
            else { qCritical() << "Error at line (count):" << str; ok = false; }
        }
        else if(key.toLower() == "vn")
        {
            if(strlist.size() == 4)
            {
                normals.append(QVector3D(strlist.at(1).toFloat(&ok),
                                         strlist.at(2).toFloat(&ok),
                                         strlist.at(3).toFloat(&ok)));
                if(!ok) { qCritical() << "Error at line (format):" << str; }
            }
            else { qCritical() << "Error at line (count):" << str; ok = false; }
        }
        else if(key.toLower() == "f")
        {
            for(int i = 1; i < strlist.size(); i++)
            {
                auto v = strlist.at(i).split('/');
                if(v.size() == 3 && !v.at(1).isEmpty() && !v.at(2).isEmpty())
                {
                    vertexes.append(VertexData(coords.at(v.at(0).toInt(&ok, 10) - 1),
                                               texturcoords.at(v.at(1).toInt(&ok, 10) - 1),
                                               normals.at(v.at(2).toInt(&ok, 10) - 1)));
                    indexes.append(static_cast<GLuint>(indexes.size()));
                }
                else
                {
                    qCritical() << "Unsupported OBJ data format:" << strlist.at(i);
                    ok = false; break;
                }
            }
            if(!ok) { qCritical() << "Error at line (format):" << str; }
        }
    }

    objfile.close();
    qDebug() <<  "... done";
    if(!ok) return false;

    m_Rover->init(vertexes, indexes, img);
    return true;
}

void OGLWidget::Drone_scale(const float s)
{
  scale = s;
}

