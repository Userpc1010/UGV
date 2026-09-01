#include "oglwidgetbase.h"
#include "simpleobject3d.h"
#include "camera_3d.h"
#include <QFile>
#include <QTextStream>
#include <QDebug>
#include <QtMath>

OGLWidgetBase::OGLWidgetBase(QWidget *parent)
    : QOpenGLWidget(parent)
     , m_mapOffset(0, 0, 0)  // Инициализируем нулевым смещением
{
    init_pos = QVector3D(199.0f, -54.0f, -199.0f);
    function = new QOpenGLFunctions_4_5_Core();
    m_camera = new Camera_3D(init_pos);
    m_Rover = new SimpleObject3D();
}

OGLWidgetBase::~OGLWidgetBase()
{
    delete m_Rover;
    delete m_camera;
    delete function;
}

void OGLWidgetBase::initializeGL()
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

    // Общая инициализация для всех виджетов - модель ровера
    if(initObj(":/Moon_Rover/14014_Moon_Rover_V1_l1.obj",
               QImage(":/Moon_Rover/14014_Moon_Rover_V1_Atlas.png")))
    {
        m_Rover->translate(init_pos);
        m_Rover->scale(m_RoverScale);
    }

    // Вызов специфичной инициализации наследника
    initializeSpecific();
}

void OGLWidgetBase::resizeGL(int w, int h)
{
    float aspect = w / (h ? static_cast<float>(h) : 1);
    m_ProjectionMatrix.setToIdentity();
    m_ProjectionMatrix.perspective(80, aspect, 1.0f, 2000.0f);
}

void OGLWidgetBase::paintGL()
{
    function->glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    function->glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    m_Program.bind();
    m_Program.setUniformValue("u_projectionMatrix", m_ProjectionMatrix);
    m_Program.setUniformValue("u_map_offset", m_mapOffset);  // Устанавливаем смещение

    // Рисуем ровер (общий для всех)
    m_Rover->draw(&m_Program, function);

    // Рисуем специфичные объекты
    paintSpecific();

    m_camera->draw(&m_Program);
    m_Program.release();
}

void OGLWidgetBase::setMapOffset(QVector3D offset)
{
    m_mapOffset = offset;

    if (m_Program.isLinked()) {
        m_Program.bind();
        m_Program.setUniformValue("u_map_offset", m_mapOffset);
        m_Program.release();
        update();
    }
}

void OGLWidgetBase::keyPressEvent(QKeyEvent *event)
{
    onKeyPress(event);

    switch(event->key())
    {
        case Qt::Key_W || 87 || 1062:
            m_camera->Front_move();
            break;
        case Qt::Key_S:
            m_camera->Back_move();
            break;
        case Qt::Key_A:
            m_camera->right_move();
            break;
        case Qt::Key_D:
            m_camera->left_move();
            break;
        case Qt::Key_Space:
            m_camera->Up_move();
            break;
        case Qt::Key_Alt:
            m_camera->Down_move();
            break;
    }

    update();
    event->accept();
}

void OGLWidgetBase::wheelEvent(QWheelEvent *event)
{
    onWheel(event);

    if (event->delta() > 0) m_camera->camera_zoom(true);
    if (event->delta() < 0) m_camera->camera_zoom(false);

    update();
    event->accept();
}

void OGLWidgetBase::mousePressEvent(QMouseEvent *event)
{
     onMousePress(event);

    if(event->buttons() == Qt::RightButton)
    {
        lastX = event->localPos().x();
        lastY = event->localPos().y();
        cameraMousePressed = true;
    }

    update();
    event->accept();
}

void OGLWidgetBase::mouseMoveEvent(QMouseEvent *event)
{
    onMouseMove(event);

    if(event->buttons() == Qt::RightButton && cameraMousePressed)
    {
        double xoffset = event->localPos().x() - lastX;
        double yoffset = lastY - event->localPos().y();

        lastX = event->localPos().x();
        lastY = event->localPos().y();

        m_camera->rotate_camera(xoffset, yoffset);

    }

    update();
    event->accept();
}

void OGLWidgetBase::mouseReleaseEvent(QMouseEvent *event)
{
    onMouseRelease(event);

    if(event->button() == Qt::RightButton)
    {
        cameraMousePressed = false;
    }
    event->accept();
}


void OGLWidgetBase::onMousePress(QMouseEvent* event)
{
    // Базовая реализация пустая
    // Наследники переопределяют
}

void OGLWidgetBase::onMouseRelease(QMouseEvent *event){}

void OGLWidgetBase::onMouseMove(QMouseEvent *event){}

void OGLWidgetBase::onWheel(QWheelEvent *event){}

void OGLWidgetBase::onKeyPress(QKeyEvent *event){}

QVector3D OGLWidgetBase::screenCoordsToPlaneCoords(const QVector2D &mouse_pose)
{
    QVector4D mouse_norm(2.0f * mouse_pose.x()/width() - 1.0f,
                         -2.0f * mouse_pose.y() / height() + 1.0f,
                         -1.0f, 1.0f);

    QMatrix4x4 offset_matrix = m_ProjectionMatrix.inverted();
    QVector4D mouse_to_cam_coord((offset_matrix * mouse_norm).toVector2D(), -1.0f, 0.0f);

    offset_matrix = m_camera->ViewMatrix().inverted();
    QVector3D Ray_Direction((offset_matrix * mouse_to_cam_coord).toVector3D().normalized());
    QVector3D CamPosWorld((offset_matrix * QVector4D(0.0f, 0.0f, 0.0f, 1.0f)).toVector3D());

    QVector3D Normal_to_Plane(0.0, 1.0f, 0.0);
    QVector3D Center_of_coord_plane(0.0f, -53.0f, 0.0f);

    float t = (QVector3D::dotProduct(Center_of_coord_plane, Normal_to_Plane) -
               QVector3D::dotProduct(CamPosWorld, Normal_to_Plane)) /
              QVector3D::dotProduct(Ray_Direction, Normal_to_Plane);

    return CamPosWorld + Ray_Direction * t;
}

void OGLWidgetBase::initShaders()
{
    if(!m_Program.addShaderFromSourceFile(QOpenGLShader::Vertex, ":/vertshader.vsh"))
        this->close();
    if(!m_Program.addShaderFromSourceFile(QOpenGLShader::Fragment, ":/fragshader.fsh"))
        this->close();
    if(!m_Program.link())
        this->close();
}

bool OGLWidgetBase::initObj(const QString &path, const QImage &img)
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
    QVector<VertexData> vertexes;
    QVector<GLuint> indexes;

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
                    vertexes.append(VertexData(coords.at(v.at(0).toInt(&ok, 10) - 1),
                                              texturcoords.at(v.at(1).toInt(&ok, 10) - 1),
                                              normals.at(v.at(2).toInt(&ok, 10) - 1)));
                    indexes.append(static_cast<GLuint>(indexes.size()));
                }
            }
        }
    }

    objfile.close();
    if(!ok) return false;

    m_Rover->init(vertexes, indexes, img);
    return true;
}
