#ifndef OGLWIDGET_H
#define OGLWIDGET_H

#include <QOpenGLWidget>
#include <QMatrix4x4>
#include <QOpenGLShaderProgram>
#include <QOpenGLTexture>
#include <QOpenGLBuffer>
#include <QOpenGLContext>
#include <QOpenGLFunctions_4_5_Core>


class SimpleObject3D;
class Camera_3D;
class Cube;
class CubeRander;
class DrawLine;
class MultiDrawLine;
class FeaturesRander;
class PointRender;
class CostmapRander;

class OGLWidget : public QOpenGLWidget
{
    Q_OBJECT

public:
    OGLWidget(QWidget *QOpenGLWidget = nullptr, uint8_t state = 0);
    ~OGLWidget();

protected:
    void initializeGL();
    void resizeGL(int w, int h);
    void paintGL();

    void mousePressEvent(QMouseEvent* event);
    void mouseReleaseEvent(QMouseEvent *event);
    void mouseMoveEvent(QMouseEvent* event);
    void wheelEvent(QWheelEvent* event);
    void keyPressEvent(QKeyEvent* event);

    void initShaders();
    bool initObj(const QString &path, const QImage &img);

public:

  void Drone_scale (const float s);

public slots:

    void DisplayingCubes (GLfloat* vertices_buffer, GLfloat* color_buffer, unsigned long long counter, QQuaternion rotation);

    void DisplayingPoint (GLfloat* vertices_buffer, GLfloat* color_buffer, unsigned long long counter, QQuaternion rotation, QVector3D translation);

    void DisplayingMapPoint (GLfloat* vertices_buffer, GLfloat* color_buffer, unsigned long long counter);

    void DisplayingCostMap(const QImage &map);

    void DrawLines (GLfloat* vertices_buffer, QVector3D color_buffer, unsigned long long counter, uint16_t index);

    void DrawCube (QVector3D pos);

    void DrawAruco (QVector3D pos);

    void OffsetCube (QVector3D pos);

    void Invisible_Aruco (bool invisible);

    void ClearLine ();

    QVector3D screenCoordsToPlaneCoords (const QVector2D &mouse_pose);

signals:

    void RayCastPosition (QVector3D CameraPos, QVector2D CameraRot);

    void reset();

private:

    QOpenGLFunctions_4_5_Core * function;
    QMatrix4x4 m_PojectionMatrix;
    QOpenGLShaderProgram m_Program;
    QQuaternion m_Rotation;
    Cube *m_cube;
    Cube *m_Aruco_mobile;
    CubeRander* m_CubeRander;
    CostmapRander * m_costmap;
    PointRender* m_PointRender;
    PointRender* m_PointRenderMap;

    MultiDrawLine *m_MultiLineRender;

    SimpleObject3D * m_Rover;
    Camera_3D *m_camera;
    QVector<float> cam_data;

    QVector3D init_pos;

    float scale = 1.0f;

    bool camera;

    uint8_t _state;

    double lastX = 0.0, lastY = 0.0;
};

#endif // OGLWIDGET_H
