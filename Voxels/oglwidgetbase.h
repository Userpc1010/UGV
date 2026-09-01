#ifndef OGLWIDGETBASE_H
#define OGLWIDGETBASE_H

#include <QOpenGLWidget>
#include <QMatrix4x4>
#include <QOpenGLShaderProgram>
#include <QOpenGLTexture>
#include <QOpenGLBuffer>
#include <QOpenGLFunctions_4_5_Core>
#include <QMouseEvent>
#include <QWheelEvent>
#include <QKeyEvent>
#include <QQuaternion>
#include <QVector3D>

class SimpleObject3D;
class Camera_3D;
struct VertexData;

class OGLWidgetBase : public QOpenGLWidget
{
    Q_OBJECT

public:
    explicit OGLWidgetBase(QWidget *parent = nullptr);
    virtual ~OGLWidgetBase();

protected:
    // Виртуальные методы событий - наследники могут переопределить
    virtual void onMousePress(QMouseEvent* event);
    virtual void onMouseRelease(QMouseEvent* event);
    virtual void onMouseMove(QMouseEvent* event);
    virtual void onWheel(QWheelEvent* event);
    virtual void onKeyPress(QKeyEvent* event);

signals:
    void RayCastPosition(QVector3D CameraPos, QVector2D CameraRot);
    void reset();

protected:
    // Абстрактные методы - должны быть реализованы в наследниках
    virtual void initializeSpecific() = 0;
    virtual void paintSpecific() = 0;

    // Общие методы OpenGL
    void initializeGL() override;
    void resizeGL(int w, int h) override;
    void paintGL() override;

    // Общие обработчики событий
    void mousePressEvent(QMouseEvent* event) final;
    void mouseReleaseEvent(QMouseEvent *event) final;
    void mouseMoveEvent(QMouseEvent* event) final;
    void wheelEvent(QWheelEvent* event) final;
    void keyPressEvent(QKeyEvent* event) final;

    // Общие вспомогательные методы
    bool initObj(const QString &path, const QImage &img);
    QVector3D screenCoordsToPlaneCoords(const QVector2D &mouse_pose);

    void setMapOffset(QVector3D offset);

protected:
    // Общие поля для всех наследников
    QOpenGLFunctions_4_5_Core *function;
    QMatrix4x4 m_ProjectionMatrix;
    QOpenGLShaderProgram m_Program;
    Camera_3D *m_camera;
    SimpleObject3D *m_Rover;

    QVector3D init_pos;

    // Для управления камерой
    double lastX = 0.0, lastY = 0.0;
    bool cameraMousePressed = false;

    // Масштаб ровера
    float m_RoverScale = 1.0f;

    QVector3D m_mapOffset;  // Текущее смещение карты

private:
    void initShaders();
};

#endif // OGLWIDGETBASE_H
