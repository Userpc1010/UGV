#ifndef WIDGETCAMERA_H
#define WIDGETCAMERA_H

#include "oglwidgetbase.h"
#include <QQuaternion>

class Cube;
class CubeRander;
class MultiDrawLine;


class WidgetCamera : public OGLWidgetBase
{
    Q_OBJECT

public:
    explicit WidgetCamera(QWidget *parent = nullptr);
    ~WidgetCamera();

public slots:
    void DisplayingCubes(GLfloat* vertices_buffer, GLfloat* color_buffer,
                        unsigned long long counter, QQuaternion rotation);
    void DrawAruco(QVector3D pos);
    void Invisible_Aruco(bool invisible);
    void Drone_scale(float s);

    void DrawLines(GLfloat* vertices_buffer, QVector3D color_buffer,
                   uint32_t counter, uint16_t index);
    void DrawCube(QVector3D pos);
    void ClearLine();

    void updateMapOffset (QVector3D offset);

protected:
    void initializeSpecific() override;
    void paintSpecific() override;
//    void onMousePress(QMouseEvent *event);

private:
    CubeRander* m_CubeRander;
    Cube* m_Aruco_mobile;
    Cube *m_cube;
    MultiDrawLine *m_MultiLineRender;
};

#endif // WIDGETCAMERA_H
