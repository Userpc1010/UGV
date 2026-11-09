#ifndef WIDGETCAMERA_H
#define WIDGETCAMERA_H

#include "oglwidgetbase.h"
#include <Eigen/Dense>
#include <Eigen/Geometry>

class Cube;
class HeightMapRenderer;
class MultiDrawLine;


class WidgetCamera : public OGLWidgetBase
{
    Q_OBJECT

public:
    explicit WidgetCamera(QWidget *parent = nullptr);
    ~WidgetCamera();

public slots:
    void DisplayingHeightMap(const float* data, float min, float max, float targetHeight,  Eigen::Quaternionf rotation);
    void DrawAruco(Eigen::Vector3f pos);
    void Invisible_Aruco(bool invisible);
    void Drone_scale(float s);

    void DrawLines(GLfloat* vertices_buffer, Eigen::Vector3f color_buffer, unsigned long long counter, uint16_t index);
    void DrawCube(Eigen::Vector3f pos);
    void ClearLine();

    void updateMapOffset(Eigen::Vector3f offset);

protected:
    void initializeSpecific() override;
    void paintSpecific() override;
//    void onMousePress(QMouseEvent *event);

private:
    HeightMapRenderer* m_HeightMap;
    Cube* m_Aruco_mobile;
    Cube *m_cube;
    MultiDrawLine *m_MultiLineRender;
};

#endif // WIDGETCAMERA_H
