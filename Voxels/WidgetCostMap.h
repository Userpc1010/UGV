#ifndef WIDGETCOSTMAP_H
#define WIDGETCOSTMAP_H

#include "oglwidgetbase.h"
#include <cmath>
#include <Eigen/Dense>
#include <Eigen/Geometry>

class Cube;
class CostmapRander;
class MultiDrawLine;

class WidgetCostMap : public OGLWidgetBase
{
    Q_OBJECT

public:
    explicit WidgetCostMap(QWidget *parent = nullptr);
    ~WidgetCostMap();

public slots:
    void DisplayingCostMap(const QImage &map, Eigen::Quaternionf rotation);
    void DrawLines(GLfloat* vertices_buffer, Eigen::Vector3f color_buffer, unsigned long long counter, uint16_t index);

    void DrawCube(Eigen::Vector3f pos);
    void DrawAruco(Eigen::Vector3f pos);
    void Invisible_Aruco(bool invisible);
    void ClearLine();

signals:

    void goalPositionSet(Eigen::Vector3f position, Eigen::Quaternionf rotation);

protected:
    void initializeSpecific() override;
    void paintSpecific() override;
    void onMousePress(QMouseEvent* event) override;
    void onMouseMove(QMouseEvent* event) override;
    void onMouseRelease(QMouseEvent* event) override;

private:
    Eigen::Quaternionf calculateLookAtRotation(const Eigen::Vector3f& from,
                                                   const Eigen::Vector3f& to,
                                                   const Eigen::Quaternionf& currentRotation);
    bool loadGoalRoverModel(const QString &path, const QImage &img);
    Eigen::Vector3f getPlaneCoordinatesFromMouse(const QVector2D& mousePos);

    CostmapRander* m_costmap;
    Cube* m_cube;
    MultiDrawLine* m_MultiLineRender;
    Cube* m_Aruco_mobile;

    // Для управления целью (goal)
    SimpleObject3D* m_GoalRover;
    bool m_GoalRoverPlacing;  // Флаг установки цели (левая кнопка нажата)
    Eigen::Vector3f m_GoalRoverPosition = Eigen::Vector3f::Zero();
    Eigen::Quaternionf m_GoalRoverRotation = Eigen::Quaternionf::Identity();
    QVector<VertexData> m_GoalRoverVertexData;
    QVector<GLuint> m_GoalRoverIndexData;
    QImage m_GoalRoverTexture;

};

#endif // WIDGETCOSTMAP_H
