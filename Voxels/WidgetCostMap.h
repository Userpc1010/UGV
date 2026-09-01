#ifndef WIDGETCOSTMAP_H
#define WIDGETCOSTMAP_H

#include "oglwidgetbase.h"

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
    void DisplayingCostMap(const QImage &map, QQuaternion rotation);
    void DrawLines(GLfloat* vertices_buffer, QVector3D color_buffer,
                  unsigned long long counter, uint16_t index);

    void DrawCube(QVector3D pos);
    void DrawAruco(QVector3D pos);
    void Invisible_Aruco(bool invisible);
    void ClearLine();

signals:

    void goalPositionSet(QVector3D position, QQuaternion rotation);

protected:
    void initializeSpecific() override;
    void paintSpecific() override;
    void onMousePress(QMouseEvent* event) override;
    void onMouseMove(QMouseEvent* event) override;
    void onMouseRelease(QMouseEvent* event) override;

private:
    QQuaternion calculateLookAtRotation(const QVector3D& from,
                                       const QVector3D& to,
                                       const QQuaternion& currentRotation);
    bool loadGoalRoverModel(const QString &path, const QImage &img);
    QVector3D getPlaneCoordinatesFromMouse(const QVector2D& mousePos);

    CostmapRander* m_costmap;
    Cube* m_cube;
    MultiDrawLine* m_MultiLineRender;
    Cube* m_Aruco_mobile;

    // Для управления целью (goal)
    SimpleObject3D* m_GoalRover;
    bool m_GoalRoverPlacing;  // Флаг установки цели (левая кнопка нажата)
    QVector3D m_GoalRoverPosition;
    QQuaternion m_GoalRoverRotation;
    QVector<VertexData> m_GoalRoverVertexData;
    QVector<GLuint> m_GoalRoverIndexData;
    QImage m_GoalRoverTexture;

};

#endif // WIDGETCOSTMAP_H
