#ifndef WIDGETLIDAR_H
#define WIDGETLIDAR_H

#include "oglwidgetbase.h"
#include <QQuaternion>

class PointRender;

class WidgetLidar : public OGLWidgetBase
{
    Q_OBJECT

public:
    explicit WidgetLidar(QWidget *parent = nullptr);
    ~WidgetLidar();

public slots:
    void DisplayingPoint(GLfloat* vertices_buffer, GLfloat* color_buffer,
                        unsigned long long counter, QQuaternion rotation,
                        QVector3D translation);
    void DisplayingMapPoint(GLfloat* vertices_buffer, GLfloat* color_buffer,
                           unsigned long long counter);

protected:
    void initializeSpecific() override;
    void paintSpecific() override;

private:
    PointRender* m_PointRender;
    PointRender* m_PointRenderMap;
};

#endif // WIDGETLIDAR_H
