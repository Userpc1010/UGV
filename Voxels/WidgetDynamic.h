#ifndef WIDGETDYNAMIC_H
#define WIDGETDYNAMIC_H

#include "oglwidgetbase.h"
#include <Eigen/Dense>
#include <Eigen/Geometry>

class PointRender;

class WidgetDynamic : public OGLWidgetBase
{
    Q_OBJECT
public:
    explicit WidgetDynamic(QWidget *parent = nullptr);
    ~WidgetDynamic();

public slots:
    // Добавили T_arr_16 в конец
    void DisplayingDynamicPoints(const float* pcl_data, uint32_t size, const float* T_arr_16);

protected:
    void initializeSpecific() override;
    void paintSpecific() override;

private:
    PointRender* m_PointRender;
};

#endif // WIDGETDYNAMIC_H
