#include "WidgetDynamic.h"
#include "pointrender.h"
#include "camera_3d.h"
#include <QDebug>

WidgetDynamic::WidgetDynamic(QWidget *parent)
    : OGLWidgetBase(parent)
    , m_PointRender(new PointRender())
{
    m_RoverScale = 10.0f;
    m_camera->SetPositionCamera(QVector3D(-1000.0f, 1000.0f, 0.0f));
}

WidgetDynamic::~WidgetDynamic()
{
    delete m_PointRender;
}

void WidgetDynamic::initializeSpecific()
{
    m_PointRender->init();
}

void WidgetDynamic::paintSpecific()
{
    m_PointRender->draw(&m_Program, function, 9.0);
}

void WidgetDynamic::DisplayingDynamicPoints(const float* pcl_data, uint32_t size, const float* T_arr_16)
{
    if (!pcl_data || size == 0) return;

    // 1. Рисуем точки (они уже в world frame)
    GLfloat* verts  = new GLfloat[3 * size];
    GLfloat* colors = new GLfloat[3 * size];

    for (uint32_t i = 0; i < size; i++) {
        // Как в publishCloud: (x, z, y) * 100 для GL
        verts[i*3+0] = pcl_data[i*3+0] * 100.0f;
        verts[i*3+1] = pcl_data[i*3+2] * 100.0f;
        verts[i*3+2] = pcl_data[i*3+1] * 100.0f;
        // Красный для движущихся
        colors[i*3+0] = 1.0f;
        colors[i*3+1] = 0.0f;
        colors[i*3+2] = 0.0f;
    }

    m_PointRender->translate(verts, colors, size);
    delete[] verts;
    delete[] colors;

    // 2. Обновляем позу ровера из T_arr_16
    if (T_arr_16) {
        Eigen::Matrix4f T;
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                T(i, j) = T_arr_16[i*4 + j];

        Eigen::Quaternionf q(T.block<3,3>(0,0));
        Eigen::Vector3f t = T.block<3,1>(0,3);


        m_Rover->rotate(q);
        m_Rover->translate(t);
    }

    update();
}
