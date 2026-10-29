#ifndef CAMERA_3D_H
#define CAMERA_3D_H

#include "simpleobject3d.h"
#include <QQuaternion>
#include <QVector3D>
#include <QMatrix4x4>

class Camera_3D

{

public:
    Camera_3D(QVector3D init_pos);

    void draw(QOpenGLShaderProgram* program, QOpenGLFunctions_4_5_Core *functions = nullptr);

    void rotate_camera (double x, double y);
    void camera_zoom   (bool zoom);

    void SetPositionCamera (QVector3D pos);
    QVector3D GetPositionCamera ();
    QMatrix4x4 ViewMatrix();

    void Front_move ();
    void Back_move  ();
    void right_move ();
    void left_move  ();
    void Up_move    ();
    void Down_move  ();

    QVector<float> GetRaycastPosition();

private:

    void updateViewMatrix();

private:
    QVector3D m_Translate;
    QVector3D m_camera_up;
    QVector3D front;

    QMatrix4x4 m_GlobalTransform;
    QMatrix4x4 m_ViewMatrix;

    qreal pitch = 0.0;
    qreal yaw = 0.0;

    double sensitivity_x = 0.5;
    double sensitivity_y = 0.5;

    QVector<float> cam_data;
};

#endif // CAMERA_3D_H
