#include "camera_3d.h"
#include <QOpenGLShaderProgram>
#include <QtMath>

Camera_3D::Camera_3D(QVector3D init_pos)
{
    front = QVector3D(0.0f, 0.0f, -1.0f); // По умолчанию смотрим по оси -Z

    m_Translate = init_pos + QVector3D(0.0f, 10.0f, 0.0f);

    m_camera_up =  QVector3D( 0.0f, 1.0f, 0.0f );

    pitch = -45.0;
    yaw =   -90.0; // Смотрим по -Z

    // Пересчитываем front из углов
    front.setX(qCos(qDegreesToRadians(yaw)) * qCos(qDegreesToRadians(pitch)));
    front.setY(qSin(qDegreesToRadians(pitch)));
    front.setZ(qSin(qDegreesToRadians(yaw)) * qCos(qDegreesToRadians(pitch)));
    front.normalize();

    updateViewMatrix(); // Обновляем матрицу вида
}

void Camera_3D::draw(QOpenGLShaderProgram *program, QOpenGLFunctions_4_5_Core *functions)
{
    if(functions != nullptr) return;

    program->setUniformValue("u_viewMatrix", m_ViewMatrix);

    updateViewMatrix();
}

void Camera_3D::rotate_camera(double x, double y)
{
  yaw += (x * sensitivity_x); pitch += (y * sensitivity_y);

  if(pitch > 89.99)  pitch = 89.99;
  if(pitch < -89.99) pitch = -89.99;

  if(yaw > 360.0) yaw = 0;
  if(yaw < 0) yaw = 360.0;

  front.setX(qCos(qDegreesToRadians(yaw)) * qCos(qDegreesToRadians(pitch)));
  front.setY(qSin(qDegreesToRadians(pitch)));
  front.setZ(qSin(qDegreesToRadians(yaw)) * qCos(qDegreesToRadians(pitch)));
  front.normalize();
}

void Camera_3D::camera_zoom(bool zoom)
{
   if(zoom)m_Translate += front;
   else m_Translate -= front;
}

QVector3D Camera_3D::GetPositionCamera()
{
    return QVector3D(m_Translate.x(), -m_Translate.y(), -m_Translate.z());
}

QMatrix4x4 Camera_3D::ViewMatrix()
{
 return m_ViewMatrix;
}

void Camera_3D::SetPositionCamera(QVector3D pos)
{
  m_Translate += pos;
}

void Camera_3D::Front_move()
{
  m_Translate += front * 5;
}

void Camera_3D::Back_move()
{
  m_Translate -= front;
}

void Camera_3D::right_move()
{
  m_Translate -=  QVector3D::crossProduct(front, m_camera_up).normalized();
}

void Camera_3D::left_move()
{
 m_Translate += QVector3D::crossProduct(front, m_camera_up).normalized();

}

void Camera_3D::Up_move()
{
  m_Translate += m_camera_up;
}

void Camera_3D::Down_move()
{
    m_Translate -= m_camera_up;
}

QVector<float> Camera_3D::GetRaycastPosition()
{
  cam_data.clear();

  cam_data.push_back(m_Translate.x());
  cam_data.push_back(m_Translate.y());
  cam_data.push_back(m_Translate.z());
  cam_data.push_back(pitch);
  cam_data.push_back(yaw);

  return cam_data;
}

void Camera_3D::updateViewMatrix()
{
    m_ViewMatrix.setToIdentity();
    m_ViewMatrix.lookAt( m_Translate,  m_Translate + front, m_camera_up );
    m_ViewMatrix = m_ViewMatrix * m_GlobalTransform.inverted();
}
